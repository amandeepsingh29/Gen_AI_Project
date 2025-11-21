import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)
from tqdm import tqdm
import argparse
from PIL import Image
import glob

CONCEPT_COLUMNS = [
    "pigment_network",
    "blue_whitish_veil",
    "vascular_structures",
    "pigmentation",
    "streaks",
    "dots_and_globules",
    "regression_structures",
]

NEGATIVE_CONCEPT_STRINGS = {
    "absent",
    "none",
    "nan",
    "not present",
    "no",
    "false",
    "0",
    "",
}

POSITIVE_HINT_STRINGS = {
    "present",
    "typical",
    "irregular",
    "localized irregular",
    "blue",
    "arborizing",
    "hairpin",
    "dotted",
    "within regression",
    "linear irregular",
    "whitish",
    "network",
    "veil",
    "streak",
    "globule",
    "vascular",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConceptDerm7ptDataset(Dataset):
    """Dataset class for CBM training with concepts and diagnosis"""
    
    def __init__(self, meta_df, concept_df, img_root, transform=None):
        self.meta_df = meta_df
        self.concept_df = concept_df
        self.img_root = img_root
        self.transform = transform
        
        # Get unique diagnosis classes
        self.diagnosis_classes = sorted(meta_df['diagnosis'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.diagnosis_classes)}
        self.concept_columns = [col for col in self.concept_df.columns if col != 'image_id']
        
        # Merge metadata with concept data
        self.merged_data = self._merge_data()
        
        print(f"CBM Dataset created with {len(self.merged_data)} samples")
        print(f"Diagnosis classes: {self.diagnosis_classes}")
        
    def _merge_data(self):
        """Merge metadata with concept annotations"""
        meta_copy = self.meta_df.copy()
        meta_copy['image_id'] = meta_copy['clinic'].apply(lambda x: os.path.basename(str(x)).lower())
        meta_copy = meta_copy.drop(columns=self.concept_columns, errors='ignore')
        
        merged = meta_copy.merge(self.concept_df, on='image_id', how='inner')
        print(f"Merged {len(merged)} samples with concept annotations")
        
        return merged
    
    def __len__(self):
        return len(self.merged_data)
    
    def __getitem__(self, idx):
        row = self.merged_data.iloc[idx]
        
        image_filename = row['image_id']
        
        possible_paths = [
            os.path.join(self.img_root, image_filename),
            os.path.join(self.img_root, image_filename.replace('.jpg', '.png')),
            os.path.join(self.img_root, image_filename.replace('.png', '.jpg'))
        ]
        
        if not any(os.path.exists(p) for p in possible_paths):
            search_pattern = os.path.join(self.img_root, '**', image_filename)
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                image_path = found_files[0]
            else:
                base_name = os.path.splitext(image_filename)[0]
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    search_pattern = os.path.join(self.img_root, '**', base_name + ext)
                    found_files = glob.glob(search_pattern, recursive=True)
                    if found_files:
                        image_path = found_files[0]
                        break
                else:
                    raise FileNotFoundError(f"Could not find image: {image_filename}")
        else:
            image_path = next(p for p in possible_paths if os.path.exists(p))
        
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        diagnosis = row['diagnosis']
        label = self.class_to_idx[diagnosis]
        
        concepts = torch.tensor([float(row[col]) for col in self.concept_columns], dtype=torch.float32)

        return image, label, concepts

class PretrainedConceptPredictor(nn.Module):
    """Concept predictor using pre-trained ResNet backbone"""
    
    def __init__(self, pretrained_model_path, num_concepts=7, freeze_backbone=False):
        super(PretrainedConceptPredictor, self).__init__()
        
        # Load the pre-trained ResNet model
        try:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu', weights_only=True)
        except TypeError:
            checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        
        # Determine the model architecture from the checkpoint
        if 'backbone.layer4.1.conv2.weight' in checkpoint:
            if checkpoint['backbone.layer4.1.conv2.weight'].shape[0] == 512:
                backbone = models.resnet18(pretrained=False)
            else:
                backbone = models.resnet50(pretrained=False)
        else:
            backbone = models.resnet18(pretrained=False)  # Default fallback
        
        # Load the pre-trained weights
        backbone.load_state_dict(checkpoint, strict=False)
        
        # Remove the final classification layer and keep feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        
        # Add concept prediction layer
        initial_raw = torch.randn(num_concepts, self.feature_dim) * 0.01 - 5.0
        self.raw_weight = nn.Parameter(initial_raw)
        self.bias = nn.Parameter(torch.zeros(num_concepts))
        
        print(f"Created concept predictor with {self.feature_dim} features -> {num_concepts} concepts")
        
    def forward(self, x, return_logits: bool = False):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten
        weight = F.softplus(self.raw_weight)
        concept_logits = F.linear(features, weight, self.bias)
        if return_logits:
            return concept_logits
        return torch.sigmoid(concept_logits)

class DiagnosisPredictor(nn.Module):
    """Diagnosis predictor using concept probabilities"""
    
    def __init__(self, num_concepts=7, num_classes=19):
        super(DiagnosisPredictor, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, concepts):
        return self.classifier(concepts)

class ConceptBottleneckModel(nn.Module):
    """Combined CBM model"""
    
    def __init__(self, concept_predictor, diagnosis_predictor):
        super(ConceptBottleneckModel, self).__init__()
        self.concept_predictor = concept_predictor
        self.diagnosis_predictor = diagnosis_predictor
        
    def forward(self, x, return_concepts=False):
        concepts = self.concept_predictor(x)
        diagnoses = self.diagnosis_predictor(concepts)
        
        if return_concepts:
            return diagnoses, concepts
        return diagnoses

def _map_concept_value(value: object) -> float:
    if pd.isna(value):
        return 0.0
    s = str(value).strip().lower()
    if s in NEGATIVE_CONCEPT_STRINGS:
        return 0.0
    try:
        numeric = float(s)
        return 1.0 if numeric > 0 else 0.0
    except ValueError:
        pass
    for hint in POSITIVE_HINT_STRINGS:
        if hint in s:
            return 1.0
    return 1.0 if s else 0.0


def load_or_create_concept_data(meta_df, concept_file_path, num_concepts=7):
    """Derive concept annotations from metadata columns."""
    expected = [c for c in CONCEPT_COLUMNS if c in meta_df.columns][:num_concepts]

    if os.path.exists(concept_file_path):
        print(f"Loading concept annotations from {concept_file_path}")
        concept_df = pd.read_csv(concept_file_path)
        loaded_columns = [c for c in concept_df.columns if c != "image_id"]

        missing_expected = [c for c in expected if c not in loaded_columns]
        if missing_expected or len(loaded_columns) != len(expected):
            print(
                "Existing concept file is missing expected columns; "
                "recomputing from metadata."
            )
        else:
            return concept_df

    if not expected:
        raise ValueError("Metadata does not contain expected concept columns; cannot build CBM labels.")

    image_ids = meta_df['clinic'].apply(lambda x: os.path.basename(str(x)).lower())
    concept_data = {"image_id": image_ids.tolist()}

    for col in expected:
        concept_data[col] = meta_df[col].apply(_map_concept_value).astype(float)

    concept_df = pd.DataFrame(concept_data)
    concept_df = concept_df.drop_duplicates(subset="image_id", keep="first")

    os.makedirs(os.path.dirname(concept_file_path), exist_ok=True)
    concept_df.to_csv(concept_file_path, index=False)
    print(f"Derived concept annotations for {len(concept_df)} images and saved to {concept_file_path}")
    return concept_df


def compute_concept_pos_weights(dataset: ConceptDerm7ptDataset) -> torch.Tensor:
    concept_frame = dataset.merged_data[dataset.concept_columns].astype(float)
    total = float(len(concept_frame))
    pos_counts_np = concept_frame.sum(axis=0).values.astype(float)
    pos_counts = torch.tensor(pos_counts_np, dtype=torch.float32)
    neg_counts = torch.tensor(total - pos_counts_np, dtype=torch.float32)
    pos_weight = torch.where(pos_counts > 0, neg_counts / pos_counts, torch.full_like(pos_counts, total))
    return pos_weight


def build_class_balanced_sampler(dataset: ConceptDerm7ptDataset) -> WeightedRandomSampler:
    label_indices = dataset.merged_data['diagnosis'].map(dataset.class_to_idx)
    class_counts = label_indices.value_counts()
    sample_weights = label_indices.map(lambda idx: 1.0 / class_counts[idx]).astype(float)
    weights_tensor = torch.as_tensor(sample_weights.values, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)


def compute_class_weights(dataset: ConceptDerm7ptDataset) -> torch.Tensor:
    label_indices = dataset.merged_data['diagnosis'].map(dataset.class_to_idx)
    class_counts = label_indices.value_counts()
    num_classes = len(dataset.diagnosis_classes)
    total_samples = float(len(label_indices))
    weights = torch.zeros(num_classes, dtype=torch.float32)
    present_classes = max(1, len(class_counts))
    for class_idx in range(num_classes):
        count = class_counts.get(class_idx, 0)
        if count > 0:
            weights[class_idx] = total_samples / (present_classes * float(count))
        else:
            weights[class_idx] = 0.0
    return weights


def compute_brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(np.mean((probs - targets) ** 2))


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(probs)
    for i in range(n_bins):
        lower, upper = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lower) & (probs <= upper)
        else:
            mask = (probs >= lower) & (probs < upper)
        if not np.any(mask):
            continue
        bin_confidence = np.mean(probs[mask])
        bin_accuracy = np.mean(targets[mask])
        ece += np.abs(bin_accuracy - bin_confidence) * (np.sum(mask) / total)
    return float(ece)


def safe_metric(metric_fn, y_true: np.ndarray, y_score: np.ndarray):
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None

def train_concept_predictor(model, train_loader, val_loader, pos_weights, num_epochs=50):
    """Train the concept predictor"""
    print("Training concept predictor...")
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights.to(DEVICE))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"Concept Epoch {epoch+1}/{num_epochs} [Train]")
        for images, _, concepts in train_pbar:
            images = images.to(DEVICE)
            concepts = concepts.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(images, return_logits=True)
            loss = criterion(logits, concepts)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Concept Epoch {epoch+1}/{num_epochs} [Val]")
            for images, _, concepts in val_pbar:
                images = images.to(DEVICE)
                concepts = concepts.to(DEVICE)
                
                logits = model(images, return_logits=True)
                loss = criterion(logits, concepts)
                val_loss += loss.item()
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        print(f"Concept Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            print(f"  New best concept model saved (Val Loss: {best_val_loss:.4f})")
    
    model.load_state_dict(best_model_state)
    return model

def evaluate_cbm(cbm, test_loader, class_names):
    """Evaluate the full CBM model with detailed debugging and concept-level analysis."""
    print("Evaluating CBM model...")

    cbm.eval()
    concept_prob_batches = []
    concept_pred_batches = []
    concept_target_batches = []
    diagnosis_prob_batches = []
    diagnosis_pred_batches = []
    diagnosis_target_batches = []

    prediction_counts = {i: 0 for i in range(len(class_names))}
    concept_names = list(test_loader.dataset.concept_columns)
    num_classes = len(class_names)

    mediation_effect_sums = {
        name: {
            'set_to_one': np.zeros(num_classes, dtype=np.float64),
            'set_to_zero': np.zeros(num_classes, dtype=np.float64),
        }
        for name in concept_names
    }
    total_samples = 0
    failures = []
    sample_pointer = 0

    with torch.no_grad():
        for batch_idx, (images, labels, concepts) in enumerate(tqdm(test_loader, desc="Evaluating")):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            concepts = concepts.to(DEVICE)

            diagnosis_logits, predicted_concepts = cbm(images, return_concepts=True)
            diagnosis_probs_batch = torch.softmax(diagnosis_logits, dim=1)
            concept_pred_binary = (predicted_concepts > 0.5).float()
            diagnosis_pred = torch.argmax(diagnosis_logits, dim=1)

            concept_prob_batches.append(predicted_concepts.cpu())
            concept_pred_batches.append(concept_pred_binary.cpu())
            concept_target_batches.append(concepts.cpu())
            diagnosis_prob_batches.append(diagnosis_probs_batch.cpu())
            diagnosis_pred_batches.append(diagnosis_pred.cpu())
            diagnosis_target_batches.append(labels.cpu())

            batch_size = images.size(0)
            total_samples += batch_size

            for idx, name in enumerate(concept_names):
                intervened_one = predicted_concepts.clone()
                intervened_one[:, idx] = 1.0
                logits_one = cbm.diagnosis_predictor(intervened_one)
                delta_one = torch.softmax(logits_one, dim=1) - diagnosis_probs_batch
                mediation_effect_sums[name]['set_to_one'] += delta_one.sum(dim=0).cpu().numpy()

                intervened_zero = predicted_concepts.clone()
                intervened_zero[:, idx] = 0.0
                logits_zero = cbm.diagnosis_predictor(intervened_zero)
                delta_zero = torch.softmax(logits_zero, dim=1) - diagnosis_probs_batch
                mediation_effect_sums[name]['set_to_zero'] += delta_zero.sum(dim=0).cpu().numpy()

            batch_indices = list(range(sample_pointer, sample_pointer + batch_size))
            image_ids = [
                test_loader.dataset.merged_data.iloc[i]['image_id']
                for i in batch_indices
            ]
            sample_pointer += batch_size

            labels_np = labels.cpu().numpy()
            diagnosis_pred_np = diagnosis_pred.cpu().numpy()
            diagnosis_probs_np = diagnosis_probs_batch.cpu().numpy()
            concept_probs_np = predicted_concepts.cpu().numpy()
            concept_pred_np = concept_pred_binary.cpu().numpy()
            concept_targets_np = concepts.cpu().numpy()

            for idx in range(batch_size):
                prediction_counts[diagnosis_pred_np[idx]] += 1

                if diagnosis_pred_np[idx] != labels_np[idx] and len(failures) < 25:
                    error_indices = np.where(concept_pred_np[idx] != concept_targets_np[idx])[0]
                    concept_errors = []
                    for concept_idx in error_indices:
                        concept_errors.append({
                            'concept': concept_names[concept_idx],
                            'target': float(concept_targets_np[idx][concept_idx]),
                            'pred_binary': float(concept_pred_np[idx][concept_idx]),
                            'pred_prob': float(concept_probs_np[idx][concept_idx]),
                        })

                    failures.append({
                        'image_id': image_ids[idx],
                        'true_diagnosis': class_names[labels_np[idx]],
                        'predicted_diagnosis': class_names[diagnosis_pred_np[idx]],
                        'predicted_confidence': float(diagnosis_probs_np[idx][diagnosis_pred_np[idx]]),
                        'concept_errors': concept_errors,
                    })

    concept_probs = torch.cat(concept_prob_batches, dim=0).numpy()
    concept_preds = torch.cat(concept_pred_batches, dim=0).numpy()
    concept_targets = torch.cat(concept_target_batches, dim=0).numpy()
    diagnosis_probs = torch.cat(diagnosis_prob_batches, dim=0).numpy()
    diagnosis_preds = torch.cat(diagnosis_pred_batches, dim=0).numpy()
    diagnosis_targets = torch.cat(diagnosis_target_batches, dim=0).numpy()

    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("=" * 60)

    total_predictions = len(diagnosis_preds)
    for i, class_name in enumerate(class_names):
        count = prediction_counts[i]
        percentage = (count / total_predictions) * 100 if total_predictions else 0.0
        print(f"{class_name:25s}: {count:4d} predictions ({percentage:5.1f}%)")

    unique_predictions = len(set(diagnosis_preds.tolist()))
    print(f"\nUnique classes predicted: {unique_predictions}/{len(class_names)}")

    if unique_predictions == 1:
        print("🚨 WARNING: Model is predicting only ONE class!")
        most_frequent = np.bincount(diagnosis_preds).argmax()
        print(f"Always predicting: {class_names[most_frequent]}")
    elif unique_predictions < max(1, int(len(class_names) * 0.3)):
        print("⚠️  WARNING: Model has limited prediction diversity")

    max_probs = np.max(diagnosis_probs, axis=1) if len(diagnosis_probs) else np.array([])
    avg_confidence = float(np.mean(max_probs)) if len(max_probs) else 0.0
    print(f"\nAverage prediction confidence: {avg_confidence:.4f}")
    if avg_confidence > 0.95:
        print("⚠️  Very high confidence - possible overconfident model")
    elif avg_confidence < 0.4:
        print("⚠️  Very low confidence - model may be uncertain")

    concept_accuracy = accuracy_score(concept_targets.flatten(), concept_preds.flatten())
    concept_precision, concept_recall, concept_f1, _ = precision_recall_fscore_support(
        concept_targets.flatten(), concept_preds.flatten(), average='binary', zero_division=0
    )

    diagnosis_accuracy = accuracy_score(diagnosis_targets, diagnosis_preds)
    diagnosis_precision, diagnosis_recall, diagnosis_f1, _ = precision_recall_fscore_support(
        diagnosis_targets, diagnosis_preds, average='weighted', zero_division=0
    )

    print(f"\n" + "=" * 60)
    print("PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Concept Accuracy: {concept_accuracy:.4f}")
    print(f"Concept Precision: {concept_precision:.4f}")
    print(f"Concept Recall: {concept_recall:.4f}")
    print(f"Concept F1: {concept_f1:.4f}")
    print()
    print(f"Diagnosis Accuracy: {diagnosis_accuracy:.4f}")
    print(f"Diagnosis Precision: {diagnosis_precision:.4f}")
    print(f"Diagnosis Recall: {diagnosis_recall:.4f}")
    print(f"Diagnosis F1: {diagnosis_f1:.4f}")

    per_concept_metrics = {}
    concept_calibration = {}
    print(f"\nCONCEPT-LEVEL PERFORMANCE")
    for idx, name in enumerate(concept_names):
        y_true = concept_targets[:, idx]
        y_prob = concept_probs[:, idx]
        y_pred = concept_preds[:, idx]

        auc = safe_metric(roc_auc_score, y_true, y_prob)
        auprc = safe_metric(average_precision_score, y_true, y_prob)
        precision_i, recall_i, f1_i, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        brier = compute_brier_score(y_prob, y_true)
        ece = compute_ece(y_prob, y_true)

        per_concept_metrics[name] = {
            'roc_auc': auc,
            'auprc': auprc,
            'precision': float(precision_i),
            'recall': float(recall_i),
            'f1': float(f1_i),
        }
        concept_calibration[name] = {
            'brier': brier,
            'ece': ece,
        }

        auc_text = f"ROC-AUC {auc:.4f}" if auc is not None else "ROC-AUC N/A"
        auprc_text = f"AUPRC {auprc:.4f}" if auprc is not None else "AUPRC N/A"
        print(
            f"- {name:25s}: {auc_text}, {auprc_text},"
            f" Precision {precision_i:.4f}, Recall {recall_i:.4f}, F1 {f1_i:.4f},"
            f" Brier {brier:.4f}, ECE {ece:.4f}"
        )

    mediation_summary = {}
    print(f"\nMEDIATION ANALYSIS (avg Δprobability)")
    for name in concept_names:
        if total_samples > 0:
            delta_one = mediation_effect_sums[name]['set_to_one'] / total_samples
            delta_zero = mediation_effect_sums[name]['set_to_zero'] / total_samples
        else:
            delta_one = np.zeros(num_classes)
            delta_zero = np.zeros(num_classes)

        delta_one_dict = {class_names[i]: float(delta_one[i]) for i in range(num_classes)}
        delta_zero_dict = {class_names[i]: float(delta_zero[i]) for i in range(num_classes)}
        mediation_summary[name] = {
            'set_to_one': delta_one_dict,
            'set_to_zero': delta_zero_dict,
        }

        if num_classes:
            top_one = max(delta_one_dict.items(), key=lambda kv: abs(kv[1]))
            top_zero = max(delta_zero_dict.items(), key=lambda kv: abs(kv[1]))
            print(
                f"- {name:25s}: force=1 -> {top_one[0]} Δ{top_one[1]:+.4f};"
                f" force=0 -> {top_zero[0]} Δ{top_zero[1]:+.4f}"
            )

    total_failures = int(np.sum(diagnosis_preds != diagnosis_targets))
    failures_to_show = failures[:5]
    if failures_to_show:
        print(f"\nFAILURE ANALYSIS (showing {len(failures_to_show)} of {total_failures} misclassifications)")
        for item in failures_to_show:
            concepts_str = ", ".join(
                f"{err['concept']} (t={int(err['target'])}, p={int(err['pred_binary'])}, prob={err['pred_prob']:.2f})"
                for err in item['concept_errors']
            ) or "no concept discrepancies"
            print(
                f"- {item['image_id']}: true={item['true_diagnosis']}, pred={item['predicted_diagnosis']}"
                f" (conf={item['predicted_confidence']:.2f}); concept errors: {concepts_str}"
            )

    return {
        'concept_metrics': {
            'accuracy': float(concept_accuracy),
            'precision': float(concept_precision),
            'recall': float(concept_recall),
            'f1': float(concept_f1),
        },
        'concept_metrics_per_concept': per_concept_metrics,
        'concept_calibration': concept_calibration,
        'diagnosis_metrics': {
            'accuracy': float(diagnosis_accuracy),
            'precision': float(diagnosis_precision),
            'recall': float(diagnosis_recall),
            'f1': float(diagnosis_f1),
        },
        'prediction_distribution': {
            class_names[i]: int(prediction_counts[i]) for i in range(len(class_names))
        },
        'confidence_stats': {
            'average_confidence': avg_confidence,
            'unique_predictions': int(unique_predictions),
        },
        'mediation_effects': mediation_summary,
        'failure_analysis': {
            'total_failures': total_failures,
            'sampled_examples': failures,
        },
    }

def train_diagnosis_predictor(
    concept_predictor,
    diagnosis_predictor,
    train_loader,
    val_loader,
    num_epochs=50,
    class_weights: Optional[torch.Tensor] = None,
    gt_warmup_epochs: int = 5,
    gt_mix_epochs: int = 5,
):
    """Train diagnosis predictor with frozen concept predictor, balanced sampling, and GT warmup."""
    print("Training diagnosis predictor...")
    
    # Freeze concept predictor
    for param in concept_predictor.parameters():
        param.requires_grad = False

    if class_weights is None:
        raise ValueError("class_weights must be provided for diagnosis training")

    print("Class weights (diagnosis):")
    for idx, weight in enumerate(class_weights):
        print(f"  Class {idx}: weight={float(weight):.3f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.Adam(diagnosis_predictor.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training phase
        diagnosis_predictor.train()
        concept_predictor.eval()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_predictions = []
        
        mix_ratio = 0.0
        if epoch >= gt_warmup_epochs:
            if gt_mix_epochs <= 0:
                mix_ratio = 1.0
            else:
                mix_ratio = min(1.0, (epoch - gt_warmup_epochs + 1) / gt_mix_epochs)

        train_pbar = tqdm(train_loader, desc=f"Diagnosis Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels, gt_concepts in train_pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            gt_concepts = gt_concepts.to(DEVICE)

            # Get concepts from frozen predictor
            with torch.no_grad():
                predicted_concepts = concept_predictor(images)

            if epoch < gt_warmup_epochs:
                concepts = gt_concepts
            elif mix_ratio < 1.0:
                concepts = mix_ratio * predicted_concepts + (1.0 - mix_ratio) * gt_concepts
            else:
                concepts = predicted_concepts
            
            optimizer.zero_grad()
            outputs = diagnosis_predictor(concepts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            train_predictions.extend(predicted.cpu().numpy())
            
            acc = 100 * train_correct / train_total
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%', 'mix': f'{mix_ratio:.2f}'})
        
        # Check training prediction diversity
        unique_train_preds = len(set(train_predictions))
        
        # Validation phase
        diagnosis_predictor.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_predictions = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Diagnosis Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels, _ in val_pbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                concepts = concept_predictor(images)
                outputs = diagnosis_predictor(concepts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_predictions.extend(predicted.cpu().numpy())
                
                acc = 100 * val_correct / val_total
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{acc:.2f}%'})
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        
        unique_val_preds = len(set(val_predictions))
        
        scheduler.step(val_loss)
        
        print(f"Diagnosis Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"                         Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"                         Train unique preds: {unique_train_preds}, Val unique preds: {unique_val_preds}")
        
        if unique_train_preds == 1:
            print("  🚨 WARNING: Training predictions are all the same class!")
        if unique_val_preds == 1:
            print("  🚨 WARNING: Validation predictions are all the same class!")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = diagnosis_predictor.state_dict().copy()
            print(f"  ⭐ New best diagnosis model saved (Val Acc: {best_val_acc:.4f})")
    
    diagnosis_predictor.load_state_dict(best_model_state)
    return diagnosis_predictor

def main():
    parser = argparse.ArgumentParser(description='Train CBM using pre-trained ResNet')
    parser.add_argument('--pretrained_path', type=str, default='results/best_model.pt',
                       help='Path to pre-trained ResNet model')
    parser.add_argument('--concept_epochs', type=int, default=15, help='Epochs for concept training')
    parser.add_argument('--diagnosis_epochs', type=int, default=15, help='Epochs for diagnosis training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_concepts', type=int, default=7, help='Number of concepts')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze ResNet backbone')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--diag_gt_warmup', type=int, default=5, help='Epochs to train diagnosis head on ground-truth concepts before mixing predictions')
    parser.add_argument('--diag_gt_mix_epochs', type=int, default=5, help='Epochs to linearly anneal from ground-truth to predicted concepts')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    print(f"Training CBM with pre-trained model: {args.pretrained_path}")
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Load metadata
    meta = pd.read_csv("dataset/meta/meta.csv")
    print(f"Loaded metadata with {len(meta)} samples")
    
    # Load data splits
    try:
        train_idx = pd.read_csv("dataset/meta/all.csv")["indexes"].tolist()
        val_idx = pd.read_csv("dataset/meta/val_indexes.csv")["indexes"].tolist()
        test_idx = pd.read_csv("dataset/meta/test_indexes_all.csv")["indexes"].tolist()
    except FileNotFoundError:
        # Create splits if not available
        from train_resnet_derm7pt import create_data_splits
        train_idx, val_idx, test_idx = create_data_splits(meta)
    
    # Validate indices
    max_index = len(meta) - 1
    train_idx = [idx for idx in train_idx if 0 <= idx <= max_index]
    val_idx = [idx for idx in val_idx if 0 <= idx <= max_index]
    test_idx = [idx for idx in test_idx if 0 <= idx <= max_index]
    
    # Create data splits
    train_meta = meta.iloc[train_idx].reset_index(drop=True)
    val_meta = meta.iloc[val_idx].reset_index(drop=True)
    test_meta = meta.iloc[test_idx].reset_index(drop=True)
    
    # Load or create concept data
    concept_df = load_or_create_concept_data(meta, "dataset/concepts.csv", args.num_concepts)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ConceptDerm7ptDataset(train_meta, concept_df, 'dataset/images', transform=transform)
    val_dataset = ConceptDerm7ptDataset(val_meta, concept_df, 'dataset/images', transform=transform)
    test_dataset = ConceptDerm7ptDataset(test_meta, concept_df, 'dataset/images', transform=transform)

    concept_pos_weights = compute_concept_pos_weights(train_dataset)
    concept_positive_rates = train_dataset.merged_data[train_dataset.concept_columns].mean()
    print("Concept positive rates (train):")
    for name, rate in concept_positive_rates.items():
        weight = concept_pos_weights[train_dataset.concept_columns.index(name)].item()
        print(f"  {name}: {rate:.4f} -> pos_weight {weight:.2f}")
    
    # Create data loaders
    train_sampler = build_class_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers)
    
    class_weights = compute_class_weights(train_dataset)
    print("Training diagnosis distribution:")
    train_counts = train_dataset.merged_data['diagnosis'].value_counts()
    for name, count in train_counts.items():
        idx = train_dataset.class_to_idx[name]
        print(f"  {name} (class {idx}): {count} samples")

    # Create models
    num_classes = len(train_dataset.diagnosis_classes)
    
    concept_predictor = PretrainedConceptPredictor(
        args.pretrained_path, 
        num_concepts=args.num_concepts,
        freeze_backbone=args.freeze_backbone
    ).to(DEVICE)
    
    diagnosis_predictor = DiagnosisPredictor(
        num_concepts=args.num_concepts,
        num_classes=num_classes
    ).to(DEVICE)
    
    print(f"Created CBM with {args.num_concepts} concepts and {num_classes} diagnosis classes")
    
    # Train concept predictor
    concept_predictor = train_concept_predictor(
        concept_predictor,
        train_loader,
        val_loader,
        concept_pos_weights,
        args.concept_epochs,
    )
    
    # Train diagnosis predictor
    diagnosis_predictor = train_diagnosis_predictor(
        concept_predictor,
        diagnosis_predictor,
        train_loader,
        val_loader,
        args.diagnosis_epochs,
        class_weights=class_weights,
        gt_warmup_epochs=args.diag_gt_warmup,
        gt_mix_epochs=args.diag_gt_mix_epochs,
    )
    
    # Create full CBM
    cbm = ConceptBottleneckModel(concept_predictor, diagnosis_predictor)
    
    # Evaluate
    results = evaluate_cbm(cbm, test_loader, train_dataset.diagnosis_classes)
    
    # Save models
    torch.save(concept_predictor.state_dict(), "results/cbm_concept_predictor.pt")
    torch.save(diagnosis_predictor.state_dict(), "results/cbm_diagnosis_predictor.pt")
    
    # Save results
    final_results = {
        'model_info': {
            'pretrained_path': args.pretrained_path,
            'num_concepts': args.num_concepts,
            'num_classes': num_classes,
            'class_names': train_dataset.diagnosis_classes,
            'freeze_backbone': args.freeze_backbone
        },
        'training_args': vars(args),
        'evaluation_results': results
    }
    
    with open("results/cbm_results.json", 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nCBM training completed!")
    print(f"Models saved to: results/cbm_concept_predictor.pt, results/cbm_diagnosis_predictor.pt")
    print(f"Results saved to: results/cbm_results.json")

if __name__ == "__main__":
    main()
