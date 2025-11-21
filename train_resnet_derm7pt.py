import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
from tqdm import tqdm
import argparse
import math
from PIL import Image
import glob
import json
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Derm7ptDataset(Dataset):
    """Dataset class for Derm7pt dermoscopic images"""
    
    def __init__(self, meta_df, img_root, transform=None):
        self.meta_df = meta_df
        self.img_root = img_root
        self.transform = transform
        
        # Get unique diagnosis classes
        self.diagnosis_classes = sorted(meta_df['diagnosis'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.diagnosis_classes)}
        
        print(f"Found {len(self.diagnosis_classes)} diagnosis classes: {self.diagnosis_classes}")
    
    def __len__(self):
        return len(self.meta_df)
    
    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        
        # Get image path
        image_filename = os.path.basename(row['clinic']).lower()
        
        # Search for the image in the images directory
        possible_paths = [
            os.path.join(self.img_root, image_filename),
            os.path.join(self.img_root, image_filename.replace('.jpg', '.png')),
            os.path.join(self.img_root, image_filename.replace('.png', '.jpg'))
        ]
        
        # Also search in subdirectories
        if not any(os.path.exists(p) for p in possible_paths):
            search_pattern = os.path.join(self.img_root, '**', image_filename)
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                image_path = found_files[0]
            else:
                # Try alternative extensions
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
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get diagnosis label
        diagnosis = row['diagnosis']
        label = self.class_to_idx[diagnosis]
        
        return image, label

class ResNetClassifier(nn.Module):
    """ResNet classifier for dermoscopic image diagnosis"""
    
    def __init__(self, model_name='resnet18', num_classes=5, pretrained=True):
        super(ResNetClassifier, self).__init__()
        
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.backbone.fc = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def create_data_splits(meta_df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Create train/validation/test splits if index files don't exist"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
    
    n_samples = len(meta_df)
    indices = list(range(n_samples))
    
    # Shuffle indices for random split
    import random
    random.seed(42)
    random.shuffle(indices)
    
    # Calculate split points
    train_end = int(train_ratio * n_samples)
    val_end = train_end + int(val_ratio * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    print(f"Created splits - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    return train_idx, val_idx, test_idx

def plot_training_history(history, save_path="results/training_plots.png"):
    """Plot training history including loss and accuracy curves"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    axes[0, 0].plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(epochs, [acc * 100 for acc in history['train_accs']], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, [acc * 100 for acc in history['val_accs']], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    if 'learning_rates' in history:
        axes[1, 0].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)
        axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    axes[1, 1].text(0.1, 0.8, f"Best Validation Accuracy: {history['best_val_acc']:.4f}", 
                   transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold')
    axes[1, 1].text(0.1, 0.6, f"Final Training Loss: {history['train_losses'][-1]:.4f}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.4, f"Final Validation Loss: {history['val_losses'][-1]:.4f}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.2, f"Total Epochs: {len(epochs)}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Training Summary', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training plots saved to {save_path}")
    return fig

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="results/confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Normalized Count'})
    
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add counts to each cell
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            plt.text(j + 0.5, i + 0.7, f'({cm[i, j]})', 
                    ha='center', va='center', fontsize=8, color='red')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    return plt.gcf()

def plot_class_performance(y_true, y_pred, class_names, save_path="results/class_performance.png"):
    """Plot per-class performance metrics"""
    from sklearn.metrics import precision_recall_fscore_support
    
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics_df = pd.DataFrame({
        'Class': class_names,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].bar(range(len(class_names)), precision, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Precision by Class', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Classes')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_xticks(range(len(class_names)))
    axes[0, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(len(class_names)), recall, color='lightcoral', alpha=0.7)
    axes[0, 1].set_title('Recall by Class', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('Recall')
    axes[0, 1].set_xticks(range(len(class_names)))
    axes[0, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].bar(range(len(class_names)), f1, color='lightgreen', alpha=0.7)
    axes[1, 0].set_title('F1-Score by Class', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Classes')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].set_xticks(range(len(class_names)))
    axes[1, 0].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].bar(range(len(class_names)), support, color='gold', alpha=0.7)
    axes[1, 1].set_title('Support (Number of Samples) by Class', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Number of Samples')
    axes[1, 1].set_xticks(range(len(class_names)))
    axes[1, 1].set_xticklabels(class_names, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class performance plots saved to {save_path}")
    return fig

def comprehensive_test_evaluation(model, test_loader, class_names, save_dir="results"):
    """Comprehensive evaluation on test set with visualizations"""
    print("=" * 60)
    print("COMPREHENSIVE TEST SET EVALUATION")
    print("=" * 60)
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    # Collect predictions
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(DEVICE)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate comprehensive metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )
    
    # Print detailed results
    print(f"\nOVERALL TEST PERFORMANCE:")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1 Score: {f1:.4f}")
    
    print(f"\nPER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name:25s} - Precision: {per_class_precision[i]:.4f}, "
              f"Recall: {per_class_recall[i]:.4f}, F1: {per_class_f1[i]:.4f}, "
              f"Support: {per_class_support[i]:3d}")
    
    # Detailed classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Generate visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names, 
                         os.path.join(save_dir, "test_confusion_matrix.png"))
    
    # Plot per-class performance
    plot_class_performance(all_labels, all_predictions, class_names,
                          os.path.join(save_dir, "test_class_performance.png"))
    
    # Save detailed results
    test_results = {
        'overall_metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'per_class_metrics': {
            'class_names': class_names,
            'precision': per_class_precision.tolist(),
            'recall': per_class_recall.tolist(),
            'f1': per_class_f1.tolist(),
            'support': per_class_support.tolist()
        },
        'predictions': {
            'true_labels': all_labels.tolist(),
            'predicted_labels': all_predictions.tolist(),
            'prediction_probabilities': all_probabilities.tolist()
        }
    }
    
    # Save comprehensive results
    with open(os.path.join(save_dir, "comprehensive_test_results.json"), 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"\nComprehensive test results saved to {save_dir}/")
    
    return test_results

def train_resnet_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
                      save_path="results/resnet_derm7pt.pt", class_weights=None,
                      use_focal=False, focal_gamma=2.0, mixup_alpha=0.0, label_smoothing=0.0,
                      grad_clip=None):
    """Train ResNet model on Derm7pt dataset"""
    
    base_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE)) if class_weights is not None else nn.CrossEntropyLoss()

    def smooth_labels(targets, num_classes, smoothing):
        with torch.no_grad():
            true_dist = torch.zeros((targets.size(0), num_classes), device=targets.device)
            true_dist.fill_(smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.data.unsqueeze(1), 1.0 - smoothing)
        return true_dist

    def focal_loss(inputs, targets_soft, gamma=2.0):
        probs = torch.softmax(inputs, dim=1)
        ce = - (targets_soft * torch.log(probs + 1e-9)).sum(dim=1)
        p_t = (probs * targets_soft).sum(dim=1)
        loss = ((1 - p_t) ** gamma) * ce
        return loss.mean()
    
    params_to_update = [
        {'params': [p for name, p in model.named_parameters() if 'fc' not in name], 'lr': learning_rate * 0.1},
        {'params': model.backbone.fc.parameters(), 'lr': learning_rate}
    ]
    
    optimizer = optim.Adam(params_to_update, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    best_val_balanced_acc = -float('inf')
    best_val_acc = 0.0
    best_model_state = None
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_balanced_accs = []
    val_macro_f1s = []
    learning_rates = []
    
    print("Starting training...")
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            if mixup_alpha and mixup_alpha > 0.0:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                batch_size = images.size(0)
                index = torch.randperm(batch_size).to(DEVICE)
                mixed_images = lam * images + (1 - lam) * images[index]
                labels_a, labels_b = labels, labels[index]
                outputs = model(mixed_images)
                num_classes = outputs.size(1)
                targets_a = smooth_labels(labels_a, num_classes, label_smoothing) if label_smoothing > 0 else torch.nn.functional.one_hot(labels_a, num_classes).float()
                targets_b = smooth_labels(labels_b, num_classes, label_smoothing) if label_smoothing > 0 else torch.nn.functional.one_hot(labels_b, num_classes).float()
                targets_a = targets_a.to(DEVICE)
                targets_b = targets_b.to(DEVICE)
                targets = lam * targets_a + (1 - lam) * targets_b
                if use_focal:
                    loss = focal_loss(outputs, targets, gamma=focal_gamma)
                else:
                    log_probs = torch.log_softmax(outputs, dim=1)
                    loss = - (targets * log_probs).sum(dim=1).mean()
                current_preds = torch.argmax(outputs, dim=1)
            else:
                outputs = model(images)
                if label_smoothing and label_smoothing > 0.0:
                    num_classes = outputs.size(1)
                    targets = smooth_labels(labels, num_classes, label_smoothing)
                    if use_focal:
                        loss = focal_loss(outputs, targets, gamma=focal_gamma)
                    else:
                        log_probs = torch.log_softmax(outputs, dim=1)
                        loss = - (targets * log_probs).sum(dim=1).mean()
                else:
                    if use_focal:
                        num_classes = outputs.size(1)
                        targets = torch.nn.functional.one_hot(labels, num_classes).float().to(DEVICE)
                        loss = focal_loss(outputs, targets, gamma=focal_gamma)
                    else:
                        loss = base_criterion(outputs, labels)
                current_preds = torch.argmax(outputs, dim=1)

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct_predictions / total_samples:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct_predictions / total_samples
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_pbar:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(images)
                if label_smoothing and label_smoothing > 0.0:
                    num_classes = outputs.size(1)
                    targets = smooth_labels(labels, num_classes, label_smoothing)
                    log_probs = torch.log_softmax(outputs, dim=1)
                    loss = - (targets * log_probs).sum(dim=1).mean()
                elif use_focal:
                    num_classes = outputs.size(1)
                    targets = torch.nn.functional.one_hot(labels, num_classes).float().to(DEVICE)
                    loss = focal_loss(outputs, targets, gamma=focal_gamma)
                else:
                    loss = base_criterion(outputs, labels)
                
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                val_predictions.append(predicted.cpu())
                val_targets.append(labels.cpu())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * val_correct / val_total:.2f}%'
                })
        
        val_loss = val_running_loss / len(val_loader)
        val_acc = val_correct / val_total
        if val_predictions:
            val_predictions_tensor = torch.cat(val_predictions).numpy()
            val_targets_tensor = torch.cat(val_targets).numpy()
            val_balanced_acc = balanced_accuracy_score(val_targets_tensor, val_predictions_tensor)
            val_macro_f1 = f1_score(val_targets_tensor, val_predictions_tensor, average='macro', zero_division=0)
        else:
            val_balanced_acc = 0.0
            val_macro_f1 = 0.0
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_balanced_accs.append(val_balanced_acc)
        val_macro_f1s.append(val_macro_f1)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Balanced Acc: {val_balanced_acc:.4f}, Val Macro F1: {val_macro_f1:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ⭐ New best validation balanced accuracy: {best_val_balanced_acc:.4f}")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_model_state, save_path)
    print(f"\nBest model saved to {save_path}")
    print(f"Best validation balanced accuracy: {best_val_balanced_acc:.4f}")
    print(f"Corresponding validation accuracy: {best_val_acc:.4f}")
    
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'val_balanced_accs': val_balanced_accs,
        'val_macro_f1s': val_macro_f1s,
        'learning_rates': learning_rates,
        'best_val_acc': best_val_acc,
        'best_val_balanced_acc': best_val_balanced_acc
    }
    
    history_path = save_path.replace('.pt', '_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_history(history, save_path.replace('.pt', '_training_plots.png'))
    
    return best_val_acc, history

def main():
    parser = argparse.ArgumentParser(description='Train ResNet on Derm7pt dataset')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['resnet18', 'resnet34', 'resnet50'],
                       help='ResNet model variant')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--use_focal', action='store_true', help='Use focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal loss gamma')
    parser.add_argument('--mixup_alpha', type=float, default=0.0, help='Mixup alpha (0 disables)')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='Label smoothing factor (0 disables)')
    parser.add_argument('--grad_clip', type=float, default=None, help='Gradient clipping norm (None disables)')
    parser.add_argument('--unfreeze_backbone', action='store_true', help='Unfreeze backbone for fine-tuning')
    
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    print(f"Training {args.model} for {args.epochs} epochs")
    
    os.makedirs("results", exist_ok=True)
    
    if not os.path.exists("dataset/meta/meta.csv"):
        raise FileNotFoundError("Could not find dataset/meta/meta.csv")
    
    meta = pd.read_csv("dataset/meta/meta.csv")
    print(f"Loaded metadata with {len(meta)} samples")
    
    try:
        train_idx = pd.read_csv("dataset/meta/all.csv")["indexes"].tolist()
        val_idx = pd.read_csv("dataset/meta/valid_indexes.csv")["indexes"].tolist()
        test_idx = pd.read_csv("dataset/meta/test_indexes.csv")["indexes"].tolist()
        print("Loaded existing data splits")
        
        max_index = len(meta) - 1
        train_idx_valid = [idx for idx in train_idx if 0 <= idx <= max_index]
        val_idx_valid = [idx for idx in val_idx if 0 <= idx <= max_index]
        test_idx_valid = [idx for idx in test_idx if 0 <= idx <= max_index]
        
        if len(train_idx_valid) < len(train_idx):
            print(f"Warning: Removed {len(train_idx) - len(train_idx_valid)} out-of-bounds train indices")
        if len(val_idx_valid) < len(val_idx):
            print(f"Warning: Removed {len(val_idx) - len(val_idx_valid)} out-of-bounds validation indices")
        if len(test_idx_valid) < len(test_idx):
            print(f"Warning: Removed {len(test_idx) - len(test_idx_valid)} out-of-bounds test indices")
        
        train_idx, val_idx, test_idx = train_idx_valid, val_idx_valid, test_idx_valid
        
        if len(train_idx) < 100 or len(val_idx) < 20 or len(test_idx) < 20:
            print("Too many invalid indices, creating new data splits...")
            train_idx, val_idx, test_idx = create_data_splits(meta)
            
    except FileNotFoundError:
        print("Creating new data splits...")
        train_idx, val_idx, test_idx = create_data_splits(meta)
    except Exception as e:
        print(f"Error loading existing splits: {e}")
        print("Creating new data splits...")
        train_idx, val_idx, test_idx = create_data_splits(meta)
    
    try:
        train_meta = meta.iloc[train_idx].reset_index(drop=True)
        val_meta = meta.iloc[val_idx].reset_index(drop=True)
        test_meta = meta.iloc[test_idx].reset_index(drop=True)
    except Exception as e:
        print(f"Error creating data splits: {e}")
        print("Creating new splits...")
        train_idx, val_idx, test_idx = create_data_splits(meta)
        train_meta = meta.iloc[train_idx].reset_index(drop=True)
        val_meta = meta.iloc[val_idx].reset_index(drop=True)
        test_meta = meta.iloc[test_idx].reset_index(drop=True)
    
    print(f"Data splits - Train: {len(train_meta)}, Val: {len(val_meta)}, Test: {len(test_meta)}")
    
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = Derm7ptDataset(train_meta, 'dataset/images', transform=train_transform)
    val_dataset = Derm7ptDataset(val_meta, 'dataset/images', transform=val_test_transform)
    test_dataset = Derm7ptDataset(test_meta, 'dataset/images', transform=val_test_transform)

    class_weights = torch.ones(len(train_dataset.diagnosis_classes), dtype=torch.float32)
    sample_weights = torch.ones(len(train_meta), dtype=torch.double)
    if len(train_meta) > 0:
        class_counts = train_meta['diagnosis'].value_counts()
        total_count = float(len(train_meta))
        num_classes = float(len(train_dataset.diagnosis_classes))
        for class_name, idx in train_dataset.class_to_idx.items():
            count = float(class_counts.get(class_name, 0.0))
            if count > 0.0:
                raw_weight = total_count / (num_classes * count)
                adjusted_weight = raw_weight ** 0.5
                class_weights[idx] = float(torch.clamp(torch.tensor(adjusted_weight), 0.25, 4.0))
            else:
                class_weights[idx] = 0.0
        sample_weights = torch.tensor([
            class_weights[train_dataset.class_to_idx[diagnosis]].item()
            for diagnosis in train_meta['diagnosis']
        ], dtype=torch.double)
        print("Class distribution in training split:")
        for class_name, idx in train_dataset.class_to_idx.items():
            count = int(class_counts.get(class_name, 0))
            weight = class_weights[idx].item()
            print(f"  {class_name:25s}: {count:4d} samples -> weight {weight:.4f}")

    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                           num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)
    
    num_classes = len(train_dataset.diagnosis_classes)
    model = ResNetClassifier(model_name=args.model, num_classes=num_classes, pretrained=True)
    model = model.to(DEVICE)
    
    print(f"Model: {args.model} with {num_classes} classes")
    print(f"Classes: {train_dataset.diagnosis_classes}")
    
    save_path = f"results/{args.model}_derm7pt.pt"
    if args.unfreeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen for full fine-tuning")

    best_val_acc, history = train_resnet_model(
        model, train_loader, val_loader, 
        num_epochs=args.epochs, 
        learning_rate=args.lr,
        save_path=save_path,
        class_weights=class_weights,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing,
        grad_clip=args.grad_clip
    )
    
    model.load_state_dict(torch.load(save_path))
    
    comprehensive_results = comprehensive_test_evaluation(
        model, test_loader, train_dataset.diagnosis_classes, save_dir="results"
    )
    
    results = {
        'model_name': args.model,
        'num_classes': num_classes,
        'class_names': train_dataset.diagnosis_classes,
        'best_val_acc': best_val_acc,
        'training_history': history,
        'comprehensive_test_results': comprehensive_results,
        'training_args': vars(args)
    }
    
    results_path = f"results/{args.model}_derm7pt_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"📁 Model saved to: {save_path}")
    print(f"📊 Results saved to: {results_path}")
    print(f"📈 Training plots: {save_path.replace('.pt', '_training_plots.png')}")
    print(f"🎯 Test confusion matrix: results/test_confusion_matrix.png")
    print(f"📋 Class performance: results/test_class_performance.png")

if __name__ == "__main__":
    main()
