import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
import json
import glob
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DirectResNetClassifier(nn.Module):
    """Direct ResNet classifier for diagnosis prediction"""
    
    def __init__(self, pretrained_model_path, num_classes):
        super(DirectResNetClassifier, self).__init__()
        
        # Load the pre-trained ResNet model
        checkpoint = torch.load(pretrained_model_path, map_location='cpu')
        
        # Create ResNet18 architecture
        self.backbone = models.resnet18(pretrained=False)
        
        # Load the pre-trained weights
        self.backbone.load_state_dict(checkpoint, strict=False)
        
        # Replace final layer for new number of classes if needed
        if self.backbone.fc.out_features != num_classes:
            self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
            print(f"Replaced final layer: {self.backbone.fc.in_features} -> {num_classes} classes")
        
        print(f"Loaded ResNet18 for direct diagnosis prediction with {num_classes} classes")
        
    def forward(self, x):
        return self.backbone(x)

def load_direct_resnet_model():
    """Load the pre-trained ResNet for direct diagnosis"""
    try:
        with open("results/resnet18_derm7pt_results.json", "r") as f:
            resnet_results = json.load(f)
        
        class_names = resnet_results['class_names']
        num_classes = len(class_names)
        
        model = DirectResNetClassifier("results/resnet18_derm7pt.pt", num_classes)
        model = model.to(DEVICE)
        model.eval()
        
        print(f"Loaded direct ResNet model with {num_classes} classes")
        print(f"Classes: {class_names}")
        
        return model, class_names
        
    except Exception as e:
        print(f"Error loading direct ResNet model: {e}")
        return None, []

def predict_with_resnet(model, img_path, class_names):
    """Make direct diagnosis prediction with ResNet"""
    
    # Image transformation (same as training)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load and preprocess image
    if isinstance(img_path, str):
        img = Image.open(img_path).convert('RGB')
    else:
        img = img_path  # Already PIL Image
    
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()
        
        # Get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            {
                "diagnosis": class_names[idx.item()],
                "confidence": prob.item()
            }
            for idx, prob in zip(top3_indices, top3_probs)
        ]
    
    return {
        "predicted_diagnosis": class_names[predicted_class_idx],
        "confidence": confidence,
        "top3_predictions": top3_predictions,
        "all_probabilities": probabilities[0].cpu().numpy().tolist()
    }

def test_resnet_on_sample_images():
    """Test the direct ResNet on some sample images"""
    
    print("Loading direct ResNet model...")
    model, class_names = load_direct_resnet_model()
    
    if model is None:
        print("Failed to load model. Make sure you have trained the ResNet model first.")
        return
    
    meta = pd.read_csv("dataset/meta/meta.csv")
    
    sample_indices = np.random.choice(len(meta), min(5, len(meta)), replace=False)
    
    print(f"\n{'='*80}")
    print("DIRECT RESNET PREDICTIONS ON SAMPLE IMAGES")
    print("="*80)
    
    for i, idx in enumerate(sample_indices):
        row = meta.iloc[idx]
        image_filename = os.path.basename(row['clinic']).lower()
        true_diagnosis = row['diagnosis']
        
        possible_paths = [
            os.path.join('dataset/images', image_filename),
            os.path.join('dataset/images', image_filename.replace('.jpg', '.png')),
            os.path.join('dataset/images', image_filename.replace('.png', '.jpg'))
        ]
        
        image_path = None
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            search_pattern = os.path.join('dataset/images', '**', image_filename)
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                image_path = found_files[0]
        
        if image_path and os.path.exists(image_path):
            try:
                result = predict_with_resnet(model, image_path, class_names)
                
                print(f"\nSample {i+1}: {image_filename}")
                print(f"True Diagnosis: {true_diagnosis}")
                print(f"Predicted: {result['predicted_diagnosis']}")
                print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.1f}%)")
                
                is_correct = result['predicted_diagnosis'] == true_diagnosis
                print(f"Correct: {'✅ YES' if is_correct else '❌ NO'}")
                
                print("Top 3 predictions:")
                for j, pred in enumerate(result['top3_predictions']):
                    marker = "👑" if j == 0 else f"{j+1}."
                    print(f"  {marker} {pred['diagnosis']}: {pred['confidence']:.4f} ({pred['confidence']*100:.1f}%)")
                
            except Exception as e:
                print(f"\nError processing {image_filename}: {e}")
        else:
            print(f"\nCould not find image: {image_filename}")

def compare_resnet_vs_cbm():
    """Compare direct ResNet predictions with CBM predictions"""
    
    print("Loading models for comparison...")
    
    # Load direct ResNet
    resnet_model, resnet_classes = load_direct_resnet_model()
    
    # Load CBM
    try:
        from train_cbm_resnet import PretrainedConceptPredictor, DiagnosisPredictor, ConceptBottleneckModel
        
        with open("results/cbm_results.json", "r") as f:
            cbm_config = json.load(f)
        
        cbm_classes = cbm_config['model_info']['class_names']
        
        # Create CBM
        concept_predictor = PretrainedConceptPredictor(
            cbm_config['model_info']['pretrained_path'], 
            num_concepts=cbm_config['model_info']['num_concepts'],
            freeze_backbone=True
        ).to(DEVICE)
        
        diagnosis_predictor = DiagnosisPredictor(
            num_concepts=cbm_config['model_info']['num_concepts'],
            num_classes=cbm_config['model_info']['num_classes']
        ).to(DEVICE)
        
        concept_predictor.load_state_dict(torch.load("results/cbm_concept_predictor.pt", map_location=DEVICE))
        diagnosis_predictor.load_state_dict(torch.load("results/cbm_diagnosis_predictor.pt", map_location=DEVICE))
        
        cbm = ConceptBottleneckModel(concept_predictor, diagnosis_predictor)
        cbm.eval()
        
        print("Both models loaded successfully!")
        
    except Exception as e:
        print(f"Could not load CBM: {e}")
        print("Running ResNet-only analysis...")
        test_resnet_on_sample_images()
        return
    
    # Test on sample images
    meta = pd.read_csv("dataset/meta/meta.csv")
    sample_indices = np.random.choice(len(meta), min(3, len(meta)), replace=False)
    
    print(f"\n{'='*100}")
    print("COMPARISON: DIRECT RESNET vs CONCEPT BOTTLENECK MODEL")
    print("="*100)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i, idx in enumerate(sample_indices):
        row = meta.iloc[idx]
        image_filename = os.path.basename(row['clinic']).lower()
        true_diagnosis = row['diagnosis']
        
        # Find image
        image_path = None
        possible_paths = [
            os.path.join('dataset/images', image_filename),
            os.path.join('dataset/images', image_filename.replace('.jpg', '.png')),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                image_path = path
                break
        
        if not image_path:
            search_pattern = os.path.join('dataset/images', '**', image_filename)
            found_files = glob.glob(search_pattern, recursive=True)
            if found_files:
                image_path = found_files[0]
        
        if image_path and os.path.exists(image_path):
            try:
                img = Image.open(image_path).convert('RGB')
                input_tensor = transform(img).unsqueeze(0).to(DEVICE)
                
                print(f"\n{'─'*60}")
                print(f"Sample {i+1}: {image_filename}")
                print(f"True Diagnosis: {true_diagnosis}")
                print("─"*60)
                
                # ResNet prediction
                resnet_result = predict_with_resnet(resnet_model, img, resnet_classes)
                
                # CBM prediction  
                with torch.no_grad():
                    cbm_logits, concepts = cbm(input_tensor, return_concepts=True)
                    cbm_probs = torch.softmax(cbm_logits, dim=1)
                    cbm_pred_idx = torch.argmax(cbm_probs, dim=1).item()
                    cbm_confidence = cbm_probs[0, cbm_pred_idx].item()
                
                print(f"🤖 Direct ResNet:")
                print(f"   Prediction: {resnet_result['predicted_diagnosis']}")
                print(f"   Confidence: {resnet_result['confidence']:.4f} ({resnet_result['confidence']*100:.1f}%)")
                
                print(f"🧠 CBM (via concepts):")
                print(f"   Prediction: {cbm_classes[cbm_pred_idx]}")
                print(f"   Confidence: {cbm_confidence:.4f} ({cbm_confidence*100:.1f}%)")
                print(f"   Key concepts: {concepts[0].cpu().numpy()[:3]}")  # Show first 3 concepts
                
                # Check which is correct
                resnet_correct = resnet_result['predicted_diagnosis'] == true_diagnosis
                cbm_correct = cbm_classes[cbm_pred_idx] == true_diagnosis
                
                print(f"\n📊 Results:")
                print(f"   ResNet correct: {'✅' if resnet_correct else '❌'}")
                print(f"   CBM correct: {'✅' if cbm_correct else '❌'}")
                
                if resnet_result['predicted_diagnosis'] == cbm_classes[cbm_pred_idx]:
                    print(f"   Agreement: ✅ Both models agree")
                else:
                    print(f"   Agreement: ❌ Models disagree")
                
            except Exception as e:
                print(f"Error processing {image_filename}: {e}")

if __name__ == "__main__":
    print("Direct ResNet Testing")
    print("====================")
    
    print("\n1. Testing direct ResNet predictions:")
    test_resnet_on_sample_images()
    
    print("\n2. Comparing ResNet vs CBM:")
    compare_resnet_vs_cbm()
