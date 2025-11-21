import streamlit as st
import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import cv2
import json
import io
import time
import torchvision.models as models
import requests

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
except Exception as e:
    # Handle other import errors (e.g., version incompatibility)
    print(f"Gemini import warning: {e}")
    GEMINI_AVAILABLE = False
    genai = None

# Backend configuration
CHAT_BACKEND_URL = "http://localhost:5000"
USE_BACKEND = True  # Set to True to use Flask backend instead of direct genai

from train_cbm_resnet import PretrainedConceptPredictor, DiagnosisPredictor, ConceptBottleneckModel

# Set page config
st.set_page_config(
    page_title="Enhanced CBM Skin Cancer Diagnosis",
    page_icon="🔬",
    layout="wide"
)

# Define concept names for visualization
CONCEPT_NAMES = [
    "Pigment Network",
    "Blue-Whitish Veil", 
    "Vascular Structures",
    "Pigmentation",
    "Streaks/Pseudopods",
    "Dots/Globules",
    "Regression Structures"
]

# Initialize Gemini
def init_gemini_chat():
    """Initialize Gemini chat model with API key from environment or user input"""
    
    # Check if backend is available
    if USE_BACKEND:
        try:
            response = requests.get(f"{CHAT_BACKEND_URL}/health", timeout=2)
            if response.status_code == 200:
                return "backend", None  # Return "backend" as model indicator
            else:
                return None, f"Chat backend unhealthy: {response.status_code}"
        except requests.exceptions.ConnectionError:
            return None, f"Chat backend not running. Start it with: python chat_backend.py (Python 3.9+ required)"
        except Exception as e:
            return None, f"Error connecting to chat backend: {str(e)}"
    
    # Fallback to direct genai (requires Python 3.9+)
    if not GEMINI_AVAILABLE:
        return None, "google-generativeai not installed or incompatible. Requires Python 3.9+. Install with: pip install google-generativeai"
    
    # Use hardcoded API key, then check session state, then environment
    api_key = "AIzaSyCVfYG8HFdh_Os8pWuzbLDH-J-r75V67sk"
    if not api_key:
        api_key = st.session_state.get('gemini_api_key', os.getenv('GEMINI_API_KEY', ''))
    
    if not api_key:
        return None, "No API key provided"
    
    try:
        genai.configure(api_key=api_key)
        # Try the correct model initialization method
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
        except AttributeError:
            # Fallback for older API versions
            import google.generativeai.types as genai_types
            model = genai.GenerativeModel(model_name='gemini-pro')
        return model, None
    except Exception as e:
        return None, f"Error initializing Gemini: {str(e)}. Note: Requires Python 3.9+ and google-generativeai>=0.3.0"

def create_diagnosis_context(explanation, resnet_result=None):
    """Create a formatted context string from diagnosis results for Gemini"""
    context_parts = []
    
    # CBM Results
    if explanation:
        context_parts.append("=== CBM (Concept Bottleneck Model) Analysis ===")
        context_parts.append(f"Predicted Diagnosis: {explanation.get('diagnosis', 'Unknown')}")
        context_parts.append(f"Confidence: {explanation.get('diagnosis_probability', 0):.3f} ({explanation.get('diagnosis_probability', 0)*100:.1f}%)")
        
        context_parts.append("\n--- Detected Concepts ---")
        for i, concept_value in enumerate(explanation.get('predicted_concepts', [])):
            concept_name = CONCEPT_NAMES[i] if i < len(CONCEPT_NAMES) else f"Concept {i}"
            level = "High" if concept_value >= 0.7 else "Medium" if concept_value >= 0.5 else "Low-Medium" if concept_value >= 0.3 else "Low"
            context_parts.append(f"- {concept_name}: {concept_value:.3f} ({level} activation)")
        
        context_parts.append("\n--- Concept Influences ---")
        for inf in explanation.get('concept_influences', [])[:5]:  # Top 5
            context_parts.append(f"- {inf['concept_name']}: influence {inf['probability_influence']:.4f}")
    
    # ResNet Results
    if resnet_result:
        context_parts.append("\n=== Direct ResNet Analysis ===")
        context_parts.append(f"Predicted Diagnosis: {resnet_result.get('predicted_diagnosis', 'Unknown')}")
        context_parts.append(f"Confidence: {resnet_result.get('confidence', 0):.3f} ({resnet_result.get('confidence', 0)*100:.1f}%)")
        
        if resnet_result.get('top3_predictions'):
            context_parts.append("\nTop 3 Predictions:")
            for i, pred in enumerate(resnet_result['top3_predictions'][:3]):
                context_parts.append(f"{i+1}. {pred['diagnosis']}: {pred['confidence']:.3f}")
    
    return "\n".join(context_parts)

def generate_diagnosis_report(cbm_explanation, resnet_result, image, model_used):
    """Generate a comprehensive text report for download"""
    from datetime import datetime
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("DERMOSCOPIC IMAGE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Model(s) Used: {model_used}")
    report_lines.append(f"Image Size: {image.size[0]}x{image.size[1]} pixels")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # CBM Results
    if cbm_explanation:
        report_lines.append("━" * 80)
        report_lines.append("CBM (CONCEPT BOTTLENECK MODEL) ANALYSIS")
        report_lines.append("━" * 80)
        report_lines.append("")
        
        # Diagnosis
        diagnosis = cbm_explanation.get('diagnosis', 'Unknown')
        confidence = cbm_explanation.get('diagnosis_probability', 0)
        report_lines.append(f"DIAGNOSIS: {diagnosis}")
        report_lines.append(f"CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
        report_lines.append("")
        
        # Confidence interpretation
        if confidence >= 0.8:
            conf_level = "High confidence"
        elif confidence >= 0.6:
            conf_level = "Moderate confidence"
        else:
            conf_level = "Low confidence - further examination recommended"
        report_lines.append(f"Confidence Level: {conf_level}")
        report_lines.append("")
        
        # Detected Concepts
        report_lines.append("-" * 80)
        report_lines.append("DETECTED DERMATOLOGICAL CONCEPTS")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        predicted_concepts = cbm_explanation.get('predicted_concepts', [])
        for i, concept_value in enumerate(predicted_concepts):
            if i < len(CONCEPT_NAMES):
                concept_name = CONCEPT_NAMES[i]
                
                # Activation level
                if concept_value >= 0.7:
                    level = "HIGH"
                    indicator = "+++"
                elif concept_value >= 0.5:
                    level = "MEDIUM-HIGH"
                    indicator = "++"
                elif concept_value >= 0.3:
                    level = "MEDIUM"
                    indicator = "+"
                else:
                    level = "LOW"
                    indicator = "-"
                
                report_lines.append(f"{concept_name:25s} [{indicator:3s}] {concept_value:.4f} ({level})")
        
        report_lines.append("")
        
        # Concept Influences
        report_lines.append("-" * 80)
        report_lines.append("CONCEPT INFLUENCE ON DIAGNOSIS")
        report_lines.append("-" * 80)
        report_lines.append("")
        
        influences = cbm_explanation.get('concept_influences', [])
        sorted_influences = sorted(influences, key=lambda x: abs(x.get('probability_influence', 0)), reverse=True)
        
        report_lines.append("Top Influential Concepts (ranked by impact):")
        report_lines.append("")
        for rank, inf in enumerate(sorted_influences[:7], 1):
            concept_name = inf.get('concept_name', 'Unknown')
            influence = inf.get('probability_influence', 0)
            binary_inf = "YES" if inf.get('binary_influence', False) else "NO"
            
            influence_dir = "↑" if influence > 0 else "↓" if influence < 0 else "→"
            report_lines.append(f"{rank}. {concept_name:25s} {influence_dir} {influence:+.6f}  (Binary: {binary_inf})")
        
        report_lines.append("")
    
    # ResNet Results
    if resnet_result:
        report_lines.append("━" * 80)
        report_lines.append("DIRECT RESNET MODEL ANALYSIS")
        report_lines.append("━" * 80)
        report_lines.append("")
        
        diagnosis = resnet_result.get('predicted_diagnosis', 'Unknown')
        confidence = resnet_result.get('confidence', 0)
        report_lines.append(f"DIAGNOSIS: {diagnosis}")
        report_lines.append(f"CONFIDENCE: {confidence:.4f} ({confidence*100:.2f}%)")
        report_lines.append("")
        
        # Top predictions
        top3 = resnet_result.get('top3_predictions', [])
        if top3:
            report_lines.append("-" * 80)
            report_lines.append("TOP 3 DIFFERENTIAL DIAGNOSES")
            report_lines.append("-" * 80)
            report_lines.append("")
            
            for i, pred in enumerate(top3[:3], 1):
                medal = ["🥇", "🥈", "🥉"][i-1]
                diag = pred.get('diagnosis', 'Unknown')
                conf = pred.get('confidence', 0)
                report_lines.append(f"{medal} Rank {i}: {diag}")
                report_lines.append(f"   Confidence: {conf:.4f} ({conf*100:.2f}%)")
                report_lines.append("")
    
    # Model Comparison
    if cbm_explanation and resnet_result:
        report_lines.append("━" * 80)
        report_lines.append("MODEL COMPARISON")
        report_lines.append("━" * 80)
        report_lines.append("")
        
        cbm_diag = cbm_explanation.get('diagnosis', '')
        resnet_diag = resnet_result.get('predicted_diagnosis', '')
        cbm_conf = cbm_explanation.get('diagnosis_probability', 0)
        resnet_conf = resnet_result.get('confidence', 0)
        
        agreement = "YES ✓" if cbm_diag == resnet_diag else "NO ✗"
        conf_diff = abs(cbm_conf - resnet_conf)
        
        report_lines.append(f"Models Agree: {agreement}")
        report_lines.append(f"CBM Prediction: {cbm_diag} ({cbm_conf:.4f})")
        report_lines.append(f"ResNet Prediction: {resnet_diag} ({resnet_conf:.4f})")
        report_lines.append(f"Confidence Difference: {conf_diff:.4f}")
        report_lines.append("")
        
        if cbm_diag != resnet_diag:
            report_lines.append("⚠ CLINICAL NOTE: Models disagree. Consider:")
            report_lines.append("  - Reviewing both diagnoses carefully")
            report_lines.append("  - Checking concept activations for CBM reasoning")
            report_lines.append("  - Seeking additional clinical evaluation")
        else:
            report_lines.append("✓ CLINICAL NOTE: Both models agree, increasing confidence.")
        report_lines.append("")
    
    # Disclaimer
    report_lines.append("=" * 80)
    report_lines.append("DISCLAIMER")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("This is an AI-assisted analysis tool for educational and research purposes.")
    report_lines.append("It is NOT a substitute for professional medical diagnosis.")
    report_lines.append("Always consult a qualified dermatologist for definitive diagnosis and treatment.")
    report_lines.append("")
    report_lines.append("Recommendations:")
    report_lines.append("- Use this report as supplementary information only")
    report_lines.append("- Seek professional evaluation for any skin lesions of concern")
    report_lines.append("- Regular skin examinations by a dermatologist are recommended")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)

def chat_with_doctor_bot(model, diagnosis_context, user_message, chat_history):
    """Send message to Gemini and get response"""
    try:
        # Check if using backend
        if model == "backend":
            # Route through Flask backend
            full_prompt = f"""You are Dr. Bot, an AI dermatology assistant helping to interpret skin lesion diagnoses. You have access to the following analysis results from AI models:

{diagnosis_context}

Your role is to:
1. Explain the diagnosis results in simple, patient-friendly language
2. Describe what each concept means in dermatology
3. Answer questions about the diagnosis, prognosis, and next steps
4. Provide educational information about skin conditions
5. Always remind users that this is AI-assisted analysis and they should consult a real dermatologist

Chat History:
{chat_history}

Patient Question: {user_message}

Dr. Bot Response:"""
            
            response = requests.post(
                f"{CHAT_BACKEND_URL}/chat",
                json={"prompt": full_prompt},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                error_detail = response.json().get("error", "Unknown error")
                return f"❌ Chat backend error: {error_detail}"
                
        else:
            # Use direct genai model
            full_prompt = f"""You are Dr. Bot, an AI dermatology assistant helping to interpret skin lesion diagnoses. You have access to the following analysis results from AI models:

{diagnosis_context}

Your role is to:
1. Explain the diagnosis results in simple, patient-friendly language
2. Describe what each concept means in dermatology
3. Answer questions about the diagnosis, prognosis, and next steps
4. Provide educational information about skin conditions
5. Always remind users that this is AI-assisted analysis and they should consult a real dermatologist

Chat History:
{chat_history}

Patient Question: {user_message}

Dr. Bot Response:"""
            
            response = model.generate_content(full_prompt)
            return response.text
            
    except requests.exceptions.Timeout:
        return "❌ Chat request timed out. The backend may be overloaded."
    except requests.exceptions.ConnectionError:
        return "❌ Cannot connect to chat backend. Make sure it's running with: python chat_backend.py"
    except Exception as e:
        return f"Error communicating with Dr. Bot: {str(e)}"

@st.cache_resource
def load_cbm_model():
    """Load the trained CBM model with ResNet backbone"""
    try:
        # Load CBM results to get model configuration
        with open("results/cbm_results.json", "r") as f:
            cbm_config = json.load(f)
        
        num_concepts = cbm_config['model_info']['num_concepts']
        num_classes = cbm_config['model_info']['num_classes']
        class_names = cbm_config['model_info']['class_names']
        pretrained_path = cbm_config['model_info']['pretrained_path']
        
        print(f"Loading CBM with {num_concepts} concepts and {num_classes} classes")
        
        # Create concept predictor
        concept_predictor = PretrainedConceptPredictor(
            pretrained_path, 
            num_concepts=num_concepts,
            freeze_backbone=True
        )
        
        # Create diagnosis predictor
        diagnosis_predictor = DiagnosisPredictor(
            num_concepts=num_concepts,
            num_classes=num_classes
        )
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            concept_state = torch.load("results/cbm_concept_predictor.pt", map_location=device, weights_only=True)
            diagnosis_state = torch.load("results/cbm_diagnosis_predictor.pt", map_location=device, weights_only=True)
        except TypeError:
            concept_state = torch.load("results/cbm_concept_predictor.pt", map_location=device)
            diagnosis_state = torch.load("results/cbm_diagnosis_predictor.pt", map_location=device)
        
        concept_predictor.load_state_dict(concept_state)
        diagnosis_predictor.load_state_dict(diagnosis_state)
        
        # Create CBM
        cbm = ConceptBottleneckModel(concept_predictor, diagnosis_predictor)
        cbm = cbm.to(device)
        cbm.eval()
        
        return cbm, class_names, device
        
    except Exception as e:
        st.error(f"Error loading CBM model: {e}")
        return None, [], torch.device("cpu")

@st.cache_resource
def load_direct_resnet_model():
    """Load the direct ResNet model for comparison"""
    try:
        # Load the ResNet results to get class information
        with open("results/resnet18_derm7pt_results.json", "r") as f:
            resnet_results = json.load(f)
        
        class_names = resnet_results['class_names']
        num_classes = len(class_names)
        
        # Create ResNet model using the same architecture as training
        from train_resnet_derm7pt import ResNetClassifier
        model = ResNetClassifier(model_name='resnet18', num_classes=num_classes, pretrained=False)
        
        # Load trained weights
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the state dict with flexible key mapping
        try:
            state_dict = torch.load("results/resnet18_derm7pt.pt", map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load("results/resnet18_derm7pt.pt", map_location=device)
        
        # Handle different checkpoint formats
        if 'classifier.weight' in state_dict and 'backbone.fc.weight' not in state_dict:
            # Old format: rename classifier -> backbone.fc
            state_dict['backbone.fc.weight'] = state_dict.pop('classifier.weight')
            state_dict['backbone.fc.bias'] = state_dict.pop('classifier.bias')
        
        model.load_state_dict(state_dict)
        
        model = model.to(device)
        model.eval()
        
        print(f"Loaded direct ResNet model with {num_classes} classes")
        print(f"Classes: {class_names}")
        
        return model, class_names, device
        
    except Exception as e:
        print(f"Error loading direct ResNet model: {e}")
        print(f"Make sure you have trained the ResNet model first by running: python train_resnet_derm7pt.py")
        return None, [], torch.device("cpu")

def generate_ice_explanation(cbm, input_tensor, device):
    """Generate ICE-style explanations for the CBM model"""
    with torch.no_grad():
        # Get original prediction
        diagnosis_logits, predicted_concepts = cbm(input_tensor, return_concepts=True)
        diagnosis_probs = torch.softmax(diagnosis_logits, dim=1)
        diagnosis_pred = torch.argmax(diagnosis_probs, dim=1).item()
        diagnosis_prob = diagnosis_probs[0, diagnosis_pred].item()
        
        concept_influences = []
        
        for concept_idx in range(predicted_concepts.shape[1]):
            # Create modified concept vector with this concept zeroed out
            modified_concepts = predicted_concepts.clone()
            modified_concepts[0, concept_idx] = 0.0
            
            # Get new diagnosis prediction with modified concepts
            modified_logits = cbm.diagnosis_predictor(modified_concepts)
            modified_probs = torch.softmax(modified_logits, dim=1)
            modified_pred = torch.argmax(modified_probs, dim=1).item()
            
            # Calculate influence
            binary_influence = 1 if modified_pred != diagnosis_pred else 0
            prob_influence = (diagnosis_probs[0, diagnosis_pred] - modified_probs[0, diagnosis_pred]).item()
            
            concept_influences.append({
                "concept_idx": concept_idx,
                "concept_name": CONCEPT_NAMES[concept_idx] if concept_idx < len(CONCEPT_NAMES) else f"Concept {concept_idx}",
                "concept_value": predicted_concepts[0, concept_idx].item(),
                "binary_influence": binary_influence,
                "probability_influence": prob_influence
            })
        
        # Sort by absolute influence
        concept_influences.sort(key=lambda x: abs(x["probability_influence"]), reverse=True)
        
        return {
            "diagnosis_index": diagnosis_pred,
            "diagnosis_probability": diagnosis_prob,
            "predicted_concepts": predicted_concepts[0].tolist(),
            "concept_influences": concept_influences
        }

def generate_concept_visualizations(cbm, img, input_tensor, device):
    """Generate concept activation visualizations using CAM"""
    img_np = np.array(img)
    
    with torch.no_grad():
        cbm.eval()
        
        # Get concept predictions first
        concept_probs = cbm.concept_predictor(input_tensor)
        
        # Hook to capture feature maps from the backbone
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Register hook to capture the last convolutional features before global avg pooling
        hook_handle = None
        
        # Try to find the layer4 output in the ResNet backbone
        try:
            # Access the backbone through the feature extractor
            for name, module in cbm.concept_predictor.feature_extractor.named_modules():
                if 'layer4' in name and isinstance(module, torch.nn.Sequential):
                    # Hook the entire layer4 block
                    hook_handle = module.register_forward_hook(get_activation('layer4_features'))
                    print(f"Hooked layer4 block")
                    break
                elif name.endswith('layer4'):
                    hook_handle = module.register_forward_hook(get_activation('layer4_features'))
                    print(f"Hooked {name}")
                    break
        except:
            pass
        
        # If we couldn't hook layer4 specifically, try the entire feature extractor
        if hook_handle is None:
            # Get the second-to-last layer (before AdaptiveAvgPool2d)
            layers = list(cbm.concept_predictor.feature_extractor.children())
            if len(layers) >= 2:
                hook_handle = layers[-2].register_forward_hook(get_activation('layer4_features'))
                print("Hooked second-to-last layer of feature extractor")
        
        # Forward pass to capture feature maps
        _ = cbm.concept_predictor(input_tensor)
        
        # Remove hook
        if hook_handle:
            hook_handle.remove()
        
        concept_visualizations = []
        
        if 'layer4_features' in activation:
            feature_maps = activation['layer4_features']
            print(f"Captured feature maps shape: {feature_maps.shape}")
            
            # Ensure we have the right shape [B, C, H, W]
            if len(feature_maps.shape) == 4:
                feature_maps = feature_maps[0]  # Remove batch dimension: [C, H, W]
            else:
                print(f"Unexpected feature map shape: {feature_maps.shape}")
                # Create fallback visualizations
                for concept_idx in range(concept_probs.shape[1]):
                    concept_visualizations.append({
                        'concept_idx': concept_idx,
                        'concept_name': CONCEPT_NAMES[concept_idx] if concept_idx < len(CONCEPT_NAMES) else f"Concept {concept_idx}",
                        'heatmap': img_np,
                        'focus_map': img_np,
                        'concept_value': concept_probs[0, concept_idx].item(),
                        'cam_max': 0.0,
                        'cam_mean': 0.0
                    })
                return concept_visualizations
            
            # Get concept layer weights (raw_weight with softplus applied)
            raw_weights = cbm.concept_predictor.raw_weight.data  # [num_concepts, feature_dim]
            concept_weights = torch.nn.functional.softplus(raw_weights)  # Apply softplus to get actual weights
            
            # For ResNet18, after layer4 we should have [512, 7, 7] feature maps
            C, H, W = feature_maps.shape
            print(f"Feature maps: C={C}, H={H}, W={W}")
            print(f"Concept weights shape: {concept_weights.shape}")
            
            # Generate CAM for each concept
            for concept_idx in range(concept_probs.shape[1]):
                try:
                    # Get weights for this concept
                    weights = concept_weights[concept_idx]  # [512] for ResNet18
                    
                    # Ensure weights match feature map channels
                    if weights.shape[0] != C:
                        print(f"Warning: Weight dimension {weights.shape[0]} doesn't match feature channels {C}")
                        # Use average pooling as fallback
                        concept_cam = torch.mean(feature_maps, dim=0)  # [H, W]
                    else:
                        # Compute weighted feature map
                        weights_expanded = weights.view(C, 1, 1)  # [C, 1, 1]
                        weighted_features = weights_expanded * feature_maps  # [C, H, W]
                        concept_cam = torch.sum(weighted_features, dim=0)  # [H, W]
                    
                    # Apply ReLU to focus on positive activations
                    concept_cam = torch.clamp(concept_cam, min=0)
                    
                    # Convert to numpy for processing
                    concept_cam_np = concept_cam.cpu().numpy()
                    
                    # Check for invalid values
                    if np.isnan(concept_cam_np).any() or np.isinf(concept_cam_np).any():
                        print(f"Warning: Invalid values in CAM for concept {concept_idx}")
                        concept_cam_np = np.zeros_like(concept_cam_np)
                    
                    # Normalize CAM
                    cam_min = concept_cam_np.min()
                    cam_max = concept_cam_np.max()
                    
                    if cam_max > cam_min:
                        concept_cam_normalized = (concept_cam_np - cam_min) / (cam_max - cam_min)
                    else:
                        concept_cam_normalized = np.zeros_like(concept_cam_np)
                    
                    # Store original statistics
                    original_max = float(cam_max)
                    original_mean = float(concept_cam_np.mean())
                    
                    # Resize CAM to image size
                    concept_cam_resized = cv2.resize(concept_cam_normalized, (img.width, img.height))
                    
                    # Create heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * concept_cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Create blended image (heatmap overlay)
                    alpha = 0.6
                    blended = (1 - alpha) * img_np + alpha * heatmap
                    blended = np.clip(blended, 0, 255).astype(np.uint8)
                    
                    # Create focus map
                    threshold = 0.3
                    important_regions = concept_cam_resized > threshold
                    
                    # Create dimmed version
                    dimmed_img = img_np.copy().astype(float) * 0.2
                    focus_img = dimmed_img.copy()
                    
                    # Restore original intensity in important regions
                    focus_img[important_regions] = img_np[important_regions]
                    
                    # Add red border around important regions
                    if np.any(important_regions):
                        kernel = np.ones((2, 2), np.uint8)
                        dilated_mask = cv2.dilate(important_regions.astype(np.uint8), kernel, iterations=1)
                        border_mask = dilated_mask - important_regions.astype(np.uint8)
                        focus_img[border_mask.astype(bool)] = [255, 0, 0]
                    
                    focus_img = np.clip(focus_img, 0, 255).astype(np.uint8)
                    
                    concept_visualizations.append({
                        'concept_idx': concept_idx,
                        'concept_name': CONCEPT_NAMES[concept_idx] if concept_idx < len(CONCEPT_NAMES) else f"Concept {concept_idx}",
                        'heatmap': blended,
                        'focus_map': focus_img,
                        'concept_value': concept_probs[0, concept_idx].item(),
                        'cam_max': original_max,
                        'cam_mean': original_mean
                    })
                    
                except Exception as e:
                    print(f"Error processing concept {concept_idx}: {e}")
                    # Create safe fallback
                    concept_visualizations.append({
                        'concept_idx': concept_idx,
                        'concept_name': CONCEPT_NAMES[concept_idx] if concept_idx < len(CONCEPT_NAMES) else f"Concept {concept_idx}",
                        'heatmap': img_np,
                        'focus_map': img_np,
                        'concept_value': concept_probs[0, concept_idx].item(),
                        'cam_max': 0.0,
                        'cam_mean': 0.0
                    })
        
        else:
            print("Could not capture feature maps, creating fallback visualizations")
            # Create simple fallback visualizations based on concept activations
            for concept_idx in range(concept_probs.shape[1]):
                concept_activation = concept_probs[0, concept_idx].item()
                
                # Create a simple radial gradient based on concept activation
                h, w = img.height, img.width
                center_y, center_x = h // 2, w // 2
                
                y, x = np.ogrid[:h, :w]
                # FIX: Change from *2 to **2 for proper distance calculation
                distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                max_distance = np.sqrt(center_x**2 + center_y**2)
                normalized_distance = distance_from_center / max_distance
                
                # Create activation map
                activation_map = concept_activation * (1 - normalized_distance * 0.5)
                activation_map = np.clip(activation_map, 0, 1)
                
                # Create heatmap
                heatmap = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                # Blend with original
                alpha = 0.4
                blended = (1 - alpha) * img_np + alpha * heatmap
                blended = np.clip(blended, 0, 255).astype(np.uint8)
                
                # Create focus map
                threshold = 0.5 * concept_activation
                important_regions = activation_map > threshold
                
                dimmed_img = img_np.copy().astype(float) * 0.3
                focus_img = dimmed_img.copy()
                focus_img[important_regions] = img_np[important_regions]
                focus_img = np.clip(focus_img, 0, 255).astype(np.uint8)
                
                concept_visualizations.append({
                    'concept_idx': concept_idx,
                    'concept_name': CONCEPT_NAMES[concept_idx] if concept_idx < len(CONCEPT_NAMES) else f"Concept {concept_idx}",
                    'heatmap': blended,
                    'focus_map': focus_img,
                    'concept_value': concept_activation,
                    'cam_max': float(activation_map.max()),
                    'cam_mean': float(activation_map.mean())
                })
    
    return concept_visualizations

def process_uploaded_image(img, cbm, class_names, device):
    """Process uploaded image and generate explanations"""
    # Image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Transform image
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Generate ICE explanations
    explanation = generate_ice_explanation(cbm, input_tensor, device)
    explanation['diagnosis'] = class_names[explanation['diagnosis_index']]
    
    # Generate concept visualizations
    concept_visualizations = generate_concept_visualizations(cbm, img, input_tensor, device)
    explanation['concept_visualizations'] = concept_visualizations
    
    return explanation

def plot_concept_influence(explanation):
    """Plot concept influence chart"""
    concept_names = [inf["concept_name"] for inf in explanation["concept_influences"]]
    influences = [inf["probability_influence"] for inf in explanation["concept_influences"]]
    concept_values = [inf["concept_value"] for inf in explanation["concept_influences"]]
    
    # Sort by absolute influence
    sort_idx = np.argsort(np.abs(influences))[::-1]
    sorted_names = [concept_names[i] for i in sort_idx]
    sorted_influences = [influences[i] for i in sort_idx]
    sorted_values = [concept_values[i] for i in sort_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot influences
    bars = ax.barh(sorted_names, sorted_influences, 
                  color=['red' if i < 0 else 'green' for i in sorted_influences])
    
    # Add concept values as text
    for i, (bar, value) in enumerate(zip(bars, sorted_values)):
        ax.text(0.01, bar.get_y() + bar.get_height()/2, f"Act: {value:.2f}", 
               va='center', fontsize=9, color='white', weight='bold')
    
    ax.set_title('Concept Influence on Diagnosis', fontsize=14, fontweight='bold')
    ax.set_xlabel('Influence Value (change in probability)', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig

def plot_concept_visualization(concept_data, viz_type="heatmap"):
    """Plot concept visualization with enhanced information"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if viz_type == "heatmap":
        ax.imshow(concept_data['heatmap'])
        title = f"Heatmap: {concept_data['concept_name']}"
        description = "Red/yellow areas show high concept activation"
    else:  # focus map
        ax.imshow(concept_data['focus_map'])
        title = f"Focus Map: {concept_data['concept_name']}"
        description = "Highlighted regions are most important for this concept"
    
    # Enhanced title with activation metrics
    activation = concept_data['concept_value']
    cam_max = concept_data.get('cam_max', 0)
    cam_mean = concept_data.get('cam_mean', 0)
    
    title_text = f"{title}\nActivation: {activation:.3f} | CAM Max: {cam_max:.3f} | CAM Mean: {cam_mean:.3f}"
    
    ax.set_title(title_text, fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add description at the bottom
    plt.figtext(0.5, 0.02, description, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    return fig

def plot_concept_predictions(explanation):
    """Plot all concept predictions as a bar chart"""
    concept_names = [CONCEPT_NAMES[i] if i < len(CONCEPT_NAMES) else f"Concept {i}" 
                    for i in range(len(explanation["predicted_concepts"]))]
    concept_values = explanation["predicted_concepts"]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create color map based on activation level
    colors = []
    for val in concept_values:
        if val >= 0.7:
            colors.append('#2E8B57')  # Dark green for high activation
        elif val >= 0.5:
            colors.append('#FFA500')  # Orange for medium activation
        elif val >= 0.3:
            colors.append('#FFD700')  # Gold for low-medium activation
        else:
            colors.append('#CD5C5C')  # Light red for low activation
    
    # Create horizontal bar chart
    bars = ax.barh(concept_names, concept_values, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, concept_values)):
        ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}', 
               va='center', fontsize=10, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Concept Activation Level', fontsize=12, fontweight='bold')
    ax.set_title('All Concept Predictions for Uploaded Image', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.1)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add reference lines
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    ax.axvline(x=0.7, color='green', linestyle='--', alpha=0.5, label='High Activation (0.7)')
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='#2E8B57', alpha=0.7, label='High (≥0.7)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFA500', alpha=0.7, label='Medium (0.5-0.7)'),
        plt.Rectangle((0,0),1,1, facecolor='#FFD700', alpha=0.7, label='Low-Medium (0.3-0.5)'),
        plt.Rectangle((0,0),1,1, facecolor='#CD5C5C', alpha=0.7, label='Low (<0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    return fig

def predict_with_direct_resnet(model, img, class_names, device):
    """Make direct diagnosis prediction with ResNet"""
    try:
        # Image transformation (same as training)
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Transform image
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class_idx = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_idx].item()
            
            # Get top 3 predictions
            top3_probs, top3_indices = torch.topk(probabilities[0], min(3, len(class_names)))
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
        
    except Exception as e:
        print(f"Error in ResNet prediction: {e}")
        return {
            "predicted_diagnosis": "Error in prediction",
            "confidence": 0.0,
            "top3_predictions": [],
            "all_probabilities": []
        }

def main():
    st.title("🔬 Enhanced CBM Skin Cancer Diagnosis")
    st.markdown("### Using ResNet-based Concept Bottleneck Model With LLM Explanations")
    
    # Load both models with better error handling
    st.sidebar.markdown("### Model Status")
    
    # Load CBM model
    cbm, cbm_class_names, cbm_device = load_cbm_model()
    cbm_available = cbm is not None
    
    if cbm_available:
        st.sidebar.success("✅ CBM Model Loaded")
    else:
        st.sidebar.error("❌ CBM Model Failed")
    
    # Load ResNet model
    resnet_model, resnet_class_names, resnet_device = load_direct_resnet_model()
    resnet_available = resnet_model is not None
    
    if resnet_available:
        st.sidebar.success("✅ ResNet Model Loaded")
    else:
        st.sidebar.error("❌ ResNet Model Failed")
    
    # Check chat backend status
    st.sidebar.markdown("### Chat Backend Status")
    if USE_BACKEND:
        try:
            response = requests.get(f"{CHAT_BACKEND_URL}/health", timeout=2)
            if response.status_code == 200:
                st.sidebar.success("✅ Chat Backend Online")
            else:
                st.sidebar.warning(f"⚠️ Chat Backend Unhealthy ({response.status_code})")
        except requests.exceptions.ConnectionError:
            st.sidebar.error("❌ Chat Backend Offline")
            st.sidebar.info("Start with: `python chat_backend.py` (Python 3.9+ required)")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Backend Check Error: {str(e)[:50]}")
    else:
        if GEMINI_AVAILABLE:
            st.sidebar.info("ℹ️ Using Direct Gemini API")
        else:
            st.sidebar.warning("⚠️ Chat Unavailable (Python 3.9+ required)")
    
    # Check if at least one model is available
    if not cbm_available and not resnet_available:
        st.error("❌ **No models available!**")
        st.markdown("""
        **To use this application, you need to train at least one model:**
        
        1. **For ResNet model**: Run `python train_resnet_derm7pt.py`
        2. **For CBM model**: Run `python train_cbm_resnet.py`
        
        Make sure your dataset is properly organized in the `dataset/` folder.
        """)
        st.stop()
    
    # Model availability status
    cbm_available = cbm is not None
    resnet_available = resnet_model is not None
    
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose Mode", 
        ["🏠 Home", "📤 Upload & Diagnose", "🤖 Chat with Dr. Bot", "❓ Help"]
    )
    
    if app_mode == "🏠 Home":
        st.markdown("""
        ## Welcome to Enhanced CBM Diagnosis System
        
        This application uses a state-of-the-art *Concept Bottleneck Model (CBM)* built on a ResNet backbone 
        for transparent skin cancer diagnosis from dermoscopic images.
        
        ### 🔧 Model Architecture
        - *Feature Extractor*: Pre-trained ResNet18 fine-tuned on dermoscopic images
        - *Concept Predictor*: Neural network identifying 7 dermatological concepts
        - *Diagnosis Predictor*: Classification based on predicted concepts
        - *Explanation Method*: Individual Conditional Expectation (ICE) analysis
        
        ### 🎯 Key Features
        - *Transparent Predictions*: See exactly which concepts influence the diagnosis
        - *Concept Visualization*: Visual heatmaps showing where concepts are detected
        - *Focus Maps*: Highlight only the most important image regions
        - *Quantitative Explanations*: Numerical influence scores for each concept
        
        ### 📋 The 7 Dermatological Concepts
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            1. *Pigment Network*: Net-like pattern of pigmented lines
            2. *Blue-Whitish Veil*: Irregular blue pigmentation areas
            3. *Vascular Structures*: Visible blood vessels
            4. *Pigmentation*: Brown to black coloration
            """)
        
        with col2:
            st.markdown("""
            5. *Streaks/Pseudopods*: Finger-like projections at periphery
            6. *Dots/Globules*: Round structures, usually brown/black
            7. *Regression Structures*: White scar-like depigmentation
            """)
        
        st.markdown("### 🚀 Get Started")
        st.info("Select '📤 Upload & Diagnose' from the sidebar to analyze your dermoscopic images!")
        
    elif app_mode == "📤 Upload & Diagnose":
        st.markdown("## Upload Dermoscopic Image for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a dermoscopic image...", 
            type=["jpg", "jpeg", "png"],
            help="Upload a high-quality dermoscopic image for analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            img = Image.open(uploaded_file)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(img, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                # Model selection
                st.markdown("### 🤖 Select Prediction Model")
                
                available_models = []
                if cbm_available:
                    available_models.append("CBM (Concept Bottleneck Model)")
                if resnet_available:
                    available_models.append("Direct ResNet")
                if cbm_available and resnet_available:
                    available_models.append("Both Models (Side-by-Side)")
                
                if not available_models:
                    st.error("No models available for prediction!")
                    st.stop()
                
                selected_model = st.selectbox(
                    "Choose prediction model:",
                    available_models,
                    help="CBM provides interpretable predictions via concepts, Direct ResNet is faster but less explainable"
                )
                
                # Add debug information
                st.markdown("**Debug Info:**")
                st.write(f"CBM Available: {cbm_available}")
                st.write(f"ResNet Available: {resnet_available}")
                st.write(f"Selected Model: {selected_model}")
                
                with st.spinner("🔍 Analyzing image... This may take a moment."):
                    try:
                        if selected_model == "CBM (Concept Bottleneck Model)" and cbm_available:
                            # CBM prediction only
                            explanation = process_uploaded_image(img, cbm, cbm_class_names, cbm_device)
                            show_cbm_results = True
                            show_resnet_results = False
                            show_both_results = False
                            
                        elif selected_model == "Direct ResNet" and resnet_available:
                            # ResNet prediction only
                            st.write("Calling ResNet prediction...")
                            resnet_result = predict_with_direct_resnet(resnet_model, img, resnet_class_names, resnet_device)
                            st.write(f"ResNet result: {resnet_result}")
                            show_cbm_results = False
                            show_resnet_results = True
                            show_both_results = False
                            
                        elif selected_model == "Both Models (Side-by-Side)" and cbm_available and resnet_available:
                            # Both models
                            explanation = process_uploaded_image(img, cbm, cbm_class_names, cbm_device)
                            resnet_result = predict_with_direct_resnet(resnet_model, img, resnet_class_names, resnet_device)
                            show_cbm_results = True
                            show_resnet_results = True
                            show_both_results = True
                        
                        else:
                            st.error(f"Invalid model selection or model not available: {selected_model}")
                            st.stop()
                    
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
                        st.write("**Error details:**")
                        st.code(str(e))
                        st.stop()
                
                # Display results based on selection
                if show_both_results:
                    st.markdown("### 🎯 Model Comparison")
                    
                    # Side-by-side comparison
                    comp_col1, comp_col2 = st.columns(2)
                    
                    with comp_col1:
                        st.markdown("#### 🧠 CBM Prediction")
                        cbm_diagnosis = explanation['diagnosis']
                        cbm_confidence = explanation['diagnosis_probability']
                        
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; background-color: #e8f4fd; margin: 5px 0;">
                            <h4 style="color: #1f77b4; margin: 0;">CBM Result:</h4>
                            <h3 style="color: #333; margin: 5px 0;">{cbm_diagnosis}</h3>
                            <p style="margin: 0;">Confidence: {cbm_confidence:.3f} ({cbm_confidence*100:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with comp_col2:
                        st.markdown("#### 🤖 Direct ResNet")
                        resnet_diagnosis = resnet_result['predicted_diagnosis']
                        resnet_confidence = resnet_result['confidence']
                        
                        st.markdown(f"""
                        <div style="padding: 15px; border-radius: 8px; background-color: #f0f8e8; margin: 5px 0;">
                            <h4 style="color: #2d7d32; margin: 0;">ResNet Result:</h4>
                            <h3 style="color: #333; margin: 5px 0;">{resnet_diagnosis}</h3>
                            <p style="margin: 0;">Confidence: {resnet_confidence:.3f} ({resnet_confidence*100:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Agreement analysis
                    models_agree = cbm_diagnosis == resnet_diagnosis
                    confidence_diff = abs(cbm_confidence - resnet_confidence)
                    
                    st.markdown("#### 📊 Analysis")
                    agreement_col1, agreement_col2, agreement_col3 = st.columns(3)
                    
                    with agreement_col1:
                        agreement_icon = "✅" if models_agree else "❌"
                        st.metric("Models Agree", f"{agreement_icon} {'Yes' if models_agree else 'No'}")
                    
                    with agreement_col2:
                        st.metric("Confidence Difference", f"{confidence_diff:.3f}")
                    
                    with agreement_col3:
                        higher_conf = "CBM" if cbm_confidence > resnet_confidence else "ResNet" if resnet_confidence > cbm_confidence else "Equal"
                        st.metric("Higher Confidence", higher_conf)
                    
                    if not models_agree:
                        st.warning("⚠️ The models disagree on the diagnosis. Consider the confidence levels and use clinical judgment.")
                    else:
                        st.success("✅ Both models agree on the diagnosis, increasing confidence in the prediction.")
                    
                    # Show ResNet top 3 predictions
                    st.markdown("#### 🏆 ResNet Top 3 Predictions")
                    for i, pred in enumerate(resnet_result['top3_predictions']):
                        rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                        st.write(f"{rank_icon} **{pred['diagnosis']}**: {pred['confidence']:.3f} ({pred['confidence']*100:.1f}%)")
                
                elif show_resnet_results:
                    st.markdown("### 🎯 ResNet Diagnosis Results")
                    
                    if resnet_result and "predicted_diagnosis" in resnet_result:
                        resnet_diagnosis = resnet_result['predicted_diagnosis']
                        resnet_confidence = resnet_result['confidence']
                        
                        st.markdown(f"""
                        <div style="padding: 20px; border-radius: 10px; background-color: #f0f8e8; margin: 10px 0;">
                            <h3 style="color: #2d7d32; margin: 0;">Direct ResNet Prediction:</h3>
                            <h2 style="color: #333; margin: 5px 0;">{resnet_diagnosis}</h2>
                            <p style="margin: 5px 0; font-size: 16px;">Confidence: {resnet_confidence:.3f} ({resnet_confidence*100:.1f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show top 3 predictions
                        if resnet_result['top3_predictions']:
                            st.markdown("#### 🏆 Top 3 Predictions")
                            for i, pred in enumerate(resnet_result['top3_predictions']):
                                rank_icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                                st.write(f"{rank_icon} **{pred['diagnosis']}**: {pred['confidence']:.3f} ({pred['confidence']*100:.1f}%)")
                        
                        st.info("💡 **Direct ResNet**: Fast prediction but no concept explanations available.")
                    else:
                        st.error("Failed to get valid prediction from ResNet model")
                        st.write("ResNet result:", resnet_result)
                    
                elif show_cbm_results:
                    st.markdown("### 🎯 CBM Diagnosis Results")
                    
                    diagnosis = explanation['diagnosis']
                    confidence = explanation['diagnosis_probability']
                    
                    st.markdown(f"""
                    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0;">
                        <h3 style="color: #1f77b4; margin: 0;">CBM Prediction:</h3>
                        <h2 style="color: #333; margin: 5px 0;">{diagnosis}</h2>
                        <p style="margin: 5px 0; font-size: 16px;">Confidence: {confidence:.3f} ({confidence*100:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Show CBM explanations only if CBM was used
            if (show_cbm_results or show_both_results) and cbm_available:
                # Concept Predictions Section
                st.markdown("### 🧠 Concept Predictions")
                
                # Display concept predictions chart
                fig_concepts = plot_concept_predictions(explanation)
                st.pyplot(fig_concepts)
                
                # Display concept predictions in a table format
                st.markdown("#### 📋 Detailed Concept Analysis")
                
                concept_data = []
                for i, concept_value in enumerate(explanation["predicted_concepts"]):
                    concept_name = CONCEPT_NAMES[i] if i < len(CONCEPT_NAMES) else f"Concept {i}"
                    
                    # Determine activation level
                    if concept_value >= 0.7:
                        level = "🟢 High"
                    elif concept_value >= 0.5:
                        level = "🟡 Medium"
                    elif concept_value >= 0.3:
                        level = "🟠 Low-Medium"
                    else:
                        level = "🔴 Low"
                    
                    concept_data.append({
                        "Concept": concept_name,
                        "Activation": f"{concept_value:.3f}",
                        "Level": level,
                        "Description": get_concept_description(concept_name)
                    })
                
                # Create DataFrame and display as table
                concept_df = pd.DataFrame(concept_data)
                
                # Display table without styling (to avoid pandas/jinja2 import issues)
                st.dataframe(concept_df, use_container_width=True)
                
                # Summary statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    high_concepts = sum(1 for val in explanation["predicted_concepts"] if val >= 0.7)
                    st.metric("High Activation Concepts", high_concepts)
                
                with col2:
                    medium_concepts = sum(1 for val in explanation["predicted_concepts"] if 0.5 <= val < 0.7)
                    st.metric("Medium Activation Concepts", medium_concepts)
                
                with col3:
                    avg_activation = sum(explanation["predicted_concepts"]) / len(explanation["predicted_concepts"])
                    st.metric("Average Activation", f"{avg_activation:.3f}")
                
                # Concept influence analysis
                st.markdown("### 📊 Concept Influence Analysis")
                
                fig_influence = plot_concept_influence(explanation)
                st.pyplot(fig_influence)
                
                st.markdown("""
                *How to interpret this chart:*
                - 🟢 *Green bars*: Concepts that support the current diagnosis
                - 🔴 *Red bars*: Concepts that, if removed, would change the diagnosis
                - *Act values*: How strongly each concept was detected (0-1 scale)
                - *Longer bars*: More influential concepts
                """)
                
                # Concept visualization section
                st.markdown("### 🔍 Concept Visualization")
                
                if explanation.get('concept_visualizations'):
                    # Visualization type selector
                    viz_type = st.radio(
                        "Choose visualization type:",
                        ["Heatmap Overlay", "Focus on Important Regions"],
                        help="Heatmap shows CAM activation intensity, Focus highlights important regions only"
                    )
                    
                    # Sort concepts by activation for better selection
                    sorted_concepts = sorted(
                        explanation['concept_visualizations'], 
                        key=lambda x: x['concept_value'], 
                        reverse=True
                    )
                    
                    # Concept selector with activation values
                    concept_options = [
                        f"{cv['concept_name']} (Act: {cv['concept_value']:.3f}, CAM: {cv.get('cam_max', 0):.3f})"
                        for cv in sorted_concepts
                    ]
                    
                    selected_concept_idx = st.selectbox(
                        "Select concept to visualize (sorted by activation):",
                        range(len(concept_options)),
                        format_func=lambda x: concept_options[x]
                    )
                    
                    # Display selected visualization
                    concept_data_viz = sorted_concepts[selected_concept_idx]
                    
                    if viz_type == "Heatmap Overlay":
                        fig_viz = plot_concept_visualization(concept_data_viz, "heatmap")
                    else:
                        fig_viz = plot_concept_visualization(concept_data_viz, "focus")
                    
                    st.pyplot(fig_viz)
                    
                    # Enhanced analysis information
                    original_idx = concept_data_viz['concept_idx']
                    influence_data = next(
                        (inf for inf in explanation['concept_influences'] 
                         if inf['concept_idx'] == original_idx), 
                        None
                    )
                    
                    if influence_data:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            *Concept Analysis:*
                            - *Activation Level*: {concept_data_viz['concept_value']:.3f}
                            - *CAM Maximum*: {concept_data_viz.get('cam_max', 0):.3f}
                            - *CAM Average*: {concept_data_viz.get('cam_mean', 0):.3f}
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            *Influence Analysis:*
                            - *Diagnosis Influence*: {influence_data['probability_influence']:.4f}
                            - *Binary Influence*: {'Yes' if influence_data['binary_influence'] else 'No'}
                            - *Rank by Influence*: {influence_data.get('rank', 'N/A')}
                            """)
                        
                        # Interpretation help
                        st.markdown(f"""
                        *Interpretation:*
                        - *High activation* ({concept_data_viz['concept_value']:.3f}) means this concept was strongly detected
                        - *CAM values* show the spatial concentration of concept detection
                        - *Influence score* ({influence_data['probability_influence']:.4f}) indicates impact on diagnosis
                        """)
                
                else:
                    st.warning("Could not generate concept visualizations. This may happen if the model architecture is not fully compatible with CAM generation.")
                    st.info("The model is still working for predictions, but visual explanations are limited.")
            
            # Dr. Bot Chatbot Section
            st.markdown("---")
            
            # Download Report Section
            st.markdown("### 📄 Download Diagnosis Report")
            
            report_text = generate_diagnosis_report(
                explanation if (show_cbm_results or show_both_results) else None,
                resnet_result if (show_resnet_results or show_both_results) else None,
                img,
                selected_model
            )
            
            col_download1, col_download2 = st.columns([1, 3])
            with col_download1:
                st.download_button(
                    label="📥 Download Report (TXT)",
                    data=report_text,
                    file_name=f"diagnosis_report_{uploaded_file.name.split('.')[0]}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col_download2:
                st.info("💡 Download a comprehensive text report with all predictions, concepts, and analysis details.")

    elif app_mode == "🤖 Chat with Dr. Bot":
        # Initialize session state for chatbot
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'chat_report_content' not in st.session_state:
            st.session_state.chat_report_content = None
        if 'chat_report_uploaded' not in st.session_state:
            st.session_state.chat_report_uploaded = False
        if 'chat_diagnosis_summary' not in st.session_state:
            st.session_state.chat_diagnosis_summary = None
        
        # Custom CSS for chatbot
        st.markdown("""
        <style>
            .user-message {
                background-color: #667eea;
                color: white;
                padding: 1rem 1.2rem;
                border-radius: 18px;
                margin: 0.5rem 0;
                margin-left: 20%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            }
            
            .bot-message {
                background-color: #1e2130;
                color: #e4e4e7;
                padding: 1rem 1.2rem;
                border-radius: 18px;
                margin: 0.5rem 0;
                margin-right: 20%;
                box-shadow: 0 2px 5px rgba(0,0,0,0.2);
                border-left: 3px solid #667eea;
            }
            
            .bot-message strong {
                color: #a78bfa;
                font-weight: 600;
            }
            
            .bot-message ul, .bot-message ol {
                margin: 0.5rem 0;
                padding-left: 1.5rem;
            }
            
            .bot-message li {
                margin: 0.3rem 0;
            }
            
            .chat-info-box {
                background-color: #1e2130;
                border-left: 4px solid #667eea;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            .chat-info-box-warning {
                background-color: #2d1f1f;
                border-left: 4px solid #ef4444;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            
            .status-online {
                background-color: #10b981;
                box-shadow: 0 0 10px #10b981;
            }
            
            .status-offline {
                background-color: #ef4444;
                box-shadow: 0 0 10px #ef4444;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Check backend status
        def check_chat_backend():
            try:
                response = requests.get(f"{CHAT_BACKEND_URL}/health", timeout=2)
                return response.status_code == 200
            except:
                return False
        
        # Extract diagnosis from report
        def extract_diagnosis_from_report(report):
            lines = report.split('\n')
            for line in lines:
                if 'DIAGNOSIS:' in line.upper():
                    return line.split(':', 1)[1].strip()
            return "Not found"
        
        # Format message with markdown
        def format_chat_message(text):
            import re
            
            lines = text.split('\n')
            formatted_lines = []
            in_list = False
            list_type = None
            
            for line in lines:
                stripped = line.strip()
                
                # Check for horizontal rule (---, ***, ___)
                if stripped in ('---', '***', '___') or re.match(r'^[-*_]{3,}$', stripped):
                    if in_list:
                        formatted_lines.append(f'</{list_type}>')
                        in_list = False
                    formatted_lines.append('<hr style="margin: 1rem 0; border: none; border-top: 1px solid #444;">')
                    continue
                
                # Check for headings (###, ##, #)
                heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
                if heading_match:
                    if in_list:
                        formatted_lines.append(f'</{list_type}>')
                        in_list = False
                    level = len(heading_match.group(1))
                    heading_text = heading_match.group(2)
                    # Apply bold/italic to heading text
                    heading_text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', heading_text)
                    heading_text = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', heading_text)
                    formatted_lines.append(f'<h{min(level+2, 6)} style="margin-top: 1rem; margin-bottom: 0.5rem;">{heading_text}</h{min(level+2, 6)}>')
                    continue
                
                # Check for bullet lists
                if stripped.startswith(('- ', '• ', '* ')):
                    if not in_list or list_type != 'ul':
                        if in_list:
                            formatted_lines.append(f'</{list_type}>')
                        formatted_lines.append('<ul>')
                        in_list = True
                        list_type = 'ul'
                    item = stripped[2:] if len(stripped) > 2 and stripped[1] == ' ' else stripped[1:]
                    # Apply formatting to list items
                    item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
                    item = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', item)
                    formatted_lines.append(f'<li>{item}</li>')
                    continue
                
                # Check for numbered lists
                numbered_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
                if numbered_match:
                    if not in_list or list_type != 'ol':
                        if in_list:
                            formatted_lines.append(f'</{list_type}>')
                        formatted_lines.append('<ol>')
                        in_list = True
                        list_type = 'ol'
                    item = numbered_match.group(2)
                    # Apply formatting to list items
                    item = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', item)
                    item = re.sub(r'\*([^*]+?)\*', r'<em>\1</em>', item)
                    formatted_lines.append(f'<li>{item}</li>')
                    continue
                
                # Regular text - close list if open
                if in_list:
                    formatted_lines.append(f'</{list_type}>')
                    in_list = False
                    list_type = None
                
                # Handle empty lines
                if not stripped:
                    formatted_lines.append('<br>')
                    continue
                
                # Apply formatting to regular text
                formatted_line = line
                # Bold text
                formatted_line = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', formatted_line)
                # Italic text (but not asterisks used as bullets)
                formatted_line = re.sub(r'(?<!\*)\*([^*\s][^*]*?[^*\s])\*(?!\*)', r'<em>\1</em>', formatted_line)
                formatted_lines.append(formatted_line + '<br>')
            
            # Close any open list
            if in_list:
                formatted_lines.append(f'</{list_type}>')
            
            return '\n'.join(formatted_lines)
        
        # Send message to backend
        def send_chat_message(message, report_content, conversation_history):
            history_text = '\n'.join([f"{msg['role']}: {msg['content']}" for msg in conversation_history])
            
            context = f"""You are Dr. Bot, an expert AI dermatology assistant specializing in skin lesion diagnosis interpretation. You have deep knowledge of dermatological concepts, diagnostic criteria, and clinical decision-making.

PATIENT'S DIAGNOSIS REPORT:
{report_content}

CONVERSATION HISTORY:
{history_text}

CURRENT PATIENT QUESTION:
"{message}"

YOUR ROLE AND RESPONSIBILITIES:
1. Analyze the diagnosis report thoroughly and provide context-aware responses
2. Explain dermatological concepts (e.g., Pigment Network, Blue-Whitish Veil, Atypical Vascular Pattern) in clear, patient-friendly language
3. Interpret confidence scores and what they mean for clinical decision-making
4. Discuss concept activations and their clinical significance
5. Provide actionable next steps while being appropriately cautious
6. Address patient concerns with empathy and clarity

RESPONSE GUIDELINES:
- Use **text** for bold emphasis on important medical terms or findings (markdown format)
- Use bullet points (- or •) or numbered lists (1. 2. 3.) for clarity when listing multiple items
- Break complex explanations into digestible paragraphs separated by blank lines
- Reference specific values from the report when relevant (e.g., "Your confidence score of 0.85...")
- If discussing multiple concepts, explain each one clearly
- Always maintain a professional yet compassionate tone
- DO NOT use HTML tags - use plain text with markdown formatting only

CRITICAL REMINDERS:
⚠️ Acknowledge the AI nature of the diagnosis system
⚠️ Emphasize that this is a screening/support tool, not a definitive diagnosis
⚠️ Always recommend consulting a board-certified dermatologist for:
   - Physical examination
   - Dermoscopy evaluation
   - Biopsy if indicated
   - Treatment planning
⚠️ For concerning findings (melanoma, high-risk lesions), stress urgency of professional evaluation

EXAMPLE RESPONSE STRUCTURE:
[Direct answer to patient's question]

[Detailed explanation with specific references to their report]

[Clinical context and what it means for them]

[Clear next steps and recommendations]

[Closing reminder about professional consultation]

Now provide a comprehensive, helpful response to the patient's question above:"""

            try:
                response = requests.post(
                    f"{CHAT_BACKEND_URL}/chat",
                    json={"prompt": context},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', 'No response received')
                else:
                    error_data = response.json()
                    return f"❌ Error: {error_data.get('error', 'Failed to get response')}"
            except requests.exceptions.Timeout:
                return "❌ Request timed out. The backend might be processing. Please try again."
            except requests.exceptions.ConnectionError:
                return "❌ Cannot connect to backend. Make sure the chat backend is running on port 5000.\n\nStart it with: python chat_backend.py"
            except Exception as e:
                return f"❌ Unexpected error: {str(e)}"
        
        # Header
        st.markdown("## 🤖 Chat with Dr. Bot")
        st.markdown("Your AI-powered dermatology assistant for understanding skin lesion diagnoses")
        
        # Check backend status
        backend_online = check_chat_backend()
        status_class = "status-online" if backend_online else "status-offline"
        status_text = "Backend Online" if backend_online else "Backend Offline"
        
        st.markdown(f"""
        <div style="text-align: right; margin-bottom: 1rem;">
            <span class="status-indicator {status_class}"></span>
            <span style="color: {'#10b981' if backend_online else '#ef4444'}; font-weight: 500;">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar - File upload
        with st.sidebar:
            st.markdown("### 📄 Upload Diagnosis Report")
            
            uploaded_report = st.file_uploader(
                "Choose a diagnosis report (TXT format)",
                type=['txt'],
                help="Upload the diagnosis report generated by the diagnosis mode",
                key="chat_report_uploader"
            )
            
            if uploaded_report is not None:
                report_content = uploaded_report.read().decode('utf-8')
                st.session_state.chat_report_content = report_content
                st.session_state.chat_report_uploaded = True
                
                # Extract diagnosis
                diagnosis = extract_diagnosis_from_report(report_content)
                st.session_state.chat_diagnosis_summary = diagnosis
                
                st.markdown(f"""
                <div class="chat-info-box">
                    <strong>📋 Report Loaded</strong><br>
                    <strong>Diagnosis:</strong> {diagnosis}
                </div>
                """, unsafe_allow_html=True)
                
                # Add initial bot message if no messages yet
                if len(st.session_state.chat_messages) == 0:
                    initial_message = f"Great! I've loaded your diagnosis report. I can see your diagnosis is: **{diagnosis}**\n\nFeel free to ask me anything about your diagnosis, the detected concepts, or what the results mean!"
                    st.session_state.chat_messages.append({
                        "role": "Dr. Bot",
                        "content": initial_message
                    })
            
            st.markdown("---")
            
            # Medical disclaimer
            st.markdown("""
            <div class="chat-info-box-warning">
                <strong>⚠️ Medical Disclaimer</strong><br>
                This is an AI assistant for educational purposes only. Always consult a qualified dermatologist for medical decisions.
            </div>
            """, unsafe_allow_html=True)
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History", use_container_width=True, key="clear_chat"):
                st.session_state.chat_messages = []
                if st.session_state.chat_report_uploaded:
                    diagnosis = st.session_state.chat_diagnosis_summary
                    initial_message = f"Great! I've loaded your diagnosis report. I can see your diagnosis is: **{diagnosis}**\n\nFeel free to ask me anything about your diagnosis, the detected concepts, or what the results mean!"
                    st.session_state.chat_messages.append({
                        "role": "Dr. Bot",
                        "content": initial_message
                    })
                st.rerun()
        
        # Main chat area
        if not st.session_state.chat_report_uploaded:
            st.markdown("""
            <div class="chat-info-box">
                <h3>👋 Welcome to Dr. Bot!</h3>
                <p>To get started, please upload your diagnosis report using the sidebar on the left.</p>
                <br>
                <strong>What I can help you with:</strong>
                <ul>
                    <li>📊 Explain your diagnosis results and confidence scores</li>
                    <li>🔍 Clarify detected dermatological concepts</li>
                    <li>💡 Discuss what your results mean clinically</li>
                    <li>📋 Provide guidance on next steps</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Display chat messages
            for message in st.session_state.chat_messages:
                if message["role"] == "You":
                    st.markdown(f"""
                    <div class="user-message">
                        {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    formatted_content = format_chat_message(message["content"])
                    st.markdown(f"""
                    <div class="bot-message">
                        {formatted_content}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Chat input
            st.markdown("<br>", unsafe_allow_html=True)
            
            if not backend_online:
                st.error("⚠️ Backend is offline. Please start the chat backend with: `python chat_backend.py`")
            
            # Suggested questions section
            if len(st.session_state.chat_messages) <= 1:  # Only show for new conversations
                st.markdown("### 💡 Suggested Questions")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📋 Explain my report in simple terms", key="q1", use_container_width=True, disabled=not backend_online):
                        st.session_state.chat_messages.append({"role": "You", "content": "Can you explain my diagnosis report in simple terms?"})
                        with st.spinner("Dr. Bot is thinking..."):
                            bot_response = send_chat_message(
                                "Can you explain my diagnosis report in simple terms?",
                                st.session_state.chat_report_content,
                                st.session_state.chat_messages[:-1]
                            )
                        st.session_state.chat_messages.append({"role": "Dr. Bot", "content": bot_response})
                        st.rerun()
                    
                    if st.button("🔬 What are the detected concepts?", key="q2", use_container_width=True, disabled=not backend_online):
                        st.session_state.chat_messages.append({"role": "You", "content": "What are the detected concepts in my report and what do they mean?"})
                        with st.spinner("Dr. Bot is thinking..."):
                            bot_response = send_chat_message(
                                "What are the detected concepts in my report and what do they mean?",
                                st.session_state.chat_report_content,
                                st.session_state.chat_messages[:-1]
                            )
                        st.session_state.chat_messages.append({"role": "Dr. Bot", "content": bot_response})
                        st.rerun()
                
                with col2:
                    if st.button("💊 What are the treatment options?", key="q3", use_container_width=True, disabled=not backend_online):
                        st.session_state.chat_messages.append({"role": "You", "content": "What are the typical treatment options for this diagnosis?"})
                        with st.spinner("Dr. Bot is thinking..."):
                            bot_response = send_chat_message(
                                "What are the typical treatment options for this diagnosis?",
                                st.session_state.chat_report_content,
                                st.session_state.chat_messages[:-1]
                            )
                        st.session_state.chat_messages.append({"role": "Dr. Bot", "content": bot_response})
                        st.rerun()
                    
                    if st.button("⚠️ How serious is this?", key="q4", use_container_width=True, disabled=not backend_online):
                        st.session_state.chat_messages.append({"role": "You", "content": "How serious is my diagnosis? What should I be concerned about?"})
                        with st.spinner("Dr. Bot is thinking..."):
                            bot_response = send_chat_message(
                                "How serious is my diagnosis? What should I be concerned about?",
                                st.session_state.chat_report_content,
                                st.session_state.chat_messages[:-1]
                            )
                        st.session_state.chat_messages.append({"role": "Dr. Bot", "content": bot_response})
                        st.rerun()
                
                with col3:
                    if st.button("📅 What are the next steps?", key="q5", use_container_width=True, disabled=not backend_online):
                        st.session_state.chat_messages.append({"role": "You", "content": "What should I do next? What are the recommended next steps?"})
                        with st.spinner("Dr. Bot is thinking..."):
                            bot_response = send_chat_message(
                                "What should I do next? What are the recommended next steps?",
                                st.session_state.chat_report_content,
                                st.session_state.chat_messages[:-1]
                            )
                        st.session_state.chat_messages.append({"role": "Dr. Bot", "content": bot_response})
                        st.rerun()
                    
                    if st.button("📊 Explain the confidence score", key="q6", use_container_width=True, disabled=not backend_online):
                        st.session_state.chat_messages.append({"role": "You", "content": "What does my confidence score mean? How reliable is this diagnosis?"})
                        with st.spinner("Dr. Bot is thinking..."):
                            bot_response = send_chat_message(
                                "What does my confidence score mean? How reliable is this diagnosis?",
                                st.session_state.chat_report_content,
                                st.session_state.chat_messages[:-1]
                            )
                        st.session_state.chat_messages.append({"role": "Dr. Bot", "content": bot_response})
                        st.rerun()
                
                st.markdown("<br>", unsafe_allow_html=True)
            
            with st.form(key="chat_form", clear_on_submit=True):
                user_input = st.text_input(
                    "Your question:",
                    placeholder="Ask me about your diagnosis...",
                    label_visibility="collapsed",
                    disabled=not backend_online
                )
                submit_button = st.form_submit_button("Send", disabled=not backend_online, use_container_width=True)
                
                if submit_button and user_input:
                    # Add user message
                    st.session_state.chat_messages.append({
                        "role": "You",
                        "content": user_input
                    })
                    
                    # Show typing indicator
                    with st.spinner("Dr. Bot is thinking..."):
                        # Get bot response
                        bot_response = send_chat_message(
                            user_input,
                            st.session_state.chat_report_content,
                            st.session_state.chat_messages[:-1]  # Exclude the just-added user message
                        )
                    
                    # Add bot response
                    st.session_state.chat_messages.append({
                        "role": "Dr. Bot",
                        "content": bot_response
                    })
                    
                    # Rerun to show new messages
                    st.rerun()

    elif app_mode == "📊 Model Information":
        st.markdown("## Model Performance & Architecture")
        
        # Try to load and display model metrics
        try:
            with open("results/cbm_results.json", "r") as f:
                cbm_results = json.load(f)
            
            # Model configuration
            st.markdown("### 🏗 Model Configuration")
            config = cbm_results['model_info']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                - *Base Model*: ResNet18
                - *Number of Concepts*: {config['num_concepts']}
                - *Number of Classes*: {config['num_classes']}
                - *Backbone Frozen*: {config['freeze_backbone']}
                """)
            
            with col2:
                device_info = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                st.markdown(f"""
                - *Pre-trained Path*: {os.path.basename(config['pretrained_path'])}
                - *Training Device*: {device_info.type.upper()}
                - *Model Files*: CBM concept + diagnosis predictors
                """)
            
            # Performance metrics
            if 'evaluation_results' in cbm_results:
                st.markdown("### 📈 Performance Metrics")
                
                eval_results = cbm_results['evaluation_results']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Concept Prediction")
                    concept_metrics = eval_results['concept_metrics']
                    st.markdown(f"""
                    - *Accuracy*: {concept_metrics['accuracy']:.4f}
                    - *Precision*: {concept_metrics['precision']:.4f}
                    - *Recall*: {concept_metrics['recall']:.4f}
                    - *F1 Score*: {concept_metrics['f1']:.4f}
                    """)
                
                with col2:
                    st.markdown("#### Diagnosis Prediction")
                    diagnosis_metrics = eval_results['diagnosis_metrics']
                    st.markdown(f"""
                    - *Accuracy*: {diagnosis_metrics['accuracy']:.4f}
                    - *Precision*: {diagnosis_metrics['precision']:.4f}
                    - *Recall*: {diagnosis_metrics['recall']:.4f}
                    - *F1 Score*: {diagnosis_metrics['f1']:.4f}
                    """)
            
            # Class information
            st.markdown("### 🏷 Diagnosis Classes")
            
            if 'class_names' in config:
                classes_per_col = len(config['class_names']) // 3 + 1
                cols = st.columns(3)
                
                for i, class_name in enumerate(config['class_names']):
                    col_idx = i // classes_per_col
                    if col_idx < 3:
                        cols[col_idx].markdown(f"- {class_name}")
            
        except FileNotFoundError:
            st.warning("Model results file not found. Please train the model first.")
        except Exception as e:
            st.error(f"Error loading model information: {e}")
        
        # Architecture diagram
        st.markdown("### 🔄 Model Architecture")
        st.markdown("""
        
        Input Image (224x224)
                ↓
        ResNet18 Feature Extractor
                ↓
        Concept Predictor → [7 Concepts]
                ↓
        Diagnosis Predictor → [19 Classes]
                ↓
        ICE Explanation Generator
        
        """)
    
    elif app_mode == "❓ Help":
        st.markdown("## Help & Documentation")
        
        st.markdown("### 🤔 Frequently Asked Questions")
        
        with st.expander("What is a Concept Bottleneck Model?"):
            st.markdown("""
            A Concept Bottleneck Model (CBM) is an interpretable machine learning approach that:
            1. First identifies human-understandable concepts in the input
            2. Then makes predictions based only on these concepts
            3. Provides transparent explanations of the decision process
            
            This makes the model's reasoning process interpretable to domain experts.
            """)
        
        with st.expander("How do ICE explanations work?"):
            st.markdown("""
            Individual Conditional Expectation (ICE) explanations measure how much each concept 
            influences the final diagnosis by:
            1. Making a prediction with all concepts present
            2. Removing each concept one by one
            3. Measuring how the prediction probability changes
            4. Ranking concepts by their influence magnitude
            """)
        
        with st.expander("What image formats are supported?"):
            st.markdown("""
            The application supports:
            - JPEG (.jpg, .jpeg)
            - PNG (.png)
            
            For best results, use high-quality dermoscopic images with good lighting and focus.
            """)
        
        with st.expander("How accurate is this model?"):
            st.markdown("""
            The model's accuracy depends on several factors:
            - Quality of training data
            - Image quality and acquisition conditions
            - Specific diagnosis type
            
            Check the 'Model Information' section for detailed performance metrics.
            This tool should be used as a diagnostic aid, not a replacement for medical expertise.
            """)

def get_concept_description(concept_name):
    """Get description for each concept"""
    descriptions = {
        "Pigment Network": "Net-like pattern of pigmented lines throughout the lesion",
        "Blue-Whitish Veil": "Irregular blue pigmentation with white scarring areas", 
        "Vascular Structures": "Visible blood vessels in various patterns",
        "Pigmentation": "Brown to black coloration intensity",
        "Streaks/Pseudopods": "Finger-like projections at the periphery",
        "Dots/Globules": "Round structures, usually brown or black",
        "Regression Structures": "White scar-like areas indicating tissue regression"
    }
    return descriptions.get(concept_name, "Clinical dermatological feature")

if __name__ == "__main__":
    main()