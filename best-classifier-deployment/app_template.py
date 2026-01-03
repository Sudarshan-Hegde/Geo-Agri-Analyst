"""
BestClassifier Deployment for HuggingFace Spaces
Land Cover Classification with SR Enhancement

CRITICAL: This template must be customized with:
1. Exact model architecture from training notebook
2. Correct preprocessing steps
3. Proper class names/labels
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
import numpy as np
from PIL import Image
import json
import os
from collections import OrderedDict

# --------------------------------------------------------------------------------
# TODO: COPY MODEL ARCHITECTURE FROM TRAINING NOTEBOOK
# --------------------------------------------------------------------------------

# PLACEHOLDER - Replace with your actual architecture
class RFB(nn.Module):
    """Receptive Field Block - Copy from training notebook"""
    def __init__(self, in_channels=64):
        super(RFB, self).__init__()
        # TODO: Copy exact implementation from training
        pass
    
    def forward(self, x):
        # TODO: Copy exact forward pass
        return x

class RRDB(nn.Module):
    """Residual in Residual Dense Block - Copy from training notebook"""
    def __init__(self, nf=64):
        super(RRDB, self).__init__()
        # TODO: Copy exact implementation
        pass
    
    def forward(self, x):
        # TODO: Copy exact forward pass
        return x

class Generator(nn.Module):
    """SR Generator - Copy exact architecture from training notebook"""
    def __init__(self, num_rrdb=12, num_rrfdb=6, nf=64):
        super(Generator, self).__init__()
        # TODO: Copy exact layer names and structure
        # CRITICAL: Layer names must match state_dict keys
        pass
    
    def forward(self, x):
        # TODO: Copy exact forward pass
        return x

class BestClassifier(nn.Module):
    """
    Complete classifier with SR enhancement
    Copy exact architecture from training notebook
    """
    def __init__(self, num_classes=19, sr_model=None):
        super().__init__()
        # TODO: Copy exact implementation
        self.sr_model = sr_model
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # TODO: Copy exact classifier head architecture
        pass
    
    def forward(self, x):
        # TODO: Copy exact forward pass including SR
        pass

# --------------------------------------------------------------------------------
# MODEL LOADING
# --------------------------------------------------------------------------------

def load_checkpoint(model, path, device):
    """
    Load model checkpoint with DataParallel handling
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    
    try:
        print(f"Loading checkpoint from {path}...")
        state_dict = torch.load(path, map_location=device)
        
        # Strip 'module.' prefix from DataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        # Load weights (strict=False allows partial loading)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            print(f"⚠️ Missing keys: {len(missing)}")
            if len(missing) < 10:
                print(missing)
        
        if unexpected:
            print(f"⚠️ Unexpected keys: {len(unexpected)}")
            if len(unexpected) < 10:
                print(unexpected)
        
        print(f"✅ Successfully loaded checkpoint")
        return model
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        raise

# --------------------------------------------------------------------------------
# PREPROCESSING
# --------------------------------------------------------------------------------

def get_transform():
    """
    TODO: Copy EXACT preprocessing from training notebook
    CRITICAL: Must match training preprocessing exactly
    """
    # Example - customize to match your training
    return transforms.Compose([
        transforms.Resize((32, 32)),  # TODO: Verify input size
        transforms.ToTensor(),
        # TODO: Add normalization if used in training
        # transforms.Normalize(mean=[...], std=[...])
    ])

def preprocess_image(image):
    """
    Preprocess input image for model
    
    Args:
        image: PIL Image
    
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
    transform = get_transform()
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    tensor = transform(image)
    
    # Add batch dimension
    tensor = tensor.unsqueeze(0)
    
    return tensor

# --------------------------------------------------------------------------------
# INITIALIZATION
# --------------------------------------------------------------------------------

print("🚀 Initializing BestClassifier...")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Device: {device}")

# Load class names
CLASS_NAMES = [
    # TODO: Replace with your actual class names
    "Class 0", "Class 1", "Class 2", "Class 3", "Class 4",
    "Class 5", "Class 6", "Class 7", "Class 8", "Class 9",
    "Class 10", "Class 11", "Class 12", "Class 13", "Class 14",
    "Class 15", "Class 16", "Class 17", "Class 18"
]

# Try to load from JSON if available
if os.path.exists('label_indices.json'):
    try:
        with open('label_indices.json', 'r') as f:
            label_data = json.load(f)
            CLASS_NAMES = label_data.get('class_names', CLASS_NAMES)
        print("✅ Loaded class names from label_indices.json")
    except Exception as e:
        print(f"⚠️ Could not load label_indices.json: {e}")

num_classes = len(CLASS_NAMES)
print(f"📊 Number of classes: {num_classes}")

# Initialize model
try:
    print("🔧 Building model...")
    
    # TODO: Initialize with correct parameters
    sr_model = Generator(num_rrdb=12, num_rrfdb=6, nf=64).to(device)
    model = BestClassifier(num_classes=num_classes, sr_model=sr_model).to(device)
    
    # Load weights
    model = load_checkpoint(model, 'bestClassifier.pth', device)
    
    # Set to eval mode
    model.eval()
    
    print("✅ Model loaded successfully")
    
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    raise

# --------------------------------------------------------------------------------
# INFERENCE
# --------------------------------------------------------------------------------

def predict(image):
    """
    Main prediction function
    
    Args:
        image: PIL Image from Gradio
    
    Returns:
        dict: Prediction results {class_name: confidence}
    """
    try:
        # Preprocess
        tensor = preprocess_image(image)
        tensor = tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
        
        # Format results
        results = {
            CLASS_NAMES[i]: float(probs[i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Sort by confidence
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return results
        
    except Exception as e:
        error_msg = f"Error during prediction: {str(e)}"
        print(f"❌ {error_msg}")
        return {"Error": error_msg}

# --------------------------------------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------------------------------------

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Satellite Image"),
    outputs=gr.Label(num_top_classes=5, label="Land Cover Predictions"),
    title="🌍 BestClassifier - Land Cover Classification",
    description="""
    Upload a satellite image to classify land cover type.
    
    **Features:**
    - Super Resolution enhancement for improved accuracy
    - ResNet50 backbone pretrained on ImageNet
    - Trained on BigEarthNet-S2 dataset
    
    **Supported Input:**
    - RGB images (any size, will be resized)
    - Optimal: 30×30 to 256×256 pixels
    
    **Output:**
    - Top 5 predicted land cover classes with confidence scores
    """,
    examples=[
        # TODO: Add example images if available
    ],
    article="""
    ### About
    This model uses a hybrid SR-enhanced ResNet50 architecture for land cover classification.
    
    ### Citation
    If you use this model, please cite: [Add your citation]
    
    ### Contact
    - GitHub: [Add your GitHub]
    - Email: [Add your email]
    """,
    allow_flagging="never",
    analytics_enabled=False
)

# --------------------------------------------------------------------------------
# LAUNCH
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    print("🚀 Launching Gradio interface...")
    
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=7860,        # HuggingFace Spaces default port
        share=False,             # HF handles public access
        show_error=True          # Show detailed errors
    )
