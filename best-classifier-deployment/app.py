"""
BestClassifier - HuggingFace Space Deployment
Land Cover Classification with SR Enhancement
Extracted from majprojsuper_new.ipynb and majprojsuperres.ipynb
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

# ===============================================================================
# SR MODEL ARCHITECTURE (from majprojsuperres.ipynb)
# ===============================================================================

class DenseBlock(nn.Module):
    """Dense Block with 5 conv layers"""
    def __init__(self, nf=64, gc=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block"""
    def __init__(self, nf=64):
        super(RRDB, self).__init__()
        self.db1 = DenseBlock(nf)
        self.db2 = DenseBlock(nf)
        self.db3 = DenseBlock(nf)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x

class RFB(nn.Module):
    """Receptive Field Block"""
    def __init__(self, in_channels, out_channels):
        super(RFB, self).__init__()
        branch_channels = out_channels // 3
        remaining = out_channels - (branch_channels * 2)
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, remaining, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(remaining, remaining, 3, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.conv_concat = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1)
        )

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        return self.conv_concat(x_cat) + x

class RRFDB(nn.Module):
    """Residual in Residual RFB Dense Block"""
    def __init__(self, nf=64):
        super(RRFDB, self).__init__()
        self.rfb1 = RFB(nf, nf)
        self.rfb2 = RFB(nf, nf)
        self.rfb3 = RFB(nf, nf)
        self.rfb4 = RFB(nf, nf)
        self.rfb5 = RFB(nf, nf)

    def forward(self, x):
        out = self.rfb1(x)
        out = self.rfb2(out)
        out = self.rfb3(out)
        out = self.rfb4(out)
        out = self.rfb5(out)
        return out * 0.2 + x

class Generator(nn.Module):
    """RFB-ESRGAN Generator for RGB images (8x SR: 32→256)"""
    def __init__(self, num_rrdb=12, num_rrfdb=6, nf=64):
        super(Generator, self).__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)  # RGB input
        
        # RRDB and RRFDB trunks
        self.trunk_a = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
        self.trunk_rfb = nn.Sequential(*[RRFDB(nf) for _ in range(num_rrfdb)])
        self.rfb_up = RFB(nf, nf)
        
        # 8x upsampling (3 stages: 2x2x2 = 8x)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2),
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.PixelShuffle(2)
        )
        
        # Final convolutions
        self.conv_final = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 3, 3, 1, 1)  # RGB output
        )

    def forward(self, x):
        fea = self.conv_first(x)
        trunk_a_out = self.trunk_a(fea)
        trunk_rfb_out = self.trunk_rfb(trunk_a_out)
        rfb_up_out = self.rfb_up(trunk_rfb_out)
        fea = fea + rfb_up_out
        
        fea = self.upsample(fea)
        out = self.conv_final(fea)
        return out

# ===============================================================================
# CLASSIFIER MODEL (from majprojsuper_new.ipynb)
# ===============================================================================

class SREnhancedClassifier(nn.Module):
    """ResNet50-based classifier with SR enhancement"""
    def __init__(self, num_classes, sr_model, pretrained=True):
        super().__init__()
        self.sr_model = sr_model
        
        # Load pretrained ResNet50
        if pretrained:
            weights = ResNet50_Weights.IMAGENET1K_V2
            self.backbone = resnet50(weights=weights)
        else:
            self.backbone = resnet50(weights=None)
        
        # Enhanced classifier head
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, lr_images):
        # SR enhancement: 32x32 → 256x256
        with torch.no_grad():
            sr_images = self.sr_model(lr_images)
            # Resize to 224x224 for ResNet
            sr_images = F.interpolate(sr_images, size=(224, 224), 
                                     mode='bilinear', align_corners=False)
        
        # ResNet classification
        x = self.backbone.conv1(sr_images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        output = self.backbone.fc(features)
        return output

# ===============================================================================
# MODEL LOADING
# ===============================================================================

def load_checkpoint(model, path, device):
    """Load checkpoint with DataParallel handling"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Checkpoint not found: {path}")
    
    try:
        print(f"📦 Loading checkpoint: {path}")
        state_dict = torch.load(path, map_location=device)
        
        # Strip 'module.' prefix from DataParallel
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v
        
        # Load weights (strict=False allows partial loading)
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            print(f"  ⚠️ Missing keys: {len(missing)}")
        if unexpected:
            print(f"  ⚠️ Unexpected keys: {len(unexpected)}")
        
        print(f"  ✅ Checkpoint loaded successfully!")
        return model
        
    except Exception as e:
        print(f"  ❌ Error loading checkpoint: {e}")
        raise

# ===============================================================================
# PREPROCESSING
# ===============================================================================

def get_transform():
    """Image preprocessing for model input"""
    return transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to 32x32 for LR input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

def preprocess_image(image):
    """Preprocess PIL Image for model"""
    transform = get_transform()
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms and add batch dimension
    tensor = transform(image).unsqueeze(0)
    return tensor

# ===============================================================================
# INITIALIZATION
# ===============================================================================

print("=" * 60)
print("🚀 Initializing BestClassifier")
print("=" * 60)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📱 Device: {device}")

# Class names (BigEarthNet-19)
CLASS_NAMES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland",
    "Moors and heathland",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]

num_classes = len(CLASS_NAMES)
print(f"📊 Number of classes: {num_classes}")

# Initialize models
try:
    print("\n🔧 Building SR Generator...")
    sr_model = Generator(num_rrdb=12, num_rrfdb=6, nf=64).to(device)
    sr_model.eval()
    print("  ✅ SR Generator created")
    
    print("\n🔧 Building Classifier...")
    classifier = SREnhancedClassifier(num_classes, sr_model, pretrained=True).to(device)
    print("  ✅ Classifier created")
    
    print("\n📦 Loading weights from best_classifier.pth...")
    classifier = load_checkpoint(classifier, 'best_classifier.pth', device)
    classifier.eval()
    
    # Test forward pass
    print("\n🧪 Testing model...")
    with torch.no_grad():
        test_input = torch.randn(1, 3, 32, 32).to(device)
        test_output = classifier(test_input)
        print(f"  Test passed: {test_input.shape} → {test_output.shape}")
    
    print("\n" + "=" * 60)
    print("✅ Model initialized successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error during initialization: {e}")
    import traceback
    traceback.print_exc()
    raise

# ===============================================================================
# INFERENCE
# ===============================================================================

def predict(image):
    """
    Main prediction function for Gradio
    
    Args:
        image: PIL Image from Gradio
    
    Returns:
        dict: Top 5 predictions {class_name: confidence}
    """
    try:
        # Preprocess
        tensor = preprocess_image(image)
        tensor = tensor.to(device)
        
        # Inference
        with torch.no_grad():
            output = classifier(tensor)
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
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback
        traceback.print_exc()
        return {"Error": error_msg}

# ===============================================================================
# GRADIO INTERFACE
# ===============================================================================

# Create interface using Blocks API with simplified schema
with gr.Blocks(title="🌍 BestClassifier - Land Cover Classification", analytics_enabled=False) as demo:
    gr.Markdown("# 🌍 BestClassifier - Land Cover Classification")
    gr.Markdown("""
    Upload a satellite image to classify land cover type using **SR-enhanced ResNet50**.
    
    **Features:**
    - **Super Resolution**: 8× enhancement (32×32 → 256×256) using RFB-ESRGAN
    - **ResNet50 Classifier**: Pretrained on ImageNet, fine-tuned on BigEarthNet
    - **19 Land Cover Classes**: CORINE Land Cover categories
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Satellite Image")
            submit_btn = gr.Button("Classify", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=5, label="Land Cover Predictions")
    
    submit_btn.click(fn=predict, inputs=input_image, outputs=output_label, api_name="predict")
    
    gr.Markdown("""
    ### About
    
    This model combines super resolution with deep learning classification for accurate land cover prediction.
    
    **Architecture:**
    - SR Generator: RFB-ESRGAN (12 RRDB + 6 RRFDB blocks)
    - Classifier: ResNet50 with enhanced head (Dropout + 512-dim FC layer)
    
    **Dataset:** [BigEarthNet-S2](http://bigearth.net/) - 19-class CORINE Land Cover
    
    **Training:** 100K samples, 50 epochs with EMA optimization
    """)

# ===============================================================================
# LAUNCH
# ===============================================================================

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    print("📍 Space URL: https://huggingface.co/spaces/HegdeSudarshan/bestClassifier")
    
    # Workaround for json_schema bug: disable API info generation
    import os
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    
    demo.launch(show_api=False)
