"""
Hugging Face Gradio App for BigEarthNet SR-ResNet50 Classifier
Enhanced Super-Resolution + Active Learning Classification
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import json

# --------------------------------------------------------------------------------
# SR MODEL ARCHITECTURE (RFB-ESRGAN Generator)
# --------------------------------------------------------------------------------

class RFB(nn.Module):
    """Receptive Field Block - EXACT architecture from training checkpoint"""
    def __init__(self, in_channels=64):
        super().__init__()
        # Branch 1: AvgPool(3) + Conv + ReLU + Conv
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        # Branch 2: AvgPool(5) + Conv + ReLU + Conv
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(5, stride=1, padding=2),
            nn.Conv2d(in_channels, 24, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        # Branch 3: AvgPool(7) + Conv + ReLU + Conv
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(7, stride=1, padding=3),
            nn.Conv2d(in_channels, 24, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )
        # Fusion layer
        self.conv_concat = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], 1)
        return self.conv_concat(out) * 0.2 + x

class DenseBlock(nn.Module):
    """Dense block with 5 layers - EXACT architecture from training"""
    def __init__(self, nf=64):
        super().__init__()
        nf_internal = 32
        self.conv1 = nn.Conv2d(nf, nf_internal, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + nf_internal, nf_internal, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + nf_internal*2, nf_internal, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + nf_internal*3, nf_internal, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + nf_internal*4, nf, 3, 1, 1)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(torch.cat([x, x1], 1)))
        x3 = F.relu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = F.relu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """Residual in Residual Dense Block - EXACT architecture from training"""
    def __init__(self, nf=64):
        super().__init__()
        self.db1 = DenseBlock(nf)
        self.db2 = DenseBlock(nf)
        self.db3 = DenseBlock(nf)
        
    def forward(self, x):
        out = self.db3(self.db2(self.db1(x)))
        return out * 0.2 + x

class RRFDB(nn.Module):
    """Residual RFB Dense Block - EXACT architecture from training"""
    def __init__(self, nf=64):
        super().__init__()
        self.rfb1 = RFB(nf)
        self.rfb2 = RFB(nf)
        self.rfb3 = RFB(nf)
        self.rfb4 = RFB(nf)
        self.rfb5 = RFB(nf)
        
    def forward(self, x):
        out = self.rfb1(x)
        out = self.rfb2(out)
        out = self.rfb3(out)
        out = self.rfb4(out)
        out = self.rfb5(out)
        return out * 0.2 + x

class Generator(nn.Module):
    """RFB-ESRGAN Generator with 12 RRDB + 6 RRFDB blocks"""
    def __init__(self, nc=64, num_rrdb=12, num_rrfdb=6, scale=8):
        super().__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(3, nc, 3, 1, 1)
        
        # RRDB trunk (12 blocks) - named trunk_a to match training checkpoint
        self.trunk_a = nn.Sequential(*[RRDB(nc) for _ in range(num_rrdb)])
        
        # RRFDB trunk (6 blocks) - named trunk_rfb to match training checkpoint
        self.trunk_rfb = nn.Sequential(*[RRFDB(nc) for _ in range(num_rrfdb)])
        
        # Trunk fusion
        self.trunk_conv = nn.Conv2d(nc, nc, 3, 1, 1)
        
        # Upsampling (3 stages for 8x: 2x2x2)
        self.upconv1 = nn.Conv2d(nc, nc * 4, 3, 1, 1)
        self.pixel_shuffle1 = nn.PixelShuffle(2)
        self.upconv2 = nn.Conv2d(nc, nc * 4, 3, 1, 1)
        self.pixel_shuffle2 = nn.PixelShuffle(2)
        self.upconv3 = nn.Conv2d(nc, nc * 4, 3, 1, 1)
        self.pixel_shuffle3 = nn.PixelShuffle(2)
        
        # HR reconstruction
        self.conv_hr = nn.Conv2d(nc, nc, 3, 1, 1)
        self.conv_last = nn.Conv2d(nc, 3, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        feat = self.lrelu(self.conv_first(x))
        trunk_a_out = self.trunk_a(feat)
        trunk_rfb_out = self.trunk_rfb(trunk_a_out)
        trunk = self.trunk_conv(trunk_rfb_out)
        feat = feat + trunk
        
        # Upsample 8x (2x2x2)
        feat = self.lrelu(self.pixel_shuffle1(self.upconv1(feat)))
        feat = self.lrelu(self.pixel_shuffle2(self.upconv2(feat)))
        feat = self.lrelu(self.pixel_shuffle3(self.upconv3(feat)))
        
        # HR refinement
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

# --------------------------------------------------------------------------------
# CLASSIFIER MODEL (SR-Enhanced ResNet50)
# --------------------------------------------------------------------------------

class SREnhancedClassifier(nn.Module):
    """ResNet50-based classifier that processes SR-enhanced images"""
    def __init__(self, num_classes, sr_model):
        super().__init__()
        self.sr_model = sr_model
        
        # Load pretrained ResNet50
        self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, lr_images):
        # SR enhancement (frozen)
        with torch.no_grad():
            sr_images = self.sr_model(lr_images)
            # Resize to 224x224 for ResNet
            sr_images = F.interpolate(sr_images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Classification
        return self.backbone(sr_images)

# --------------------------------------------------------------------------------
# LOAD MODELS
# --------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")

# Load class names (19 classes for the new model)
# These are BigEarthNet-S2 19-class labels
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
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]

num_classes = 19

print("📦 Loading SR model...")
sr_model = Generator(nc=64, num_rrdb=12, num_rrfdb=6, scale=8).to(device)

# Load SR weights (download from your Kaggle or provide path)
try:
    sr_weights = torch.load('sr_generator_weights.pth', map_location=device)
    sr_model.load_state_dict(sr_weights)
    print("✅ SR model loaded successfully")
except FileNotFoundError:
    print("⚠️ SR weights not found, using random initialization")

sr_model.eval()

print("📦 Loading classifier...")
# Initialize classifier
classifier = SREnhancedClassifier(num_classes, sr_model).to(device)

# Load classifier weights
try:
    # Handle DataParallel wrapped state dict
    state_dict = torch.load('best_classifier.pth', map_location=device)
    
    # Remove 'module.' prefix if present
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    classifier.load_state_dict(state_dict)
    print("✅ Classifier loaded successfully")
except Exception as e:
    print(f"⚠️ Error loading classifier: {e}")
    print("Using randomly initialized classifier")

classifier.eval()

# --------------------------------------------------------------------------------
# PREPROCESSING & INFERENCE
# --------------------------------------------------------------------------------

def preprocess_image(image):
    """Convert PIL image to tensor"""
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 32x32 (LR input size for 8x SR)
    lr_image = image.resize((32, 32), Image.BILINEAR)
    
    # Convert to numpy and normalize to [0, 1]
    img_np = np.array(lr_image).astype(np.float32) / 255.0
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor.to(device)

def postprocess_image(tensor):
    """Convert tensor to PIL image"""
    img = tensor.squeeze(0).cpu().clamp(0, 1)
    img_np = img.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_np)

def predict(image):
    """Main inference function"""
    try:
        # Preprocess
        lr_tensor = preprocess_image(image)
        
        with torch.no_grad():
            # Get SR image for visualization
            sr_tensor = classifier.sr_model(lr_tensor)
            sr_tensor = F.interpolate(sr_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Classification
            logits = classifier(lr_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Get top 5 predictions
        top_indices = np.argsort(probs)[-5:][::-1]
        predictions = {
            CLASS_NAMES[idx]: float(probs[idx]) 
            for idx in top_indices
        }
        
        # Convert SR image for display
        sr_image = postprocess_image(sr_tensor)
        
        return sr_image, predictions
    
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return None, {"error": str(e)}

# --------------------------------------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------------------------------------

title = "🛰️ BigEarthNet SR-ResNet50 Classifier"

description = """
### Super-Resolution Enhanced Land Cover Classification

Upload a satellite image to:
1. **Enhance** resolution using RFB-ESRGAN (32×32 → 256×256)
2. **Classify** land cover into 19 BigEarthNet categories

**Model Architecture:**
- SR: RFB-ESRGAN (12 RRDB + 6 RRFDB blocks, 8× upscaling)
- Classifier: ResNet50 with enhanced head (trained with Active Learning)
- Training: 100k samples, 50 epochs with EMA and label smoothing

**Features:**
- ✅ Exponential Moving Average (EMA) for stable predictions
- ✅ Multi-scale feature extraction with RFB blocks
- ✅ Active Learning trained (DBSS + SSAS strategies)
"""

article = """
### About This Model

**Training Details:**
- Dataset: BigEarthNet-S2 (19 classes)
- Samples: 100,000 training samples
- Epochs: 50 with cosine annealing LR
- Regularization: Label smoothing (0.15), Dropout (0.4, 0.3), EMA (0.9995)
- Augmentation: Random rotation, flips, color jitter
- Hardware: Dual GPU training on Kaggle

**Performance Metrics:**
- Validation Accuracy: 100% (reported)
- Training enhanced with comprehensive visualizations:
  - ROC curves & AUC analysis
  - Precision-Recall curves
  - Feature map visualizations
  - Learning dynamics tracking

**Links:**
- [GitHub Repository](https://github.com/Sudarshan-Hegde/Geo-Agri-Analyst.git)
- [BigEarthNet Dataset](http://bigearth.net/)
- [Training Notebook](https://www.kaggle.com/code/hegdesudarshan/majprojsuper-new)

**Citation:**
```
@misc{hegde2025sr-resnet50-bigearthnet,
  title={Super-Resolution Enhanced ResNet50 for BigEarthNet Classification},
  author={Hegde, Sudarshan},
  year={2025}
}
```
"""

# Create interface
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Satellite Image (32×32 or any size)"),
    outputs=[
        gr.Image(label="Enhanced Image (SR 256×256)", type="pil"),
        gr.Label(num_top_classes=5, label="Top 5 Land Cover Predictions")
    ],
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft(),
    allow_flagging="never",
    examples=None  # Add example images if available
)

if __name__ == "__main__":
    print("\n🚀 Launching Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # HuggingFace Spaces handles public access
    )
