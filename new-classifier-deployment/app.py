import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
from PIL import Image
import json
import os
from collections import OrderedDict

# --------------------------------------------------------------------------------
# MODEL ARCHITECTURE (Must match majProjUltra.ipynb EXACTLY)
# --------------------------------------------------------------------------------

class RFB(nn.Module):
    """Receptive Field Block"""
    def __init__(self, in_channels=64):
        super(RFB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(5, stride=1, padding=2),
            nn.Conv2d(in_channels, 24, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(7, stride=1, padding=3),
            nn.Conv2d(in_channels, 24, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )
        self.conv_concat = nn.Sequential(
            nn.Conv2d(64, in_channels, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        concat = torch.cat([b1, b2, b3], dim=1)
        out = self.conv_concat(concat)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.db1 = DenseBlock(nf, gc)
        self.db2 = DenseBlock(nf, gc)
        self.db3 = DenseBlock(nf, gc)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return out * 0.2 + x

class RRFDB(nn.Module):
    def __init__(self, nf=64):
        super(RRFDB, self).__init__()
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
    """RFB-ESRGAN Generator - Corrected to match majProjUltra.ipynb"""
    def __init__(self, num_rrdb=12, num_rrfdb=6, nf=64):
        super(Generator, self).__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        
        # Trunk-A
        self.trunk_a = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
        
        # Trunk-RFB
        self.trunk_rfb = nn.Sequential(*[RRFDB(nf) for _ in range(num_rrfdb)])
        
        # Single RFB before upsampling
        self.rfb_up = RFB(nf)
        
        # Upsampling (3 stages for 8x: 2x2x2)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final convs
        self.conv_final = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 3, 3, 1, 1),
            nn.Tanh() # Note: Tanh outputs [-1, 1]
        )
        
    def forward(self, x):
        feat = self.conv_first(x)
        trunk_a_out = self.trunk_a(feat)
        trunk_rfb_out = self.trunk_rfb(trunk_a_out)
        rfb_out = self.rfb_up(trunk_rfb_out)
        upsampled = self.upsample(rfb_out)
        out = self.conv_final(upsampled)
        return out

# --------------------------------------------------------------------------------
# CLASSIFIER MODEL
# --------------------------------------------------------------------------------

class SREnhancedClassifier(nn.Module):
    """Matches majprojsuper_new.ipynb"""
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
            # Resize from 256x256 -> 224x224 for ResNet
            sr_images = F.interpolate(sr_images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Classification
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

# --------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------

def load_checkpoint(model, path, device):
    """Helper to load checkpoint and strip 'module.' prefix if present"""
    if not os.path.exists(path):
        print(f"⚠️ Warning: Checkpoint not found at {path}")
        return model
    
    try:
        state_dict = torch.load(path, map_location=device)
        
        # Create new OrderedDict that does not contain `module.`
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove `module.`
            new_state_dict[name] = v
            
        # Load weights
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        
        if missing:
            print(f"ℹ️ Missing keys in {path}: {len(missing)} (Expected if loading partial models)")
        if unexpected:
            print(f"⚠️ Unexpected keys in {path}: {unexpected}")
        
        print(f"✅ Loaded checkpoint: {path}")
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
    
    return model

# --------------------------------------------------------------------------------
# INITIALIZATION
# --------------------------------------------------------------------------------

device = torch.device('cpu') # Hugging Face Spaces free tier uses CPU usually
print(f"🔧 Using device: {device}")

# 1. Load Labels
CLASS_NAMES = [
    "Urban fabric", "Industrial/Commercial", "Arable land", "Permanent crops", 
    "Pastures", "Complex cultivation", "Land principally agriculture", "Agro-forestry",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest", "Natural grassland",
    "Moors/Heathland", "Transitional woodland", "Beaches/Dunes", "Inland wetlands",
    "Coastal wetlands", "Inland waters", "Marine waters"
]
num_classes = len(CLASS_NAMES)

# 2. Load Models
# Instantiating Generator with params from majProjUltra.ipynb (12 RRDB, 6 RRFDB)
sr_model = Generator(num_rrdb=12, num_rrfdb=6, nf=64).to(device)

# Load Classifier (which contains the SR model inside it usually, or we pass it)
classifier = SREnhancedClassifier(num_classes, sr_model).to(device)

# Load Weights
# Note: Corrected filename to 'generator_ensemble.pth'
classifier = load_checkpoint(classifier, 'best_classifier.pth', device)
# If the classifier checkpoint doesn't contain the SR weights fully, we might need to load SR separately:
# sr_model = load_checkpoint(sr_model, 'generator_ensemble.pth', device)

classifier.eval()

# --------------------------------------------------------------------------------
# INFERENCE LOGIC
# --------------------------------------------------------------------------------

def predict(image):
    if image is None:
        return None, None

    try:
        # Preprocess
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize input to 32x32 (standard LR input)
        lr_img = image.resize((32, 32), Image.BICUBIC)
        
        # To Tensor: Normalize to [-1, 1] as per training logic
        img_t = torch.tensor(np.array(lr_img)).float() / 255.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0) # CHW -> BCHW
        img_t = (img_t * 2) - 1 # [0,1] -> [-1, 1]
        img_t = img_t.to(device)
        
        with torch.no_grad():
            # Run Inference
            # 1. Get SR Output (for display)
            sr_output = classifier.sr_model(img_t)
            
            # 2. Get Classification
            logits = classifier(img_t) # Forward pass handles resizing internally
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        # Post-process SR Image
        # Denormalize from [-1, 1] to [0, 255]
        sr_np = sr_output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        sr_np = (sr_np + 1) / 2.0
        sr_np = np.clip(sr_np, 0, 1)
        sr_img_out = Image.fromarray((sr_np * 255).astype(np.uint8))
        
        # Process Predictions
        top_k_indices = probs.argsort()[-5:][::-1]
        top_preds = {CLASS_NAMES[i]: float(probs[i]) for i in top_k_indices}
        
        return sr_img_out, top_preds
        
    except Exception as e:
        print(f"Error: {e}")
        return None, {"Error": str(e)}

# --------------------------------------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------------------------------------

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Satellite Image (Input will be resized to 32x32)"),
    outputs=[
        gr.Image(label="Enhanced Super-Resolution (256x256)"),
        gr.Label(label="Classification Results")
    ],
    title="BigEarthNet SR-Classifier (RFB-ESRGAN)",
    description="Upload a low-res satellite patch. The model will upscale it using RFB-ESRGAN and classify the land cover."
    # allow_flagging="never"  <-- REMOVED THIS LINE TO FIX THE ERROR
)

if __name__ == "__main__":
    demo.launch()