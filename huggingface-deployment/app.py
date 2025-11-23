"""
Hugging Face Gradio App for BigEarthNet SR-AL Classifier
Deploy this to HuggingFace Spaces for free hosting with GPU support
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from PIL import Image
import json

# --------------------------------------------------------------------------------
# MODEL ARCHITECTURE DEFINITIONS (Same as training notebook)
# --------------------------------------------------------------------------------

class BasicConv(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.conv(x))

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth):
        super().__init__()
        # Sequential with indexed modules [0] = Conv2d to match saved weights
        layers = [nn.Conv2d(in_channels, growth, 3, 1, 1)]
        self.layers = nn.Sequential(*layers)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        return self.lrelu(self.layers(x))

class RRDB(nn.Module):
    def __init__(self, nc=64, growth=32):
        super().__init__()
        # Dense block - match the trained model structure
        self.db = nn.Module()
        self.db.convs = nn.ModuleList([
            DenseLayer(nc + i * growth, growth) for i in range(5)
        ])
        self.conv = nn.Conv2d(5 * growth, nc, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        feats = [x]
        for layer in self.db.convs:
            feats.append(layer(torch.cat(feats, 1)))
        res = self.conv(torch.cat(feats[1:], 1))
        return self.lrelu(res * 0.2 + x)

class RFB_Branch(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        # Match trained model structure with .b1, .b2, .b3, .b4 naming
        self.b1 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 4, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c // 2, in_c // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c // 2, in_c // 4, 5, 1, 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_c, in_c // 2, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_c // 2, in_c // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.cl = nn.Conv2d(in_c, in_c, 1, 1, 0)  # Final convolution layer
        self.sc = nn.LeakyReLU(0.2, inplace=True)  # Activation
    
    def forward(self, x):
        b1 = self.b1(x)
        b2 = self.b2(x)
        b3 = self.b3(x)
        b4 = self.b4(x)
        fused = torch.cat([b1, b2, b3, b4], 1)
        return self.sc(self.cl(fused))

class RFB_Modified(nn.Module):
    def __init__(self, nc=64):
        super().__init__()
        # Match trained model - has db (dense block) first
        self.db = nn.Module()
        self.db.convs = nn.ModuleList([
            DenseLayer(nc + i * 32, 32) for i in range(5)  # growth=32
        ])
        self.rfb = RFB_Branch(nc)
        self.conv = nn.Conv2d(nc, nc, 1, 1, 0)
    
    def forward(self, x):
        # Dense block processing
        feats = [x]
        for layer in self.db.convs:
            feats.append(layer(torch.cat(feats, 1)))
        db_out = self.conv(torch.cat(feats[1:], 1))  # Fuse dense features
        db_out = F.leaky_relu(db_out * 0.2 + x, 0.2, inplace=True)
        
        # RFB processing
        rfb_out = self.rfb(db_out)
        return rfb_out

class RFBESRGANGenerator(nn.Module):
    def __init__(self, growth=32, num_rrdb=16, num_rfb=8, nc=64, upscale=4):
        super().__init__()
        # Initial convolution
        self.c1 = nn.Conv2d(3, nc, 3, 1, 1)
        self.cl = nn.LeakyReLU(0.2, inplace=True)
        
        # Trunk A - RRDB blocks
        self.trunk_a = nn.ModuleList([RRDB(nc, growth) for _ in range(num_rrdb)])
        
        # Trunk RFB - RFB-Modified blocks
        self.trunk_rfb = nn.ModuleList([RFB_Modified(nc) for _ in range(num_rfb)])
        
        # RFB fusion - uses RFB_Branch structure
        self.rfb_fuse = RFB_Branch(nc * num_rfb)
        
        # Second convolution
        self.c2 = nn.Conv2d(nc, nc, 3, 1, 1)
        
        # Upsampling
        self.u1 = nn.Conv2d(nc, nc * upscale**2, 3, 1, 1)
        self.u2 = nn.PixelShuffle(upscale) if hasattr(nn, 'PixelShuffle') else None
        self.u3 = nn.LeakyReLU(0.2, inplace=True)
        
        # HR refinement
        self.hr = nn.Conv2d(nc, nc, 3, 1, 1)
        
        # Output - no separate module, directly returns
    
    def forward(self, x):
        # Initial feature extraction
        feat = self.cl(self.c1(x))
        
        # RRDB trunk
        trunk_a = feat
        for db in self.trunk_a:
            trunk_a = db(trunk_a)
        
        # RFB trunk
        trunk_rfb_feats = []
        feat_in = feat
        for rfb in self.trunk_rfb:
            feat_in = rfb(feat_in)
            trunk_rfb_feats.append(feat_in)
        
        # Fuse RFB features
        fuse = self.rfb_fuse(torch.cat(trunk_rfb_feats, 1))
        
        # Combine with trunk_a
        feat = F.leaky_relu(self.c2(fuse + trunk_a), 0.2, inplace=True)
        
        # Upsample
        up = self.u3(self.u2(self.u1(feat)) if self.u2 else self.u1(feat))
        
        # HR refinement
        hr_feat = F.leaky_relu(self.hr(up), 0.2, inplace=True)
        
        # Output activation
        return torch.tanh(hr_feat)

class RobustClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=False)
        self.backbone.fc = nn.Identity()
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Normalize from [-1, 1] to ImageNet stats
        x = (x + 1) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        feats = self.backbone(x)
        return self.fc(feats)

# --------------------------------------------------------------------------------
# LOAD MODELS AND CLASS NAMES
# --------------------------------------------------------------------------------

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load class names
with open('label_indices.json', 'r') as f:
    label_data = json.load(f)
class_names = list(label_data['original_labels'].keys())
num_classes = len(class_names)

# Initialize models
sr_model = RFBESRGANGenerator(upscale=4).to(device)
classifier = RobustClassifier(num_classes=num_classes).to(device)

# Load trained weights
sr_model.load_state_dict(torch.load('sr_model.pth', map_location=device))
classifier.load_state_dict(torch.load('classifier.pth', map_location=device))

sr_model.eval()
classifier.eval()

print("✓ Models loaded successfully")

# --------------------------------------------------------------------------------
# INFERENCE FUNCTION
# --------------------------------------------------------------------------------

def preprocess_image(image):
    """Convert PIL image to tensor in [-1, 1] range"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 30x30 (LR size)
    lr_image = image.resize((30, 30), Image.BILINEAR)
    
    # Convert to numpy array
    img_np = np.array(lr_image).astype(np.float32) / 255.0  # [0, 1]
    
    # Normalize to [-1, 1]
    img_np = img_np * 2.0 - 1.0
    
    # Convert to tensor (H, W, C) -> (C, H, W)
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor.to(device)

def postprocess_image(tensor):
    """Convert tensor to PIL image"""
    # Convert from [-1, 1] to [0, 1]
    img = (tensor.squeeze(0).cpu() + 1) / 2.0
    img = img.clamp(0, 1)
    
    # Convert to numpy (C, H, W) -> (H, W, C)
    img_np = img.permute(1, 2, 0).numpy()
    
    # Convert to uint8
    img_np = (img_np * 255).astype(np.uint8)
    
    return Image.fromarray(img_np)

def predict(image):
    """Main inference function"""
    try:
        # Preprocess
        lr_tensor = preprocess_image(image)
        
        # Super-resolution
        with torch.no_grad():
            sr_tensor = sr_model(lr_tensor)
            
            # Classification
            logits = classifier(sr_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Get top predictions
        top_indices = np.argsort(probs)[-5:][::-1]  # Top 5
        predictions = {
            class_names[idx]: float(probs[idx]) 
            for idx in top_indices
        }
        
        # Convert SR image for display
        sr_image = postprocess_image(sr_tensor)
        
        return sr_image, predictions
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None, {"error": str(e)}

# --------------------------------------------------------------------------------
# GRADIO INTERFACE
# --------------------------------------------------------------------------------

title = "🛰️ BigEarthNet SR-AL Classifier"
description = """
Upload a satellite image to:
1. **Enhance** it using Super-Resolution (30×30 → 120×120)
2. **Classify** land cover types (43 BigEarthNet classes)

Powered by RFB-ESRGAN + ResNet-18 trained with Active Learning.
"""

article = """
### About
This model was trained on the BigEarthNet V2 dataset using:
- **Super-Resolution**: RFB-ESRGAN (4× upscaling)
- **Active Learning**: DBSS + SSAS strategies for efficient labeling
- **Multi-label Classification**: 43 land cover classes

### Usage Tips
- Upload satellite images (preferably Sentinel-2 RGB bands)
- The model works best with 30×30 pixel images
- Top 5 predicted classes are shown with confidence scores

### Links
- [GitHub Repository](https://github.com/Sudarshan-Hegde/Agri-SR-AL-Net.git)
- [BigEarthNet Dataset](http://bigearth.net/)
"""

examples = [
    # Add example image paths here if you have sample images
]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Satellite Image"),
    outputs=[
        gr.Image(type="pil", label="Enhanced Image (SR)"),
        gr.Label(num_top_classes=5, label="Land Cover Classification")
    ],
    title=title,
    description=description,
    article=article,
    examples=examples,
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
