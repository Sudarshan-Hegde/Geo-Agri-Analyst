"""
BestClassifier - Land Cover Classification with Super Resolution
Streamlit version for HuggingFace Spaces deployment
FIXED: Architecture now matches training checkpoint exactly
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np
from typing import Dict

# ===============================================================================
# CONFIGURATION
# ===============================================================================

NUM_CLASSES = 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    "Urban fabric", "Industrial or commercial units", "Arable land",
    "Permanent crops", "Pastures", "Complex cultivation patterns",
    "Land principally occupied by agriculture", "Agro-forestry areas",
    "Broad-leaved forest", "Coniferous forest", "Mixed forest",
    "Natural grassland and sparsely vegetated areas", "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub", "Beaches, dunes, sands", "Inland wetlands",
    "Coastal wetlands", "Inland waters", "Marine waters"
]

# ===============================================================================
# MODEL ARCHITECTURE - EXACT MATCH TO TRAINING CHECKPOINT
# ===============================================================================

class DenseBlock(nn.Module):
    """Dense Block with 5 conv layers - EXACT match to training"""
    def __init__(self, nf=64):
        super().__init__()
        # Training uses nf_internal=32 regardless of nf parameter
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
    """Residual-in-Residual Dense Block (3 DenseBlocks) - named db1, db2, db3"""
    def __init__(self, nf=64):
        super().__init__()
        self.db1 = DenseBlock(nf)
        self.db2 = DenseBlock(nf)
        self.db3 = DenseBlock(nf)
        
    def forward(self, x):
        out = self.db3(self.db2(self.db1(x)))
        return out * 0.2 + x


class RFB(nn.Module):
    """Receptive Field Block - EXACT match to checkpoint"""
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
        # Output conv (matches checkpoint: conv_concat)
        self.conv_concat = nn.Sequential(
            nn.Conv2d(64, 64, 1, 1, 0)
        )
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], 1)
        return self.conv_concat(out) * 0.2 + x


class RRFDB(nn.Module):
    """Residual RFB Dense Block - named rfb1-rfb5"""
    def __init__(self, nf=64):
        super().__init__()
        # Use named attributes to match checkpoint keys
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
    """Generator: 12 RRDB + 6 RRFDB + 8x upscale - EXACT checkpoint architecture"""
    def __init__(self, num_rrdb=12, num_rrfdb=6, nf=64):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        
        # Trunk A: 12 RRDB blocks (checkpoint uses trunk_a, not rrdb_blocks)
        self.trunk_a = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
        
        # Trunk RFB: 6 RRFDB blocks (checkpoint uses trunk_rfb, not rrfdb_blocks)
        self.trunk_rfb = nn.Sequential(*[RRFDB(nf) for _ in range(num_rrfdb)])
        
        # RFB upsampling
        self.rfb_up = RFB(nf)
        
        # 8x upscaling (3 PixelShuffle layers: 2x each = 2^3 = 8x)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf*4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Final conv layers (matches checkpoint: conv_final)
        self.conv_final = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 3, 3, 1, 1)
        )
        
    def forward(self, x):
        feat = self.conv_first(x)
        trunk_a_out = self.trunk_a(feat)
        trunk_rfb_out = self.trunk_rfb(trunk_a_out)
        rfb_up_out = self.rfb_up(trunk_rfb_out)
        up_out = self.upsample(rfb_up_out)
        out = self.conv_final(up_out)
        return out


class SREnhancedClassifier(nn.Module):
    """ResNet50 Classifier with SR Enhancement - EXACT match to training"""
    def __init__(self, num_classes=19, sr_model=None):
        super().__init__()
        self.sr_model = sr_model
        
        # Freeze SR model
        if self.sr_model:
            for param in self.sr_model.parameters():
                param.requires_grad = False
        
        # Backbone: ResNet50 (matches checkpoint: module.backbone.*)
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.backbone = models.resnet50(weights=weights)
        
        # Replace final fc layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, lr_images):
        # SR enhancement: 32x32 -> 256x256
        if self.sr_model:
            with torch.no_grad():
                sr_images = self.sr_model(lr_images)
        else:
            sr_images = lr_images
        
        # Resize to 224x224 for ResNet50
        sr_images = F.interpolate(sr_images, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Classification
        output = self.backbone(sr_images)
        return output


# ===============================================================================
# MODEL LOADING
# ===============================================================================

@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        st.info("🔧 Building SR Generator...")
        sr_model = Generator(num_rrdb=12, num_rrfdb=6, nf=64)
        
        st.info("🔧 Building Classifier...")
        model = SREnhancedClassifier(num_classes=NUM_CLASSES, sr_model=sr_model)
        
        st.info("📦 Loading weights from best_classifier.pth...")
        checkpoint = torch.load("best_classifier.pth", map_location=DEVICE)
        
        # Strip 'module.' prefix from DataParallel checkpoint
        if isinstance(checkpoint, dict):
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix from keys
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')  # Remove 'module.' prefix
            new_state_dict[name] = v
        
        # Load with strict=False to see any remaining issues
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            st.warning(f"⚠️ Missing keys: {len(missing_keys)} keys")
            st.text(missing_keys[:10])  # Show first 10
        
        if unexpected_keys:
            st.warning(f"⚠️ Unexpected keys: {len(unexpected_keys)} keys")
            st.text(unexpected_keys[:10])  # Show first 10
        
        model.to(DEVICE)
        model.eval()
        
        st.success("✅ Model loaded successfully!")
        return model
    
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        raise


# ===============================================================================
# PREDICTION
# ===============================================================================

def predict(image: Image.Image, model) -> Dict[str, float]:
    """Run inference on image"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 32x32 (model expects this size)
    image = image.resize((32, 32), Image.BICUBIC)
    
    # Convert to tensor (normalize to [0, 1])
    img_array = np.array(image, dtype=np.float32) / 255.0
    input_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    
    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
    
    # Create results dictionary
    results = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(NUM_CLASSES)}
    
    # Sort by probability
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    
    return results


# ===============================================================================
# STREAMLIT UI
# ===============================================================================

def main():
    st.set_page_config(
        page_title="BestClassifier - Land Cover Classification",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 BestClassifier: Land Cover Classification")
    st.markdown("""
    Upload a **32x32 satellite image** to classify land cover type. 
    The model uses super-resolution enhancement (32x32 → 256x256 → 224x224) 
    before classification with ResNet50.
    """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a 32x32 satellite image"
    )
    
    if uploaded_file is not None:
        # Load and display image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, caption=f"Original ({image.size[0]}x{image.size[1]})", use_column_width=True)
        
        with col2:
            st.subheader("Top 5 Predictions")
            
            # Run prediction
            with st.spinner("Classifying..."):
                results = predict(image, model)
            
            # Display top 5
            top5 = list(results.items())[:5]
            for i, (class_name, prob) in enumerate(top5, 1):
                st.metric(
                    label=f"{i}. {class_name}",
                    value=f"{prob*100:.2f}%"
                )
                st.progress(prob)
        
        # Show all results in expander
        with st.expander("📊 See all class probabilities"):
            for class_name, prob in results.items():
                st.write(f"**{class_name}**: {prob*100:.2f}%")
    
    # Sidebar info
    with st.sidebar:
        st.header("ℹ️ Model Info")
        st.markdown("""
        **Architecture:**
        - SR Generator: 12 RRDB + 6 RRFDB blocks
        - 8x Super Resolution (32→256)
        - ResNet50 Classifier (256→224)
        
        **Input:** 32x32 RGB satellite image
        **Output:** 19 land cover classes
        
        **Classes:**
        1. Urban fabric
        2. Industrial/commercial
        3. Arable land
        4. Permanent crops
        5. Pastures
        ... and 14 more
        """)
        
        st.markdown("---")
        st.markdown(f"**Device:** {DEVICE}")
        st.markdown(f"**Model Size:** ~400MB")


if __name__ == "__main__":
    main()
