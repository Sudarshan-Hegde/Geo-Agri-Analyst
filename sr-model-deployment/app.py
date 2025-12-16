"""
RFB-ESRGAN Super-Resolution Model - Hugging Face Gradio App
Upscales low-resolution satellite/agricultural images from 32x32 to 256x256 (8x)
"""

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import io

# ========== MODEL ARCHITECTURE ==========

class RFB(nn.Module):
    """Receptive Field Block - Multi-scale feature extraction"""
    def __init__(self, in_channels=64):
        super(RFB, self).__init__()
        # Branch 1: AvgPool(3) + 1x1 conv + dilated 3x3 (d=1)
        self.branch1 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, 16, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, padding=1, dilation=1),
            nn.ReLU(inplace=True)
        )
        
        # Branch 2: AvgPool(5) + 1x1 conv + dilated 3x3 (d=2)
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(5, stride=1, padding=2),
            nn.Conv2d(in_channels, 24, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, padding=2, dilation=2),
            nn.ReLU(inplace=True)
        )
        
        # Branch 3: AvgPool(7) + 1x1 conv + dilated 3x3 (d=3)
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(7, stride=1, padding=3),
            nn.Conv2d(in_channels, 24, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, 3, 1, padding=3, dilation=3),
            nn.ReLU(inplace=True)
        )
        
        # Concat 16+24+24=64 → 1x1 conv to 64
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
    """Dense Block with 5 convolutions (from ESRGAN RRDB)"""
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
    """Residual-in-Residual Dense Block (ESRGAN)"""
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
    """Residual Receptive Field Dense Block"""
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
    """RFB-ESRGAN Generator - x8 upscale (32→256)"""
    def __init__(self, num_rrdb=12, num_rrfdb=6, nf=64):
        super(Generator, self).__init__()
        # First conv
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        
        # Trunk-A: 12 RRDBs
        self.trunk_a = nn.Sequential(*[RRDB(nf) for _ in range(num_rrdb)])
        
        # Trunk-RFB: 6 RRFDBs
        self.trunk_rfb = nn.Sequential(*[RRFDB(nf) for _ in range(num_rrfdb)])
        
        # Single RFB before upsampling
        self.rfb_up = RFB(nf)
        
        # Upsampling for x8: x2 → x2 → x2 (32 → 64 → 128 → 256)
        self.upsample = nn.Sequential(
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # x2: 32 → 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # x2: 64 → 128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, nf * 4, 3, 1, 1),
            nn.PixelShuffle(2),  # x2: 128 → 256
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final convs
        self.conv_final = nn.Sequential(
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf, 3, 3, 1, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        feat = self.conv_first(x)
        trunk_a_out = self.trunk_a(feat)
        trunk_rfb_out = self.trunk_rfb(trunk_a_out)
        rfb_out = self.rfb_up(trunk_rfb_out)
        upsampled = self.upsample(rfb_out)
        out = self.conv_final(upsampled)
        return out


# ========== MODEL LOADING ==========

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize model
model = Generator(num_rrdb=12, num_rrfdb=6).to(device)
model.eval()

# Load ensemble weights (download from your trained model)
# You'll need to upload generator_ensemble.pth to Hugging Face
try:
    model_path = "generator_ensemble.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"⚠️ Model loading failed: {e}")
    print("Using randomly initialized weights (for testing only)")


# ========== IMAGE PROCESSING ==========

def preprocess_image(image):
    """Convert PIL image to tensor and normalize to [-1, 1]"""
    transform = transforms.Compose([
        transforms.Resize((32, 32), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform(image).unsqueeze(0)


def postprocess_image(tensor):
    """Convert tensor back to PIL image"""
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and PIL
    img_array = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)


# ========== GRADIO INTERFACE ==========

def upscale_image(input_image):
    """
    Main function for super-resolution
    Input: PIL Image (any size)
    Output: Upscaled PIL Image (256x256)
    """
    if input_image is None:
        return None
    
    try:
        # Ensure RGB
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        # Preprocess
        lr_tensor = preprocess_image(input_image).to(device)
        
        # Super-resolve
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        
        # Postprocess
        sr_image = postprocess_image(sr_tensor)
        
        # Also create bicubic baseline for comparison
        bicubic_image = input_image.resize((256, 256), Image.BICUBIC)
        
        return bicubic_image, sr_image
        
    except Exception as e:
        print(f"Error during upscaling: {e}")
        return None, None


# ========== GRADIO APP ==========

demo = gr.Interface(
    fn=upscale_image,
    inputs=gr.Image(type="pil", label="Upload Low-Resolution Image"),
    outputs=[
        gr.Image(type="pil", label="Bicubic Upscale (Baseline)"),
        gr.Image(type="pil", label="RFB-ESRGAN Super-Resolution (8x)")
    ],
    title="🌾 RFB-ESRGAN Agricultural Super-Resolution (8x)",
    description="""
    **Upload a low-resolution agricultural/satellite image** to upscale it 8x (from 32×32 to 256×256).
    
    This model uses RFB-ESRGAN trained specifically for agricultural imagery:
    - **Architecture**: 12 RRDBs + 6 RRFDBs with Receptive Field Blocks
    - **Training**: 2-stage (PSNR + GAN) with perceptual losses
    - **Upscale Factor**: 8x (32×32 → 256×256)
    - **Optimized For**: Satellite imagery, crop fields, agricultural scenes
    
    The model preserves fine details like crop textures, field boundaries, and vegetation patterns.
    """,
    examples=[
        # Add example images here if available
    ],
    article="""
    ### About This Model
    
    **RFB-ESRGAN** combines Enhanced Super-Resolution GAN (ESRGAN) with Receptive Field Blocks (RFB) 
    for superior multi-scale feature extraction. 
    
    **Training Details:**
    - Stage 1: 20 epochs of PSNR-oriented training (L1 pixel loss)
    - Stage 2: 200,000 iterations of GAN training with:
      - Pixel loss (L1)
      - Perceptual loss (VGG19)
      - Adversarial loss (Relativistic GAN)
      - Spectral normalization for stability
    
    **Use Cases:**
    - Enhancing low-resolution satellite imagery
    - Improving crop monitoring images
    - Agricultural land analysis
    - Precision farming applications
    
    **Integration:**
    Use the API endpoint for programmatic access in your Geo-Agri-Analyst application.
    
    **Citation:** Based on ESRGAN and RFB-ESRGAN architectures
    """,
    allow_flagging="never",
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch(share=True)
