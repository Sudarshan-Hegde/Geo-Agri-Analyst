#!/bin/bash

# Automated Deployment Script for Hugging Face Space
# Usage: ./deploy.sh

set -e  # Exit on error

echo "🚀 Starting Hugging Face Space Deployment..."
echo "================================================"

# Configuration
SPACE_URL="https://huggingface.co/spaces/HegdeSudarshan/Classifier"
DEPLOYMENT_DIR="/home/sudarshanhegde/Sudarshan_Hegde/majorProject/new-classifier-deployment"
CLASSIFIER_WEIGHTS="/home/sudarshanhegde/Sudarshan_Hegde/majorProject/classifierHuggingFace/best_classifier.pth"
TEMP_CLONE_DIR="/tmp/hf-classifier-deploy"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli not found. Installing..."
    pip install -U "huggingface_hub[cli]"
fi

# Check if Git LFS is installed
if ! command -v git-lfs &> /dev/null; then
    echo "⚠️  Git LFS not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y git-lfs
    git lfs install
fi

# Check if classifier weights exist
if [ ! -f "$CLASSIFIER_WEIGHTS" ]; then
    echo "❌ Classifier weights not found at: $CLASSIFIER_WEIGHTS"
    echo "Please ensure the file exists before deploying."
    exit 1
fi

echo "✅ Prerequisites checked"

# Login check
echo ""
echo "🔑 Checking Hugging Face authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "Please login to Hugging Face:"
    huggingface-cli login
else
    echo "✅ Already logged in as: $(huggingface-cli whoami)"
fi

# Clean up previous clone if exists
if [ -d "$TEMP_CLONE_DIR" ]; then
    echo "🧹 Cleaning up previous clone..."
    rm -rf "$TEMP_CLONE_DIR"
fi

# Clone the space
echo ""
echo "📦 Cloning Hugging Face Space..."
git clone "$SPACE_URL" "$TEMP_CLONE_DIR"
cd "$TEMP_CLONE_DIR"

# Initialize Git LFS
echo ""
echo "🔧 Setting up Git LFS..."
git lfs install
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.onnx"
git add .gitattributes

# Copy deployment files
echo ""
echo "📋 Copying deployment files..."
cp "$DEPLOYMENT_DIR/app.py" .
cp "$DEPLOYMENT_DIR/requirements.txt" .
cp "$DEPLOYMENT_DIR/README.md" .
cp "$DEPLOYMENT_DIR/DEPLOYMENT_GUIDE.md" .

# Copy model weights
echo "📦 Copying model weights..."
cp "$CLASSIFIER_WEIGHTS" best_classifier.pth

# Check file sizes
echo ""
echo "📊 File sizes:"
ls -lh *.pth 2>/dev/null || echo "No .pth files yet"
ls -lh app.py requirements.txt README.md

# Verify model file
if [ -f "best_classifier.pth" ]; then
    SIZE=$(stat -f%z "best_classifier.pth" 2>/dev/null || stat -c%s "best_classifier.pth")
    SIZE_MB=$((SIZE / 1024 / 1024))
    echo "✅ Classifier weights: ${SIZE_MB} MB"
    
    if [ $SIZE_MB -lt 10 ]; then
        echo "⚠️  Warning: Classifier weights seem too small (${SIZE_MB} MB)"
        echo "   Expected size: 100-500 MB for ResNet50"
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "❌ Failed to copy classifier weights"
    exit 1
fi

# Git status
echo ""
echo "📝 Git status:"
git status --short

# Commit and push
echo ""
read -p "🚀 Ready to deploy. Continue? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📤 Committing and pushing to Hugging Face..."
    
    git add .
    
    git commit -m "Deploy SR-ResNet50 Classifier

- RFB-ESRGAN: 12 RRDB + 6 RRFDB blocks (8× super-resolution)
- ResNet50: Enhanced classifier head with dual dropout
- Training: 100k samples, 50 epochs, EMA optimization
- Classes: 19 BigEarthNet-S2 land cover categories
- Features: Label smoothing, warmup LR, comprehensive evaluation

Performance:
- Validation Accuracy: 100%
- Training Time: ~8-10 hours
- Regularization: EMA (0.9995), Label Smoothing (0.15)

Deployment fixes:
- ✅ DataParallel wrapper handling
- ✅ Correct image sizes (32→256→224)
- ✅ Proper model architecture match
- ✅ Device handling (CUDA/CPU)
- ✅ Stable dependency versions"
    
    git push
    
    echo ""
    echo "✅ Deployment complete!"
    echo ""
    echo "🌐 Your space is being built at:"
    echo "   $SPACE_URL"
    echo ""
    echo "⏱️  Build typically takes 5-10 minutes"
    echo ""
    echo "Next steps:"
    echo "1. Visit the space URL above"
    echo "2. Wait for 'Building' status to change to 'Running'"
    echo "3. Test with sample satellite images"
    echo ""
    echo "📚 Check DEPLOYMENT_GUIDE.md for troubleshooting"
else
    echo "❌ Deployment cancelled"
    exit 1
fi

# Cleanup
echo ""
read -p "🧹 Clean up temporary clone directory? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd /
    rm -rf "$TEMP_CLONE_DIR"
    echo "✅ Cleaned up"
fi

echo ""
echo "🎉 All done!"
