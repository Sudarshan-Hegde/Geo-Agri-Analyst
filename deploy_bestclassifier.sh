#!/bin/bash

# Deployment script for bestClassifier to HuggingFace Spaces
# Usage: ./deploy_bestclassifier.sh

set -e  # Exit on error

echo "========================================="
echo "BestClassifier Deployment Script"
echo "========================================="
echo ""

# Configuration
SPACE_NAME="HegdeSudarshan/bestClassifier"
MODEL_PATH="bestClassifier.pth"
DEPLOYMENT_DIR="best-classifier-deployment"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check prerequisites
echo "Checking prerequisites..."

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}✗ huggingface-cli not found${NC}"
    echo "Install with: pip install -U 'huggingface_hub[cli]'"
    exit 1
fi
echo -e "${GREEN}✓ huggingface-cli found${NC}"

# Check if git-lfs is installed
if ! command -v git-lfs &> /dev/null; then
    echo -e "${RED}✗ git-lfs not found${NC}"
    echo "Install with: sudo apt-get install git-lfs"
    exit 1
fi
echo -e "${GREEN}✓ git-lfs found${NC}"

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}✗ Model file not found: $MODEL_PATH${NC}"
    echo "Please ensure bestClassifier.pth is in the current directory"
    exit 1
fi
echo -e "${GREEN}✓ Model file found${NC}"

# Check if deployment files exist
if [ ! -d "$DEPLOYMENT_DIR" ]; then
    echo -e "${RED}✗ Deployment directory not found: $DEPLOYMENT_DIR${NC}"
    echo "Please create deployment files first"
    exit 1
fi
echo -e "${GREEN}✓ Deployment directory found${NC}"

echo ""
echo "All prerequisites met!"
echo ""

# Login check
echo "Checking HuggingFace authentication..."
if ! huggingface-cli whoami &> /dev/null; then
    echo -e "${YELLOW}Not logged in to HuggingFace${NC}"
    echo "Please login:"
    huggingface-cli login
fi
echo -e "${GREEN}✓ Authenticated with HuggingFace${NC}"
echo ""

# Clone or update repository
if [ -d "$SPACE_NAME" ]; then
    echo -e "${YELLOW}Space directory exists, pulling latest...${NC}"
    cd "$SPACE_NAME"
    git pull
else
    echo "Cloning HuggingFace Space..."
    git clone https://huggingface.co/spaces/$SPACE_NAME
    cd "$(basename $SPACE_NAME)"
fi

# Setup Git LFS
echo "Setting up Git LFS..."
git lfs install
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.onnx"
git add .gitattributes

# Copy deployment files
echo "Copying deployment files..."
cp ../$DEPLOYMENT_DIR/app.py .
cp ../$DEPLOYMENT_DIR/requirements.txt .
cp ../$DEPLOYMENT_DIR/README.md .

# Copy label mapping if exists
if [ -f "../$DEPLOYMENT_DIR/label_indices.json" ]; then
    cp ../$DEPLOYMENT_DIR/label_indices.json .
    echo -e "${GREEN}✓ Copied label_indices.json${NC}"
fi

# Copy model weights
echo "Copying model weights..."
cp ../$MODEL_PATH .
echo -e "${GREEN}✓ Model file copied ($(du -h $MODEL_PATH | cut -f1))${NC}"

# Show status
echo ""
echo "Files to be committed:"
git status --short

# Ask for confirmation
echo ""
read -p "Proceed with deployment? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 0
fi

# Commit and push
echo ""
echo "Committing changes..."
git add .

# Get commit message
echo ""
read -p "Enter commit message (or press Enter for default): " COMMIT_MSG
if [ -z "$COMMIT_MSG" ]; then
    COMMIT_MSG="Update bestClassifier deployment - $(date +%Y-%m-%d)"
fi

git commit -m "$COMMIT_MSG"

echo ""
echo "Pushing to HuggingFace..."
git push origin main

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}Deployment Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Your space is deploying at:"
echo "https://huggingface.co/spaces/$SPACE_NAME"
echo ""
echo "Monitor build logs:"
echo "https://huggingface.co/spaces/$SPACE_NAME/settings"
echo ""
echo -e "${YELLOW}Note: Build takes 5-10 minutes${NC}"
echo ""
