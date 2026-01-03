# BestClassifier Deployment Script for Windows PowerShell
# Usage: .\deploy_bestclassifier.ps1

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " BestClassifier Deployment Script" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "✗ Git not found" -ForegroundColor Red
    Write-Host "Install from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Git found" -ForegroundColor Green

# Check git-lfs
if (!(Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Host "✗ Git LFS not found" -ForegroundColor Red
    Write-Host "Install from: https://git-lfs.github.com/" -ForegroundColor Yellow
    exit 1
}
Write-Host "✓ Git LFS found" -ForegroundColor Green

# Check model file
$modelPath = "C:\Users\sudar\OneDrive\Desktop\majorProject\best_classifier.pth"
if (!(Test-Path $modelPath)) {
    Write-Host "✗ Model file not found: $modelPath" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Model file found" -ForegroundColor Green

# Check deployment files
$deploymentDir = "C:\Users\sudar\OneDrive\Desktop\majorProject\best-classifier-deployment"
if (!(Test-Path $deploymentDir)) {
    Write-Host "✗ Deployment directory not found: $deploymentDir" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Deployment directory found" -ForegroundColor Green

Write-Host ""
Write-Host "All prerequisites met!" -ForegroundColor Green
Write-Host ""

# Navigate to project directory
Set-Location "C:\Users\sudar\OneDrive\Desktop\majorProject"

# Clone or update repository
$spaceName = "bestClassifier"
if (Test-Path $spaceName) {
    Write-Host "Space directory exists, pulling latest..." -ForegroundColor Yellow
    Set-Location $spaceName
    git pull
} else {
    Write-Host "Cloning HuggingFace Space..." -ForegroundColor Yellow
    git clone https://huggingface.co/spaces/HegdeSudarshan/bestClassifier
    Set-Location $spaceName
}

# Setup Git LFS
Write-Host ""
Write-Host "Setting up Git LFS..." -ForegroundColor Yellow
git lfs install
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.onnx"
git add .gitattributes

# Copy deployment files
Write-Host ""
Write-Host "Copying deployment files..." -ForegroundColor Yellow
Copy-Item "$deploymentDir\app.py" -Destination . -Force
Copy-Item "$deploymentDir\requirements.txt" -Destination . -Force
Copy-Item "$deploymentDir\README.md" -Destination . -Force

Write-Host "✓ Deployment files copied" -ForegroundColor Green

# Copy model weights
Write-Host ""
Write-Host "Copying model weights..." -ForegroundColor Yellow
Copy-Item $modelPath -Destination "best_classifier.pth" -Force
$fileSize = (Get-Item "best_classifier.pth").Length / 1MB
Write-Host "✓ Model file copied ($([math]::Round($fileSize, 2)) MB)" -ForegroundColor Green

# Show status
Write-Host ""
Write-Host "Files to be committed:" -ForegroundColor Yellow
git status --short

# Ask for confirmation
Write-Host ""
$confirm = Read-Host "Proceed with deployment? (y/n)"
if ($confirm -ne "y") {
    Write-Host "Deployment cancelled" -ForegroundColor Yellow
    exit 0
}

# Commit and push
Write-Host ""
Write-Host "Committing changes..." -ForegroundColor Yellow
git add .

$commitMsg = Read-Host "Enter commit message (or press Enter for default)"
if ([string]::IsNullOrWhiteSpace($commitMsg)) {
    $commitMsg = "Update bestClassifier deployment - $(Get-Date -Format 'yyyy-MM-dd')"
}

git commit -m $commitMsg

Write-Host ""
Write-Host "Pushing to HuggingFace..." -ForegroundColor Yellow
git push origin main

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host " Deployment Complete!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Your space is deploying at:" -ForegroundColor Cyan
Write-Host "https://huggingface.co/spaces/HegdeSudarshan/bestClassifier" -ForegroundColor Cyan
Write-Host ""
Write-Host "Monitor build logs:" -ForegroundColor Yellow
Write-Host "https://huggingface.co/spaces/HegdeSudarshan/bestClassifier/settings" -ForegroundColor Yellow
Write-Host ""
Write-Host "Note: Build takes 5-10 minutes" -ForegroundColor Yellow
Write-Host ""
