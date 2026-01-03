# BestClassifier Deployment Script - FIXED VERSION
# Usage: .\deploy_fixed.ps1

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host " BestClassifier Deployment (Fixed)" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Define paths at the top
$projectDir = "C:\Users\sudar\OneDrive\Desktop\majorProject"
$modelPath = Join-Path $projectDir "best_classifier.pth"
$deploymentDir = Join-Path $projectDir "best-classifier-deployment"
$spaceName = "bestClassifier"

# Check prerequisites
Write-Host "Checking prerequisites..." -ForegroundColor Yellow

# Check git
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Git not found" -ForegroundColor Red
    Write-Host "Install from: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Git found" -ForegroundColor Green

# Check git-lfs
if (!(Get-Command git-lfs -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] Git LFS not found" -ForegroundColor Red
    Write-Host "Install from: https://git-lfs.github.com/" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Git LFS found" -ForegroundColor Green

# Check model file
if (!(Test-Path $modelPath)) {
    Write-Host "[ERROR] Model file not found: $modelPath" -ForegroundColor Red
    exit 1
}
$modelSize = (Get-Item $modelPath).Length / 1MB
$modelSizeMB = [math]::Round($modelSize, 2)
Write-Host "[OK] Model file found ($modelSizeMB MB)" -ForegroundColor Green

# Check deployment files
if (!(Test-Path $deploymentDir)) {
    Write-Host "[ERROR] Deployment directory not found: $deploymentDir" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Deployment directory found" -ForegroundColor Green

# Check individual deployment files
$requiredFiles = @("app.py", "requirements.txt", "README.md")
foreach ($file in $requiredFiles) {
    $filePath = Join-Path $deploymentDir $file
    if (!(Test-Path $filePath)) {
        Write-Host "[ERROR] Missing file: $file" -ForegroundColor Red
        exit 1
    }
    Write-Host "  [OK] Found: $file" -ForegroundColor Gray
}

Write-Host ""
Write-Host "All prerequisites met!" -ForegroundColor Green
Write-Host ""

# Navigate to project directory
Set-Location $projectDir

# Clone or update repository
$spaceDir = Join-Path $projectDir $spaceName
if (Test-Path $spaceDir) {
    Write-Host "Space directory exists, cleaning..." -ForegroundColor Yellow
    $confirm = Read-Host "Delete existing '$spaceName' directory? (y/n)"
    if ($confirm -eq "y") {
        Remove-Item -Path $spaceDir -Recurse -Force
    } else {
        Write-Host "Using existing directory..." -ForegroundColor Yellow
        Set-Location $spaceDir
        git pull origin main
        Write-Host ""
    }
}

if (!(Test-Path $spaceDir)) {
    Write-Host "Cloning HuggingFace Space..." -ForegroundColor Yellow
    git clone https://huggingface.co/spaces/HegdeSudarshan/bestClassifier $spaceName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to clone repository" -ForegroundColor Red
        Write-Host "Make sure you have access to the space or create it first:" -ForegroundColor Yellow
        Write-Host "https://huggingface.co/new-space" -ForegroundColor Cyan
        exit 1
    }
    Set-Location $spaceDir
}

# Setup Git LFS
Write-Host ""
Write-Host "Setting up Git LFS..." -ForegroundColor Yellow
git lfs install
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.onnx"

# Make sure .gitattributes exists
if (Test-Path ".gitattributes") {
    git add .gitattributes
    Write-Host "[OK] Git LFS configured" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to create .gitattributes" -ForegroundColor Red
    exit 1
}

# Copy deployment files
Write-Host ""
Write-Host "Copying deployment files..." -ForegroundColor Yellow

$filesToCopy = @(
    @{Source = Join-Path $deploymentDir "app.py"; Dest = "app.py"},
    @{Source = Join-Path $deploymentDir "requirements.txt"; Dest = "requirements.txt"},
    @{Source = Join-Path $deploymentDir "README.md"; Dest = "README.md"}
)

foreach ($file in $filesToCopy) {
    Write-Host "  Copying $($file.Dest)..." -ForegroundColor Gray
    Copy-Item -Path $file.Source -Destination $file.Dest -Force
    if (Test-Path $file.Dest) {
        Write-Host "  [OK] Copied $($file.Dest)" -ForegroundColor Green
    } else {
        Write-Host "  [ERROR] Failed to copy $($file.Dest)" -ForegroundColor Red
        exit 1
    }
}

# Copy model weights
Write-Host ""
Write-Host "Copying model weights (this may take a moment)..." -ForegroundColor Yellow
Copy-Item -Path $modelPath -Destination "best_classifier.pth" -Force
if (Test-Path "best_classifier.pth") {
    $copiedSize = (Get-Item "best_classifier.pth").Length / 1MB
    $copiedSizeMB = [math]::Round($copiedSize, 2)
    Write-Host "[OK] Model file copied ($copiedSizeMB MB)" -ForegroundColor Green
} else {
    Write-Host "[ERROR] Failed to copy model file" -ForegroundColor Red
    exit 1
}

# Show status
Write-Host ""
Write-Host "Files to be committed:" -ForegroundColor Yellow
git status --short

# Verify we have changes
$hasChanges = git status --porcelain
if ([string]::IsNullOrWhiteSpace($hasChanges)) {
    Write-Host ""
    Write-Host "No changes detected. Files may already be up to date." -ForegroundColor Yellow
    Write-Host "Check your space at: https://huggingface.co/spaces/HegdeSudarshan/bestClassifier" -ForegroundColor Cyan
    exit 0
}

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
    $commitMsg = "Update bestClassifier deployment - $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
}

git commit -m $commitMsg

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Commit failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Pushing to HuggingFace (this may take several minutes for large model)..." -ForegroundColor Yellow
Write-Host "Progress:" -ForegroundColor Gray
git push origin main

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Push failed" -ForegroundColor Red
    Write-Host ""
    Write-Host "Common fixes:" -ForegroundColor Yellow
    Write-Host "1. Make sure you're logged in to HuggingFace" -ForegroundColor Gray
    Write-Host "   git config --global credential.helper store" -ForegroundColor Gray
    Write-Host ""
    Write-Host "2. Use your HuggingFace token (not password) when prompted" -ForegroundColor Gray
    Write-Host "   Get token: https://huggingface.co/settings/tokens" -ForegroundColor Cyan
    Write-Host ""
    exit 1
}

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
Write-Host "Build typically takes 5-10 minutes. Check the space URL to see when it's ready!" -ForegroundColor Yellow
Write-Host ""
