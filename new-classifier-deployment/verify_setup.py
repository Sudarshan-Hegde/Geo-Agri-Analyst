#!/usr/bin/env python3
"""
Pre-deployment verification script
Checks if all files and dependencies are ready
"""

import os
import sys
import subprocess
from pathlib import Path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status message"""
    if status == 'success':
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == 'error':
        print(f"{RED}❌ {message}{RESET}")
    elif status == 'warning':
        print(f"{YELLOW}⚠️  {message}{RESET}")
    else:
        print(f"{BLUE}ℹ️  {message}{RESET}")

def check_file(filepath, description):
    """Check if file exists and show size"""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        print_status(f"{description}: {size_mb:.2f} MB", 'success')
        return True
    else:
        print_status(f"{description} not found at: {filepath}", 'error')
        return False

def check_command(command, description):
    """Check if command exists"""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            print_status(f"{description} installed", 'success')
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    print_status(f"{description} not found", 'warning')
    return False

def check_python_package(package, description):
    """Check if Python package is installed"""
    try:
        __import__(package)
        print_status(f"{description} installed", 'success')
        return True
    except ImportError:
        print_status(f"{description} not installed", 'warning')
        return False

def main():
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Pre-Deployment Verification{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    all_checks_passed = True
    
    # Check deployment files
    print(f"\n{BLUE}📁 Checking Deployment Files{RESET}")
    print("-" * 60)
    
    deployment_dir = "/home/sudarshanhegde/Sudarshan_Hegde/majorProject/new-classifier-deployment"
    
    files_to_check = [
        (f"{deployment_dir}/app.py", "Gradio app"),
        (f"{deployment_dir}/requirements.txt", "Requirements file"),
        (f"{deployment_dir}/README.md", "README"),
        (f"{deployment_dir}/DEPLOYMENT_GUIDE.md", "Deployment guide"),
        (f"{deployment_dir}/deploy.sh", "Deployment script"),
    ]
    
    for filepath, desc in files_to_check:
        if not check_file(filepath, desc):
            all_checks_passed = False
    
    # Check model weights
    print(f"\n{BLUE}🧠 Checking Model Weights{RESET}")
    print("-" * 60)
    
    classifier_path = "/home/sudarshanhegde/Sudarshan_Hegde/majorProject/classifierHuggingFace/best_classifier.pth"
    
    if check_file(classifier_path, "Classifier weights"):
        size_mb = os.path.getsize(classifier_path) / (1024 * 1024)
        if size_mb < 10:
            print_status(f"Classifier weights seem small ({size_mb:.2f} MB). Expected: 100-500 MB", 'warning')
            all_checks_passed = False
    else:
        all_checks_passed = False
    
    # Check system dependencies
    print(f"\n{BLUE}🔧 Checking System Dependencies{RESET}")
    print("-" * 60)
    
    check_command('git', 'Git')
    check_command('git-lfs', 'Git LFS')
    
    # Check Python packages
    print(f"\n{BLUE}🐍 Checking Python Packages{RESET}")
    print("-" * 60)
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('gradio', 'Gradio'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
    ]
    
    for package, desc in packages:
        check_python_package(package, desc)
    
    # Check Hugging Face CLI
    print(f"\n{BLUE}🤗 Checking Hugging Face CLI{RESET}")
    print("-" * 60)
    
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            username = result.stdout.strip()
            print_status(f"Logged in as: {username}", 'success')
        else:
            print_status("Not logged in to Hugging Face", 'warning')
            print("   Run: huggingface-cli login")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("Hugging Face CLI not installed", 'warning')
        print("   Install: pip install -U 'huggingface_hub[cli]'")
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    if all_checks_passed:
        print_status("All critical checks passed! Ready to deploy.", 'success')
        print(f"\n{GREEN}Run: ./deploy.sh{RESET}")
    else:
        print_status("Some checks failed. Please fix issues before deploying.", 'error')
        print(f"\n{YELLOW}Check the messages above for details{RESET}")
        return 1
    
    print(f"{BLUE}{'='*60}{RESET}\n")
    return 0

if __name__ == "__main__":
    sys.exit(main())
