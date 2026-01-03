# 🔧 Manual Deployment Guide - BestClassifier

Since the automated script had issues, follow these manual steps:

## Step-by-Step Deployment

### 1. Navigate to Project Directory
```powershell
cd C:\Users\sudar\OneDrive\Desktop\majorProject
```

### 2. Clone HuggingFace Space
```powershell
git clone https://huggingface.co/spaces/HegdeSudarshan/bestClassifier
cd bestClassifier
```

### 3. Setup Git LFS
```powershell
git lfs install
git lfs track "*.pth"
git lfs track "*.bin"
git add .gitattributes
```

### 4. Copy Files from Deployment Directory
```powershell
# Copy app.py
Copy-Item ..\best-classifier-deployment\app.py -Destination . -Force

# Copy requirements.txt
Copy-Item ..\best-classifier-deployment\requirements.txt -Destination . -Force

# Copy README
Copy-Item ..\best-classifier-deployment\README.md -Destination . -Force

# Copy model file
Copy-Item ..\best_classifier.pth -Destination . -Force
```

### 5. Verify Files
```powershell
# Check what we have
dir

# You should see:
# - app.py
# - requirements.txt
# - README.md
# - best_classifier.pth (should be ~400-500 MB)
# - .gitattributes
```

### 6. Check Status
```powershell
git status
```

You should see the new files listed as untracked.

### 7. Commit and Push
```powershell
# Add all files
git add .

# Commit
git commit -m "Initial deployment of bestClassifier"

# Push to HuggingFace
git push origin main
```

**Note:** The first push might take a while (5-15 minutes) because the model file is large (~400-500 MB).

### 8. Monitor Build

1. Go to: https://huggingface.co/spaces/HegdeSudarshan/bestClassifier
2. You should see "Building..." status
3. Wait 5-10 minutes for build to complete
4. Once complete, you'll see the Gradio interface

## 🔍 Troubleshooting

### "Push rejected" Error
```powershell
# Make sure you're logged in to HuggingFace
git config --global credential.helper store
git push origin main
```

When prompted, enter:
- **Username:** HegdeSudarshan
- **Password:** Your HuggingFace Access Token (not your password!)

**Get Token:** https://huggingface.co/settings/tokens

### "File too large" Error
```powershell
# Make sure Git LFS is tracking the file
git lfs track "*.pth"
git add .gitattributes
git add .
git commit --amend --no-edit
git push origin main --force
```

### Files Not Copying
If `Copy-Item` doesn't work, check paths:
```powershell
# Check if source file exists
Test-Path ..\best-classifier-deployment\app.py

# If False, use full path:
Copy-Item "C:\Users\sudar\OneDrive\Desktop\majorProject\best-classifier-deployment\app.py" -Destination . -Force
```

### Build Fails on HuggingFace
1. Check build logs in Space Settings
2. Common issues:
   - **Missing dependencies:** Check requirements.txt
   - **Model file corrupted:** Re-upload using Git LFS
   - **Architecture mismatch:** Verify app.py has correct model code

## ✅ Success Checklist

After pushing, verify:

- [ ] Space shows "Building" status
- [ ] No errors in git push output
- [ ] Model file size is correct (~400-500 MB)
- [ ] All files present: app.py, requirements.txt, README.md, best_classifier.pth

## 🎉 After Successful Build

Once build completes:

1. **Test the Interface**
   - Upload a sample 32×32 satellite image
   - Verify predictions appear
   
2. **Test the API**
   ```python
   from gradio_client import Client
   client = Client("HegdeSudarshan/bestClassifier")
   result = client.predict("test_image.png", api_name="/predict")
   print(result)
   ```

3. **Integrate with Backend**
   - Update `geo-agri-analyst/backend/app/huggingface_service.py`
   - Change space URL to your new deployment

## 📊 Expected Timeline

- Copy files: 1 minute
- Push to HF: 5-15 minutes (large model file)
- Build: 5-10 minutes
- **Total: ~20-25 minutes**

---

**Need help?** Check the [complete guide](BESTCLASSIFIER_DEPLOYMENT_GUIDE.md) or HuggingFace documentation.
