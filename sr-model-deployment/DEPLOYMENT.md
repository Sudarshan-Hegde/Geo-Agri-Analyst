# Quick Deployment Checklist

## 🚀 Steps to Deploy RFB-ESRGAN to Hugging Face

### 1. Prerequisites
- ✅ Hugging Face account created
- ✅ Trained model file: `generator_ensemble.pth`
- ✅ Git LFS installed (for large files)

### 2. Get Your Trained Model
After training completes in Colab:
```python
# In your Colab notebook, run this cell:
from google.colab import files
files.download('/content/drive/MyDrive/RFB-ESRGAN-Output/generator_ensemble.pth')
```
Place the downloaded file in the `sr-model-deployment/` directory.

### 3. Test Locally (Optional but Recommended)
```bash
cd sr-model-deployment
python app.py
# Open http://localhost:7860 and test the interface
```

### 4. Create Hugging Face Space
1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `rfb-esrgan-agricultural-sr`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free) or GPU ($)
3. Click "Create Space"

### 5. Upload Files

**Option A: Web Upload** (Easier)
1. Go to your Space page
2. Click "Files" → "Add file" → "Upload files"
3. Upload all files:
   - `app.py`
   - `generator_ensemble.pth`
   - `requirements.txt`
   - `README.md`
4. Commit changes

**Option B: Git Push** (Recommended for large files)
```bash
# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/rfb-esrgan-agricultural-sr
cd rfb-esrgan-agricultural-sr

# Setup Git LFS for model file
git lfs install
git lfs track "*.pth"

# Copy files
cp ../sr-model-deployment/* .

# Commit and push
git add .
git commit -m "Deploy RFB-ESRGAN SR model"
git push
```

### 6. Wait for Build
- Hugging Face will automatically build your Space
- Watch the logs for any errors
- Build time: ~2-5 minutes

### 7. Test Deployment
- Visit your Space URL: `https://huggingface.co/spaces/YOUR_USERNAME/rfb-esrgan-agricultural-sr`
- Upload a test image
- Verify the output looks correct

### 8. Get API Endpoint
```python
# Test the API
from gradio_client import Client

client = Client("YOUR_USERNAME/rfb-esrgan-agricultural-sr")
result = client.predict("test_image.jpg", api_name="/predict")
```

### 9. Update Your Project
Add the Space URL to your `geo-agri-analyst` backend configuration.

---

## 🔧 Quick Commands Reference

```bash
# Install Git LFS (one-time)
git lfs install

# Track large files
git lfs track "*.pth"

# Check what's being tracked
git lfs ls-files

# Test app locally
python app.py

# Check file size
ls -lh generator_ensemble.pth
# Should be ~68MB
```

---

## ⚠️ Common Issues

**Issue**: Model file too large for Git  
**Solution**: Use Git LFS (see Option B above)

**Issue**: Build fails with dependency errors  
**Solution**: Check requirements.txt versions match your local environment

**Issue**: Model doesn't load in deployed Space  
**Solution**: Ensure generator_ensemble.pth is in the root directory

**Issue**: Out of memory during inference  
**Solution**: Upgrade to GPU hardware tier in Space settings

---

## 📝 What Each File Does

| File | Purpose |
|------|---------|
| `app.py` | Gradio interface + model code |
| `generator_ensemble.pth` | Trained model weights (~68MB) |
| `requirements.txt` | Python dependencies |
| `README.md` | Model documentation & card |
| `DEPLOYMENT.md` | This checklist |

---

## ✅ Final Verification

Before marking as complete:
- [ ] Model loads without errors
- [ ] Interface displays correctly
- [ ] Sample images produce good results
- [ ] API endpoint works
- [ ] Documentation is clear
- [ ] Space is public (or private, as needed)

---

**Next Step**: Integrate the API into your Geo-Agri-Analyst backend!
