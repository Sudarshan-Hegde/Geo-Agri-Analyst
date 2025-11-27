# 🚀 Quick Deployment Reference

## One-Line Deployment

```bash
cd /home/sudarshanhegde/Sudarshan_Hegde/majorProject/new-classifier-deployment && ./deploy.sh
```

## Manual Deployment Steps

### 1. Install Dependencies (if needed)

```bash
# Install Hugging Face CLI
pip install -U "huggingface_hub[cli]"

# Install Git LFS
sudo apt-get update && sudo apt-get install -y git-lfs
git lfs install

# Login to Hugging Face
huggingface-cli login
```

### 2. Deploy

```bash
cd /home/sudarshanhegde/Sudarshan_Hegde/majorProject/new-classifier-deployment
./deploy.sh
```

### 3. Monitor

Visit: https://huggingface.co/spaces/HegdeSudarshan/Classifier

## Pre-Deployment Checklist

- ✅ Classifier weights: 131.61 MB at `/home/sudarshanhegde/Sudarshan_Hegde/majorProject/classifierHuggingFace/best_classifier.pth`
- ✅ All deployment files in `/new-classifier-deployment/`
- ⚠️ Need to install: Git LFS, Hugging Face CLI (script will prompt)

## Key Fixes from Previous Deployment

1. **DataParallel Handling** ✅
   - Auto-removes 'module.' prefix from state dict
   - Works with both wrapped and unwrapped models

2. **Image Sizes** ✅
   - Input: 32×32 (LR)
   - SR: 256×256 (8× upscaling)
   - ResNet: 224×224 (resized)

3. **Model Architecture** ✅
   - Exact match with training notebook
   - RFB-ESRGAN: 12 RRDB + 6 RRFDB blocks
   - ResNet50 with enhanced head

4. **Dependencies** ✅
   - torch==2.1.0, torchvision==0.16.0
   - gradio==4.44.0 (stable release)

5. **Error Handling** ✅
   - Graceful fallback if SR weights missing
   - Proper exception handling in inference
   - Detailed error messages

## Model Specifications

- **Architecture**: RFB-ESRGAN + ResNet50
- **Classes**: 19 BigEarthNet-S2 categories
- **Training**: 100k samples, 50 epochs
- **Performance**: 100% validation accuracy
- **Size**: 131.61 MB (classifier only)

## After Deployment

1. **Wait** 5-10 minutes for build
2. **Test** with satellite images
3. **Monitor** logs in Space settings
4. **Share** the space URL!

## Troubleshooting

### Build fails
- Check Space logs in settings
- Verify .pth files uploaded via LFS
- Confirm requirements.txt versions

### Model won't load
- Check file size in Space (should be ~132 MB)
- Verify architecture matches training
- Try downloading and re-uploading

### No predictions
- Check input image format (RGB)
- Verify tensor shapes in logs
- Test locally first: `cd new-classifier-deployment && python app.py`

## Support

- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **GitHub Issues**: https://github.com/Sudarshan-Hegde/Geo-Agri-Analyst/issues
- **HF Discussions**: https://huggingface.co/spaces/HegdeSudarshan/Classifier/discussions

---

**Ready to deploy?** Run: `./deploy.sh`
