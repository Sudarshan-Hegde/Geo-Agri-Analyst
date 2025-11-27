# Deployment Guide for Hugging Face Space

## Prerequisites

1. Hugging Face account
2. Git and Git LFS installed
3. Model weights files:
   - `best_classifier.pth` (from `/home/sudarshanhegde/Sudarshan_Hegde/majorProject/classifierHuggingFace/`)
   - `sr_generator_weights.pth` (SR model weights)

## Step-by-Step Deployment

### 1. Clone the Hugging Face Space Repository

```bash
# Login to Hugging Face (first time only)
huggingface-cli login

# Clone your space
git clone https://huggingface.co/spaces/HegdeSudarshan/Classifier
cd Classifier
```

### 2. Copy Deployment Files

```bash
# Copy all files from new-classifier-deployment directory
cp /home/sudarshanhegde/Sudarshan_Hegde/majorProject/new-classifier-deployment/* .

# Copy the classifier weights
cp /home/sudarshanhegde/Sudarshan_Hegde/majorProject/classifierHuggingFace/best_classifier.pth .

# Copy SR model weights (if you have them)
# cp /path/to/sr_generator_weights.pth .
```

### 3. Setup Git LFS (for large files)

```bash
# Initialize Git LFS
git lfs install

# Track large model files
git lfs track "*.pth"
git lfs track "*.bin"
git lfs track "*.onnx"

# Add .gitattributes
git add .gitattributes
```

### 4. Commit and Push

```bash
# Add all files
git add .

# Commit
git commit -m "Initial deployment: SR-ResNet50 classifier with comprehensive training

- RFB-ESRGAN (12 RRDB + 6 RRFDB blocks, 8x SR)
- ResNet50 classifier with enhanced head
- Trained on 100k samples, 50 epochs
- EMA optimization (0.9995 decay)
- 19 BigEarthNet-S2 classes"

# Push to Hugging Face
git push
```

### 5. Verify Deployment

1. Visit: https://huggingface.co/spaces/HegdeSudarshan/Classifier
2. Wait for build to complete (~5-10 minutes)
3. Test with sample images

## Important Notes

### Avoiding Previous Errors

1. **DataParallel Issue**: ✅ Fixed
   - App automatically removes 'module.' prefix from state dict keys
   - Handles both wrapped and unwrapped models

2. **Image Size**: ✅ Fixed
   - Input: 32×32 (LR for 8× SR)
   - SR output: 256×256
   - ResNet input: 224×224 (resized from SR)

3. **Model Architecture Match**: ✅ Fixed
   - Exact architecture from training notebook
   - Proper RRDB, RRFDB, RFB block structures
   - Correct forward pass logic

4. **Device Handling**: ✅ Fixed
   - Auto-detects CUDA/CPU
   - Proper .to(device) calls
   - map_location in torch.load()

5. **Dependencies**: ✅ Fixed
   - Specific versions matching training
   - torch==2.1.0, torchvision==0.16.0
   - gradio==4.44.0 (stable)

### If SR Weights Missing

If you don't have `sr_generator_weights.pth`, the app will use random initialization with a warning. To get proper SR weights:

```python
# In your training notebook, save SR model:
torch.save(classifier.module.sr_model.state_dict(), 'sr_generator_weights.pth')

# Or if not wrapped in DataParallel:
torch.save(classifier.sr_model.state_dict(), 'sr_generator_weights.pth')
```

### File Size Limits

- Hugging Face Space: 50GB limit
- Git LFS: 10GB per file recommended
- If classifier.pth > 10GB, consider model compression

### Testing Locally First

```bash
cd new-classifier-deployment
python app.py
```

Then visit: http://localhost:7860

## Troubleshooting

### Build Fails

1. Check requirements.txt versions
2. Verify all .pth files are tracked by LFS
3. Check logs in Space settings

### Model Load Error

1. Verify .pth files uploaded correctly
2. Check file sizes (should be hundreds of MB)
3. Verify architecture matches training code

### Runtime Error

1. Check GPU memory (Space has 16GB)
2. Verify input/output tensor shapes
3. Check normalization ranges

### No Predictions

1. Verify model in eval mode
2. Check softmax/sigmoid output
3. Verify class names mapping

## Advanced: Model Optimization

### Quantization (Optional)

```python
# For faster inference
import torch.quantization as quantization

classifier_quantized = quantization.quantize_dynamic(
    classifier, {nn.Linear}, dtype=torch.qint8
)
torch.save(classifier_quantized.state_dict(), 'classifier_quantized.pth')
```

### ONNX Export (Optional)

```python
# For cross-platform deployment
dummy_input = torch.randn(1, 3, 32, 32)
torch.onnx.export(
    classifier, 
    dummy_input, 
    "classifier.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)
```

## Monitoring

- **Space Metrics**: Check in Space settings
- **Gradio Analytics**: Enable in app.py
- **Error Logs**: View in Space runtime logs

## Updating the Model

```bash
# Update model weights
cp /path/to/new_best_classifier.pth best_classifier.pth

# Commit and push
git add best_classifier.pth
git commit -m "Update: Improved classifier weights"
git push
```

## Support

- GitHub Issues: [Geo-Agri-Analyst Issues](https://github.com/Sudarshan-Hegde/Geo-Agri-Analyst/issues)
- Hugging Face Community: [Discussions](https://huggingface.co/spaces/HegdeSudarshan/Classifier/discussions)
