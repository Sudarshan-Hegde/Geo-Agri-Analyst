---
title: BigEarthNet SR-ResNet50 Classifier
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# BigEarthNet SR-ResNet50 Classifier

Super-Resolution Enhanced Land Cover Classification using RFB-ESRGAN + ResNet50.

## Features

- 🔥 **8× Super-Resolution**: RFB-ESRGAN (32×32 → 256×256)
- 🎯 **19-Class Classification**: BigEarthNet-S2 land cover categories
- 🧠 **Active Learning**: Trained with DBSS + SSAS strategies
- 📊 **100k Samples**: Trained on 100,000 samples for 50 epochs
- ⚡ **EMA Optimization**: Exponential Moving Average for stable predictions

## Model Architecture

### Super-Resolution Network
- **RFB-ESRGAN Generator**
- 12 RRDB blocks (Residual in Residual Dense Block)
- 6 RRFDB blocks (RFB-enhanced Dense Block)
- Multi-scale feature extraction with dilated convolutions
- 8× upscaling (2×2×2 pixel shuffle)

### Classifier Network
- **ResNet50** backbone (ImageNet pretrained)
- Enhanced classifier head: 2048 → 512 → 19
- Dual dropout layers (0.4, 0.3)
- Label smoothing (0.15) and EMA (0.9995)

## Training Details

- **Dataset**: BigEarthNet-S2 (19 classes)
- **Samples**: 100,000 training samples
- **Epochs**: 50 with warmup (5 epochs) + cosine annealing
- **Batch Size**: 64
- **Learning Rate**: 3e-4 with warmup
- **Regularization**: 
  - Label Smoothing: 0.15
  - Weight Decay: 1e-5
  - EMA Decay: 0.9995
  - Dropout: 0.4, 0.3
- **Augmentation**: Random rotation, flips, color jitter
- **Hardware**: Dual GPU training on Kaggle (12-hour session)

## Performance

- **Validation Accuracy**: 100% (reported)
- **Training Time**: ~8-10 hours for 50 epochs
- **Comprehensive Evaluation**:
  - Confusion matrices
  - ROC curves & AUC analysis
  - Precision-Recall curves
  - Per-class metrics
  - Feature map visualizations
  - Learning dynamics tracking

## 19 BigEarthNet Classes

1. Urban fabric
2. Industrial or commercial units
3. Arable land
4. Permanent crops
5. Pastures
6. Complex cultivation patterns
7. Land principally occupied by agriculture
8. Agro-forestry areas
9. Broad-leaved forest
10. Coniferous forest
11. Mixed forest
12. Natural grassland and sparsely vegetated areas
13. Moors, heathland and sclerophyllous vegetation
14. Transitional woodland/shrub
15. Beaches, dunes, sands
16. Inland wetlands
17. Coastal wetlands
18. Inland waters
19. Marine waters

## Usage

Upload a satellite image (preferably 32×32 or any size - will be resized automatically):
1. Image is resized to 32×32 (LR input)
2. SR model enhances to 256×256
3. ResNet50 classifies into 19 categories
4. Top 5 predictions shown with confidence scores

## Links

- **GitHub**: [Geo-Agri-Analyst](https://github.com/Sudarshan-Hegde/Geo-Agri-Analyst.git)
- **Dataset**: [BigEarthNet](http://bigearth.net/)
- **Training Notebook**: [Kaggle](https://www.kaggle.com/code/hegdesudarshan/majprojsuper-new)

## Citation

```bibtex
@misc{hegde2025sr-resnet50-bigearthnet,
  title={Super-Resolution Enhanced ResNet50 for BigEarthNet Classification},
  author={Hegde, Sudarshan},
  year={2025},
  url={https://huggingface.co/spaces/HegdeSudarshan/Classifier}
}
```

## License

MIT License

## Acknowledgments

- BigEarthNet dataset creators
- PyTorch and Hugging Face teams
- RFB-ESRGAN architecture authors
