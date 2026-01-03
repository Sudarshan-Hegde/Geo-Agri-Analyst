---
title: BestClassifier
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🌍 BestClassifier - Land Cover Classification

State-of-the-art land cover classification using Super Resolution enhanced ResNet50.

## Model Overview

**BestClassifier** combines super resolution (SR) enhancement with deep learning classification to achieve high-accuracy land cover prediction from satellite imagery.

### Architecture

- **SR Module**: RFB-ESRGAN with 12 RRDB + 6 RRFDB blocks (8× upscaling)
- **Classifier**: ResNet50 backbone pretrained on ImageNet
- **Enhancement Pipeline**: 30×30 → 256×256 (SR) → 224×224 (classification)

### Dataset

Trained on [BigEarthNet-S2](http://bigearth.net/) dataset:
- 19 CORINE Land Cover classes
- Multi-spectral Sentinel-2 satellite imagery
- European land cover across 10 countries

## Performance

<!-- TODO: Add your actual metrics -->
- **Accuracy**: XX%
- **Precision**: XX%
- **Recall**: XX%
- **F1 Score**: XX%

## Usage

### Web Interface

Simply upload a satellite image to get instant predictions with confidence scores.

### API Access

```python
from gradio_client import Client

client = Client("HegdeSudarshan/bestClassifier")
result = client.predict(
    "path/to/satellite_image.png",
    api_name="/predict"
)
print(result)
```

### cURL

```bash
curl -X POST https://hegdesudarshan-bestclassifier.hf.space/api/predict \
  -F "data=@satellite_image.png"
```

## Input Specifications

- **Format**: RGB images (JPG, PNG)
- **Size**: Any size (automatically resized to 30×30)
- **Optimal**: 30×30 to 256×256 pixels
- **Channels**: 3 (RGB)

## Output Format

Returns top 5 land cover predictions with confidence scores:

```json
{
  "Urban fabric": 0.856,
  "Industrial units": 0.089,
  "Pastures": 0.032,
  "Mixed forest": 0.015,
  "Arable land": 0.008
}
```

## Land Cover Classes

The model predicts 19 CORINE Land Cover classes:

1. Urban fabric
2. Industrial/Commercial units
3. Arable land (non-irrigated)
4. Permanent crops (vineyards, fruit trees)
5. Pastures
6. Complex cultivation patterns
7. Land principally occupied by agriculture
8. Agro-forestry areas
9. Broad-leaved forest
10. Coniferous forest
11. Mixed forest
12. Natural grassland
13. Moors and heathland
14. Transitional woodland/shrub
15. Beaches, dunes, and sands
16. Inland wetlands
17. Coastal wetlands
18. Inland waters
19. Marine waters

## Training Details

<!-- TODO: Fill in your actual training details -->
- **Framework**: PyTorch 2.1.0
- **Training Samples**: XX,XXX
- **Epochs**: XX
- **Batch Size**: XX
- **Optimizer**: Adam with EMA
- **Loss Function**: Cross-Entropy
- **Augmentation**: Random flip, rotation, color jitter

## Model Files

- `bestClassifier.pth` - Complete model weights (XXX MB)
- `label_indices.json` - Class name mappings

## Integration Examples

### Python Backend (FastAPI)

```python
from fastapi import FastAPI, UploadFile
from gradio_client import Client

app = FastAPI()
client = Client("HegdeSudarshan/bestClassifier")

@app.post("/classify")
async def classify_image(file: UploadFile):
    # Save uploaded file
    with open("temp.png", "wb") as f:
        f.write(await file.read())
    
    # Get prediction
    result = client.predict("temp.png", api_name="/predict")
    return result
```

### JavaScript/Node.js

```javascript
const fetch = require('node-fetch');
const FormData = require('form-data');
const fs = require('fs');

async function classifyImage(imagePath) {
  const form = new FormData();
  form.append('data', fs.createReadStream(imagePath));
  
  const response = await fetch(
    'https://hegdesudarshan-bestclassifier.hf.space/api/predict',
    { method: 'POST', body: form }
  );
  
  return await response.json();
}
```

## Limitations

- Best performance on European land cover types (training data bias)
- May struggle with rare or underrepresented classes
- Requires clear satellite imagery (cloud-free)
- Input images should be properly georeferenced for best results

## Citation

<!-- TODO: Add your citation -->
```bibtex
@misc{bestclassifier2026,
  title={BestClassifier: SR-Enhanced Land Cover Classification},
  author={Your Name},
  year={2026},
  publisher={HuggingFace},
  url={https://huggingface.co/spaces/HegdeSudarshan/bestClassifier}
}
```

## References

- [BigEarthNet Dataset](http://bigearth.net/)
- [CORINE Land Cover](https://land.copernicus.eu/pan-european/corine-land-cover)
- [Sentinel-2 Satellite](https://sentinel.esa.int/web/sentinel/missions/sentinel-2)

## License

MIT License - See LICENSE file for details

## Contact

- **GitHub**: [Add your GitHub]
- **Email**: [Add your email]
- **Project**: [Add project link]

## Acknowledgments

- BigEarthNet dataset creators
- European Space Agency (Sentinel-2 data)
- HuggingFace Spaces for hosting

---

**Last Updated**: January 2026  
**Version**: 1.0.0  
**Status**: Production Ready
