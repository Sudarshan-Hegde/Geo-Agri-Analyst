# SR-Model Integration

This document describes the integration of the HuggingFace SR-Model for super resolution image enhancement.

## Overview

The SR-Model service provides high-quality image enhancement for satellite imagery using a dedicated super resolution model hosted on HuggingFace Spaces.

**API Endpoint**: `HegdeSudarshan/SR-Model`  
**Model Function**: `/enhance_image`

## Architecture

### Services

1. **sr_service.py** - Dedicated service for SR-Model API calls
2. **huggingface_service.py** - Updated to use SR-Model for image enhancement
3. **main.py** - Includes SR health check and test endpoint

## API Endpoints

### 1. Test SR Enhancement
```http
POST /api/v1/enhance-image
Content-Type: application/json

{
  "lat": 12.9716,
  "lng": 77.5946
}
```

**Response:**
```json
{
  "status": "success",
  "coordinates": {"lat": 12.9716, "lng": 77.5946},
  "original_image_b64": "iVBORw0KG...",
  "enhanced_image_b64": "iVBORw0KG..."
}
```

### 2. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "weather": "available",
    "huggingface_ml": "available",
    "sr_model": "available"
  }
}
```

### 3. Location Analysis (includes SR)
```http
POST /api/v1/analyze
Content-Type: application/json

{
  "type": "point",
  "lat": 12.9716,
  "lng": 77.5946
}
```

The analysis endpoint automatically uses SR-Model for image enhancement when available.

## Usage Examples

### Python (Direct Service Usage)

```python
from sr_service import get_sr_service
from PIL import Image

# Initialize service
sr_service = get_sr_service()

# Check if available
is_healthy = await sr_service.check_health()

# Enhance an image
image = Image.open('satellite.png')
enhanced = await sr_service.enhance_image(image)

# Get as base64
enhanced_b64 = await sr_service.enhance_image_to_base64(image)
```

### Python (Gradio Client)

```python
from gradio_client import Client

client = Client("HegdeSudarshan/SR-Model")
result = client.predict(
    "path/to/image.png",
    api_name="/enhance_image"
)
# result is the path to the enhanced image
```

### JavaScript/Frontend

```javascript
const response = await fetch('http://localhost:8000/api/v1/enhance-image', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    lat: 12.9716,
    lng: 77.5946
  })
});

const data = await response.json();
// data.enhanced_image_b64 contains the enhanced image
```

## Features

- ✅ Automatic image enhancement for all analysis requests
- ✅ Fallback to original/classifier image if SR fails
- ✅ Dedicated test endpoint for SR functionality
- ✅ Health monitoring for SR service
- ✅ Support for PIL Images and file paths
- ✅ Base64 encoding for easy frontend integration

## Testing

### 1. Test SR Service Directly

```bash
cd geo-agri-analyst/backend/app
python test_sr_service.py
```

### 2. Test via API

```bash
# Start the backend
python main.py

# In another terminal, test the endpoint
curl -X POST http://localhost:8000/api/v1/enhance-image \
  -H "Content-Type: application/json" \
  -d '{"lat": 12.9716, "lng": 77.5946}'
```

### 3. Check Health

```bash
curl http://localhost:8000/health
```

## Integration Details

### How It Works

1. **Satellite Image Acquisition**: System fetches 30x30 satellite image from Sentinel Hub
2. **Classification**: Image is sent to HuggingFace Classifier for land classification
3. **SR Enhancement**: Original image is enhanced using SR-Model
4. **Fallback**: If SR-Model fails, uses classifier's enhanced image
5. **Response**: Returns both classification results and enhanced image

### Workflow

```
User Request
    ↓
Fetch Satellite Image (30x30)
    ↓
    ├─→ Send to Classifier → Get land class predictions
    │
    └─→ Send to SR-Model → Get enhanced image (120x120)
          ↓
    Combine Results
          ↓
    Return to User
```

## Configuration

### Environment Variables

```bash
# Optional: For private HuggingFace spaces
export HF_TOKEN="your_huggingface_token"
```

### Service Settings

Edit `sr_service.py` to customize:

```python
class SRModelService:
    def __init__(self, space_url: str = None, hf_token: str = None):
        self.space_url = space_url or "HegdeSudarshan/SR-Model"
        self.timeout = 60.0  # Adjust timeout
```

## Error Handling

The SR service includes comprehensive error handling:

- **Connection Errors**: Falls back to original/classifier image
- **Timeout**: Configurable timeout (default 60s)
- **Invalid Images**: Automatic format conversion and validation
- **Service Unavailable**: Returns original image with status message

## Performance

- **Cold Start**: ~10-15 seconds (first request after idle)
- **Warm Requests**: ~2-5 seconds per image
- **Image Size**: Typically 30x30 → 120x120 (4x upscaling)

## Troubleshooting

### SR-Model not available

```bash
# Check health endpoint
curl http://localhost:8000/health

# Check service logs
python test_sr_service.py
```

### Enhancement fails

The system automatically falls back to:
1. Classifier's enhanced image
2. Original satellite image

Check logs for:
- Connection errors
- Timeout issues
- Image format problems

### Dependencies

Ensure these packages are installed:

```bash
pip install gradio_client pillow
```

## Future Enhancements

- [ ] Batch enhancement for polygon analysis
- [ ] Caching for frequently requested locations
- [ ] Support for different upscaling factors
- [ ] Custom SR model parameters
- [ ] Progress tracking for batch operations

## References

- SR-Model Space: https://huggingface.co/spaces/HegdeSudarshan/SR-Model
- Gradio Client Docs: https://www.gradio.app/guides/getting-started-with-the-python-client
- HuggingFace API: https://huggingface.co/docs/api-inference
