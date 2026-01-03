# Integration Guide: Adding SR to Geo-Agri-Analyst

## 🔌 Backend Integration

### Step 1: Copy SR Service

Copy `sr_service.py` to your backend:

```bash
cp sr-model-deployment/sr_service.py geo-agri-analyst/backend/app/sr_service.py
```

### Step 2: Update main.py

Add to `geo-agri-analyst/backend/app/main.py`:

```python
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from .sr_service import get_sr_service
import tempfile
import os

app = FastAPI()

# ... your existing code ...

@app.post("/api/upscale-satellite")
async def upscale_satellite_image(file: UploadFile = File(...)):
    """
    Endpoint to upscale a satellite image using RFB-ESRGAN
    
    Upload a low-resolution image and get back an 8x upscaled version
    """
    # Get SR service
    sr_service = get_sr_service()
    
    if not sr_service.is_available():
        return {"error": "Super-resolution service unavailable"}
    
    try:
        # Read uploaded file
        image_bytes = await file.read()
        
        # Upscale
        sr_bytes = sr_service.upscale_from_bytes(image_bytes)
        
        if sr_bytes is None:
            return {"error": "Upscaling failed"}
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(sr_bytes)
            temp_path = temp_file.name
        
        # Return as file response
        return FileResponse(
            temp_path,
            media_type="image/png",
            filename=f"sr_{file.filename}"
        )
        
    except Exception as e:
        return {"error": f"Error processing image: {str(e)}"}


@app.get("/api/sr-status")
async def sr_service_status():
    """Check if SR service is available"""
    sr_service = get_sr_service()
    return {
        "available": sr_service.is_available(),
        "model_url": sr_service.hf_space_url
    }
```

### Step 3: Update requirements.txt

Add to `geo-agri-analyst/backend/requirements.txt`:

```txt
gradio-client>=0.7.0
```

### Step 4: Set Environment Variable

In `geo-agri-analyst/backend/.env`:

```env
SR_MODEL_URL=YOUR_USERNAME/rfb-esrgan-agricultural-sr
```

---

## 🎨 Frontend Integration

### Option 1: Add to MapComponent

In `geo-agri-analyst/frontend/src/components/MapComponent.jsx`:

```jsx
import { useState } from 'react';

function MapComponent() {
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [upscaledImage, setUpscaledImage] = useState(null);
  
  const handleUpscale = async (imageFile) => {
    setIsUpscaling(true);
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await fetch('http://localhost:8000/api/upscale-satellite', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const blob = await response.blob();
        const imageUrl = URL.createObjectURL(blob);
        setUpscaledImage(imageUrl);
      } else {
        console.error('Upscaling failed');
      }
    } catch (error) {
      console.error('Error upscaling image:', error);
    } finally {
      setIsUpscaling(false);
    }
  };
  
  return (
    <div>
      {/* Your existing map component */}
      
      <div className="sr-controls">
        <button 
          onClick={() => handleUpscale(selectedSatelliteImage)}
          disabled={isUpscaling}
          className="btn-primary"
        >
          {isUpscaling ? 'Enhancing...' : '🔍 Enhance Image Quality (8x)'}
        </button>
        
        {upscaledImage && (
          <div className="upscaled-preview">
            <img src={upscaledImage} alt="Upscaled" />
            <a href={upscaledImage} download="enhanced_satellite.png">
              Download Enhanced Image
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
```

### Option 2: Create Dedicated SR Panel

Create `geo-agri-analyst/frontend/src/components/SuperResolutionPanel.jsx`:

```jsx
import React, { useState } from 'react';

export default function SuperResolutionPanel() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [upscaled, setUpscaled] = useState(null);
  const [loading, setLoading] = useState(false);
  
  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setUpscaled(null);
    }
  };
  
  const handleUpscale = async () => {
    if (!selectedFile) return;
    
    setLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);
    
    try {
      const response = await fetch('http://localhost:8000/api/upscale-satellite', {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        const blob = await response.blob();
        setUpscaled(URL.createObjectURL(blob));
      }
    } catch (error) {
      console.error('Upscaling error:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="sr-panel">
      <h2>🌾 Image Super-Resolution</h2>
      <p>Enhance satellite image quality with AI (8x upscaling)</p>
      
      <div className="upload-section">
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileSelect}
        />
        
        <button 
          onClick={handleUpscale}
          disabled={!selectedFile || loading}
        >
          {loading ? 'Processing...' : 'Upscale Image'}
        </button>
      </div>
      
      <div className="comparison-view">
        {preview && (
          <div className="image-box">
            <h3>Original</h3>
            <img src={preview} alt="Original" />
          </div>
        )}
        
        {upscaled && (
          <div className="image-box">
            <h3>Enhanced (8x)</h3>
            <img src={upscaled} alt="Upscaled" />
            <a href={upscaled} download="enhanced.png">
              Download
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
```

Add to your main App:

```jsx
import SuperResolutionPanel from './components/SuperResolutionPanel';

// In your app routing or tabs
<SuperResolutionPanel />
```

---

## 🎯 Use Cases in Geo-Agri-Analyst

### 1. Enhance Satellite Imagery Before Analysis
```javascript
// Before running crop detection
const enhancedImage = await upscaleSatelliteImage(rawSatelliteImage);
const cropAnalysis = await analyzeCrops(enhancedImage);
```

### 2. Improve Polygon Detail
```javascript
// When user draws polygon, enhance that region
const polygonImage = extractPolygonRegion(map, polygon);
const enhancedPolygon = await upscaleImage(polygonImage);
displayEnhancedView(enhancedPolygon);
```

### 3. Historical Image Enhancement
```javascript
// Enhance low-res historical satellite data
const historicalImages = await fetchHistoricalData(coordinates);
const enhancedHistory = await Promise.all(
  historicalImages.map(img => upscaleImage(img))
);
showTimeSeries(enhancedHistory);
```

---

## 🧪 Testing the Integration

### Backend Test

```bash
cd geo-agri-analyst/backend

# Start the server
uvicorn app.main:app --reload

# In another terminal, test the endpoint
curl -X POST "http://localhost:8000/api/upscale-satellite" \
  -F "file=@test_image.jpg" \
  --output enhanced.png

# Check status
curl "http://localhost:8000/api/sr-status"
```

### Frontend Test

```bash
cd geo-agri-analyst/frontend

# Start dev server
npm run dev

# Open http://localhost:5173
# Test file upload and upscaling
```

---

## 📊 Performance Considerations

### Latency
- **Local model**: ~0.5s per image (GPU)
- **HF API**: ~2-5s per image (includes network)
- **Recommendation**: Show loading indicator

### Caching
Consider caching upscaled images:

```python
# In sr_service.py
from functools import lru_cache
import hashlib

class SuperResolutionService:
    def __init__(self):
        self.cache_dir = "cache/sr_images"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def upscale_with_cache(self, image_path):
        # Generate cache key
        with open(image_path, 'rb') as f:
            image_hash = hashlib.md5(f.read()).hexdigest()
        
        cache_path = f"{self.cache_dir}/{image_hash}_sr.png"
        
        # Check cache
        if os.path.exists(cache_path):
            return cache_path
        
        # Upscale and cache
        result = self.upscale_image(image_path, cache_path)
        return result
```

---

## ✅ Integration Checklist

- [ ] Copy `sr_service.py` to backend
- [ ] Add endpoint to `main.py`
- [ ] Update backend `requirements.txt`
- [ ] Set `SR_MODEL_URL` environment variable
- [ ] Test backend endpoint
- [ ] Add frontend component/button
- [ ] Test file upload flow
- [ ] Add loading states
- [ ] Handle errors gracefully
- [ ] Test with real satellite images
- [ ] (Optional) Add caching
- [ ] (Optional) Add batch processing
- [ ] Document for team

---

## 🚀 Next Steps

1. Deploy SR model to Hugging Face (see DEPLOYMENT.md)
2. Get Space URL
3. Update environment variable
4. Test integration locally
5. Deploy to production

---

**Ready to enhance your agricultural imagery! 🌾**
