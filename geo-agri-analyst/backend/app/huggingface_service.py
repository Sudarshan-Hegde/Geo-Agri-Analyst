"""
HuggingFace Model Integration Service
Handles communication with the deployed HuggingFace Space for SR + Classification
"""

from gradio_client import Client, handle_file
import base64
import io
from PIL import Image
import numpy as np
from typing import Dict, Optional, List, Tuple
import asyncio
import os
import tempfile
from app.satellite_service import get_satellite_service


class HuggingFaceModelService:
    """
    Service for calling the HuggingFace Spaces API using Gradio Client
    """
    
    def __init__(self, space_url: str = None, hf_token: str = None):
        """
        Initialize the HuggingFace service
        
        Args:
            space_url: HuggingFace Space identifier
                      Format: USERNAME/SPACENAME
            hf_token: HuggingFace API token (required for private Spaces)
        """
        # Update this with your actual HuggingFace Space name
        self.space_url = space_url or "HegdeSudarshan/Classifier"
        # For private Spaces, set your HF token here or via environment variable
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.client = None
        self.timeout = 90.0  # 90 seconds timeout for cold starts
        
        # Land cover class names (43 classes from BigEarthNet)
        self.class_names = [
            "Continuous urban fabric",
            "Discontinuous urban fabric",
            "Industrial or commercial units",
            "Road and rail networks and associated land",
            "Port areas",
            "Airports",
            "Mineral extraction sites",
            "Dump sites",
            "Construction sites",
            "Green urban areas",
            "Sport and leisure facilities",
            "Non-irrigated arable land",
            "Permanently irrigated land",
            "Rice fields",
            "Vineyards",
            "Fruit trees and berry plantations",
            "Olive groves",
            "Pastures",
            "Annual crops associated with permanent crops",
            "Complex cultivation patterns",
            "Land principally occupied by agriculture",
            "Agro-forestry areas",
            "Broad-leaved forest",
            "Coniferous forest",
            "Mixed forest",
            "Natural grassland",
            "Moors and heathland",
            "Sclerophyllous vegetation",
            "Transitional woodland/shrub",
            "Beaches, dunes, sands",
            "Bare rock",
            "Sparsely vegetated areas",
            "Burnt areas",
            "Inland marshes",
            "Peatbogs",
            "Salt marshes",
            "Salines",
            "Intertidal flats",
            "Water courses",
            "Water bodies",
            "Coastal lagoons",
            "Estuaries",
            "Sea and ocean"
        ]
        
        print(f"✅ HuggingFace Service initialized")
        print(f"📡 Space URL: {self.space_url}")
        if self.hf_token:
            print(f"🔐 Using HuggingFace token for authentication")
    
    def _get_client(self) -> Client:
        """Get or create Gradio client"""
        if self.client is None:
            try:
                # Connect to public Space URL
                print(f"🔄 Attempting to connect to: {self.space_url}")
                self.client = Client(self.space_url, hf_token=self.hf_token)
                print(f"✅ Connected to HuggingFace Space: {self.space_url}")
            except Exception as e:
                print(f"⚠️ Could not connect to Space: {type(e).__name__}: {str(e)}")
                import traceback
                traceback.print_exc()
                self.client = None
        return self.client
    
    async def check_health(self) -> bool:
        """
        Check if HuggingFace Space is running and accessible
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Try to get client in executor to avoid blocking
            loop = asyncio.get_event_loop()
            client = await loop.run_in_executor(None, self._get_client)
            if client is not None:
                print(f"✅ HuggingFace Space is healthy")
                return True
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def _create_fake_satellite_image(self, lat: float, lng: float) -> Image.Image:
        """
        Create a fake satellite image based on coordinates (for testing)
        
        Args:
            lat: Latitude
            lng: Longitude
        
        Returns:
            PIL Image (30x30 RGB)
        """
        np.random.seed(int((lat + lng) * 1000) % 10000)
        
        # Create realistic-looking satellite image
        image = np.random.rand(30, 30, 3) * 0.3
        
        # Add vegetation patterns
        for i in range(0, 30, 4):
            for j in range(0, 30, 4):
                if np.random.rand() > 0.6:
                    # Green areas
                    image[i:i+4, j:j+4, 1] += 0.4
                elif np.random.rand() > 0.3:
                    # Brown areas (soil)
                    image[i:i+4, j:j+4, 0] += 0.3
                    image[i:i+4, j:j+4, 1] += 0.2
        
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(image)
    
    async def predict(
        self, 
        lat: float, 
        lng: float,
        image: Optional[Image.Image] = None
    ) -> Dict:
        """
        Get land classification prediction from HuggingFace model
        
        Args:
            lat: Latitude
            lng: Longitude
            image: Optional PIL Image (if None, will fetch real satellite image)
        
        Returns:
            Dict containing:
                - land_class: Predicted land cover class
                - confidence: Confidence score (0-1)
                - before_image_b64: Original/LR image as base64
                - after_image_b64: Enhanced/SR image as base64
                - predictions: Top 5 predictions with scores
        """
        try:
            # Fetch real satellite image if not provided
            if image is None:
                print(f"🛰️ Fetching real satellite image for lat={lat}, lng={lng}")
                satellite_svc = get_satellite_service()
                image = satellite_svc.get_satellite_image(lat, lng, size=30, zoom=17)
                
                # Fallback to fake image if satellite fetch fails
                if image is None:
                    print("⚠️ Satellite image fetch failed, using fallback")
                    image = self._create_fake_satellite_image(lat, lng)
                else:
                    print("✅ Successfully fetched real satellite image")
            
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Ensure image is exactly 30x30
            if image.size != (30, 30):
                print(f"⚠️ Resizing image from {image.size} to 30x30")
                image = image.resize((30, 30), Image.Resampling.LANCZOS)
            
            # Save to temporary file for Gradio client
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                image.save(tmp, format='PNG')
                temp_path = tmp.name
            
            # Call HuggingFace Space using Gradio client
            print(f"📡 Calling HuggingFace Space: {self.space_url}")
            print(f"⏱️ Timeout set to {self.timeout} seconds")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def call_predict():
                client = self._get_client()
                if client is None:
                    raise Exception("Could not connect to HuggingFace Space")
                # Use handle_file to properly format the image for Gradio
                result = client.predict(handle_file(temp_path), api_name="/predict")
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                return result
            
            result = await loop.run_in_executor(None, call_predict)
            
            print(f"✅ Received response from HuggingFace")
            
            # Parse response from Gradio
            # Result format: (enhanced_image_path, {"label": ..., "confidences": [...]})
            if result and len(result) >= 2:
                enhanced_image_path = result[0]  # Path to enhanced image
                predictions_data = result[1]  # Dict with label and confidences
                
                # Read enhanced image
                with open(enhanced_image_path, 'rb') as f:
                    enhanced_b64 = base64.b64encode(f.read()).decode('utf-8')
                
                # Convert original image to base64
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_b64 = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                
                # Extract predictions
                if isinstance(predictions_data, dict):
                    top_label = predictions_data.get('label', 'Unknown')
                    confidences = predictions_data.get('confidences', [])
                    
                    # Build predictions dict from confidences list
                    predictions_dict = {}
                    for item in confidences[:5]:  # Top 5
                        label = item.get('label', '')
                        conf = item.get('confidence', 0.0)
                        predictions_dict[label] = conf
                    
                    # Get top prediction
                    if predictions_dict:
                        top_class = list(predictions_dict.keys())[0]
                        confidence = predictions_dict[top_class]
                    else:
                        top_class = top_label
                        confidence = 0.0
                    
                    return {
                        "land_class": top_class,
                        "confidence": confidence,
                        "before_image_b64": img_b64,
                        "after_image_b64": enhanced_b64,
                        "predictions": predictions_dict,
                        "source": "huggingface"
                    }
                else:
                    raise Exception(f"Unexpected prediction format: {predictions_data}")
            else:
                raise Exception(f"Unexpected API response format: {result}")
                
        except Exception as e:
            print(f"❌ Error calling HuggingFace API: {e}")
            print(f"📋 Full error details: {type(e).__name__}")
            print("💡 Try again - the first request may wake up a sleeping Space")
            return self._get_fallback_prediction(lat, lng, image)
    
    async def predict_batch(
        self,
        coordinates: List[Tuple[float, float]],
        zoom: int = 17,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Get land classification predictions for multiple coordinate points
        
        Args:
            coordinates: List of (lat, lng) tuples
            zoom: Zoom level for satellite images
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of prediction dicts, one for each coordinate
        """
        predictions = []
        total = len(coordinates)
        
        print(f"🔄 Processing batch of {total} locations...")
        
        for idx, (lat, lng) in enumerate(coordinates):
            try:
                # Fetch satellite image with specified zoom
                satellite_svc = get_satellite_service()
                image = satellite_svc.get_satellite_image(lat, lng, size=30, zoom=zoom)
                
                # Get prediction for this location
                pred = await self.predict(lat, lng, image=image)
                pred['coordinates'] = {'lat': lat, 'lng': lng}
                predictions.append(pred)
                
                # Progress update
                if progress_callback:
                    progress_callback(idx + 1, total)
                
                if (idx + 1) % 5 == 0 or idx == total - 1:
                    print(f"   Processed {idx + 1}/{total} locations")
                    
            except Exception as e:
                print(f"⚠️ Error processing location ({lat}, {lng}): {e}")
                # Add fallback prediction for failed location
                predictions.append(self._get_fallback_prediction(lat, lng))
                predictions[-1]['coordinates'] = {'lat': lat, 'lng': lng}
        
        print(f"✅ Batch processing complete: {len(predictions)}/{total} predictions")
        return predictions
    
    def _get_fallback_prediction(
        self, 
        lat: float, 
        lng: float,
        image: Optional[Image.Image] = None
    ) -> Dict:
        """
        Return fallback prediction when HuggingFace API is unavailable
        
        Args:
            lat: Latitude
            lng: Longitude
            image: Optional PIL Image
        
        Returns:
            Dict with placeholder predictions
        """
        # Create or use provided image
        if image is None:
            image = self._create_fake_satellite_image(lat, lng)
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Simple hash-based class selection
        class_idx = hash(f"{lat}{lng}") % len(self.class_names)
        selected_class = self.class_names[class_idx]
        confidence = 0.75 + (hash(f"{lat}{lng}") % 20) / 100
        
        return {
            "land_class": selected_class,
            "confidence": confidence,
            "before_image_b64": img_b64,
            "after_image_b64": img_b64,  # Same image (no SR)
            "predictions": {
                selected_class: confidence,
                self.class_names[(class_idx + 1) % len(self.class_names)]: confidence - 0.1,
                self.class_names[(class_idx + 2) % len(self.class_names)]: confidence - 0.2,
            },
            "source": "fallback",
            "note": "HuggingFace API unavailable - using fallback predictions"
        }


# Global service instance
hf_service = None

def get_hf_service() -> HuggingFaceModelService:
    """
    Get or create the global HuggingFace service instance
    
    Returns:
        HuggingFaceModelService instance
    """
    global hf_service
    if hf_service is None:
        hf_service = HuggingFaceModelService()
    return hf_service
