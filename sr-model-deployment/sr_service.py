"""
Super-Resolution Service for Geo-Agri-Analyst Backend
Integrates RFB-ESRGAN model from Hugging Face
"""

from gradio_client import Client
from PIL import Image
import os
import tempfile
from typing import Optional

class SuperResolutionService:
    """Service to upscale satellite images using RFB-ESRGAN from Hugging Face"""
    
    def __init__(self, hf_space_url: str = None):
        """
        Initialize the SR service
        
        Args:
            hf_space_url: Hugging Face Space URL 
                         (e.g., "username/rfb-esrgan-agricultural-sr")
        """
        self.hf_space_url = hf_space_url or os.getenv(
            "SR_MODEL_URL", 
            "YOUR_USERNAME/rfb-esrgan-agricultural-sr"  # Update this!
        )
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Gradio client connection to HF Space"""
        try:
            self.client = Client(self.hf_space_url)
            print(f"✓ Connected to SR model: {self.hf_space_url}")
        except Exception as e:
            print(f"⚠️ Failed to connect to SR model: {e}")
            self.client = None
    
    def upscale_image(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Upscale a low-resolution image 8x (32×32 → 256×256)
        
        Args:
            image_path: Path to input low-resolution image
            output_path: Path to save upscaled image (optional)
            
        Returns:
            Path to upscaled image, or None if failed
        """
        if not self.client:
            print("⚠️ SR service not initialized")
            return None
        
        try:
            # Call the HF Space API
            result = self.client.predict(
                image_path,
                api_name="/predict"
            )
            
            # Result is tuple: (bicubic_baseline, sr_output)
            _, sr_image_path = result
            
            # Save to output path if specified
            if output_path:
                sr_image = Image.open(sr_image_path)
                sr_image.save(output_path)
                return output_path
            
            return sr_image_path
            
        except Exception as e:
            print(f"⚠️ SR upscaling failed: {e}")
            return None
    
    def upscale_from_bytes(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Upscale image from bytes (useful for API endpoints)
        
        Args:
            image_bytes: Input image as bytes
            
        Returns:
            Upscaled image as bytes, or None if failed
        """
        if not self.client:
            return None
        
        try:
            # Save bytes to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_input:
                temp_input.write(image_bytes)
                temp_input_path = temp_input.name
            
            # Upscale
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_output:
                temp_output_path = temp_output.name
            
            result_path = self.upscale_image(temp_input_path, temp_output_path)
            
            if result_path:
                # Read result as bytes
                with open(result_path, 'rb') as f:
                    result_bytes = f.read()
                
                # Cleanup
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
                
                return result_bytes
            
            # Cleanup on failure
            os.unlink(temp_input_path)
            return None
            
        except Exception as e:
            print(f"⚠️ SR upscaling from bytes failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if SR service is available"""
        return self.client is not None


# Singleton instance
_sr_service = None

def get_sr_service() -> SuperResolutionService:
    """Get or create SR service singleton"""
    global _sr_service
    if _sr_service is None:
        _sr_service = SuperResolutionService()
    return _sr_service
