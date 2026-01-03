"""
Super Resolution Service
Handles image enhancement using the HuggingFace SR-Model
API: https://huggingface.co/spaces/HegdeSudarshan/SR-Model
"""

from gradio_client import Client, handle_file
import base64
import io
from PIL import Image
import os
import tempfile
import asyncio
from typing import Optional


class SRModelService:
    """
    Service for calling the HuggingFace SR-Model Space API
    """
    
    def __init__(self, space_url: str = None, hf_token: str = None):
        """
        Initialize the SR Model service
        
        Args:
            space_url: HuggingFace Space identifier (default: HegdeSudarshan/SR-Model)
            hf_token: HuggingFace API token (optional, for private spaces)
        """
        self.space_url = space_url or "HegdeSudarshan/SR-Model"
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.client = None
        self.timeout = 60.0  # 60 seconds timeout
        
        print(f"✅ SR Model Service initialized")
        print(f"📡 Space URL: {self.space_url}")
    
    def _get_client(self) -> Optional[Client]:
        """Get or create Gradio client"""
        if self.client is None:
            try:
                print(f"🔄 Connecting to SR-Model Space: {self.space_url}")
                self.client = Client(self.space_url, hf_token=self.hf_token)
                print(f"✅ Connected to SR-Model Space")
            except Exception as e:
                print(f"⚠️ Could not connect to SR-Model Space: {type(e).__name__}: {str(e)}")
                self.client = None
        return self.client
    
    async def enhance_image(
        self,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> Optional[Image.Image]:
        """
        Enhance a low-resolution image using super resolution
        
        Args:
            image: PIL Image to enhance (either this or image_path required)
            image_path: Path to image file (either this or image required)
        
        Returns:
            Enhanced PIL Image, or None if enhancement fails
        """
        temp_input_path = None
        
        try:
            # Prepare input image
            if image_path is None:
                # Save PIL Image to temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(tmp, format='PNG')
                    temp_input_path = tmp.name
                    image_path = temp_input_path
            
            print(f"📡 Calling SR-Model API for image enhancement")
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            def call_enhance():
                client = self._get_client()
                if client is None:
                    raise Exception("Could not connect to SR-Model Space")
                
                # Call the /enhance_image endpoint
                result = client.predict(
                    handle_file(image_path),
                    api_name="/enhance_image"
                )
                return result
            
            # Get enhanced image path
            enhanced_image_path = await loop.run_in_executor(None, call_enhance)
            
            print(f"✅ Received enhanced image from SR-Model")
            
            # Load enhanced image
            if isinstance(enhanced_image_path, str) and os.path.exists(enhanced_image_path):
                enhanced_image = Image.open(enhanced_image_path)
                print(f"📐 Enhanced image size: {enhanced_image.size}")
                return enhanced_image
            else:
                print(f"⚠️ Unexpected response format from SR-Model: {type(enhanced_image_path)}")
                return None
                
        except Exception as e:
            print(f"❌ Error calling SR-Model API: {e}")
            import traceback
            traceback.print_exc()
            return None
        
        finally:
            # Clean up temporary file
            if temp_input_path and os.path.exists(temp_input_path):
                try:
                    os.unlink(temp_input_path)
                except:
                    pass
    
    async def enhance_image_to_base64(
        self,
        image: Image.Image,
        image_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Enhance image and return as base64 string
        
        Args:
            image: PIL Image to enhance
            image_path: Optional path to image file
        
        Returns:
            Base64 encoded string of enhanced image, or None if enhancement fails
        """
        enhanced_image = await self.enhance_image(image, image_path)
        
        if enhanced_image is None:
            return None
        
        try:
            # Convert to base64
            buffer = io.BytesIO()
            enhanced_image.save(buffer, format='PNG')
            img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_b64
        except Exception as e:
            print(f"❌ Error converting enhanced image to base64: {e}")
            return None
    
    async def check_health(self) -> bool:
        """
        Check if SR-Model Space is running and accessible
        
        Returns:
            True if service is healthy, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            client = await loop.run_in_executor(None, self._get_client)
            if client is not None:
                print(f"✅ SR-Model Space is healthy")
                return True
            return False
        except Exception as e:
            print(f"❌ SR-Model health check failed: {e}")
            return False


# Global service instance
_sr_service = None


def get_sr_service() -> SRModelService:
    """
    Get or create the global SR Model service instance
    
    Returns:
        SRModelService instance
    """
    global _sr_service
    if _sr_service is None:
        _sr_service = SRModelService()
    return _sr_service
