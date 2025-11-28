
from gradio_client import Client, handle_file
from PIL import Image
import numpy as np
import base64
import io


class ModelService:
    """
    Service class for loading and running the ML pipeline
    """
    
    def __init__(self):
        self.client = Client("HegdeSudarshan/Classifier")
        print("ModelService initialized with Hugging Face Classifier API.")
    
    def _load_models(self):
        """Load both SR and Classification models from saved weights"""
        try:
            # Load SR Model
            sr_weights_path = self.model_weights_dir / "sr_model_final.pth"
            if sr_weights_path.exists():
                self.sr_model = SR_Model()
                self.sr_model.load_state_dict(torch.load(sr_weights_path, map_location=self.device))
                self.sr_model.to(self.device)
                self.sr_model.eval()
                print("✅ SR Model loaded successfully")
            else:
                print(f"⚠️  SR model weights not found at {sr_weights_path}")
                print("   Using placeholder SR model")
                self.sr_model = SR_Model().to(self.device)
                self.sr_model.eval()
            
            # Load Classification Model
            clf_weights_path = self.model_weights_dir / "clf_model_final.pth"
            if clf_weights_path.exists():
                self.clf_model = CLF_Model(num_classes=10)
                self.clf_model.load_state_dict(torch.load(clf_weights_path, map_location=self.device))
                self.clf_model.to(self.device)
                self.clf_model.eval()
                print("✅ Classification Model loaded successfully")
            else:
                print(f"⚠️  Classification model weights not found at {clf_weights_path}")
                print("   Using placeholder Classification model")
                self.clf_model = CLF_Model(num_classes=10).to(self.device)
                self.clf_model.eval()
                
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            # Fallback to placeholder models
            self.sr_model = SR_Model().to(self.device)
            self.clf_model = CLF_Model(num_classes=10).to(self.device)
            self.sr_model.eval()
            self.clf_model.eval()
    
    def _preprocess_image(self, image_data, target_size=(16, 16)):
        """
        Preprocess image data to create LR input tensor
        
        Args:
            image_data: Input image (PIL Image or numpy array)
            target_size: Target size for LR image (width, height)
        
        Returns:
            LR tensor (1, 3, H, W)
        """
        try:
            # Convert to PIL Image if needed
            if isinstance(image_data, np.ndarray):
                if image_data.dtype != np.uint8:
                    image_data = (image_data * 255).astype(np.uint8)
                image_data = Image.fromarray(image_data)
            
            # Resize to LR size
            lr_image = image_data.resize(target_size, Image.BICUBIC)
            
            # Convert to tensor and normalize
            lr_array = np.array(lr_image).astype(np.float32) / 255.0
            
            # Handle grayscale
            if len(lr_array.shape) == 2:
                lr_array = np.stack([lr_array] * 3, axis=-1)
            
            # Transpose to CHW format and add batch dimension
            lr_tensor = torch.from_numpy(lr_array.transpose(2, 0, 1)).unsqueeze(0)
            
            return lr_tensor.to(self.device)
            
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            # Return random tensor as fallback
            return torch.rand(1, 3, *target_size).to(self.device)
    
    def _tensor_to_base64(self, tensor):
        """
        Convert tensor to base64 encoded image string
        
        Args:
            tensor: Image tensor (C, H, W) or (1, C, H, W)
        
        Returns:
            Base64 encoded image string
        """
        try:
            # Remove batch dimension if present
            if tensor.dim() == 4:
                tensor = tensor.squeeze(0)
            
            # Convert to numpy and transpose to HWC
            img_array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
            
            # Clip and convert to uint8
            img_array = np.clip(img_array * 255, 0, 255).astype(np.uint8)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(img_array)
            
            # Convert to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return img_base64
            
        except Exception as e:
            print(f"Error converting tensor to base64: {e}")
            # Return placeholder base64 (1x1 pixel)
            return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    
    def run_pipeline(self, image_data):
        """
        Run the ML pipeline using Hugging Face Classifier API
        Args:
            image_data: Input image data (PIL Image, numpy array, or file path)
        Returns:
            Dict containing:
                - land_class_name: Predicted class name
                - confidence_score: Confidence of the prediction
                - sr_image_b64: Enhanced image as base64
                - error: Error message if any
        """
        import requests
        try:
            # Convert image_data to file if needed
            if isinstance(image_data, Image.Image):
                buffer = io.BytesIO()
                image_data.save(buffer, format='PNG')
                buffer.seek(0)
                image_path = "temp_input.png"
                with open(image_path, "wb") as f:
                    f.write(buffer.getvalue())
            elif isinstance(image_data, str):
                image_path = image_data
            else:
                # Assume numpy array
                pil_img = Image.fromarray(image_data)
                buffer = io.BytesIO()
                pil_img.save(buffer, format='PNG')
                buffer.seek(0)
                image_path = "temp_input.png"
                with open(image_path, "wb") as f:
                    f.write(buffer.getvalue())

            result = self.client.predict(
                image=handle_file(image_path),
                api_name="/predict"
            )
            # result[0]: enhanced image dict, result[1]: classification dict
            sr_img_url = result[0].get("url")
            sr_img_b64 = ""
            error_msg = None
            if sr_img_url:
                try:
                    resp = requests.get(sr_img_url)
                    if resp.status_code == 200:
                        sr_img_b64 = base64.b64encode(resp.content).decode('utf-8')
                    else:
                        error_msg = f"Failed to fetch SR image, status code: {resp.status_code}"
                except Exception as img_err:
                    error_msg = f"Exception fetching SR image: {img_err}"
            else:
                error_msg = "SR image URL missing from API response."

            label = result[1].get("label")
            confidences = result[1].get("confidences", [])
            confidence_score = 0.0
            if confidences and isinstance(confidences, list):
                for c in confidences:
                    if c.get("label") == label:
                        confidence_score = c.get("confidence", 0.0)
                        break

            # If SR image is missing, add error info
            if not sr_img_b64:
                print(f"[MLService] Warning: SR image not returned. Reason: {error_msg}")

            return {
                "land_class_name": label,
                "confidence_score": confidence_score,
                "sr_image_b64": sr_img_b64,
                "error": error_msg
            }
        except Exception as e:
            print(f"Error in ML pipeline: {e}")
            return {
                "land_class_name": None,
                "confidence_score": 0.0,
                "sr_image_b64": "",
                "error": str(e)
            }
    
    def create_fake_satellite_image(self, width=64, height=64):
        """
        Create a fake satellite image for testing
        
        Args:
            width: Image width
            height: Image height
        
        Returns:
            PIL Image
        """
        # Create a realistic-looking satellite image
        np.random.seed(42)  # For consistent results
        
        # Create base terrain
        image = np.random.rand(height, width, 3) * 0.3
        
        # Add some patterns that look like fields/vegetation
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                if np.random.rand() > 0.6:
                    # Green areas (vegetation)
                    image[i:i+8, j:j+8, 1] += 0.4
                elif np.random.rand() > 0.3:
                    # Brown areas (soil)
                    image[i:i+8, j:j+8, 0] += 0.3
                    image[i:i+8, j:j+8, 1] += 0.2
        
        # Clip values
        image = np.clip(image, 0, 1)
        
        # Convert to PIL
        return Image.fromarray((image * 255).astype(np.uint8))