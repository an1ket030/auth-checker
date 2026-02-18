
import torch
from torchvision import transforms
from PIL import Image
import io
import yaml
import os
from ml.training.train import build_model
from ml.scripts.augment import get_val_test_transforms

class MLInferenceEngine:
    def __init__(self, model_path="ml/model/best.pth", config_path="ml/training/config.yaml"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self._load_config(config_path)
        
        # Load Model Structure
        self.model = build_model(self.config)
        
        # Load Weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print(f"ML Engine loaded model from {model_path}")
        else:
            print(f"WARNING: Model file not found at {model_path}. Inference will fail.")
            self.model = None

        # Preprocessing
        self.transform = get_val_test_transforms()
        
        # Classes (Assumed order, verify with training)
        # Typically ImageFolder correlates index 0 to first alphabetical folder.
        # If folders are 'fake', 'real' -> 0: Fake, 1: Real
        self.classes = ['FAKE', 'AUTHENTIC'] 

    def _load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def predict(self, image_bytes):
        if not self.model:
            return {"label": "ERROR", "confidence": 0.0, "reason": "Model not loaded"}

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
                
                label_idx = pred.item()
                confidence = conf.item()
                
                label = self.classes[label_idx] if label_idx < len(self.classes) else "UNKNOWN"
                
                return {
                    "label": label,
                    "confidence": confidence
                }
        except Exception as e:
            print(f"Inference Error: {e}")
            return {"label": "ERROR", "confidence": 0.0, "reason": str(e)}
