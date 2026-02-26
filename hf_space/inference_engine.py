# hf_space/inference_engine.py — Self-contained ML inference for HuggingFace Space
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import yaml
import os


# ---------- Model Builder (inlined from ml/training/train.py) ----------
def build_model(config):
    """Build EfficientNet-B3 with custom classifier head."""
    weights = None  # No pretrained weights — we load our own .pth
    model = models.efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=config["model"]["dropout"]),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, config["model"]["num_classes"]),
    )
    return model


# ---------- Transforms (inlined from ml/scripts/augment.py) ----------
def get_val_test_transforms():
    """Preprocessing transforms matching training validation pipeline."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


# ---------- Inference Engine ----------
class MLInferenceEngine:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "best.pth")
        config_path = os.path.join(base_dir, "config.yaml")

        self.device = torch.device("cpu")
        self.config = self._load_config(config_path)

        # Build model architecture
        self.model = build_model(self.config)

        # Load trained weights
        if os.path.exists(model_path):
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
            self.model.to(self.device)
            self.model.eval()
            print(f"[ML Engine] Model loaded from {model_path}")
        else:
            print(f"[ML Engine] WARNING: Model not found at {model_path}")
            self.model = None

        self.transform = get_val_test_transforms()

        # Class labels: ImageFolder alphabetical order -> fake=0, real=1
        self.classes = ["FAKE", "AUTHENTIC"]

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
                label = (
                    self.classes[label_idx]
                    if label_idx < len(self.classes)
                    else "UNKNOWN"
                )

                return {"label": label, "confidence": confidence}

        except Exception as e:
            print(f"[ML Engine] Inference Error: {e}")
            return {"label": "ERROR", "confidence": 0.0, "reason": str(e)}
