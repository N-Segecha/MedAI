import os
import torch
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

from .skin_classifier import SkinClassifier  # ✅ Assumes skin_classifier.py is in models/

class SkinGPTModel:
    def __init__(
        self,
        model_path="backend/app/models/skin_gpt/skin_gpt.pth",
        label_csv="backend/app/models/skin_gpt/rf_class_weights.csv",
        device=None
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not os.path.exists(model_path):
            raise RuntimeError(f"❌ Checkpoint not found: {model_path}")
        if not os.path.exists(label_csv):
            raise RuntimeError(f"❌ Label CSV not found: {label_csv}")

        try:
            df = pd.read_csv(label_csv)
            self.index_to_label = {i: label for i, label in enumerate(df.columns)}

            self.model = SkinClassifier(num_classes=len(self.index_to_label))
            weights = torch.load(model_path, map_location=self.device)
            # 🔍 Inspect checkpoint keys
            print("🔍 Checkpoint contains the following keys:")
            for k in weights.keys():
               print(k)


            try:
                self.model.load_state_dict(weights)  # strict=True by default
            except RuntimeError as mismatch:
                raise RuntimeError("🔍 Checkpoint mismatch with model architecture:\n" + str(mismatch))

            self.model.to(self.device)
            self.model.eval()

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])

            print("✅ SkinClassifier initialized with ResNet50 backbone and", len(self.index_to_label), "output classes")
            print("🧩 Sample checkpoint keys:", [k for k in weights.keys() if "fc" in k][-2:])

        except Exception as e:
            raise RuntimeError("Model setup failed. Try a smaller checkpoint or verify your cache.\n" + str(e))

    def diagnose(self, image_path):
        if not os.path.exists(image_path):
            return {"error": f"Image not found: {image_path}", "source": image_path}
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.softmax(outputs, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0][pred_idx].item()
                label = self.index_to_label[pred_idx]

                return {
                    "prediction": label,
                    "confidence": round(confidence, 4),
                    "source": image_path
                }
        except Exception as e:
            return {"error": str(e), "source": image_path}

