import torch
from backend.app.models.oct_classifier import OCTClassifier  # unchanged

class EyeAgent:
    def __init__(self, weight_path="backend/app/models/oct_classifier/model_epoch10.pth", device="cpu"):
        self.device = device
        self.model = OCTClassifier()
        self.model.load_state_dict(torch.load(weight_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()
        self.label_map = {0: 'CNV', 1: 'DME', 2: 'DRUSEN', 3: 'NORMAL'}

    def predict(self, tensor):
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
            probs = torch.nn.functional.softmax(output, dim=1)
            top = torch.argmax(probs, dim=1).item()
            return {
                "label": self.label_map[top],
                "confidence": round(probs[0][top].item(), 4),
                "source": "eye_agent_v1"
            }
