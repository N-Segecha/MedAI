import torch
from PIL import Image
from torchvision import transforms
from backend.app.models.skin_gpt import SkinGPTModel  # Hypothetical import path

class SkinGPTAgent:
    def __init__(self, model_path="skin_gpt.pth", device="cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = SkinGPTModel()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def diagnose(self, image_path, prompt):
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        prompt_tensor = torch.tensor([prompt]).to(self.device)  # Replace with tokenizer logic
        with torch.no_grad():
            output = self.model(image_tensor, prompt_tensor)
        return output  # Adjust to match structured return format
