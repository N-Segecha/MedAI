# backend/app/utils/preprocessing.py

import torch
from PIL import Image
import torchvision.transforms as transforms
from fastapi import UploadFile

def preprocess_oct(file: UploadFile) -> torch.Tensor:
    image = Image.open(file.file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # [1, C, H, W]
