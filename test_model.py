print("📍 Running test_model.py")


import torch

try:
    weights = torch.load("backend/app/models/skin_gpt/skingpt4_vicuna_v1.pth", map_location="cpu")
    print("✅ Checkpoint loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load checkpoint: {e}")

print(type(weights))
print(weights.keys())

