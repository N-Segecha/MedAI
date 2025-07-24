import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SkinGPTModel:
    def __init__(self, model_id="microsoft/phi-2"):
        self.device = "cpu"  # 💡 Explicit CPU assignment
        print(f"🔧 Initializing SkinGPT on {self.device} using model: {model_id}")

        # 🧠 Load tokenizer safely
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                use_fast=False,
                legacy=True,
                local_files_only=False
            )
            print("✅ Tokenizer loaded")
        except Exception as e:
            print(f"❌ Tokenizer failed: {type(e).__name__} — {e}")
            raise RuntimeError("Tokenizer setup failed. Check model name or internet connection.")

        # ⚡ Load model with fallback
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                local_files_only=False
            )
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded and ready")
        except Exception as e:
            print(f"❌ Model load failed: {type(e).__name__} — {e}")
            raise RuntimeError("Model setup failed. Try a smaller checkpoint or verify your cache.")

    def diagnose(self, prompt, max_tokens=512):
        print(f"🩺 Diagnosing: {prompt[:50]}...")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_tokens)
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"📋 Diagnosis complete")
        return result
