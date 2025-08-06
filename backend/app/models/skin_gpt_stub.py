# backend/app/models/skin_gpt_stub.py

class SkinGPTStub:
    def diagnose(self, prompt):
        return {
            "status": "unavailable",
            "message": "SkinGPT model not loaded due to memory constraints or missing checkpoint."
        }
