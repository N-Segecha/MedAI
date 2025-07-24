# llama_first_aid.py
from typing import Dict
from backend.app.utils.first_aid_decision_tree import get_triage_steps
from backend.app.utils.model_loader import load_llama_first_aid_agent


class FirstAidAgentStub:
    def __init__(self, model_version="LLAMA_Stub_v0.1"):
        self.model_version = model_version

    def triage(self, symptoms: str, context: Dict = None) -> Dict:
        if not validate_symptoms(symptoms):
            return {"error": "Invalid symptom input"}

        # Decision tree stub logic
        response = {
            "triage_steps": ["Apply pressure", "Elevate limb", "Seek emergency help"],
            "confidence_score": 0.7,
            "escalation_flag": True,
            "model_version": self.model_version
        }

        log_decision("LLAMA_FirstAid", symptoms, response, self.model_version)
        return response
