from rf_wrapper import RFClassifier
from backend.app.models.gpt_wrapper import SkinGPTAgent

class SkinDiagnosisRouter:
    def __init__(self):
        self.rf_agent = RFClassifier(
            model_path="rf_model.joblib",
            weights_path="rf_class_weights.csv"
        )
        self.gpt_agent = SkinGPTAgent(model_path="skin_gpt.pth")

    def run_rf(self, csv_path):
        return self.rf_agent.predict(csv_path)

    def run_gpt(self, image_path, prompt):
        return self.gpt_agent.diagnose(image_path, prompt)

    def metadata(self):
        return {
            "rf": self.rf_agent.explain(),
            "gpt": "SkinGPT v1.0 - fine-tuned on medical skin images"
        }
