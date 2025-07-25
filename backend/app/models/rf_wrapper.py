import joblib
import pandas as pd

class RFClassifier:
    def __init__(self, model_path="rf_model.joblib", weights_path="rf_class_weights.csv"):
        self.model = joblib.load(model_path)
        self.class_weights = pd.read_csv(weights_path)

    def predict(self, X_path):
        X = pd.read_csv(X_path)
        preds = self.model.predict(X)
        return pd.DataFrame({"prediction": preds})

    def explain(self):
        return {
            "classes": self.class_weights["class"].tolist(),
            "weights": self.class_weights["weight"].tolist()
        }
