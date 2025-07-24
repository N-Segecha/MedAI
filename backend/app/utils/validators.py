# validators.py
import re

def validate_symptoms(symptoms: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", symptoms)) and len(symptoms.strip()) > 5
