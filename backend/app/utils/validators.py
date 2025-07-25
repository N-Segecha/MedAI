import re
import os

def validate_symptoms(symptoms: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", symptoms)) and len(symptoms.strip()) > 5

def validate_prompt(prompt: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", prompt)) and len(prompt.strip()) > 10

def validate_csv_extension(filename: str) -> bool:
    return filename.lower().endswith(".csv")

def validate_image_extension(filename: str) -> bool:
    allowed_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def validate_file_exists(filepath: str) -> bool:
    return os.path.exists(filepath) and os.path.isfile(filepath)

