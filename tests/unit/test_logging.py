import os
import json
import tempfile
from backend.app.utils.logging import log_decision, BASE_LOG_DIR, AGENT_LOG_SUBDIRS
from datetime import datetime

def test_log_creation_for_first_aid():
    input_data = {"symptoms": "burn on hand", "context": {"age": 35}}
    output_data = {"triage_steps": ["cool burn", "cover wound"], "confidence_score": 0.92}
    version = "LLAMA-v1.0"

    log_decision(agent="LLAMA_FirstAid", input=input_data, output=output_data, version=version)

    log_dir = os.path.join(BASE_LOG_DIR, AGENT_LOG_SUBDIRS["LLAMA_FirstAid"])
    recent_log = sorted(os.listdir(log_dir))[-1]
    log_path = os.path.join(log_dir, recent_log)

    with open(log_path, "r") as f:
        log_entry = json.load(f)

    assert log_entry["agent"] == "LLAMA_FirstAid"
    assert log_entry["version"] == version
    assert isinstance(log_entry["input"], dict)
    assert isinstance(log_entry["output"], dict)

def test_log_creation_for_skin_gpt():
    input_data = {"prompt": "pink lesion on scalp", "image": "lesion_01.png"}
    output_data = {"diagnosis": "Seborrheic Dermatitis"}
    version = "SkinGPT-v1.0-Vicuna"

    log_decision(agent="SkinGPT", input=input_data, output=output_data, version=version)

    log_dir = os.path.join(BASE_LOG_DIR, AGENT_LOG_SUBDIRS["SkinGPT"])
    recent_log = sorted(os.listdir(log_dir))[-1]
    log_path = os.path.join(log_dir, recent_log)

    with open(log_path, "r") as f:
        log_entry = json.load(f)

    assert log_entry["agent"] == "SkinGPT"
    assert "diagnosis" in log_entry["output"]
    assert log_entry["version"].startswith("SkinGPT-v1")

def test_log_creation_for_skin_rf():
    input_data = {"csv_file": "dermatology_input.csv"}
    output_data = {"diagnosis": [{"row": 0, "label": "Eczema"}]}
    version = "SkinRF-v1.0"

    log_decision(agent="SkinRF", input=input_data, output=output_data, version=version)

    log_dir = os.path.join(BASE_LOG_DIR, AGENT_LOG_SUBDIRS["SkinRF"])
    recent_log = sorted(os.listdir(log_dir))[-1]
    log_path = os.path.join(log_dir, recent_log)

    with open(log_path, "r") as f:
        log_entry = json.load(f)

    assert log_entry["agent"] == "SkinRF"
    assert isinstance(log_entry["output"]["diagnosis"], list)

def test_log_creation_for_eye_agent():
    input_data = {"image": "oct_scan_001.png"}
    output_data = {"label": "Glaucoma", "confidence": 0.88, "source": "EyeAgent-v1.0"}
    version = "EyeAgent-v1.0"

    log_decision(agent="EyeAgent", input=input_data, output=output_data, version=version)

    log_dir = os.path.join(BASE_LOG_DIR, AGENT_LOG_SUBDIRS["EyeAgent"])
    recent_log = sorted(os.listdir(log_dir))[-1]
    log_path = os.path.join(log_dir, recent_log)

    with open(log_path, "r") as f:
        log_entry = json.load(f)

    assert log_entry["agent"] == "EyeAgent"
    assert log_entry["output"]["label"] == "Glaucoma"
