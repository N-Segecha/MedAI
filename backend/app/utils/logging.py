import json
from datetime import datetime
import os

BASE_LOG_DIR = "MedAI/data/logs/"
AGENT_LOG_SUBDIRS = {
    "LLAMA_FirstAid": "first_aid",
    "SkinRF": "skin_rf",
    "SkinGPT": "skin_gpt",
    "EyeAgent": "eye_agent",
    "DeepChest": "deep_chest"
}

def log_decision(agent: str, input: dict, output: dict, version: str):
    subdir = AGENT_LOG_SUBDIRS.get(agent, "general")
    log_dir = os.path.join(BASE_LOG_DIR, subdir)
    os.makedirs(log_dir, exist_ok=True)

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent,
        "version": version,
        "input": input,
        "output": output
    }

    fname = f"log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S%f')[:-3]}.json"
    filepath = os.path.join(log_dir, fname)

    with open(filepath, 'w') as f:
        json.dump(log_entry, f, indent=2)
