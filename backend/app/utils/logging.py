# logging.py
import json
from datetime import datetime
import os

LOG_DIR = "MedAI/data/logs/first_aid/"
os.makedirs(LOG_DIR, exist_ok=True)

def log_decision(agent: str, input: str, output: dict, version: str):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "agent": agent,
        "version": version,
        "input": input,
        "output": output
    }
    fname = f"{LOG_DIR}log_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, 'w') as f:
        json.dump(log_entry, f, indent=2)
