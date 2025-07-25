from llama_first_aid import FirstAidAgentStub
from backend.app.models.skin_gpt.skin_gpt_model import SkinGPTModel
from backend.app.models.skin_gpt.skin_rf_model import SkinRFModel
from backend.app.models.eye_agent import EyeAgent
import yaml

def load_llama_first_aid_agent():
    return FirstAidAgentStub()

def load_eye_agent():
    return EyeAgent()

def load_skin_rf_model(config_path="./configs/agent_weights/skin_gpt_agents.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    rf_cfg = config["skin_gpt_agents"]["random_forest"]
    return SkinRFModel(
        model_path=rf_cfg["artifact_path"],
        features_path=rf_cfg["feature_spec"]
    )

def load_skin_gpt_model(agent_key="skin_gpt_vicuna", config_path="./configs/agent_weights/skin_gpt_agents.yml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    agent_cfg = config["skin_gpt_agents"].get(agent_key)
    if not agent_cfg:
        raise ValueError(f"❌ Agent '{agent_key}' not found in config file.")

    return SkinGPTModel(
        model_id=agent_cfg["weight_path"],
        prompt_template=agent_cfg["prompt_template"],
        base_model=agent_cfg["metadata"]["base_model"]
    )
