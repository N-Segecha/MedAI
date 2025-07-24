# model_loader.py
from llama_first_aid import FirstAidAgentStub

def load_llama_first_aid_agent():
    return FirstAidAgentStub()
def load_skin_gpt_model(fallback=True):
    try:
        return SkinGPTModel(model_id="lmsys/vicuna-13b-v1.3")
    except:
        if fallback:
            print("🔁 Falling back to lightweight Phi-2")
            return SkinGPTModel(model_id="microsoft/phi-2")
        else:
            raise
