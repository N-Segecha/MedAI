# routing.py
def route_to_agent(symptoms: str) -> str:
    keywords = ["bleeding", "burn", "faint", "labor"]
    if any(k in symptoms.lower() for k in keywords):
        return "LLAMA_FirstAid"
    return "CHW_Toolkit"
