# first_aid_decision_tree.py
def get_triage_steps(symptoms: str) -> list:
    if "bleeding" in symptoms.lower():
        return ["Apply direct pressure", "Clean wound", "Call emergency services"]
    if "burn" in symptoms.lower():
        return ["Cool area with water", "Cover with clean cloth", "Seek medical help"]
    return ["Monitor symptoms", "Provide basic support", "Refer to CHW if needed"]
