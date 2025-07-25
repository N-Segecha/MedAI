def route_to_agent(symptoms: str, modality: str = "text") -> str:
    """Routes input to appropriate diagnostic agent based on symptoms and modality."""

    keywords_first_aid = ["bleeding", "burn", "faint", "labor"]
    keywords_skin = ["rash", "itch", "lesion", "skin", "scalp"]
    keywords_eye = ["blurred", "vision", "ocular", "eye", "sight"]

    symptoms_lower = symptoms.lower()

    if any(k in symptoms_lower for k in keywords_first_aid):
        return "LLAMA_FirstAid"
    elif any(k in symptoms_lower for k in keywords_skin):
        if modality == "csv":
            return "SkinRF"
        elif modality == "image":
            return "SkinGPT"
    elif any(k in symptoms_lower for k in keywords_eye):
        return "EyeAgent"

    return "CHW_Toolkit"

