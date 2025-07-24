# response_models.py
class FirstAidResponse(BaseModel):
    triage_steps: list
    confidence_score: float
    escalation_flag: bool
    model_version: str