# request_models.py
from pydantic import BaseModel

class FirstAidRequest(BaseModel):
    symptoms: str
    context: dict = {}