# test_llama_first_aid.py
from backend.app.models.llama_first_aid import FirstAidAgentStub

def test_stub_agent_output():
    agent = FirstAidAgentStub()
    symptoms = "The patient has heavy bleeding after a cut"
    result = agent.triage(symptoms)

    assert isinstance(result, dict)
    assert "triage_steps" in result
    assert result["confidence_score"] <= 1.0
