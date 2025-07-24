def test_eye_agent_prediction():
    from backend.app.models.eye_agent import EyeAgent
    import torch
    dummy = torch.randn(1, 3, 224, 224)
    agent = EyeAgent()
    result = agent.predict(dummy)
    assert "label" in result
    assert "confidence" in result
    assert isinstance(result["confidence"], float)
