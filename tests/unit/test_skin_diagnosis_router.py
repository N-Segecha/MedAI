import tempfile
import os
import pandas as pd
from PIL import Image
from backend.app.models.skin_gpt.agent_router import SkinDiagnosisRouter

router = SkinDiagnosisRouter()

def test_run_rf():
    # 🧪 Create dummy tabular CSV
    df = pd.DataFrame({
        "feature1": [0.2],
        "feature2": [0.8],
        "feature3": [0.5]
    })
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        results = router.run_rf(tmp.name)
    os.remove(tmp.name)

    # ✅ Assertions
    assert isinstance(results, pd.DataFrame), "Expected DataFrame output"
    assert not results.empty, "RF output should not be empty"
    assert "diagnosis" in results.columns, "Missing 'diagnosis' column in RF results"

def test_run_gpt():
    # 🖼️ Create dummy image + prompt
    img = Image.new("RGB", (224, 224), color=(255, 192, 203))  # light pink
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        img.save(tmp.name)
        result = router.run_gpt(tmp.name, prompt="Red rash with scaling on elbows")
    os.remove(tmp.name)

    # ✅ Assertions
    assert isinstance(result, str), "Expected string output from GPT agent"
    assert len(result) > 0, "GPT response is empty"
    assert "rash" in result.lower() or "diagnosis" in result.lower(), "GPT response lacks clinical keywords"

def test_metadata():
    # 🧾 Validate agent metadata
    meta = router.metadata()

    assert isinstance(meta, dict), "Metadata must be a dictionary"
    assert "version" in meta, "Missing 'version' in metadata"
    assert "weights" in meta, "Missing 'weights' in metadata"
