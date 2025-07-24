from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
import tempfile, os

# 🔧 Unified App Instance
app = FastAPI(title="MedAI - Diagnostic API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
def ping():
    return {"status": "MedAI backend is alive!"}

# 🚑 DeepChest
from backend.app.models.deep_chest import DeepChestModel
deep_chest_model = DeepChestModel()

@app.post("/predict/xray")
async def predict_xray(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    results = deep_chest_model.predict(tmp_path)
    os.remove(tmp_path)
    return {"diagnosis": results}

# 🧬 SkinGPT
from backend.app.models.skin_gpt import SkinGPTModel
skin_gpt_model = SkinGPTModel()

@app.post("/predict/skin")
async def predict_skin(prompt: str = Form(...)):
    result = skin_gpt_model.diagnose(prompt)
    return {
        "diagnosis": result,
        "model": "SkinGPT-4 (Vicuna-13B)",
        "status": "Success"
    }

# 🩺 LLAMA First Aid Stub
from backend.app.models.llama_first_aid import FirstAidAgentStub
agent = FirstAidAgentStub()

@app.post("/predict/first_aid")
async def predict_first_aid(request: Request):
    payload = await request.json()
    symptoms = payload.get("symptoms", "")
    context = payload.get("context", {})
    result = agent.triage(symptoms, context)
    return result

from backend.app.models.eye_agent import EyeAgent
from backend.app.schemas.response_models import OCTDiagnosisOutput

eye_agent = EyeAgent()

@app.post("/predict/eye", response_model=OCTDiagnosisOutput)
async def predict_eye(image: UploadFile = File(...)):
    tensor = await preprocess_oct(image)  # your loader function
    result = eye_agent.predict(tensor)
    return result
