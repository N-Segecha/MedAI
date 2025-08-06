import sys, os, tempfile, datetime, contextlib
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware

# 🕒 Capture startup timestamp
startup_timestamp = datetime.datetime.now().isoformat()

# 🔧 Unified App Instance
app = FastAPI(title="MedAI - Diagnostic API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping", tags=["Health Check"])
def ping():
    return {
        "status": "MedAI backend is alive!",
        "timestamp": startup_timestamp
    }

# 🚑 DeepChest Agent
from backend.app.models.deep_chest import DeepChestModel
deep_chest_model = DeepChestModel()

@app.post("/predict/xray", tags=["Chest X-ray"])
async def predict_xray(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        results = deep_chest_model.predict(tmp_path)
        return {"diagnosis": results}
    finally:
        os.remove(tmp_path)

# 🧬 SkinGPT Agent (Restart-safe)
from backend.app.models.skin_gpt import SkinGPTModel
try:
    skin_gpt_model = SkinGPTModel(
        model_path="backend/app/models/skin_gpt/skin_gpt.pth",
        label_csv="backend/app/models/skin_gpt/rf_class_weights.csv"
    )
    print("✅ SkinGPTModel loaded successfully.")
except RuntimeError as e:
    print(f"⚠️ SkinGPTModel failed to load: {e}")
    from backend.app.models.skin_gpt_stub import SkinGPTStub
    skin_gpt_model = SkinGPTStub()
    os.makedirs("logs", exist_ok=True)
    with open("logs/model_load.log", "a") as log_file:
        log_file.write(f"[SkinGPTModel] Load failed: {str(e)}\n")

@app.post("/predict/skin", tags=["Skin Diagnosis"])
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

@app.post("/predict/first_aid", tags=["First Aid"])
async def predict_first_aid(request: Request):
    payload = await request.json()
    result = agent.triage(payload.get("symptoms", ""), payload.get("context", {}))
    return result

# 👁️ EyeAgent
from backend.app.models.eye_agent import EyeAgent
from backend.app.schemas.response_models import OCTDiagnosisOutput
from backend.app.utils.preprocessing import preprocess_oct

eye_agent = EyeAgent()

@app.post("/predict/eye", response_model=OCTDiagnosisOutput, tags=["Eye Diagnosis"])
async def predict_eye(image: UploadFile = File(...)):
    tensor = await preprocess_oct(image)
    result = eye_agent.predict(tensor)
    return result

# 🧬 Skin Diagnosis Router (RF + GPT)
from backend.app.models.agent_router import SkinDiagnosisRouter
router = SkinDiagnosisRouter(skin_gpt_model=skin_gpt_model)

@app.post("/predict/skin/rf", tags=["Skin Diagnosis"])
async def predict_skin_rf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result_df = router.run_rf(tmp_path)
        return {
            "diagnosis": result_df.to_dict(orient="records"),
            "model": "Random Forest - Dermatology CSV",
            "status": "Success"
        }
    finally:
        os.remove(tmp_path)

@app.post("/predict/skin/gpt", tags=["Skin Diagnosis"])
async def predict_skin_gpt(image: UploadFile = File(...), prompt: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        result = router.run_gpt(tmp_path, prompt)
        return {
            "diagnosis": str(result),
            "model": "SkinGPT-4 (Vicuna-13B)",
            "status": "Success"
        }
    finally:
        os.remove(tmp_path)

# 📊 Agent Status Route
@app.get("/status/agents", tags=["Health Check"])
def agent_status():
    return {
        "DeepChest": deep_chest_model.__class__.__name__,
        "SkinGPT": skin_gpt_model.__class__.__name__,
        "FirstAid": agent.__class__.__name__,
        "EyeAgent": eye_agent.__class__.__name__,
        "Router": router.__class__.__name__
    }
