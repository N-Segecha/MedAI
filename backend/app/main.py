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

# 🧬 SkinGPT + RF from Tabular + Vision
from backend.app.models.skin_gpt import SkinGPTModel

skin_gpt_model = SkinGPTModel(
    model_path="backend/app/models/skin_gpt/skin_gpt.pth",
    label_csv="backend/app/models/skin_gpt/rf_class_weights.csv"
)


from backend.app.models.skin_gpt.agent_router import SkinDiagnosisRouter
router = SkinDiagnosisRouter()

@app.post("/predict/skin/rf")
async def predict_skin_rf(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name
    result_df = router.run_rf(tmp_path)
    os.remove(tmp_path)
    return {
        "diagnosis": result_df.to_dict(orient="records"),
        "model": "Random Forest - Dermatology CSV",
        "status": "Success"
    }

@app.post("/predict/skin/gpt")
async def predict_skin_gpt(image: UploadFile = File(...), prompt: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        contents = await image.read()
        tmp.write(contents)
        tmp_path = tmp.name
    result = router.run_gpt(tmp_path, prompt)
    os.remove(tmp_path)
    return {
        "diagnosis": str(result),
        "model": "SkinGPT-4 (Vicuna-13B)",
        "status": "Success"
    }

@app.get("/predict/skin/metadata")
def get_skin_model_metadata():
    return router.metadata()
