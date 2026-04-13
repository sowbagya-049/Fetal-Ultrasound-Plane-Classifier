"""
FastAPI Prediction API — Fetal Ultrasound Plane Classifier
Endpoint: POST /predict
Usage:  curl -X POST http://localhost:8000/predict -F "file=@scan.png"
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import efficientnet_b0
from PIL import Image
import io, os, time

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fetal Ultrasound Classifier API",
    description="Classifies fetal ultrasound images into 6 anatomical planes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES = [
    "Fetal Abdomen", "Fetal Brain", "Fetal Femur",
    "Fetal Thorax",  "Maternal Cervix", "Other"
]
DEVICE = torch.device("cpu")

# ── Model ─────────────────────────────────────────────────────────────────────
def load_model():
    model = efficientnet_b0(weights=None)
    in_f  = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 6),
    )
    for path in ["best_efficientnet.pth", "/app/best_efficientnet.pth"]:
        if os.path.exists(path):
            ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            print(f"[INFO] Model loaded from {path}")
            break
    model.eval()
    return model

_model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_probabilities: dict
    inference_time_ms: float

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Fetal Ultrasound Classifier API is running.", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "EfficientNet-B0", "classes": CLASSES}

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only PNG/JPG images are accepted.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    t0 = time.time()
    with torch.no_grad():
        probs = torch.softmax(_model(tensor), dim=1).squeeze().tolist()
    latency = (time.time() - t0) * 1000

    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
    return PredictionResponse(
        predicted_class    = CLASSES[pred_idx],
        confidence         = round(probs[pred_idx], 4),
        all_probabilities  = {cls: round(p, 4) for cls, p in zip(CLASSES, probs)},
        inference_time_ms  = round(latency, 2),
    )