"""
main.py — FastAPI routes for AutoML web application
GET  /           -> health check
POST /upload/    -> upload CSV, get preview + columns
POST /train/     -> train AutoML model, get metrics
POST /predict/   -> predict on new data
GET  /download/  -> download trained model file
"""

import os, shutil, uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Any

from backend.model import load_data, preprocess_data, train_automl_model, make_prediction

app = FastAPI(title="AutoML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Simple in-memory state
STATE: dict = {"filepath": None, "columns": [], "train_result": None}


# Schemas
class TrainRequest(BaseModel):
    target_column: str

class PredictRequest(BaseModel):
    features: dict[str, Any]


# Routes
@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "AutoML API is running"}


@app.post("/upload/", tags=["Data"])
async def upload_file(file: UploadFile = File(...)):
    """Upload a CSV file. Returns preview, column names, shape, dtypes, missing counts."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    unique_name = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, unique_name)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    df = load_data(filepath)
    summary = preprocess_data(df)

    STATE["filepath"] = filepath
    STATE["columns"] = summary["columns"]

    return {
        "filename": file.filename,
        "shape": list(summary["shape"]),
        "columns": summary["columns"],
        "dtypes": summary["dtypes"],
        "missing_values": summary["missing_values"],
        "preview": summary["preview"],
    }


@app.post("/train/", tags=["ML"])
def train_model(request: TrainRequest):
    """Train AutoML on the uploaded dataset. Body: { target_column: '...' }"""
    if STATE["filepath"] is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet.")
    if request.target_column not in STATE["columns"]:
        raise HTTPException(
            status_code=400,
            detail=f"Column '{request.target_column}' not found. Available: {STATE['columns']}"
        )
    df = load_data(STATE["filepath"])
    try:
        result = train_automl_model(df, request.target_column)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
    STATE["train_result"] = result
    return result


@app.post("/predict/", tags=["ML"])
def predict_output(request: PredictRequest):
    """Run inference. Body: { features: { col1: val1, ... } }"""
    if STATE["train_result"] is None:
        raise HTTPException(status_code=400, detail="No model trained yet.")
    try:
        return make_prediction(request.features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/download/", tags=["ML"])
def download_model():
    """Download the trained model .pkl file."""
    model_file = "backend/trained_model.pkl"
    if not os.path.exists(model_file):
        raise HTTPException(status_code=404, detail="No trained model found.")
    return FileResponse(
        path=model_file, media_type="application/octet-stream", filename="automl_model.pkl"
    )
