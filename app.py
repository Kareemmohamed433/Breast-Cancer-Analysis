from fastapi import FastAPI, Request, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
import pickle
import numpy as np
import json
from pathlib import Path
import os
from starlette.middleware.sessions import SessionMiddleware
from typing import List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Breast Cancer Classifier")

# Add session middleware for flashed messages
app.add_middleware(SessionMiddleware, secret_key="your-secret-key")  # Replace with a secure key

# Ensure required folders exist
os.makedirs("models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

MODEL_DIR = Path("models")

# Helper function to flash messages
def flash(request: Request, message: str, category: str = "info"):
    if "_messages" not in request.session:
        request.session["_messages"] = []
    request.session["_messages"].append({"message": message, "category": category})

# Dependency to get flashed messages
def get_flashed_messages(request: Request) -> List[Tuple[str, str]]:
    messages = request.session.get("_messages", [])
    request.session["_messages"] = []  # Clear messages after retrieval
    return [(m["category"], m["message"]) for m in messages]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    feature_info_path = MODEL_DIR / "feature_info.json"
    try:
        if not feature_info_path.exists():
            logger.error(f"Feature info file not found at {feature_info_path}")
            flash(request, "Feature information file is missing. Please contact the administrator.", "danger")
            return templates.TemplateResponse("index.html", {
                "request": request,
                "features": [],
                "descriptions": [],
                "get_flashed_messages": get_flashed_messages
            })

        with open(feature_info_path) as f:
            features = json.load(f)

        logger.info("Successfully loaded feature_info.json")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "features": features["feature_names"],
            "descriptions": features["feature_descriptions"],
            "get_flashed_messages": get_flashed_messages
        })
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse feature_info.json: {str(e)}")
        flash(request, f"Invalid feature information file: {str(e)}", "danger")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "features": [],
            "descriptions": [],
            "get_flashed_messages": get_flashed_messages
        })
    except Exception as e:
        logger.error(f"Unexpected error in home endpoint: {str(e)}")
        flash(request, f"Failed to load feature information: {str(e)}", "danger")
        return templates.TemplateResponse("index.html", {
            "request": request,
            "features": [],
            "descriptions": [],
            "get_flashed_messages": get_flashed_messages
        })

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    try:
        features = [float(form_data[f"feature_{i}"]) for i in range(30)]
        if any(f < 0 for f in features):
            logger.error("Negative values detected in input")
            flash(request, "All features must be non-negative.", "danger")
            return RedirectResponse(url="/", status_code=303)
    except ValueError as e:
        logger.error(f"Invalid input format: {str(e)}")
        flash(request, "Please enter valid numerical values.", "danger")
        return RedirectResponse(url="/", status_code=303)

    try:
        model_path = MODEL_DIR / "model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            logger.error(f"Model or scaler file missing: model={model_path.exists()}, scaler={scaler_path.exists()}")
            flash(request, "Model or scaler file is missing. Please contact the administrator.", "danger")
            return RedirectResponse(url="/", status_code=303)

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        scaled = scaler.transform(np.array(features).reshape(1, -1))
        logger.info(f"Scaled input: {scaled.tolist()}")
        prediction = model.predict(scaled)[0]
        proba = round(model.predict_proba(scaled)[0].max() * 100, 2)

        with open(MODEL_DIR / "feature_info.json") as f:
            info = json.load(f)

        flash(request, "Prediction successful!", "success")
        logger.info(f"Prediction made: class={info['target_names'][prediction]}, probability={proba}%")
        return templates.TemplateResponse("results.html", {
            "request": request,
            "prediction": int(prediction),
            "probability": float(proba),
            "class_name": info["target_names"][prediction].capitalize(),
            "model_performance": info["model_performance"],
            "get_flashed_messages": get_flashed_messages
        })
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        flash(request, f"Prediction failed: {str(e)}", "danger")
        return RedirectResponse(url="/", status_code=303)

@app.post("/predict_json")
async def predict_json(request: Request):
    try:
        form_data = await request.form()
        features = [float(form_data[f"feature_{i}"]) for i in range(30)]
        if any(f < 0 for f in features):
            logger.error("Negative values detected in input")
            raise HTTPException(status_code=400, detail="All features must be non-negative.")

        model_path = MODEL_DIR / "model.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"

        if not model_path.exists() or not scaler_path.exists():
            logger.error(f"Model or scaler file missing: model={model_path.exists()}, scaler={scaler_path.exists()}")
            raise HTTPException(status_code=500, detail="Model or scaler file is missing.")

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        scaled = scaler.transform(np.array(features).reshape(1, -1))
        prediction = model.predict(scaled)[0]
        proba = round(model.predict_proba(scaled)[0].max() * 100, 2)

        with open(MODEL_DIR / "feature_info.json") as f:
            info = json.load(f)

        logger.info(f"JSON Prediction made: class={info['target_names'][prediction]}, probability={proba}%")
        return {
            "prediction": info["target_names"][prediction].capitalize(),
            "probability": proba
        }
    except ValueError as e:
        logger.error(f"Invalid input format: {str(e)}")
        raise HTTPException(status_code=400, detail="Please enter valid numerical values.")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)