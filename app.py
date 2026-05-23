import joblib
import json
import numpy as np
import pandas as pd
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List
import uvicorn

MODEL_DIR = "models/"
SCALER_PATH = "data/scaler.pkl"
XGB_PATH = MODEL_DIR + "xgb_optuna_tuned.joblib"
CAT_PATH = MODEL_DIR + "catboost_optuna_tuned.joblib"
HIST_PATH = MODEL_DIR + "histgb_optuna_tuned.joblib"
WEIGHTS_PATH = MODEL_DIR + "ensemble_weights.json"
THRESHOLD = 0.5

scaler = joblib.load(SCALER_PATH)
model_xgb = joblib.load(XGB_PATH)
model_cat = joblib.load(CAT_PATH)
model_hist = joblib.load(HIST_PATH)

with open(WEIGHTS_PATH, 'r') as f:
    weights = json.load(f)

app = FastAPI(title="Sepsis Prediction API",
              description="Предсказание вероятности сепсиса по медицинским признакам",
              version="1.0.0")

class PatientFeatures(BaseModel):
    features: List[float] = Field(..., min_items=1, description="Признаки пациента (числовой вектор)")

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
    threshold: float

def predict_one(features: np.ndarray) -> float:
    scaled = scaler.transform(features.reshape(1, -1))
    return _predict_scaled(scaled[0])

def _predict_scaled(scaled_features: np.ndarray) -> float:
    p_xgb = model_xgb.predict_proba(scaled_features.reshape(1, -1))[:, 1][0]
    p_cat = model_cat.predict_proba(scaled_features.reshape(1, -1))[:, 1][0]
    p_hist = model_hist.predict_proba(scaled_features.reshape(1, -1))[:, 1][0]
    total_w = weights.get('xgb_weight', 0) + weights.get('cat_weight', 0) + weights.get('hist_weight', 0)
    if total_w == 0:
        raise ValueError("Сумма весов ансамбля равна нулю. Проверьте ensemble_weights.json")
    prob = (weights.get('xgb_weight', 0) * p_xgb +
            weights.get('cat_weight', 0) * p_cat +
            weights.get('hist_weight', 0) * p_hist) / total_w
    return prob

def _predict_batch_scaled(scaled_array: np.ndarray) -> np.ndarray:
    p_xgb = model_xgb.predict_proba(scaled_array)[:, 1]
    p_cat = model_cat.predict_proba(scaled_array)[:, 1]
    p_hist = model_hist.predict_proba(scaled_array)[:, 1]
    total_w = weights.get('xgb_weight', 0) + weights.get('cat_weight', 0) + weights.get('hist_weight', 0)
    if total_w == 0:
        raise ValueError("Сумма весов ансамбля равна нулю. Проверьте ensemble_weights.json")
    probs = (weights.get('xgb_weight', 0) * p_xgb +
             weights.get('cat_weight', 0) * p_cat +
             weights.get('hist_weight', 0) * p_hist) / total_w
    return probs

def _predict_batch_scaled(scaled_array: np.ndarray) -> np.ndarray:
    p_xgb = model_xgb.predict_proba(scaled_array)[:, 1]
    p_cat = model_cat.predict_proba(scaled_array)[:, 1]
    p_hist = model_hist.predict_proba(scaled_array)[:, 1]
    total_w = weights.get("xgb_weight", 0) + weights.get("cat_weight", 0) + weights.get("hist_weight", 0)
    probs = (weights.get("xgb_weight", 0) * p_xgb +
            weights.get("cat_weight", 0) * p_cat +
            weights.get("hist_weight", 0) * p_hist) / total_w
    return probs

def check_feature_count(feature_array, expected_n_features):
    if feature_array.shape[1] != expected_n_features:
        raise HTTPException(
            status_code=400,
            detail=f"Ожидалось {expected_n_features} признаков, получено {feature_array.shape[1]}"
        )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"message": "Sepsis Prediction API. Перейдите на /docs для тестирования."}

@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures):
    try:
        features_array = np.array(patient.features, dtype=np.float64)
        prob = predict_one(features_array)
        pred_label = int(prob >= THRESHOLD)
        return PredictionResponse(probability=round(prob, 6),
                                  prediction=pred_label,
                                  threshold=THRESHOLD)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка предсказания: {str(e)}")

@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Файл не содержит данных (пустой).")

    if 'patient_id' in df.columns:
        ids = df['patient_id'].values
        df = df.drop(columns=['patient_id'])
    else:
        ids = np.arange(len(df))

    try:
        feature_array = df.values.astype(np.float64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Все признаки должны быть числовыми: {str(e)}")

    expected_n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else scaler.mean_.shape[0]
    check_feature_count(feature_array, expected_n_features)

    probs = predict_batch(feature_array)
    preds = (probs >= THRESHOLD).astype(int)

    result_df = pd.DataFrame({
        'patient_id': ids,
        'probability': probs.round(6),
        'prediction': preds,
        'threshold': THRESHOLD
    })

    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sepsis_predictions.csv"}
    )
    return response

@app.post("/predict-scaled")
async def predict_scaled(file: UploadFile = File(...)):
    """
    Принимает CSV с уже масштабированными признаками (без patient_id или с ним).
    Возвращает предсказания (вероятность и класс) для каждой строки.
    """
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка чтения файла: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Файл не содержит данных (пустой).")

    if 'patient_id' in df.columns:
        ids = df['patient_id'].values
        df = df.drop(columns=['patient_id'])
    else:
        ids = np.arange(len(df))

    try:
        feature_array = df.values.astype(np.float64)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Все признаки должны быть числовыми: {str(e)}")

    # Проверяем число признаков: должно совпадать с ожидаемым моделями (обычно равно числу фич после scaler)
    expected_n_features = model_xgb.n_features_in_
    check_feature_count(feature_array, expected_n_features)

    probs = _predict_batch_scaled(feature_array)
    preds = (probs >= THRESHOLD).astype(int)

    result_df = pd.DataFrame({
        'patient_id': ids,
        'probability': probs.round(6),
        'prediction': preds,
        'threshold': THRESHOLD
    })

    stream = io.StringIO()
    result_df.to_csv(stream, index=False)
    response = StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sepsis_predictions_scaled.csv"}
    )
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)