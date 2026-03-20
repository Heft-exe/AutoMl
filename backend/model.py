"""
model.py — AutoML ML logic using PyCaret
Handles: data loading, preprocessing, training, evaluation, saving, prediction
"""

import os
import pandas as pd
import numpy as np
from pycaret.classification import (
    setup as cls_setup, compare_models as cls_compare, pull as cls_pull,
    save_model as cls_save, load_model as cls_load, predict_model as cls_predict,
    get_config as cls_get_config,
)
from pycaret.regression import (
    setup as reg_setup, compare_models as reg_compare, pull as reg_pull,
    save_model as reg_save, load_model as reg_load, predict_model as reg_predict,
    get_config as reg_get_config,
)

MODEL_PATH = "backend/trained_model"
TASK_FILE  = "task_type.txt"


# 1. Load data
def load_data(filepath: str) -> pd.DataFrame:
    """Read a CSV file and return a DataFrame."""
    return pd.read_csv(filepath)


# 2. Preprocess / summarise
def preprocess_data(df: pd.DataFrame) -> dict:
    """Return a lightweight summary. PyCaret handles actual preprocessing internally."""
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "preview": df.head(5).to_dict(orient="records"),
    }


# 3. Detect task type
def detect_task_type(df: pd.DataFrame, target: str) -> str:
    """Heuristic: <=20 unique values or object dtype => classification, else regression."""
    col = df[target]
    if col.dtype == object or col.dtype == bool or col.nunique() <= 20:
        return "classification"
    return "regression"


# 4. Train AutoML model
def train_automl_model(df: pd.DataFrame, target: str) -> dict:
    """Run PyCaret AutoML. Returns metrics, feature importance, and comparison table."""
    task = detect_task_type(df, target)
    with open(TASK_FILE, "w") as f:
        f.write(task)

    if task == "classification":
        cls_setup(data=df, target=target, session_id=42, verbose=False, html=False)
        best_model = cls_compare(n_select=1, verbose=False)
        results_df = cls_pull()
        cls_save(best_model, MODEL_PATH)
        importance = _get_feature_importance(best_model, cls_get_config("X_train"))
    else:
        reg_setup(data=df, target=target, session_id=42, verbose=False, html=False)
        best_model = reg_compare(n_select=1, verbose=False)
        results_df = reg_pull()
        reg_save(best_model, MODEL_PATH)
        importance = _get_feature_importance(best_model, reg_get_config("X_train"))

    top_row = results_df.iloc[0].to_dict()
    metrics = {
        k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else str(v))
        for k, v in top_row.items()
    }

    return {
        "task_type": task,
        "best_model_name": type(best_model).__name__,
        "metrics": metrics,
        "feature_importance": importance,
        "comparison_table": results_df.head(10).to_dict(orient="records"),
    }


# 5. Evaluate
def evaluate_model(train_result: dict) -> dict:
    return {"task_type": train_result["task_type"],
            "best_model": train_result["best_model_name"],
            "metrics": train_result["metrics"]}


# 6. Save / Load
def save_model():
    return MODEL_PATH

def load_trained_model():
    task = _read_task_type()
    if task == "classification":
        return cls_load(MODEL_PATH), task
    return reg_load(MODEL_PATH), task


# 7. Predict
def make_prediction(input_data: dict) -> dict:
    model, task = load_trained_model()
    input_df = pd.DataFrame([input_data])
    if task == "classification":
        pred_df = cls_predict(model, data=input_df, verbose=False)
    else:
        pred_df = reg_predict(model, data=input_df, verbose=False)
    prediction = pred_df["prediction_label"].iloc[0]
    score = pred_df.get("prediction_score", pd.Series([None])).iloc[0]
    return {
        "prediction": str(prediction),
        "confidence": round(float(score), 4) if score is not None and not pd.isna(score) else None,
    }


# Helpers
def _get_feature_importance(model, X_train) -> list:
    try:
        importances = model.feature_importances_
        features = X_train.columns.tolist()
        pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
        return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]
    except AttributeError:
        try:
            coef = np.abs(model.coef_).flatten()
            features = X_train.columns.tolist()
            pairs = sorted(zip(features, coef), key=lambda x: x[1], reverse=True)
            return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]
        except Exception:
            return []

def _read_task_type() -> str:
    if os.path.exists(TASK_FILE):
        with open(TASK_FILE) as f:
            return f.read().strip()
    return "classification"
