# tools.py
from smolagents import Tool
from catboost import CatBoostClassifier, Pool
import pandas as pd
import json
from pathlib import Path
import uuid
from datetime import datetime

MODEL_PATH = Path("cat_model.cbm")
BUFFER_PATH = Path("intake_buffer.parquet")

_model = CatBoostClassifier()
_model.load_model(str(MODEL_PATH))

KEY_FEATURES = []

def _to_df_row(feat_dict: dict) -> pd.DataFrame:
    return pd.DataFrame([feat_dict])

@Tool
def predict_dropout(features: dict) -> dict:

    missing = [k for k in KEY_FEATURES if k not in features or features[k] in [None, ""]]
    if missing:
        return {"y_hat": None, "top_missing": missing[:3]}
    X = _to_df_row(features)
    pool = Pool(X)
    y_hat = float(_model.predict_proba(pool)[:,1][0])
    return {"y_hat": y_hat, "top_missing": []}

@Tool
def suggest_next_questions(features: dict) -> list:

    priority = ["stress","absence_rate","gpa","age","gender","major"]
    return [k for k in priority if k not in features or features[k] in [None, ""]][:2]

@Tool
def save_intake(student_id: str, week: int, features: dict) -> str:

    session_id = str(uuid.uuid4())
    row = {
        "session_id": session_id,
        "student_id": student_id,
        "week": int(week),
        "collected_features": json.dumps(features),
        "timestamp": datetime.now().isoformat(),
        "y_next": None
    }
    df = pd.DataFrame([row])
    if BUFFER_PATH.exists():
        old = pd.read_parquet(BUFFER_PATH)
        pd.concat([old, df], ignore_index=True).to_parquet(BUFFER_PATH, index=False)
    else:
        df.to_parquet(BUFFER_PATH, index=False)
    return session_id

@Tool
def refresh_model() -> str:

    global _model
    _model = CatBoostClassifier()
    _model.load_model(str(MODEL_PATH))
    return "model reloaded"
