import joblib
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_MODEL_PATH = PROJECT_ROOT / "models" / "active" / "model.pkl"

def load_model():
    return joblib.load(ACTIVE_MODEL_PATH)

def predict_survival(input_data: dict, model):
    df = pd.DataFrame([input_data])
    df = df.drop(df.columns[0], axis=1)

    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0][1]

    return int(prediction), float(probability)
