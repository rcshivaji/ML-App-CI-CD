import joblib
from pathlib import Path
import pandas as pd
from src.preprocessing.preprocess import preprocess_data

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "active" / "model_v1.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "active" / "scaler_v1.pkl"

def load_artifacts():
    """
    Load trained model and scaler from disk.
    This should be called once at app startup.
    """
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

def predict_survival(input_data: dict, model, scaler):
    """
    Predict survival and probability for a single passenger.
    """
    df = pd.DataFrame([input_data])

    X, _, _ = preprocess_data(
        df,
        fit=False,
        scaler=scaler
    )

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0][1]

    return int(prediction), float(probability)