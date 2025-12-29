import os
from datetime import datetime

import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.s3_loader import load_titanic_from_s3
from src.preprocessing.preprocess import preprocess_data

# =========================
# Configuration
# =========================

BUCKET_NAME = "titanic-ml-dataset-cicd"
KEY = "Titanic-Dataset.csv"   # must exactly match S3 object key

MODEL_NAME = "LogisticRegression"
MODEL_VERSION = "v1"

MODEL_DIR = "models/active"
REPORTS_DIR = "reports"
METRICS_PATH = os.path.join(REPORTS_DIR, "model_metrics.csv")


# =========================
# Training Function
# =========================

def train_model():
    # 1. Load dataset from S3
    df = load_titanic_from_s3(
        bucket_name=BUCKET_NAME,
        key=KEY
    )

    # 2. Preprocess data
    X, y, scaler = preprocess_data(df, fit=True)

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 4. Train baseline model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Evaluate model
    y_pred = model.predict(X_test)

    metrics = {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "trained_on": datetime.now().strftime("%d/%m/%Y"),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    # 6. Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # 7. Persist model artifacts
    model_path = os.path.join(MODEL_DIR, f"model_{MODEL_VERSION}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{MODEL_VERSION}.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    # 8. Append metrics to CSV (preserve history)
    metrics_df = pd.DataFrame([metrics])

    if os.path.exists(METRICS_PATH):
        existing_df = pd.read_csv(METRICS_PATH)
        combined_df = pd.concat(
            [existing_df, metrics_df],
            ignore_index=True
        )
    else:
        combined_df = metrics_df

    combined_df.to_csv(METRICS_PATH, index=False)

    # 9. Log output
    print("Training completed successfully")
    print("Model saved to:", model_path)
    print("Metrics appended to:", METRICS_PATH)
    print(metrics_df)


# =========================
# Entry Point
# =========================

if __name__ == "__main__":
    train_model()
