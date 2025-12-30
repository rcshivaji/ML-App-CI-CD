# =========================
# PYTHON PATH FIX
# =========================
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# =========================
# IMPORTS
# =========================
import os
from datetime import datetime
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data.s3_loader import load_titanic_from_s3

# =========================
# CONFIG
# =========================
BUCKET_NAME = "titanic-ml-dataset-cicd"
KEY = "Titanic-Dataset.csv"

MODELS_DIR = "models/candidates"
REPORTS_DIR = "reports"
METRICS_PATH = os.path.join(REPORTS_DIR, "model_metrics.csv")

NUMERIC_FEATURES = ["Age", "Fare", "Pclass"]
CATEGORICAL_FEATURES = ["Sex"]

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# =========================
# METRIC LOGGER
# =========================
def log_metrics(model_name, model_version, y_test, y_pred):
    metrics = {
        "model_name": model_name,
        "model_version": model_version,
        "trained_on": datetime.now().strftime("%d/%m/%Y"),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    df_new = pd.DataFrame([metrics])

    if os.path.exists(METRICS_PATH):
        df_old = pd.read_csv(METRICS_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(METRICS_PATH, index=False)
    print(f"📊 Metrics logged for {model_name}")

# =========================
# TRAINING
# =========================
def train_models():
    df = load_titanic_from_s3(BUCKET_NAME, KEY)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Logistic Regression Pipeline ----------
    lr_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]),
                NUMERIC_FEATURES
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                CATEGORICAL_FEATURES
            )
        ]
    )

    lr_pipeline = Pipeline([
        ("preprocessing", lr_preprocessor),
        ("model", LogisticRegression(max_iter=1000))
    ])

    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)

    joblib.dump(lr_pipeline, f"{MODELS_DIR}/model_lr_v1.pkl")
    log_metrics("LogisticRegression", "v1", y_test, y_pred_lr)

    # ---------- Random Forest Pipeline ----------
    rf_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
                ]),
                NUMERIC_FEATURES
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                CATEGORICAL_FEATURES
            )
        ]
    )

    rf_pipeline = Pipeline([
        ("preprocessing", rf_preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            random_state=42
        ))
    ])

    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)

    joblib.dump(rf_pipeline, f"{MODELS_DIR}/model_rf_v2.pkl")
    log_metrics("RandomForest", "v2", y_test, y_pred_rf)

    print("✅ Training completed successfully")

# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    train_models()
