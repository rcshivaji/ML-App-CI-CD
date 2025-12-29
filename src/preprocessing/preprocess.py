import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["Pclass", "Sex", "Age", "Fare"]
TARGET_COLUMN = "Survived"


def preprocess_data(df: pd.DataFrame, fit: bool = True, scaler=None):
    """
    Preprocess Titanic dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Raw Titanic dataframe
    fit : bool
        Whether to fit the scaler (True for training)
    scaler : StandardScaler or None
        Existing scaler (used during inference)

    Returns
    -------
    X : pd.DataFrame
        Processed feature matrix
    y : pd.Series or None
        Target variable (None if not present)
    scaler : StandardScaler
        Fitted scaler
    """

    df = df.copy()

    # Select relevant columns
    df = df[FEATURE_COLUMNS + ([TARGET_COLUMN] if TARGET_COLUMN in df else [])]

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Handle missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN] if TARGET_COLUMN in df else None

    # Scale features
    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y, scaler
