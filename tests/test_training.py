import os


def test_model_artifacts_exist():
    assert os.path.exists("models/active/model_v1.pkl")
    assert os.path.exists("models/active/scaler_v1.pkl")
