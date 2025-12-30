import os


def test_model_artifacts_exist():
    assert os.path.exists("models/active/model.pkl")
