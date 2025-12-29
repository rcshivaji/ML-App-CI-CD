from src.preprocessing.preprocess import preprocess_data
import pandas as pd


def test_preprocess_shape():
    data = {
        "Pclass": [1, 3],
        "Sex": ["male", "female"],
        "Age": [22, None],
        "Fare": [7.25, 71.83],
        "Survived": [0, 1],
    }

    df = pd.DataFrame(data)

    X, y, scaler = preprocess_data(df)

    assert X.shape == (2, 4)
    assert y.shape == (2,)
