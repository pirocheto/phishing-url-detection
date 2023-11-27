from src.helper import load_data


def test_load_data():
    X_train, y_train = load_data("data/train.parquet")
    assert len(X_train.shape) == 1
    assert X_train.shape[0] > 0
    assert X_train.shape == y_train.shape
