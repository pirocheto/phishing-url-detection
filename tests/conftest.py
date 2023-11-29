import pytest

from src.helper import load_data, load_onnx_session, load_pickle_model


# Fixture to provide a sample of input data (X_sample) for testing
@pytest.fixture
def X_sample():
    X, _ = load_data("data/test.parquet")
    return X[:50]


# Fixture to provide an ONNX session (onnx_sess) for testing
@pytest.fixture
def onnx_sess():
    return load_onnx_session("live/model/model.onnx")


# Fixture to provide a pickled model (pkl_model) for testing
@pytest.fixture
def pkl_model():
    return load_pickle_model("live/model/model.pkl")
