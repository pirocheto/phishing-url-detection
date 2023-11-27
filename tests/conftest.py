import pytest

from src.helper import load_data, load_onnx_session, load_pickle_model


@pytest.fixture
def X_sample():
    X, _ = load_data("data/test.parquet")
    return X[:50]


@pytest.fixture
def onnx_sess():
    return load_onnx_session("live/model/model.onnx")


@pytest.fixture
def pkl_model():
    return load_pickle_model("live/model/model.pkl")
