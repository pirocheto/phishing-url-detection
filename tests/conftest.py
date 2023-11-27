import dvc.api
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder

from src.helper import create_model, load_model


@pytest.fixture
def sample_data():
    params = dvc.api.params_show()
    df_test = pd.read_parquet(params["data"]["test"])
    df_sample = df_test.head(50)
    y = LabelEncoder().fit_transform(df_sample["status"])
    return df_sample["url"], y


@pytest.fixture
def dummy_model():
    return create_model()


@pytest.fixture
def fitted_dummy_model(dummy_model, sample_data):
    X_train, y_train = sample_data
    dummy_model.fit(X_train, y_train)
    return dummy_model


@pytest.fixture
def model():
    model = create_model()
    return model


@pytest.fixture
def pkl_model():
    params = dvc.api.params_show()
    model = load_model(params["model"]["pickle"])
    return model


@pytest.fixture
def onnx_model():
    params = dvc.api.params_show()
    model = load_model(params["model"]["onnx"], model_format="onnx")
    return model
