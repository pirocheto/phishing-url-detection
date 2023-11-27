import tempfile
from pathlib import Path

import numpy as np
import onnxruntime

from src.create_onnx import pkl2onnx
from src.helper import save_model, score_model
from src.modelcard import load_code, load_metrics, render_modelcard
from src.train import format_hyperparams


def test_score(dummy_model, sample_data):
    X_train, y_train = sample_data
    scores = score_model(dummy_model, X_train, y_train)
    assert len(scores) == 7

    assert "test_recall" in scores
    assert "test_precision" in scores
    assert "test_f1" in scores
    assert "test_accuracy" in scores
    assert "test_roc_auc" in scores


def test_save_pickle(dummy_model):
    _, temp_file_name = tempfile.mkstemp(suffix=".pkl")
    model_path = Path(temp_file_name)

    save_model(dummy_model, model_path)
    assert model_path.exists()
    assert model_path.stat().st_size > 0
    model_path.unlink()


def test_hyperparams():
    params = {
        "C": 9.783081707940896,
        "loss": "hinge",
        "lowercase": True,
        "max_ngram_char": 5,
        "max_ngram_word": 2,
        "tol": 0.0003837000703754547,
        "use_idf": False,
    }

    hyperparams = format_hyperparams(params)
    assert type(hyperparams) is dict
    assert len(hyperparams) == 9


def test_load_metrics():
    metrics = load_metrics("live/metrics.json")
    assert type(metrics) is str
    assert "roc_auc" in metrics
    assert "accuracy" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics


def test_load_code():
    snippets = load_code()
    assert type(snippets) is dict
    assert "py" in snippets
    assert "js" in snippets
    assert "node" in snippets
    assert "onnx" in snippets["py"]
    assert "pkl" in snippets["py"]


def test_render_modelcard():
    modelcard = render_modelcard("", load_code())
    assert type(modelcard) is str
    assert len(modelcard) > 0


def test_create_onnx(fitted_dummy_model, sample_data):
    X_test, _ = sample_data
    model = pkl2onnx(fitted_dummy_model)
    sess = onnxruntime.InferenceSession(model)

    inputs = np.array(X_test, dtype="str")

    X_pred = sess.run(None, {"inputs": inputs})[0]
    assert X_pred.shape == X_test.shape
