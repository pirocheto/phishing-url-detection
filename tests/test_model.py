import time
from pathlib import Path

import dvc.api
import numpy as np
import pytest

# ! Constants for test conditions
# Maximum allowed model size in megabytes (Mo)
MAX_SIZE = 50

# Minimum required F1 score
MIN_F1 = 0.90

# Minimum required precision score
MIN_PRECISION = 0.90

# Minimum required recall score
MIN_RECALL = 0.90

# Minimum required ROC AUC score
MIN_ROC_AUC = 0.90

# Minimum required accuracy score
MIN_ACCURACY = 0.90

# Maximum allowed inference time in seconds
MAX_INFERENCE_TIME = 0.5


def test_metrics():
    # Retrieve metrics using dvc.api
    metrics = dvc.api.metrics_show()

    # Assertions for metric values
    f1 = metrics["f1"]
    assert f1 > MIN_F1, f"F1 score below ({f1}) the minimum required ({MIN_F1})"

    precision = metrics["precision"]
    assert (
        precision > MIN_PRECISION
    ), f"Precision score ({precision}) below the minimum required ({MIN_PRECISION})"

    recall = metrics["recall"]
    assert (
        recall > MIN_RECALL
    ), f"Recall score ({recall}) below the minimum required ({MIN_RECALL})"

    roc_auc = metrics["roc_auc"]
    assert (
        roc_auc > MIN_ROC_AUC
    ), f"ROC AUC score ({roc_auc}) below the minimum required ({MIN_ROC_AUC})"

    accuracy = metrics["accuracy"]
    assert (
        accuracy > MIN_ACCURACY
    ), f"Accuracy score ({accuracy}) below the minimum required ({MIN_ACCURACY})"


@pytest.mark.parametrize("path", [("live/model/model.onnx"), ("live/model/model.pkl")])
def test_model_size(path):
    """Check the size of the model file"""

    model_path = Path(path)
    size_mo = model_path.stat().st_size / (1024**2)
    assert (
        size_mo < MAX_SIZE
    ), f"Model size ({size_mo} MB) exceeds the maximum allowed size ({MAX_SIZE} MB)"


def get_inference_time(data, predict):
    """Calculate the average inference time per input"""

    inputs = np.array(data, dtype="str")

    start_time = time.time()
    for x in inputs:
        predict(x)

    inference_time = (time.time() - start_time) / len(inputs)
    return inference_time


def test_onnx_inference_time(X_sample, onnx_sess):
    """Test the inference time for the ONNX model"""

    def predict(x):
        return onnx_sess.run(None, {"inputs": [x]})

    inference_time = get_inference_time(X_sample, predict)
    assert inference_time < MAX_INFERENCE_TIME, (
        f"Inference time ({inference_time:.4f} seconds) for ONNX model "
        f"exceeds the maximum allowed time ({MAX_INFERENCE_TIME} seconds)"
    )


def test_pickle_inference_time(X_sample, pkl_model):
    """Test the inference time for the Pickle model"""

    def predict(x):
        return pkl_model.predict([x])

    inference_time = get_inference_time(X_sample, predict)
    assert inference_time < MAX_INFERENCE_TIME, (
        f"Inference time ({inference_time:.4f} seconds) for pickle model "
        f"exceeds the maximum allowed time ({MAX_INFERENCE_TIME} seconds)"
    )


def test_equivalence(X_sample, pkl_model, onnx_sess):
    """Test equivalence between Pickle and ONNX models"""

    pickle_proba = pkl_model.predict_proba(X_sample)
    onnx_proba = onnx_sess.run(None, {"inputs": X_sample})[1]

    # Assert that the probabilities are close within specified tolerances
    np.testing.assert_allclose(pickle_proba, onnx_proba, rtol=1e-2, atol=1e-2)
