"""
Module to evaluate a machine learning model and generate evaluation metrics and plots.

Metrics:
- ROC AUC
- Accuracy
- F1 Score
- Precision
- Recall

Plots:
- Confusion Matrix
- Calibration Curve
- Precision-Recall Curve
- ROC Curve
- Score Distribution
"""

import json
from pathlib import Path

import dvc.api
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from helper import load_data, load_pickle_model
from plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_distribution,
)


def evaluate():
    """
    Evaluate a machine learning model, generate evaluation metrics, and create plots.
    """
    params = dvc.api.params_show()
    model = load_pickle_model(params["model"]["pickle"])

    X_test, y_test = load_data(params["data"]["test"])
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    scores = {
        "roc_auc": roc_auc_score(y_test, y_score),
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    metrics_path = Path("live/metrics.json")
    metrics_path.parent.mkdir(exist_ok=True, parents=True)
    metrics_path.write_text(json.dumps(scores, indent=4), "utf8")

    images_path = Path("live/images")
    images_path.mkdir(exist_ok=True, parents=True)

    plot_confusion_matrix(y_test, y_pred)
    plt.savefig(images_path / "confusion_matrix.png")

    for path, plot_func in [
        ("calibration_curve.png", plot_calibration_curve),
        ("precision_recall_curve.png", plot_precision_recall_curve),
        ("roc_curve.png", plot_roc_curve),
        ("score_distribution.png", plot_score_distribution),
    ]:
        plt.figure()
        plot_func(y_test, y_score)
        plt.savefig(images_path / path)


if __name__ == "__main__":
    evaluate()
