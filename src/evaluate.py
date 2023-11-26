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

from helper import load_data, load_model
from plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_score_distribution,
)


def evaluate():
    params = dvc.api.params_show()
    model = load_model("model")

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

    Path("metrics.json").write_text(json.dumps(scores, indent=4))

    images_path = Path("images")
    images_path.mkdir(exist_ok=True)

    plot_confusion_matrix(y_test, y_pred)
    plt.savefig(images_path / "confusion_matrix.png")
    plt.close()

    plot_calibration_curve(y_test, y_score)
    plt.savefig(images_path / "calibration_curve.png")
    plt.close()

    plot_precision_recall_curve(y_test, y_score)
    plt.savefig(images_path / "precision_recall_curve.png")
    plt.close()

    plot_roc_curve(y_test, y_score)
    plt.savefig(images_path / "roc_curve.png")
    plt.close()

    plot_score_distribution(y_test, y_score)
    plt.savefig(images_path / "score_distribution.png")
    plt.close()


if __name__ == "__main__":
    evaluate()
