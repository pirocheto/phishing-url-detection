import json
from pathlib import Path

import dvc.api
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from helper import load_data, load_model


def evaluate():
    params = dvc.api.params_show()
    model = load_model("model")

    X_test, y_test = load_data(params["data"])
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    scores = {}
    scores["roc_auc"] = roc_auc_score(y_test, y_score[:, 1])

    for name, metric in [
        ("accuracy", accuracy_score),
        ("f1", f1_score),
        ("precision", precision_score),
        ("recall", recall_score),
    ]:
        scores[name] = metric(y_test, y_pred)

    Path("metrics.json").write_text(json.dumps(scores, indent=4))


if __name__ == "__main__":
    evaluate()
