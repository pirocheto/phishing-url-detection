"""
Module for utiliy functions.
"""

# pylint: disable=import-outside-toplevel


def create_model(params: dict | None = None) -> any:
    """
    Create a machine learning model pipeline,
    using TfidfVectorizer as preprecessing and LinearSVM as classifier.
    """

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.svm import LinearSVC

    tfidf = FeatureUnion(
        [
            ("word", TfidfVectorizer()),
            ("char", TfidfVectorizer(analyzer="char")),
        ]
    )

    classifier = CalibratedClassifierCV(
        LinearSVC(dual="auto"),
        cv=5,
        method="isotonic",
        ensemble=False,
    )

    pipeline = Pipeline([("tfidf", tfidf), ("cls", classifier)])

    if params:  # pragma: no cover
        pipeline.set_params(**params)
    return pipeline


def load_data(path: str):
    """Load data from a file."""
    import pandas as pd

    df_train = pd.read_parquet(path)
    df_train = df_train.replace({"status": {"legitimate": 0, "phishing": 1}})
    X_train = df_train["url"].values
    y_train = df_train["status"].values

    return X_train, y_train


def score_model(model, X_train, y_train, train_score=False) -> dict:
    """Score a machine learning model using cross-validation."""
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.model_selection import cross_validate

    # Ignore this warnings to don't flood the terminal
    # Doesn't work for n_jobs > 1
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    scoring = ["recall", "precision", "f1", "accuracy", "roc_auc"]
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        n_jobs=1,
        return_train_score=train_score,
        scoring=scoring,
    )

    return scores


def save_model(model: any, path: str) -> str:
    """Save a machine learning model to a file."""
    import pickle
    from pathlib import Path

    # Create a directory to save the model
    model_path = Path(path)
    model_path.parent.mkdir(exist_ok=True, parents=True)

    # Save the model to a pickle file
    model_path.write_bytes(pickle.dumps(model))
    return str(model_path)


def load_model(path: str, model_format="pickle"):
    """Load a machine learning model from a file."""
    from pathlib import Path

    model_path = Path(path)

    if model_format == "pickle":
        import pickle

        model = pickle.loads(model_path.read_bytes())

    if model_format == "onnx":
        import onnxruntime

        model = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
    return model
