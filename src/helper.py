def create_model(params: dict | None = None) -> any:
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
    )

    pipeline = Pipeline([("tfidf", tfidf), ("cls", classifier)])

    if params and params.get("classifier", None):
        pipeline_params = {"cls__estimator": params["classifier"]}
        pipeline.set_params(**pipeline_params)
        return pipeline

    if params:
        pipeline_params = {
            "tfidf__word__ngram_range": (1, params["max_ngram_word"]),
            "tfidf__char__ngram_range": (1, params["max_ngram_char"]),
            "tfidf__word__lowercase": params["lowercase"],
            "tfidf__char__lowercase": params["lowercase"],
            "tfidf__word__use_idf": params["use_idf"],
            "tfidf__char__use_idf": params["use_idf"],
            "cls__estimator__C": params["C"],
            "cls__estimator__loss": params["loss"],
            "cls__estimator__tol": params["tol"],
        }
        pipeline.set_params(**pipeline_params)
        return pipeline

    return pipeline


def load_data(path: str) -> ("np.array", "np.array"):
    import pandas as pd

    df_train = pd.read_csv(path)
    df_train = df_train.replace({"status": {"legitimate": 0, "phishing": 1}})
    X_train = df_train["url"].values
    y_train = df_train["status"].values

    return X_train, y_train


def score_model(model, X_train, y_train) -> dict:
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
        n_jobs=5,
        return_train_score=True,
        scoring=scoring,
    )

    return scores


def save_model(model: any, dir: str) -> str:
    import pickle
    from pathlib import Path

    # Create a directory to save the model
    model_dir = Path(dir)
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.pkl"

    # Save the model to a pickle file
    model_path.write_bytes(pickle.dumps(model))
    return str(model_path)


def load_model(dir: str):
    import pickle
    from pathlib import Path

    model_dir = Path(dir)
    model_path = model_dir / "model.pkl"

    model = pickle.loads(model_path.read_bytes())
    return model


def pkl2onnx(model_path: str) -> str:
    from pathlib import Path

    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import StringTensorType

    model_path = Path(model_path)
    model_dir = model_path.parent
    onnx_path = model_dir / "model.onnx"

    onx = to_onnx(
        model_path,
        initial_types=[("inputs", StringTensorType((None,)))],
        options={"zipmap": False},
    )

    onnx_path.write_bytes(onx.SerializeToString())
    return onnx_path
