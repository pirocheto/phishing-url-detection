import pickle
import tempfile
from functools import lru_cache
from pathlib import Path
from pprint import pprint

import numpy as np
import optuna
import pandas as pd
from dvclive.optuna import DVCLiveCallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.svm import LinearSVC

from dvclive import Live

seed = 796856567


def cached(transformer):
    class CachedTransformer(transformer):
        @lru_cache(maxsize=None)
        def _fit_cached(self, raw_documents, y=None):
            raw_documents = np.array(raw_documents)
            return super().fit(raw_documents, y)

        def fit(self, raw_documents, y=None):
            raw_documents = tuple(raw_documents)
            y = y if y is None else tuple(y)
            return self._fit_cached(raw_documents, y)

        @lru_cache(maxsize=None)
        def _transform_cached(self, raw_documents):
            raw_documents = np.array(raw_documents)
            return super().transform(raw_documents)

        def transform(self, raw_documents):
            raw_documents = tuple(raw_documents)
            return self._transform_cached(raw_documents)

        @lru_cache(maxsize=None)
        def _fit_transform_cached(self, raw_documents, y=None):
            raw_documents = np.array(raw_documents)
            return super().fit_transform(raw_documents, y)

        def fit_transform(self, raw_documents, y=None):
            raw_documents = tuple(raw_documents)
            y = y if y is None else tuple(y)
            return self._fit_transform_cached(raw_documents, y)

    wrapper_class = type(f"Cached{transformer.__name__}", (CachedTransformer,), {})
    return wrapper_class


df = pd.read_csv("data/all.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df["url"],
    df["status"],
    test_size=0.2,
    random_state=seed,
    stratify=df["status"],
)

tfidf = FeatureUnion(
    [
        ("1", cached(TfidfVectorizer)()),
        ("2", cached(TfidfVectorizer)(analyzer="char")),
    ]
)

classifier = LinearSVC(dual="auto")

pipeline = Pipeline(
    [
        ("tfidf", tfidf),
        ("svm", classifier),
    ]
)


def get_params(trial):
    # TF-IDF
    max_ngram_1 = trial.suggest_int("max_ngram_1", 1, 5)
    max_ngram_2 = trial.suggest_int("max_ngram_2", 1, 5)

    # SVM
    C = trial.suggest_float("C", 1e-7, 10.0, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])

    return {
        "tfidf__1__ngram_range": (0, max_ngram_1),
        "tfidf__2__ngram_range": (0, max_ngram_2),
        "svm__C": C,
        "svm__loss": loss,
    }


def objective(trial):
    exp_name = f"svm-opt-{trial.number+1}"

    with Live(exp_name=exp_name) as live:
        params = get_params(trial)

        live.log_params(params)

        model = pipeline.set_params(**params)
        model.fit(X_train, y_train)

        model_path = Path(live.dir) / "model.pkl"
        model_path.write_bytes(pickle.dumps(model))

        model.fit(X_train, y_train)

        live.log_artifact(model_path, type="model", cache=False)

        predictions = model.predict(X_test)

        metrics = classification_report(y_test, predictions, output_dict=True)
        live.log_metric("accuracy", metrics["accuracy"])
        for name, value in metrics["phishing"].items():
            live.log_metric(name, value)

    return metrics["phishing"]["f1-score"]


study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
)

study.optimize(objective, n_trials=30)


best_trial = study.best_trial

# Afficher les informations sur le meilleur essai
print("Meilleur essai:")
print("Valeur de l'objectif (score) : ", best_trial.value)
pprint(best_trial.params)

import matplotlib.pyplot as plt

optuna_plots_path = Path("optuna-plots")
optuna_plots_path.mkdir(exist_ok=True)

for name, plot in [
    (
        "param_importances",
        optuna.visualization.matplotlib.plot_param_importances,
    ),
    (
        "optimization_history",
        optuna.visualization.matplotlib.plot_optimization_history,
    ),
    (
        "timeline",
        optuna.visualization.matplotlib.plot_timeline,
    ),
]:
    plt.Figure()
    plot(study)
    plt.savefig(optuna_plots_path / f"{name}.png")
    plt.close()
