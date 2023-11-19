import pickle
from functools import lru_cache
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from dvclive import Live

SEED = 796856567
DATA_PATH = "data/data.csv"
N_TRIALS = 1


@lru_cache(maxsize=None)
def load_data():
    df = pd.read_csv(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        df["url"],
        df["status"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["status"],
    )
    return X_train, X_test, y_train, y_test


def create_model(params):
    tfidf = FeatureUnion(
        [
            ("1", TfidfVectorizer()),
            ("2", TfidfVectorizer(analyzer="char")),
        ]
    )

    classifier = CalibratedClassifierCV(
        LinearSVC(dual="auto"),
        cv=5,
        method="isotonic",
    )

    pipeline = Pipeline(
        [
            ("tfidf", tfidf),
            ("cls", classifier),
        ]
    )

    pipeline.set_params(**params)
    return pipeline


def get_params(trial):
    # TF-IDF
    max_ngram_1 = trial.suggest_int("max_ngram_1", 1, 5)
    max_ngram_2 = trial.suggest_int("max_ngram_2", 1, 5)

    # SVM
    C = trial.suggest_float("C", 1e-7, 10.0, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])

    return {
        "tfidf__1__ngram_range": (1, max_ngram_1),
        "tfidf__2__ngram_range": (1, max_ngram_2),
        "cls__base_estimator__C": C,
        "cls__base_estimator__loss": loss,
    }


def optuna_plots(study):
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_timeline,
    )

    optuna_plots_path = Path("optuna_plots")
    optuna_plots_path.mkdir(exist_ok=True)

    for name, plot in [
        ("param_importances", plot_param_importances),
        ("optimization_history", plot_optimization_history),
        ("timeline", plot_timeline),
    ]:
        plt.Figure()
        plot(study)
        plt.savefig(optuna_plots_path / f"{name}.png")
        plt.close()


def objective(trial):
    exp_name = f"svm-opt-{trial.number+1}"
    X_train, X_test, y_train, y_test = load_data()

    with Live(exp_name=exp_name) as live:
        params = get_params(trial)
        live.log_params(trial.params)

        model = create_model(params)
        model.fit(X_train, y_train)

        model_path = Path(live.dir) / "model.pkl"
        model_path.write_bytes(pickle.dumps(model))
        live.log_artifact(model_path, type="model", cache=False)

        predictions = model.predict(X_test)

        metrics = classification_report(y_test, predictions, output_dict=True)
        live.log_metric("accuracy", metrics["accuracy"])
        for name, value in metrics["phishing"].items():
            live.log_metric(name, value)

    return metrics["phishing"]["f1-score"]


if __name__ == "__main__":
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )

    study.optimize(objective, n_trials=N_TRIALS)

    best_trial = study.best_trial

    print("Best trial:")
    print("score:", best_trial.value)
    print("params:")
    pprint(best_trial.params)
