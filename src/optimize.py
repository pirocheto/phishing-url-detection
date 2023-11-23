import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.visualization.matplotlib import plot_optimization_history
from rich.logging import RichHandler
from sklearn.pipeline import Pipeline

from dvclive import Live
from helper import create_model, load_data, score_model

# Initialize a colored logger
optuna.logging.disable_default_handler()
logger = logging.getLogger("optuna")
logger.addHandler(RichHandler())

# Constants
N_TRIALS = 2
SEED = 796856567


def get_params(trial: optuna.Trial) -> dict:
    # For TF-IDF
    max_ngram_word = trial.suggest_int("max_ngram_word", 1, 3)
    max_ngram_char = trial.suggest_int("max_ngram_char", 1, 5)
    use_idf = trial.suggest_categorical("use_idf", [True, False])
    lowercase = trial.suggest_categorical("lowercase", [True, False])

    # For
    C = trial.suggest_float("C", 1e-7, 10, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)

    return {
        "max_ngram_word": max_ngram_word,
        "max_ngram_char": max_ngram_char,
        "lowercase": lowercase,
        "use_idf": use_idf,
        "C": C,
        "loss": loss,
        "tol": tol,
    }


class Objective:
    def __init__(self, X_train, y_train, live: Live) -> None:
        self.X_train = X_train
        self.y_train = y_train
        self.live = live

    def __call__(self, trial: optuna.Trial) -> Any:
        live = self.live
        live.step = trial.number

        params = get_params(trial)
        model: Pipeline = create_model(params)

        scores = score_model(model, self.X_train, self.y_train)
        mean_scores = {}
        for metric, values in scores.items():
            value = values.mean()
            live.log_metric(metric, value)
            mean_scores[metric] = values.mean()
        trial.set_user_attr("scores", mean_scores)

        f1_score = scores["test_f1"]
        return min(np.mean(f1_score), np.median(f1_score))


def optimization_history(study):
    fig = plt.figure()
    plot_optimization_history(study)
    return fig


def optimize():
    with Live(save_dvc_exp=False, resume=True, dvcyaml=False) as live:
        optuna_journal = Path(live.dir) / "optunalog/journal.log"
        optuna_journal.parent.mkdir(exist_ok=True)

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(str(optuna_journal)),
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            storage=storage,
            study_name="optimize-model",
            load_if_exists=True,
        )

        X_train, y_train = load_data("data/data.csv")
        objective = Objective(X_train, y_train, live=live)
        study.optimize(objective, n_trials=N_TRIALS)

        live.summary = study.best_trial.user_attrs["scores"]
        live.log_params(study.best_params)
        live.log_artifact(optuna_journal)

    fig = optimization_history(study)
    live.log_image("optimization_history.png", fig)


if __name__ == "__main__":
    optimize()
