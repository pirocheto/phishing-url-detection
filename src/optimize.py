import logging
import warnings
from pathlib import Path
from typing import Any

import dvc.api
import numpy as np
import optuna
import pandas as pd
from rich.logging import RichHandler
from sklearn.pipeline import Pipeline

from dvclive import Live
from helper import create_model, exp_show, load_data, save_model, score_model

# Initialize the logger
logger = logging.getLogger("optuna")
logger.addHandler(RichHandler())
optuna.logging.disable_default_handler()


# Set the random seed
SEED = 796856567

# Number of trials to perform
N_TRIALS = 3


# Function to get parameters for a trial
def get_params(trial):
    max_ngram_word = trial.suggest_int("max_ngram_word", 1, 3)
    max_ngram_char = trial.suggest_int("max_ngram_char", 1, 5)
    use_idf = trial.suggest_categorical("use_idf", [True, False])

    C = trial.suggest_float("C", 1e-7, 10.0, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])
    tol = trial.suggest_float("tol", 1e-5, 1e-1, log=True)
    lowercase = trial.suggest_categorical("lowercase", [True, False])

    return {
        "tfidf__word__ngram_range": (1, max_ngram_word),
        "tfidf__char__ngram_range": (1, max_ngram_char),
        "tfidf__word__lowercase": lowercase,
        "tfidf__char__lowercase": lowercase,
        "tfidf__word__use_idf": use_idf,
        "tfidf__char__use_idf": use_idf,
        "cls__estimator__C": C,
        "cls__estimator__loss": loss,
        "cls__estimator__tol": tol,
    }


# Class to define the objective of the Optuna study
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

        score = scores["test_f1"]
        score = min(np.mean(score), np.median(score))

        if score > trial.study.best_value:
            model_dir = Path(live.dir) / "model"
            model_path = save_model(model, model_dir)
            live.log_artifact(model_path, type="model")

        return score


def optimize():
    with Live(save_dvc_exp=False, resume=True) as live:
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

    exp_show()


if __name__ == "__main__":
    optimize()
