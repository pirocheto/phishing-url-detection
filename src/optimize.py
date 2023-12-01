"""
Module for hyperparameter optimization.
"""

from functools import partial
from pathlib import Path

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import optuna
import yaml
from optuna.storages import JournalFileStorage, JournalStorage

from helper import create_model, load_data, score_model
from plots import plot_optimization_history


def get_hyperparams(trial: optuna.Trial, space: dict) -> dict:
    """Generate hyperparameters based on the search space."""

    # For TF-IDF
    max_ngram_word = trial.suggest_int("max_ngram_word", **space["max_ngram_word"])
    max_ngram_char = trial.suggest_int("max_ngram_char", **space["max_ngram_char"])
    use_idf = trial.suggest_categorical("use_idf", space["use_idf"])
    lowercase = trial.suggest_categorical("lowercase", space["lowercase"])

    # For SVM
    C = trial.suggest_float("C", **space["C"], log=True)
    tol = trial.suggest_float("tol", **space["tol"], log=True)
    loss = trial.suggest_categorical("loss", space["loss"])

    calibration = trial.suggest_categorical("calibration", space["calibration"])

    # Define hyperparameters dictionary
    hyperparams = {
        "tfidf__word__ngram_range": (1, max_ngram_word),
        "tfidf__char__ngram_range": (1, max_ngram_char),
        "tfidf__word__lowercase": lowercase,
        "tfidf__char__lowercase": lowercase,
        "tfidf__word__use_idf": use_idf,
        "tfidf__char__use_idf": use_idf,
        "cls__method": calibration,
        "cls__estimator__C": C,
        "cls__estimator__loss": loss,
        "cls__estimator__tol": tol,
    }

    return hyperparams


def objective(X_train, y_train, space, trial: optuna.Trial):
    """Objective function for Optuna optimization."""

    # Get hyperparameters from Optuna trial
    hyperparams = get_hyperparams(trial, space)

    # Create a model pipeline with the specified hyperparameters
    model = create_model(hyperparams)

    # Score the model on training data
    scores = score_model(model, X_train, y_train)

    # Calculate mean scores for each metric
    mean_scores = {}
    for metric, values in scores.items():
        mean_scores[metric] = values.mean()

    # Set the user attribute "scores" in the trial
    trial.set_user_attr("scores", mean_scores)

    # Use the ROC AUC score as the optimization objective
    f1_score = scores["test_roc_auc"]
    return min(np.mean(f1_score), np.median(f1_score))


def optimize():
    """Perform hyperparameter optimization using Optuna."""

    params = dvc.api.params_show()
    X_train, y_train = load_data(params["data"]["train"])

    # Create an Optuna study for optimization
    storage = JournalStorage(JournalFileStorage("optuna-journal.log"))
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=4242),
        storage=storage,
    )

    # Partially apply the objective function with training data
    objective_with_data = partial(
        objective,
        X_train,
        y_train,
        params["train"]["hyperparams"],
    )

    # Run the optimization study
    study.optimize(
        objective_with_data,
        n_trials=params["train"]["n_trials"],
        show_progress_bar=True,
    )

    plot_optimization_history(study.trials)
    plt.savefig("live/images/optimization_history.png")

    hyperparams = get_hyperparams(
        study.best_trial,
        params["train"]["hyperparams"],
    )

    best_hyperparams_path = Path("live/hyperparams.yaml")
    best_hyperparams_path.write_text(yaml.dump(hyperparams), "utf8")


if __name__ == "__main__":
    optimize()
