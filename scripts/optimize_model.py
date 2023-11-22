import logging
import pickle
import warnings
from pathlib import Path
from typing import Any

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import yaml
from model import create_model
from rich.logging import RichHandler
from rich.pretty import pprint
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder

from dvclive import Live

# Ignore this warnings to don't flood the terminal
# Doesn't work for n_jobs > 1
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Initialize the logger
logger = logging.getLogger("optuna")
logger.addHandler(RichHandler())
optuna.logging.disable_default_handler()

# Create a folder to store Optuna logs
optunalog_path = Path("optunalog")
optunalog_path.mkdir(exist_ok=True)

# Set the random seed
SEED = 796856567
# Path to the data file
DATA_PATH = "data/data.csv"
# Number of trials to perform
N_TRIALS = 3


# Function to load data
def load_data(path):
    df_train = pd.read_csv(path)
    X_train = df_train["url"].values
    y_train = df_train["status"].values
    y_train = LabelEncoder().fit_transform(y_train)

    return X_train, y_train


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
    def __init__(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    def __call__(self, trial) -> Any:
        # Unique name for each experiment based on the trial number
        exp_name = f"svm-opt-{trial.number}"

        # Use dvclive for live experiment tracking
        with Live(exp_name=exp_name) as live:
            trial.set_user_attr("exp_name", live._exp_name)

            # Get parameters to test for this trial
            params = get_params(trial)
            # Log parameters
            live.log_params(trial.params)

            if trial.should_prune():
                raise optuna.TrialPruned()

            # Create the model using the current parameters
            model = create_model(params)

            # Train the model on the training data
            model.fit(self.X_train, self.y_train)

            # Create a directory to save the model
            model_dir = Path(live.dir) / "model"
            model_dir.mkdir(exist_ok=True)

            # Save the model to a pickle file
            model_path = model_dir / "model.pkl"
            model_path.write_bytes(pickle.dumps(model))

            # Save parameters to a YAML file
            with open(model_dir / "params.yaml", "w") as fp:
                yaml.dump(params, fp)

            # Log the model as an artifact using dvclive
            live.log_artifact(model_path, type="model", cache=False)

            scores = cross_validate(
                model,
                self.X_train,
                self.y_train,
                cv=5,
                n_jobs=5,
                scoring=[
                    "recall",
                    "precision",
                    "f1",
                    "accuracy",
                    "roc_auc",
                ],
            )

            for name, values in scores.items():
                if name.startswith("test_"):
                    value = np.mean(values)
                    live.log_metric(name.replace("test_", ""), value)
                    trial.set_user_attr(name, value)

        # The returned value for optimization
        score = scores["test_f1"]
        return min(np.mean(score), np.median(score))


# Function to print the best trial results
def print_best_exps(n=10):
    pd.set_option("display.max_columns", None)

    df = pd.DataFrame(
        dvc.api.exp_show(),
        columns=[
            "Experiment",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "C",
            "loss",
            "tol",
            "lowercase",
            "max_ngram_word",
            "max_ngram_char",
            "use_idf",
        ],
    )
    df = df.dropna(subset=["Experiment"])
    df = df.set_index("Experiment")
    df = df.sort_values("f1", ascending=False)
    df = df.head(n)
    pprint(df)


def main():
    storage = "sqlite:///optunalog/optuna.db"

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=storage,
        study_name="optimize-model",
        load_if_exists=True,
    )

    # Initialize the study objective
    objective = Objective(*load_data(DATA_PATH))
    study.optimize(objective, n_trials=N_TRIALS)

    # Display the results of the best trial and plot visualizations
    print_best_exps()


if __name__ == "__main__":
    main()
