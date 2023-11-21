import logging
import pickle
import sys
from pathlib import Path
from pprint import pprint
from typing import Any

import matplotlib.pyplot as plt
import optuna
import pandas as pd
import yaml
from model import create_model
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from dvclive import Live

# Initialize the logger
logging.getLogger("optuna").addHandler(logging.StreamHandler(sys.stdout))

# Create a folder to store Optuna logs
optunalog_path = Path("optunalog")
optunalog_path.mkdir(exist_ok=True)

# Configure storage for Optuna trials
storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"{str(optunalog_path)}/journal.log"),
)

# Set the random seed
SEED = 796856567
# Path to the data file
DATA_PATH = "data/data.csv"
# Number of trials to perform
N_TRIALS = 1


# Function to load data
def load_data(path):
    df = pd.read_csv(path)

    X_train, X_test, y_train, y_test = train_test_split(
        df["url"],
        df["status"],
        test_size=0.2,
        random_state=SEED,
        stratify=df["status"],
    )
    return X_train, X_test, y_train, y_test


# Function to get parameters for a trial
def get_params(trial):
    max_ngram_1 = trial.suggest_int("max_ngram_1", 1, 5)
    max_ngram_2 = trial.suggest_int("max_ngram_2", 1, 5)

    C = trial.suggest_float("C", 1e-7, 10.0, log=True)
    loss = trial.suggest_categorical("loss", ["hinge", "squared_hinge"])

    return {
        "tfidf__1__ngram_range": (1, max_ngram_1),
        "tfidf__2__ngram_range": (1, max_ngram_2),
        "cls__estimator__C": C,
        "cls__estimator__loss": loss,
    }


# Function to plot various visualizations of the Optuna study
def plot_study(study):
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
    )

    for name, plot in [
        ("param_importances", plot_param_importances),
        ("optimization_history", plot_optimization_history),
    ]:
        plt.Figure()
        plot(study)
        plt.savefig(optunalog_path / f"{name}.png")
        plt.close()


# Class to define the objective of the Optuna study
class Objective:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    # Method called for each Optuna trial
    def __call__(self, trial) -> Any:
        # Unique name for each experiment based on the trial number
        exp_name = f"svm-opt-{trial.number}"

        # Use dvclive for live experiment tracking
        with Live(exp_name=exp_name) as live:
            # Get parameters to test for this trial
            params = get_params(trial)
            # Log parameters
            live.log_params(trial.params)

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

            # Predictions on the test data
            y_preds = model.predict(self.X_test)

            # Calculate classification metrics
            metrics = classification_report(self.y_test, y_preds, output_dict=True)

            # Log accuracy in the live experiment
            live.log_metric("accuracy", metrics["accuracy"])

            # Log metrics specific to the "phishing" class
            for name, value in metrics["phishing"].items():
                live.log_metric(name, value)

        # The returned value is the F1-score metric
        return metrics["phishing"]["f1-score"]


# Function to print the best trial results
def print_best_trial(study):
    best_trial = study.best_trial
    print(f"Best trial: {best_trial.number}")
    print(f"Score: {best_trial.value}")
    print("Params:")
    pprint(best_trial.params)


def main():
    # Create the Optuna study object
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
    print_best_trial(study)
    plot_study(study)


if __name__ == "__main__":
    main()
