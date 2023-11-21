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

# logger = optuna.logging.get_logger("optuna")
# logger.addHandler(logging.StreamHandler(sys.stdout))

optunalog_path = Path("optunalog")
optunalog_path.mkdir(exist_ok=True)

storage = optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage(f"{str(optunalog_path)}/journal.log"),
)

SEED = 796856567
DATA_PATH = "data/data.csv"
N_TRIALS = 10


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


def plot_study(study):
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_param_importances,
        plot_timeline,
    )

    for name, plot in [
        ("param_importances", plot_param_importances),
        ("optimization_history", plot_optimization_history),
        ("timeline", plot_timeline),
    ]:
        plt.Figure()
        plot(study)
        plt.savefig(optunalog_path / f"{name}.png")
        plt.close()


class Objective:
    def __init__(self, X_train, X_test, y_train, y_test) -> None:
        self.X_train = X_train
        self.X_test = X_test

        self.y_train = y_train
        self.y_test = y_test

    def __call__(self, trial) -> Any:
        exp_name = f"svm-opt-{trial.number}"

        with Live(exp_name=exp_name) as live:
            params = get_params(trial)
            live.log_params(trial.params)

            model = create_model(params)
            model.fit(self.X_train, self.y_train)

            model_dir = Path(live.dir) / "model"
            model_dir.mkdir(exist_ok=True)

            model_path = model_dir / "model.pkl"
            model_path.write_bytes(pickle.dumps(model))

            with open(model_dir / "params.yaml", "w") as fp:
                yaml.dump(params, fp)

            live.log_artifact(model_path, type="model", cache=False)

            y_preds = model.predict(self.X_test)

            metrics = classification_report(self.y_test, y_preds, output_dict=True)
            live.log_metric("accuracy", metrics["accuracy"])
            for name, value in metrics["phishing"].items():
                live.log_metric(name, value)

        return metrics["phishing"]["f1-score"]


def main():
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        storage=storage,
        study_name="optimize-model",
        load_if_exists=True,
    )

    objective = Objective(*load_data(DATA_PATH))
    study.optimize(objective, n_trials=N_TRIALS)
    plot_study(study)

    best_trial = study.best_trial
    print("Best trial:", best_trial.number)
    print("score:", best_trial.value)
    print("params:")
    pprint(best_trial.params)


if __name__ == "__main__":
    main()
