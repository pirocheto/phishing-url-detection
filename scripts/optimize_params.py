import logging
import random
import subprocess

import dvc.api
import matplotlib.pyplot as plt
import optuna

logger = logging.getLogger(__name__)


def generate_experiment_name():
    p1 = random.randint(100, 999)
    p2 = random.randint(100, 999)
    p3 = random.randint(100, 999)
    return f"exp-{p1}.{p2}.{p3}"


def objective(trial):
    exp_name = generate_experiment_name()

    classifier = trial.suggest_categorical(
        "classifier",
        choices=[
            "GaussianNB",
            "LinearSVC",
            "LogisticRegression",
            "RandomForest",
        ],
    )

    feature_selection = trial.suggest_categorical(
        "feature_selection",
        choices=[
            "UnivariateFeatureSelection",
            "SelectFromModel",
        ],
    )

    max_features = trial.suggest_int("max_features", 5, 50)

    if feature_selection == "SelectFromModel":
        run_exp = (
            f"dvc exp run -n {exp_name} "
            f"--temp "
            f"-S classifier={classifier} "
            f"-S feature_selection={feature_selection} "
            f"-S feature_selection.max_features={max_features}"
        )

    if feature_selection == "UnivariateFeatureSelection":
        run_exp = (
            f"dvc exp run -n {exp_name} "
            f"--temp "
            f"-S classifier={classifier} "
            f"-S feature_selection={feature_selection} "
            f"-S feature_selection.param={max_features}"
        )

    try:
        subprocess.run(run_exp, shell=True, check=True)
    except subprocess.CalledProcessError as error:
        logger.error(error)
        return float("-inf")

    metric = dvc.api.metrics_show(rev=exp_name)["f1-score"]
    return metric


N_TRIALS = 10
N_JOBS = 5


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS)
# print(study.best_params)
# print(study.best_value)

optuna.visualization.matplotlib.plot_timeline(study)
plt.savefig("comparaison/timeline.png")
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.savefig("comparaison/optimization_history.png")
