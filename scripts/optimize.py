from pprint import pprint

import optuna
import pandas as pd
from dvclive.optuna import DVCLiveCallback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from dvclive import Live

df = pd.read_csv("data/all.csv")

X = df["url"]
y = df["status"]  # Labels correspondants à chaque exemple de texte

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


def objective(trial):
    with Live(exp_name=f"svm-opt-{trial.number+1}") as live:
        C = trial.suggest_float("C", 1e-7, 10.0, log=True)

        live.log_params(trial.params)

        pipeline = make_pipeline(
            make_union(
                TfidfVectorizer(),
                TfidfVectorizer(),
            ),
            LinearSVC(C=C),
        )

        # Entraîner le modèle sur les données d'entraînement
        pipeline.fit(X_train, y_train)

        # Faire des prédictions sur les données de test
        predictions = pipeline.predict(X_test)

        metrics = classification_report(y_test, predictions, output_dict=True)

        live.log_metric("accuracy", metrics["accuracy"])
        for name, value in metrics["phishing"].items():
            live.log_metric(name, value)

    return metrics["phishing"]["f1-score"]


study = optuna.create_study(direction="maximize")

study.optimize(objective, n_trials=4)
