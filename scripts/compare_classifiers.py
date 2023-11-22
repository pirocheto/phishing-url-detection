import dvc.api
import numpy as np
import pandas as pd
from rich.pretty import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from dvclive import Live

SEED = 796856567

DATA_PATH = "data/data.csv"


# Function to load data
def load_data(path):
    df_train = pd.read_csv(path)
    df_train = df_train.replace({"status": {"legitimate": 0, "phishing": 1}})
    X_train = df_train["url"].values
    y_train = df_train["status"].values

    return X_train, y_train


classifiers = [
    ("svm", LinearSVC(dual="auto")),
    ("lr", LogisticRegression()),
    ("knn", KNeighborsClassifier()),
    ("nb", MultinomialNB()),
]


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
        ],
    )
    df = df.dropna(subset=["Experiment"])
    df = df.set_index("Experiment")
    df = df.sort_values("f1", ascending=False)
    df = df.head(n)
    pprint(df)


def main():
    for exp_name, classifier in classifiers:
        print(f"Experiment '{exp_name}' in progress...")
        with Live(exp_name=exp_name) as live:
            tfidf = FeatureUnion(
                [
                    ("word", TfidfVectorizer()),
                    ("char", TfidfVectorizer(analyzer="char")),
                ]
            )

            model = Pipeline([("tfidf", tfidf), ("cls", classifier)])

            X_train, y_train = load_data(DATA_PATH)

            scores = cross_validate(
                model,
                X_train,
                y_train,
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
                    name = name.replace("test_", "")
                    value = np.mean(values)
                    live.log_metric(name, value)

    print_best_exps()


if __name__ == "__main__":
    main()
