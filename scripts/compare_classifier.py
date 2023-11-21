from pprint import pprint

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC

from dvclive import Live

SEED = 796856567
DATA_PATH = "data/data.csv"

df = pd.read_csv("data/all.csv")

X = df["url"]
y = df["status"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


classifiers = [
    ("svm", LinearSVC()),
    ("lr", LogisticRegression()),
    ("knn", KNeighborsClassifier()),
]


for code_name, classifier in classifiers:
    with Live(exp_name=code_name) as live:
        pipeline = make_pipeline(
            make_union(
                TfidfVectorizer(max_features=500),
                TfidfVectorizer(analyzer="char", max_features=500),
            ),
            classifier,
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        metrics = classification_report(y_test, predictions, output_dict=True)
        live.log_metric("accuracy", metrics["accuracy"])
        for name, value in metrics["phishing"].items():
            live.log_metric(name, value)
