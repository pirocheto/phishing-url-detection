from datetime import datetime

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygit2
import yaml

from plots.advantages import plot as plot_advantages
from plots.confusion_matrix import plot as plot_confusion_matrix
from plots.data import plot as plot_data
from plots.data_proportion import plot as plot_data_proportion
from plots.details import plot as plot_details
from plots.feature_importance import plot as plot_feature_importance
from plots.intended_use import plot as plot_intended_use
from plots.limitations import plot as plot_limitations
from plots.metrics import plot as plot_metrics
from plots.roc_curve import plot as plot_roc_curve
from plots.score_distribution import plot as plot_score_distribution

params = dvc.api.params_show()

plt.style.use(params["model_card_style"]["style"])
plt.rcParams["font.sans-serif"] = params["model_card_style"]["font"]


fig = plt.figure(figsize=(8.27, 11.7))

subplots = fig.subplot_mosaic(
    [
        ["details", "intended_use", "metrics"],
        ["feature_importance", "intended_use", "confusion_matrix"],
        ["feature_importance", "data", "roc_curve"],
        ["feature_importance", "data_proportion", "score_distribution"],
        ["feature_importance", "advantages", "limitations"],
    ]
)

with open(params["metrics"], "r", encoding="utf8") as fp:
    metrics = yaml.safe_load(fp)

ndigits = 3
metrics = [
    ("Accuracy", round(metrics["accuracy"], ndigits)),
    ("Precision", round(metrics["precision"], ndigits)),
    ("Recall", round(metrics["recall"], ndigits)),
    ("F1-Score", round(metrics["f1"], ndigits)),
]

with open(params["feature_importance"], "r", encoding="utf8") as fp:
    feature_importance = yaml.safe_load(fp)

target = params["column_mapping"]["target"]
prediction = params["column_mapping"]["prediction"]

df_train = pd.read_csv(params["data_train_pred"], index_col="url")
df_test = pd.read_csv(params["data_test_pred"], index_col="url")

y_scores = df_test[prediction]
y_pred = df_test[f"{prediction}_label"]
y_true = df_test[target]

pos_label = params["column_mapping"]["pos_label"]

labels = df_train[target].value_counts().index
data_proportion = np.transpose(
    [
        df_train[target].value_counts().values,
        df_test[target].value_counts().values,
    ]
)

date = datetime.now()
algo_name = params["classifier"]["_target_"].split(".")[-1]
repo = pygit2.Repository(".")
last_commit_hash = str(repo.head.target)
details = [
    ("Author", "Pirocheto"),
    ("Date", date.strftime("%Y/%m/%d")),
    ("Time", date.now().strftime("%H:%M:%S")),
    ("Model Version", last_commit_hash[:8]),
    ("Algorithm", algo_name),
    ("License", "Apache 2.0"),
]

advantages = [
    "Very fast",
    "Very light",
    "Consumes few resources",
    "Scikit-learn API",
    "Easy to serve",
    "Simple to reproduce",
    "Probability easy to interpret",
]

limitations = [
    "Input are features not URL",
    "May be less efficient than more sophisticated architectures",
    "May make more errors on more modern sites due to old training data",
]

data_pres = (
    "The data is described here:  https://arxiv.org/pdf/1810.03993.pdf.\n"
    "They were randomly split into 70% for training and 30% for testing in a stratified "
    "fashion as illustrated in the graph below."
)

intented_uses = (
    "Provides the probability that a website is a phishing site. "
    "It is necessary to extract the website's features before sending them to the model. "
    "It can be used as the primary engine of an AI-based anti-phishing system or "
    "as an additional criterion in an existing anti-phishing system."
)

intented_users = (
    "This is a demonstration model not intended for production and without end users. "
    "However, anyone can use this model at their own risk while complying with the terms of the license."
)

plot_metrics(metrics, subplots["metrics"])
plot_intended_use(intented_uses, intented_users, subplots["intended_use"])
plot_data_proportion(data_proportion, labels, subplots["data_proportion"])
plot_details(details, subplots["details"])
plot_data(data_pres, subplots["data"])
plot_roc_curve(y_scores, y_true, pos_label, subplots["roc_curve"])
plot_score_distribution(y_scores, y_true, labels, subplots["score_distribution"])
plot_advantages(advantages, subplots["advantages"])
plot_limitations(limitations, subplots["limitations"])
plot_confusion_matrix(y_pred, y_true, labels, subplots["confusion_matrix"])
plot_feature_importance(feature_importance, subplots["feature_importance"])

prop_cycle = plt.rcParams["axes.prop_cycle"]
primary_color = prop_cycle.by_key()["color"][0]

fig.text(
    s="Model Card",
    x=0.02,
    y=0.975,
    fontweight="bold",
    fontsize="14",
    ha="left",
    va="bottom",
    alpha=0.4,
    style="italic",
)

fig.suptitle(
    "Phishing Detection",
    x=0.98,
    y=0.975,
    fontsize="14",
    fontweight="bold",
    ha="right",
    va="bottom",
    style="italic",
    # alpha=0.4,
    color=primary_color,
)

fig.text(
    x=0,
    y=0.965,
    s="_" * 120,
    va="bottom",
    fontweight="bold",
    alpha=0.4,
)

fig.text(
    x=0,
    y=0.0,
    s=" " * 300,
    fontsize=11,
    va="bottom",
    color=plt.rcParams["figure.facecolor"],
    backgroundcolor=primary_color,
    # alpha=0.7,
)

fig.text(
    x=0.01,
    y=0.0,
    s="Created by @pirocheto",
    fontsize=11,
    va="bottom",
    ha="left",
    color=plt.rcParams["figure.facecolor"],
    fontweight="bold",
)
fig.text(
    x=0.99,
    y=0.0,
    s="For more details see: https://github.com/pirocheto/phishing-detection",
    fontsize=11,
    va="bottom",
    ha="right",
    color=plt.rcParams["figure.facecolor"],
    backgroundcolor=primary_color,
)

fig.subplots_adjust(
    top=0.96, left=0.02, right=0.98, bottom=0.04, wspace=0.15, hspace=0.3
)

plt.savefig(
    params["model_card"],
    dpi=300,
)
