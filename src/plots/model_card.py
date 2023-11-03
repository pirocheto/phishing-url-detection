from datetime import datetime
from textwrap import fill

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygit2
import yaml

from plots.data_proportion import plot_data_proportion
from plots.feature_importances import plot_feature_importance
from plots.roc_curve import plot_roc_curve
from plots.score_distribution import plot_score_distribution
from src.plots.confusion_matrix import plot_confusion_matrix

params = dvc.api.params_show()

plt.style.use(params["model_card_style"]["style"])
plt.rcParams["font.sans-serif"] = params["model_card_style"]["font"]


def plot_intended_use(intented_uses, intended_users, ax):
    def wrap_text(text):
        return fill(text, width=30)

    x0, y0 = 0.05, 0.975

    text = ax.text(
        x0,
        y0,
        "Intended uses",
        size=11,
        va="top",
        weight="bold",
    )

    text = ax.annotate(
        wrap_text(intented_uses),
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=text,
        textcoords="offset fontsize",
        size=10,
        va="top",
    )

    text = ax.annotate(
        "Intended users",
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=text,
        textcoords="offset fontsize",
        size=11,
        va="top",
        weight="bold",
    )

    text = ax.annotate(
        wrap_text(intended_users),
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=text,
        textcoords="offset fontsize",
        size=10,
        va="top",
    )

    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.set_yticks([])
    ax.set_xticks([])
    # ax.axis("off")


def plot_details(data, ax):
    table = ax.table(
        cellText=data,
        loc="center",
        edges="B",
        cellLoc="left",
    )
    table.scale(1, 1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for i, cell in enumerate(table.get_celld().values()):
        cell.set_facecolor(plt.rcParams["figure.facecolor"])
        cell.set_edgecolor(plt.rcParams["text.color"])
        if i % 2 == 0:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_style("italic")
    ax.axis("off")


def plot_advantages(data, ax):
    wrapped_data = [
        fill(line, initial_indent="• ", subsequent_indent="   ", width=28)
        for line in data
    ]

    x0, y0 = 0.1, 0.95
    text = ax.text(
        x0,
        y0,
        wrapped_data[0],
        fontsize=10,
        va="top",
        ha="left",
    )

    for line in wrapped_data[1:]:
        text = ax.annotate(
            line,
            xy=(0, 0),
            xytext=(0, -0.5),
            xycoords=text,
            textcoords="offset fontsize",
            size=10,
            va="top",
        )

    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(
        0,
        0.5,
        "Advantages",
        fontsize=12,
        va="center",
        ha="left",
        rotation=90,
        bbox=dict(facecolor="green", alpha=0.1),
    )


def plot_limitations(data, ax):
    wrapped_data = [
        fill(line, initial_indent="• ", subsequent_indent="   ", width=25)
        for line in data
    ]

    x0, y0 = 0.1, 0.95
    text = ax.text(
        x0,
        y0,
        wrapped_data[0],
        fontsize=10,
        va="top",
        ha="left",
    )

    for line in wrapped_data[1:]:
        text = ax.annotate(
            line,
            xy=(0, 0),
            xytext=(0, -0.6),
            xycoords=text,
            textcoords="offset fontsize",
            size=10,
            va="top",
        )

    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.text(
        0,
        0.5,
        "Limitations",
        fontsize=12,
        va="center",
        ha="left",
        rotation=90,
        bbox=dict(facecolor="red", alpha=0.1),
    )


def plot_data_infos(text, ax):
    def wrap_text(text):
        return fill(text, width=29, replace_whitespace=False)

    x0, y0 = 0.05, 0.975

    title = ax.text(
        x0,
        y0,
        "Data",
        size=11,
        va="top",
        weight="bold",
    )

    ax.annotate(
        wrap_text(text),
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=title,
        textcoords="offset fontsize",
        size=10,
        va="top",
    )

    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.set_yticks([])
    ax.set_xticks([])


def plot_metrics(metrics, ax):
    table = ax.table(cellText=metrics, loc="center", edges="horizontal")

    for i, cell in enumerate(table.get_celld().values()):
        cell.set_facecolor(plt.rcParams["figure.facecolor"])
        cell.set_edgecolor(plt.rcParams["text.color"])
        if i % 2 == 0:
            cell.get_text().set_ha("right")
        else:
            cell.get_text().set_ha("center")
            cell.get_text().set_fontweight("bold")

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    ax.axis("off")

    ax.text(
        -0.08,
        0.5,
        "Performances",
        fontsize=12,
        va="center",
        ha="center",
        rotation=90,
    )


def plot_model_card():
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

    ndigits = 3
    metrics = [
        ("Accuracy", round(metrics["accuracy"], ndigits)),
        ("Precision", round(metrics["precision"], ndigits)),
        ("Recall", round(metrics["recall"], ndigits)),
        ("F1-Score", round(metrics["f1"], ndigits)),
    ]

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
        "However, anyone can use this model at their own risk while complying "
        "with the terms of the license."
    )

    plot_metrics(metrics, subplots["metrics"])
    plot_intended_use(intented_uses, intented_users, subplots["intended_use"])
    plot_data_proportion(data_proportion, labels, subplots["data_proportion"])
    plot_details(details, subplots["details"])
    plot_data_infos(data_pres, subplots["data"])
    plot_roc_curve(y_scores, y_true, pos_label, subplots["roc_curve"])
    plot_score_distribution(y_scores, y_true, labels, subplots["score_distribution"])
    plot_advantages(advantages, subplots["advantages"])
    plot_limitations(limitations, subplots["limitations"])
    plot_confusion_matrix(y_true, y_pred, labels, subplots["confusion_matrix"])
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


if __name__ == "__main__":
    plot_model_card()
