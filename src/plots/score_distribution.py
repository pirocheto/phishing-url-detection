import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_score_distribution(y_scores, y_true, labels, ax):
    y_scores = np.array(y_scores)
    data1 = y_scores[np.where(y_true == labels[0])]
    data2 = y_scores[np.where(y_true == labels[1])]

    ax = sns.histplot(
        data1,
        kde=True,
        ax=ax,
        edgecolor=None,
        legend=True,
        label=labels[0],
    )
    ax = sns.histplot(
        data2,
        kde=True,
        ax=ax,
        edgecolor=None,
        legend=True,
        label=labels[1],
    )

    ax.legend(
        loc="upper center",
        framealpha=0.4,
        fancybox=False,
        bbox_to_anchor=(0.5, 0),
        ncols=2,
    )
    ax.set_box_aspect(1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(None)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Score distribution", fontweight="bold", fontsize=10)
