import numpy as np
import seaborn as sns


def plot_score_distribution(y_true, y_scores, labels, ax):
    y_scores = np.array(y_scores)
    scores1 = y_scores[y_true == labels[0]]
    scores2 = y_scores[y_true == labels[1]]

    ax = sns.histplot(
        scores1,
        kde=True,
        ax=ax,
        bins="auto",
        edgecolor=None,
        legend=True,
        label=labels[0],
    )
    ax = sns.histplot(
        scores2,
        kde=True,
        ax=ax,
        bins="auto",
        edgecolor=None,
        legend=True,
        label=labels[1],
    )

    ax.legend(
        # loc="upper center",
        framealpha=0.4,
        # fancybox=False,
        # bbox_to_anchor=(0.5, 0),
        # ncols=2,
    )
    ax.set_box_aspect(1)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")

    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True)
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)
    ax.set_title("Score distribution", fontweight="bold", fontsize=10)
