import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ks_2samp


def plot_cumulative_distribution(y_true, y_scores, labels, ax):
    y_scores = np.array(y_scores)
    scores1 = y_scores[y_true == labels[0]]
    scores2 = y_scores[y_true == labels[1]]
    # ax.ecdf(scores1, label=labels[0])
    # ax.ecdf(scores2, label=labels[1])

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Predicted Probabilities")
    ax.set_ylabel("Cumulative Distribution")
    ax.set_title(
        "Empirical Cumulative Distribution Function (ECDF)",
        fontweight="bold",
        fontsize=10,
    )

    sorted_data1 = np.sort(scores1)
    ecdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
    ax.plot(sorted_data1, ecdf1, label=labels[0])

    sorted_data2 = np.sort(scores2)
    ecdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
    ax.plot(sorted_data2, ecdf2, label=labels[1])

    # y_line = np.linspace(0, 1, len(sorted_data1))

    # ax.fill_between(
    #     sorted_data1,
    #     ecdf1,
    #     alpha=0.2,
    #     interpolate=False,
    #     zorder=-1,
    #     # color="grey",
    #     linewidth=0.0,
    # )
    # ax.fill_between(
    #     sorted_data2,
    #     ecdf2,
    #     # alpha=0.2,
    #     interpolate=False,
    #     zorder=0,
    #     color=plt.rcParams["figure.facecolor"],
    #     linewidth=0.0,
    # )

    ks_statistic, p_value = ks_2samp(scores1, scores2)
    ax.text(
        0.5,
        0.5,
        "KS Statistic:" + r"$\bf{" + f"{ks_statistic:.3f}" + "}$\n"
        "p-value:" + r"$\bf{" + f"{p_value:.3}" + "}$",
        va="center",
        ha="center",
        # fontsize="12",
        bbox=dict(
            boxstyle="square",
            fc=ax.get_facecolor(),
            ec="grey",
            alpha=0.6,
            pad=0.5,
        ),
    )
    # ax.set_axisbelow(False)
    ax.grid(True)
    ax.legend()
    ax.set_box_aspect(1)
