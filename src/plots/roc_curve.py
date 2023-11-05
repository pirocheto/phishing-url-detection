import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(y_true, y_scores, pos_label, ax):
    fpr, tpr, thresholds = roc_curve(
        y_true, y_scores, pos_label=pos_label, drop_intermediate=False
    )

    roc_auc = auc(fpr, tpr)
    # best_threshold_index = np.argmin(np.abs(fpr + tpr - 1))
    # best_threshold = thresholds[best_threshold_index]

    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    ax.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC Curve \n(auc = {roc_auc:.3f})",
    )
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.plot(
        [0, 1],
        [0, 1],
        lw=2,
        linestyle="dotted",
        alpha=0.3,
    )
    ax.fill_between(fpr, tpr, alpha=0.2)

    ax.scatter(
        fpr[best_threshold_index],
        tpr[best_threshold_index],
        color=color,
    )

    ax.annotate(
        "Best Threshold\n" + r"$\bf{" + f"{best_threshold:.3f}" + "}$",
        xy=(
            fpr[best_threshold_index],
            tpr[best_threshold_index],
        ),
        xytext=(0.5, -0.5),
        textcoords="offset fontsize",
        va="top",
        # ha="left",
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(
        "Receiver Operating Characteristic (ROC)", fontweight="bold", fontsize=10
    )

    ax.text(
        0.8,
        0.15,
        "AUC\n" + r"$\bf{" + f"{roc_auc:.3f}" + "}$",
        va="center",
        ha="center",
        # fontsize="12",
        bbox=dict(
            boxstyle="square",
            fc=ax.get_facecolor(),
            ec="grey",
            alpha=0.4,
            pad=0.5,
        ),
    )
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.spines["left"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    ax.grid(True)
    ax.set_box_aspect(1)
