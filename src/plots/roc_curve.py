from sklearn.metrics import auc, roc_curve


def plot_roc_curve(y_scores, y_true, pos_label, ax):
    fpr, tpr, _ = roc_curve(
        y_true, y_scores, pos_label=pos_label, drop_intermediate=False
    )

    roc_auc = round(auc(fpr, tpr), 3)

    ax.plot(fpr, tpr, lw=2)
    ax.plot([0, 1], [0, 1], lw=2, linestyle="dotted", alpha=0.3)
    ax.fill_between(fpr, tpr, alpha=0.2)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve", fontweight="bold", fontsize=10)

    ax.text(
        0.5,
        0.5,
        "AUC\n" + r"$\bf{" + str(roc_auc) + "}$",
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_box_aspect(1)
