import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_precision_recall_curve(y_true, y_scores, pos_label, ax):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color = colors[1]

    precision, recall, threshold = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label
    )
    fscore = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = fscore.argmax()
    best_threshold = threshold[best_threshold_index]
    average_precision = average_precision_score(y_true, y_scores, pos_label=pos_label)

    ax.step(recall, precision, where="post", color=color)
    ax.fill_between(recall, precision, step="post", alpha=0.2, color=color)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=10)

    ax.text(
        0.20,
        0.15,
        "AP\n" + r"$\bf{" + f"{average_precision:.3f}" + "}$",
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

    ax.scatter(
        recall[best_threshold_index],
        precision[best_threshold_index],
        color=color,
    )

    ax.annotate(
        "Best Threshold\n" + r"$\bf{" + f"{best_threshold:.3f}" + "}$",
        xy=(
            recall[best_threshold_index],
            precision[best_threshold_index],
        ),
        xytext=(-0.5, -0.5),
        textcoords="offset fontsize",
        va="top",
        ha="right",
    )

    ax.grid(True)
    ax.set_box_aspect(1)
    return ax
