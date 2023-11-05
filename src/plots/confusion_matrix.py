import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_pred, y_true, labels, ax):
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    colors = colors[0 : 1 + 1]
    n = 2
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax,
        cmap=cmap,
        linecolor=plt.rcParams["figure.facecolor"],
        linewidths="1",
        vmin=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        robust=True,
        square=True,
    )

    # ax.set_box_aspect(1)

    ax.tick_params(axis="both", which="both", length=0)
    # ax.xaxis.set_label_position("top")
    ax.set_xlabel("Pred Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=10)
