import matplotlib.pyplot as plt


def plot_data_proportion(data, labels, ax):
    bar_width = 0.7

    labels_x = ("Train rows", "Test rows")

    b_0 = ax.bar(
        labels_x,
        data[0],
        label=labels[0],
        width=bar_width,
    )

    b_1 = ax.bar(
        labels_x,
        data[1],
        label=labels[1],
        bottom=data[0],
        width=bar_width,
    )

    ax.bar_label(
        b_0,
        labels=data[0],
        label_type="center",
        color=plt.rcParams["figure.facecolor"],
    )
    ax.bar_label(
        b_1,
        labels=data[1],
        label_type="center",
        color=plt.rcParams["figure.facecolor"],
    )

    ax.tick_params(axis="x", which="both", length=0)
    # ax.xaxis.set_ticks_position("top")
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    # ax.spines["left"].set_visible(False)
    # ax.spines["right"].set_visible(False)
    # ax.spines["top"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    # ax.set_yticks([])
    # ax.set_xticks([])
    ax.legend(
        # loc="upper center",
        framealpha=0.4,
        # fancybox=False,
        # bbox_to_anchor=(0.5, 0),
        # ncols=2,
    )
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    ax.set_box_aspect(1)
