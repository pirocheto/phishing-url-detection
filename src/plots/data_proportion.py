import matplotlib.pyplot as plt


def plot(data, labels, ax):
    bar_width = 0.7

    labels_x = ("Train rows", "Test rows")
    values_legitimate = data[0]
    values_phishing = data[1]

    b_0 = ax.bar(
        labels_x,
        values_legitimate,
        label=labels[0],
        width=bar_width,
    )

    b_1 = ax.bar(
        labels_x,
        values_phishing,
        label=labels[1],
        bottom=values_legitimate,
        width=bar_width,
    )

    ax.bar_label(
        b_0,
        labels=values_legitimate,
        label_type="center",
        color=plt.rcParams["figure.facecolor"],
    )
    ax.bar_label(
        b_1,
        labels=values_phishing,
        label_type="center",
        color=plt.rcParams["figure.facecolor"],
    )

    ax.tick_params(axis="both", which="both", length=0)
    ax.xaxis.set_ticks_position("top")
    # ax.set_xticklabels([])
    ax.set_yticklabels([])

    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    ax.spines["left"].set_visible(False)
    ax.set_yticks([])
    # ax.set_xticks([])
    ax.legend(
        loc="upper center",
        framealpha=0.4,
        fancybox=False,
        bbox_to_anchor=(0.5, 0),
        ncols=2,
    )
    ax.grid(False)
    ax.set_box_aspect(1)
