from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib import patches


def plot(data, ax):
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
    # ax.set_title("Advantages", weight="bold", size="10")

    # ax.axis("off")
