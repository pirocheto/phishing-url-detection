import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def normalize_values(values):
    min_value = min(values)
    max_value = max(values)

    new_min = 0
    new_max = 1

    normalized_values = [
        (x - min_value) / (max_value - min_value) * (new_max - new_min) + new_min
        for x in values
    ]
    return normalized_values


def plot(feature_importance, ax):
    feature_importance = dict(
        sorted(feature_importance.items(), key=lambda feat: feat[1])
    )

    feature_names = list(feature_importance.keys())
    feature_values = list(feature_importance.values())

    feature_values = normalize_values(feature_values)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors = colors[0 : 2 + 1][::-1]
    n = 256
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n)
    colors = [cmap(x) for x in np.linspace(0, 1, len(feature_importance))]

    ax.barh(feature_names, feature_values, color=colors, height=1, alpha=0.5)
    ax.set_title("Feature Importance", weight="bold", fontsize=11)

    ax.spines["top"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.yaxis.set_ticks_position("right")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.margins(y=0)
    ax.set_xlim(0, 1)

    ax.grid(False)
    ax.axis("off")

    for i, label in enumerate(feature_names):
        ax.text(
            0.08,
            i,
            label,
            size=8,
            ha="left",
            va="center",
            # color="black",
        )

    # ax.text(
    #     0,
    #     len(feature_importance) + 0.5,
    #     "Feature importance",
    #     size=11,
    #     ha="center",
    #     va="bottom",
    #     weight="bold",
    # )
    # for label in ax.get_yticklabels():
    #     label.set_fontsize(8)

    # ax.set_xticks([])
