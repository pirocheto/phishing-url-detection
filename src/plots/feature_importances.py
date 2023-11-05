import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def plot_feature_importances(
    feature_importance, ax, sort=True, title="Feature Importances"
):
    if sort:
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda feat: feat[1])
        )

    feature_names = list(feature_importance.keys())
    feature_values = list(feature_importance.values())

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors = colors[0 : 2 + 1][::-1]
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=len(feature_importance)
    )
    colors = [cmap(x) for x in np.linspace(0, 1, len(feature_importance))]

    ax.barh(
        feature_names,
        feature_values,
        color=colors,
        height=1,
        alpha=0.5,
    )

    ax.set_title(title, weight="bold", fontsize=11)

    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.set_yticks([])
    ax.set_xticks([0])
    ax.margins(0.05, 0.025)
    ax.tick_params(axis="both", which="both", length=0)

    ax.grid(True)

    for i, label in enumerate(feature_names):
        ax.annotate(
            label,
            xy=(0.5, i),
            xycoords=("axes fraction", "data"),
            size=8,
            ha="center",
            va="center",
            # color="black",
        )
