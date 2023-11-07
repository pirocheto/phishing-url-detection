import numpy as np
import seaborn as sns


def plot_correlation_matrix(x, y, color, size, ax):
    # Mapping from column names to integer coordinates
    # x_labels = [v for v in x.unique()]
    # y_labels = [v for v in y.unique()]
    # x_to_num = {p[1]:p[0] for p in enumerate(x_labels)}
    # y_to_num = {p[1]:p[0] for p in enumerate(y_labels)}

    size_scale = 400

    n_colors = 256
    palette = sns.diverging_palette(20, 220, n=n_colors)
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    ax.scatter(
        x=x,
        y=y,
        s=size * size_scale,
        c=[value_to_color(v) for v in color],
        marker="o",
    )

    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")
    ax.yaxis.set_ticks_position("right")
    ax.tick_params(axis="both", labelsize=10, which="both", length=0)

    ax.grid(False, "major")
    ax.grid(True, "minor")
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    # ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    # ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_xlim(-0.5, len(np.unique(x)) - 0.5)
    ax.set_ylim(-0.5, len(np.unique(y)) - 0.5)
    ax.set_title("Correlation Matrix", fontweight="bold", fontsize=10)
    ax.set_box_aspect(1)
