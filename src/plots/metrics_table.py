import matplotlib.pyplot as plt


def plot_metrics_table(metrics, ax):
    metrics = [(name, f"{value:.3f}") for name, value in metrics]
    table = ax.table(cellText=metrics, loc="upper center", edges="horizontal")

    for i, cell in enumerate(table.get_celld().values()):
        cell.set_facecolor(plt.rcParams["figure.facecolor"])
        cell.set_edgecolor(plt.rcParams["text.color"])
        if i % 2 == 0:
            cell.get_text().set_ha("right")
        else:
            cell.get_text().set_ha("center")
            cell.get_text().set_fontweight("bold")

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    ax.axis("off")
    ax.set_title("Metrics", fontweight="bold", fontsize=10)

    # ax.text(
    #     -0.08,
    #     0.5,
    #     "Performances",
    #     fontsize=12,
    #     va="center",
    #     ha="center",
    #     rotation=90,
    # )
