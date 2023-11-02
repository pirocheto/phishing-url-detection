import matplotlib.pyplot as plt


def plot(metrics: list[tuple[str, float]], ax):
    table = ax.table(cellText=metrics, loc="center", edges="horizontal")

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
    table.scale(1, 1.7)
    ax.axis("off")

    ax.text(
        -0.08,
        0.5,
        "Performances",
        fontsize=12,
        va="center",
        ha="center",
        rotation=90,
    )
