import matplotlib.pyplot as plt


def plot(ax):
    data = [
        ("Author", "Pirocheto"),
        ("Model Date", "2023/03/12"),
        ("Model Version", "1.0.12"),
        ("Algorithm", "Naïve Bayes"),
        ("License", "Apache 2.0"),
    ]

    table = ax.table(
        cellText=data,
        loc="center",
        edges="B",
        cellLoc="left",
    )
    table.scale(1, 1.5)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for i, cell in enumerate(table.get_celld().values()):
        cell.set_facecolor(plt.rcParams["figure.facecolor"])
        cell.set_edgecolor(plt.rcParams["text.color"])
        if i % 2 == 0:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_style("italic")
    ax.axis("off")
