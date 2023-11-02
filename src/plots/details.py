import matplotlib.pyplot as plt


def plot(data, ax):
    table = ax.table(
        cellText=data,
        loc="center",
        edges="B",
        cellLoc="left",
    )
    table.scale(1, 1.2)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for i, cell in enumerate(table.get_celld().values()):
        cell.set_facecolor(plt.rcParams["figure.facecolor"])
        cell.set_edgecolor(plt.rcParams["text.color"])
        if i % 2 == 0:
            cell.get_text().set_ha("left")
        else:
            cell.get_text().set_style("italic")
    ax.axis("off")
