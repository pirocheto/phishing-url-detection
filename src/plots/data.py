from textwrap import fill


def wrap_text(text):
    return fill(text, width=29, replace_whitespace=False)


def plot(text, ax):
    x0, y0 = 0.05, 0.975

    title = ax.text(
        x0,
        y0,
        "Data",
        size=11,
        va="top",
        weight="bold",
    )

    ax.annotate(
        wrap_text(text),
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=title,
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
