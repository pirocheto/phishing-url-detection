from textwrap import fill


def wrap_text(text):
    return fill(text, width=30)


def plot(intented_uses, intended_users, ax):
    x0, y0 = 0.05, 0.975

    text = ax.text(
        x0,
        y0,
        "Intended uses",
        size=11,
        va="top",
        weight="bold",
    )

    text = ax.annotate(
        wrap_text(intented_uses),
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=text,
        textcoords="offset fontsize",
        size=10,
        va="top",
    )

    text = ax.annotate(
        "Intended users",
        xy=(0, 0),
        xytext=(0, -1),
        xycoords=text,
        textcoords="offset fontsize",
        size=11,
        va="top",
        weight="bold",
    )

    text = ax.annotate(
        wrap_text(intended_users),
        xy=(0, 0),
        xytext=(0, -1),
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
    # ax.axis("off")
