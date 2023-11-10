import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Get the dvc params
params = dvc.api.params_show(stages="split_data")

# Use a matplotlib style to make more beautiful graphics
plt.style.use(params["plt_style"])


def plot_data_proportion(data, labels, ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Define the width of the bars
    bar_width = 0.7

    # Labels for the x-axis
    labels_x = ("Train rows", "Test rows")

    # Create the first set of bars (data[0])
    b_0 = ax.bar(labels_x, data[0], label=labels[0], width=bar_width)

    # Create the second set of bars (data[1]) stacked on top of the first set
    b_1 = ax.bar(labels_x, data[1], label=labels[1], bottom=data[0], width=bar_width)

    # Add labels on top of the bars
    ax.bar_label(
        b_0, labels=data[0], label_type="center", color=plt.rcParams["figure.facecolor"]
    )
    ax.bar_label(
        b_1, labels=data[1], label_type="center", color=plt.rcParams["figure.facecolor"]
    )

    # Remove tick mark (-) on the x-axis
    ax.tick_params(axis="x", which="both", length=0)

    # Set font weight to "bold" for x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")

    # Display legend with slight transparency
    ax.legend(framealpha=0.4, reverse=True)

    # Configure grid lines
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax


if __name__ == "__main__":
    # 1. Split the data
    df = pd.read_csv(
        params["path"]["data_all"],
        index_col=params["column_mapping"]["id"],
    )

    X_train, X_test = train_test_split(df, **params["train_test_split"])
    X_train.to_csv(params["path"]["data_train_raw"])
    X_test.to_csv(params["path"]["data_test_raw"])

    # 2. Make the plot showing the data proportion
    target = params["column_mapping"]["target"]
    labels = X_train[target].value_counts().index
    data_proportion = np.transpose(
        [
            X_train[target].value_counts().values,
            X_test[target].value_counts().values,
        ]
    )
    plot_data_proportion(data_proportion, labels)
    plt.savefig(params["path"]["data_proportion"])
