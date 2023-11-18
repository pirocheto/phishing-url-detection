import pickle
from typing import Dict

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

# Get the dvc params
params = dvc.api.params_show(stages="model_interpretation")
path_data_selected = params["path"]["data"]["transformed"]

# Use a matplotlib style to make more beautiful graphics
plt.style.use(params["plt_style"])


def plot_feature_importances(
    feature_importances: Dict[str, float],
    sort=True,
    title="Feature Importances",
    ax=None,
):
    # If ax is not provided, create a new axis
    if ax is None:
        ax: plt.axes.Axes = plt.gca()

    # Sort feature importances if required
    if sort:
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda feat: feat[1])
        )

    # Extract feature names and values
    feature_names = list(feature_importances.keys())
    feature_values = list(feature_importances.values())

    # Create a color map based on the number of features
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors = colors[0 : 2 + 1][::-1]
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", colors, N=len(feature_importances)
    )
    colors = [cmap(x) for x in np.linspace(0, 1, len(feature_importances))]

    # Plot the horizontal bar chart
    ax.barh(
        feature_names,
        feature_values,
        color=colors,
        height=1,
        alpha=0.5,
    )

    # Set the title of the plot
    ax.set_title(title, weight="bold", fontsize=11)

    # Remove spines and ticks for a cleaner appearance
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.margins(0, 0)

    # Keep the value 0 to see negative values
    ax.set_xticks([0])
    ax.set_yticks([])

    # Display grid for better readability
    ax.grid(True)

    # Annotate feature names next to the bars
    for i, label in enumerate(feature_names):
        ax.annotate(
            label,
            xy=(0.5, i),
            xycoords=("axes fraction", "data"),
            size=8,
            ha="center",
            va="center",
        )

    return ax


if __name__ == "__main__":
    # Load test and train datasets
    df_test = pd.read_csv(
        path_data_selected["test"],
        index_col=params["column_mapping"]["id"],
    )

    df_train = pd.read_csv(
        path_data_selected["train"],
        index_col=params["column_mapping"]["id"],
    )

    # Load the trained model from the saved file
    with open(params["path"]["results"]["models"]["classifier"], "rb") as fp:
        classifier = pickle.load(fp)

    # Create a subplot with two columns and specified figure size
    fig, ax = plt.subplots(
        ncols=2,
        figsize=(5, 7),
    )

    # Iterate over train and test datasets
    for i, (title, df) in enumerate([("Train Data", df_train), ("Test Data", df_test)]):
        # Extract target and feature data
        targets = df[params["column_mapping"]["target"]]
        data = df.drop(params["column_mapping"]["target"], axis=1)

        # Perform permutation importance analysis
        results = permutation_importance(
            classifier,
            data,
            targets,
            scoring="f1_macro",
            n_repeats=3,
            n_jobs=1,
            random_state=42,
        )

        # Extract feature importances and create a dictionary
        importances_mean = [value.item() for value in results["importances_mean"]]
        feature_importances = dict(zip(df_train.columns, importances_mean))

        # Plot feature importances for each dataset
        plot_feature_importances(feature_importances, title=title, sort=False, ax=ax[i])

    # Set the title of the entire plot
    fig.suptitle("Feature Importances", fontweight="bold")

    # Adjust layout parameters for better spacing
    fig.subplots_adjust(top=0.90, bottom=0.05, wspace=0.05)

    # Save the figure with feature importances
    fig.savefig(params["path"]["results"]["plots"]["feature_importances"])
