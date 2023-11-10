import warnings

import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from hydra.utils import instantiate

# Ignore this warnings to don't flood the terminal
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get the dc params
params = dvc.api.params_show(stages="feature_selection")
target = params["column_mapping"]["target"]

# Use a matplotlib style to make more beautiful graphics
plt.style.use(params["plt_style"])


# TO DO: Put in a gist
def plot_correlation_matrix(x, y, color, size, size_scale=400, ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    n_colors = 256

    # Create a color palette for the correlation values
    palette = sns.diverging_palette(20, 220, n=n_colors)
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        # Convert a correlation value to a color in the palette
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    # Create a scatter plot with size and color based on correlation values
    ax.scatter(
        x=x,
        y=y,
        s=size * size_scale,
        c=[value_to_color(v) for v in color],
        marker="o",
    )

    # Configure the plot to become a matrix
    ax.set_xlim(-0.5, len(np.unique(x)) - 0.5)
    ax.set_ylim(-0.5, len(np.unique(y)) - 0.5)
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    # Configure grid lines
    ax.grid(False, "major")
    ax.grid(True, "minor")

    # Change y-axis ticks position to appear on the right
    ax.yaxis.set_ticks_position("right")
    ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha="right")

    # Remove ticks (-) on the axis
    ax.tick_params(axis="both", labelsize=10, which="both", length=0)

    ax.set_title("Correlation Matrix", fontweight="bold", fontsize=10)
    ax.set_box_aspect(1)

    return ax


if __name__ == "__main__":
    # Read the training dataset
    df_train = pd.read_csv(
        params["path"]["data_train_transformed"],
        index_col=params["column_mapping"]["id"],
    )

    # Read the test dataset
    df_test = pd.read_csv(
        params["path"]["data_test_transformed"],
        index_col=params["column_mapping"]["id"],
    )

    # 1. Select the best features
    # Seperate data into features and target to be compliant with sklean API
    y_train = df_train[target]
    X_train = df_train.drop(target, axis=1)
    y_test = df_test[target]
    X_test = df_test.drop(target, axis=1)

    # Load and fit the selector on the training dataset
    feature_selector = instantiate(params["feature_selection"])
    feature_selector.fit(X_train, y_train)

    # Transform data
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)

    # Save list of selected features to access later if need
    selected_features = list(feature_selector.get_feature_names_out(X_train.columns))
    with open("results/selected_features.yaml", "w", encoding="utf8") as fp:
        yaml.dump(selected_features, fp)

    # Save transformed data
    df_train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_features,
        index=df_train.index,
    )
    df_test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_features,
        index=df_test.index,
    )
    df_train_selected[target] = y_train
    df_test_selected[target] = y_test
    df_train_selected.to_csv(params["path"]["data_train_selected"])
    df_test_selected.to_csv(params["path"]["data_test_selected"])

    # 2. Plot correlation matrix of selected features
    df_train = df_train[[*selected_features, "status"]]

    # TO DO: replace status and phishing by params
    df_train["status"] = df_train["status"].replace({"phishing": 1, "legitimate": 0})

    # Compute correlation values and format the dataframe to get a lower triangle in the graph
    df_corr = df_train.corr()
    df_corr = df_corr.fillna(0)
    df_corr = df_corr.where((np.tril(np.ones(df_corr.shape), k=-1)).astype(bool))
    df_corr = pd.melt(df_corr.reset_index(), id_vars="index")
    df_corr = df_corr.dropna()
    df_corr.columns = ["x", "y", "value"]

    # Plot correlation matrix and save it in file
    plt.figure(figsize=(12, 12))
    plot_correlation_matrix(
        df_corr["x"],
        df_corr["y"],
        df_corr["value"],
        df_corr["value"].abs(),
    )
    plt.tight_layout()
    plt.savefig(params["path"]["correlation_matrix"])
