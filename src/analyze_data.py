import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Get the dc params
params = dvc.api.params_show(stages="analyze_data")
target = params["column_mapping"]["target"]
pos_label = params["column_mapping"]["pos_label"]

# Use a matplotlib style to make more beautiful graphics
plt.style.use(params["plt_style"])


def plot_correlation_matrix(x, y, color, size, size_scale=100, ax=None):
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

    # Remove tick marks (-)
    ax.tick_params(axis="both", labelsize=10, which="both", length=0)

    # Set title
    ax.set_title("Correlation Matrix", fontweight="bold", fontsize=10)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax


if __name__ == "__main__":
    df = pd.read_csv(
        params["path"]["data"]["all"],
        index_col=params["column_mapping"]["id"],
        dtype=params["feature_dtypes"],
    )

    # 1. Plot correlation matrix of selected features
    # Compute correlation values and format the dataframe to get a lower triangle in the graph
    df[target] = [1 if label == pos_label else 0 for label in df[target]]
    df_corr = df.corr()
    df_corr = df_corr.fillna(0)
    df_corr = df_corr.where((np.tril(np.ones(df_corr.shape), k=-1)).astype(bool))
    df_corr = pd.melt(df_corr.reset_index(), id_vars="index")
    df_corr = df_corr.dropna()
    df_corr.columns = ["x", "y", "value"]

    # Plot correlation matrix and save it in file
    plt.figure(figsize=(20, 20))
    plot_correlation_matrix(
        df_corr["x"],
        df_corr["y"],
        df_corr["value"],
        df_corr["value"].abs(),
        size_scale=150,
    )
    plt.tight_layout()
    plt.savefig(params["path"]["results"]["plots"]["correlation_matrix"])
    plt.close()

    # 2. Plot feature statistics
    # /!\ Warning: String features should be removed before plotting

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"][:2]

    df_features = df.drop(target, axis=1)

    with PdfPages(params["path"]["results"]["plots"]["feature_statistics"]) as pdf:
        for col in df_features.columns:
            # Plot a boxplot for numerical features
            if df_features[col].dtype in ["int64", "float64"]:
                sns.boxplot(
                    x=target,
                    y=col,
                    data=df,
                    boxprops=dict(alpha=0.8),
                    linewidth=1,
                    hue=target,
                    width=0.3,
                    linecolor="grey",
                )
                plt.tick_params(axis="both", which="both", length=0)
                plt.title(col)
                pdf.savefig()
                plt.close()
            # Plot a countplot for categorical features
            elif df_features[col].dtype in ["object", "categorical", "bool"]:
                sns.countplot(
                    x=col,
                    hue=target,
                    data=df,
                    alpha=0.8,
                    width=0.3,
                    dodge=True,
                    palette=colors,
                    edgecolor=plt.rcParams["figure.facecolor"],
                )
                plt.tick_params(axis="both", which="both", length=0)
                plt.title(col)
                pdf.savefig()
                plt.close()
