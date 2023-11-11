import dvc.api
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import ks_2samp
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

# Get the dvc params
params = dvc.api.params_show(stages="test_model")
target = params["column_mapping"]["target"]
proba_pred = params["column_mapping"]["prediction_proba"]
label_pred = params["column_mapping"]["prediction_label"]
pos_label = params["column_mapping"]["pos_label"]

# Use a matplotlib style to make more beautiful graphics
plt.style.use(params["plt_style"])


def plot_confusion_matrix(y_true, y_pred, labels=[0, 1], ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Compute confusion matrix using scikit-learn's confusion_matrix function
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Get the default color cycle and select the first two colors
    # Then create a custom colormap with the selected colors
    # Useful when a global style is used i.e. with 'plt.style.use()'
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    colors = colors[0:2]
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

    # Plot the confusion matrix as a heatmap using seaborn's heatmap function
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax,
        cmap=cmap,
        linecolor=plt.rcParams["figure.facecolor"],
        linewidths="2",
        vmin=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        robust=True,
        square=True,
    )

    # Remove tick marks (-)
    ax.tick_params(axis="both", which="both", length=0)

    # Set labels and title for the plot
    ax.set_xlabel("Pred Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=10)
    return ax


def plot_roc_curve(y_true, y_scores, pos_label=1, ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(
        y_true, y_scores, pos_label=pos_label, drop_intermediate=False
    )

    # Compute area under the curve (AUC)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold based on maximizing (TPR - FPR)
    best_threshold_index = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_threshold_index]

    # Plot the ROC curve
    ax.plot(
        fpr,
        tpr,
        lw=2,
        label=f"ROC Curve \n(AUC = {roc_auc:.3f})",
    )

    # Plot the diagonal line
    ax.plot(
        [0, 1],
        [0, 1],
        lw=2,
        linestyle="dotted",
        alpha=0.3,
    )

    # Fill the area under the ROC curve
    ax.fill_between(fpr, tpr, alpha=0.2)

    # Mark the point on the ROC curve corresponding to the best threshold
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][0]
    ax.scatter(
        fpr[best_threshold_index],
        tpr[best_threshold_index],
        color=color,
    )

    # Annotate the best threshold on the plot
    ax.annotate(
        "Best Threshold\n" + r"$\bf{" + f"{best_threshold:.3f}" + "}$",
        xy=(
            fpr[best_threshold_index],
            tpr[best_threshold_index],
        ),
        xytext=(0.5, -0.5),
        textcoords="offset fontsize",
        va="top",
    )

    # Set axis limits, labels and title
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title(
        "Receiver Operating Characteristic (ROC)", fontweight="bold", fontsize=10
    )

    # Add text annotation for AUC
    ax.text(
        0.8,
        0.15,
        "AUC\n" + r"$\bf{" + f"{roc_auc:.3f}" + "}$",
        va="center",
        ha="center",
        bbox=dict(
            boxstyle="square",
            fc=ax.get_facecolor(),
            ec="grey",
            alpha=0.4,
            pad=0.5,
        ),
    )

    # Display the grid
    ax.grid(True)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax


def plot_score_distribution(y_true, y_scores, labels=(0, 1), ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Convert y_scores to a NumPy array
    y_scores = np.array(y_scores)

    # Extract scores for each class
    scores1 = y_scores[y_true == labels[0]]
    scores2 = y_scores[y_true == labels[1]]

    # Plot the score distribution for the first class
    ax = sns.histplot(
        scores1,
        kde=True,
        ax=ax,
        bins="auto",
        edgecolor=None,
        legend=True,
        label=str(labels[0]),
    )

    # Plot the score distribution for the second class
    ax = sns.histplot(
        scores2,
        kde=True,
        ax=ax,
        bins="auto",
        edgecolor=None,
        legend=True,
        label=str(labels[1]),
    )

    # Set axis limits and labels
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.xaxis.set_ticks_position("bottom")

    # Display the grid
    ax.grid(True)

    # Add legend with some transparency
    ax.legend(framealpha=0.4)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    # Set the title of the plot
    ax.set_title("Score distribution", fontweight="bold", fontsize=10)
    return ax


def plot_precision_recall_curve(y_true, y_scores, pos_label=1, ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Get the default color cycle and select the first color
    # Useful when a global style is used i.e. with 'plt.style.use()'
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    color = colors[1]

    # Compute precision-recall curve
    precision, recall, threshold = precision_recall_curve(
        y_true, y_scores, pos_label=pos_label
    )

    # Find the best threshold based on maximizing F1 score
    fscore = 2 * (precision * recall) / (precision + recall)
    best_threshold_index = fscore.argmax()
    best_threshold = threshold[best_threshold_index]

    # Plot the precision-recall curve
    ax.step(recall, precision, where="post", color=color)
    ax.fill_between(recall, precision, step="post", alpha=0.2, color=color)

    # Compute average precision
    average_precision = average_precision_score(y_true, y_scores, pos_label=pos_label)

    # Add text annotation for Average Precision (AP)
    ax.text(
        0.20,
        0.15,
        "AP\n" + r"$\bf{" + f"{average_precision:.3f}" + "}$",
        va="center",
        ha="center",
        bbox=dict(
            boxstyle="square",
            fc=ax.get_facecolor(),
            ec="grey",
            alpha=0.4,
            pad=0.5,
        ),
    )

    # Mark the point on the curve corresponding to the best threshold
    ax.scatter(
        recall[best_threshold_index],
        precision[best_threshold_index],
        color=color,
    )

    # Annotate the best threshold on the plot
    ax.annotate(
        "Best Threshold\n" + r"$\bf{" + f"{best_threshold:.3f}" + "}$",
        xy=(
            recall[best_threshold_index],
            precision[best_threshold_index],
        ),
        xytext=(-0.5, -0.5),
        textcoords="offset fontsize",
        va="top",
        ha="right",
    )

    # Set axis limits and labels
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    # Set the title of the plot
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=10)

    # Display the grid
    ax.grid(True)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax


def plot_calibration_curve(y_true, y_scores, pos_label=1, ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_true, y_scores, pos_label=pos_label, n_bins=10
    )

    # Plot the calibration curve
    ax.plot(prob_pred, prob_true, marker="o", linestyle="-")

    # Plot the diagonal line (perfect line)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Set axis labels
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")

    # Set title
    ax.set_title("Calibration Curve", fontweight="bold", fontsize=10)

    # Display the grid
    ax.grid(True)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax


def plot_cumulative_distribution(y_true, y_scores, labels=(0, 1), ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Extract scores for each class
    y_scores = np.array(y_scores)
    scores1 = y_scores[y_true == labels[0]]
    scores2 = y_scores[y_true == labels[1]]

    # Sort the data for ECDF calculation
    sorted_data1 = np.sort(scores1)
    ecdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
    sorted_data2 = np.sort(scores2)
    ecdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)

    # Plots the curves
    ax.plot(sorted_data1, ecdf1, label=labels[0])
    ax.plot(sorted_data2, ecdf2, label=labels[1])

    # Compute the Kolmogorov-Smirnov (KS) statistic and p-value
    ks_statistic, p_value = ks_2samp(scores1, scores2)

    # Add text annotation for KS Statistic and p-value
    ax.text(
        0.5,
        0.5,
        "KS Statistic:" + r"$\bf{" + f"{ks_statistic:.3f}" + "}$\n"
        "p-value:" + r"$\bf{" + f"{p_value:.3}" + "}$",
        va="center",
        ha="center",
        bbox=dict(
            boxstyle="square",
            fc=ax.get_facecolor(),
            ec="grey",
            alpha=0.6,
            pad=0.5,
        ),
    )
    # Set axis limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)

    # Set labels
    ax.set_xlabel("Predicted Probabilities")
    ax.set_ylabel("Cumulative Distribution")

    # Set the title of the plot
    ax.set_title(
        "Empirical Cumulative Distribution Function",
        fontweight="bold",
        fontsize=10,
    )

    # Display legend and the grid
    ax.legend()
    ax.grid(True)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax


def plot_metrics_table(metrics, ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Format metrics to display as strings with three decimal places
    metrics = [(name, f"{value:.3f}") for name, value in metrics]

    # Create a table with the specified metrics
    table = ax.table(cellText=metrics, loc="upper center", edges="horizontal")

    # Customize the appearance of the table cells
    for i, cell in enumerate(table.get_celld().values()):
        cell.set_facecolor(plt.rcParams["figure.facecolor"])
        cell.set_edgecolor(plt.rcParams["text.color"])

        # Alternate background colors for better readability
        if i % 2 == 0:
            cell.get_text().set_ha("right")
        else:
            cell.get_text().set_ha("center")
            cell.get_text().set_fontweight("bold")

    # Adjust font size and scale of the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Hide axis
    ax.axis("off")

    # Set the title of the plot
    ax.set_title("Metrics", fontweight="bold", fontsize=10)

    return ax


if __name__ == "__main__":
    # 1. Make predictions on test dataset
    # Read the test dataset
    df_test = pd.read_csv(
        params["path"]["data_test_selected"],
        index_col=params["column_mapping"]["id"],
    )
    X_test = df_test.drop(target, axis=1)

    # Load the trained model
    pipeline = joblib.load(params["path"]["model_bin"])

    # Make predictions on test dataset
    df_test[proba_pred] = pipeline.predict_proba(X_test)[:, 1]
    df_test[label_pred] = pipeline.predict(X_test)

    # Save predictions to read them leter if need
    df_test.to_csv(params["path"]["data_test_predicted"])

    # 2. Compute metrics
    metrics = classification_report(
        df_test[target],
        df_test[label_pred],
        output_dict=True,
    )
    metrics = {
        "accuracy": metrics["accuracy"],
        "f1-score": metrics["1"]["f1-score"],
        "precision": metrics["1"]["precision"],
        "recall": metrics["1"]["recall"],
    }

    # Save metrics to read them leter if need
    with open(params["path"]["metrics"], "w", encoding="utf8") as fp:
        yaml.safe_dump(metrics, fp)

    # 2. Creates several plots to more easily interpret the results
    # Plot and save metrics table
    plt.figure()
    plot_metrics_table(metrics.items())
    plt.savefig(params["path"]["metrics_table"])

    # Plot and save confusion matrix
    plt.figure()
    plot_confusion_matrix(df_test[target], df_test[label_pred])
    plt.savefig(params["path"]["confusion_matrix"])

    # Plot and save roc curve
    plt.figure()
    plot_roc_curve(df_test[target], df_test[proba_pred])
    plt.savefig(params["path"]["roc_curve"])

    # Plot and save score distribution
    plt.figure()
    plot_score_distribution(df_test[target], df_test[proba_pred])
    plt.savefig(params["path"]["score_distribution"])

    # Plot and save precision recall curve
    plt.figure()
    plot_precision_recall_curve(df_test[target], df_test[proba_pred])
    plt.savefig(params["path"]["precision_recall_curve"])

    # Plot and save calibration curve
    plt.figure()
    plot_calibration_curve(df_test[target], df_test[proba_pred])
    plt.savefig(params["path"]["calibration_curve"])

    # Plot and save cumulative distribution
    plt.figure()
    plot_cumulative_distribution(df_test[target], df_test[proba_pred])
    plt.savefig(params["path"]["cumulative_distribution"])

    # Create classification report containing multiple charts to see results on a single page
    fig, ax = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(8.27, 11.69),
    )

    plot_metrics_table(metrics.items(), ax=ax[0][0])
    plot_confusion_matrix(df_test[target], df_test[label_pred], ax=ax[0][1])
    plot_roc_curve(df_test[target], df_test[proba_pred], ax=ax[1][0])
    plot_precision_recall_curve(df_test[target], df_test[proba_pred], ax=ax[1][1])
    plot_score_distribution(df_test[target], df_test[proba_pred], ax=ax[2][0])
    plot_cumulative_distribution(df_test[target], df_test[proba_pred], ax=ax[2][1])
    fig.subplots_adjust(
        left=0.05,
        right=1,
        top=0.92,
        bottom=0.08,
        wspace=0.10,
        hspace=0.4,
    )
    fig.savefig(params["path"]["classification_report"])
