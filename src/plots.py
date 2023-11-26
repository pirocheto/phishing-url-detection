import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

plt.style.use("Solarize_Light2")


def plot_score_distribution(y_true, y_scores, labels=[0, 1], ax=None):
    # If ax is not provided, create a new axis
    if ax is None:
        ax = plt.gca()

    # Convert y_scores to a NumPy array
    y_scores = np.array(y_scores)

    for label in labels:
        scores = y_scores[y_true == label]

        sns.histplot(
            scores,
            kde=True,
            ax=ax,
            bins="auto",
            edgecolor=None,
            legend=True,
            label=str(label),
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
    ax.set_title("Score distribution", fontweight="bold", fontsize=12)

    return ax


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
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix", fontweight="bold", fontsize=12)

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
    ax.set_title("Calibration Curve", fontweight="bold", fontsize=12)

    # Display the grid
    ax.grid(True)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

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
        "Receiver Operating Characteristic (ROC)", fontweight="bold", fontsize=12
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
    ax.set_title("Precision-Recall Curve", fontweight="bold", fontsize=12)

    # Display the grid
    ax.grid(True)

    # Set the aspect ratio of the plot to be equal
    ax.set_box_aspect(1)

    return ax
