from sklearn.calibration import calibration_curve


def plot_calibration_curve(y_true, y_scores, pos_label, ax):
    prob_true, prob_pred = calibration_curve(
        y_true, y_scores, pos_label=pos_label, n_bins=10
    )
    ax.plot(prob_pred, prob_true, marker="o", linestyle="-")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve", fontweight="bold", fontsize=10)
    ax.grid()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.set_box_aspect(1)
