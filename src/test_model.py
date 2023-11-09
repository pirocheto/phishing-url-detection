import dvc.api
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from rich.console import Console
from rich.syntax import Syntax
from sklearn.metrics import classification_report

from plots.calibration_curve import plot_calibration_curve
from plots.confusion_matrix import plot_confusion_matrix
from plots.cumulative_distribution import plot_cumulative_distribution
from plots.metrics_table import plot_metrics_table
from plots.precision_recall_curve import plot_precision_recall_curve
from plots.roc_curve import plot_roc_curve
from plots.score_distribution import plot_score_distribution

stage = "test_model"

params = dvc.api.params_show(stages=stage)

console = Console()

console.log(
    f"[purple]\['{stage}' stage config]",
    Syntax(
        yaml.dump(params),
        "yaml",
        theme="monokai",
        background_color="default",
    ),
)

plt.style.use(params["plt_style"]["style"])
plt.rcParams["font.sans-serif"] = params["plt_style"]["font"]


target = params["column_mapping"]["target"]
prediction = params["column_mapping"]["prediction"]
label_pred = f"{prediction}_label"
pos_label = params["column_mapping"]["pos_label"]

# =========== Predict data ===========
with open(params["path"]["selected_features"], "r", encoding="utf8") as fp:
    selected_features = yaml.safe_load(fp)


df_test = pd.read_csv(
    params["path"]["data_test"],
    index_col=params["column_mapping"]["id"],
    usecols=[
        params["column_mapping"]["id"],
        params["column_mapping"]["target"],
        *selected_features,
    ],
)
X_test = df_test.drop(target, axis=1)

df_train = pd.read_csv(
    params["path"]["data_train"],
    index_col=params["column_mapping"]["id"],
    usecols=[
        params["column_mapping"]["id"],
        params["column_mapping"]["target"],
        *selected_features,
    ],
)
X_train = df_train.drop(target, axis=1)

pipeline = joblib.load(params["path"]["model_bin"])
labels = df_test[target].value_counts().index

df_test[prediction] = pipeline.predict_proba(X_test)[:, 1]
df_test[label_pred] = pipeline.predict(X_test)
df_train[prediction] = pipeline.predict_proba(X_train)[:, 1]
df_train[label_pred] = pipeline.predict(X_train)

df_test.to_csv(params["path"]["data_test_pred"])
df_train.to_csv(params["path"]["data_train_pred"])


y_scores = df_test[prediction]
y_pred = df_test[label_pred]
y_true = df_test[target]

# =========== Compute metrics ===========
metrics = classification_report(
    df_test[target],
    df_test[label_pred],
    output_dict=True,
)


metrics = {
    "accuracy": metrics["accuracy"],
    "f1-score": metrics[pos_label]["f1-score"],
    "precision": metrics[pos_label]["precision"],
    "recall": metrics[pos_label]["recall"],
}

with open(params["path"]["metrics"], "w", encoding="utf8" "") as fp:
    yaml.safe_dump(metrics, fp)

console.log(
    "[purple]\[metrics]",
    Syntax(
        yaml.dump(metrics),
        "yaml",
        theme="monokai",
        background_color="default",
    ),
)

# =========== Plotting confusion matrix ===========
fig, ax = plt.subplots()
plot_confusion_matrix(
    df_test[target],
    df_test[label_pred],
    labels,
    ax,
)

fig.savefig(params["path"]["confusion_matrix"])


# =========== Plotting ROC curve ===========
fig, ax = plt.subplots()
plot_roc_curve(
    df_test[target],
    df_test[prediction],
    pos_label,
    ax,
)

fig.savefig(params["path"]["roc_curve"])

# =========== Plotting score distribution ===========
fig, ax = plt.subplots()
plot_score_distribution(
    df_test[target],
    df_test[prediction],
    labels,
    ax,
)

fig.savefig(params["path"]["score_distribution"])


# =========== Plotting score distribution ===========
fig, ax = plt.subplots()

plot_precision_recall_curve(
    df_test[target],
    df_test[prediction],
    pos_label,
    ax,
)

fig.savefig(params["path"]["precision_recall_curve"])


# =========== Plotting calibration curve ===========
fig, ax = plt.subplots()

plot_calibration_curve(
    df_test[target],
    df_test[prediction],
    pos_label,
    ax,
)

fig.savefig(params["path"]["calibration_curve"])

# =========== Plotting cumulative distribution ===========
fig, ax = plt.subplots()

plot_cumulative_distribution(
    df_test[target],
    df_test[prediction],
    labels,
    ax,
)

fig.savefig(params["path"]["cumulative_distribution"])

# =========== Plotting Metrics table ===========
fig, ax = plt.subplots()

plot_metrics_table(list(metrics.items()), ax=ax)

fig.savefig(params["path"]["metrics_table"])

# =========== Plotting combined curves ===========
fig, ax = plt.subplots(
    nrows=3,
    ncols=2,
    figsize=(8.27, 11.69),
)


plot_confusion_matrix(df_test[target], df_test[label_pred], labels, ax[0][0])
plot_metrics_table(list(metrics.items()), ax=ax[0][1])
plot_roc_curve(df_test[target], df_test[prediction], pos_label, ax[1][0])
plot_precision_recall_curve(df_test[target], df_test[prediction], pos_label, ax[1][1])

plot_score_distribution(df_test[target], df_test[prediction], labels, ax[2][0])
plot_cumulative_distribution(df_test[target], df_test[prediction], labels, ax[2][1])

fig.suptitle("Classification Report", fontsize=13, fontweight="bold")

fig.subplots_adjust(wspace=0.5, hspace=0.5)
fig.savefig(params["path"]["classification_report"], dpi=300)
