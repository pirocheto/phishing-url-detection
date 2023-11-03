import dvc.api
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import classification_report

from plots.confusion_matrix import plot_confusion_matrix
from plots.roc_curve import plot_roc_curve
from plots.score_distribution import plot_score_distribution

params = dvc.api.params_show(stages="test_model")

plt.style.use(params["plt_style"]["style"])
plt.rcParams["font.sans-serif"] = params["plt_style"]["font"]


target = params["column_mapping"]["target"]
prediction = params["column_mapping"]["prediction"]
label_pred = f"{prediction}_label"
pos_label = params["column_mapping"]["pos_label"]

# =========== Predict data ===========
df_test = pd.read_csv(
    params["path"]["data_test"],
    index_col=params["column_mapping"]["id"],
)
X_test = df_test.drop(target, axis=1)

df_train = pd.read_csv(
    params["path"]["data_train"],
    index_col=params["column_mapping"]["id"],
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
