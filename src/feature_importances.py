import dvc.api
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from plots.feature_importances import plot_feature_importances

params = dvc.api.params_show()

plt.style.use(params["plt_style"]["style"])
plt.rcParams["font.sans-serif"] = params["plt_style"]["font"]

# ----------- Loading data -----------
df_test = pd.read_csv(
    params["path"]["data_test"],
    index_col=params["column_mapping"]["id"],
)
df_train = pd.read_csv(
    params["path"]["data_train"],
    index_col=params["column_mapping"]["id"],
)

# ----------- Loading model -----------
pipeline = joblib.load(params["path"]["model_bin"])


# ----------- Compute feature importances -----------
def _compute_feature_importances(df):
    targets = df[params["column_mapping"]["target"]]
    data = df.drop(params["column_mapping"]["target"], axis=1)
    results = permutation_importance(
        pipeline,
        data,
        targets,
        scoring="f1_macro",
        n_repeats=3,
        n_jobs=-1,
        random_state=42,
    )

    importances_mean = [value.item() for value in results["importances_mean"]]
    feature_importances = dict(zip(pipeline.feature_names_in_, importances_mean))
    return feature_importances


fig, ax = plt.subplots(ncols=2, figsize=(10, 15))

plot_feature_importances(
    _compute_feature_importances(df_train),
    ax[0],
    title="Train Data",
    sort=False,
)
plot_feature_importances(
    _compute_feature_importances(df_test),
    ax[1],
    title="Test Data",
    sort=False,
)

fig.suptitle("Feature Importances", fontweight="bold")
fig.subplots_adjust(top=0.94, bottom=0.05)
fig.savefig(params["path"]["feature_importances"])
