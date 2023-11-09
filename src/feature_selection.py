import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from hydra.utils import instantiate
from rich.console import Console
from rich.pretty import Pretty
from rich.rule import Rule
from rich.syntax import Syntax

from plots.correlation_matrix import plot_correlation_matrix

stage = "feature_selection"

params = dvc.api.params_show(stages=stage)


plt.style.use(params["plt_style"]["style"])
plt.rcParams["font.sans-serif"] = params["plt_style"]["font"]

df_train = pd.read_csv(
    params["path"]["data_train"], index_col=params["column_mapping"]["id"]
)
df_test = pd.read_csv(
    params["path"]["data_test"], index_col=params["column_mapping"]["id"]
)
y_train = df_train[params["column_mapping"]["target"]]
X_train = df_train.drop(params["column_mapping"]["target"], axis=1)

y_test = df_test[params["column_mapping"]["target"]]
X_test = df_test.drop(params["column_mapping"]["target"], axis=1)

feature_selector = instantiate(params["feature_selection"])

feature_selector = feature_selector.fit(X_train, y_train)

selected_features = list(feature_selector.get_feature_names_out(X_train.columns))


with open("results/selected_features.yaml", "w", encoding="utf8") as fp:
    yaml.dump(selected_features, fp)


# ---------- Plot correlation matrix ----------
fig, ax = plt.subplots(figsize=(12, 12))

df = pd.read_csv(
    params["path"]["data_train"],
    index_col=params["column_mapping"]["id"],
    usecols=[
        params["column_mapping"]["id"],
        params["column_mapping"]["target"],
        *selected_features,
    ],
)

df["status"] = df["status"].replace({"phishing": 1, "legitimate": 0})

df_corr = df.corr()
df_corr = df_corr.fillna(0)

df_corr = df_corr.where((np.tril(np.ones(df_corr.shape), k=-1)).astype(bool))

corr = pd.melt(df_corr.reset_index(), id_vars="index")
corr = corr.dropna()
corr.columns = ["x", "y", "value"]

x = corr["x"]
y = corr["y"]
color = corr["value"]
size = corr["value"].abs()

plot_correlation_matrix(x, y, color, size, ax)
fig.tight_layout()
fig.savefig(params["path"]["correlation_matrix"])
