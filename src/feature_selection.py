import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plots.correlation_matrix import plot_correlation_matrix

params = dvc.api.params_show(stages="feature_selection")
fig, ax = plt.subplots(figsize=(20, 20))

df = pd.read_csv(params["path"]["data_all"], index_col=params["column_mapping"]["id"])

df = df[df.columns[-50:]]
df["status"] = df["status"].replace({"phishing": 1, "legitimate": 0})

df_corr = df.corr()
df_corr = df_corr.fillna(0)

df_corr = df_corr.where(np.triu(np.ones(df_corr.shape), k=1).astype(bool))

corr = pd.melt(df_corr.reset_index(), id_vars="index")
corr = corr.dropna()
corr.columns = ["x", "y", "value"]

x = corr["x"]
y = corr["y"]
color = corr["value"]
size = corr["value"].abs()

plot_correlation_matrix(x, y, color, size, ax)
fig.savefig(params["path"]["correlation_matrix"])
