import dvc.api
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.syntax import Syntax
from sklearn.model_selection import train_test_split

from plots.data_proportion import plot_data_proportion

console = Console()

stage = "split_data"

params = dvc.api.params_show(stages=stage)

console.log(
    f"[purple]['{stage}' stage config]",
    Syntax(
        yaml.dump(params),
        "yaml",
        theme="monokai",
        background_color="default",
    ),
)


# =========== Splitting data ===========
df = pd.read_csv(
    params["path"]["data_all"],
    index_col=params["column_mapping"]["id"],
)

X_train, X_test = train_test_split(df, **params["train_test_split"])

X_train.to_csv(params["path"]["data_train"])
X_test.to_csv(params["path"]["data_test"])

# =========== Plotting data proportion ===========
plt.style.use(params["plt_style"]["style"])
plt.rcParams["font.sans-serif"] = params["plt_style"]["font"]

target = params["column_mapping"]["target"]
labels = X_train[target].value_counts().index
data_proportion = np.transpose(
    [
        X_train[target].value_counts().values,
        X_test[target].value_counts().values,
    ]
)

fig, ax = plt.subplots()
plot_data_proportion(data_proportion, labels, ax)
fig.savefig(params["path"]["data_proportion"])
