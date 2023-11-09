from datetime import datetime

import dvc.api
import joblib
import pandas as pd
import yaml
from hydra.utils import instantiate
from rich.console import Console
from rich.syntax import Syntax

stage = "train_model"

params = dvc.api.params_show(stages=stage)

console = Console()

console.log(
    f"[purple]['{stage}' stage config][purple]",
    Syntax(
        yaml.dump(params),
        "yaml",
        theme="monokai",
        background_color="default",
    ),
)
params = dvc.api.params_show(stages="train_model")


with open(params["path"]["selected_features"], "r", encoding="utf8") as fp:
    selected_features = yaml.safe_load(fp)


df_train = pd.read_csv(
    params["path"]["data_train"],
    index_col=params["column_mapping"]["id"],
    usecols=[
        params["column_mapping"]["id"],
        params["column_mapping"]["target"],
        *selected_features,
    ],
)

y_train = df_train[params["column_mapping"]["target"]]
X_train = df_train.drop(params["column_mapping"]["target"], axis=1)

pipeline = instantiate(params["pipeline"])


console.log("[purple]\[pipeline object][purple]", pipeline, sep="\n")

start_time = datetime.now()
pipeline = pipeline.fit(X_train, y_train)
end_time = datetime.now()

console.log(f"[purple]\[training time][/purple] {end_time - start_time}")

joblib.dump(pipeline, params["path"]["model_bin"])
