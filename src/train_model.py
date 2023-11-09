from datetime import datetime

import dvc.api
import joblib
import pandas as pd
import yaml
from hydra.utils import instantiate

stage = "train_model"

params = dvc.api.params_show(stages=stage)


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


start_time = datetime.now()
pipeline = pipeline.fit(X_train, y_train)
end_time = datetime.now()


joblib.dump(pipeline, params["path"]["model_bin"])
