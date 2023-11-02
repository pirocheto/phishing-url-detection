import dvc.api
import joblib
import pandas as pd
from hydra.utils import instantiate

params = dvc.api.params_show()

df_train = pd.read_csv(params["data_train"], index_col=params["column_mapping"]["id"])
y_train = df_train[params["column_mapping"]["target"]]
X_train = df_train.drop(params["column_mapping"]["target"], axis=1)

pipeline = instantiate(params["pipeline"])
pipeline = pipeline.fit(X_train, y_train)

joblib.dump(pipeline, params["model_dst"])
