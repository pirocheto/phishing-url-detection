import dvc.api
import joblib
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance

params = dvc.api.params_show()

# df_test = pd.read_csv(params["data_test"], index_col=params["column_mapping"]["id"])
# X_test = df_test.drop(params["column_mapping"]["target"], axis=1)

df_train = pd.read_csv(params["data_train"], index_col=params["column_mapping"]["id"])
y_train = df_train[params["column_mapping"]["target"]]
X_train = df_train.drop(params["column_mapping"]["target"], axis=1)

pipeline = joblib.load(params["model_dst"])

result = permutation_importance(
    pipeline,
    X_train,
    y_train,
    scoring="f1_macro",
    n_repeats=10,
    # n_jobs=-1,
    random_state=42,
)

importances_mean = [value.item() for value in result["importances_mean"]]
feature_importance = dict(zip(pipeline.feature_names_in_, importances_mean))

with open(params["feature_importance"], "w", encoding="utf8") as fp:
    yaml.dump(feature_importance, fp)
