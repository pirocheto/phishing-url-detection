import dvc.api
import joblib
import pandas as pd

params = dvc.api.params_show()

df_test = pd.read_csv(params["data_test"], index_col=params["column_mapping"]["id"])
X_test = df_test.drop(params["column_mapping"]["target"], axis=1)

df_train = pd.read_csv(params["data_train"], index_col=params["column_mapping"]["id"])
X_train = df_train.drop(params["column_mapping"]["target"], axis=1)

pipeline = joblib.load(params["model_dst"])

df_test["phishing"] = pipeline.predict_proba(X_test)[:, 1]
df_train["phishing"] = pipeline.predict_proba(X_train)[:, 1]

df_test.to_csv(params["data_test_pred"])
df_train.to_csv(params["data_train_pred"])
