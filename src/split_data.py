import dvc.api
import pandas as pd
from sklearn.model_selection import train_test_split

params = dvc.api.params_show()

df = pd.read_csv(params["data_all"], index_col=params["column_mapping"]["id"])
# df = df.replace({params["column_mapping"]["target"]: {"legitimate": 0, "phishing": 1}})

X_train, X_test = train_test_split(df, **params["train_test_split"])

X_train.to_csv(params["data_train"])
X_test.to_csv(params["data_test"])
