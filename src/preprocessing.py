import os
import pickle

import dvc.api
import pandas as pd
from hydra.utils import instantiate

# Get the dvc params
params = dvc.api.params_show(stages="preprocessing")
path_data_raw = params["path"]["data"]["raw"]
feature = params["column_mapping"]["feature"]
target = params["column_mapping"]["target"]

if __name__ == "__main__":
    # Load the test dataset
    df_test = pd.read_csv(path_data_raw["test"])

    # Load the training dataset
    df_train = pd.read_csv(path_data_raw["train"])

    # Seperate data into features and target to be compliant with sklean API
    X_test = df_test[feature]
    y_test = df_test[target]
    X_train = df_train[feature]
    y_train = df_train[target]

    # 1. Process target
    target_preprocessor = instantiate(params["preprocessing"]["target"])

    # Fit the target preprocessor
    target_preprocessor.fit(y_train)

    # Transform the labels
    y_train = target_preprocessor.transform(y_train)
    y_test = target_preprocessor.transform(y_test)

    # 2. Process features
    feature_preprocessor = instantiate(params["preprocessing"]["feature"])

    # Fit the feature preprocessor
    feature_preprocessor.fit(X_train)

    # Transform the features
    X_train = feature_preprocessor.transform(X_train)
    X_test = feature_preprocessor.transform(X_test)

    os.makedirs(params["path"]["results"]["models"]["dir"], exist_ok=True)

    path_preprocessor = params["path"]["results"]["models"]["preprocessor"]
    path_label_encoder = params["path"]["results"]["models"]["label_encoder"]

    with open(path_preprocessor, "wb") as fp:
        pickle.dump(feature_preprocessor, fp)

    with open(path_label_encoder, "wb") as fp:
        pickle.dump(target_preprocessor, fp)
