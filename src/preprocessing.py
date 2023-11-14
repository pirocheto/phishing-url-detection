import os

import dvc.api
import pandas as pd
from hydra.utils import instantiate

# Get the dvc params
params = dvc.api.params_show(stages="preprocessing")
path_data_raw = params["path"]["data"]["raw"]
path_data_transformed = params["path"]["data"]["transformed"]
target = params["column_mapping"]["target"]

if __name__ == "__main__":
    # Load the test dataset
    df_test = pd.read_csv(
        path_data_raw["test"],
        index_col=params["column_mapping"]["id"],
    )
    # Load the training dataset
    df_train = pd.read_csv(
        path_data_raw["train"],
        index_col=params["column_mapping"]["id"],
    )

    # Seperate data into features and target to be compliant with sklean API
    X_test = df_test.drop(target, axis=1)
    y_test = df_test[target]
    X_train = df_train.drop(target, axis=1)
    y_train = df_train[target]

    # 1. Process target
    target_preprocessor = instantiate(params["target_preprocessing"])

    # Fit the target preprocessor
    target_preprocessor.fit(y_train)

    # Transform the labels
    y_train = target_preprocessor.transform(y_train)
    y_test = target_preprocessor.transform(y_test)

    # 2. Process features
    feature_preprocessor = instantiate(params["feature_preprocessing"])

    # Fit the feature preprocessor
    feature_preprocessor.fit(X_train)

    # Transform the features
    X_train = feature_preprocessor.transform(X_train)
    X_test = feature_preprocessor.transform(X_test)

    # Save the transformed data
    df_train_transformed = pd.DataFrame(
        X_train,
        columns=df_train.columns[:-1],
        index=df_train.index,
    )
    df_test_transformed = pd.DataFrame(
        X_test,
        columns=df_test.columns[:-1],
        index=df_test.index,
    )

    df_train_transformed[target] = y_train
    df_test_transformed[target] = y_test

    # Create directory for transformed data
    os.makedirs(path_data_transformed["dir"], exist_ok=True)

    df_train_transformed.to_csv(path_data_transformed["train"])
    df_test_transformed.to_csv(path_data_transformed["test"])
