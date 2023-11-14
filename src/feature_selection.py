import os
import warnings

import dvc.api
import pandas as pd
import yaml
from hydra.utils import instantiate

# Ignore this warnings to don't flood the terminal
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Get the dc params
params = dvc.api.params_show(stages="feature_selection")
target = params["column_mapping"]["target"]
path_data_transformed = params["path"]["data"]["transformed"]
path_data_selected = params["path"]["data"]["selected"]


if __name__ == "__main__":
    # Read the training dataset
    df_train = pd.read_csv(
        path_data_transformed["train"],
        index_col=params["column_mapping"]["id"],
    )

    # Read the test dataset
    df_test = pd.read_csv(
        path_data_transformed["test"],
        index_col=params["column_mapping"]["id"],
    )

    # 1. Select the best features
    # Seperate data into features and target to be compliant with sklean API
    y_train = df_train[target]
    X_train = df_train.drop(target, axis=1)
    y_test = df_test[target]
    X_test = df_test.drop(target, axis=1)

    # Load and fit the selector on the training dataset
    feature_selector = instantiate(params["feature_selection"])
    feature_selector.fit(X_train, y_train)

    # Transform data
    X_train_selected = feature_selector.transform(X_train)
    X_test_selected = feature_selector.transform(X_test)

    # Get list of selected features
    selected_features = list(feature_selector.get_feature_names_out(X_train.columns))

    # Save selected features to read them later if need
    with open(params["path"]["results"]["selected_features"], "w") as fp:
        yaml.dump(selected_features, fp)

    # Save transformed data
    df_train_selected = pd.DataFrame(
        X_train_selected,
        columns=selected_features,
        index=df_train.index,
    )
    df_test_selected = pd.DataFrame(
        X_test_selected,
        columns=selected_features,
        index=df_test.index,
    )
    df_train_selected[target] = y_train
    df_test_selected[target] = y_test

    # Create directory for selected data
    os.makedirs(path_data_selected["dir"], exist_ok=True)

    df_train_selected.to_csv(path_data_selected["train"])
    df_test_selected.to_csv(path_data_selected["test"])
