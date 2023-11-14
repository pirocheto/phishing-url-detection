import warnings

import dvc.api
import joblib
import pandas as pd
from hydra.utils import instantiate
from sklearn.exceptions import ConvergenceWarning

# Ignore this warnings to don't flood the terminal
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Get the dvc params
params = dvc.api.params_show(stages="train_model")

if __name__ == "__main__":
    # Get the training dataset with only best features
    df_train = pd.read_csv(
        params["path"]["data"]["selected"]["train"],
        index_col=params["column_mapping"]["id"],
    )

    # Seperate data into features (X_train) and target (y_train) to be compliant with sklean API
    y_train = df_train[params["column_mapping"]["target"]]
    X_train = df_train.drop(params["column_mapping"]["target"], axis=1)

    # Load and fit the model on the training dataset, then save it.
    pipeline = instantiate(params["pipeline"])
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, params["path"]["results"]["model_bin"])
