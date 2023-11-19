import pickle
import warnings

import dvc.api
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
    with open(params["path"]["results"]["models"]["preprocessor"], "rb") as fp:
        preprocessor = pickle.load(fp)

    with open(params["path"]["results"]["models"]["label_encoder"], "rb") as fp:
        label_encoder = pickle.load(fp)

    # Get the training dataset with only best features
    df_train = pd.read_csv(params["path"]["data"]["raw"]["train"])

    # Seperate data into features (X_train) and target (y_train) to be compliant with sklean API
    X_train = df_train[params["column_mapping"]["feature"]]
    y_train = df_train[params["column_mapping"]["target"]]

    # Load and fit the model on the training dataset, then save it.
    X_train = preprocessor.transform(X_train)
    y_train = label_encoder.transform(y_train)

    classifier = instantiate(params["classifier"])

    classifier.fit(X_train, y_train)

    with open(params["path"]["results"]["models"]["classifier"], "wb") as fp:
        pickle.dump(classifier, fp)
