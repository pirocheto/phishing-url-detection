import os
import warnings

import dill
import dvc.api
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from skl2onnx import to_onnx
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline

# Ignore this warnings to don't flood the terminal
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Get the dvc params
params = dvc.api.params_show(stages="finalize_model")


if __name__ == "__main__":
    target_preprocessor = instantiate(params["preprocessing"]["target"])
    classifier = instantiate(params["classifier"])
    feature_selector = instantiate(params["feature_selection"])

    feature_preprocessing = params["preprocessing"].get("feature", None)
    if feature_preprocessing:
        feature_preprocessor = instantiate(params["preprocessing"]["feature"])

    df_train = pd.read_csv("data/all.csv", index_col=params["column_mapping"]["id"])
    y_train = df_train[params["column_mapping"]["target"]]
    df_train = df_train.drop(params["column_mapping"]["target"], axis=1)

    if feature_preprocessing:
        model = make_pipeline(feature_preprocessor, classifier)
    else:
        model = classifier

    y_train = target_preprocessor.fit_transform(y_train)

    X_train = feature_selector.fit_transform(df_train, y_train)

    model.fit(X_train, y_train)

    onx = to_onnx(model, X_train[:1].astype(np.float32))

    os.makedirs(params["path"]["final_models"]["dir"], exist_ok=True)

    with open(params["path"]["final_models"]["onnx_model"], "wb") as fp:
        fp.write(onx.SerializeToString())

    with open(params["path"]["final_models"]["pkl_model"], "wb") as fp:
        dill.dump(model, fp)
