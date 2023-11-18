import os
import warnings

import dvc.api
import joblib
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
    # Load all Sklearn objects
    classifier = instantiate(params["classifier"])
    feature_selector = instantiate(params["feature_selection"])
    target_preprocessor = instantiate(params["preprocessing"]["target"])

    feature_preprocessing = params["preprocessing"].get("feature", None)
    if feature_preprocessing:
        feature_preprocessor = instantiate(params["preprocessing"]["feature"])

    # 1. Load data
    df_train = pd.read_csv(
        params["path"]["data"]["all"],
        index_col=params["column_mapping"]["id"],
    )

    # Get features and labels
    y_train = df_train[params["column_mapping"]["target"]]
    df_train = df_train.drop(params["column_mapping"]["target"], axis=1)

    # Encode the labels
    y_train = target_preprocessor.fit_transform(y_train)

    # Keep only the relevant features
    X_train = feature_selector.fit_transform(df_train, y_train)

    # 2. Build and train the model
    if feature_preprocessing:
        pipeline = make_pipeline(feature_preprocessor, classifier)
        model = pipeline.fit(X_train, y_train)
    else:
        model = classifier.fit(X_train, y_train)

    # 3. Save the model
    # Save model in pickle format
    os.makedirs(params["path"]["final_models"]["dir"], exist_ok=True)
    joblib.dump(model, params["path"]["final_models"]["pkl_model"])

    # Save model in onnx format
    onx = to_onnx(model, X_train[:1].astype(np.float32), options={"zipmap": False})
    with open(params["path"]["final_models"]["onnx_model"], "wb") as fp:
        fp.write(onx.SerializeToString())
