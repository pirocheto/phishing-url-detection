import pickle
from pathlib import Path

import pandas as pd
import typer
import yaml
from model import create_model
from skl2onnx import to_onnx
from skl2onnx.common.data_types import StringTensorType
from sklearn.preprocessing import LabelEncoder
from typing_extensions import Annotated


def load_data(path):
    df_train = pd.read_csv(path)
    X_train = df_train["url"].values
    y_train = df_train["status"].values
    y_train = LabelEncoder().fit_transform(y_train)

    return X_train, y_train


def get_params(path="dvclive/model/params.yaml"):
    with open(path, "r") as fp:
        params = yaml.load(fp, Loader=yaml.Loader)

    return params


def save_model(model, dir="models"):
    models_dir = Path(dir)
    models_dir.mkdir(exist_ok=True)

    onnx_path = models_dir / "model.onnx"
    pkl_path = models_dir / "model.pkl"

    onx = to_onnx(
        model,
        initial_types=[("inputs", StringTensorType((None,)))],
        options={"zipmap": False},
    )

    onnx_path.write_bytes(onx.SerializeToString())
    pkl_path.write_bytes(pickle.dumps(model))


def main(
    data: Annotated[
        str,
        typer.Option(
            "-d",
            help="Path to the CSV file containing training data.",
        ),
    ] = "data/data.csv",
    params: Annotated[
        str,
        typer.Option(
            "-m",
            help="Path to the yaml file containing the params.",
        ),
    ] = "dvclive/model/params.yaml",
    output: Annotated[
        str,
        typer.Option(
            "-o",
            help="Directory to save the trained model files.",
        ),
    ] = "models",
):
    """
    Load a pickle model, train it on the data, then save it in pickle and onnx format.
    """

    X_train, y_train = load_data(data)

    model = create_model(get_params(params))
    model.fit(X_train, y_train)

    save_model(model, output)
    print("Training done.")


if __name__ == "__main__":
    typer.run(main)
