import pickle
from pathlib import Path

import pandas as pd
import typer
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


def load_model(path):
    model_path = "dvclive/model.pkl"
    with open(model_path, "rb") as fp:
        model = pickle.load(fp)

    return model


def save_model(model, dir="models"):
    models_dir = Path(dir)
    models_dir.mkdir(exist_ok=True)

    onx = to_onnx(
        model,
        initial_types=[("inputs", StringTensorType((None,)))],
    )

    with open(models_dir / "model.onnx", "wb") as fp:
        fp.write(onx.SerializeToString())

    with open(models_dir / "model.pkl", "wb") as fp:
        pickle.dump(model, fp)


def main(
    data: Annotated[
        str, typer.Option("-d", help="Path to the CSV file containing training data.")
    ] = "data/data.csv",
    model: Annotated[
        str,
        typer.Option(
            "-m", help="Path to the pickle file containing the trained model."
        ),
    ] = "dvclive/model.pkl",
    output: Annotated[
        str, typer.Option("-o", help="Directory to save the trained model files.")
    ] = "models",
):
    """
    Load a pickle model, train it on the data, then save it in pickle and onnx format.
    """

    X_train, y_train = load_data(data)
    model = load_model(model)
    model.fit(X_train, y_train)
    save_model(model, output)

    print("Training done.")


if __name__ == "__main__":
    typer.run(main)
