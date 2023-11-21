import json
from pathlib import Path

import joblib
import typer
from jinja2 import Template
from tabulate import tabulate
from typing_extensions import Annotated


def get_metrics(path="dvclive/metrics.json"):
    with open(path, "r") as fp:
        metrics = json.load(fp)

    metrics = tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="github")
    return metrics


def get_model_type(path="dvclive/model/model.pkl"):
    model = joblib.load(path)
    model_type = model[-1].estimator.__class__.__name__
    return model_type


def get_code():
    code_js = Path("load_model/javascript/index.html").read_text()
    code_node = Path("load_model/nodejs/index.js").read_text()
    code_py_onnx = Path("load_model/python/load_onnx.py").read_text()
    code_py_pkl = Path("load_model/python/load_pickle.py").read_text()

    code = {
        "py": {"onnx": code_py_onnx, "pkl": code_py_pkl},
        "js": code_js,
        "node": code_node,
    }
    return code


def main(
    output: Annotated[
        str,
        typer.Option("-o", help="Path to save the model card."),
    ] = "modelcard.md"
):
    """Create the model card based on the project."""

    template_str = Path("templates/modelcard.md.j2").read_text()
    template = Template(template_str)

    model_type = get_model_type()
    metrics = get_metrics()
    code = get_code()

    params = {
        "model_type": model_type,
        "metrics": metrics,
        "code": code,
    }
    modelcard_str = template.render(params)
    Path(output).write_text(modelcard_str)


if __name__ == "__main__":
    typer.run(main)
