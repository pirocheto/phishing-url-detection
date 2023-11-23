import json
from pathlib import Path

import joblib
from jinja2 import Template
from tabulate import tabulate


def load_metrics(path: str) -> str:
    """Load metrics from a JSON file and format them as a table."""
    mertrics = json.loads(Path(path).read_text())

    return tabulate(
        mertrics.items(),
        headers=["Metric", "Value"],
        tablefmt="github",
    )


def load_model_type(path: str) -> str:
    """Load the type of the model from a saved model file."""
    model = joblib.load(path)
    model_type = model[-1].estimator.__class__.__name__
    return model_type


def load_code() -> dict:
    """Load code snippets from specified files."""

    code = {
        "py": {
            "onnx": Path("load_model/python/load_onnx.py").read_text(),
            "pkl": Path("load_model/python/load_pickle.py").read_text(),
        },
        "js": Path("load_model/javascript/index.html").read_text(),
        "node": Path("load_model/nodejs/index.js").read_text(),
    }

    return code


def render_model_card(model_type: str, metrics: str, code: dict) -> str:
    """Render the model card using a Jinja2 template."""
    template_str = Path("templates/modelcard.md.j2").read_text()
    template = Template(template_str)

    params = {"model_type": model_type, "metrics": metrics, "code": code}
    return template.render(params)


def create_modelcard(output: str = "modelcard.md") -> None:
    """Main function to generate and save the model card."""
    metrics = load_metrics("dvclive/metrics.json")
    model_type = load_model_type("dvclive/model/model.pkl")
    code = load_code()

    modelcard_str = render_model_card(model_type, metrics, code)
    Path(output).write_text(modelcard_str)


if __name__ == "__main__":
    create_modelcard()
