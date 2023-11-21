import json
from pathlib import Path

import joblib
import typer
from jinja2 import Template
from tabulate import tabulate
from typing_extensions import Annotated

METRICS_FILE_PATH = "dvclive/metrics.json"
MODEL_FILE_PATH = "dvclive/model/model.pkl"
OUTPUT_DEFAULT_PATH = "modelcard.md"
CODE_PATHS = {
    "py": {
        "onnx": "load_model/python/load_onnx.py",
        "pkl": "load_model/python/load_pickle.py",
    },
    "js": "load_model/javascript/index.html",
    "node": "load_model/nodejs/index.js",
}


def load_metrics(path: str) -> str:
    """Load metrics from a JSON file and format them as a table."""
    with open(path, "r") as file_pointer:
        metrics_data = json.load(file_pointer)
    return tabulate(
        metrics_data.items(), headers=["Metric", "Value"], tablefmt="github"
    )


def load_model_type(path: str) -> str:
    """Load the type of the model from a saved model file."""
    model = joblib.load(path)
    model_type = model[-1].estimator.__class__.__name__
    return model_type


def load_code(code_paths: dict) -> dict:
    """Load code snippets from specified files."""
    code = {}
    for lang, lang_paths in code_paths.items():
        if isinstance(lang_paths, dict):
            code[lang] = {
                key: Path(path).read_text() for key, path in lang_paths.items()
            }
        else:
            code[lang] = Path(lang_paths).read_text()
    return code


def render_model_card(model_type: str, metrics: str, code: dict) -> str:
    """Render the model card using a Jinja2 template."""
    template_str = Path("templates/modelcard.md.j2").read_text()
    template = Template(template_str)

    params = {"model_type": model_type, "metrics": metrics, "code": code}
    return template.render(params)


def write_model_card(modelcard_str: str, output_path: str) -> None:
    """Write the rendered model card to an output file."""
    Path(output_path).write_text(modelcard_str)


def main(
    output: Annotated[
        str, typer.Option("-o", help="Path to save the model card.")
    ] = OUTPUT_DEFAULT_PATH
) -> None:
    """Main function to generate and save the model card."""
    metrics = load_metrics(METRICS_FILE_PATH)
    model_type = load_model_type(MODEL_FILE_PATH)
    code = load_code(CODE_PATHS)

    modelcard_str = render_model_card(model_type, metrics, code)
    write_model_card(modelcard_str, output)


if __name__ == "__main__":
    typer.run(main)
