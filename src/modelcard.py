import json
from pathlib import Path

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


def load_code() -> dict:
    """Load code snippets from specified files."""

    path = Path("resources/modelcard/scripts")

    code = {
        "py": {
            "onnx": (path / "python/load_onnx.py").read_text(),
            "pkl": (path / "python/load_pickle.py").read_text(),
        },
        "js": (path / "javascript/index.html").read_text(),
        "node": (path / "nodejs/index.js").read_text(),
    }

    return code


def render_modelcard(metrics: str, code: dict) -> str:
    """Render the model card using a Jinja2 template."""
    template_str = Path("resources/modelcard/template.md.j2").read_text()
    template = Template(template_str)

    params = {"metrics": metrics, "code": code}
    return template.render(params)


def create_modelcard() -> None:  # pragma: no cover
    """Main function to generate and save the model card."""
    metrics = load_metrics("live/metrics.json")
    code = load_code()

    modelcard_str = render_modelcard(metrics, code)
    Path("live/model/README.md").write_text(modelcard_str)


if __name__ == "__main__":
    create_modelcard()
