"""
Module for generating a model card by loading metrics and code snippets.
"""

import json
from pathlib import Path

import yaml
from jinja2 import Template
from tabulate import tabulate


def get_metrics(path: str) -> str:
    metrics = json.loads(Path(path).read_text("utf8"))

    return tabulate(
        metrics.items(),
        headers=["Metric", "Value"],
        tablefmt="github",
    )


def get_hyperparams(path: str, key: str = "hyperparams") -> str:
    params = yaml.safe_load(Path(path).read_text("utf8"))
    hyperparams = params[key]

    return tabulate(
        hyperparams.items(),
        headers=["Params", "Value"],
        tablefmt="github",
    )


def get_sizes(paths: list[str]) -> str:
    sizes = {}

    for path in paths:
        path = Path(path)
        name = path.name
        size = path.stat().st_size / (1024**2)
        sizes[name] = size

    return tabulate(
        sizes.items(),
        headers=["File", "Size (Mo)"],
        tablefmt="github",
    )


def get_plots(paths: list[str]) -> str:
    plots = "\n".join([f"![]({path})" for path in paths])
    return plots


def render_report(metrics: str, hyperparams: str, sizes: str, plots: str) -> str:
    template_str = Path("resources/templates/report.md.j2").read_text("utf8")
    template = Template(template_str)

    params = {
        "metrics": metrics,
        "hyperparams": hyperparams,
        "sizes": sizes,
        "plots": plots,
    }
    return template.render(params)


def create_report() -> None:  # pragma: no cover
    """Main function to generate and save the model card."""

    metrics = get_metrics("live/metrics.json")
    hyperparams = get_hyperparams("params.yaml")
    sizes = get_sizes(["live/model/model.onnx", "live/model/model.pkl"])
    plots = get_plots(["live/images/confusion_matrix.png"])

    report_str = render_report(metrics, hyperparams, sizes, plots)
    Path("report.md").write_text(report_str, "utf8")


if __name__ == "__main__":
    create_report()
