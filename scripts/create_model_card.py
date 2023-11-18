import shutil
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import yaml
from skops import card, hub_utils

dst = Path("hf_model_hub")

try:
    shutil.rmtree(dst)
except FileNotFoundError:
    pass


model = joblib.load("models/model.pkl")

df_test = pd.read_csv("data/selected/test.csv", index_col="url")

hub_utils.init(
    model="models/model.pkl",
    task="tabular-classification",
    requirements=[
        f"scikit-learn={sklearn.__version__}",
        f"joblib={joblib.__version__}",
    ],
    dst=dst,
    data=df_test.head(5),
)

hub_utils.add_files("models/model.onnx", dst=dst)


# Read code snippets from files for model deployment
with open("scripts/load_model/onnx_model.py") as fp:
    onnx_py_code = fp.read()

# Load metrics from a YAML file
with open("results/metrics.yaml", "r") as fp:
    metrics = yaml.safe_load(fp)


# Create a model card object
model_card = card.Card(
    model,
    metadata=card.metadata_from_config(dst),
    template=None,
)

description = """
The model predicts the probability that a URL is a phishing site using a list of features.

- **Developed by:** [pirocheto](https://github.com/pirocheto)
- **Model type:** Traditional machine learning
- **Task:** Tabular classification (Binary)
- **License:** {{ license }}
- **Repository:** {{ repo }}
"""

get_started_onnx = f"```python\n{onnx_py_code}\n```"

sections = {
    "Model Description": description,
    "How to Get Started with the Model/With ONNX (recommanded)": get_started_onnx,
}


model_card.add_metrics(**metrics["test"])
model_card.add(**sections)

metadata = {
    "license": "mit",
    "inference": False,
    "pipeline_tag": "tabular-classification",
    "tags": [
        "tabular-classification",
        "sklearn",
        "phishing",
        "onnx",
    ],
}

for name, value in metadata.items():
    setattr(model_card.metadata, name, value)


model_card.save(Path(dst) / "README.md")
