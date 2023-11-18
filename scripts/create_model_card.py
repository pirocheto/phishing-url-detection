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

# Read code snippets from files for model deployment
with open("scripts/load_model/onnx_model.js") as fp:
    onnx_js_code = fp.read()

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

- **Model type:** Traditional machine learning
- **Task:** Tabular classification (Binary)
- **License:**: MIT
- **Repository:** https://github.com/pirocheto/phishing-url-detection
"""

get_started = """
Using pickle in Python is discouraged due to security risks during data deserialization, potentially allowing code injection.
It lacks portability across Python versions and interoperability with other languages.

Instead, we recommend using the ONNX model, which is more secure. 
It is half the size and almost twice as fast compared to the pickle model. 
Additionally, it can be utilized by languages supported by the [ONNX runtime](https://onnxruntime.ai/docs/get-started/) (see below for an example using JavaScript).
"""

get_started_onnx_py = f"```python\n{onnx_py_code}\n```"
get_started_onnx_js = f"```javascript\n{onnx_js_code}\n```"


sections = {
    "Model Description": description,
    "How to Get Started with the Model": get_started,
    "How to Get Started with the Model/With ONNX (recommanded)/Python": get_started_onnx_py,
    "How to Get Started with the Model/With ONNX (recommanded)/JavaScript": get_started_onnx_js,
}


model_card.add_metrics("Model Description/Evaluation", **metrics["test"])
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
