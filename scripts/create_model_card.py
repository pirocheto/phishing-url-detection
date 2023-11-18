import shutil
from pathlib import Path

import joblib
import pandas as pd
import sklearn
import skops
import yaml
from skops import card, hub_utils

dst = Path("hf_model_hub")

try:
    shutil.rmtree(dst)
except FileNotFoundError:
    pass

# Load the pre-trained model
model = joblib.load("models/model.pkl")


df_test = pd.read_csv("data/selected/test.csv", index_col="url")

hub_utils.init(
    model="models/model.pkl",
    task="tabular-classification",
    requirements=[f"scikit-learn={sklearn.__version__}"],
    dst=dst,
    data=df_test[:5],
)

plots_dir = dst / "plots"
plots_dir.mkdir(exist_ok=True)

hub_utils.add_files(
    "./results/classification_report.png",
    "./results/plots/feature_importances.png",
    dst=plots_dir,
)

hub_utils.add_files("models/model.onnx", dst=dst)

# Create a model card object
model_card = card.Card(model, metadata=card.metadata_from_config(dst))

# Set metadata for the model card
model_card.metadata.license = "mit"
# model_card.metadata.tags = ["classification", "phishing"]
# model_card.metadata.library_name = "sklearn"
# model_card.metadata.pipeline_tag = "tabular-classification"


# Add plots to the model card
model_card.add_plot(
    **{
        "Model description/Test Report": "plots/classification_report.png",
        "Model description/Model Interpretation/Feature Importances": "plots/feature_importances.png",
    }
)

# Read code snippets from files for model deployment
with open("scripts/load_model/onnx_model.py") as fp:
    onnx_py_code = fp.read()

with open("scripts/load_model/onnx_model.js") as fp:
    onnx_js_code = fp.read()

with open("scripts/load_model/joblib_model.py") as fp:
    joblib_code = fp.read()

# Add code snippets to the model card
model_card.add(
    **{
        "Model description": "",
        "Model description/Training Procedure": "",
        "How to Get Started with the Model": "Below are some code snippets to load the model.",
        "How to Get Started with the Model/With joblib (not recommended)": f"```python  \n{joblib_code}  \n```",
        "How to Get Started with the Model/With ONNX (recommended)/Python": f"```python  \n{onnx_py_code}  \n```",
        "How to Get Started with the Model/With ONNX (recommended)/JavaScript": f"```javascript  \n{onnx_js_code}  \n```",
    },
)

# Add a plot describing the model architecture
model_card.add_model_plot(
    description="This is the architecture of the model loaded by joblib."
)

# Load metrics from a YAML file
with open("results/metrics.yaml", "r") as fp:
    metrics = yaml.safe_load(fp)

# Add metrics to the model card
model_card.add_metrics(**metrics["test"])

# Delete unnecessary sections from the model card
model_card.delete("Model description/Intended uses & limitations")
model_card.delete("Model Card Authors")
model_card.delete("Model Card Contact")
model_card.delete("Citation")

# Save the model card
model_card.save(Path(dst) / "README.md")
