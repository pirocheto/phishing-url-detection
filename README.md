<div align="center">
<h1 align="center">
Phishing URL Detection <br>
<em style="font-size: 18px;color:grey">with</em>
<em style="font-size: 22px;color:grey">Machine Learning</em>
</h1>
</div>

This repository contains the code for training a machine learning model for phishing URL detection.

## Reproduce The Model

```bash
# 1. Clone the repository
git clone https://github.com/pirocheto/phishing-url-detection.git

# 2. Go inside the project
cd phishing-url-detection

# 3. Install dependencies
poetry install --no-root

# 4. Run the pipeline
dvc repro train_model
```

For more details, see the pipeline in the [dvc.yaml](dvc.yaml) file.

## Useful Links

The dataset used and the latest model are hosted on Hugging Face:

- Dataset: https://huggingface.co/datasets/pirocheto/phishing-url
- Model: https://huggingface.co/pirocheto/phishing-url-detection

## Main Tools Used in This Project

- [DVC](https://dvc.org/): Version data and experiments
- [CML](https://cml.dev/): Post a comment to the pull request showing the metrics and parameters of an experiment
- [Scikit-Learn](https://scikit-learn.org/stable/): Framework to train the model
- [Optuna](https://optuna.readthedocs.io/en/stable/): Find the best hyperparameters for model
