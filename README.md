<div align="center">
    <h1 align="center">
        Phishing URL Detection<br>
        <em style="font-size: 18px;color:grey">with</em>
        <em style="font-size: 22px;color:grey">Machine Learning</em>
    </h1>
</div>

This repository contains the code for training a machine learning model for phishing URL detection.
The dataset used and the latest model are hosted on Hugging Face:

- Dataset: https://huggingface.co/datasets/pirocheto/phishing-url
- Model: https://huggingface.co/pirocheto/phishing-url-detection

> ℹ️ You can test the model on the demo page [here](https://pirocheto.github.io/phishing-url-detection/).

## Consideration Regarding The Model

The model architecture consists of a TF-IDF (character n-grams + word n-grams) for vectorization and a linear SVM for classification.

:white_check_mark: **Lightweight**: Easy to handle, you can embed it in your applications without the need for a remote server to host it.

:white_check_mark: **Fast**: Your application will experience no additional latency due to model inferences.

:white_check_mark: **Works Offline**: The use of URL tokens alone enables usage without an internet connection.

On the other hand, it could be less efficient than more complex models or those using external features.

## Reproduce The Model

```bash
# 1. Clone the repository
git clone https://github.com/pirocheto/phishing-url-detection.git

# 2. Go inside the project
cd phishing-url-detection

# 3. Install dependencies
poetry install --no-root

# 4. Run the pipeline
dvc repro -s download_data
dvc repro -s train
```

For more details, see the pipeline in the [dvc.yaml](dvc.yaml) file.

## Project Structure

- `live`: Artifacts created during pipeline execution
- `notebooks`: Contains the code for the exploration phase
- `ressources`: Miscellaneous resources used by scripts
- `tests`: Test files
- `src`: Python scripts
- `params.yaml`: Parameters for the DVC experiment
- `dvc.yaml`: Pipeline to run the experiment and reproduce executions

## Main Tools Used in This Project

- [DVC](https://dvc.org/): Version data and experiments
- [CML](https://cml.dev/): Post a comment to the pull request showing the metrics and parameters of an experiment
- [Scikit-Learn](https://scikit-learn.org/stable/): Framework to train the model
- [Optuna](https://optuna.readthedocs.io/en/stable/): Find the best hyperparameters for model
