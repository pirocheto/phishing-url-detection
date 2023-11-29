"""
Module for training a machine learning model with specified hyperparameters.
"""

import dvc.api

from helper import create_model, load_data, save_model


def format_hyperparams(params):
    """Format hyperparameters for the machine learning model."""

    hyperparams = {
        "tfidf__word__ngram_range": (1, params["max_ngram_word"]),
        "tfidf__char__ngram_range": (1, params["max_ngram_char"]),
        "tfidf__word__lowercase": params["lowercase"],
        "tfidf__char__lowercase": params["lowercase"],
        "tfidf__word__use_idf": params["use_idf"],
        "tfidf__char__use_idf": params["use_idf"],
        "cls__method": params["calibration"],
        "cls__estimator__C": params["C"],
        "cls__estimator__loss": params["loss"],
        "cls__estimator__tol": params["tol"],
    }
    return hyperparams


def train():
    """
    Train a machine learning model with specified hyperparameters and save the model.

    Reads hyperparameters and data paths from DVC parameters, creates a model,
    trains the model, and saves the trained model to a specified file.
    """
    params = dvc.api.params_show()
    hyperparams = format_hyperparams(params["hyperparams"])
    model = create_model(hyperparams)

    X_train, y_train = load_data(params["data"]["train"])
    model.fit(X_train, y_train)
    save_model(model, params["model"]["pickle"])


if __name__ == "__main__":
    train()
