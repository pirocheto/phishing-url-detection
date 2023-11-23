import dvc.api

from helper import create_model, load_data, save_model


def train():
    params = dvc.api.params_show()
    model = create_model(params)
    X_train, y_train = load_data(params["data"])
    model.fit(X_train, y_train)
    save_model(model, "model")


if __name__ == "__main__":
    train()
