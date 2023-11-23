from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from dvclive import Live
from helper import create_model, exp_show, load_data, save_model, score_model


def compare():
    classifiers = [
        ("svm", LinearSVC(dual="auto")),
        ("lr", LogisticRegression()),
        # ("knn", KNeighborsClassifier()),
        ("nb", MultinomialNB()),
    ]

    X_train, y_train = load_data("data/data.csv")

    for exp_name, classifier in classifiers:
        print(f"Experiment '{exp_name}' in progress...")

        with Live(exp_name=exp_name, save_dvc_exp=True) as live:
            live.log_param("classifier", classifier.__class__.__name__)

            model: Pipeline = create_model({"cls__estimator": classifier})
            claasifier_name = model[-1].estimator.__class__.__name__
            live.log_param("classifier", claasifier_name)

            # Log the model as an artifact using dvclive
            model_path = save_model(model, f"{live.dir}/model")
            live.log_artifact(model_path, type="model", cache=False)

            # Test the model
            scores = score_model(model, X_train, y_train)

            for name, values in scores.items():
                live.log_metric(name, values.mean())

    exp_show()


if __name__ == "__main__":
    compare()
