from skl2onnx.sklapi import TraceableTfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def create_model(params=None):
    tfidf = TraceableTfidfVectorizer(analyzer="char")

    classifier = CalibratedClassifierCV(
        LinearSVC(dual="auto"),
        cv=5,
        method="isotonic",
    )

    pipeline = Pipeline([("tfidf", tfidf), ("cls", classifier)])

    if params:
        pipeline.set_params(**params)

    return pipeline
