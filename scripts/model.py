from skl2onnx.sklapi import TraceableTfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def create_model(params=None):
    # tfidf = FeatureUnion(
    #     [
    #         ("w", TraceableTfidfVectorizer(token_pattern=r"\b\w\w+\b")),
    #         ("c", TraceableTfidfVectorizer(analyzer="char")),
    #     ]
    # )
    tfidf = TfidfVectorizer(analyzer="char")

    classifier = CalibratedClassifierCV(
        LinearSVC(dual="auto"),
        cv=5,
        method="isotonic",
    )

    pipeline = Pipeline([("tfidf", tfidf), ("cls", classifier)])

    if params:
        pipeline.set_params(**params)

    return pipeline
