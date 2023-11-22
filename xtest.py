import pprint

import numpy
import skl2onnx.sklapi.register  # noqa
from numpy.testing import assert_almost_equal
from onnxruntime import InferenceSession
from skl2onnx import to_onnx
from skl2onnx.sklapi import TraceableTfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = numpy.array(
    [
        "This is the first document.",
        "This document is the second document.",
        "Is this the first document?",
        "",
    ]
)


mod1 = TfidfVectorizer(ngram_range=(1, 2))
mod1.fit(corpus)


mod2 = TraceableTfidfVectorizer(ngram_range=(1, 2))
mod2.fit(corpus)

pprint.pprint(mod2.vocabulary_)
