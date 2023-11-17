# SPDX-License-Identifier: Apache-2.0

"""
Dataframe as an input
=====================

.. index:: dataframe

A pipeline usually ingests data as a matrix. It may be converted in a matrix
if all the data share the same type. But data held in a dataframe
have usually multiple types, float, integer or string for categories.
ONNX also supports that case.

A dataset with categories
+++++++++++++++++++++++++

"""
import pprint

import numpy
from onnx.reference import ReferenceEvaluator
from onnxruntime import InferenceSession
from pandas import DataFrame
from skl2onnx import to_onnx
from skl2onnx.algebra.type_helper import guess_initial_types
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

data = DataFrame(
    [
        dict(CAT1="a", CAT2="c", num1=0.5, num2=0.6, y=0),
        dict(CAT1="b", CAT2="d", num1=0.4, num2=0.8, y=1),
        dict(CAT1="a", CAT2="d", num1=0.5, num2=0.56, y=0),
        dict(CAT1="a", CAT2="d", num1=0.55, num2=0.56, y=1),
        dict(CAT1="a", CAT2="c", num1=0.35, num2=0.86, y=0),
        dict(CAT1="a", CAT2="c", num1=0.5, num2=0.68, y=1),
    ]
)

cat_cols = ["CAT1", "CAT2"]
train_data = data.drop("y", axis=1)


categorical_transformer = Pipeline(
    [("onehot", OneHotEncoder(sparse=False, handle_unknown="ignore"))]
)
preprocessor = ColumnTransformer(
    transformers=[("cat", categorical_transformer, cat_cols)], remainder="passthrough"
)
pipe = Pipeline([("preprocess", preprocessor), ("rf", RandomForestClassifier())])
pipe.fit(train_data, data["y"])


######################################
# It looks ok. Let's dig into the details to
# directly use *onnxruntime*.
#
# Unhide conversion logic with a dataframe
# ++++++++++++++++++++++++++++++++++++++++
#
# A dataframe can be seen as a set of columns with
# different types. That's what ONNX should see:
# a list of inputs, the input name is the column name,
# the input type is the column type.


from skl2onnx.common.data_types import FloatTensorType, StringTensorType

init = [
    ("CAT1", StringTensorType(shape=[None, 1])),
    ("CAT2", StringTensorType(shape=[None, 1])),
    ("num1", FloatTensorType(shape=[None, 1])),
    ("num2", FloatTensorType(shape=[None, 1])),
]

print("--------- INIT -----------")
pprint.pprint(init)

##############################
# Let's convert with *skl2onnx* only.

onx2 = to_onnx(
    pipe, initial_types=init, options={RandomForestClassifier: {"zipmap": False}}
)

#####################################
# Let's run it with onnxruntime.
# We need to convert the dataframe into a dictionary
# where column names become keys, and column values become
# values.

for c in train_data.columns:
    if c not in cat_cols:
        train_data[c] = train_data[c].astype(numpy.float32)


print("------- INPUT ------")
inputs = {c: train_data[c].values.reshape((-1, 1)) for c in train_data.columns}
pprint.pprint(inputs)


#############################
# Inference.

sess2 = InferenceSession(onx2.SerializeToString(), providers=["CPUExecutionProvider"])

got2 = sess2.run(None, inputs)


print(got2[1])
