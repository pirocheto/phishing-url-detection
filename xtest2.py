from datetime import datetime
from pprint import pprint

import dill
import numpy as np
import onnxruntime
import pandas as pd
import yaml

# Charger le modèle ONNX
onnx_model_path = "models/model.onnx"
sess = onnxruntime.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

pkl_model_path = "models/model.pkl"
with open(pkl_model_path, "rb") as fp:
    pkl_model = dill.load(fp)

with open("results/selected_features.yaml", "r") as fp:
    selected_features = yaml.safe_load(fp)

df = pd.read_csv("data/all.csv")
X_test = df[selected_features]

inputs = X_test.astype(np.float32).to_numpy()


start = datetime.now()
# Effectuer des prédictions avec le modèle ONNX
output = sess.run(["output_probability"], {"X": inputs})
time_onnx = datetime.now() - start


start = datetime.now()
# Effectuer des prédictions avec le modèle ONNX
output = pkl_model.predict_proba(inputs)
time_pkl = datetime.now() - start


print(time_pkl / time_onnx)

# # Les prédictions sont dans la variable 'output'. Vous pouvez les utiliser comme nécessaire.
# for result in output[0]:
#     print(result[1])
