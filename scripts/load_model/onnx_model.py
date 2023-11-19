import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download

# REPO_ID = "pirocheto/phishing-url-detection"
# FILENAME = "model.onnx"
# model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

model_path = "./models/model.onnx"

# Initializing the ONNX Runtime session with the pre-trained model
sess = onnxruntime.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"],
)

# Defining a list of URLs with characteristics
urls = ["https://www.rga.com/about/workplace"]


inputs = np.array(urls, dtype="str")

# Using the ONNX model to make predictions on the input data
probas = sess.run(None, {"X": inputs})[1]

# Displaying the results
for url, proba in zip(urls, probas):
    print(f"URL: {url}")
    print(f"Likelihood of being a phishing site: {proba[1] * 100:.2f}%")
    print("----")

# Expected output:
# URL: https://www.rga.com/about/workplace
# Likelihood of being a phishing site: 0.25%
# ----
