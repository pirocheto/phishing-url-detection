import onnxruntime
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "pirocheto/phishing-url-detection"
FILENAME = "model.onnx"

# Initializing the ONNX Runtime session with the pre-trained model
sess = onnxruntime.InferenceSession(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME),
    providers=["CPUExecutionProvider"],
)

# Defining a list of URLs with characteristics
data = [
    {
        "url": "https://www.rga.com/about/workplace",
        "nb_hyperlinks": 97,
        "ratio_intHyperlinks": 0.969072165,
        "ratio_extHyperlinks": 0.030927835,
        "ratio_extRedirection": 0,
        "safe_anchor": 25,
        "domain_registration_length": 3571,
        "domain_age": 11039,
        "web_traffic": 178542,
        "google_index": 0,
        "page_rank": 5,
    },
]

# Converting data to a float32 NumPy array
df = pd.DataFrame(data).set_index("url")
inputs = df.to_numpy(dtype="float32")

# Using the ONNX model to make predictions on the input data
probas = sess.run(None, {"X": inputs})[1]

# Displaying the results
for url, proba in zip(data, probas):
    print(f"URL: {url['url']}")
    print(f"Likelihood of being a phishing site: {proba[1] * 100:.2f}%")
    print("----")

# Output:
# URL: https://www.rga.com/about/workplace
# Likelihood of being a phishing site: 0.89%
# ----
