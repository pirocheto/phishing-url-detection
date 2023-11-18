import numpy as np
import onnxruntime
import pandas as pd

# Defining a list of URLs with characteristics
urls = [
    {
        "url": "https://www.rga.com/about/workplace",
        "nb_hyperlinks": 97.0,
        "ratio_intHyperlinks": 0.969072165,
        "ratio_extHyperlinks": 0.030927835,
        "ratio_extRedirection": 0.0,
        "safe_anchor": 25.0,
        "domain_registration_length": 3571.0,
        "domain_age": 11039,
        "web_traffic": 178542.0,
        "google_index": 0.0,
        "page_rank": 5,
    },
]

# Initializing the ONNX Runtime session with the pre-trained model
sess = onnxruntime.InferenceSession(
    "models/model.onnx",
    providers=["CPUExecutionProvider"],
)

# Creating a DataFrame from the list of URLs
df = pd.DataFrame(urls)
df = df.set_index("url")

# Converting DataFrame data to a float32 NumPy array
inputs = df.astype(np.float32).to_numpy()


# Using the ONNX model to make predictions on the input data
probas = sess.run(None, {"X": inputs})[1]


# Displaying the results
for url, proba in zip(urls, probas):
    print(proba)
    print(f"URL: {url['url']}")
    print(f"Likelihood of being a phishing site: {proba[1] * 100:.2f}%")
    print("----")

# output:
# URL: https://www.rga.com/about/workplace
# Likelihood of being a phishing site: 0.89%
# ----
