import pickle
from pathlib import Path

model_path = Path(__file__).parents[2] / "models/model.pkl"

urls = [
    "https://www.rga.com/about/workplace",
    "http://www.iracing.com/tracks/gateway-motorsports-park/",
]

with open(model_path, "rb") as fp:
    model = pickle.load(fp)


probas = model.predict_proba(urls)


# Displaying the results
for url, proba in zip(urls, probas):
    print(f"URL: {url}")
    print(f"Likelihood of being a phishing site: {proba[1] * 100:.2f} %")
    print("----")
