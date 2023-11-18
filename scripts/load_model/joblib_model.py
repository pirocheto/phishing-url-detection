import joblib
import pandas as pd

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


model = joblib.load("models/model.pkl")

df = pd.DataFrame(urls)
df = df.set_index("url")

probas = model.predict_proba(df.values)

for url, proba in zip(urls, probas):
    print(f"URL: {url['url']}")
    print(f"Likelihood of being a phishing site: {proba[1] * 100:.2f}%")
    print("----")


# output:
# URL: https://www.rga.com/about/workplace
# Likelihood of being a phishing site: 0.89%
# ----
