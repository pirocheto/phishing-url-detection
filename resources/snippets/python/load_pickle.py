import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "pirocheto/phishing-url-detection"
FILENAME = "model.pkl"

# Download the model from the Hugging Face Model Hub
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)

urls = [
    "https://clubedemilhagem.com/home.php",
    "http://www.medicalnewstoday.com/articles/188939.php",
]

# Load the downloaded model using joblib
model = joblib.load(model_path)

# Predict probabilities for each URL
probas = model.predict_proba(urls)

for url, proba in zip(urls, probas):
    print(f"URL: {url}")
    print(f"Likelihood of being a phishing site: {proba[1] * 100:.2f} %")
    print("----")
