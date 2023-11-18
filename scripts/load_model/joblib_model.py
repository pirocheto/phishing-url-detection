import joblib
from huggingface_hub import hf_hub_download

REPO_ID = "pirocheto/phishing-url-detection"
FILENAME = "model.pkl"

model = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))
