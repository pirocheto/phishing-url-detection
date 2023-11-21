load_data:
	mkdir -p data && \
	curl -X GET https://huggingface.co/datasets/pirocheto/phishing-url/raw/main/data.csv -o "data/data.csv"

train:
	python scripts/train_model.py -d data/data.csv -m dvclive/model.pkl -o models

create_modelcard:
	python scripts/create_modelcard.py -o modelcard.md