load_data:
	mkdir -p data && \
	curl -X GET https://huggingface.co/datasets/pirocheto/phishing-url/raw/main/data.csv -o "data/data.csv"

train:
	python scripts/train_model.py -d data/data.csv -p dvclive/model/params.yaml -o models

create_modelcard:
	mkdir -p models
	python scripts/create_modelcard.py -o models/README.md

purge_exp:
	dvc exp remove -A && rm optunalog/optuna.db

optuna-dashboard:
	optuna-dashboard optunalog/optuna.db