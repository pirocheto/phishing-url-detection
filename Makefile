
.ONESHELL:
load_data:
	mkdir -p data
	dvc import-url https://huggingface.co/datasets/pirocheto/phishing-url/resolve/main/data/train.parquet data/train.parquet
	dvc import-url https://huggingface.co/datasets/pirocheto/phishing-url/resolve/main/data/test.parquet data/test.parquet

modelcard:
	python src/modelcard.py

optuna_dashboard:
	optuna-dashboard notebooks/optunalog/optuna.db