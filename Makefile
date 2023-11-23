
load_data:
	mkdir -p data && \
	dvc import-url https://huggingface.co/datasets/pirocheto/phishing-url/resolve/main/data.csv data/data.csv

train:
	python scripts/train_model.py -d data/data.csv -p dvclive/model/params.yaml -o models

modelcard:
	mkdir -p models
	python scripts/create_modelcard.py -o models/README.md

exp.save:
	dvc exp save -n svm-opt -f

exp.purge:
	dvc exp remove -A; \
    [ -e "optunalog/optuna.db" ] && rm "optunalog/optuna.db"

optuna_dashboard:
	optuna-dashboard optunalog/optuna.db