
pull_model:
	dvc pull model/model.pkl model/model.onnx model/README.md

modelcard:
	python src/modelcard.py

optuna_dashboard:
	optuna-dashboard notebooks/optunalog/optuna.db


test:
	coverage run --source=src -m pytest -s
	coverage report -m