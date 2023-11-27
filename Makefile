
pull_model:
	dvc pull live/model/model.pkl live/model/model.onnx

modelcard:
	python src/modelcard.py

optuna_dashboard:
	optuna-dashboard notebooks/optunalog/optuna.db


test:
	coverage run --source=src -m pytest -s
	coverage report -m