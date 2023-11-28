download_data:
	dvc repro download_data

pull_model:
	dvc pull live/model/model.pkl live/model/model.onnx

modelcard:
	python src/modelcard.py

report:
	python src/report.py

optuna_dashboard:
	optuna-dashboard notebooks/optunalog/optuna.db

