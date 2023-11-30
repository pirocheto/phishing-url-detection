# Download data using DVC
download_data:
	dvc repro -s download_data

# Train model using DVC
train:
	dvc repro -s train

# Pull the pickled model and ONNX model using DVC
pull_model:
	dvc pull live/model/model.pkl live/model/model.onnx

# Generate a model card using the src/modelcard.py script
modelcard:
	python src/modelcard.py

# Generate a report using the src/report.py script
report:
	python src/report.py

# Launch the Optuna dashboard using the optuna-dashboard command
optuna_dashboard:
	optuna-dashboard sqlite:///notebooks/optuna.db
