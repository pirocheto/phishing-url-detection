"""
DVC Experiment Automation Script

This script automates the execution of DVC experiments for various classifiers.
It iterates through a list of classifiers, executes DVC experiments, and processes the DVC queue.

Script Configuration:
- STAGE: DVC stage name for the experiments.
- N_JOBS: Number of jobs for DVC queue processing.
- REMOVE_EXISTING: Flag to remove existing DVC experiments.

List of Classifiers:
- classifiers: A list of tuples containing classifier codes and names.

Usage:
1. Configure the script parameters.
2. Run the script to execute DVC experiments for each classifier.

Note:
- This script assumes that the DVC project is properly configured with the specified stage.
- Make sure to adjust the list of classifiers based on your project's requirements.

"""

import subprocess
import time

# DVC stage name
STAGE = "test_model"

# Number of jobs for DVC queue processing
N_JOBS = 5

# Flag to remove existing DVC experiments
REMOVE_EXISTING = True

# List of classifiers with their codes
classifiers = [
    ("nb", "GaussianNB"),
    ("svm", "LinearSVC"),
    ("lr", "LogisticRegression"),
    ("rf", "RandomForest"),
    ("dt", "TreeDecisionClassifier"),
    ("pct", "Perceptron"),
    ("xgboost", "XGBClassifier"),
    ("gbc", "GradientBoostingClassifier"),
    ("ada", "AdaBoostClassifier"),
    ("et", "ExtraTreesClassifier"),
    ("knn", "Normalizer", "KNeighborsClassifier"),
    ("ridge", "RidgeClassifier"),
    ("lda", "LinearDiscriminantAnalysis"),
    ("qda", "QuadraticDiscriminantAnalysis"),
    ("lightgdm", "LGBMClassifier"),
    ("catboost", "CatBoostClassifier"),
]


# Loop over the classifiers
for exp_name, classifier in classifiers:
    # Remove existing DVC experiment if specified
    if REMOVE_EXISTING:
        subprocess.run(f"dvc exp remove {exp_name} -q", shell=True)

    # Build the DVC experiment run command
    cmd = f"dvc exp run {STAGE} -n {exp_name} --queue -S estimator={classifier}"

    try:
        # Execute the DVC experiment run command
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as err:
        print(err)

    print("-" * 15)

# Pause for a short time to allow DVC to process the queue
time.sleep(2)

# Start DVC queue processing with a specified number of jobs
subprocess.run(f"dvc queue start --jobs {N_JOBS}", shell=True, check=True)
