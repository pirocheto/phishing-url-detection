import subprocess
import time

# DVC stage name
STAGE = "test_model"

# Number of jobs for DVC queue processing
N_JOBS = 5

# Flag to remove existing DVC experiments
REMOVE_EXISTING = False

# List of classifiers with their codes
classifiers = [
    ("nb", "GaussianNB"),
    ("svm", "LinearSVC"),
    ("lr", "LogisticRegression"),
    ("rf", "RandomForest"),
    ("dt", "TreeDecisionClassifier"),
    ("pct", "Perceptron"),
    ("xgboost", "XGBoostClassifier"),
    ("gbc", "GradientBoostingClassifier"),
    ("ada", "AdaBoostClassifier"),
    ("et", "ExtraTreesClassifier"),
    ("knn", "KNeighborsClassifier"),
    ("ridge", "RidgeClassifier"),
    ("lda", "LinearDiscriminantAnalysis"),
    ("qda", "QuadraticDiscriminantAnalysis"),
    ("lightgdm", "LGBMClassifier"),
    ("catboost", "CatBoostClassifier"),
]

# Loop over the classifiers
for code, classifier in classifiers:
    exp_name = f"base-{code}"

    # Remove existing DVC experiment if specified
    if REMOVE_EXISTING:
        subprocess.run(f"dvc exp remove {exp_name} -q", shell=True)

    # Build the DVC experiment run command
    cmd = f"dvc exp run {STAGE} -n {exp_name} -S classifier={classifier} --queue"
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
