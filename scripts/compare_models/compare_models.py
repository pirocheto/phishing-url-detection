import subprocess
import time

FORCE = False
N_JOBS = 5

classifiers = [
    "GaussianNB",
    "LinearSVC",
    "LogisticRegression",
    "RandomForest",
    "TreeDecisionClassifier",
    "Perceptron",
    "XGBoostClassifier",
    "GradientBoostingClassifier",
    "AdaBoostClassifier",
    "ExtraTreesClassifier",
    "KNeighborsClassifier",
    "RidgeClassifier",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "LGBMClassifier",
    "CatBoostClassifier",
]

force_flag = "--force" if FORCE else ""
for classifier in classifiers:
    command = f"dvc exp run -n base-{classifier} -S classifier={classifier} --queue {force_flag}"
    subprocess.run(command, shell=True, check=True)

subprocess.run(f"dvc queue start --jobs {N_JOBS}", shell=True, check=True)
