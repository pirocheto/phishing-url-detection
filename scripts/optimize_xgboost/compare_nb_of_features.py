import subprocess
import time

# DVC stage name
STAGE = "test_model"

# Number of jobs for DVC queue processing
N_JOBS = 5

# Flag to remove existing DVC experiments
REMOVE_EXISTING = True

# Specify the classifier
classifier = "XGBClassifier"
feature_selector = "RecursiveFeatureElimination"

# Define the space for the number of features
params = [3, 5, 8, 10, 15, 20, 30, 50]

for nb_features in params:
    # Remove existing DVC experiment if specified
    exp_name = f"xgb-nf-{nb_features}"

    # Remove existing DVC experiment if specified
    if REMOVE_EXISTING:
        subprocess.run(f"dvc exp remove {exp_name} -q", shell=True)

    # Build the DVC experiment run command
    cmd = (
        f"dvc exp run {STAGE} --queue  -n {exp_name}"
        f" -S feature_selection={feature_selector}"
        f" -S feature_selection.param={feature_selector}"
        f" -S classifier={classifier}"
    )

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
