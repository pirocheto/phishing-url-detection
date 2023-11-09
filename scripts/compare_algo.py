import random
import subprocess

N_JOBS = 5


def generate_experiment_name():
    p1 = random.randint(100, 999)
    p2 = random.randint(100, 999)
    p3 = random.randint(100, 999)
    return f"exp-{p1}.{p2}.{p3}"


algos = [
    "GaussianNB",
    "LinearSVC",
    "LogisticRegression",
    "RandomForest",
]


for classifier in algos:
    exp_name = generate_experiment_name()
    command = f"dvc exp run -n {exp_name} -S classifier={classifier} --queue"
    subprocess.run(command, shell=True, check=True)


subprocess.run(f"dvc queue start --jobs {N_JOBS}", shell=True, check=True)
