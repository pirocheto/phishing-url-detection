import subprocess


def show_metrics(max_exp=10):
    # Columns to display
    columns = ["Experiment", "test.f1-score"]

    # Convert columns into a regular expression for filtering metrics
    column_regex = "|".join(columns)

    # Use dvc exp show to obtain filtered and sorted metrics
    cmd = f'dvc exp show --drop "^(?!({column_regex})$).*" --sort-by test.f1-score --sort-order desc'
    process = subprocess.run(
        cmd, shell=True, check=True, capture_output=True, text=True
    )
    metrics = process.stdout
    metrics = metrics.splitlines()

    # Calculate the number of experiments not shown
    exps_not_shown = len(metrics) - max_exp - 6

    # Display metrics with headers and the specified number of experiments
    print(*metrics[0:5], sep="\n")
    print(*metrics[5:-1][:max_exp], sep="\n")
    if exps_not_shown > 0:
        print(f"  ... ({exps_not_shown} experiments not shown)")
    print(metrics[-1])


if __name__ == "__main__":
    show_metrics()
