"""
DVC Metrics Display Script

This script displays experiment metrics using DVC's exp show command.
It allows filtering and sorting metrics based on specified columns.

Script Functionality:
- The show_metrics function displays experiment metrics with specified columns.
- The number of experiments to display can be controlled using the max_exp parameter.

Usage:
1. Adjust the columns variable to include the desired metric columns.
2. Run the script to display experiment metrics.

Note:
- This script assumes that DVC experiments are properly configured in the project.
- Make sure to customize the columns variable based on the metrics available in your experiments.

"""

import subprocess
import sys


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
    n_exps = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    show_metrics(n_exps)
