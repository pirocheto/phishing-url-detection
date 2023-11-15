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


def show_metrics():
    # Columns to display
    columns = [
        "Experiment",
        "test.f1-score",
        "State",
    ]

    # Convert columns into a regular expression for filtering metrics
    column_regex = "|".join(columns)

    # Use dvc exp show to obtain filtered and sorted metrics
    cmd = f'dvc exp show --drop "^(?!({column_regex})$).*" --sort-by test.f1-score --sort-order desc'

    try:
        subprocess.run(cmd, shell=True, check=True)

    except subprocess.CalledProcessError:
        print("DVC processing in progress ... run this script later")


if __name__ == "__main__":
    show_metrics()
