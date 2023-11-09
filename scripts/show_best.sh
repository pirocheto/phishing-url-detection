#!/bin/bash

# Number of experiments to display
NB_EXPS=10

# Columns to display
columns=(
    "Experiment"
    "precision"
    "recall"
    "f1-score"
    "accuracy"
)

# Convert columns into a regular expression for filtering metrics
columns_regex=$(echo "${columns[*]}" | tr ' ' '|')

# Use dvc exp show to obtain filtered and sorted metrics
metrics=$(dvc exp show --drop "^(?!($columns_regex)$).*" --sort-by f1-score --sort-order desc)

# Calculate the number of lines to display (including headers)
nb_lines=$((NB_EXPS + 5))

# Display metrics with headers and the specified number of experiments
echo "$metrics" | head -n $nb_lines