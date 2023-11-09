#!/bin/bash

# Set the output format for dvc exp show (json, csv, md)
FORMAT=""

# Create the format flag based on the specified format (if any)
format_flag="$([ -n "$FORMAT" ] && echo "--$FORMAT")"

# Define the columns to be displayed
columns=(
    "Experiment" 
    "precision" 
    "recall" 
    "f1-score" 
    "accuracy"
)

# Convert the columns into a regular expression for filtering metrics
columns_regex=$(echo "${columns[*]}" | tr ' ' '|')

# Use dvc exp show to obtain filtered metrics based on the specified columns and format
dvc exp show --drop "^(?!($columns_regex)$).*" $format_flag
