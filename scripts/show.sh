COLUMNS=(
    "Experiment" 
    "precision" 
    "recall" 
    "f1-score" 
    "accuracy"
)

COLUMNS=$(echo "${COLUMNS[*]}" | tr ' ' '|')

dvc exp show \
    --sort-by f1-score \
    --sort-order desc \
    --drop "^(?!($COLUMNS)$).*"

# results=$(dvc exp show --sort-by f1-score --sort-order desc --drop "^(?!($COLUMNS)$).*" --precision 3 --csv)

# line1=$(echo "$results" | sed -n '1p')
# line3=$(echo "$results" | sed -n '4p')

# echo $line1
# echo $line3