FORCE=true
JOBS=5

force_flag="$([ "$FORCE" = true ] && echo '--force' || echo '')"
dvc exp run --queue -S classifier=GaussianNB -n GaussianNB $force_flag
dvc exp run --queue -S classifier=LinearSVC -n LinearSVC $force_flag
dvc exp run --queue -S classifier=LogisticRegression -n LogisticRegression $force_flag
dvc exp run --queue -S classifier=RandomForest -n RandomForest $force_flag
dvc exp run --queue -S classifier=TreeDecision -n TreeDecision $force_flag
dvc queue start --jobs $JOBS

