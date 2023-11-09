FORCE=true
N_JOBS=5

classifiers=(
    "GaussianNB"
    "LinearSVC"
    "LogisticRegression"
    "RandomForest"
    "TreeDecisionClassifier"
    "Perceptron"
    "XGBoostClassifier"
    "GradientBoostingClassifier"
    "AdaBoostClassifier"
    "ExtraTreesClassifier"
    "KNeighborsClassifier"
    "RidgeClassifier"
    "LinearDiscriminantAnalysis"
    "QuadraticDiscriminantAnalysis"
    "LGBMClassifier"
    "CatBoostClassifier"
)


force_flag="$([ "$FORCE" = true ] && echo '--force' || echo '')"
for classifier in "${classifiers[@]}"; do
    dvc exp run --queue -S classifier="$classifier" -n "$classifier" "$force_flag"
done

dvc queue start --jobs $N_JOBS