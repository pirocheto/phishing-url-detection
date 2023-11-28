## Metrics

| Metric    |    Value |
|-----------|----------|
| roc_auc   | 0.986002 |
| accuracy  | 0.949364 |
| f1        | 0.94867  |
| precision | 0.961853 |
| recall    | 0.935843 |

## Hyperparameters

| Params         | Value                 |
|----------------|-----------------------|
| C              | 9.783081707940896     |
| loss           | hinge                 |
| lowercase      | True                  |
| max_ngram_char | 5                     |
| max_ngram_word | 1                     |
| tol            | 0.0003837000703754547 |
| use_idf        | False                 |

## Model size

| File       |   Size (Mo) |
|------------|-------------|
| model.onnx |    11.1121  |
| model.pkl  |     7.18834 |

## Plots

![](live/images/confusion_matrix.png)
![](live/images/calibration_curve.png)
![](live/images/precision_recall_curve.png)
![](live/images/roc_curve.png)
![](live/images/score_distribution.png)