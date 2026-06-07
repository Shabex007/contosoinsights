# Churn Model Report

## Model Comparison

| model_name          |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|
| logistic_regression |      0.855 |      0.9972 |   0.8338 | 0.9082 |    0.9708 |
| random_forest       |      0.816 |      0.9967 |   0.7888 | 0.8806 |    0.9513 |

## Selection Criteria

The best churn model was selected using ROC-AUC as the primary metric. Recall and precision were also reviewed because churn prediction is an imbalanced classification problem.
