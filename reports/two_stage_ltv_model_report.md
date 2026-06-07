# Two-Stage LTV Prediction Report

## Why Two-Stage Modeling?

The future 6-month revenue target is highly zero-inflated. Only a small percentage of customers generate future revenue, so a single regression model struggles. The two-stage approach first predicts whether a customer will spend, then estimates revenue amount for likely spenders.

## Stage 1: Future Revenue Classifier

| model_name                     |   accuracy |   precision |   recall |     f1 |   roc_auc |
|:-------------------------------|-----------:|------------:|---------:|-------:|----------:|
| ltv_stage1_random_forest       |     0.9339 |      0.4972 |   0.966  | 0.6565 |    0.9912 |
| ltv_stage1_logistic_regression |     0.8836 |      0.3566 |   0.9706 | 0.5216 |    0.9625 |

## Stage 2: Revenue Amount Regressor

| model_name                         |     mae |    rmse |     r2 |
|:-----------------------------------|--------:|--------:|-------:|
| ltv_stage2_ridge_regression        | 1788.96 | 2932.35 | 0.4595 |
| ltv_stage2_random_forest_regressor | 1759.59 | 2980.59 | 0.4416 |

