# LTV Prediction Model Report

## Model Comparison

| model_name              |     mae |    rmse |     r2 |   mape_nonzero_actuals |
|:------------------------|--------:|--------:|-------:|-----------------------:|
| random_forest_regressor | 174.623 |  897.11 | 0.3846 |                 3.4729 |
| linear_regression       | 286.598 | 1057.79 | 0.1444 |                 2.7805 |
| ridge_regression        | 286.745 | 1057.95 | 0.1441 |                 2.7797 |

Best model selected by lowest RMSE: **random_forest_regressor**
