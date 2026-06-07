# Customer Segmentation Report

## Method

Customers were segmented using K-Means clustering on RFM and behavioral features. Features were log-transformed where appropriate and standardized before clustering.

## K Selection

|   k |   inertia |   silhouette_score |
|----:|----------:|-------------------:|
|   2 |    345143 |             0.2819 |
|   3 |    276957 |             0.2506 |
|   4 |    244130 |             0.2526 |
|   5 |    220869 |             0.217  |
|   6 |    197916 |             0.199  |
|   7 |    181511 |             0.2013 |
|   8 |    169857 |             0.2054 |

## Segment Profiles

|   cluster |   customers |   avg_recency_days |   avg_frequency |   avg_total_revenue |   total_revenue |   avg_order_value |   avg_tenure_days |   avg_product_diversity |   avg_category_diversity |   churn_rate |   avg_future_6m_revenue |   customer_pct |   revenue_pct | business_segment            |
|----------:|------------:|-------------------:|----------------:|--------------------:|----------------:|------------------:|------------------:|------------------------:|-------------------------:|-------------:|------------------------:|---------------:|--------------:|:----------------------------|
|         1 |       22002 |            688.066 |          2.3495 |             7483.11 |     1.64643e+08 |           3788.43 |         1045.67   |                   6.357 |                   4.1003 |       0.8035 |                358.385  |         0.4446 |        0.7977 | Loyal High-Value Customers  |
|         0 |       27485 |           1231.58  |          1.1438 |             1519.53 |     4.17642e+07 |           1427.6  |           99.8905 |                   2.166 |                   1.8621 |       0.9058 |                 17.2591 |         0.5554 |        0.2023 | Dormant Low-Value Customers |

