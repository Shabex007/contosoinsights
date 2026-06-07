# Phase 3 EDA Summary

## Dataset Overview

- Total customers: 49,487
- Total revenue: $206,407,538.58
- Average revenue per customer: $4,170.94
- Churn rate 180d: 86.03%
- Median purchase frequency: 1

## Segment Summary

| historical_ltv_segment   |   customers |     revenue |   churn_rate |   avg_revenue |
|:-------------------------|------------:|------------:|-------------:|--------------:|
| High-Value               |       12372 | 1.35429e+08 |         0.83 |      10946.4  |
| Mid-Value                |       24743 | 6.66365e+07 |         0.86 |       2693.14 |
| Low-Value                |       12372 | 4.34181e+06 |         0.9  |        350.94 |

## Key Initial Insights

1. The customer base has a high 180-day churn rate.
2. Revenue is highly skewed, meaning a smaller group of customers contributes disproportionately to sales.
3. Many customers purchase only once, making repeat purchase behavior a key modeling signal.
4. LTV segment, recency, frequency, and monetary value should be strong predictors for churn and customer value.
