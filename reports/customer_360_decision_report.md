# Customer 360 Decision Engine Report

## Overview

This report combines churn prediction, two-stage LTV prediction, and customer segmentation to generate customer-level business recommendations.

## Summary

- Customers scored: 9,898
- Total revenue at risk: $2,465,374.37
- Average churn probability: 76.29%
- Average predicted LTV: $333.31
- Recommended action types: 7

## Recommended Action Summary

| recommended_action               |   customers |   total_revenue_at_risk |   avg_churn_probability |   avg_predicted_ltv |
|:---------------------------------|------------:|------------------------:|------------------------:|--------------------:|
| Personalized Win-Back Offer      |         362 |             1.07798e+06 |                  0.9858 |           3018.75   |
| Targeted Discount Campaign       |         378 |        543236           |                  0.9665 |           1498.55   |
| Immediate VIP Retention Outreach |          50 |        333749           |                  0.9954 |           6710.71   |
| Email Re-Engagement Campaign     |        6331 |        323462           |                  0.9749 |             52.4199 |
| Standard Lifecycle Marketing     |        1223 |         79434.3         |                  0.2235 |            310.202  |
| Low-Cost Re-Activation Email     |        1523 |         72148.1         |                  0.2156 |            239.733  |
| VIP Loyalty / Upsell Program     |          31 |         35363.7         |                  0.1668 |           7352.5    |

## Top 10 Highest Priority Customers

|   customerkey | customer_name          | business_segment            |   churn_probability |   predicted_ltv |   revenue_at_risk | recommended_action               |
|--------------:|:-----------------------|:----------------------------|--------------------:|----------------:|------------------:|:---------------------------------|
|        293472 | Bryant Hawkins         | Loyal High-Value Customers  |              0.9952 |        14348.1  |          14279    | Immediate VIP Retention Outreach |
|        185961 | Gabriella MacGillivray | Dormant Low-Value Customers |              0.9986 |        11134.5  |          11118.9  | Immediate VIP Retention Outreach |
|        205574 | Ralph Rich             | Loyal High-Value Customers  |              0.9997 |         9820.7  |           9818.17 | Immediate VIP Retention Outreach |
|        112076 | Jeremy Massola         | Dormant Low-Value Customers |              1      |         9590.81 |           9590.81 | Immediate VIP Retention Outreach |
|         80933 | Jasmine Lyons          | Dormant Low-Value Customers |              0.9126 |         9964.43 |           9093.86 | Immediate VIP Retention Outreach |
|        157702 | Edward Millen          | Dormant Low-Value Customers |              0.9956 |         8698.8  |           8660.89 | Immediate VIP Retention Outreach |
|        245088 | Diane Gist             | Dormant Low-Value Customers |              0.9997 |         8595.66 |           8593.02 | Immediate VIP Retention Outreach |
|        411578 | Tanja Vogel            | Loyal High-Value Customers  |              0.9937 |         8612.7  |           8558.06 | Immediate VIP Retention Outreach |
|         32695 | Jordan Woollard        | Dormant Low-Value Customers |              1      |         8340.39 |           8340.39 | Immediate VIP Retention Outreach |
|        123879 | Chloe Cox              | Dormant Low-Value Customers |              0.9939 |         8343.98 |           8293.39 | Immediate VIP Retention Outreach |
