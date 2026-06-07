# Contoso Customer Intelligence Platform

![Project Architecture](images/project_architecture.png)

## End-to-End Customer Analytics, Machine Learning, Segmentation, Retention Strategy and ROI Optimization

Contoso Customer Intelligence Platform is an end-to-end data science and machine learning project built on the Microsoft Contoso retail database. The project transforms raw retail transaction data into a complete customer intelligence system that can identify churn risk, forecast future customer value, segment customers, recommend business actions and estimate campaign ROI.

This project is designed as a realistic data science portfolio case study, covering the complete workflow from SQL feature engineering to model training, business decisioning and dashboard delivery.

---

## Table of Contents

- [Business Problem](#business-problem)
- [Solution Overview](#solution-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Highlights](#project-highlights)
- [Repository Structure](#repository-structure)
- [SQL Analytics and Feature Store](#sql-analytics-and-feature-store)
- [Data Extraction Pipeline](#data-extraction-pipeline)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Feature Engineering](#feature-engineering)
- [Churn Prediction](#churn-prediction)
- [Customer Lifetime Value Prediction](#customer-lifetime-value-prediction)
- [Two-Stage LTV Modeling](#two-stage-ltv-modeling)
- [Customer Segmentation](#customer-segmentation)
- [Customer 360 Decision Engine](#customer-360-decision-engine)
- [Campaign ROI Calculator](#campaign-roi-calculator)
- [Streamlit Dashboard](#streamlit-dashboard)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Tech Stack](#tech-stack)
- [Future Improvements](#future-improvements)
- [Resume Bullet](#resume-bullet)

---

## Business Problem

Retail and e-commerce companies need to understand customer behavior before revenue is lost. The core business questions addressed in this project are:

- Which customers are most likely to churn?
- Which customers are expected to generate future revenue?
- Which customers are most valuable to retain?
- Which customer segments should marketing teams prioritize?
- Which retention campaigns produce the best financial return?
- How can model outputs be converted into actionable business decisions?

The goal is not only to build machine learning models, but to connect those models to practical retention and campaign strategy.

---

## Solution Overview

The platform combines multiple layers of analytics and machine learning:

1. SQL analytics and customer-level feature engineering
2. Python data extraction pipeline
3. Exploratory data analysis and visual reporting
4. Leakage-safe feature engineering pipeline
5. Churn classification model
6. Single-stage and two-stage LTV prediction models
7. K-Means customer segmentation
8. Customer 360 scoring engine
9. Campaign ROI calculator
10. Streamlit dashboard for business users

The final output is a customer-level decision table containing:

- Customer segment
- Churn probability
- Predicted LTV
- Revenue at risk
- Priority tier
- Recommended campaign
- Estimated ROI

---

## Architecture

![Advanced Architecture Diagram](images/project_architecture.png)

```text
PostgreSQL Contoso Database
        ↓
SQL customer feature store
        ↓
Python extraction pipeline
        ↓
EDA and feature engineering
        ↓
Machine learning models
        ↓
Customer 360 decision engine
        ↓
Campaign ROI calculator
        ↓
Streamlit dashboard
```

---

## Dataset

**Data Source:** Microsoft Contoso retail database  
**Database:** PostgreSQL  
**Domain:** Retail / E-commerce  
**Customer Records:** 49,487  
**ML Feature Columns:** 35+ raw customer-level features and 40+ model-ready features

The dataset contains:

- Customer profile data
- Sales transactions
- Product catalog information
- Category information
- Geographic customer data
- Historical purchase behavior

---

## Project Highlights

| Area                             | Output |
| -------------------------------- | -----: |
| Customers analyzed               | 49,487 |
| Churn model ROC-AUC              | 0.9708 |
| Churn model F1 score             | 0.9082 |
| Single-stage LTV R²              | 0.3846 |
| Two-stage LTV classifier ROC-AUC | 0.9912 |
| Two-stage LTV regressor R²       | 0.4595 |
| Revenue at risk identified       | $2.46M |
| Expected saved revenue           |  $511K |
| Estimated net ROI                |  $462K |
| ROI multiple                     | 10.56x |

---

## Repository Structure

```text
contosoinsights/
│
├── sql/                         # SQL analytics and feature views
├── src/                         # Python pipeline, ML, ROI, decision engine
├── notebooks/                   # EDA and modeling notebooks
├── dashboard/                   # Streamlit dashboard
├── data/                        # Processed outputs, predictions, business tables
├── models/                      # Saved ML models
├── artifacts/                   # Preprocessors and metadata
├── reports/                     # Model reports, ROI reports, summaries
├── images/                      # Plots, dashboard screenshots, architecture
├── requirements.txt
└── README.md
```

---

# SQL Analytics and Feature Store

The project begins with SQL analytics and customer-level feature engineering in PostgreSQL.

The SQL layer creates customer-level metrics such as:

- Recency
- Frequency
- Monetary value
- Customer tenure
- Total revenue
- Gross profit
- Average order value
- Product diversity
- Category diversity
- Churn label
- Future revenue target

## Customer Segmentation SQL Analysis

![Customer Segmentation SQL Analysis](images/1_customer_segementation.png)

## Cohort Analysis

![Cohort Analysis](images/2_cohort_analysis.png)

## Retention Analysis

![Customer Churn by Cohort Year](images/3_customer_churn_cohort_year.png)

---

# Data Extraction Pipeline

A Python extraction pipeline connects to PostgreSQL using SQLAlchemy, extracts customer-level features and saves data as both CSV and Parquet.

Key outputs:

```text
data/processed/customer_ml_features.csv
data/processed/customer_ml_features.parquet
data/raw/sales_transaction_sample.csv
data/raw/sales_transaction_sample.parquet
```

The pipeline validates:

- Dataset is not empty
- Required columns exist
- One row exists per customer
- Target labels are available

---

# Exploratory Data Analysis

EDA was used to understand customer behavior, revenue concentration, churn imbalance and modeling risks before training machine learning models.

## Churn Distribution

![Churn Distribution](images/eda/01_churn_distribution.png)

## Revenue Distribution

![Revenue Distribution](images/eda/02_ltv_distribution_log.png)

## Revenue by Segment

![Revenue by Segment](images/eda/03_revenue_by_segment.png)

## Churn by Segment

![Churn by Segment](images/eda/04_churn_by_segment.png)

## Cohort Revenue

![Average Revenue by Cohort](images/eda/05_avg_revenue_by_cohort.png)

## Country Revenue Analysis

![Top Countries by Revenue](images/eda/06_top_countries_by_revenue.png)

## Recency vs Frequency

![Recency Frequency Scatter](images/eda/07_recency_frequency_scatter.png)

## Correlation Heatmap

![Correlation Heatmap](images/eda/08_correlation_heatmap.png)

Key EDA findings:

- Churn rate is high under the 180-day inactivity rule.
- Revenue is heavily right-skewed.
- Many customers purchase only once.
- Customer value is concentrated in a smaller group of high-revenue customers.
- Future revenue is highly zero-inflated.

---

# Feature Engineering

A reusable feature engineering pipeline was created for machine learning.

Feature engineering steps:

- Leakage column removal
- Engineered ratio features
- Log transformations for skewed revenue features
- One-hot encoding for categorical variables
- Standard scaling for numeric features
- Stratified train/test split for churn modeling
- Model-ready Parquet datasets
- Preprocessor and feature metadata persistence

Output files:

```text
data/features/X_train.parquet
data/features/X_test.parquet
data/features/y_train.parquet
data/features/y_test.parquet
artifacts/preprocessor.pkl
artifacts/feature_names.pkl
artifacts/feature_engineering_metadata.json
```

Final churn feature matrix:

```text
X_train: 39,589 rows × 44 features
X_test: 9,898 rows × 44 features
```

---

# Churn Prediction

The churn model predicts whether a customer is likely to churn based on behavioral and customer-value features.

Target:

```text
is_churned_180d
```

Models trained:

- Logistic Regression
- Random Forest Classifier

A leakage check was performed during modeling. An initial model produced unrealistic perfect scores because `recency_days` was directly related to the churn label. The feature was removed and the model was retrained to produce a more realistic evaluation.

Best model:

```text
Logistic Regression
```

Performance:

| Metric    |  Value |
| --------- | -----: |
| Accuracy  | 0.8550 |
| Precision | 0.9972 |
| Recall    | 0.8338 |
| F1 Score  | 0.9082 |
| ROC-AUC   | 0.9708 |

## Churn ROC Curve

![Churn ROC Curve](images/modeling/churn_roc_curve.png)

## Churn Confusion Matrix

![Churn Confusion Matrix](images/modeling/churn_confusion_matrix.png)

## Churn Feature Importance

![Churn Feature Importance](images/modeling/churn_feature_importance.png)

> If the best churn model is Logistic Regression, feature importance may not be generated by the current script. This image is optional unless you add coefficient-based importance later.

---

# Customer Lifetime Value Prediction

The LTV model predicts future 6-month revenue.

Target:

```text
future_6m_revenue
```

Models trained:

- Linear Regression
- Ridge Regression
- Random Forest Regressor

Best single-stage model:

```text
Random Forest Regressor
```

Performance:

| Metric |   Value |
| ------ | ------: |
| MAE    | $174.62 |
| RMSE   | $897.11 |
| R²     |  0.3846 |

## LTV Actual vs Predicted

![LTV Actual vs Predicted](images/modeling/ltv_actual_vs_predicted.png)

## LTV Residual Plot

![LTV Residual Plot](images/modeling/ltv_residual_plot.png)

## LTV Feature Importance

![LTV Feature Importance](images/modeling/ltv_feature_importance.png)

---

# Two-Stage LTV Modeling

EDA showed that future revenue is highly zero-inflated:

```text
Only 6.54% of customers generated future 6-month revenue.
```

Because of this, a two-stage LTV modeling strategy was implemented.

Stage 1:

```text
Predict whether the customer will generate future revenue.
```

Stage 2:

```text
Predict revenue amount for customers with non-zero future revenue.
```

Stage 1 best model:

```text
Random Forest Classifier
```

| Metric    |  Value |
| --------- | -----: |
| ROC-AUC   | 0.9912 |
| Recall    | 0.9660 |
| Precision | 0.4972 |
| F1 Score  | 0.6565 |

Stage 2 best model:

```text
Ridge Regression
```

| Metric |     Value |
| ------ | --------: |
| MAE    | $1,788.96 |
| RMSE   | $2,932.35 |
| R²     |    0.4595 |

## Two-Stage LTV Stage 1 ROC

![Two-Stage LTV ROC](images/modeling/two_stage_ltv_stage1_roc.png)

## Two-Stage LTV Confusion Matrix

![Two-Stage LTV Confusion Matrix](images/modeling/two_stage_ltv_stage1_confusion_matrix.png)

## Two-Stage LTV Actual vs Predicted

![Two-Stage LTV Actual vs Predicted](images/modeling/two_stage_ltv_stage2_actual_vs_predicted.png)

---

# Customer Segmentation

K-Means clustering was used to segment customers based on RFM and behavioral features.

Features used:

- Recency
- Frequency
- Monetary value
- Total revenue
- Average order value
- Customer tenure
- Product diversity
- Category diversity
- Purchase velocity
- Gross margin

The model selected two primary clusters based on silhouette score.

Final business segments:

| Segment                      | Customers | Revenue Share | Interpretation                                  |
| ---------------------------- | --------: | ------------: | ----------------------------------------------- |
| At-Risk High-Value Customers |    22,002 |           80% | High-value customers with meaningful churn risk |
| Dormant Low-Value Customers  |    27,485 |           20% | Lower-value customers with high inactivity      |

## K Selection Elbow Curve

![Elbow Curve](images/segmentation/segmentation_elbow_curve.png)

## Silhouette Scores

![Silhouette Scores](images/segmentation/segmentation_silhouette_scores.png)

## PCA Cluster Visualization

![PCA Cluster Visualization](images/segmentation/segmentation_pca_clusters.png)

## Segment Revenue

![Segment Revenue](images/segmentation/segment_revenue.png)

## Segment Churn Rate

![Segment Churn Rate](images/segmentation/segment_churn_rate.png)

---

# Customer 360 Decision Engine

The Customer 360 Decision Engine combines model outputs into customer-level business recommendations.

Inputs:

- Customer profile
- Churn probability
- Predicted LTV
- Customer segment
- Revenue at risk

Generated outputs:

- Churn risk segment
- Predicted LTV segment
- Revenue at risk
- Priority tier
- Recommended action

Customer 360 output:

```text
data/business/customer_360_scores.csv
```

Business actions include:

- Immediate VIP Retention Outreach
- Personalized Win-Back Offer
- Targeted Discount Campaign
- Email Re-Engagement Campaign
- Low-Cost Re-Activation Email
- VIP Loyalty / Upsell Program
- Standard Lifecycle Marketing

## Priority Distribution

![Priority Distribution](images/business/priority_distribution.png)

## Recommended Action Distribution

![Recommended Action Distribution](images/business/recommended_action_distribution.png)

## Segment Revenue at Risk

![Segment Revenue at Risk](images/business/segment_revenue_at_risk.png)

---

# Campaign ROI Calculator

The ROI calculator estimates the expected financial impact of different retention campaigns.

Formula:

```text
revenue_at_risk = churn_probability × predicted_ltv
expected_saved_revenue = revenue_at_risk × retention_success_rate
campaign_cost = targeted_customers × cost_per_customer
net_roi = expected_saved_revenue - campaign_cost
roi_multiple = expected_saved_revenue / campaign_cost
```

Campaign ROI results:

| Metric                 |         Value |
| ---------------------- | ------------: |
| Revenue at risk        | $2,465,374.37 |
| Expected saved revenue |   $511,088.94 |
| Total campaign cost    |    $48,393.00 |
| Net ROI                |   $462,695.94 |
| ROI multiple           |        10.56x |

Best campaign by net ROI:

```text
Personalized Win-Back Offer
Net ROI: $251,395.37
ROI Multiple: 14.89x
```

Best campaign by ROI multiple:

```text
Immediate VIP Retention Outreach
ROI Multiple: 23.36x
```

## Expected Saved Revenue by Campaign

![Expected Saved Revenue by Campaign](images/business/expected_saved_revenue_by_campaign.png)

## Net ROI by Campaign

![Net ROI by Campaign](images/business/net_roi_by_campaign.png)

## ROI Multiple by Campaign

![ROI Multiple by Campaign](images/business/roi_multiple_by_campaign.png)

---

# Streamlit Dashboard

An interactive dashboard was created to make the customer intelligence system usable by business teams.

Run the dashboard:

```bash
streamlit run dashboard/app.py
```

Dashboard pages:

- Executive Overview
- Customer Segmentation
- Churn Prediction
- LTV Forecasting
- Campaign ROI
- Customer Lookup

## Dashboard: Executive Overview

![Dashboard Executive Overview](images/dashboard/dashboard_executive_overview.jpeg)

## Dashboard: Customer Segmentation

![Dashboard Customer Segmentation](images/dashboard/dashboard_segmentation.jpeg)

## Dashboard: Churn Prediction

![Dashboard Churn Prediction](images/dashboard/dashboard_churn.jpeg)

## Dashboard: LTV Forecasting

![Dashboard LTV Forecasting](images/dashboard/dashboard_ltv.jpeg)

## Dashboard: Campaign ROI

![Dashboard Campaign ROI](images/dashboard/dashboard_roi.jpeg)

## Dashboard: Customer Lookup

![Dashboard Customer Lookup](images/dashboard/dashboard_lookup.jpeg)

> If your dashboard screenshots are currently stored directly inside `images/`, move them into `images/dashboard/` and rename them using the names above, or update these paths to match your actual screenshot filenames.

---

# How to Run

## 1. Clone repository

```bash
git clone <your-repo-url>
cd contosoinsights
```

## 2. Create virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure environment variables

Create `.env`:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=contoso
DB_USER=postgres
DB_PASSWORD=your_password
DB_SCHEMA=public
```

## 5. Run data extraction

```bash
python -m src.extract_data
python -m src.quick_check
```

## 6. Run EDA

```bash
python -m src.eda_analysis
```

## 7. Run feature engineering

```bash
python -m src.feature_engineering
python -m src.check_features
```

## 8. Train churn models

```bash
python -m src.train_churn_models
python -m src.check_churn_model
```

## 9. Train LTV models

```bash
python -m src.prepare_ltv_dataset
python -m src.train_ltv_models
python -m src.check_ltv_model
```

## 10. Train two-stage LTV model

```bash
python -m src.train_two_stage_ltv_model
python -m src.check_two_stage_ltv_model
```

## 11. Run segmentation

```bash
python -m src.customer_segmentation
python -m src.check_segmentation
```

## 12. Generate Customer 360 recommendations

```bash
python -m src.generate_customer_actions
python -m src.check_customer_360
```

## 13. Run ROI calculator

```bash
python -m src.roi_calculator
python -m src.check_roi
```

## 14. Launch dashboard

```bash
streamlit run dashboard/app.py
```

---

# Results Summary

| Component                        | Result |
| -------------------------------- | -----: |
| Customers analyzed               | 49,487 |
| Customer-level features          |    35+ |
| Churn model ROC-AUC              | 0.9708 |
| Churn F1 score                   | 0.9082 |
| Single-stage LTV R²              | 0.3846 |
| Two-stage LTV classifier ROC-AUC | 0.9912 |
| Two-stage LTV regressor R²       | 0.4595 |
| Customer segments                |      2 |
| Revenue at risk                  | $2.46M |
| Expected saved revenue           |  $511K |
| Net ROI                          |  $462K |
| ROI multiple                     | 10.56x |

---

# Tech Stack

## Database and SQL

- PostgreSQL
- SQL feature engineering
- SQL analytics views

## Python and Data Science

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## Machine Learning

- Logistic Regression
- Random Forest Classifier
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- K-Means Clustering
- PCA

## Visualization and Dashboarding

- Matplotlib
- Seaborn
- Plotly
- Streamlit

## Project Engineering

- Modular Python scripts
- Reproducible pipelines
- Model artifacts
- Reports and metadata
- Git-friendly project structure

---

# Key Data Science Concepts Demonstrated

- SQL-based feature engineering
- Data extraction from PostgreSQL
- Exploratory data analysis
- Churn modeling
- Leakage detection and correction
- Regression modeling
- Zero-inflated target handling
- Two-stage modeling
- Unsupervised clustering
- Customer segmentation
- Model evaluation
- Business rules engine
- ROI modeling
- Dashboard deployment

---

# Future Improvements

Potential future improvements:

- Add XGBoost and LightGBM models
- Add SHAP explainability for churn and LTV models
- Save customer IDs directly in prediction outputs
- Add MLflow experiment tracking
- Add Airflow orchestration
- Add FastAPI scoring service
- Deploy dashboard to Streamlit Cloud
- Add A/B testing simulation
- Add customer-level intervention history
- Add model monitoring and drift detection

---
