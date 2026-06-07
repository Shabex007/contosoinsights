/*
Validation queries for Phase 1 customer_ml_features view.
Run these after executing 4_customer_ml_features.sql.
*/

-- 1. Check total customers and churn rate
SELECT
    COUNT(*) AS total_customers,
    SUM(is_churned_180d) AS churned_customers,
    ROUND(AVG(is_churned_180d)::numeric, 4) AS churn_rate
FROM public.customer_ml_features;

-- 2. Check value segment distribution
SELECT
    historical_ltv_segment,
    COUNT(*) AS customer_count,
    ROUND(SUM(total_revenue)::numeric, 2) AS total_revenue,
    ROUND(AVG(total_revenue)::numeric, 2) AS avg_revenue
FROM public.customer_ml_features
GROUP BY historical_ltv_segment
ORDER BY total_revenue DESC;

-- 3. Check churn by historical value segment
SELECT
    historical_ltv_segment,
    customer_status_180d,
    COUNT(*) AS customer_count,
    ROUND(AVG(total_revenue)::numeric, 2) AS avg_total_revenue,
    ROUND(SUM(total_revenue)::numeric, 2) AS total_revenue
FROM public.customer_ml_features
GROUP BY historical_ltv_segment, customer_status_180d
ORDER BY historical_ltv_segment, customer_status_180d;

-- 4. Check cohort-level ML readiness
SELECT
    cohort_year,
    COUNT(*) AS customers,
    ROUND(AVG(is_churned_180d)::numeric, 4) AS churn_rate,
    ROUND(AVG(total_revenue)::numeric, 2) AS avg_ltv,
    ROUND(AVG(future_6m_revenue)::numeric, 2) AS avg_future_6m_revenue
FROM public.customer_ml_features
GROUP BY cohort_year
ORDER BY cohort_year;

-- 5. Preview ML feature table
SELECT *
FROM public.customer_ml_features
LIMIT 20;
