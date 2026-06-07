/*
Phase 1: Customer ML Feature View
Project: ContosoInsights
Purpose: Create one row per customer for downstream Data Science / ML tasks:
         - Churn prediction
         - LTV prediction
         - Customer segmentation

Assumptions:
- Existing tables: sales, customer, product
- Existing view: public.cohort_analysis from 0_view_cohort_analysis.sql
- Churn definition: customer has not purchased in the last 180 days from the dataset's latest order date
- Future 6-month revenue is calculated from each customer's first purchase date + 6 months onward.
  This is a starter supervised-learning target. In Phase 2/3 we can improve it with time-based snapshots.
*/

CREATE OR REPLACE VIEW public.customer_ml_features AS
WITH dataset_dates AS (
    SELECT
        MAX(orderdate)::date AS max_order_date,
        MIN(orderdate)::date AS min_order_date
    FROM sales
),
transaction_base AS (
    SELECT
        s.customerkey,
        s.orderkey,
        s.orderdate::date AS orderdate,
        s.quantity,
        s.netprice,
        s.exchangerate,
        s.unitcost,
        s.productkey,
        c.age,
        c.countryfull,
        c.givenname,
        c.surname,
        p.productname,
        p.categoryname,
        (s.quantity * s.netprice * s.exchangerate) AS revenue,
        (s.quantity * s.unitcost * s.exchangerate) AS cost,
        ((s.quantity * s.netprice * s.exchangerate) - (s.quantity * s.unitcost * s.exchangerate)) AS gross_profit
    FROM sales s
    JOIN customer c
        ON c.customerkey = s.customerkey
    LEFT JOIN product p
        ON p.productkey = s.productkey
),
customer_dates AS (
    SELECT
        customerkey,
        MIN(orderdate) AS first_purchase_date,
        MAX(orderdate) AS last_purchase_date,
        EXTRACT(YEAR FROM MIN(orderdate))::int AS cohort_year
    FROM transaction_base
    GROUP BY customerkey
),
order_level AS (
    SELECT
        customerkey,
        orderkey,
        MIN(orderdate) AS orderdate,
        SUM(revenue) AS order_revenue,
        SUM(gross_profit) AS order_gross_profit,
        SUM(quantity) AS order_quantity,
        COUNT(DISTINCT productkey) AS products_in_order,
        COUNT(DISTINCT categoryname) AS categories_in_order
    FROM transaction_base
    GROUP BY customerkey, orderkey
),
customer_agg AS (
    SELECT
        tb.customerkey,
        MAX(TRIM(tb.givenname) || ' ' || TRIM(tb.surname)) AS customer_name,
        MAX(tb.age) AS age,
        MAX(tb.countryfull) AS countryfull,

        COUNT(DISTINCT tb.orderkey) AS total_orders,
        COUNT(*) AS total_order_lines,
        SUM(tb.quantity) AS total_quantity,
        COUNT(DISTINCT tb.productkey) AS unique_products_bought,
        COUNT(DISTINCT tb.categoryname) AS unique_categories_bought,

        SUM(tb.revenue) AS total_revenue,
        SUM(tb.cost) AS total_cost,
        SUM(tb.gross_profit) AS total_gross_profit,
        AVG(tb.revenue) AS avg_line_revenue
    FROM transaction_base tb
    GROUP BY tb.customerkey
),
order_agg AS (
    SELECT
        customerkey,
        AVG(order_revenue) AS avg_order_value,
        MAX(order_revenue) AS max_order_value,
        MIN(order_revenue) AS min_order_value,
        STDDEV(order_revenue) AS std_order_value,
        AVG(order_quantity) AS avg_items_per_order,
        AVG(products_in_order) AS avg_products_per_order,
        AVG(categories_in_order) AS avg_categories_per_order
    FROM order_level
    GROUP BY customerkey
),
inter_purchase AS (
    SELECT
        customerkey,
        AVG(days_between_orders) AS avg_days_between_orders,
        STDDEV(days_between_orders) AS std_days_between_orders
    FROM (
        SELECT
            customerkey,
            orderdate,
            orderdate - LAG(orderdate) OVER (
                PARTITION BY customerkey
                ORDER BY orderdate
            ) AS days_between_orders
        FROM (
            SELECT DISTINCT customerkey, orderdate
            FROM order_level
        ) d
    ) x
    WHERE days_between_orders IS NOT NULL
    GROUP BY customerkey
),
ltv_percentiles AS (
    SELECT
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY total_revenue) AS ltv_25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY total_revenue) AS ltv_75
    FROM customer_agg
),
customer_ltv_segments AS (
    SELECT
        ca.customerkey,
        ca.total_revenue,
        CASE
            WHEN ca.total_revenue < lp.ltv_25 THEN 'Low-Value'
            WHEN ca.total_revenue <= lp.ltv_75 THEN 'Mid-Value'
            ELSE 'High-Value'
        END AS historical_ltv_segment
    FROM customer_agg ca
    CROSS JOIN ltv_percentiles lp
),
future_revenue AS (
    SELECT
        tb.customerkey,
        SUM(
            CASE
                WHEN tb.orderdate > cd.first_purchase_date
                 AND tb.orderdate <= cd.first_purchase_date + INTERVAL '6 months'
                THEN tb.revenue
                ELSE 0
            END
        ) AS revenue_first_6m_after_first_purchase,
        SUM(
            CASE
                WHEN tb.orderdate > cd.first_purchase_date + INTERVAL '6 months'
                 AND tb.orderdate <= cd.first_purchase_date + INTERVAL '12 months'
                THEN tb.revenue
                ELSE 0
            END
        ) AS revenue_month_7_to_12
    FROM transaction_base tb
    JOIN customer_dates cd
        ON cd.customerkey = tb.customerkey
    GROUP BY tb.customerkey
)
SELECT
    ca.customerkey,
    ca.customer_name,
    ca.age,
    ca.countryfull,
    cd.cohort_year,
    cd.first_purchase_date,
    cd.last_purchase_date,

    dd.max_order_date AS dataset_max_order_date,
    (dd.max_order_date - cd.last_purchase_date) AS recency_days,
    (cd.last_purchase_date - cd.first_purchase_date) AS customer_tenure_days,

    ca.total_orders AS frequency,
    ca.total_order_lines,
    ca.total_quantity,
    ca.unique_products_bought,
    ca.unique_categories_bought,

    ca.total_revenue AS monetary_value,
    ca.total_revenue,
    ca.total_cost,
    ca.total_gross_profit,
    CASE
        WHEN ca.total_revenue = 0 THEN 0
        ELSE ca.total_gross_profit / ca.total_revenue
    END AS gross_margin_pct,

    oa.avg_order_value,
    oa.max_order_value,
    oa.min_order_value,
    COALESCE(oa.std_order_value, 0) AS std_order_value,
    oa.avg_items_per_order,
    oa.avg_products_per_order,
    oa.avg_categories_per_order,

    COALESCE(ip.avg_days_between_orders, 9999) AS avg_days_between_orders,
    COALESCE(ip.std_days_between_orders, 0) AS std_days_between_orders,

    CASE
        WHEN (cd.last_purchase_date - cd.first_purchase_date) = 0 THEN ca.total_orders::numeric
        ELSE ca.total_orders::numeric / NULLIF(((cd.last_purchase_date - cd.first_purchase_date)::numeric / 365.0), 0)
    END AS purchase_velocity_orders_per_year,

    cls.historical_ltv_segment,

    CASE
        WHEN cd.last_purchase_date < dd.max_order_date - INTERVAL '180 days' THEN 1
        ELSE 0
    END AS is_churned_180d,

    CASE
        WHEN cd.last_purchase_date < dd.max_order_date - INTERVAL '180 days' THEN 'Churned'
        ELSE 'Active'
    END AS customer_status_180d,

    fr.revenue_first_6m_after_first_purchase,
    fr.revenue_month_7_to_12 AS future_6m_revenue

FROM customer_agg ca
JOIN customer_dates cd
    ON ca.customerkey = cd.customerkey
JOIN order_agg oa
    ON ca.customerkey = oa.customerkey
LEFT JOIN inter_purchase ip
    ON ca.customerkey = ip.customerkey
JOIN customer_ltv_segments cls
    ON ca.customerkey = cls.customerkey
JOIN future_revenue fr
    ON ca.customerkey = fr.customerkey
CROSS JOIN dataset_dates dd;
