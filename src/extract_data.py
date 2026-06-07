"""Extract ML-ready data from PostgreSQL.

Run from project root:
    python src/extract_data.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import CUSTOMER_FEATURE_VIEW, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.database import read_sql, test_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


CUSTOMER_FEATURE_QUERY = f"""
SELECT *
FROM {CUSTOMER_FEATURE_VIEW};
"""


TRANSACTION_SAMPLE_QUERY = """
SELECT
    s.customerkey,
    s.orderkey,
    s.orderdate,
    s.quantity,
    s.netprice,
    s.exchangerate,
    s.productkey
FROM public.sales s
ORDER BY s.orderdate
LIMIT 100000;
"""


def validate_customer_features(df: pd.DataFrame) -> None:
    """Basic validation checks for customer ML features."""
    if df.empty:
        raise ValueError("customer_ml_features returned 0 rows. Check whether the SQL view was created correctly.")

    required_columns = {
        "customerkey",
        "recency_days",
        "frequency",
        "monetary_value",
        "is_churned_180d",
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns from customer_ml_features: {sorted(missing_columns)}")

    duplicate_customers = df["customerkey"].duplicated().sum()
    if duplicate_customers > 0:
        raise ValueError(f"customer_ml_features should have 1 row per customer, but found {duplicate_customers} duplicates.")

    logger.info("Validation passed: customer_ml_features is non-empty and customer-level.")


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Save dataframe as CSV and Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = output_path.with_suffix(".csv")
    parquet_path = output_path.with_suffix(".parquet")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    logger.info("Saved CSV: %s", csv_path)
    logger.info("Saved Parquet: %s", parquet_path)


def extract_customer_features() -> pd.DataFrame:
    """Extract ML-ready customer features from PostgreSQL."""
    logger.info("Extracting customer features from %s", CUSTOMER_FEATURE_VIEW)
    df = read_sql(CUSTOMER_FEATURE_QUERY)
    validate_customer_features(df)
    save_dataframe(df, PROCESSED_DATA_DIR / "customer_ml_features")
    return df


def extract_transaction_sample() -> pd.DataFrame:
    """Extract a transaction sample for EDA sanity checks."""
    logger.info("Extracting transaction sample from public.sales")
    df = read_sql(TRANSACTION_SAMPLE_QUERY)
    save_dataframe(df, RAW_DATA_DIR / "sales_transaction_sample")
    return df


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print simple dataset summary."""
    print("\n" + "=" * 80)
    print("CUSTOMER ML FEATURES SUMMARY")
    print("=" * 80)
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Unique customers: {df['customerkey'].nunique():,}")

    if "is_churned_180d" in df.columns:
        churn_rate = df["is_churned_180d"].mean()
        print(f"Churn rate 180d: {churn_rate:.2%}")

    numeric_cols = [
        col for col in ["recency_days", "frequency", "monetary_value", "avg_order_value"]
        if col in df.columns
    ]

    if numeric_cols:
        print("\nNumeric feature summary:")
        print(df[numeric_cols].describe().round(2))

    print("=" * 80 + "\n")


def main() -> None:
    logger.info("Testing database connection...")
    if not test_connection():
        raise RuntimeError("Database connection failed.")
    logger.info("Database connection successful.")

    customer_features = extract_customer_features()
    print_dataset_summary(customer_features)

    # Optional but useful for Phase 3 EDA
    extract_transaction_sample()

    logger.info("Phase 2 extraction complete.")


if __name__ == "__main__":
    main()
