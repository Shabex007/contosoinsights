from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASE_DIR, PROCESSED_DATA_DIR


TARGET_COLUMN = "future_6m_revenue"

LTV_FEATURES_DIR = BASE_DIR / "data" / "ltv_features"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

LTV_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DROP_COLUMNS = [
    "customerkey",
    "customer_name",
    "future_6m_revenue",
    "dataset_max_order_date",
    "first_purchase_date",
    "last_purchase_date",
    "customer_status_180d",
]

NUMERIC_FEATURES = [
    "age",
    "cohort_year",
    "recency_days",
    "customer_tenure_days",
    "frequency",
    "total_order_lines",
    "total_quantity",
    "unique_products_bought",
    "unique_categories_bought",
    "monetary_value",
    "total_revenue",
    "total_cost",
    "total_gross_profit",
    "gross_margin_pct",
    "avg_order_value",
    "max_order_value",
    "min_order_value",
    "std_order_value",
    "avg_items_per_order",
    "avg_products_per_order",
    "avg_categories_per_order",
    "avg_days_between_orders",
    "std_days_between_orders",
    "purchase_velocity_orders_per_year",
    "revenue_first_6m_after_first_purchase",
    "is_churned_180d",
]

CATEGORICAL_FEATURES = [
    "countryfull",
    "historical_ltv_segment",
]

ENGINEERED_FEATURES = [
    "revenue_per_product",
    "revenue_per_category",
    "quantity_per_order",
    "profit_per_order",
    "product_diversity_ratio",
    "category_diversity_ratio",
    "gross_profit_per_product",
    "orders_per_tenure_day",
    "revenue_per_tenure_day",
]

LOG_FEATURES = [
    "monetary_value",
    "total_revenue",
    "total_cost",
    "total_gross_profit",
    "avg_order_value",
    "max_order_value",
    "min_order_value",
    "revenue_first_6m_after_first_purchase",
]


def load_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "customer_ml_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract_data` first."
        )

    return pd.read_parquet(path)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["revenue_per_product"] = df["total_revenue"] / (df["unique_products_bought"] + 1)
    df["revenue_per_category"] = df["total_revenue"] / (df["unique_categories_bought"] + 1)
    df["quantity_per_order"] = df["total_quantity"] / (df["frequency"] + 1)
    df["profit_per_order"] = df["total_gross_profit"] / (df["frequency"] + 1)
    df["product_diversity_ratio"] = df["unique_products_bought"] / (df["total_quantity"] + 1)
    df["category_diversity_ratio"] = df["unique_categories_bought"] / (df["total_quantity"] + 1)
    df["gross_profit_per_product"] = df["total_gross_profit"] / (df["unique_products_bought"] + 1)
    df["orders_per_tenure_day"] = df["frequency"] / (df["customer_tenure_days"] + 1)
    df["revenue_per_tenure_day"] = df["total_revenue"] / (df["customer_tenure_days"] + 1)

    return df


def clean_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in LOG_FEATURES:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    return df


def build_preprocessor(numeric_features, categorical_features):
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def get_feature_names(preprocessor, numeric_features, categorical_features):
    feature_names = list(numeric_features)

    if categorical_features:
        encoder = preprocessor.named_transformers_["categorical"]
        feature_names.extend(
            encoder.get_feature_names_out(categorical_features).tolist()
        )

    return feature_names


def main() -> None:
    print("=" * 80)
    print("PHASE 6: PREPARE LTV DATASET")
    print("=" * 80)

    df = load_data()
    print(f"Raw dataset shape: {df.shape}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    print("\nTarget summary:")
    print(df[TARGET_COLUMN].describe())
    print(f"Customers with future revenue > 0: {(df[TARGET_COLUMN] > 0).mean():.2%}")

    df = add_engineered_features(df)
    df = clean_values(df)
    df = apply_log_transforms(df)

    available_numeric = [col for col in NUMERIC_FEATURES + ENGINEERED_FEATURES if col in df.columns]
    available_categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]

    X_raw = df[available_numeric + available_categorical].copy()
    y = df[TARGET_COLUMN].clip(lower=0)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = build_preprocessor(available_numeric, available_categorical)

    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names(
        preprocessor,
        available_numeric,
        available_categorical,
    )

    X_train = pd.DataFrame(X_train_array, columns=feature_names)
    X_test = pd.DataFrame(X_test_array, columns=feature_names)

    X_train.to_parquet(LTV_FEATURES_DIR / "X_train_ltv.parquet", index=False)
    X_test.to_parquet(LTV_FEATURES_DIR / "X_test_ltv.parquet", index=False)

    y_train.to_frame(name=TARGET_COLUMN).to_parquet(
        LTV_FEATURES_DIR / "y_train_ltv.parquet",
        index=False,
    )

    y_test.to_frame(name=TARGET_COLUMN).to_parquet(
        LTV_FEATURES_DIR / "y_test_ltv.parquet",
        index=False,
    )

    joblib.dump(preprocessor, ARTIFACTS_DIR / "ltv_preprocessor.pkl")
    joblib.dump(feature_names, ARTIFACTS_DIR / "ltv_feature_names.pkl")

    metadata = {
        "target_column": TARGET_COLUMN,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "customers_with_future_revenue_pct": float((df[TARGET_COLUMN] > 0).mean()),
        "numeric_features": available_numeric,
        "categorical_features": available_categorical,
        "final_feature_names": feature_names,
    }

    with open(ARTIFACTS_DIR / "ltv_feature_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\nLTV feature dataset created.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Saved to: {LTV_FEATURES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()