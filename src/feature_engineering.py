from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import BASE_DIR, PROCESSED_DATA_DIR


FEATURES_DIR = BASE_DIR / "data" / "features"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

FEATURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "is_churned_180d"

ID_COLUMNS = [
    "customerkey",
    "customer_name",
]

DATE_COLUMNS = [
    "first_purchase_date",
    "last_purchase_date",
    "dataset_max_order_date",
]

LEAKAGE_COLUMNS = [
    "customer_status_180d",
    "future_6m_revenue",
]

DROP_COLUMNS = (
    ID_COLUMNS
    + DATE_COLUMNS
    + LEAKAGE_COLUMNS
    + [TARGET_COLUMN]
)

BASE_NUMERIC_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "countryfull",
    "historical_ltv_segment",
]

ENGINEERED_NUMERIC_FEATURES = [
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

LOG_TRANSFORM_FEATURES = [
    "monetary_value",
    "total_revenue",
    "total_gross_profit",
    "avg_order_value",
]


def load_customer_features() -> pd.DataFrame:
    """Load the customer-level ML feature dataset created in Phase 2."""
    path = PROCESSED_DATA_DIR / "customer_ml_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract_data` first."
        )

    df = pd.read_parquet(path)
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional ratio and behavior features."""
    df = df.copy()

    df["revenue_per_product"] = (
        df["total_revenue"] / (df["unique_products_bought"] + 1)
    )

    df["revenue_per_category"] = (
        df["total_revenue"] / (df["unique_categories_bought"] + 1)
    )

    df["quantity_per_order"] = (
        df["total_quantity"] / (df["frequency"] + 1)
    )

    df["profit_per_order"] = (
        df["total_gross_profit"] / (df["frequency"] + 1)
    )

    df["product_diversity_ratio"] = (
        df["unique_products_bought"] / (df["total_quantity"] + 1)
    )

    df["category_diversity_ratio"] = (
        df["unique_categories_bought"] / (df["total_quantity"] + 1)
    )

    df["gross_profit_per_product"] = (
        df["total_gross_profit"] / (df["unique_products_bought"] + 1)
    )

    df["orders_per_tenure_day"] = (
        df["frequency"] / (df["customer_tenure_days"] + 1)
    )

    df["revenue_per_tenure_day"] = (
        df["total_revenue"] / (df["customer_tenure_days"] + 1)
    )

    return df


def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace infinite values with NaN, then fill numeric NaNs with 0."""
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    categorical_cols = df.select_dtypes(exclude=np.number).columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    return df


def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Apply log1p transform to highly skewed revenue features."""
    df = df.copy()

    for col in LOG_TRANSFORM_FEATURES:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
            df[col] = np.log1p(df[col])

    return df


def get_available_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that exist in the dataframe."""
    return [col for col in columns if col in df.columns]


def prepare_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Create final X/y datasets before preprocessing."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing: {TARGET_COLUMN}")

    df = add_engineered_features(df)
    df = clean_infinite_values(df)
    df = apply_log_transforms(df)

    y = df[TARGET_COLUMN].astype(int)

    numeric_features = get_available_columns(
        df,
        BASE_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES,
    )

    categorical_features = get_available_columns(
        df,
        CATEGORICAL_FEATURES,
    )

    selected_features = numeric_features + categorical_features
    X = df[selected_features].copy()

    return X, y, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build preprocessing transformer for numeric and categorical columns."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                numeric_features,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def get_transformed_feature_names(
    preprocessor: ColumnTransformer,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    """Get final feature names after preprocessing."""
    feature_names = []

    feature_names.extend(numeric_features)

    if categorical_features:
        encoder = preprocessor.named_transformers_["categorical"]
        categorical_names = encoder.get_feature_names_out(
            categorical_features
        ).tolist()
        feature_names.extend(categorical_names)

    return feature_names


def save_parquet_outputs(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save model-ready train/test datasets."""
    X_train.to_parquet(FEATURES_DIR / "X_train.parquet", index=False)
    X_test.to_parquet(FEATURES_DIR / "X_test.parquet", index=False)

    y_train.to_frame(name=TARGET_COLUMN).to_parquet(
        FEATURES_DIR / "y_train.parquet",
        index=False,
    )

    y_test.to_frame(name=TARGET_COLUMN).to_parquet(
        FEATURES_DIR / "y_test.parquet",
        index=False,
    )


def save_metadata(
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Save feature engineering metadata for documentation and modeling."""
    metadata = {
        "target_column": TARGET_COLUMN,
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "final_feature_count": len(feature_names),
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "final_feature_names": feature_names,
        "dropped_columns": DROP_COLUMNS,
        "log_transformed_features": LOG_TRANSFORM_FEATURES,
        "engineered_features": ENGINEERED_NUMERIC_FEATURES,
    }

    metadata_path = ARTIFACTS_DIR / "feature_engineering_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    print("=" * 80)
    print("PHASE 4: FEATURE ENGINEERING PIPELINE")
    print("=" * 80)

    print("\nLoading customer features...")
    df = load_customer_features()
    print(f"Raw dataset shape: {df.shape}")

    print("\nPreparing features and target...")
    X_raw, y, numeric_features, categorical_features = prepare_features_and_target(df)

    print(f"Initial feature matrix shape: {X_raw.shape}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    print(f"Target churn rate: {y.mean():.2%}")

    print("\nCreating stratified train/test split...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Raw X_train shape: {X_train_raw.shape}")
    print(f"Raw X_test shape: {X_test_raw.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    print("\nFitting preprocessor...")
    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    feature_names = get_transformed_feature_names(
        preprocessor=preprocessor,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    X_train = pd.DataFrame(
        X_train_array,
        columns=feature_names,
    )

    X_test = pd.DataFrame(
        X_test_array,
        columns=feature_names,
    )

    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final X_test shape: {X_test.shape}")
    print(f"Final feature count: {len(feature_names)}")

    print("\nSaving model-ready datasets...")
    save_parquet_outputs(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("Saving preprocessing artifacts...")
    joblib.dump(preprocessor, ARTIFACTS_DIR / "preprocessor.pkl")
    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_names.pkl")

    save_metadata(
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print("\nPhase 4 complete.")
    print(f"Saved feature datasets to: {FEATURES_DIR}")
    print(f"Saved artifacts to: {ARTIFACTS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()