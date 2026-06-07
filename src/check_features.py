import joblib
import pandas as pd

from src.config import BASE_DIR


FEATURES_DIR = BASE_DIR / "data" / "features"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def main() -> None:
    X_train = pd.read_parquet(FEATURES_DIR / "X_train.parquet")
    X_test = pd.read_parquet(FEATURES_DIR / "X_test.parquet")
    y_train = pd.read_parquet(FEATURES_DIR / "y_train.parquet")
    y_test = pd.read_parquet(FEATURES_DIR / "y_test.parquet")
    feature_names = joblib.load(ARTIFACTS_DIR / "feature_names.pkl")

    print("=" * 80)
    print("FEATURE ENGINEERING CHECK")
    print("=" * 80)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    print(f"\nFeature names count: {len(feature_names)}")
    print(f"X_train columns count: {len(X_train.columns)}")

    print("\nTrain churn distribution:")
    print(y_train["is_churned_180d"].value_counts(normalize=True).round(4))

    print("\nTest churn distribution:")
    print(y_test["is_churned_180d"].value_counts(normalize=True).round(4))

    print("\nMissing values in X_train:")
    print(X_train.isna().sum().sum())

    print("\nFirst 10 features:")
    print(list(X_train.columns[:10]))

    print("=" * 80)


if __name__ == "__main__":
    main()