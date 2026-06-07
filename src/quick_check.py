"""Quick checks after extraction.

Run:
    python src/quick_check.py
"""

from __future__ import annotations

import pandas as pd

from src.config import PROCESSED_DATA_DIR


def main() -> None:
    path = PROCESSED_DATA_DIR / "customer_ml_features.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/extract_data.py` first."
        )

    df = pd.read_csv(path)

    print("\nDataset loaded successfully")
    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values by column:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    if "is_churned_180d" in df.columns:
        print("\nChurn label distribution:")
        print(df["is_churned_180d"].value_counts(normalize=True).round(4))

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()
