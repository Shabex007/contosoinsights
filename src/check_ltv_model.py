import json
import joblib
import pandas as pd

from src.config import BASE_DIR


MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"


def main() -> None:
    best_model_path = MODELS_DIR / "best_ltv_model.pkl"
    metadata_path = REPORTS_DIR / "ltv_model_metadata.json"
    comparison_path = REPORTS_DIR / "ltv_model_comparison.csv"
    predictions_path = PREDICTIONS_DIR / "customer_ltv_predictions_test.csv"

    if not best_model_path.exists():
        raise FileNotFoundError(
            "Best LTV model not found. Run `python -m src.train_ltv_models` first."
        )

    model = joblib.load(best_model_path)
    comparison = pd.read_csv(comparison_path)
    predictions = pd.read_csv(predictions_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=" * 80)
    print("LTV MODEL CHECK")
    print("=" * 80)

    print(f"Best model: {metadata['best_model_name']}")
    print(f"Feature count: {metadata['feature_count']}")
    print(f"Train rows: {metadata['train_rows']:,}")
    print(f"Test rows: {metadata['test_rows']:,}")

    print("\nModel comparison:")
    print(comparison.round(4).to_string(index=False))

    print("\nPrediction sample:")
    print(predictions.head())

    print("\nPrediction summary:")
    print(predictions.describe().round(2))

    print("\nLoaded model type:")
    print(type(model))

    print("=" * 80)


if __name__ == "__main__":
    main()