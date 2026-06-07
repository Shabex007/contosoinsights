import json
import pandas as pd

from src.config import BASE_DIR


REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"


def main() -> None:
    metadata_path = REPORTS_DIR / "two_stage_ltv_metadata.json"
    stage1_path = REPORTS_DIR / "two_stage_ltv_stage1_classifier_comparison.csv"
    stage2_path = REPORTS_DIR / "two_stage_ltv_stage2_regressor_comparison.csv"
    predictions_path = PREDICTIONS_DIR / "customer_two_stage_ltv_predictions_test.csv"

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    stage1 = pd.read_csv(stage1_path)
    stage2 = pd.read_csv(stage2_path)
    predictions = pd.read_csv(predictions_path)

    print("=" * 80)
    print("TWO-STAGE LTV MODEL CHECK")
    print("=" * 80)

    print(f"Best Stage 1 model: {metadata['best_stage1_model']}")
    print(f"Best Stage 2 model: {metadata['best_stage2_model']}")
    print(f"Feature count: {metadata['feature_count']}")
    print(f"Train spender rate: {metadata['train_spender_rate']:.2%}")
    print(f"Test spender rate: {metadata['test_spender_rate']:.2%}")

    print("\nStage 1 comparison:")
    print(stage1.round(4).to_string(index=False))

    print("\nStage 2 comparison:")
    print(stage2.round(4).to_string(index=False))

    print("\nPrediction sample:")
    print(predictions.head())

    print("\nFinal two-stage prediction summary:")
    print(predictions["final_two_stage_ltv_prediction"].describe().round(2))

    print("=" * 80)


if __name__ == "__main__":
    main()