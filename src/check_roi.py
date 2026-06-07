import json
import pandas as pd

from src.config import BASE_DIR


BUSINESS_DIR = BASE_DIR / "data" / "business"
REPORTS_DIR = BASE_DIR / "reports"


def main() -> None:
    customer_roi_path = BUSINESS_DIR / "customer_roi_scores.csv"
    campaign_roi_path = BUSINESS_DIR / "campaign_roi_by_action.csv"
    metadata_path = REPORTS_DIR / "roi_metadata.json"

    if not customer_roi_path.exists():
        raise FileNotFoundError(
            "customer_roi_scores.csv not found. Run `python -m src.roi_calculator` first."
        )

    customer_roi = pd.read_csv(customer_roi_path)
    campaign_roi = pd.read_csv(campaign_roi_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=" * 80)
    print("ROI CALCULATOR CHECK")
    print("=" * 80)

    print(f"Customers scored: {metadata['customers_scored']:,}")
    print(f"Campaigns: {metadata['campaigns']}")
    print(f"Total revenue at risk: ${metadata['total_revenue_at_risk']:,.2f}")
    print(f"Expected saved revenue: ${metadata['total_expected_saved_revenue']:,.2f}")
    print(f"Total campaign cost: ${metadata['total_campaign_cost']:,.2f}")
    print(f"Total net ROI: ${metadata['total_net_roi']:,.2f}")
    print(f"Overall ROI multiple: {metadata['overall_roi_multiple']:.2f}x")

    print("\nCampaign ROI Summary:")
    print(campaign_roi.round(2).to_string(index=False))

    print("\nTop 10 customer ROI opportunities:")
    cols = [
        "customerkey",
        "customer_name",
        "recommended_action",
        "churn_probability",
        "predicted_ltv",
        "revenue_at_risk",
        "expected_saved_revenue",
        "campaign_cost",
        "net_roi",
        "roi_multiple",
    ]

    available_cols = [col for col in cols if col in customer_roi.columns]

    print(
        customer_roi.sort_values("net_roi", ascending=False)[available_cols]
        .head(10)
        .round(2)
        .to_string(index=False)
    )

    print("=" * 80)


if __name__ == "__main__":
    main()