import json
import pandas as pd

from src.config import BASE_DIR


BUSINESS_DIR = BASE_DIR / "data" / "business"
REPORTS_DIR = BASE_DIR / "reports"


def main() -> None:
    customer_360_path = BUSINESS_DIR / "customer_360_scores.csv"
    metadata_path = REPORTS_DIR / "customer_360_metadata.json"
    action_summary_path = REPORTS_DIR / "business_action_recommendations.csv"
    top_customers_path = REPORTS_DIR / "top_50_customer_priorities.csv"

    if not customer_360_path.exists():
        raise FileNotFoundError(
            "customer_360_scores.csv not found. Run `python -m src.generate_customer_actions` first."
        )

    customer_360 = pd.read_csv(customer_360_path)
    action_summary = pd.read_csv(action_summary_path)
    top_customers = pd.read_csv(top_customers_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=" * 80)
    print("CUSTOMER 360 CHECK")
    print("=" * 80)

    print(f"Customers scored: {metadata['customers_scored']:,}")
    print(f"Total revenue at risk: ${metadata['total_revenue_at_risk']:,.2f}")
    print(f"Average churn probability: {metadata['avg_churn_probability']:.2%}")
    print(f"Average predicted LTV: ${metadata['avg_predicted_ltv']:,.2f}")
    print(f"Actions generated: {metadata['actions_generated']}")

    print("\nCustomer 360 columns:")
    print(list(customer_360.columns))

    print("\nPriority tier distribution:")
    print(customer_360["priority_tier"].value_counts())

    print("\nRecommended action summary:")
    print(action_summary.round(2).to_string(index=False))

    print("\nTop 10 priority customers:")
    display_cols = [
        "customerkey",
        "customer_name",
        "business_segment",
        "churn_probability",
        "predicted_ltv",
        "revenue_at_risk",
        "priority_tier",
        "recommended_action",
    ]

    existing_cols = [col for col in display_cols if col in top_customers.columns]
    print(top_customers[existing_cols].head(10).round(2).to_string(index=False))

    print("=" * 80)


if __name__ == "__main__":
    main()