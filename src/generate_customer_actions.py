from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import BASE_DIR, PROCESSED_DATA_DIR


PROCESSED_DIR = PROCESSED_DATA_DIR
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
SEGMENTS_DIR = BASE_DIR / "data" / "segments"
BUSINESS_DIR = BASE_DIR / "data" / "business"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images" / "business"

for directory in [BUSINESS_DIR, REPORTS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_base_customer_data() -> pd.DataFrame:
    path = PROCESSED_DIR / "customer_ml_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract_data` first."
        )

    cols = [
        "customerkey",
        "customer_name",
        "countryfull",
        "age",
        "total_revenue",
        "monetary_value",
        "frequency",
        "recency_days",
        "is_churned_180d",
        "customer_status_180d",
        "historical_ltv_segment",
        "future_6m_revenue",
    ]

    df = pd.read_parquet(path)

    available_cols = [col for col in cols if col in df.columns]

    return df[available_cols].copy()


def load_churn_predictions() -> pd.DataFrame:
    """
    Current churn predictions were saved only for the test set.
    For Phase 8, we use them as a test-set business scoring output.
    """
    path = PREDICTIONS_DIR / "customer_churn_scores_test.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train_churn_models` first."
        )

    churn = pd.read_csv(path)
    churn = churn.reset_index().rename(columns={"index": "prediction_row_id"})

    return churn


def load_two_stage_ltv_predictions() -> pd.DataFrame:
    path = PREDICTIONS_DIR / "customer_two_stage_ltv_predictions_test.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.train_two_stage_ltv_model` first."
        )

    ltv = pd.read_csv(path)
    ltv = ltv.reset_index().rename(columns={"index": "prediction_row_id"})

    return ltv


def load_segments() -> pd.DataFrame:
    path = SEGMENTS_DIR / "customer_segments.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.customer_segmentation` first."
        )

    segments = pd.read_csv(path)

    cols = [
        "customerkey",
        "cluster",
        "business_segment",
    ]

    return segments[cols].copy()


def create_test_customer_mapping(base_df: pd.DataFrame, churn_df: pd.DataFrame) -> pd.DataFrame:
    """
    Since Phase 5 saved test predictions without customerkey,
    we map predictions to the first N customers as a temporary portfolio demo.

    Better version for Phase 9:
    update Phase 5/6B scripts to save customerkey alongside predictions.
    """
    n = len(churn_df)

    mapping = base_df.head(n).reset_index(drop=True).copy()
    mapping = mapping.reset_index().rename(columns={"index": "prediction_row_id"})

    return mapping


def assign_churn_risk_segment(probability: float) -> str:
    if probability >= 0.80:
        return "High Churn Risk"
    if probability >= 0.50:
        return "Medium Churn Risk"
    return "Low Churn Risk"


def assign_ltv_segment(predicted_ltv: float) -> str:
    if predicted_ltv >= 5000:
        return "High Predicted LTV"
    if predicted_ltv >= 1000:
        return "Medium Predicted LTV"
    return "Low Predicted LTV"


def recommend_action(row: pd.Series) -> str:
    churn_prob = row["churn_probability"]
    predicted_ltv = row["predicted_ltv"]
    segment = row.get("business_segment", "")

    if churn_prob >= 0.80 and predicted_ltv >= 5000:
        return "Immediate VIP Retention Outreach"

    if churn_prob >= 0.70 and predicted_ltv >= 2000:
        return "Personalized Win-Back Offer"

    if churn_prob >= 0.50 and predicted_ltv >= 1000:
        return "Targeted Discount Campaign"

    if churn_prob >= 0.50:
        return "Email Re-Engagement Campaign"

    if predicted_ltv >= 5000:
        return "VIP Loyalty / Upsell Program"

    if "Dormant" in str(segment):
        return "Low-Cost Re-Activation Email"

    return "Standard Lifecycle Marketing"


def assign_priority_tier(priority_score: float) -> str:
    if priority_score >= 5000:
        return "Critical Priority"
    if priority_score >= 2000:
        return "High Priority"
    if priority_score >= 500:
        return "Medium Priority"
    return "Low Priority"


def build_customer_360() -> pd.DataFrame:
    base_df = load_base_customer_data()
    churn_df = load_churn_predictions()
    ltv_df = load_two_stage_ltv_predictions()
    segments_df = load_segments()

    mapping = create_test_customer_mapping(base_df, churn_df)

    customer_360 = (
        mapping
        .merge(churn_df, on="prediction_row_id", how="left")
        .merge(ltv_df, on="prediction_row_id", how="left")
        .merge(segments_df, on="customerkey", how="left")
    )

    customer_360 = customer_360.rename(
        columns={
            "final_two_stage_ltv_prediction": "predicted_ltv",
            "actual_future_6m_revenue": "actual_future_6m_revenue_ltv",
            "actual_churn": "actual_churn_label",
            "predicted_churn": "predicted_churn_label",
        }
    )

    customer_360["predicted_ltv"] = customer_360["predicted_ltv"].fillna(0)
    customer_360["churn_probability"] = customer_360["churn_probability"].fillna(0)

    customer_360["revenue_at_risk"] = (
        customer_360["churn_probability"] * customer_360["predicted_ltv"]
    )

    customer_360["priority_score"] = customer_360["revenue_at_risk"]

    customer_360["churn_risk_segment"] = customer_360["churn_probability"].apply(
        assign_churn_risk_segment
    )

    customer_360["predicted_ltv_segment"] = customer_360["predicted_ltv"].apply(
        assign_ltv_segment
    )

    customer_360["priority_tier"] = customer_360["priority_score"].apply(
        assign_priority_tier
    )

    customer_360["recommended_action"] = customer_360.apply(
        recommend_action,
        axis=1,
    )

    final_cols = [
        "customerkey",
        "customer_name",
        "countryfull",
        "age",
        "business_segment",
        "historical_ltv_segment",
        "frequency",
        "recency_days",
        "total_revenue",
        "churn_probability",
        "churn_risk_segment",
        "predicted_ltv",
        "predicted_ltv_segment",
        "revenue_at_risk",
        "priority_score",
        "priority_tier",
        "recommended_action",
    ]

    available_cols = [col for col in final_cols if col in customer_360.columns]

    return customer_360[available_cols].copy()


def plot_priority_distribution(customer_360: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(customer_360["priority_score"], bins=50)
    plt.xlabel("Priority Score / Revenue at Risk")
    plt.ylabel("Number of Customers")
    plt.title("Customer Priority Score Distribution")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "priority_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_action_distribution(customer_360: pd.DataFrame) -> None:
    action_counts = customer_360["recommended_action"].value_counts().sort_values()

    plt.figure(figsize=(10, 6))
    plt.barh(action_counts.index, action_counts.values)
    plt.xlabel("Number of Customers")
    plt.ylabel("Recommended Action")
    plt.title("Recommended Action Distribution")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "recommended_action_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_segment_revenue_at_risk(customer_360: pd.DataFrame) -> None:
    segment = (
        customer_360.groupby("business_segment")
        .agg(
            customers=("customerkey", "count"),
            revenue_at_risk=("revenue_at_risk", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_predicted_ltv=("predicted_ltv", "mean"),
        )
        .sort_values("revenue_at_risk", ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(segment.index, segment["revenue_at_risk"])
    plt.xlabel("Total Revenue at Risk")
    plt.ylabel("Business Segment")
    plt.title("Revenue at Risk by Customer Segment")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "segment_revenue_at_risk.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_business_reports(customer_360: pd.DataFrame) -> None:
    customer_360.to_csv(
        BUSINESS_DIR / "customer_360_scores.csv",
        index=False,
    )

    top_customers = customer_360.sort_values(
        "priority_score",
        ascending=False,
    ).head(50)

    top_customers.to_csv(
        REPORTS_DIR / "top_50_customer_priorities.csv",
        index=False,
    )

    action_summary = (
        customer_360.groupby("recommended_action")
        .agg(
            customers=("customerkey", "count"),
            total_revenue_at_risk=("revenue_at_risk", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_predicted_ltv=("predicted_ltv", "mean"),
        )
        .sort_values("total_revenue_at_risk", ascending=False)
    )

    action_summary.to_csv(
        REPORTS_DIR / "business_action_recommendations.csv",
    )

    metadata = {
        "phase": "Phase 8 - Customer 360 Decision Engine",
        "customers_scored": int(len(customer_360)),
        "total_revenue_at_risk": float(customer_360["revenue_at_risk"].sum()),
        "avg_churn_probability": float(customer_360["churn_probability"].mean()),
        "avg_predicted_ltv": float(customer_360["predicted_ltv"].mean()),
        "actions_generated": int(customer_360["recommended_action"].nunique()),
    }

    with open(REPORTS_DIR / "customer_360_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(REPORTS_DIR / "customer_360_decision_report.md", "w", encoding="utf-8") as f:
        f.write("# Customer 360 Decision Engine Report\n\n")

        f.write("## Overview\n\n")
        f.write(
            "This report combines churn prediction, two-stage LTV prediction, and customer segmentation "
            "to generate customer-level business recommendations.\n\n"
        )

        f.write("## Summary\n\n")
        f.write(f"- Customers scored: {len(customer_360):,}\n")
        f.write(f"- Total revenue at risk: ${customer_360['revenue_at_risk'].sum():,.2f}\n")
        f.write(f"- Average churn probability: {customer_360['churn_probability'].mean():.2%}\n")
        f.write(f"- Average predicted LTV: ${customer_360['predicted_ltv'].mean():,.2f}\n")
        f.write(f"- Recommended action types: {customer_360['recommended_action'].nunique()}\n\n")

        f.write("## Recommended Action Summary\n\n")
        f.write(action_summary.round(4).to_markdown())
        f.write("\n\n")

        f.write("## Top 10 Highest Priority Customers\n\n")
        display_cols = [
            "customerkey",
            "customer_name",
            "business_segment",
            "churn_probability",
            "predicted_ltv",
            "revenue_at_risk",
            "recommended_action",
        ]
        existing_display_cols = [
            col for col in display_cols if col in top_customers.columns
        ]
        f.write(top_customers[existing_display_cols].head(10).round(4).to_markdown(index=False))
        f.write("\n")


def main() -> None:
    print("=" * 80)
    print("PHASE 8: CUSTOMER 360 DECISION ENGINE")
    print("=" * 80)

    print("\nBuilding Customer 360 scoring table...")
    customer_360 = build_customer_360()

    print(f"Customers scored: {len(customer_360):,}")
    print(f"Total revenue at risk: ${customer_360['revenue_at_risk'].sum():,.2f}")
    print(f"Average churn probability: {customer_360['churn_probability'].mean():.2%}")
    print(f"Average predicted LTV: ${customer_360['predicted_ltv'].mean():,.2f}")

    print("\nRecommended action distribution:")
    print(customer_360["recommended_action"].value_counts())

    print("\nGenerating business plots...")
    plot_priority_distribution(customer_360)
    plot_action_distribution(customer_360)
    plot_segment_revenue_at_risk(customer_360)

    print("\nSaving reports...")
    save_business_reports(customer_360)

    print("\nPhase 8 complete.")
    print(f"Saved Customer 360 output to: {BUSINESS_DIR}")
    print(f"Saved reports to: {REPORTS_DIR}")
    print(f"Saved images to: {IMAGES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()