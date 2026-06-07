from __future__ import annotations

import json
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.config import BASE_DIR


BUSINESS_DIR = BASE_DIR / "data" / "business"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images" / "business"

for directory in [BUSINESS_DIR, REPORTS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


CAMPAIGN_ASSUMPTIONS = {
    "Immediate VIP Retention Outreach": {
        "cost_per_customer": 100,
        "retention_success_rate": 0.35,
    },
    "Personalized Win-Back Offer": {
        "cost_per_customer": 50,
        "retention_success_rate": 0.25,
    },
    "Targeted Discount Campaign": {
        "cost_per_customer": 20,
        "retention_success_rate": 0.18,
    },
    "Email Re-Engagement Campaign": {
        "cost_per_customer": 2,
        "retention_success_rate": 0.05,
    },
    "Low-Cost Re-Activation Email": {
        "cost_per_customer": 1,
        "retention_success_rate": 0.03,
    },
    "VIP Loyalty / Upsell Program": {
        "cost_per_customer": 75,
        "retention_success_rate": 0.20,
    },
    "Standard Lifecycle Marketing": {
        "cost_per_customer": 1,
        "retention_success_rate": 0.02,
    },
}


def load_customer_360() -> pd.DataFrame:
    path = BUSINESS_DIR / "customer_360_scores.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.generate_customer_actions` first."
        )

    return pd.read_csv(path)


def add_campaign_assumptions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["campaign_cost_per_customer"] = df["recommended_action"].map(
        lambda action: CAMPAIGN_ASSUMPTIONS.get(
            action,
            {"cost_per_customer": 1},
        )["cost_per_customer"]
    )

    df["retention_success_rate"] = df["recommended_action"].map(
        lambda action: CAMPAIGN_ASSUMPTIONS.get(
            action,
            {"retention_success_rate": 0.02},
        )["retention_success_rate"]
    )

    return df


def calculate_customer_roi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["expected_saved_revenue"] = (
        df["revenue_at_risk"] * df["retention_success_rate"]
    )

    df["campaign_cost"] = df["campaign_cost_per_customer"]

    df["net_roi"] = df["expected_saved_revenue"] - df["campaign_cost"]

    df["roi_multiple"] = df.apply(
        lambda row: (
            row["expected_saved_revenue"] / row["campaign_cost"]
            if row["campaign_cost"] > 0
            else 0
        ),
        axis=1,
    )

    return df


def summarize_roi_by_campaign(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("recommended_action")
        .agg(
            targeted_customers=("customerkey", "count"),
            total_revenue_at_risk=("revenue_at_risk", "sum"),
            avg_churn_probability=("churn_probability", "mean"),
            avg_predicted_ltv=("predicted_ltv", "mean"),
            cost_per_customer=("campaign_cost_per_customer", "mean"),
            retention_success_rate=("retention_success_rate", "mean"),
            total_campaign_cost=("campaign_cost", "sum"),
            expected_saved_revenue=("expected_saved_revenue", "sum"),
            net_roi=("net_roi", "sum"),
        )
        .reset_index()
    )

    summary["roi_multiple"] = summary.apply(
        lambda row: (
            row["expected_saved_revenue"] / row["total_campaign_cost"]
            if row["total_campaign_cost"] > 0
            else 0
        ),
        axis=1,
    )

    summary = summary.sort_values("net_roi", ascending=False)

    return summary


def plot_expected_saved_revenue(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("expected_saved_revenue", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["recommended_action"], plot_df["expected_saved_revenue"])
    plt.xlabel("Expected Saved Revenue")
    plt.ylabel("Campaign")
    plt.title("Expected Saved Revenue by Campaign")
    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "expected_saved_revenue_by_campaign.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_net_roi(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("net_roi", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["recommended_action"], plot_df["net_roi"])
    plt.xlabel("Net ROI")
    plt.ylabel("Campaign")
    plt.title("Net ROI by Campaign")
    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "net_roi_by_campaign.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def plot_roi_multiple(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("roi_multiple", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["recommended_action"], plot_df["roi_multiple"])
    plt.xlabel("ROI Multiple")
    plt.ylabel("Campaign")
    plt.title("ROI Multiple by Campaign")
    plt.tight_layout()
    plt.savefig(
        IMAGES_DIR / "roi_multiple_by_campaign.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()


def save_outputs(customer_roi: pd.DataFrame, campaign_summary: pd.DataFrame) -> None:
    customer_roi.to_csv(
        BUSINESS_DIR / "customer_roi_scores.csv",
        index=False,
    )

    campaign_summary.to_csv(
        BUSINESS_DIR / "campaign_roi_by_action.csv",
        index=False,
    )

    metadata = {
        "phase": "Phase 9 - ROI Calculator",
        "customers_scored": int(len(customer_roi)),
        "campaigns": int(campaign_summary.shape[0]),
        "total_revenue_at_risk": float(customer_roi["revenue_at_risk"].sum()),
        "total_expected_saved_revenue": float(customer_roi["expected_saved_revenue"].sum()),
        "total_campaign_cost": float(customer_roi["campaign_cost"].sum()),
        "total_net_roi": float(customer_roi["net_roi"].sum()),
        "overall_roi_multiple": float(
            customer_roi["expected_saved_revenue"].sum()
            / customer_roi["campaign_cost"].sum()
        )
        if customer_roi["campaign_cost"].sum() > 0
        else 0,
        "campaign_assumptions": CAMPAIGN_ASSUMPTIONS,
    }

    with open(REPORTS_DIR / "roi_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(REPORTS_DIR / "roi_summary_report.md", "w", encoding="utf-8") as f:
        f.write("# Campaign ROI Summary Report\n\n")

        f.write("## Overview\n\n")
        f.write(
            "This report estimates the financial impact of retention and re-engagement campaigns "
            "using churn probability, predicted LTV, revenue at risk, campaign costs, and assumed retention success rates.\n\n"
        )

        f.write("## ROI Formula\n\n")
        f.write("```text\n")
        f.write("revenue_at_risk = churn_probability × predicted_ltv\n")
        f.write("expected_saved_revenue = revenue_at_risk × retention_success_rate\n")
        f.write("campaign_cost = targeted_customers × cost_per_customer\n")
        f.write("net_roi = expected_saved_revenue - campaign_cost\n")
        f.write("roi_multiple = expected_saved_revenue / campaign_cost\n")
        f.write("```\n\n")

        f.write("## Overall Financial Impact\n\n")
        f.write(f"- Customers scored: {len(customer_roi):,}\n")
        f.write(f"- Total revenue at risk: ${metadata['total_revenue_at_risk']:,.2f}\n")
        f.write(f"- Total expected saved revenue: ${metadata['total_expected_saved_revenue']:,.2f}\n")
        f.write(f"- Total campaign cost: ${metadata['total_campaign_cost']:,.2f}\n")
        f.write(f"- Total net ROI: ${metadata['total_net_roi']:,.2f}\n")
        f.write(f"- Overall ROI multiple: {metadata['overall_roi_multiple']:.2f}x\n\n")

        f.write("## Campaign ROI Summary\n\n")
        f.write(campaign_summary.round(4).to_markdown(index=False))
        f.write("\n")


def main() -> None:
    print("=" * 80)
    print("PHASE 9: ROI CALCULATOR")
    print("=" * 80)

    print("\nLoading Customer 360 scores...")
    customer_360 = load_customer_360()

    print(f"Customers loaded: {len(customer_360):,}")

    print("\nAdding campaign assumptions...")
    customer_roi = add_campaign_assumptions(customer_360)

    print("\nCalculating customer-level ROI...")
    customer_roi = calculate_customer_roi(customer_roi)

    print("\nSummarizing campaign ROI...")
    campaign_summary = summarize_roi_by_campaign(customer_roi)

    print("\nCampaign ROI Summary:")
    print(campaign_summary.round(2).to_string(index=False))

    print("\nGenerating ROI plots...")
    plot_expected_saved_revenue(campaign_summary)
    plot_net_roi(campaign_summary)
    plot_roi_multiple(campaign_summary)

    print("\nSaving ROI outputs...")
    save_outputs(customer_roi, campaign_summary)

    print("\nPhase 9 complete.")
    print(f"Saved customer ROI scores to: {BUSINESS_DIR / 'customer_roi_scores.csv'}")
    print(f"Saved campaign ROI summary to: {BUSINESS_DIR / 'campaign_roi_by_action.csv'}")
    print(f"Saved ROI report to: {REPORTS_DIR / 'roi_summary_report.md'}")
    print("=" * 80)


if __name__ == "__main__":
    main()