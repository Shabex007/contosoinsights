from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import PROCESSED_DATA_DIR, BASE_DIR


EDA_IMAGE_DIR = BASE_DIR / "images" / "eda"
EDA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "customer_ml_features.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.extract_data"
        )
    return pd.read_parquet(path)


def save_plot(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(EDA_IMAGE_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_churn_distribution(df: pd.DataFrame) -> None:
    counts = df["customer_status_180d"].value_counts()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Customer Status Distribution")
    plt.xlabel("Customer Status")
    plt.ylabel("Number of Customers")
    save_plot("01_churn_distribution.png")


def plot_ltv_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(df["total_revenue"]), bins=50)
    plt.title("Customer Revenue Distribution Log Scale")
    plt.xlabel("log(1 + Total Revenue)")
    plt.ylabel("Number of Customers")
    save_plot("02_ltv_distribution_log.png")


def plot_revenue_by_segment(df: pd.DataFrame) -> None:
    segment_revenue = (
        df.groupby("historical_ltv_segment")["total_revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    segment_revenue.plot(kind="bar")
    plt.title("Total Revenue by LTV Segment")
    plt.xlabel("LTV Segment")
    plt.ylabel("Total Revenue")
    save_plot("03_revenue_by_segment.png")


def plot_churn_by_segment(df: pd.DataFrame) -> None:
    churn_segment = (
        df.groupby("historical_ltv_segment")["is_churned_180d"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(8, 5))
    churn_segment.plot(kind="bar")
    plt.title("Churn Rate by LTV Segment")
    plt.xlabel("LTV Segment")
    plt.ylabel("Churn Rate")
    save_plot("04_churn_by_segment.png")


def plot_cohort_revenue(df: pd.DataFrame) -> None:
    cohort = (
        df.groupby("cohort_year")
        .agg(
            customers=("customerkey", "count"),
            revenue=("total_revenue", "sum"),
            avg_revenue=("total_revenue", "mean"),
            churn_rate=("is_churned_180d", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(9, 5))
    plt.plot(cohort["cohort_year"], cohort["avg_revenue"], marker="o")
    plt.title("Average Revenue per Customer by Cohort Year")
    plt.xlabel("Cohort Year")
    plt.ylabel("Average Revenue per Customer")
    save_plot("05_avg_revenue_by_cohort.png")


def plot_country_revenue(df: pd.DataFrame) -> None:
    country = (
        df.groupby("countryfull")
        .agg(
            customers=("customerkey", "count"),
            revenue=("total_revenue", "sum"),
            churn_rate=("is_churned_180d", "mean"),
        )
        .sort_values("revenue", ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    country["revenue"].sort_values().plot(kind="barh")
    plt.title("Top 10 Countries by Revenue")
    plt.xlabel("Total Revenue")
    plt.ylabel("Country")
    save_plot("06_top_countries_by_revenue.png")


def plot_recency_frequency_scatter(df: pd.DataFrame) -> None:
    sample = df.sample(min(5000, len(df)), random_state=42)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        sample["recency_days"],
        sample["frequency"],
        alpha=0.4,
    )
    plt.title("Recency vs Frequency")
    plt.xlabel("Recency Days")
    plt.ylabel("Purchase Frequency")
    save_plot("07_recency_frequency_scatter.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    selected_cols = [
        "age",
        "recency_days",
        "customer_tenure_days",
        "frequency",
        "unique_products_bought",
        "unique_categories_bought",
        "monetary_value",
        "total_revenue",
        "total_gross_profit",
        "gross_margin_pct",
        "avg_order_value",
        "avg_items_per_order",
        "purchase_velocity_orders_per_year",
        "is_churned_180d",
        "future_6m_revenue",
    ]

    available_cols = [col for col in selected_cols if col in df.columns]
    corr = df[available_cols].corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    save_plot("08_correlation_heatmap.png")


def generate_summary_report(df: pd.DataFrame) -> None:
    report_path = BASE_DIR / "reports" / "phase3_eda_summary.md"

    total_customers = len(df)
    churn_rate = df["is_churned_180d"].mean()
    total_revenue = df["total_revenue"].sum()
    avg_revenue = df["total_revenue"].mean()
    median_frequency = df["frequency"].median()

    segment_summary = (
        df.groupby("historical_ltv_segment")
        .agg(
            customers=("customerkey", "count"),
            revenue=("total_revenue", "sum"),
            churn_rate=("is_churned_180d", "mean"),
            avg_revenue=("total_revenue", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Phase 3 EDA Summary\n\n")
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total customers: {total_customers:,}\n")
        f.write(f"- Total revenue: ${total_revenue:,.2f}\n")
        f.write(f"- Average revenue per customer: ${avg_revenue:,.2f}\n")
        f.write(f"- Churn rate 180d: {churn_rate:.2%}\n")
        f.write(f"- Median purchase frequency: {median_frequency:.0f}\n\n")

        f.write("## Segment Summary\n\n")
        f.write(segment_summary.round(2).to_markdown())
        f.write("\n\n")

        f.write("## Key Initial Insights\n\n")
        f.write("1. The customer base has a high 180-day churn rate.\n")
        f.write("2. Revenue is highly skewed, meaning a smaller group of customers contributes disproportionately to sales.\n")
        f.write("3. Many customers purchase only once, making repeat purchase behavior a key modeling signal.\n")
        f.write("4. LTV segment, recency, frequency, and monetary value should be strong predictors for churn and customer value.\n")

    print(f"Saved report: {report_path}")


def main() -> None:
    df = load_data()

    print("Dataset loaded")
    print(f"Shape: {df.shape}")

    plot_churn_distribution(df)
    plot_ltv_distribution(df)
    plot_revenue_by_segment(df)
    plot_churn_by_segment(df)
    plot_cohort_revenue(df)
    plot_country_revenue(df)
    plot_recency_frequency_scatter(df)
    plot_correlation_heatmap(df)
    generate_summary_report(df)

    print(f"EDA charts saved to: {EDA_IMAGE_DIR}")


if __name__ == "__main__":
    main()