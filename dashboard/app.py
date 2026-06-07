from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent.parent

BUSINESS_DIR = BASE_DIR / "data" / "business"
SEGMENTS_DIR = BASE_DIR / "data" / "segments"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_MODELING_DIR = BASE_DIR / "images" / "modeling"
IMAGES_SEGMENTATION_DIR = BASE_DIR / "images" / "segmentation"
IMAGES_BUSINESS_DIR = BASE_DIR / "images" / "business"


st.set_page_config(
    page_title="Contoso Customer Intelligence",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_customer_360() -> pd.DataFrame:
    path = BUSINESS_DIR / "customer_360_scores.csv"
    return pd.read_csv(path)


@st.cache_data
def load_roi_summary() -> pd.DataFrame:
    path = BUSINESS_DIR / "campaign_roi_by_action.csv"
    return pd.read_csv(path)


@st.cache_data
def load_segment_profiles() -> pd.DataFrame:
    path = SEGMENTS_DIR / "customer_segment_profiles.csv"
    return pd.read_csv(path)


@st.cache_data
def load_churn_model_results() -> pd.DataFrame:
    path = REPORTS_DIR / "churn_model_comparison.csv"
    return pd.read_csv(path)


@st.cache_data
def load_ltv_model_results() -> pd.DataFrame:
    path = REPORTS_DIR / "ltv_model_comparison.csv"
    return pd.read_csv(path)


def format_currency(value: float) -> str:
    return f"${value:,.2f}"


def show_image_if_exists(path: Path, caption: str) -> None:
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"Image not found: {path.name}")


def executive_overview(customer_360: pd.DataFrame, roi_summary: pd.DataFrame) -> None:
    st.title("📊 Contoso Customer Intelligence Dashboard")
    st.subheader("Executive Overview")

    total_customers = len(customer_360)
    total_revenue_at_risk = customer_360["revenue_at_risk"].sum()
    avg_churn_probability = customer_360["churn_probability"].mean()
    avg_predicted_ltv = customer_360["predicted_ltv"].mean()
    total_net_roi = roi_summary["net_roi"].sum()

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Customers Scored", f"{total_customers:,}")
    col2.metric("Revenue at Risk", format_currency(total_revenue_at_risk))
    col3.metric("Avg Churn Probability", f"{avg_churn_probability:.2%}")
    col4.metric("Avg Predicted LTV", format_currency(avg_predicted_ltv))
    col5.metric("Estimated Net ROI", format_currency(total_net_roi))

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Priority Tier Distribution")
        priority_counts = (
            customer_360["priority_tier"]
            .value_counts()
            .reset_index()
        )
        priority_counts.columns = ["priority_tier", "customers"]

        fig = px.bar(
            priority_counts,
            x="priority_tier",
            y="customers",
            title="Customers by Priority Tier",
            text="customers",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Recommended Actions")
        action_counts = (
            customer_360["recommended_action"]
            .value_counts()
            .reset_index()
        )
        action_counts.columns = ["recommended_action", "customers"]

        fig = px.bar(
            action_counts,
            x="customers",
            y="recommended_action",
            orientation="h",
            title="Recommended Action Distribution",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Priority Customers")
    top_customers = customer_360.sort_values(
        "priority_score",
        ascending=False,
    ).head(10)

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

    st.dataframe(top_customers[display_cols], use_container_width=True)


def segmentation_page(customer_360: pd.DataFrame, segment_profiles: pd.DataFrame) -> None:
    st.title("🧩 Customer Segmentation")

    st.write(
        "Customers are grouped using K-Means clustering based on RFM and behavioral features."
    )

    st.subheader("Segment Profiles")
    st.dataframe(segment_profiles, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(
            segment_profiles,
            x="business_segment",
            y="customers",
            title="Customers by Segment",
            text="customers",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.bar(
            segment_profiles,
            x="business_segment",
            y="total_revenue",
            title="Total Revenue by Segment",
            text="total_revenue",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segmentation Visuals")

    col3, col4 = st.columns(2)

    with col3:
        show_image_if_exists(
            IMAGES_SEGMENTATION_DIR / "segmentation_pca_clusters.png",
            "PCA Cluster Visualization",
        )

    with col4:
        show_image_if_exists(
            IMAGES_SEGMENTATION_DIR / "segmentation_silhouette_scores.png",
            "Silhouette Scores",
        )


def churn_page(customer_360: pd.DataFrame) -> None:
    st.title("⚠️ Churn Prediction")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Churn Probability",
        f"{customer_360['churn_probability'].mean():.2%}",
    )
    col2.metric(
        "High Churn Risk Customers",
        f"{(customer_360['churn_risk_segment'] == 'High Churn Risk').sum():,}",
    )
    col3.metric(
        "Revenue at Risk",
        format_currency(customer_360["revenue_at_risk"].sum()),
    )

    st.subheader("Churn Risk Distribution")

    fig = px.histogram(
        customer_360,
        x="churn_probability",
        nbins=50,
        title="Churn Probability Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn Risk by Segment")

    segment_churn = (
        customer_360.groupby("business_segment")
        .agg(
            customers=("customerkey", "count"),
            avg_churn_probability=("churn_probability", "mean"),
            revenue_at_risk=("revenue_at_risk", "sum"),
        )
        .reset_index()
    )

    fig = px.bar(
        segment_churn,
        x="business_segment",
        y="avg_churn_probability",
        title="Average Churn Probability by Segment",
        text="avg_churn_probability",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model Evaluation")

    try:
        churn_results = load_churn_model_results()
        st.dataframe(churn_results, use_container_width=True)
    except Exception:
        st.warning("Churn model comparison report not found.")

    col1, col2 = st.columns(2)

    with col1:
        show_image_if_exists(
            IMAGES_MODELING_DIR / "churn_confusion_matrix.png",
            "Churn Confusion Matrix",
        )

    with col2:
        show_image_if_exists(
            IMAGES_MODELING_DIR / "churn_roc_curve.png",
            "Churn ROC Curve",
        )


def ltv_page(customer_360: pd.DataFrame) -> None:
    st.title("💰 LTV Forecasting")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Predicted LTV",
        format_currency(customer_360["predicted_ltv"].mean()),
    )
    col2.metric(
        "Total Predicted LTV",
        format_currency(customer_360["predicted_ltv"].sum()),
    )
    col3.metric(
        "High Predicted LTV Customers",
        f"{(customer_360['predicted_ltv_segment'] == 'High Predicted LTV').sum():,}",
    )

    st.subheader("Predicted LTV Distribution")

    fig = px.histogram(
        customer_360,
        x="predicted_ltv",
        nbins=50,
        title="Predicted LTV Distribution",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Predicted Future Value Customers")

    top_ltv = customer_360.sort_values(
        "predicted_ltv",
        ascending=False,
    ).head(20)

    st.dataframe(
        top_ltv[
            [
                "customerkey",
                "customer_name",
                "business_segment",
                "churn_probability",
                "predicted_ltv",
                "recommended_action",
            ]
        ],
        use_container_width=True,
    )

    st.subheader("LTV Model Evaluation")

    try:
        ltv_results = load_ltv_model_results()
        st.dataframe(ltv_results, use_container_width=True)
    except Exception:
        st.warning("LTV model comparison report not found.")

    col1, col2 = st.columns(2)

    with col1:
        show_image_if_exists(
            IMAGES_MODELING_DIR / "ltv_actual_vs_predicted.png",
            "LTV Actual vs Predicted",
        )

    with col2:
        show_image_if_exists(
            IMAGES_MODELING_DIR / "ltv_residual_plot.png",
            "LTV Residual Plot",
        )


def roi_page(customer_360: pd.DataFrame, roi_summary: pd.DataFrame) -> None:
    st.title("📈 Campaign ROI Calculator")

    st.write(
        "Adjust campaign assumptions to simulate expected saved revenue, cost, and net ROI."
    )

    default_success_rate = st.slider(
        "Global Retention Success Rate Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    default_cost_multiplier = st.slider(
        "Global Campaign Cost Multiplier",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
    )

    simulated = roi_summary.copy()

    simulated["simulated_expected_saved_revenue"] = (
        simulated["expected_saved_revenue"] * default_success_rate
    )

    simulated["simulated_campaign_cost"] = (
        simulated["total_campaign_cost"] * default_cost_multiplier
    )

    simulated["simulated_net_roi"] = (
        simulated["simulated_expected_saved_revenue"]
        - simulated["simulated_campaign_cost"]
    )

    total_saved = simulated["simulated_expected_saved_revenue"].sum()
    total_cost = simulated["simulated_campaign_cost"].sum()
    total_net_roi = simulated["simulated_net_roi"].sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Saved Revenue", format_currency(total_saved))
    col2.metric("Campaign Cost", format_currency(total_cost))
    col3.metric("Net ROI", format_currency(total_net_roi))

    st.subheader("Campaign ROI Summary")

    st.dataframe(simulated, use_container_width=True)

    fig = px.bar(
        simulated.sort_values("simulated_net_roi", ascending=True),
        x="simulated_net_roi",
        y="recommended_action",
        orientation="h",
        title="Simulated Net ROI by Campaign",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top ROI Customers")

    top_roi = customer_360.sort_values(
        "priority_score",
        ascending=False,
    ).head(25)

    st.dataframe(
        top_roi[
            [
                "customerkey",
                "customer_name",
                "business_segment",
                "churn_probability",
                "predicted_ltv",
                "revenue_at_risk",
                "recommended_action",
            ]
        ],
        use_container_width=True,
    )


def customer_lookup_page(customer_360: pd.DataFrame) -> None:
    st.title("🔍 Customer Lookup")

    customer_ids = customer_360["customerkey"].sort_values().unique()

    selected_customer = st.selectbox(
        "Select Customer ID",
        customer_ids,
    )

    customer = customer_360[
        customer_360["customerkey"] == selected_customer
    ].iloc[0]

    st.subheader(f"Customer: {customer['customer_name']}")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Churn Probability", f"{customer['churn_probability']:.2%}")
    col2.metric("Predicted LTV", format_currency(customer["predicted_ltv"]))
    col3.metric("Revenue at Risk", format_currency(customer["revenue_at_risk"]))
    col4.metric("Priority Tier", customer["priority_tier"])

    st.divider()

    st.write("### Customer Details")

    details = {
        "Customer Key": customer["customerkey"],
        "Name": customer["customer_name"],
        "Country": customer["countryfull"],
        "Age": customer["age"],
        "Business Segment": customer["business_segment"],
        "Historical LTV Segment": customer["historical_ltv_segment"],
        "Frequency": customer["frequency"],
        "Recency Days": customer["recency_days"],
        "Total Revenue": format_currency(customer["total_revenue"]),
        "Churn Risk Segment": customer["churn_risk_segment"],
        "Predicted LTV Segment": customer["predicted_ltv_segment"],
        "Recommended Action": customer["recommended_action"],
    }

    details_df = pd.DataFrame(
        list(details.items()),
        columns=["Metric", "Value"],
    )

    st.dataframe(details_df, use_container_width=True)


def main() -> None:
    try:
        customer_360 = load_customer_360()
        roi_summary = load_roi_summary()
        segment_profiles = load_segment_profiles()
    except Exception as e:
        st.error("Required dashboard data not found.")
        st.code(str(e))
        st.stop()

    st.sidebar.title("Contoso Intelligence")
    st.sidebar.write("Customer Analytics + ML Dashboard")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive Overview",
            "Customer Segmentation",
            "Churn Prediction",
            "LTV Forecasting",
            "Campaign ROI",
            "Customer Lookup",
        ],
    )

    if page == "Executive Overview":
        executive_overview(customer_360, roi_summary)

    elif page == "Customer Segmentation":
        segmentation_page(customer_360, segment_profiles)

    elif page == "Churn Prediction":
        churn_page(customer_360)

    elif page == "LTV Forecasting":
        ltv_page(customer_360)

    elif page == "Campaign ROI":
        roi_page(customer_360, roi_summary)

    elif page == "Customer Lookup":
        customer_lookup_page(customer_360)


if __name__ == "__main__":
    main()