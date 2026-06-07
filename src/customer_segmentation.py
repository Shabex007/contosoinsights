from __future__ import annotations

import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.config import BASE_DIR, PROCESSED_DATA_DIR


SEGMENTS_DIR = BASE_DIR / "data" / "segments"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
IMAGES_DIR = BASE_DIR / "images" / "segmentation"

for directory in [SEGMENTS_DIR, MODELS_DIR, REPORTS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


SEGMENT_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_value",
    "total_revenue",
    "avg_order_value",
    "customer_tenure_days",
    "unique_products_bought",
    "unique_categories_bought",
    "purchase_velocity_orders_per_year",
    "gross_margin_pct",
]


LOG_FEATURES = [
    "monetary_value",
    "total_revenue",
    "avg_order_value",
]


def load_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "customer_ml_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract_data` first."
        )

    return pd.read_parquet(path)


def prepare_segmentation_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    available_features = [col for col in SEGMENT_FEATURES if col in df.columns]

    if len(available_features) < 4:
        raise ValueError("Not enough segmentation features available.")

    X = df[available_features].copy()

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    for col in LOG_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col].clip(lower=0))

    return X, available_features


def scale_features(X: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def evaluate_k_values(X_scaled: np.ndarray, k_min: int = 2, k_max: int = 8) -> pd.DataFrame:
    results = []

    for k in range(k_min, k_max + 1):
        print(f"Evaluating k={k}")

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = model.fit_predict(X_scaled)

        inertia = model.inertia_
        silhouette = silhouette_score(X_scaled, labels)

        results.append(
            {
                "k": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
            }
        )

    return pd.DataFrame(results)


def plot_elbow_curve(k_results: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(k_results["k"], k_results["inertia"], marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Curve for Customer Segmentation")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "segmentation_elbow_curve.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_silhouette_scores(k_results: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(k_results["k"], k_results["silhouette_score"], marker="o")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Scores by Number of Clusters")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "segmentation_silhouette_scores.png", dpi=150, bbox_inches="tight")
    plt.close()


def choose_best_k(k_results: pd.DataFrame) -> int:
    best_row = k_results.sort_values("silhouette_score", ascending=False).iloc[0]
    best_k = int(best_row["k"])
    return best_k


def train_final_kmeans(X_scaled: np.ndarray, best_k: int) -> KMeans:
    model = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10,
    )

    model.fit(X_scaled)
    return model


def create_segment_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    features: list[str],
) -> pd.DataFrame:
    segmented_df = df.copy()
    segmented_df["cluster"] = labels

    profile = (
        segmented_df.groupby("cluster")
        .agg(
            customers=("customerkey", "count"),
            avg_recency_days=("recency_days", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_total_revenue=("total_revenue", "mean"),
            total_revenue=("total_revenue", "sum"),
            avg_order_value=("avg_order_value", "mean"),
            avg_tenure_days=("customer_tenure_days", "mean"),
            avg_product_diversity=("unique_products_bought", "mean"),
            avg_category_diversity=("unique_categories_bought", "mean"),
            churn_rate=("is_churned_180d", "mean"),
            avg_future_6m_revenue=("future_6m_revenue", "mean"),
        )
        .reset_index()
    )

    profile["customer_pct"] = profile["customers"] / profile["customers"].sum()
    profile["revenue_pct"] = profile["total_revenue"] / profile["total_revenue"].sum()

    return segmented_df, profile


def assign_business_segment_names(profile: pd.DataFrame) -> dict:
    """
    Assign business-friendly names to clusters based on their profile.

    This function uses relative cluster characteristics instead of generic rank logic.
    """

    profile = profile.copy()
    segment_names = {}

    revenue_median = profile["avg_total_revenue"].median()
    recency_median = profile["avg_recency_days"].median()
    frequency_median = profile["avg_frequency"].median()
    churn_median = profile["churn_rate"].median()

    for _, row in profile.iterrows():
        cluster = int(row["cluster"])

        high_revenue = row["avg_total_revenue"] >= revenue_median
        high_frequency = row["avg_frequency"] >= frequency_median
        high_recency = row["avg_recency_days"] >= recency_median
        high_churn = row["churn_rate"] >= churn_median

        if high_revenue and high_frequency and high_churn:
            name = "At-Risk High-Value Customers"

        elif high_revenue and high_frequency and not high_churn:
            name = "Loyal High-Value Customers"

        elif high_revenue and not high_frequency:
            name = "High-Value Occasional Customers"

        elif not high_revenue and high_recency and high_churn:
            name = "Dormant Low-Value Customers"

        elif not high_revenue and not high_recency:
            name = "Recent Low-to-Mid Value Customers"

        else:
            name = "Low-Value Occasional Customers"

        segment_names[cluster] = name

    return segment_names

def plot_pca_clusters(X_scaled: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        {
            "pca_1": components[:, 0],
            "pca_2": components[:, 1],
            "cluster": labels,
        }
    )

    sample = pca_df.sample(min(10000, len(pca_df)), random_state=42)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        sample["pca_1"],
        sample["pca_2"],
        c=sample["cluster"],
        alpha=0.5,
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title("Customer Segments Visualized with PCA")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "segmentation_pca_clusters.png", dpi=150, bbox_inches="tight")
    plt.close()

    return pca_df


def plot_segment_revenue(profile: pd.DataFrame) -> None:
    plot_df = profile.sort_values("total_revenue", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["business_segment"], plot_df["total_revenue"])
    plt.xlabel("Total Revenue")
    plt.ylabel("Business Segment")
    plt.title("Total Revenue by Customer Segment")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "segment_revenue.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_segment_churn(profile: pd.DataFrame) -> None:
    plot_df = profile.sort_values("churn_rate", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["business_segment"], plot_df["churn_rate"])
    plt.xlabel("Churn Rate")
    plt.ylabel("Business Segment")
    plt.title("Churn Rate by Customer Segment")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "segment_churn_rate.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_reports(
    k_results: pd.DataFrame,
    profile: pd.DataFrame,
    best_k: int,
    feature_names: list[str],
) -> None:
    k_results.to_csv(REPORTS_DIR / "segmentation_k_selection.csv", index=False)
    profile.to_csv(REPORTS_DIR / "customer_segment_profiles.csv", index=False)

    metadata = {
        "phase": "Phase 7 - ML Customer Segmentation",
        "algorithm": "KMeans",
        "best_k": best_k,
        "features_used": feature_names,
        "segment_count": int(best_k),
        "total_customers": int(profile["customers"].sum()),
    }

    with open(REPORTS_DIR / "segmentation_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(REPORTS_DIR / "segmentation_report.md", "w", encoding="utf-8") as f:
        f.write("# Customer Segmentation Report\n\n")
        f.write("## Method\n\n")
        f.write(
            "Customers were segmented using K-Means clustering on RFM and behavioral features. "
            "Features were log-transformed where appropriate and standardized before clustering.\n\n"
        )

        f.write("## K Selection\n\n")
        f.write(k_results.round(4).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Segment Profiles\n\n")
        f.write(profile.round(4).to_markdown(index=False))
        f.write("\n\n")


def main() -> None:
    print("=" * 80)
    print("PHASE 7: ML CUSTOMER SEGMENTATION")
    print("=" * 80)

    print("\nLoading customer dataset...")
    df = load_data()
    print(f"Raw dataset shape: {df.shape}")

    print("\nPreparing segmentation features...")
    X, feature_names = prepare_segmentation_features(df)
    print(f"Segmentation feature matrix shape: {X.shape}")
    print(f"Features used: {feature_names}")

    print("\nScaling features...")
    X_scaled, scaler = scale_features(X)

    print("\nEvaluating k values...")
    k_results = evaluate_k_values(X_scaled, k_min=2, k_max=8)
    print(k_results.round(4).to_string(index=False))

    best_k = choose_best_k(k_results)
    print(f"\nBest k based on silhouette score: {best_k}")

    print("\nTraining final KMeans model...")
    model = train_final_kmeans(X_scaled, best_k)
    labels = model.predict(X_scaled)

    print("\nCreating segment profiles...")
    segmented_df, profile = create_segment_profiles(df, labels, feature_names)

    segment_names = assign_business_segment_names(profile)
    segmented_df["business_segment"] = segmented_df["cluster"].map(segment_names)
    profile["business_segment"] = profile["cluster"].map(segment_names)

    profile = profile.sort_values("total_revenue", ascending=False)

    print("\nSegment profiles:")
    print(profile.round(2).to_string(index=False))

    print("\nSaving outputs...")
    segmented_df.to_csv(SEGMENTS_DIR / "customer_segments.csv", index=False)
    profile.to_csv(SEGMENTS_DIR / "customer_segment_profiles.csv", index=False)

    k_results.to_csv(REPORTS_DIR / "segmentation_k_selection.csv", index=False)
    profile.to_csv(REPORTS_DIR / "customer_segment_profiles.csv", index=False)

    joblib.dump(model, MODELS_DIR / "customer_segmentation_kmeans.pkl")
    joblib.dump(scaler, MODELS_DIR / "customer_segmentation_scaler.pkl")
    joblib.dump(feature_names, MODELS_DIR / "customer_segmentation_features.pkl")

    plot_elbow_curve(k_results)
    plot_silhouette_scores(k_results)
    plot_pca_clusters(X_scaled, labels)
    plot_segment_revenue(profile)
    plot_segment_churn(profile)

    save_reports(
        k_results=k_results,
        profile=profile,
        best_k=best_k,
        feature_names=feature_names,
    )

    print("\nPhase 7 complete.")
    print(f"Saved segmented customers to: {SEGMENTS_DIR}")
    print(f"Saved reports to: {REPORTS_DIR}")
    print(f"Saved images to: {IMAGES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()