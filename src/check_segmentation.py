import json
import pandas as pd

from src.config import BASE_DIR


SEGMENTS_DIR = BASE_DIR / "data" / "segments"
REPORTS_DIR = BASE_DIR / "reports"


def main() -> None:
    segments_path = SEGMENTS_DIR / "customer_segments.csv"
    profiles_path = SEGMENTS_DIR / "customer_segment_profiles.csv"
    metadata_path = REPORTS_DIR / "segmentation_metadata.json"
    k_selection_path = REPORTS_DIR / "segmentation_k_selection.csv"

    if not segments_path.exists():
        raise FileNotFoundError(
            "customer_segments.csv not found. Run `python -m src.customer_segmentation` first."
        )

    segments = pd.read_csv(segments_path)
    profiles = pd.read_csv(profiles_path)
    k_selection = pd.read_csv(k_selection_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=" * 80)
    print("CUSTOMER SEGMENTATION CHECK")
    print("=" * 80)

    print(f"Best k: {metadata['best_k']}")
    print(f"Total customers: {metadata['total_customers']:,}")
    print(f"Segment count: {metadata['segment_count']}")

    print("\nK selection results:")
    print(k_selection.round(4).to_string(index=False))

    print("\nSegment profiles:")
    print(profiles.round(2).to_string(index=False))

    print("\nCustomer segment distribution:")
    print(segments["business_segment"].value_counts())

    print("\nSample segmented customers:")
    print(
        segments[
            [
                "customerkey",
                "customer_name",
                "cluster",
                "business_segment",
                "total_revenue",
                "frequency",
                "recency_days",
                "is_churned_180d",
            ]
        ].head()
    )

    print("=" * 80)


if __name__ == "__main__":
    main()