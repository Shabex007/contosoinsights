from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import BASE_DIR


FEATURES_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
IMAGES_DIR = BASE_DIR / "images" / "modeling"
REPORTS_DIR = BASE_DIR / "reports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

TARGET_COLUMN = "is_churned_180d"

for directory in [MODELS_DIR, PREDICTIONS_DIR, IMAGES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_training_data():
    """Load Phase 4 model-ready train/test datasets."""
    X_train = pd.read_parquet(FEATURES_DIR / "X_train.parquet")
    X_test = pd.read_parquet(FEATURES_DIR / "X_test.parquet")
    y_train = pd.read_parquet(FEATURES_DIR / "y_train.parquet")[TARGET_COLUMN]
    y_test = pd.read_parquet(FEATURES_DIR / "y_test.parquet")[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """Create candidate churn models."""
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=2000,
            n_jobs=-1,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained classification model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """Train candidate models and return metrics."""
    models = build_models()
    trained_models = {}
    results = []

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        results.append(
            {
                "model_name": model_name,
                **metrics,
            }
        )

        trained_models[model_name] = model

        print(
            f"{model_name} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values(
        "roc_auc",
        ascending=False,
    )

    best_model_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]

    return trained_models, results_df, best_model_name, best_model


def save_models(trained_models: dict, best_model_name: str, best_model) -> None:
    """Save trained churn models."""
    for model_name, model in trained_models.items():
        joblib.dump(model, MODELS_DIR / f"churn_{model_name}.pkl")

    joblib.dump(best_model, MODELS_DIR / "best_churn_model.pkl")

    with open(MODELS_DIR / "best_churn_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_model_name)


def save_model_results(results_df: pd.DataFrame) -> None:
    """Save model comparison metrics."""
    results_df.to_csv(REPORTS_DIR / "churn_model_comparison.csv", index=False)

    with open(REPORTS_DIR / "churn_model_report.md", "w", encoding="utf-8") as f:
        f.write("# Churn Model Report\n\n")
        f.write("## Model Comparison\n\n")
        f.write(results_df.round(4).to_markdown(index=False))
        f.write("\n\n")
        f.write("## Selection Criteria\n\n")
        f.write(
            "The best churn model was selected using ROC-AUC as the primary metric. "
            "Recall and precision were also reviewed because churn prediction is an imbalanced classification problem.\n"
        )


def plot_confusion_matrix(model, X_test, y_test, best_model_name: str) -> None:
    """Save confusion matrix for best model."""
    y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)

    display.plot(values_format="d")
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.savefig(IMAGES_DIR / "churn_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(model, X_test, y_test, best_model_name: str) -> None:
    """Save ROC curve for best model."""
    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"{best_model_name} AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {best_model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "churn_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, X_train: pd.DataFrame, best_model_name: str) -> None:
    """Save top feature importance chart if supported."""
    if not hasattr(model, "feature_importances_"):
        print(f"{best_model_name} does not support feature_importances_. Skipping plot.")
        return

    importance_df = (
        pd.DataFrame(
            {
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }
        )
        .sort_values("importance", ascending=False)
        .head(20)
    )

    plt.figure(figsize=(10, 8))
    plt.barh(
        importance_df["feature"].iloc[::-1],
        importance_df["importance"].iloc[::-1],
    )
    plt.title(f"Top 20 Feature Importances - {best_model_name}")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "churn_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(
        REPORTS_DIR / "churn_feature_importance.csv",
        index=False,
    )


def create_risk_segment(probability: float) -> str:
    """Convert churn probability to business-friendly risk segment."""
    if probability >= 0.80:
        return "High Risk"
    if probability >= 0.50:
        return "Medium Risk"
    return "Low Risk"


def save_test_predictions(model, X_test, y_test) -> None:
    """Save test-set churn predictions."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "actual_churn": y_test.values,
            "predicted_churn": y_pred,
            "churn_probability": y_proba,
        }
    )

    predictions["risk_segment"] = predictions["churn_probability"].apply(
        create_risk_segment
    )

    predictions.to_csv(
        PREDICTIONS_DIR / "customer_churn_scores_test.csv",
        index=False,
    )


def save_metadata(
    best_model_name: str,
    results_df: pd.DataFrame,
    X_train,
    X_test,
    y_train,
    y_test,
) -> None:
    """Save modeling metadata."""
    metadata = {
        "phase": "Phase 5 - Churn Prediction Modeling",
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
        "best_model_metrics": results_df.iloc[0].to_dict(),
    }

    with open(REPORTS_DIR / "churn_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def print_classification_report(model, X_test, y_test, best_model_name: str) -> None:
    """Print and save detailed classification report."""
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\nClassification Report:")
    print(report)

    with open(REPORTS_DIR / "churn_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Classification Report - {best_model_name}\n")
        f.write("=" * 80)
        f.write("\n")
        f.write(report)


def main() -> None:
    print("=" * 80)
    print("PHASE 5: CHURN PREDICTION MODELING")
    print("=" * 80)

    print("\nLoading Phase 4 feature datasets...")
    X_train, X_test, y_train, y_test = load_training_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean():.2%}")
    print(f"Test churn rate: {y_test.mean():.2%}")

    trained_models, results_df, best_model_name, best_model = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))

    print(f"\nBest model: {best_model_name}")

    save_models(
        trained_models=trained_models,
        best_model_name=best_model_name,
        best_model=best_model,
    )

    save_model_results(results_df)

    plot_confusion_matrix(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_model_name=best_model_name,
    )

    plot_roc_curve(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_model_name=best_model_name,
    )

    plot_feature_importance(
        model=best_model,
        X_train=X_train,
        best_model_name=best_model_name,
    )

    save_test_predictions(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
    )

    save_metadata(
        best_model_name=best_model_name,
        results_df=results_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    print_classification_report(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        best_model_name=best_model_name,
    )

    print("\nPhase 5 complete.")
    print(f"Saved models to: {MODELS_DIR}")
    print(f"Saved reports to: {REPORTS_DIR}")
    print(f"Saved images to: {IMAGES_DIR}")
    print(f"Saved predictions to: {PREDICTIONS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()