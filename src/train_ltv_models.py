from __future__ import annotations

import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import BASE_DIR


TARGET_COLUMN = "future_6m_revenue"

LTV_FEATURES_DIR = BASE_DIR / "data" / "ltv_features"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
IMAGES_DIR = BASE_DIR / "images" / "modeling"

for directory in [MODELS_DIR, REPORTS_DIR, PREDICTIONS_DIR, IMAGES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_ltv_data():
    X_train = pd.read_parquet(LTV_FEATURES_DIR / "X_train_ltv.parquet")
    X_test = pd.read_parquet(LTV_FEATURES_DIR / "X_test_ltv.parquet")
    y_train = pd.read_parquet(LTV_FEATURES_DIR / "y_train_ltv.parquet")[TARGET_COLUMN]
    y_test = pd.read_parquet(LTV_FEATURES_DIR / "y_test_ltv.parquet")[TARGET_COLUMN]

    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_regression_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)
    predictions = np.clip(predictions, 0, None)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    nonzero_mask = y_test > 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(
            np.abs((y_test[nonzero_mask] - predictions[nonzero_mask]) / y_test[nonzero_mask])
        )
    else:
        mape = np.nan

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape_nonzero_actuals": mape,
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    models = build_models()
    trained_models = {}
    results = []

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)

        metrics = evaluate_regression_model(model, X_test, y_test)

        results.append(
            {
                "model_name": model_name,
                **metrics,
            }
        )

        trained_models[model_name] = model

        print(
            f"{model_name} | "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"MAE: {metrics['mae']:.2f} | "
            f"R2: {metrics['r2']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("rmse", ascending=True)

    best_model_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_model_name]

    return trained_models, results_df, best_model_name, best_model


def save_models(trained_models, best_model_name, best_model):
    for model_name, model in trained_models.items():
        joblib.dump(model, MODELS_DIR / f"ltv_{model_name}.pkl")

    joblib.dump(best_model, MODELS_DIR / "best_ltv_model.pkl")

    with open(MODELS_DIR / "best_ltv_model_name.txt", "w", encoding="utf-8") as f:
        f.write(best_model_name)


def save_results_report(results_df, best_model_name):
    results_df.to_csv(REPORTS_DIR / "ltv_model_comparison.csv", index=False)

    with open(REPORTS_DIR / "ltv_model_report.md", "w", encoding="utf-8") as f:
        f.write("# LTV Prediction Model Report\n\n")
        f.write("## Model Comparison\n\n")
        f.write(results_df.round(4).to_markdown(index=False))
        f.write("\n\n")
        f.write(f"Best model selected by lowest RMSE: **{best_model_name}**\n")


def plot_actual_vs_predicted(model, X_test, y_test, best_model_name):
    preds = np.clip(model.predict(X_test), 0, None)

    sample_size = min(5000, len(y_test))
    sample_idx = np.random.default_rng(42).choice(len(y_test), sample_size, replace=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test.iloc[sample_idx], preds[sample_idx], alpha=0.4)
    plt.xlabel("Actual Future 6M Revenue")
    plt.ylabel("Predicted Future 6M Revenue")
    plt.title(f"Actual vs Predicted LTV - {best_model_name}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "ltv_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_residuals(model, X_test, y_test, best_model_name):
    preds = np.clip(model.predict(X_test), 0, None)
    residuals = y_test - preds

    sample_size = min(5000, len(y_test))
    sample_idx = np.random.default_rng(42).choice(len(y_test), sample_size, replace=False)

    plt.figure(figsize=(7, 6))
    plt.scatter(preds[sample_idx], residuals.iloc[sample_idx], alpha=0.4)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Future 6M Revenue")
    plt.ylabel("Residual")
    plt.title(f"Residual Plot - {best_model_name}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "ltv_residual_plot.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_feature_importance(model, X_train, best_model_name):
    if not hasattr(model, "feature_importances_"):
        print(f"{best_model_name} does not support feature importances. Skipping.")
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
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(f"Top 20 LTV Feature Importances - {best_model_name}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "ltv_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()

    importance_df.to_csv(REPORTS_DIR / "ltv_feature_importance.csv", index=False)


def save_predictions(model, X_test, y_test):
    preds = np.clip(model.predict(X_test), 0, None)

    predictions_df = pd.DataFrame(
        {
            "actual_future_6m_revenue": y_test.values,
            "predicted_future_6m_revenue": preds,
            "absolute_error": np.abs(y_test.values - preds),
        }
    )

    predictions_df.to_csv(
        PREDICTIONS_DIR / "customer_ltv_predictions_test.csv",
        index=False,
    )


def save_metadata(best_model_name, results_df, X_train, X_test, y_train, y_test):
    metadata = {
        "phase": "Phase 6 - LTV Prediction",
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "target_mean_train": float(y_train.mean()),
        "target_mean_test": float(y_test.mean()),
        "target_nonzero_rate_train": float((y_train > 0).mean()),
        "target_nonzero_rate_test": float((y_test > 0).mean()),
        "best_model_metrics": results_df.iloc[0].to_dict(),
    }

    with open(REPORTS_DIR / "ltv_model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def main() -> None:
    print("=" * 80)
    print("PHASE 6: LTV PREDICTION MODELING")
    print("=" * 80)

    print("\nLoading LTV feature datasets...")
    X_train, X_test, y_train, y_test = load_ltv_data()

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Train target mean: {y_train.mean():.2f}")
    print(f"Test target mean: {y_test.mean():.2f}")
    print(f"Train non-zero target rate: {(y_train > 0).mean():.2%}")
    print(f"Test non-zero target rate: {(y_test > 0).mean():.2%}")

    trained_models, results_df, best_model_name, best_model = train_and_evaluate(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nModel comparison:")
    print(results_df.round(4).to_string(index=False))

    print(f"\nBest model: {best_model_name}")

    save_models(trained_models, best_model_name, best_model)
    save_results_report(results_df, best_model_name)

    plot_actual_vs_predicted(best_model, X_test, y_test, best_model_name)
    plot_residuals(best_model, X_test, y_test, best_model_name)
    plot_feature_importance(best_model, X_train, best_model_name)

    save_predictions(best_model, X_test, y_test)
    save_metadata(best_model_name, results_df, X_train, X_test, y_train, y_test)

    print("\nPhase 6 complete.")
    print(f"Saved models to: {MODELS_DIR}")
    print(f"Saved reports to: {REPORTS_DIR}")
    print(f"Saved images to: {IMAGES_DIR}")
    print(f"Saved predictions to: {PREDICTIONS_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()