from __future__ import annotations

import json
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from src.config import BASE_DIR, PROCESSED_DATA_DIR
from src.prepare_ltv_dataset import (
    add_engineered_features,
    apply_log_transforms,
    build_preprocessor,
    clean_values,
    get_feature_names,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
)


TARGET_COLUMN = "future_6m_revenue"
SPEND_FLAG_COLUMN = "will_generate_future_revenue"

OUTPUT_DIR = BASE_DIR / "data" / "two_stage_ltv"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
PREDICTIONS_DIR = BASE_DIR / "data" / "predictions"
IMAGES_DIR = BASE_DIR / "images" / "modeling"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

for directory in [
    OUTPUT_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    PREDICTIONS_DIR,
    IMAGES_DIR,
    ARTIFACTS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    path = PROCESSED_DATA_DIR / "customer_ml_features.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m src.extract_data` first."
        )

    return pd.read_parquet(path)


def prepare_dataset() -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    df = load_data()

    df = add_engineered_features(df)
    df = clean_values(df)
    df = apply_log_transforms(df)

    available_numeric = [
        col for col in NUMERIC_FEATURES + ENGINEERED_FEATURES if col in df.columns
    ]

    available_categorical = [
        col for col in CATEGORICAL_FEATURES if col in df.columns
    ]

    X_raw = df[available_numeric + available_categorical].copy()

    y_spend = (df[TARGET_COLUMN] > 0).astype(int)
    y_amount = df[TARGET_COLUMN].clip(lower=0)

    return X_raw, y_spend, y_amount, available_numeric, available_categorical


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def evaluate_regressor(model, X_test, y_test) -> dict:
    preds = np.clip(model.predict(X_test), 0, None)

    return {
        "mae": mean_absolute_error(y_test, preds),
        "rmse": np.sqrt(mean_squared_error(y_test, preds)),
        "r2": r2_score(y_test, preds),
    }


def train_stage1_classifiers(X_train, X_test, y_train, y_test):
    models = {
        "ltv_stage1_logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        "ltv_stage1_random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining Stage 1 classifier: {name}")
        model.fit(X_train, y_train)

        metrics = evaluate_classifier(model, X_test, y_test)

        results.append({"model_name": name, **metrics})
        trained_models[name] = model

        print(
            f"{name} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Precision: {metrics['precision']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)

    best_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_name]

    return trained_models, results_df, best_name, best_model


def train_stage2_regressors(X_train_nonzero, X_test_nonzero, y_train_nonzero, y_test_nonzero):
    models = {
        "ltv_stage2_ridge_regression": Ridge(alpha=1.0),
        "ltv_stage2_random_forest_regressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining Stage 2 regressor: {name}")
        model.fit(X_train_nonzero, y_train_nonzero)

        metrics = evaluate_regressor(model, X_test_nonzero, y_test_nonzero)

        results.append({"model_name": name, **metrics})
        trained_models[name] = model

        print(
            f"{name} | "
            f"RMSE: {metrics['rmse']:.2f} | "
            f"MAE: {metrics['mae']:.2f} | "
            f"R2: {metrics['r2']:.4f}"
        )

    results_df = pd.DataFrame(results).sort_values("rmse", ascending=True)

    best_name = results_df.iloc[0]["model_name"]
    best_model = trained_models[best_name]

    return trained_models, results_df, best_name, best_model


def plot_stage1_roc(model, X_test, y_test, model_name):
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Stage 1 Future Revenue Classifier ROC - {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "two_stage_ltv_stage1_roc.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_stage1_confusion_matrix(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot(values_format="d")
    plt.title(f"Stage 1 Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "two_stage_ltv_stage1_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_stage2_actual_vs_predicted(model, X_test_nonzero, y_test_nonzero, model_name):
    preds = np.clip(model.predict(X_test_nonzero), 0, None)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test_nonzero, preds, alpha=0.5)
    plt.xlabel("Actual Future Revenue")
    plt.ylabel("Predicted Future Revenue")
    plt.title(f"Stage 2 Actual vs Predicted - {model_name}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "two_stage_ltv_stage2_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()


def save_predictions(
    stage1_model,
    stage2_model,
    X_test,
    y_spend_test,
    y_amount_test,
):
    spend_probability = stage1_model.predict_proba(X_test)[:, 1]
    predicted_spend_flag = (spend_probability >= 0.5).astype(int)
    predicted_amount_if_spender = np.clip(stage2_model.predict(X_test), 0, None)

    final_predicted_ltv = spend_probability * predicted_amount_if_spender

    predictions = pd.DataFrame(
        {
            "actual_will_generate_revenue": y_spend_test.values,
            "actual_future_6m_revenue": y_amount_test.values,
            "spend_probability": spend_probability,
            "predicted_spend_flag": predicted_spend_flag,
            "predicted_amount_if_spender": predicted_amount_if_spender,
            "final_two_stage_ltv_prediction": final_predicted_ltv,
            "absolute_error": np.abs(y_amount_test.values - final_predicted_ltv),
        }
    )

    predictions.to_csv(
        PREDICTIONS_DIR / "customer_two_stage_ltv_predictions_test.csv",
        index=False,
    )


def save_reports(
    stage1_results,
    stage2_results,
    best_stage1_name,
    best_stage2_name,
    X_train,
    X_test,
    y_spend_train,
    y_spend_test,
    y_amount_train,
    y_amount_test,
):
    stage1_results.to_csv(
        REPORTS_DIR / "two_stage_ltv_stage1_classifier_comparison.csv",
        index=False,
    )

    stage2_results.to_csv(
        REPORTS_DIR / "two_stage_ltv_stage2_regressor_comparison.csv",
        index=False,
    )

    metadata = {
        "phase": "Phase 6B - Two Stage LTV Prediction",
        "stage1_target": SPEND_FLAG_COLUMN,
        "stage2_target": TARGET_COLUMN,
        "best_stage1_model": best_stage1_name,
        "best_stage2_model": best_stage2_name,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "train_spender_rate": float(y_spend_train.mean()),
        "test_spender_rate": float(y_spend_test.mean()),
        "train_future_revenue_mean": float(y_amount_train.mean()),
        "test_future_revenue_mean": float(y_amount_test.mean()),
        "stage1_best_metrics": stage1_results.iloc[0].to_dict(),
        "stage2_best_metrics": stage2_results.iloc[0].to_dict(),
    }

    with open(REPORTS_DIR / "two_stage_ltv_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    with open(REPORTS_DIR / "two_stage_ltv_model_report.md", "w", encoding="utf-8") as f:
        f.write("# Two-Stage LTV Prediction Report\n\n")
        f.write("## Why Two-Stage Modeling?\n\n")
        f.write(
            "The future 6-month revenue target is highly zero-inflated. "
            "Only a small percentage of customers generate future revenue, so a single regression model struggles. "
            "The two-stage approach first predicts whether a customer will spend, then estimates revenue amount for likely spenders.\n\n"
        )

        f.write("## Stage 1: Future Revenue Classifier\n\n")
        f.write(stage1_results.round(4).to_markdown(index=False))
        f.write("\n\n")

        f.write("## Stage 2: Revenue Amount Regressor\n\n")
        f.write(stage2_results.round(4).to_markdown(index=False))
        f.write("\n\n")


def main() -> None:
    print("=" * 80)
    print("PHASE 6B: TWO-STAGE LTV PREDICTION")
    print("=" * 80)

    X_raw, y_spend, y_amount, numeric_features, categorical_features = prepare_dataset()

    print(f"Raw X shape: {X_raw.shape}")
    print(f"Future spender rate: {y_spend.mean():.2%}")
    print(f"Future revenue mean: {y_amount.mean():.2f}")

    X_train_raw, X_test_raw, y_spend_train, y_spend_test, y_amount_train, y_amount_test = train_test_split(
        X_raw,
        y_spend,
        y_amount,
        test_size=0.2,
        random_state=42,
        stratify=y_spend,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    X_train_array = preprocessor.fit_transform(X_train_raw)
    X_test_array = preprocessor.transform(X_test_raw)

    feature_names = get_feature_names(preprocessor, numeric_features, categorical_features)

    X_train = pd.DataFrame(X_train_array, columns=feature_names)
    X_test = pd.DataFrame(X_test_array, columns=feature_names)

    print(f"Processed X_train shape: {X_train.shape}")
    print(f"Processed X_test shape: {X_test.shape}")
    print(f"Train future spender rate: {y_spend_train.mean():.2%}")
    print(f"Test future spender rate: {y_spend_test.mean():.2%}")

    stage1_models, stage1_results, best_stage1_name, best_stage1_model = train_stage1_classifiers(
        X_train,
        X_test,
        y_spend_train,
        y_spend_test,
    )

    nonzero_train_mask = y_amount_train > 0
    nonzero_test_mask = y_amount_test > 0

    X_train_nonzero = X_train.loc[nonzero_train_mask.reset_index(drop=True)]
    X_test_nonzero = X_test.loc[nonzero_test_mask.reset_index(drop=True)]
    y_train_nonzero = y_amount_train[nonzero_train_mask]
    y_test_nonzero = y_amount_test[nonzero_test_mask]

    print(f"\nStage 2 non-zero train rows: {len(y_train_nonzero):,}")
    print(f"Stage 2 non-zero test rows: {len(y_test_nonzero):,}")

    stage2_models, stage2_results, best_stage2_name, best_stage2_model = train_stage2_regressors(
        X_train_nonzero,
        X_test_nonzero,
        y_train_nonzero,
        y_test_nonzero,
    )

    print("\nStage 1 comparison:")
    print(stage1_results.round(4).to_string(index=False))

    print("\nStage 2 comparison:")
    print(stage2_results.round(4).to_string(index=False))

    joblib.dump(preprocessor, ARTIFACTS_DIR / "two_stage_ltv_preprocessor.pkl")
    joblib.dump(feature_names, ARTIFACTS_DIR / "two_stage_ltv_feature_names.pkl")

    for name, model in stage1_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    for name, model in stage2_models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    joblib.dump(best_stage1_model, MODELS_DIR / "best_two_stage_ltv_stage1_model.pkl")
    joblib.dump(best_stage2_model, MODELS_DIR / "best_two_stage_ltv_stage2_model.pkl")

    plot_stage1_roc(best_stage1_model, X_test, y_spend_test, best_stage1_name)
    plot_stage1_confusion_matrix(best_stage1_model, X_test, y_spend_test, best_stage1_name)
    plot_stage2_actual_vs_predicted(
        best_stage2_model,
        X_test_nonzero,
        y_test_nonzero,
        best_stage2_name,
    )

    save_predictions(
        stage1_model=best_stage1_model,
        stage2_model=best_stage2_model,
        X_test=X_test,
        y_spend_test=y_spend_test,
        y_amount_test=y_amount_test,
    )

    save_reports(
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        best_stage1_name=best_stage1_name,
        best_stage2_name=best_stage2_name,
        X_train=X_train,
        X_test=X_test,
        y_spend_train=y_spend_train,
        y_spend_test=y_spend_test,
        y_amount_train=y_amount_train,
        y_amount_test=y_amount_test,
    )

    print("\nPhase 6B complete.")
    print(f"Saved two-stage predictions to: {PREDICTIONS_DIR}")
    print(f"Saved reports to: {REPORTS_DIR}")
    print(f"Saved images to: {IMAGES_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()