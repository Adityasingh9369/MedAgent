import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import shap
from loguru import logger
from pathlib import Path
from src.ml.feature_engineering import build_feature_matrix

MLFLOW_EXPERIMENT = "clinicalagent-readmission"
MODEL_OUTPUT_DIR = Path("models")

FEATURE_COLS = [
    "age", "gender_m", "admission_type_emergency",
    "los_days", "diagnosis_count",
    "has_diabetes", "has_heart_failure", "has_hypertension",
    "has_renal_disease", "has_pneumonia", "has_sepsis", "has_copd",
    "num_icu_stays", "total_icu_los", "max_icu_los",
]

TARGET_COL = "readmitted_30d"


def train_and_evaluate():
    """
    Full training pipeline:
    1. Build feature matrix from MIMIC data
    2. Train XGBoost with cross-validation
    3. Log everything to MLflow
    4. Save model artifact
    """
    logger.info("=" * 55)
    logger.info("PHASE 4: ML RISK MODEL TRAINING")
    logger.info("=" * 55)

    # Build features
    df = build_feature_matrix()
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    logger.info(f"Dataset: {len(df)} samples | {y.sum()} positive | {y.mean():.2%} readmission rate")

    # Class imbalance — compute scale_pos_weight
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    scale_pos_weight = neg / pos if pos > 0 else 1
    logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="xgboost-readmission-v1"):

        # Model params
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "auc",
            "random_state": 42,
            # "use_label_encoder": False,
        }

        mlflow.log_params(params)
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("n_features", len(FEATURE_COLS))
        mlflow.log_param("positive_rate", round(float(y.mean()), 4))

        model = XGBClassifier(**params)

        # Cross-validation
        logger.info("Running 5-fold cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

        logger.info(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        mlflow.log_metric("cv_auc_mean", round(float(cv_scores.mean()), 4))
        mlflow.log_metric("cv_auc_std", round(float(cv_scores.std()), 4))

        # Train on full dataset
        logger.info("Training on full dataset...")
        model.fit(X, y)

        # Predictions
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)

        # Metrics
        train_auc = roc_auc_score(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba)

        mlflow.log_metric("train_auc", round(float(train_auc), 4))
        mlflow.log_metric("brier_score", round(float(brier), 4))

        logger.info(f"Train AUC: {train_auc:.4f}")
        logger.info(f"Brier Score: {brier:.4f}")

        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        mlflow.log_metric("precision_class1", round(float(report["1"]["precision"]), 4))
        mlflow.log_metric("recall_class1", round(float(report["1"]["recall"]), 4))
        mlflow.log_metric("f1_class1", round(float(report["1"]["f1-score"]), 4))

        # Feature importance
        importance_df = pd.DataFrame({
            "feature": FEATURE_COLS,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        logger.info("Top 5 features:")
        for _, row in importance_df.head(5).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            mlflow.log_metric(f"importance_{row['feature']}", round(float(row['importance']), 4))

        # Save model
        MODEL_OUTPUT_DIR.mkdir(exist_ok=True)
        mlflow.xgboost.log_model(model, "xgboost_readmission_model")
        model.save_model(str(MODEL_OUTPUT_DIR / "readmission_model.json"))

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run ID: {run_id}")
        logger.info(f"Model saved to models/readmission_model.json")

        print("\n" + "=" * 55)
        print("TRAINING COMPLETE")
        print("=" * 55)
        print(f"  CV AUC (5-fold):   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Train AUC:         {train_auc:.4f}")
        print(f"  Brier Score:       {brier:.4f}")
        print(f"  Positive Rate:     {y.mean():.2%}")
        print("=" * 55)
        print("\nTop Features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']:<30} {row['importance']:.4f}")
        print("=" * 55)

        return model, run_id