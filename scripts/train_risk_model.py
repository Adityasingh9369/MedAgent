"""
Training entry point for the XGBoost readmission risk model.
Runs feature engineering from MIMIC data, trains the model,
and logs everything to MLflow.

Run with:
    python -m scripts.train_risk_model
"""

from src.ml.risk_model import train_and_evaluate
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting readmission risk model training...")
    model, run_id = train_and_evaluate()
    logger.info(f"Training complete. MLflow run ID: {run_id}")
    logger.info("To view results, run: mlflow ui")
    logger.info("Then open http://localhost:5000 in your browser")