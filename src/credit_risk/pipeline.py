"""
Main training pipeline orchestrator.
Runs the complete end-to-end credit risk modeling pipeline.
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
import logging
import json
import sys

from credit_risk.config import config
from credit_risk.data.ingestion import ingest_pipeline
from credit_risk.data.preprocessing import preprocess_pipeline, save_preprocessor
from credit_risk.models.training import (
    train_all_models, select_best_model, save_model
)
from credit_risk.models.evaluation import evaluate_model_comprehensive
from credit_risk.scoring import create_scorer_from_training

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "INFO", log_file: Path = None) -> None:
    """Configure logging."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )


def run_pipeline(
    raw_data_path: Path = None,
    output_dir: Path = None,
    log_level: str = "INFO"
) -> Dict[str, Any]:
    """
    Run complete training pipeline.

    Args:
        raw_data_path: Path to raw data file.
        output_dir: Directory for all outputs.
        log_level: Logging level.

    Returns:
        Dictionary with pipeline results and paths.
    """
    # Setup
    if output_dir is None:
        output_dir = Path("outputs") / f"run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    reports_dir = output_dir / "reports"
    plots_dir = output_dir / "plots"

    setup_logging(log_level, output_dir / "pipeline.log")

    logger.info("=" * 60)
    logger.info("Starting Credit Risk Modeling Pipeline")
    logger.info("=" * 60)

    results = {
        "pipeline_start": pd.Timestamp.now().isoformat(),
        "output_dir": str(output_dir),
        "stages": {}
    }

    try:
        # ============ STAGE 1: DATA INGESTION ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 1: DATA INGESTION")
        logger.info("=" * 60)

        df, validation_report = ingest_pipeline(raw_data_path)
        results["stages"]["ingestion"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "default_rate": float(df["default"].mean()),
            "validation_report": validation_report
        }

        # Save raw processed data
        processed_path = output_dir / "data" / "processed.parquet"
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed_path, index=False)
        logger.info(f"Saved processed data to {processed_path}")

        # ============ STAGE 2: PREPROCESSING ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 2: PREPROCESSING & FEATURE ENGINEERING")
        logger.info("=" * 60)

        X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df)
        results["stages"]["preprocessing"] = {
            "train_shape": list(X_train.shape),
            "test_shape": list(X_test.shape),
            "n_features": X_train.shape[1],
            "train_target_dist": y_train.value_counts().to_dict(),
            "test_target_dist": y_test.value_counts().to_dict()
        }

        # Save preprocessor
        save_preprocessor(preprocessor, models_dir / "preprocessor.pkl")

        # Save train/test splits
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        train_df.to_parquet(output_dir / "data" / "train.parquet")
        test_df.to_parquet(output_dir / "data" / "test.parquet")

        # ============ STAGE 3: MODEL TRAINING ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 3: MODEL TRAINING & HYPERPARAMETER TUNING")
        logger.info("=" * 60)

        model_results = train_all_models(X_train, y_train, X_test, y_test)
        results["stages"]["training"] = {}

        for name, result in model_results.items():
            results["stages"]["training"][name] = {
                "best_params": result["best_params"],
                "cv_scores": {k: v.tolist() for k, v in result["cv_scores"].items()},
                "optimal_threshold": result["optimal_threshold"],
                "val_metrics": result["val_metrics"]
            }

        # ============ STAGE 4: MODEL SELECTION ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 4: MODEL SELECTION")
        logger.info("=" * 60)

        best_name, best_model, best_result = select_best_model(
            model_results, metric="roc_auc"
        )

        results["stages"]["selection"] = {
            "best_model": best_name,
            "best_roc_auc": best_result["val_metrics"]["roc_auc"],
            "best_threshold": best_result["optimal_threshold"]
        }

        # ============ STAGE 5: COMPREHENSIVE EVALUATION ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 5: COMPREHENSIVE EVALUATION")
        logger.info("=" * 60)

        eval_results = evaluate_model_comprehensive(
            best_model, preprocessor, X_train.columns.tolist(),
            X_test, y_test, best_result["optimal_threshold"],
            reports_dir / best_name
        )

        results["stages"]["evaluation"] = {
            "metrics": eval_results["metrics"],
            "risk_distribution": eval_results["risk_bands"]["distribution"]
        }

        # ============ STAGE 6: MODEL PERSISTENCE ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 6: MODEL PERSISTENCE & SCORER CREATION")
        logger.info("=" * 60)

        # Save best model
        save_model(
            best_model,
            models_dir / "best_model.pkl",
            metadata={
                "model_name": best_name,
                "threshold": best_result["optimal_threshold"],
                "val_metrics": best_result["val_metrics"],
                "feature_names": X_train.columns.tolist(),
                "training_date": pd.Timestamp.now().isoformat()
            }
        )

        # Create scorer
        scorer = create_scorer_from_training(
            best_model, preprocessor, best_result["optimal_threshold"], models_dir
        )

        results["stages"]["persistence"] = {
            "model_path": str(models_dir / "best_model.pkl"),
            "preprocessor_path": str(models_dir / "preprocessor.pkl"),
            "scorer_ready": True
        }

        # ============ STAGE 7: GENERATE ARTIFACTS FOR POWER BI ============
        logger.info("\n" + "=" * 60)
        logger.info("STAGE 7: POWER BI EXPORT")
        logger.info("=" * 60)

        powerbi_dir = output_dir / "powerbi"
        powerbi_dir.mkdir(parents=True, exist_ok=True)

        # Score test set for Power BI
        test_scores = scorer.score_batch(test_df.drop(columns=["default"]))
        powerbi_df = pd.DataFrame([s.to_dict() for s in test_scores.scores])
        powerbi_df["actual_default"] = y_test.values
        powerbi_df.to_csv(powerbi_dir / "scored_applicants.csv", index=False)

        # Feature importance for Power BI
        importance_df = pd.DataFrame(
            list(eval_results["feature_importance"].items()),
            columns=["Feature", "Importance"]
        )
        importance_df.to_csv(powerbi_dir / "feature_importance.csv", index=False)

        # Risk band reference
        risk_band_df = pd.DataFrame(config.evaluation.risk_bands)
        risk_band_df.to_csv(powerbi_dir / "risk_bands.csv", index=False)

        # Model performance summary
        perf_df = pd.DataFrame([{
            "model": best_name,
            "metric": k,
            "value": v
        } for k, v in eval_results["metrics"].items()])
        perf_df.to_csv(powerbi_dir / "model_performance.csv", index=False)

        results["stages"]["powerbi"] = {
            "export_dir": str(powerbi_dir),
            "files": [f.name for f in powerbi_dir.glob("*")]
        }

        # ============ COMPLETE ============
        results["pipeline_end"] = pd.Timestamp.now().isoformat()
        results["status"] = "success"

        # Save pipeline results
        results_path = output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)
        logger.info(f"Best Model: {best_name}")
        logger.info(f"Test ROC-AUC: {eval_results['metrics']['roc_auc']:.4f}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Results saved to: {results_path}")

    except Exception as e:
        logger.exception("Pipeline failed")
        results["status"] = "failed"
        results["error"] = str(e)
        results["pipeline_end"] = pd.Timestamp.now().isoformat()

        results_path = output_dir / "pipeline_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        raise

    return results


def run_inference_pipeline(
    input_path: Path,
    model_dir: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Run inference on new data.

    Args:
        input_path: Path to input CSV/Parquet.
        model_dir: Directory with model artifacts.
        output_path: Path to save predictions.

    Returns:
        DataFrame with predictions.
    """
    logger.info(f"Running inference on {input_path}")

    # Load data
    if input_path.suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        df = pd.read_parquet(input_path)

    # Create scorer
    scorer = CreditRiskScorer(
        model_path=model_dir / "best_model.pkl",
        preprocessor_path=model_dir / "preprocessor.pkl"
    )

    # Score
    result = scorer.score_batch(df)

    # Save
    output_df = pd.DataFrame([s.to_dict() for s in result.scores])
    output_df.to_csv(output_path, index=False)

    logger.info(f"Scored {len(output_df)} applicants, saved to {output_path}")
    return output_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Credit Risk Pipeline")
    parser.add_argument("--raw-data", type=Path, help="Raw data file path")
    parser.add_argument("--output-dir", type=Path, help="Output directory")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    parser.add_argument("--inference", action="store_true", help="Run inference mode")
    parser.add_argument("--input", type=Path, help="Input file for inference")
    parser.add_argument("--model-dir", type=Path, help="Model directory for inference")
    parser.add_argument("--output", type=Path, help="Output file for inference")

    args = parser.parse_args()

    if args.inference:
        run_inference_pipeline(args.input, args.model_dir, args.output)
    else:
        run_pipeline(args.raw_data, args.output_dir, args.log_level)