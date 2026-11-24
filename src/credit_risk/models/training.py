"""
Model training module with hyperparameter tuning and cross-validation.
Supports multiple algorithms with production-ready configuration.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
import pandas as pd
import numpy as np
import joblib
import logging
import warnings
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, RandomizedSearchCV, GridSearchCV
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
import inspect

from credit_risk.config import config

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


def get_model_class(model_name: str):
    """Get model class from string name."""
    model_map = {
        "sklearn.linear_model.LogisticRegression": LogisticRegression,
        "sklearn.tree.DecisionTreeClassifier": DecisionTreeClassifier,
        "sklearn.ensemble.RandomForestClassifier": RandomForestClassifier,
        "sklearn.ensemble.GradientBoostingClassifier": GradientBoostingClassifier
    }
    return model_map.get(model_name)


def instantiate_model(model_config: Dict) -> Any:
    """Instantiate model from configuration."""
    model_class = get_model_class(model_config["class"])
    if model_class is None:
        raise ValueError(f"Unknown model class: {model_config['class']}")

    # Filter params to only include valid ones for the model
    valid_params = inspect.signature(model_class.__init__).parameters
    filtered_params = {k: v for k, v in model_config["params"].items() if k in valid_params}

    return model_class(**filtered_params)


def tune_hyperparameters(model: Any, param_grid: Dict,
                         X_train: pd.DataFrame, y_train: pd.Series,
                         cv: int = 5, scoring: str = "roc_auc",
                         n_iter: int = 20, random_state: int = 42,
                         method: str = "random") -> Tuple[Any, Dict]:
    """
    Tune hyperparameters using RandomizedSearchCV or GridSearchCV.

    Args:
        model: Base estimator.
        param_grid: Parameter grid/distribution.
        X_train: Training features.
        y_train: Training target.
        cv: Cross-validation folds.
        scoring: Scoring metric.
        n_iter: Number of iterations for random search.
        random_state: Random seed.
        method: 'random' or 'grid'.

    Returns:
        Tuple of (best_estimator, best_params).
    """
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    if method == "random":
        search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv_strategy,
            scoring=scoring, n_jobs=-1, random_state=random_state, verbose=1
        )
    else:
        search = GridSearchCV(
            model, param_grid, cv=cv_strategy,
            scoring=scoring, n_jobs=-1, verbose=1
        )

    logger.info(f"Starting hyperparameter tuning with {method} search...")
    search.fit(X_train, y_train)

    logger.info(f"Best score: {search.best_score_:.4f}")
    logger.info(f"Best params: {search.best_params_}")

    return search.best_estimator_, search.best_params_


def evaluate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                   threshold: float = 0.5) -> Dict[str, float]:
    """
    Evaluate model on given data.

    Args:
        model: Fitted model.
        X: Features.
        y: True labels.
        threshold: Decision threshold for probability.

    Returns:
        Dictionary of metrics.
    """
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_pred_proba),
        "average_precision": average_precision_score(y, y_pred_proba)
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series,
                         cv: int = 5, scoring: Union[str, List[str]] = None) -> Dict[str, np.ndarray]:
    """
    Perform cross-validation with multiple metrics.

    Args:
        model: Model to evaluate.
        X: Features.
        y: Target.
        cv: Number of folds.
        scoring: List of scoring metrics or single metric string.

    Returns:
        Dictionary of metric arrays per fold.
    """
    if scoring is None:
        scoring = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]
    elif isinstance(scoring, str):
        scoring = [scoring]

    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.data.random_state)
    results = {}

    for metric in scoring:
        scores = cross_val_score(model, X, y, cv=cv_strategy, scoring=metric, n_jobs=-1)
        results[metric] = scores
        logger.info(f"CV {metric}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    return results


def find_optimal_threshold(model: Any, X_val: pd.DataFrame, y_val: pd.Series,
                           method: str = "f1", target_recall: float = 0.8,
                           target_precision: float = 0.7) -> Tuple[float, Dict]:
    """
    Find optimal decision threshold.

    Args:
        model: Fitted model.
        X_val: Validation features.
        y_val: Validation target.
        method: Optimization method ('f1', 'youden', 'precision_at_recall', 'recall_at_precision').
        target_recall: Target recall for precision_at_recall.
        target_precision: Target precision for recall_at_precision.

    Returns:
        Tuple of (optimal_threshold, threshold_metrics).
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    if method == "f1":
        # Maximize F1
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        # Remove last element (threshold=1.0 gives no positive predictions)
        optimal_idx = np.argmax(f1_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]

    elif method == "youden":
        # Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]

    elif method == "precision_at_recall":
        # Find threshold achieving target recall with max precision
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        valid = recall[:-1] >= target_recall
        if valid.any():
            optimal_idx = np.argmax(precision[:-1][valid])
            optimal_threshold = thresholds[valid][optimal_idx]
        else:
            optimal_threshold = 0.5
            logger.warning(f"Could not achieve target recall {target_recall}")

    elif method == "recall_at_precision":
        # Find threshold achieving target precision with max recall
        precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
        valid = precision[:-1] >= target_precision
        if valid.any():
            optimal_idx = np.argmax(recall[:-1][valid])
            optimal_threshold = thresholds[valid][optimal_idx]
        else:
            optimal_threshold = 0.5
            logger.warning(f"Could not achieve target precision {target_precision}")

    else:
        optimal_threshold = 0.5

    # Evaluate at optimal threshold
    metrics = evaluate_model(model, X_val, y_val, threshold=optimal_threshold)
    metrics["threshold"] = optimal_threshold

    logger.info(f"Optimal threshold ({method}): {optimal_threshold:.4f}")
    logger.info(f"Metrics at threshold: {metrics}")

    return optimal_threshold, metrics


def calibrate_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series,
                    method: str = "isotonic", cv: int = 3) -> Any:
    """
    Calibrate model probabilities.

    Args:
        model: Fitted model.
        X_train: Training features.
        y_train: Training target.
        method: Calibration method ('isotonic' or 'sigmoid').
        cv: Cross-validation folds for calibration.

    Returns:
        Calibrated classifier.
    """
    calibrated = CalibratedClassifierCV(model, method=method, cv=cv)
    calibrated.fit(X_train, y_train)
    logger.info(f"Model calibrated using {method} method")
    return calibrated


def train_all_models(X_train: pd.DataFrame, y_train: pd.DataFrame,
                     X_val: pd.DataFrame, y_val: pd.Series,
                     model_configs: Dict = None) -> Dict[str, Dict]:
    """
    Train and evaluate all configured models.

    Args:
        X_train: Training features.
        y_train: Training target.
        X_val: Validation features.
        y_val: Validation target.
        model_configs: Model configurations (uses global config if None).

    Returns:
        Dictionary of model results.
    """
    if model_configs is None:
        model_configs = config.model.models

    results = {}

    for model_name, model_config in model_configs.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}")
        logger.info(f"{'='*50}")

        # Instantiate base model
        base_model = instantiate_model(model_config)

        # Cross-validation on training data
        cv_results = cross_validate_model(
            base_model, X_train, y_train,
            cv=config.model.cv_folds,
            scoring=config.model.scoring
        )

        # Hyperparameter tuning
        best_model, best_params = tune_hyperparameters(
            base_model, model_config["search_space"],
            X_train, y_train,
            cv=config.model.cv_folds,
            scoring=config.model.scoring,
            n_iter=config.model.n_iter_search
        )

        # Calibrate probabilities
        calibrated_model = calibrate_model(best_model, X_train, y_train)

        # Find optimal threshold on validation set
        optimal_threshold, threshold_metrics = find_optimal_threshold(
            calibrated_model, X_val, y_val,
            method=config.evaluation.threshold_optimization,
            target_recall=config.evaluation.target_recall,
            target_precision=config.evaluation.target_precision
        )

        # Final evaluation on validation set
        val_metrics = evaluate_model(calibrated_model, X_val, y_val, threshold=optimal_threshold)

        # Store results
        results[model_name] = {
            "model": calibrated_model,
            "best_params": best_params,
            "cv_scores": cv_results,
            "optimal_threshold": optimal_threshold,
            "threshold_metrics": threshold_metrics,
            "val_metrics": val_metrics,
            "feature_importance": get_feature_importance(calibrated_model, X_train.columns)
        }

        logger.info(f"{model_name} validation ROC-AUC: {val_metrics['roc_auc']:.4f}")

    return results


def get_feature_importance(model: Any, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importance from model."""
    importance = {}

    try:
        # For calibrated classifier, get base estimator
        base_model = model.calibrated_classifiers_[0].base_estimator if hasattr(model, 'calibrated_classifiers_') else model

        if hasattr(base_model, "feature_importances_"):
            importance = dict(zip(feature_names, base_model.feature_importances_))
        elif hasattr(base_model, "coef_"):
            # For linear models, use absolute coefficient values
            coef = np.abs(base_model.coef_).flatten()
            importance = dict(zip(feature_names, coef))
    except Exception as e:
        logger.warning(f"Could not extract feature importance: {e}")

    # Sort by importance
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    return importance


def select_best_model(results: Dict[str, Dict], metric: str = "roc_auc") -> Tuple[str, Any, Dict]:
    """
    Select best model based on validation metric.

    Args:
        results: Results from train_all_models.
        metric: Metric to optimize.

    Returns:
        Tuple of (best_model_name, best_model, best_model_results).
    """
    best_score = -1
    best_name = None
    best_result = None

    for name, result in results.items():
        score = result["val_metrics"].get(metric, 0)
        if score > best_score:
            best_score = score
            best_name = name
            best_result = result

    logger.info(f"Best model: {best_name} with {metric} = {best_score:.4f}")
    return best_name, best_result["model"], best_result


def save_model(model: Any, filepath: Path, metadata: Dict = None) -> None:
    """Save model with metadata."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {"model": model}
    if metadata:
        save_dict["metadata"] = metadata

    joblib.dump(save_dict, filepath)
    logger.info(f"Saved model to {filepath}")


def load_model(filepath: Path) -> Tuple[Any, Dict]:
    """Load model and metadata."""
    save_dict = joblib.load(filepath)
    model = save_dict["model"]
    metadata = save_dict.get("metadata", {})
    logger.info(f"Loaded model from {filepath}")
    return model, metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    from credit_risk.data.ingestion import ingest_pipeline
    from credit_risk.data.preprocessing import preprocess_pipeline

    df, _ = ingest_pipeline()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df)

    # Quick test with one model
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    print(f"Test metrics: {metrics}")