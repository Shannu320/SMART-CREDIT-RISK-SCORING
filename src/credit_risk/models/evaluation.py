"""
Model evaluation and risk banding module.
Provides comprehensive evaluation metrics, visualizations, and risk segmentation.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import json
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve

from credit_risk.config import config

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and risk analysis."""

    def __init__(self, model: Any, preprocessor: Any, feature_names: List[str],
                 risk_bands: List[Dict] = None):
        self.model = model
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.risk_bands = risk_bands or config.evaluation.risk_bands

    def evaluate(self, X: pd.DataFrame, y: pd.Series,
                 threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.

        Args:
            X: Features.
            y: True labels.
            threshold: Decision threshold.

        Returns:
            Dictionary with all evaluation results.
        """
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= threshold).astype(int)

        results = {
            "threshold": threshold,
            "metrics": self._compute_metrics(y, y_pred, y_pred_proba),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(y, y_pred, output_dict=True),
            "roc_curve": self._compute_roc_curve(y, y_pred_proba),
            "pr_curve": self._compute_pr_curve(y, y_pred_proba),
            "calibration": self._compute_calibration(y, y_pred_proba),
            "brier_score": float(brier_score_loss(y, y_pred_proba)),
            "risk_bands": self._assign_risk_bands(y_pred_proba),
            "feature_importance": self._get_feature_importance()
        }

        return results

    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Compute all metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, average_precision_score, matthews_corrcoef
        )

        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
            "average_precision": float(average_precision_score(y_true, y_pred_proba)),
            "matthews_corrcoef": float(matthews_corrcoef(y_true, y_pred)),
            "specificity": self._specificity(y_true, y_pred),
            "npv": self._npv(y_true, y_pred)
        }

    def _specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    def _npv(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate negative predictive value."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        return float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    def _compute_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Compute ROC curve data."""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(roc_auc)
        }

    def _compute_pr_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """Compute Precision-Recall curve data."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(pr_auc)
        }

    def _compute_calibration(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                             n_bins: int = 10) -> Dict:
        """Compute calibration curve data."""
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')
        return {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
            "mean_predicted_prob": float(np.mean(y_pred_proba)),
            "fraction_positives": float(np.mean(y_true))
        }

    def _assign_risk_bands(self, y_pred_proba: np.ndarray) -> Dict:
        """Assign risk bands based on predicted probabilities."""
        bands = []
        for band in self.risk_bands:
            mask = (y_pred_proba >= band["min_score"]) & (y_pred_proba < band["max_score"])
            if band == self.risk_bands[-1]:  # Last band includes max
                mask = (y_pred_proba >= band["min_score"]) & (y_pred_proba <= band["max_score"])

            count = int(mask.sum())
            pct = float(count / len(y_pred_proba) * 100)

            bands.append({
                "name": band["name"],
                "min_score": band["min_score"],
                "max_score": band["max_score"],
                "color": band["color"],
                "count": count,
                "percentage": pct
            })

        return {"bands": bands, "distribution": {b["name"]: b["percentage"] for b in bands}}

    def _get_feature_importance(self) -> Dict[str, float]:
        """Extract feature importance."""
        importance = {}

        try:
            base_model = self.model
            if hasattr(self.model, 'calibrated_classifiers_'):
                base_model = self.model.calibrated_classifiers_[0].base_estimator

            if hasattr(base_model, "feature_importances_"):
                importance = dict(zip(self.feature_names, base_model.feature_importances_))
            elif hasattr(base_model, "coef_"):
                coef = np.abs(base_model.coef_).flatten()
                importance = dict(zip(self.feature_names, coef))
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {e}")

        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def plot_roc_curve(self, roc_data: Dict, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot ROC curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(roc_data["fpr"], roc_data["tpr"], linewidth=2,
                label=f'ROC Curve (AUC = {roc_data["auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {save_path}")

        return fig

    def plot_pr_curve(self, pr_data: Dict, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pr_data["recall"], pr_data["precision"], linewidth=2,
                label=f'PR Curve (AUC = {pr_data["auc"]:.3f})')
        ax.axhline(y=np.mean(pr_data["precision"]), color='k', linestyle='--',
                   linewidth=1, label='Baseline')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {save_path}")

        return fig

    def plot_calibration(self, cal_data: Dict, save_path: Optional[Path] = None) -> plt.Figure:
        """Plot calibration curve."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(cal_data["prob_pred"], cal_data["prob_true"], 'o-', linewidth=2,
                label='Model')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly Calibrated')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title('Calibration Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved calibration curve to {save_path}")

        return fig

    def plot_feature_importance(self, importance: Dict, top_n: int = 20,
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """Plot feature importance."""
        top_features = dict(list(importance.items())[:top_n])

        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        features = list(top_features.keys())[::-1]
        values = list(top_features.values())[::-1]

        colors = ['#C62828' if v > np.percentile(values, 75) else
                  '#EF6C00' if v > np.percentile(values, 50) else
                  '#FFCA28' if v > np.percentile(values, 25) else '#66BB6A'
                  for v in values]

        bars = ax.barh(features, values, color=colors)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(val + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {save_path}")

        return fig

    def plot_confusion_matrix(self, cm: List[List[int]],
                              save_path: Optional[Path] = None) -> plt.Figure:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_array = np.array(cm)
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Good', 'Predicted Bad'],
                    yticklabels=['Actual Good', 'Actual Bad'], ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        return fig

    def plot_risk_distribution(self, risk_bands: Dict,
                               save_path: Optional[Path] = None) -> plt.Figure:
        """Plot risk band distribution."""
        fig, ax = plt.subplots(figsize=(10, 6))

        bands = risk_bands["bands"]
        names = [b["name"] for b in bands]
        counts = [b["count"] for b in bands]
        colors = [b["color"] for b in bands]

        bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=1.5)

        # Add count and percentage labels
        total = sum(counts)
        for bar, count in zip(bars, counts):
            pct = count / total * 100
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01,
                    f'{count}\n({pct:.1f}%)', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Number of Applicants')
        ax.set_title('Risk Band Distribution')
        ax.grid(True, alpha=0.3, axis='y')

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved risk distribution plot to {save_path}")

        return fig

    def generate_report(self, results: Dict, output_dir: Path) -> Path:
        """Generate comprehensive evaluation report."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics as JSON
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            # Convert numpy types to native Python
            def convert(obj):
                if isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj

            json.dump(convert(results), f, indent=2)

        # Generate plots
        self.plot_roc_curve(results["roc_curve"], output_dir / "roc_curve.png")
        self.plot_pr_curve(results["pr_curve"], output_dir / "pr_curve.png")
        self.plot_calibration(results["calibration"], output_dir / "calibration_curve.png")
        self.plot_feature_importance(results["feature_importance"], save_path=output_dir / "feature_importance.png")
        self.plot_confusion_matrix(results["confusion_matrix"], output_dir / "confusion_matrix.png")
        self.plot_risk_distribution(results["risk_bands"], output_dir / "risk_distribution.png")

        # Generate markdown report
        report_path = output_dir / "evaluation_report.md"
        self._generate_markdown_report(results, report_path)

        logger.info(f"Generated evaluation report in {output_dir}")
        return report_path

    def _generate_markdown_report(self, results: Dict, report_path: Path) -> None:
        """Generate markdown evaluation report."""
        metrics = results["metrics"]
        risk_dist = results["risk_bands"]["distribution"]

        with open(report_path, 'w') as f:
            f.write("# Model Evaluation Report\n\n")

            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in metrics.items():
                f.write(f"| {metric} | {value:.4f} |\n")

            f.write("\n## Confusion Matrix\n\n")
            cm = results["confusion_matrix"]
            f.write("| | Predicted Good | Predicted Bad |\n")
            f.write("|---|---|---|\n")
            f.write(f"| Actual Good | {cm[0][0]} | {cm[0][1]} |\n")
            f.write(f"| Actual Bad | {cm[1][0]} | {cm[1][1]} |\n")

            f.write("\n## Risk Band Distribution\n\n")
            f.write("| Risk Band | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for band, pct in risk_dist.items():
                f.write(f"| {band} | - | {pct:.1f}% |\n")

            f.write("\n## Top 10 Features\n\n")
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for i, (feat, imp) in enumerate(list(results["feature_importance"].items())[:10]):
                f.write(f"| {i+1} | {feat} | {imp:.4f} |\n")

            f.write("\n## ROC Curve\n\n")
            f.write(f"![ROC Curve](roc_curve.png)\n\n")

            f.write("## Precision-Recall Curve\n\n")
            f.write(f"![PR Curve](pr_curve.png)\n\n")

            f.write("## Calibration Curve\n\n")
            f.write(f"![Calibration](calibration_curve.png)\n\n")

            f.write("## Feature Importance\n\n")
            f.write(f"![Feature Importance](feature_importance.png)\n\n")

            f.write("## Risk Distribution\n\n")
            f.write(f"![Risk Distribution](risk_distribution.png)\n")


def evaluate_model_comprehensive(model: Any, preprocessor: Any, feature_names: List[str],
                                  X_test: pd.DataFrame, y_test: pd.Series,
                                  threshold: float, output_dir: Path) -> Dict:
    """
    Convenience function for comprehensive evaluation.

    Args:
        model: Fitted model.
        preprocessor: Fitted preprocessor.
        feature_names: List of feature names.
        X_test: Test features.
        y_test: Test target.
        threshold: Decision threshold.
        output_dir: Directory to save outputs.

    Returns:
        Evaluation results dictionary.
    """
    evaluator = ModelEvaluator(model, preprocessor, feature_names)
    results = evaluator.evaluate(X_test, y_test, threshold)
    evaluator.generate_report(results, output_dir)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Evaluation module ready")