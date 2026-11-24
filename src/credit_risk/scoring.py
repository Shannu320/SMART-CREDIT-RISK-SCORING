"""
Credit risk scoring module.
Provides applicant-level risk scores, risk bands, and explanations.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import pandas as pd
import numpy as np
import joblib
import logging
from dataclasses import dataclass, asdict
from datetime import datetime

from credit_risk.config import config
from credit_risk.data.preprocessing import create_full_pipeline, load_preprocessor
from credit_risk.models.training import load_model

logger = logging.getLogger(__name__)


@dataclass
class RiskScore:
    """Individual applicant risk score result."""
    applicant_id: str
    default_probability: float
    risk_band: str
    risk_score: int  # 0-1000 scale
    threshold_used: float
    decision: str  # "Approve", "Review", "Decline"
    key_factors: List[Dict[str, Any]]
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BatchScoreResult:
    """Batch scoring result summary."""
    total_applicants: int
    approved: int
    review: int
    declined: int
    risk_distribution: Dict[str, int]
    average_default_prob: float
    scores: List[RiskScore]


class CreditRiskScorer:
    """Production-ready credit risk scoring engine."""

    def __init__(self, model_path: Optional[Path] = None,
                 preprocessor_path: Optional[Path] = None,
                 threshold: float = 0.5):
        """
        Initialize scorer with model and preprocessor.

        Args:
            model_path: Path to saved model.
            preprocessor_path: Path to saved preprocessor.
            threshold: Decision threshold for default probability.
        """
        self.model_path = model_path or config.api.model_path
        self.preprocessor_path = preprocessor_path or config.api.preprocessor_path
        self.threshold = threshold
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.model_metadata = {}
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model and preprocessor from disk."""
        logger.info("Loading model and preprocessor...")

        # Load model
        model_data = joblib.load(self.model_path)
        if isinstance(model_data, dict) and "model" in model_data:
            self.model = model_data["model"]
            self.model_metadata = model_data.get("metadata", {})
        else:
            self.model = model_data

        # Load preprocessor
        self.preprocessor = joblib.load(self.preprocessor_path)

        # Get feature names from preprocessor
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            self.feature_names = self.preprocessor.get_feature_names_out().tolist()
        else:
            self.feature_names = []

        logger.info(f"Loaded model: {type(self.model).__name__}")
        logger.info(f"Model metadata: {self.model_metadata}")
        logger.info(f"Number of features: {len(self.feature_names)}")

    def score_applicant(self, applicant_data: Dict,
                        applicant_id: str = None) -> RiskScore:
        """
        Score a single applicant.

        Args:
            applicant_data: Dictionary with applicant features.
            applicant_id: Optional applicant identifier.

        Returns:
            RiskScore object.
        """
        # Convert to DataFrame
        df = pd.DataFrame([applicant_data])

        # Generate ID if not provided
        if applicant_id is None:
            applicant_id = f"APP_{np.random.randint(100000, 999999)}"

        # Preprocess
        X = self._preprocess(df)

        # Predict probability
        default_prob = float(self.model.predict_proba(X)[:, 1][0])

        # Determine risk band
        risk_band = self._get_risk_band(default_prob)

        # Convert to 0-1000 score (higher = better)
        risk_score = int((1 - default_prob) * 1000)

        # Make decision
        decision = self._make_decision(default_prob, risk_band)

        # Get key factors (top contributing features)
        key_factors = self._get_key_factors(applicant_data, X, default_prob)

        return RiskScore(
            applicant_id=applicant_id,
            default_probability=default_prob,
            risk_band=risk_band,
            risk_score=risk_score,
            threshold_used=self.threshold,
            decision=decision,
            key_factors=key_factors
        )

    def score_batch(self, applicants_df: pd.DataFrame,
                    id_column: str = "applicant_id") -> BatchScoreResult:
        """
        Score a batch of applicants.

        Args:
            applicants_df: DataFrame with applicant data.
            id_column: Column name for applicant IDs.

        Returns:
            BatchScoreResult with summary and individual scores.
        """
        # Get applicant IDs
        if id_column in applicants_df.columns:
            applicant_ids = applicants_df[id_column].astype(str).tolist()
        else:
            applicant_ids = [f"APP_{i:06d}" for i in range(len(applicants_df))]

        # Preprocess
        X = self._preprocess(applicants_df)

        # Predict probabilities
        default_probs = self.model.predict_proba(X)[:, 1]

        # Score each applicant
        scores = []
        risk_dist = {band["name"]: 0 for band in config.evaluation.risk_bands}
        decisions = {"Approve": 0, "Review": 0, "Decline": 0}

        for i, (app_id, prob) in enumerate(zip(applicant_ids, default_probs)):
            risk_band = self._get_risk_band(prob)
            risk_score = int((1 - prob) * 1000)
            decision = self._make_decision(prob, risk_band)

            # Get key factors for this applicant
            key_factors = self._get_key_factors(
                applicants_df.iloc[i].to_dict(), X[i:i+1], prob
            )

            score = RiskScore(
                applicant_id=app_id,
                default_probability=float(prob),
                risk_band=risk_band,
                risk_score=risk_score,
                threshold_used=self.threshold,
                decision=decision,
                key_factors=key_factors
            )
            scores.append(score)

            risk_dist[risk_band] += 1
            decisions[decision] += 1

        return BatchScoreResult(
            total_applicants=len(scores),
            approved=decisions["Approve"],
            review=decisions["Review"],
            declined=decisions["Decline"],
            risk_distribution=risk_dist,
            average_default_prob=float(np.mean(default_probs)),
            scores=scores
        )

    def _preprocess(self, df: pd.DataFrame) -> np.ndarray:
        """Apply preprocessing pipeline."""
        # Ensure all required columns exist
        for col in config.features.numeric_features + config.features.categorical_features:
            if col not in df.columns:
                df[col] = np.nan

        # Apply preprocessor
        X = self.preprocessor.transform(df)
        return X

    def _get_risk_band(self, default_prob: float) -> str:
        """Map default probability to risk band."""
        for band in config.evaluation.risk_bands:
            if band["min_score"] <= default_prob <= band["max_score"]:
                return band["name"]
        return config.evaluation.risk_bands[-1]["name"]

    def _make_decision(self, default_prob: float, risk_band: str) -> str:
        """Make lending decision based on risk."""
        if default_prob < self.threshold:
            if risk_band in ["Very Low", "Low"]:
                return "Approve"
            else:
                return "Review"
        else:
            return "Decline"

    def _get_key_factors(self, raw_data: Dict, X: np.ndarray,
                         default_prob: float) -> List[Dict[str, Any]]:
        """Identify key factors driving the risk score."""
        factors = []

        # Risk-increasing factors
        risk_factors = []

        # Check high-impact features
        if raw_data.get("credit_score", 650) < 600:
            risk_factors.append({
                "factor": "Low Credit Score",
                "value": raw_data.get("credit_score"),
                "impact": "high",
                "direction": "increases_risk"
            })

        if raw_data.get("debt_to_income_ratio", 0) > 0.5:
            risk_factors.append({
                "factor": "High Debt-to-Income Ratio",
                "value": raw_data.get("debt_to_income_ratio"),
                "impact": "high",
                "direction": "increases_risk"
            })

        if raw_data.get("past_defaults", 0) > 0:
            risk_factors.append({
                "factor": "Past Defaults",
                "value": raw_data.get("past_defaults"),
                "impact": "high",
                "direction": "increases_risk"
            })

        if raw_data.get("employment_type") == "Unemployed":
            risk_factors.append({
                "factor": "Unemployed",
                "value": "Unemployed",
                "impact": "medium",
                "direction": "increases_risk"
            })

        # Protective factors
        protective_factors = []

        if raw_data.get("credit_score", 650) > 750:
            protective_factors.append({
                "factor": "Excellent Credit Score",
                "value": raw_data.get("credit_score"),
                "impact": "high",
                "direction": "decreases_risk"
            })

        if raw_data.get("on_time_payment_pct", 0) > 0.95:
            protective_factors.append({
                "factor": "Strong Payment History",
                "value": raw_data.get("on_time_payment_pct"),
                "impact": "high",
                "direction": "decreases_risk"
            })

        if raw_data.get("employment_tenure_years", 0) > 5:
            protective_factors.append({
                "factor": "Stable Employment",
                "value": raw_data.get("employment_tenure_years"),
                "impact": "medium",
                "direction": "decreases_risk"
            })

        # Combine and limit to top 5
        all_factors = risk_factors + protective_factors
        all_factors.sort(key=lambda x: 0 if x["impact"] == "high" else 1)
        return all_factors[:5]

    def explain_score(self, risk_score: RiskScore) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Credit Risk Assessment for {risk_score.applicant_id}",
            "=" * 50,
            f"Default Probability: {risk_score.default_probability:.1%}",
            f"Risk Score: {risk_score.risk_score}/1000",
            f"Risk Band: {risk_score.risk_band}",
            f"Decision: {risk_score.decision}",
            "",
            "Key Factors:"
        ]

        for factor in risk_score.key_factors:
            direction = "↑" if factor["direction"] == "increases_risk" else "↓"
            lines.append(f"  {direction} {factor['factor']}: {factor['value']} ({factor['impact']} impact)")

        return "\n".join(lines)


def create_scorer_from_training(model: Any, preprocessor: Any,
                                threshold: float,
                                output_dir: Path) -> CreditRiskScorer:
    """
    Create scorer from training artifacts and save.

    Args:
        model: Trained model.
        preprocessor: Fitted preprocessor.
        threshold: Optimal threshold.
        output_dir: Directory to save artifacts.

    Returns:
        Configured CreditRiskScorer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save model with metadata
    metadata = {
        "threshold": threshold,
        "model_type": type(model).__name__,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "feature_names": preprocessor.get_feature_names_out().tolist() if hasattr(preprocessor, 'get_feature_names_out') else []
    }

    model_path = output_dir / "best_model.pkl"
    preprocessor_path = output_dir / "preprocessor.pkl"

    joblib.dump({"model": model, "metadata": metadata}, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    logger.info(f"Saved scorer artifacts to {output_dir}")

    return CreditRiskScorer(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        threshold=threshold
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    from credit_risk.data.ingestion import ingest_pipeline
    from credit_risk.data.preprocessing import preprocess_pipeline
    from credit_risk.models.training import train_all_models, select_best_model

    df, _ = ingest_pipeline()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df)

    # Quick train
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    # Create scorer
    scorer = create_scorer_from_training(model, preprocessor, 0.5, Path("models/test"))

    # Score sample
    sample = X_test.iloc[0].to_dict()
    result = scorer.score_applicant(sample, "TEST_001")
    print(scorer.explain_score(result))