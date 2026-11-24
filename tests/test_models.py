"""
Tests for model training and evaluation modules.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from credit_risk.models.training import (
    instantiate_model, tune_hyperparameters, evaluate_model,
    cross_validate_model, find_optimal_threshold, train_all_models,
    select_best_model, save_model, load_model
)
from credit_risk.models.evaluation import ModelEvaluator
from credit_risk.scoring import CreditRiskScorer, create_scorer_from_training
from credit_risk.data.preprocessing import create_preprocessing_pipeline
from credit_risk.config import config


# Module-level fixtures
@pytest.fixture
def sample_train_data():
    """Create sample training data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randint(0, 5, n),
        'feature_4': np.random.randn(n)  # numeric only
    })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def sample_val_data():
    """Create sample validation data."""
    np.random.seed(123)
    n = 100
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randint(0, 5, n),
        'feature_4': np.random.randn(n)  # numeric only
    })
    y = pd.Series(np.random.randint(0, 2, n))
    return X, y


@pytest.fixture
def trained_scorer(sample_train_data):
    """Create a trained scorer for testing."""
    X, y = sample_train_data

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer

    # Create a simple preprocessor for test data
    numeric_features = X.columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )
    preprocessor.fit(X)

    # Train model
    X_transformed = preprocessor.transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_transformed, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        preprocessor_path = Path(tmpdir) / "preprocessor.pkl"

        import joblib
        joblib.dump({"model": model, "metadata": {"threshold": 0.5}}, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        scorer = CreditRiskScorer(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            threshold=0.5
        )
        return scorer


class TestModels:
    """Test model training functionality."""

    def test_instantiate_model(self):
        """Test model instantiation from config."""
        model_config = {
            "class": "sklearn.linear_model.LogisticRegression",
            "params": {"max_iter": 1000, "random_state": 42}
        }
        model = instantiate_model(model_config)
        assert model is not None
        assert hasattr(model, 'fit')

    def test_evaluate_model(self, sample_train_data, sample_val_data):
        """Test model evaluation."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_val, y_val, threshold=0.5)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'roc_auc' in metrics
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, float))

    def test_cross_validate_model(self, sample_train_data):
        """Test cross-validation."""
        X, y = sample_train_data

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)

        results = cross_validate_model(model, X, y, cv=3, scoring=['accuracy', 'roc_auc'])

        assert 'accuracy' in results
        assert 'roc_auc' in results
        assert len(results['accuracy']) == 3

    def test_find_optimal_threshold(self, sample_train_data, sample_val_data):
        """Test threshold optimization."""
        X_train, y_train = sample_train_data
        X_val, y_val = sample_val_data

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        threshold, metrics = find_optimal_threshold(model, X_val, y_val, method='f1')

        assert 0 <= threshold <= 1
        assert 'threshold' in metrics
        assert 'f1' in metrics

    def test_save_load_model(self, sample_train_data):
        """Test model serialization."""
        X, y = sample_train_data

        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.pkl"
            save_model(model, path, metadata={"test": "metadata"})

            loaded_model, metadata = load_model(path)

            assert loaded_model is not None
            assert metadata.get("test") == "metadata"


class TestScorer:
    """Test scoring functionality."""

    def test_score_applicant(self, trained_scorer):
        """Test single applicant scoring."""
        applicant_data = {
            'feature_1': 0.5,
            'feature_2': -0.3,
            'feature_3': 2,
            'feature_4': 0.5
        }

        result = trained_scorer.score_applicant(applicant_data, "TEST_001")

        assert isinstance(result.default_probability, float)
        assert 0 <= result.default_probability <= 1
        assert result.risk_band in ["Very Low", "Low", "Medium", "High", "Very High"]
        assert 0 <= result.risk_score <= 1000
        assert result.decision in ["Approve", "Review", "Decline"]
        assert isinstance(result.key_factors, list)

    def test_score_batch(self, trained_scorer):
        """Test batch scoring."""
        df = pd.DataFrame({
            'feature_1': [0.5, -0.2, 1.0],
            'feature_2': [-0.3, 0.1, -0.5],
            'feature_3': [2, 3, 1],
            'feature_4': [0.5, -0.5, 1.2]
        })

        result = trained_scorer.score_batch(df)

        assert result.total_applicants == 3
        assert result.approved + result.review + result.declined == 3
        assert len(result.scores) == 3
        assert all(isinstance(s.default_probability, float) for s in result.scores)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])