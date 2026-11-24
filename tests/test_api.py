"""
Tests for FastAPI endpoints.
"""
import pytest
from httpx import ASGITransport, AsyncClient
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import joblib

from credit_risk.api import app
from credit_risk.data.preprocessing import create_preprocessing_pipeline
from sklearn.linear_model import LogisticRegression


@pytest.fixture
def mock_scorer(monkeypatch):
    """Create and inject mock scorer using actual preprocessing pipeline."""
    np.random.seed(42)
    n = 100

    # Create data matching the actual feature set used by the pipeline
    X = pd.DataFrame({
        'checking_account': np.random.choice(['< 0 DM', '0 <= ... < 200 DM', '>= 200 DM', 'no checking account'], n),
        'duration_months': np.random.randint(6, 72, n),
        'credit_history': np.random.choice(['no credits/all paid back', 'all credits at this bank paid back duly',
                                             'existing credits paid back duly till now', 'delay in paying off in the past',
                                             'critical account/other credits existing'], n),
        'purpose': np.random.choice(['car (new)', 'car (used)', 'furniture/equipment', 'radio/television',
                                      'domestic appliances', 'repairs', 'education', 'vacation',
                                      'retraining', 'business', 'others'], n),
        'credit_amount': np.random.randint(500, 20000, n),
        'savings': np.random.choice(['< 100 DM', '100 <= ... < 500 DM', '500 <= ... < 1000 DM',
                                      '>= 1000 DM', 'unknown/no savings account'], n),
        'employment_duration': np.random.choice(['unemployed', '< 1 year', '1 <= ... < 4 years',
                                                  '4 <= ... < 7 years', '>= 7 years'], n),
        'installment_rate_pct': np.random.randint(1, 5, n),
        'personal_status': np.random.choice(['male: divorced/separated', 'female: divorced/separated/married',
                                              'male: single', 'male: married/widowed', 'female: single'], n),
        'other_debtors': np.random.choice(['none', 'co-applicant', 'guarantor'], n),
        'residence_years': np.random.randint(1, 10, n),
        'property': np.random.choice(['real estate', 'building society savings/life insurance',
                                       'car or other', 'unknown/no property'], n),
        'age': np.random.randint(18, 75, n),
        'installment_plans': np.random.choice(['bank', 'stores', 'none'], n),
        'housing': np.random.choice(['rent', 'own', 'for free'], n),
        'existing_credits': np.random.randint(1, 5, n),
        'job': np.random.choice(['unemployed/unskilled non-resident', 'unskilled resident',
                                  'skilled employee/official', 'management/self-employed/highly qualified'], n),
        'dependents': np.random.randint(0, 3, n),
        'telephone': np.random.choice(['none', 'yes'], n),
        'foreign_worker': np.random.choice(['yes', 'no'], n),
    })
    y = pd.Series(np.random.randint(0, 2, n))

    # Use the actual preprocessing pipeline
    from credit_risk.data.preprocessing import preprocess_pipeline, FeatureEngineer
    from credit_risk.data.preprocessing import create_preprocessing_pipeline

    # Split data
    from credit_risk.data.preprocessing import split_data
    X_train, X_test, y_train, y_test = split_data(
        X.assign(default=y), target_col='default', test_size=0.2, random_state=42
    )

    # Feature engineering
    feature_engineer = FeatureEngineer()
    X_train_eng = feature_engineer.fit_transform(X_train)
    X_test_eng = feature_engineer.transform(X_test)

    # Fit preprocessor
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X_train_eng, y_train)
    X_train_transformed = preprocessor.transform(X_train_eng)

    # Train model
    model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    model.fit(X_train_transformed, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.pkl"
        preprocessor_path = Path(tmpdir) / "preprocessor.pkl"

        joblib.dump({"model": model, "metadata": {"threshold": 0.5, "model_type": "LogisticRegression"}}, model_path)
        joblib.dump(preprocessor, preprocessor_path)

        from credit_risk.scoring import CreditRiskScorer
        scorer = CreditRiskScorer(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            threshold=0.5
        )

        # Patch the global scorer
        import credit_risk.api as api_module
        monkeypatch.setattr(api_module, 'scorer', scorer)

        yield scorer


class TestAPI:
    """Test API endpoints using AsyncClient."""

    @pytest.mark.asyncio
    async def test_health_check(self, mock_scorer):
        """Test health endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert data["model_type"] == "LogisticRegression"

    @pytest.mark.asyncio
    async def test_score_single(self, mock_scorer):
        """Test single applicant scoring."""
        payload = {
            "applicant_id": "TEST_001",
            "age": 35,
            "gender": "M",
            "marital_status": "Married",
            "dependents": 2,
            "employment_type": "Full-time",
            "employment_tenure_years": 5.5,
            "annual_income": 75000,
            "credit_score": 720,
            "credit_history_length_years": 10,
            "num_credit_accounts": 5,
            "num_delinquent_accounts": 0,
            "past_defaults": 0,
            "bankruptcy_flag": 0,
            "total_debt": 15000,
            "monthly_debt_payment": 500,
            "debt_to_income_ratio": 0.08,
            "loan_amount": 25000,
            "loan_term_months": 60,
            "loan_purpose": "Debt Consolidation",
            "interest_rate": 12.5,
            "num_late_payments_30d": 0,
            "num_late_payments_60d": 0,
            "num_late_payments_90d": 0,
            "on_time_payment_pct": 0.98
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/score", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert "applicant_id" in data
            assert "default_probability" in data
            assert "risk_band" in data
            assert "risk_score" in data
            assert "decision" in data
            assert 0 <= data["default_probability"] <= 1
            assert 0 <= data["risk_score"] <= 1000
            assert data["decision"] in ["Approve", "Review", "Decline"]

    @pytest.mark.asyncio
    async def test_score_batch(self, mock_scorer):
        """Test batch scoring."""
        payload = {
            "applicants": [
                {
                    "applicant_id": "TEST_001",
                    "age": 35,
                    "gender": "M",
                    "marital_status": "Married",
                    "dependents": 2,
                    "employment_type": "Full-time",
                    "employment_tenure_years": 5.5,
                    "annual_income": 75000,
                    "credit_score": 720,
                    "credit_history_length_years": 10,
                    "num_credit_accounts": 5,
                    "num_delinquent_accounts": 0,
                    "past_defaults": 0,
                    "bankruptcy_flag": 0,
                    "total_debt": 15000,
                    "monthly_debt_payment": 500,
                    "debt_to_income_ratio": 0.08,
                    "loan_amount": 25000,
                    "loan_term_months": 60,
                    "loan_purpose": "Debt Consolidation",
                    "interest_rate": 12.5,
                    "num_late_payments_30d": 0,
                    "num_late_payments_60d": 0,
                    "num_late_payments_90d": 0,
                    "on_time_payment_pct": 0.98
                },
                {
                    "applicant_id": "TEST_002",
                    "age": 45,
                    "gender": "F",
                    "marital_status": "Single",
                    "dependents": 0,
                    "employment_type": "Self-employed",
                    "employment_tenure_years": 3.0,
                    "annual_income": 90000,
                    "credit_score": 680,
                    "credit_history_length_years": 8,
                    "num_credit_accounts": 4,
                    "num_delinquent_accounts": 1,
                    "past_defaults": 0,
                    "bankruptcy_flag": 0,
                    "total_debt": 30000,
                    "monthly_debt_payment": 800,
                    "debt_to_income_ratio": 0.15,
                    "loan_amount": 40000,
                    "loan_term_months": 48,
                    "loan_purpose": "Business",
                    "interest_rate": 15.0,
                    "num_late_payments_30d": 1,
                    "num_late_payments_60d": 0,
                    "num_late_payments_90d": 0,
                    "on_time_payment_pct": 0.92
                }
            ]
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/score/batch", json=payload)
            assert response.status_code == 200
            data = response.json()

            assert data["total_applicants"] == 2
            assert data["approved"] + data["review"] + data["declined"] == 2
            assert len(data["scores"]) == 2
            assert "risk_distribution" in data
            assert "average_default_prob" in data

    @pytest.mark.asyncio
    async def test_model_info(self, mock_scorer):
        """Test model info endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/model/info")
            assert response.status_code == 200
            data = response.json()

            assert "model_type" in data
            assert "threshold" in data
            assert "feature_count" in data
            assert "risk_bands" in data

    @pytest.mark.asyncio
    async def test_model_features(self, mock_scorer):
        """Test model features endpoint."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.get("/model/features")
            assert response.status_code == 200
            data = response.json()

            assert "features" in data
            assert "count" in data

    @pytest.mark.asyncio
    async def test_invalid_input(self, mock_scorer):
        """Test validation error handling."""
        payload = {
            "age": 150,  # Invalid age
            "gender": "M"
        }

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            response = await client.post("/score", json=payload)
            assert response.status_code == 422  # Validation error