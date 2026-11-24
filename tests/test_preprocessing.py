"""
Tests for preprocessing module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from credit_risk.data.preprocessing import (
    create_preprocessing_pipeline, create_full_pipeline,
    FeatureEngineer, OutlierHandler, split_data,
    preprocess_pipeline, save_preprocessor, load_preprocessor
)
from credit_risk.config import config


class TestPreprocessing:
    """Test preprocessing functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data matching German Credit format."""
        np.random.seed(42)
        n = 100
        return pd.DataFrame({
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
            'default': np.random.randint(0, 2, n)
        })

    def test_feature_engineer(self, sample_data):
        """Test feature engineering transformer."""
        engineer = FeatureEngineer()
        df_transformed = engineer.fit_transform(sample_data)

        # Check new features created
        assert 'credit_amount_per_month' in df_transformed.columns
        assert 'age_group' in df_transformed.columns
        assert 'credit_amount_bin' in df_transformed.columns
        assert 'duration_bin' in df_transformed.columns
        assert 'employment_stability' in df_transformed.columns
        assert 'savings_level' in df_transformed.columns
        assert 'checking_status' in df_transformed.columns

    def test_outlier_handler(self, sample_data):
        """Test outlier handling."""
        handler = OutlierHandler(method='iqr', threshold=1.5,
                                  columns=['credit_amount', 'age'])
        handler.fit(sample_data)
        df_transformed = handler.transform(sample_data)

        # Check clipping
        assert df_transformed['credit_amount'].max() <= handler.bounds_['credit_amount'][1]
        assert df_transformed['credit_amount'].min() >= handler.bounds_['credit_amount'][0]

    def test_create_preprocessing_pipeline(self):
        """Test preprocessing pipeline creation."""
        preprocessor = create_preprocessing_pipeline()
        assert preprocessor is not None
        assert hasattr(preprocessor, 'transformers')

    def test_split_data(self, sample_data):
        """Test train/test split."""
        X_train, X_test, y_train, y_test = split_data(
            sample_data, target_col='default', test_size=0.2, random_state=42
        )

        assert len(X_train) + len(X_test) == len(sample_data)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        assert 'default' not in X_train.columns
        assert 'default' not in X_test.columns

    def test_preprocess_pipeline(self, sample_data):
        """Test full preprocessing pipeline."""
        X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(sample_data)

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == X_test.shape[1]
        assert len(y_train) == X_train.shape[0]
        assert len(y_test) == X_test.shape[0]
        assert preprocessor is not None

    def test_save_load_preprocessor(self, sample_data):
        """Test preprocessor serialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "preprocessor.pkl"

            # Get original train/test split
            from credit_risk.data.preprocessing import split_data
            X_train_orig, X_test_orig, y_train, y_test = split_data(
                sample_data, target_col='default', test_size=0.2, random_state=42
            )

            # Apply feature engineering
            from credit_risk.data.preprocessing import FeatureEngineer
            feature_engineer = FeatureEngineer()
            X_train_eng = feature_engineer.fit_transform(X_train_orig)
            X_test_eng = feature_engineer.transform(X_test_orig)

            # Fit preprocessor
            preprocessor = create_preprocessing_pipeline()
            preprocessor.fit(X_train_eng, y_train)
            save_preprocessor(preprocessor, path)

            assert path.exists()

            loaded = load_preprocessor(path)
            assert loaded is not None

            # Test loaded preprocessor works on feature-engineered data
            X_test_transformed = loaded.transform(X_test_eng)
            assert X_test_transformed.shape[0] == X_test_eng.shape[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])