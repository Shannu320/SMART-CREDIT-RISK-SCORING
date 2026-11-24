"""
Tests for data ingestion module.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os

from credit_risk.data.ingestion import (
    load_raw_data, decode_categorical, validate_data, ingest_pipeline
)
from credit_risk.config import config


class TestIngestion:
    """Test data ingestion functionality."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        data = """A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1
A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2
A14 12 A34 A46 2096 A61 A74 2 A93 A101 3 A121 49 A143 A152 1 A172 2 A191 A201 1"""
        return data

    @pytest.fixture
    def temp_raw_file(self, sample_raw_data):
        """Create temporary raw data file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.data', delete=False) as f:
            f.write(sample_raw_data)
            temp_path = f.name
        yield Path(temp_path)
        os.unlink(temp_path)

    def test_load_raw_data(self, temp_raw_file):
        """Test loading raw data."""
        df = load_raw_data(temp_raw_file)
        assert len(df) == 3
        assert len(df.columns) == 21  # 20 features + target
        # Check that we have the expected columns (order may differ)
        expected_cols = set(config.features.numeric_features +
                           config.features.categorical_features + ['target'])
        assert set(df.columns) == expected_cols

    def test_decode_categorical(self, temp_raw_file):
        """Test categorical decoding."""
        df = load_raw_data(temp_raw_file)
        df = decode_categorical(df)

        # Check target decoding
        assert 'default' in df.columns
        assert 'target' not in df.columns
        assert set(df['default'].unique()).issubset({0, 1})

        # Check categorical decoding
        assert df['checking_account'].iloc[0] == '< 0 DM'
        assert df['credit_history'].iloc[0] == 'critical account/other credits existing'

    def test_validate_data(self, temp_raw_file):
        """Test data validation."""
        df = load_raw_data(temp_raw_file)
        df = decode_categorical(df)
        df, report = validate_data(df)

        assert 'n_rows' in report
        assert 'missing_values' in report
        assert 'duplicate_rows' in report
        assert report['n_rows'] == 3

    def test_ingest_pipeline(self, temp_raw_file):
        """Test full ingestion pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            processed_path = Path(tmpdir) / "processed.parquet"
            df, report = ingest_pipeline(temp_raw_file, processed_path)

            assert len(df) == 3
            assert 'default' in df.columns
            assert processed_path.exists()

            # Check report structure
            assert 'n_rows' in report
            assert 'target_distribution' in report


class TestConfig:
    """Test configuration."""

    def test_config_structure(self):
        """Test config has all required sections."""
        assert hasattr(config, 'data')
        assert hasattr(config, 'features')
        assert hasattr(config, 'model')
        assert hasattr(config, 'evaluation')
        assert hasattr(config, 'api')

    def test_default_values(self):
        """Test default config values."""
        assert config.data.test_size == 0.2
        assert config.data.random_state == 42
        assert config.model.cv_folds == 5
        assert config.evaluation.threshold_optimization == 'f1'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])