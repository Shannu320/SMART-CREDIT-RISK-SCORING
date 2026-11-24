"""
Data ingestion module for German Credit Data.
Handles loading, parsing, and initial validation of raw data.
"""
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np
import logging

from credit_risk.config import config

logger = logging.getLogger(__name__)


# German Credit Data column definitions based on UCI documentation
COLUMN_NAMES = [
    "checking_account",      # A11-A14: Status of existing checking account
    "duration_months",       # A2: Duration in months (numeric)
    "credit_history",        # A30-A34: Credit history
    "purpose",               # A40-A410: Purpose
    "credit_amount",         # A5: Credit amount (numeric)
    "savings",               # A61-A65: Savings account/bonds
    "employment_duration",   # A71-A75: Present employment since
    "installment_rate_pct",  # A8: Installment rate in percentage of disposable income (numeric)
    "personal_status",       # A91-A95: Personal status and sex
    "other_debtors",         # A101-A103: Other debtors / guarantors
    "residence_years",       # A11: Present residence since (numeric)
    "property",              # A121-A124: Property
    "age",                   # A13: Age in years (numeric)
    "installment_plans",     # A141-A143: Other installment plans
    "housing",               # A151-A153: Housing
    "existing_credits",      # A16: Number of existing credits at this bank (numeric)
    "job",                   # A171-A174: Job
    "dependents",            # A18: Number of people being liable to provide maintenance for (numeric)
    "telephone",             # A191-A192: Telephone
    "foreign_worker",        # A201-A202: Foreign worker
    "target"                 # Target: 1 = Good, 2 = Bad
]

# Value mappings for categorical features
CATEGORICAL_MAPPINGS = {
    "checking_account": {
        "A11": "< 0 DM", "A12": "0 <= ... < 200 DM", "A13": ">= 200 DM", "A14": "no checking account"
    },
    "credit_history": {
        "A30": "no credits/all paid back", "A31": "all credits at this bank paid back duly",
        "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past",
        "A34": "critical account/other credits existing"
    },
    "purpose": {
        "A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/television",
        "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation",
        "A48": "retraining", "A49": "business", "A410": "others"
    },
    "savings": {
        "A61": "< 100 DM", "A62": "100 <= ... < 500 DM", "A63": "500 <= ... < 1000 DM",
        "A64": ">= 1000 DM", "A65": "unknown/no savings account"
    },
    "employment_duration": {
        "A71": "unemployed", "A72": "< 1 year", "A73": "1 <= ... < 4 years",
        "A74": "4 <= ... < 7 years", "A75": ">= 7 years"
    },
    "personal_status": {
        "A91": "male: divorced/separated", "A92": "female: divorced/separated/married",
        "A93": "male: single", "A94": "male: married/widowed", "A95": "female: single"
    },
    "other_debtors": {
        "A101": "none", "A102": "co-applicant", "A103": "guarantor"
    },
    "property": {
        "A121": "real estate", "A122": "building society savings/life insurance",
        "A123": "car or other", "A124": "unknown/no property"
    },
    "installment_plans": {
        "A141": "bank", "A142": "stores", "A143": "none"
    },
    "housing": {
        "A151": "rent", "A152": "own", "A153": "for free"
    },
    "job": {
        "A171": "unemployed/unskilled non-resident", "A172": "unskilled resident",
        "A173": "skilled employee/official", "A174": "management/self-employed/highly qualified"
    },
    "telephone": {
        "A191": "none", "A192": "yes"
    },
    "foreign_worker": {
        "A201": "yes", "A202": "no"
    }
}


def load_raw_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load raw German Credit Data from file.

    Args:
        filepath: Path to raw data file. Uses config default if None.

    Returns:
        DataFrame with raw data and proper column names.
    """
    if filepath is None:
        filepath = config.data.raw_data_path

    logger.info(f"Loading raw data from {filepath}")

    # Read space-separated data
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)

    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def decode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Decode categorical codes to human-readable labels.

    Args:
        df: DataFrame with encoded categorical columns.

    Returns:
        DataFrame with decoded categorical values.
    """
    df = df.copy()

    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])

    # Decode target: 1=Good (0), 2=Bad (1) for binary classification
    df["default"] = (df["target"] == 2).astype(int)
    df = df.drop(columns=["target"])

    logger.info("Decoded categorical features and target")
    return df


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Validate data quality and return validation report.

    Args:
        df: DataFrame to validate.

    Returns:
        Tuple of (validated DataFrame, validation report dict).
    """
    report = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_distribution": df["default"].value_counts().to_dict() if "default" in df.columns else {},
        "numeric_stats": {},
        "categorical_stats": {}
    }

    # Check for missing values
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        logger.warning(f"Missing values found in columns: {missing_cols}")

    # Check duplicates
    if report["duplicate_rows"] > 0:
        logger.warning(f"Found {report['duplicate_rows']} duplicate rows")
        df = df.drop_duplicates()

    # Numeric feature statistics
    numeric_cols = config.features.numeric_features
    for col in numeric_cols:
        if col in df.columns:
            report["numeric_stats"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75))
            }

    # Categorical feature statistics
    cat_cols = config.features.categorical_features
    for col in cat_cols:
        if col in df.columns:
            report["categorical_stats"][col] = {
                "n_unique": int(df[col].nunique()),
                "value_counts": df[col].value_counts().head(10).to_dict()
            }

    logger.info("Data validation complete")
    return df, report


def save_processed_data(df: pd.DataFrame, filepath: Optional[Path] = None) -> None:
    """Save processed DataFrame to parquet."""
    if filepath is None:
        filepath = config.data.processed_data_path

    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filepath, index=False)
    logger.info(f"Saved processed data to {filepath}")


def load_processed_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """Load processed DataFrame from parquet."""
    if filepath is None:
        filepath = config.data.processed_data_path

    logger.info(f"Loading processed data from {filepath}")
    return pd.read_parquet(filepath)


def ingest_pipeline(raw_path: Optional[Path] = None,
                    processed_path: Optional[Path] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Complete data ingestion pipeline.

    Args:
        raw_path: Path to raw data file.
        processed_path: Path to save processed data.

    Returns:
        Tuple of (processed DataFrame, validation report).
    """
    # Load raw data
    df = load_raw_data(raw_path)

    # Decode categoricals
    df = decode_categorical(df)

    # Validate
    df, report = validate_data(df)

    # Save processed
    save_processed_data(df, processed_path)

    return df, report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df, report = ingest_pipeline()
    print(f"Processed {len(df)} rows")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Columns: {df.columns.tolist()}")