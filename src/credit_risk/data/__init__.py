"""
Data ingestion and preprocessing module.
"""
from credit_risk.data.ingestion import (
    load_raw_data,
    decode_categorical,
    validate_data,
    ingest_pipeline,
    save_processed_data,
    load_processed_data
)

from credit_risk.data.preprocessing import (
    create_preprocessing_pipeline,
    create_full_pipeline,
    FeatureEngineer,
    OutlierHandler,
    split_data,
    preprocess_pipeline,
    save_preprocessor,
    load_preprocessor
)

__all__ = [
    # Ingestion
    "load_raw_data",
    "decode_categorical",
    "validate_data",
    "ingest_pipeline",
    "save_processed_data",
    "load_processed_data",
    # Preprocessing
    "create_preprocessing_pipeline",
    "create_full_pipeline",
    "FeatureEngineer",
    "OutlierHandler",
    "split_data",
    "preprocess_pipeline",
    "save_preprocessor",
    "load_preprocessor",
]