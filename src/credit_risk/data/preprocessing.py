"""
Preprocessing pipeline for credit risk data.
Handles feature engineering, encoding, scaling, and train/test splitting.
"""
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.config import config

logger = logging.getLogger(__name__)


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Handle outliers using IQR method. Works with column indices from ColumnTransformer."""

    def __init__(self, method: str = "iqr", threshold: float = 1.5,
                 columns: Optional[List[str]] = None):
        self.method = method
        self.threshold = threshold
        self.columns = columns
        self.bounds_ = {}

    def fit(self, X, y=None) -> "OutlierHandler":
        # ColumnTransformer passes numpy arrays with specific columns
        # X is a 2D array of the selected columns
        X_df = pd.DataFrame(X, columns=self.columns if self.columns else [f"col_{i}" for i in range(X.shape[1])])

        for col in X_df.columns:
            Q1 = X_df[col].quantile(0.25)
            Q3 = X_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - self.threshold * IQR
            upper = Q3 + self.threshold * IQR
            self.bounds_[col] = (lower, upper)

        return self

    def transform(self, X):
        # Check if input is DataFrame
        is_dataframe = hasattr(X, 'columns')
        if is_dataframe:
            X_df = X.copy()
        else:
            X_df = pd.DataFrame(X, columns=self.columns if self.columns else [f"col_{i}" for i in range(X.shape[1])])

        for col, (lower, upper) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower, upper)

        if is_dataframe:
            return X_df
        return X_df.values

    def get_feature_names_out(self, input_features=None):
        """Return output feature names (same as input)."""
        if input_features is None:
            return np.array(self.columns) if self.columns else np.array([f"col_{i}" for i in range(len(self.bounds_))])
        return np.array(input_features)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Create additional features for credit risk modeling."""

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Credit amount per month
        if "credit_amount" in X.columns and "duration_months" in X.columns:
            X["credit_amount_per_month"] = X["credit_amount"] / X["duration_months"].clip(lower=1)

        # Age groups
        if "age" in X.columns:
            X["age_group"] = pd.cut(X["age"], bins=[0, 25, 35, 45, 55, 65, 100],
                                     labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])

        # Credit amount bins
        if "credit_amount" in X.columns:
            X["credit_amount_bin"] = pd.qcut(X["credit_amount"], q=5,
                                              labels=["Very Low", "Low", "Medium", "High", "Very High"],
                                              duplicates="drop")

        # Duration bins
        if "duration_months" in X.columns:
            X["duration_bin"] = pd.cut(X["duration_months"], bins=[0, 12, 24, 36, 48, 72, 100],
                                        labels=["0-12", "13-24", "25-36", "37-48", "49-72", "72+"])

        # Debt burden proxy: installment rate * duration
        if "installment_rate_pct" in X.columns and "duration_months" in X.columns:
            X["total_installment_burden"] = X["installment_rate_pct"] * X["duration_months"] / 100

        # Employment stability
        if "employment_duration" in X.columns:
            emp_map = {"unemployed": 0, "< 1 year": 1, "1 <= ... < 4 years": 2,
                       "4 <= ... < 7 years": 3, ">= 7 years": 4}
            X["employment_stability"] = X["employment_duration"].map(emp_map).fillna(0)

        # Savings level
        if "savings" in X.columns:
            sav_map = {"< 100 DM": 1, "100 <= ... < 500 DM": 2, "500 <= ... < 1000 DM": 3,
                       ">= 1000 DM": 4, "unknown/no savings account": 0}
            X["savings_level"] = X["savings"].map(sav_map).fillna(0)

        # Checking account status
        if "checking_account" in X.columns:
            chk_map = {"< 0 DM": 0, "0 <= ... < 200 DM": 1, ">= 200 DM": 2, "no checking account": -1}
            X["checking_status"] = X["checking_account"].map(chk_map).fillna(-1)

        return X


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Create the preprocessing pipeline for original features only.
    Feature engineering is handled separately by FeatureEngineer.

    Returns:
        ColumnTransformer for numeric and categorical features.
    """
    numeric_features = config.features.numeric_features.copy()
    categorical_features = config.features.categorical_features.copy()

    # Numeric transformer
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_handler", OutlierHandler(
            method=config.features.outlier_method,
            threshold=config.features.outlier_threshold,
            columns=numeric_features
        )),
        ("scaler", StandardScaler())
    ])

    # Categorical transformer
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    return preprocessor


def create_full_pipeline(model: Any = None) -> Pipeline:
    """
    Create full ML pipeline with preprocessing and model.

    Args:
        model: Scikit-learn compatible estimator.

    Returns:
        Complete pipeline.
    """
    preprocessor = create_preprocessing_pipeline()

    steps = [
        ("feature_engineer", FeatureEngineer()),
        ("preprocessor", preprocessor)
    ]

    if model is not None:
        steps.append(("classifier", model))

    return Pipeline(steps=steps)


def split_data(df: pd.DataFrame,
               target_col: str = "default",
               test_size: float = 0.2,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets.

    Args:
        df: Input DataFrame.
        target_col: Target column name.
        test_size: Proportion of test set.
        random_state: Random seed.
        stratify: Whether to stratify by target.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_param = y if stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor: ColumnTransformer, filepath: Optional[Path] = None) -> None:
    """Save fitted preprocessor to disk."""
    if filepath is None:
        filepath = config.api.preprocessor_path

    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, filepath)
    logger.info(f"Saved preprocessor to {filepath}")


def load_preprocessor(filepath: Optional[Path] = None) -> ColumnTransformer:
    """Load fitted preprocessor from disk."""
    if filepath is None:
        filepath = config.api.preprocessor_path

    logger.info(f"Loading preprocessor from {filepath}")
    return joblib.load(filepath)


def preprocess_pipeline(df: pd.DataFrame,
                        target_col: str = "default") -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Complete preprocessing pipeline: feature engineering, splitting, and transformation.

    Args:
        df: Input DataFrame.
        target_col: Target column name.

    Returns:
        Tuple of (X_train_transformed, X_test_transformed, y_train, y_test, fitted_preprocessor).
    """
    # Split first (to prevent data leakage)
    X_train, X_test, y_train, y_test = split_data(
        df, target_col=target_col,
        test_size=config.data.test_size,
        random_state=config.data.random_state,
        stratify=config.data.stratify
    )

    # Apply feature engineering to both sets
    feature_engineer = FeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_test = feature_engineer.transform(X_test)

    # Create and fit preprocessor on training data only
    preprocessor = create_preprocessing_pipeline()
    preprocessor.fit(X_train, y_train)

    # Transform both sets
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    # Get feature names
    feature_names = preprocessor.get_feature_names_out()

    # Convert to DataFrames for easier handling
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

    logger.info(f"Transformed train shape: {X_train_df.shape}, test shape: {X_test_df.shape}")
    logger.info(f"Number of features: {len(feature_names)}")

    return X_train_df, X_test_df, y_train, y_test, preprocessor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test with sample data
    from credit_risk.data.ingestion import ingest_pipeline
    df, _ = ingest_pipeline()

    X_train, X_test, y_train, y_test, preprocessor = preprocess_pipeline(df)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Features: {list(X_train.columns)[:10]}...")