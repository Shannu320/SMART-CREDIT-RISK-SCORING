"""
Configuration management for the credit risk pipeline.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class DataConfig:
    """Data ingestion and processing configuration."""
    raw_data_path: Path = Path("data/raw/german_credit.data")
    processed_data_path: Path = Path("data/processed/credit_risk_processed.parquet")
    train_data_path: Path = Path("data/processed/train.parquet")
    test_data_path: Path = Path("data/processed/test.parquet")
    target_column: str = "default"
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    numeric_features: List[str] = field(default_factory=lambda: [
        "duration_months", "credit_amount", "installment_rate_pct",
        "residence_years", "age", "existing_credits", "dependents"
    ])
    categorical_features: List[str] = field(default_factory=lambda: [
        "checking_account", "credit_history", "purpose", "savings",
        "employment_duration", "personal_status", "other_debtors",
        "property", "installment_plans", "housing", "job", "telephone",
        "foreign_worker"
    ])
    drop_features: List[str] = field(default_factory=list)
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 1.5


@dataclass
class ModelConfig:
    """Model training configuration."""
    models: Dict[str, Dict] = field(default_factory=lambda: {
        "logistic_regression": {
            "class": "sklearn.linear_model.LogisticRegression",
            "params": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "solver": "lbfgs"
            },
            "search_space": {
                "C": [0.01, 0.1, 1.0, 10.0, 100.0],
                "penalty": ["l2"]
            }
        },
        "decision_tree": {
            "class": "sklearn.tree.DecisionTreeClassifier",
            "params": {
                "class_weight": "balanced",
                "random_state": 42
            },
            "search_space": {
                "max_depth": [3, 5, 7, 10, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 4, 8]
            }
        },
        "random_forest": {
            "class": "sklearn.ensemble.RandomForestClassifier",
            "params": {
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1
            },
            "search_space": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"]
            }
        },
        "gradient_boosting": {
            "class": "sklearn.ensemble.GradientBoostingClassifier",
            "params": {
                "random_state": 42
            },
            "search_space": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "subsample": [0.8, 1.0]
            }
        }
    })
    cv_folds: int = 5
    scoring: str = "roc_auc"
    n_iter_search: int = 20
    refit: bool = True


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""
    metrics: List[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"
    ])
    threshold_optimization: str = "f1"  # f1, youden, precision_at_recall, recall_at_precision
    target_recall: float = 0.8
    target_precision: float = 0.7
    risk_bands: List[Dict] = field(default_factory=lambda: [
        {"name": "Very Low", "min_score": 0.0, "max_score": 0.15, "color": "#2E7D32"},
        {"name": "Low", "min_score": 0.15, "max_score": 0.30, "color": "#66BB6A"},
        {"name": "Medium", "min_score": 0.30, "max_score": 0.50, "color": "#FFCA28"},
        {"name": "High", "min_score": 0.50, "max_score": 0.70, "color": "#EF6C00"},
        {"name": "Very High", "min_score": 0.70, "max_score": 1.0, "color": "#C62828"}
    ])


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    model_path: Path = Path("models/best_model.pkl")
    preprocessor_path: Path = Path("models/preprocessor.pkl")
    log_level: str = "INFO"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    project_root: Path = Path(__file__).parent.parent.parent

    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            "data": self.data.__dict__,
            "features": self.features.__dict__,
            "model": self.model.__dict__,
            "evaluation": self.evaluation.__dict__,
            "api": self.api.__dict__
        }

    @classmethod
    def from_json(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        with open(path) as f:
            data = json.load(f)
        config = cls()
        # Update with loaded values (simplified - in production use pydantic)
        return config


# Global config instance
config = Config()