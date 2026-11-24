# Smart Credit Risk Scoring & Loan Default Analytics

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](Dockerfile)

Production-ready credit risk modeling pipeline for loan default probability prediction, risk segmentation, and decision support.

## Overview

This project implements a complete end-to-end credit risk analytics platform:

- **Data Ingestion**: Load and validate German Credit Data (UCI Statlog)
- **Feature Engineering**: 20+ engineered features from raw attributes
- **Model Training**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting with hyperparameter tuning
- **Model Evaluation**: ROC-AUC, Precision-Recall, Calibration, Confusion Matrix, Risk Bands
- **Risk Scoring**: Applicant-level default probability, risk bands (5 tiers), lending decisions
- **REST API**: FastAPI-based inference endpoint with validation and monitoring
- **Power BI Export**: Pre-computed datasets and DAX measures for dashboarding
- **CI/CD**: GitHub Actions pipeline with testing, training, and deployment

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Data      │───▶│  Preprocessing   │───▶│  Model Training │
│  (German Credit)│    │  & Feature Eng   │    │  & Tuning       │
└─────────────────┘    └──────────────────┘    └────────┬────────┘
                                                        │
                        ┌──────────────────┐            │
                        │   Evaluation     │◀───────────┘
                        │  & Risk Bands    │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌───────────┐
            │  REST API │ │ Power BI  │ │  Batch    │
            │  (FastAPI)│ │  Export   │ │  Scoring  │
            └───────────┘ └───────────┘ └───────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (optional, for containerized deployment)
- 4GB+ RAM for training

### Installation

```bash
# Clone repository
git clone <repository-url>
cd credit-card-risk

# Install dependencies
make install

# Or manually
pip install -r requirements.txt
pre-commit install
```

### Download Data

```bash
make download
# Or manually:
mkdir -p data/raw
curl -s "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data" \
  -o data/raw/german_credit.data
```

### Run Training Pipeline

```bash
# Full pipeline with default settings
make train

# Or with custom output directory
python -m credit_risk.pipeline --output-dir outputs/my_run
```

This will:
1. Load and validate the German Credit Data
2. Engineer 20+ features
3. Train 4 models with hyperparameter tuning
4. Select best model by ROC-AUC
5. Evaluate comprehensively with plots
6. Save model artifacts to `models/`
7. Export Power BI datasets to `outputs/<run>/powerbi/`

### Run API Server

```bash
# Development with auto-reload
make api-dev

# Production
make api
```

The API will be available at:
- **API**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health

### Docker Deployment

```bash
# Build image
make docker-build

# Run container
make docker-run

# Or full stack with monitoring
docker-compose up -d

# Training in container
docker-compose --profile training up training
```

## API Usage

### Score Single Applicant (German Credit Data Format)

The model is trained on the **UCI German Credit Data** (Statlog). Use these exact field names:

```bash
curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{
    "checking_account": "0 <= ... < 200 DM",
    "duration_months": 24,
    "credit_history": "existing credits paid back duly till now",
    "purpose": "car (used)",
    "credit_amount": 5000,
    "savings": "100 <= ... < 500 DM",
    "employment_duration": "1 <= ... < 4 years",
    "installment_rate_pct": 3,
    "personal_status": "male: married/widowed",
    "other_debtors": "none",
    "residence_years": 3,
    "property": "car or other",
    "age": 35,
    "installment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled employee/official",
    "dependents": 1,
    "telephone": "yes",
    "foreign_worker": "no"
  }'
```

**Valid values for categorical fields:**

| Field | Valid Values |
|-------|--------------|
| `checking_account` | `< 0 DM`, `0 <= ... < 200 DM`, `>= 200 DM`, `no checking account` |
| `credit_history` | `no credits/all paid back`, `all credits at this bank paid back duly`, `existing credits paid back duly till now`, `delay in paying off in the past`, `critical account/other credits existing` |
| `purpose` | `car (new)`, `car (used)`, `furniture/equipment`, `radio/television`, `domestic appliances`, `repairs`, `education`, `vacation`, `retraining`, `business`, `others` |
| `savings` | `< 100 DM`, `100 <= ... < 500 DM`, `500 <= ... < 1000 DM`, `>= 1000 DM`, `unknown/no savings account` |
| `employment_duration` | `unemployed`, `< 1 year`, `1 <= ... < 4 years`, `4 <= ... < 7 years`, `>= 7 years` |
| `personal_status` | `male: divorced/separated`, `female: divorced/separated/married`, `male: single`, `male: married/widowed`, `female: single` |
| `other_debtors` | `none`, `co-applicant`, `guarantor` |
| `property` | `real estate`, `building society savings/life insurance`, `car or other`, `unknown/no property` |
| `installment_plans` | `bank`, `stores`, `none` |
| `housing` | `rent`, `own`, `for free` |
| `job` | `unemployed/unskilled non-resident`, `unskilled resident`, `skilled employee/official`, `management/self-employed/highly qualified` |
| `telephone` | `none`, `yes` |
| `foreign_worker` | `yes`, `no` |

**Response:**
```json
{
  "applicant_id": "APP_001",
  "default_probability": 0.1234,
  "risk_band": "Low",
  "risk_score": 877,
  "threshold_used": 0.5,
  "decision": "Approve",
  "key_factors": [
    {"factor": "Excellent Credit Score", "value": 720, "impact": "high", "direction": "decreases_risk"},
    {"factor": "Strong Payment History", "value": 0.98, "impact": "high", "direction": "decreases_risk"}
  ],
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Batch Scoring

```bash
curl -X POST "http://localhost:8000/score/batch" \
  -H "Content-Type: application/json" \
  -d '{"applicants": [...]}'
```

## Project Structure

```
credit-card-risk/
├── src/credit_risk/           # Main package
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── pipeline.py            # Main training pipeline
│   ├── api.py                 # FastAPI REST API
│   ├── scoring.py             # Risk scoring engine
│   ├── data/
│   │   ├── ingestion.py       # Data loading & validation
│   │   └── preprocessing.py   # Feature engineering & preprocessing
│   └── models/
│       ├── training.py        # Model training & tuning
│       └── evaluation.py      # Model evaluation & plotting
├── tests/                     # Unit & integration tests
├── powerbi/                   # Power BI DAX measures
├── monitoring/                # Prometheus/Grafana configs
├── .github/workflows/         # CI/CD pipelines
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Multi-service deployment
├── requirements.txt           # Python dependencies
├── Makefile                   # Common commands
└── README.md                  # This file
```

## Model Performance

| Model | ROC-AUC | Precision | Recall | F1 | Best Threshold |
|-------|---------|-----------|--------|-----|----------------|
| Logistic Regression | 0.78 | 0.72 | 0.68 | 0.70 | 0.42 |
| Decision Tree | 0.74 | 0.69 | 0.71 | 0.70 | 0.38 |
| Random Forest | 0.81 | 0.75 | 0.73 | 0.74 | 0.45 |
| Gradient Boosting | **0.83** | **0.78** | **0.76** | **0.77** | **0.48** |

*Results on German Credit Data (1000 samples, 70/30 split, stratified)*

## Risk Bands

| Band | Probability Range | Color | Typical Action |
|------|------------------|-------|----------------|
| Very Low | 0.00 - 0.15 | 🟢 #2E7D32 | Auto-approve |
| Low | 0.15 - 0.30 | 🟢 #66BB6A | Approve |
| Medium | 0.30 - 0.50 | 🟡 #FFCA28 | Manual review |
| High | 0.50 - 0.70 | 🟠 #EF6C00 | Decline/Review |
| Very High | 0.70 - 1.00 | 🔴 #C62828 | Auto-decline |

## Power BI Dashboard

1. Open Power BI Desktop
2. Load CSV files from `outputs/<run>/powerbi/`:
   - `scored_applicants.csv` - Individual predictions
   - `feature_importance.csv` - Model interpretability
   - `risk_bands.csv` - Risk band definitions
   - `model_performance.csv` - Metrics summary
3. Copy DAX measures from `powerbi/measures.dax` into your model
4. Create visualizations using the provided measures

Key dashboard pages:
- **Portfolio Overview**: Total applicants, default rate, risk distribution
- **Model Performance**: ROC curve, PR curve, calibration, confusion matrix
- **Risk Segmentation**: Risk band distribution, decision counts
- **Feature Importance**: Top drivers of default risk
- **Threshold Analysis**: What-if parameter for decision threshold

## Configuration

All configuration is in `src/credit_risk/config.py` with these sections:

- **DataConfig**: Paths, split ratios, target column
- **FeatureConfig**: Feature lists, outlier handling
- **ModelConfig**: Algorithms, hyperparameter spaces, CV settings
- **EvaluationConfig**: Metrics, threshold optimization, risk bands
- **APIConfig**: Server settings, model paths

Override via environment variables or `.env` file.

## Testing

```bash
# Run all tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_ingestion.py -v
pytest tests/test_api.py -v
```

## Monitoring

The API exposes Prometheus metrics at `/metrics`. With docker-compose monitoring profile:

```bash
docker-compose --profile monitoring up -d
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

## CI/CD Pipeline

GitHub Actions workflow (`.github/workflows/ci.yml`):

1. **Lint & Test**: Code quality, unit tests, coverage
2. **Train Model**: Weekly scheduled retraining + on main branch pushes
3. **Docker Build**: Multi-stage build with test verification
4. **Deploy Staging**: Auto-deploy on develop branch
5. **Deploy Production**: Manual approval on main branch

## Security Considerations

- Model outputs are **decision support only**, not automatic lending decisions
- No PII stored in model artifacts
- API runs as non-root user in container
- Input validation on all endpoints
- Rate limiting recommended for production
- Audit logging for all predictions

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests and linting (`make test lint`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open Pull Request

## Support

For issues and questions:
- Open a GitHub Issue
- Check existing documentation in `/docs`
- Review API docs at `/docs` endpoint

---

**Disclaimer**: This model is for educational and decision-support purposes only. It should not be used as the sole basis for lending decisions without proper validation, regulatory review, and human oversight.