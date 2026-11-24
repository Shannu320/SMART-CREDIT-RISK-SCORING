"""
FastAPI REST API for credit risk scoring.
Production-ready inference endpoint with validation, monitoring, and documentation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import uvicorn

from credit_risk.config import config
from credit_risk.scoring import CreditRiskScorer, RiskScore, BatchScoreResult

logger = logging.getLogger(__name__)

# Global scorer instance
scorer: Optional[CreditRiskScorer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global scorer
    logger.info("Starting up credit risk API...")
    try:
        scorer = CreditRiskScorer()
        logger.info("Scorer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize scorer: {e}")
        scorer = None
    yield
    logger.info("Shutting down credit risk API...")


app = FastAPI(
    title="Credit Risk Scoring API",
    description="Production-ready API for loan default probability prediction and risk assessment",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response Models
class ApplicantRequest(BaseModel):
    """Single applicant scoring request."""
    applicant_id: Optional[str] = None
    age: int = Field(..., ge=18, le=100, description="Applicant age")
    gender: str = Field(..., pattern="^(M|F)$", description="Gender")
    marital_status: str = Field(..., description="Marital status")
    dependents: int = Field(..., ge=0, le=10, description="Number of dependents")
    employment_type: str = Field(..., description="Employment type")
    employment_tenure_years: float = Field(..., ge=0, le=50, description="Employment tenure in years")
    annual_income: float = Field(..., ge=0, description="Annual income")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    credit_history_length_years: float = Field(..., ge=0, le=50, description="Credit history length")
    num_credit_accounts: int = Field(..., ge=0, le=50, description="Number of credit accounts")
    num_delinquent_accounts: int = Field(..., ge=0, le=20, description="Number of delinquent accounts")
    past_defaults: int = Field(..., ge=0, le=10, description="Number of past defaults")
    bankruptcy_flag: int = Field(..., ge=0, le=1, description="Bankruptcy flag (0/1)")
    total_debt: float = Field(..., ge=0, description="Total debt")
    monthly_debt_payment: float = Field(..., ge=0, description="Monthly debt payment")
    debt_to_income_ratio: float = Field(..., ge=0, le=2, description="Debt to income ratio")
    loan_amount: float = Field(..., ge=1000, le=1000000, description="Loan amount requested")
    loan_term_months: int = Field(..., ge=6, le=120, description="Loan term in months")
    loan_purpose: str = Field(..., description="Loan purpose")
    interest_rate: float = Field(..., ge=0, le=50, description="Interest rate")
    num_late_payments_30d: int = Field(..., ge=0, le=50, description="30-day late payments")
    num_late_payments_60d: int = Field(..., ge=0, le=30, description="60-day late payments")
    num_late_payments_90d: int = Field(..., ge=0, le=20, description="90-day late payments")
    on_time_payment_pct: float = Field(..., ge=0, le=1, description="On-time payment percentage")

    class Config:
        schema_extra = {
            "example": {
                "applicant_id": "APP_001",
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
        }


class ApplicantResponse(BaseModel):
    """Single applicant scoring response."""
    applicant_id: str
    default_probability: float
    risk_band: str
    risk_score: int
    threshold_used: float
    decision: str
    key_factors: List[Dict[str, Any]]
    timestamp: str


class BatchRequest(BaseModel):
    """Batch scoring request."""
    applicants: List[ApplicantRequest]


class BatchResponse(BaseModel):
    """Batch scoring response."""
    total_applicants: int
    approved: int
    review: int
    declined: int
    risk_distribution: Dict[str, int]
    average_default_prob: float
    scores: List[ApplicantResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: Optional[str]
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str
    timestamp: str


# Dependency
def get_scorer() -> CreditRiskScorer:
    """Get scorer instance."""
    if scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return scorer


# Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if scorer is not None else "unhealthy",
        model_loaded=scorer is not None,
        model_type=scorer.model_metadata.get("model_type") if scorer else None,
        timestamp=datetime.utcnow().isoformat() + "Z"
    )


@app.post("/score", response_model=ApplicantResponse, tags=["Scoring"])
async def score_applicant(
    request: ApplicantRequest,
    background_tasks: BackgroundTasks,
    scorer_instance: CreditRiskScorer = Depends(get_scorer)
):
    """
    Score a single applicant for credit risk.

    Returns default probability, risk band, risk score, and lending decision.
    """
    try:
        # Convert to dict, excluding None applicant_id
        data = request.dict(exclude_none=True)
        applicant_id = data.pop("applicant_id", None)

        # Score
        result = scorer_instance.score_applicant(data, applicant_id)

        # Log for monitoring (background)
        background_tasks.add_task(log_prediction, result)

        return ApplicantResponse(**result.to_dict())

    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/score/batch", response_model=BatchResponse, tags=["Scoring"])
async def score_batch(
    request: BatchRequest,
    background_tasks: BackgroundTasks,
    scorer_instance: CreditRiskScorer = Depends(get_scorer)
):
    """
    Score multiple applicants in batch.

    Returns summary statistics and individual scores.
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([a.dict(exclude_none=True) for a in request.applicants])

        # Score batch
        result = scorer_instance.score_batch(df)

        # Log for monitoring
        background_tasks.add_task(log_batch_prediction, result)

        return BatchResponse(
            total_applicants=result.total_applicants,
            approved=result.approved,
            review=result.review,
            declined=result.declined,
            risk_distribution=result.risk_distribution,
            average_default_prob=result.average_default_prob,
            scores=[ApplicantResponse(**s.to_dict()) for s in result.scores]
        )

    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info", tags=["Model"])
async def model_info(scorer_instance: CreditRiskScorer = Depends(get_scorer)):
    """Get model information."""
    return {
        "model_type": scorer_instance.model_metadata.get("model_type"),
        "threshold": scorer_instance.threshold,
        "feature_count": len(scorer_instance.feature_names) if scorer_instance.feature_names else 0,
        "created_at": scorer_instance.model_metadata.get("created_at"),
        "risk_bands": config.evaluation.risk_bands
    }


@app.get("/model/features", tags=["Model"])
async def model_features(scorer_instance: CreditRiskScorer = Depends(get_scorer)):
    """Get model feature names."""
    return {
        "features": scorer_instance.feature_names,
        "count": len(scorer_instance.feature_names) if scorer_instance.feature_names else 0
    }


# Background logging functions
async def log_prediction(result: RiskScore):
    """Log prediction for monitoring."""
    logger.info(f"Prediction: {result.applicant_id} | "
                f"Prob: {result.default_probability:.4f} | "
                f"Band: {result.risk_band} | "
                f"Decision: {result.decision}")


async def log_batch_prediction(result: BatchScoreResult):
    """Log batch prediction for monitoring."""
    logger.info(f"Batch: {result.total_applicants} applicants | "
                f"Approved: {result.approved} | "
                f"Review: {result.review} | "
                f"Declined: {result.declined} | "
                f"Avg Prob: {result.average_default_prob:.4f}")


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc.detail),
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.utcnow().isoformat() + "Z"
        ).dict()
    )


def run_server(host: str = "0.0.0.0", port: int = 8000, workers: int = 4):
    """Run the API server."""
    uvicorn.run(
        "credit_risk.api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_server()