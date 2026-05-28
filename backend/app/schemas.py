"""Pydantic models for API requests and responses."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """Input payload for prediction requests."""

    text: str = Field(min_length=1, max_length=10_000)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text cannot be empty")
        return value


class PredictResponse(BaseModel):
    """Prediction payload returned by the API."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    """Health check payload for deployment monitors."""

    status: str
    model_loaded: bool
    device: str
    model_name: str


class BatchPredictionItem(BaseModel):
    """One row from a CSV batch prediction result."""

    row_number: int
    message: str
    predicted_label: Literal["bullish", "neutral", "bearish", "unknown"]
    confidence: float = Field(ge=0.0, le=1.0)


class BatchAnalysisSummary(BaseModel):
    """Aggregate metrics for a CSV batch analysis."""

    report_id: str
    detected_text_column: str
    total_rows: int
    analyzed_rows: int
    bullish_count: int
    neutral_count: int
    bearish_count: int
    unknown_count: int
    bullish_pct: float
    neutral_pct: float
    bearish_pct: float
    unknown_pct: float
    net_sentiment: float
    net_sentiment_label: str
    average_confidence: float
    report_pdf_url: str


class BatchAnalysisResponse(BaseModel):
    """Response returned for CSV batch analysis."""

    summary: BatchAnalysisSummary
    predictions: list[BatchPredictionItem]
