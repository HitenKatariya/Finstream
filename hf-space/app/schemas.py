from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    confidence: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str


class BatchPredictionItem(BaseModel):
    row_number: int
    message: str
    predicted_label: str
    confidence: float


class BatchAnalysisSummary(BaseModel):
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
    summary: BatchAnalysisSummary
    predictions: list[dict]
