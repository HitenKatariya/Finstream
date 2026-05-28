from io import BytesIO
import logging
import os
import re
from uuid import uuid4

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

from ..schemas import (
    BatchAnalysisResponse,
    BatchAnalysisSummary,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)


router = APIRouter()
logger = logging.getLogger("finstream.api")

TEXT_COLUMN_HINTS = (
    "text", "message", "sentence", "content", "news",
    "headline", "comment", "description", "article", "body", "post",
)

REPORTS_DIR = "/tmp/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def _get_model_manager(request: Request):
    return request.app.state.model_manager


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column_name.lower())


def _detect_text_column(frame: pd.DataFrame) -> str:
    if frame.empty:
        raise ValueError("CSV file is empty")
    normalized_columns = {col: _normalize_column_name(str(col)) for col in frame.columns}
    for column, normalized in normalized_columns.items():
        if normalized in TEXT_COLUMN_HINTS or any(hint in normalized for hint in TEXT_COLUMN_HINTS):
            return column
    object_columns = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    if object_columns:
        scored_columns = []
        for column in object_columns:
            series = frame[column].dropna().astype(str).str.strip()
            if series.empty:
                continue
            average_length = series.str.len().mean()
            non_empty_ratio = (series != "").mean()
            scored_columns.append((float(average_length * non_empty_ratio), column))
        if scored_columns:
            scored_columns.sort(reverse=True)
            return scored_columns[0][1]
        return object_columns[0]
    return frame.columns[0]


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    mm = _get_model_manager(request)
    return HealthResponse(
        status="ok" if mm.is_ready else "degraded",
        model_loaded=mm.is_ready,
        device=mm.device,
        model_name=mm.model_name,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    mm = _get_model_manager(request)
    if not mm.is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    try:
        result = await run_in_threadpool(mm.predict, payload.text)
        return PredictResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@router.post("/analyze-csv", response_model=BatchAnalysisResponse)
async def analyze_csv(
    request: Request,
    file: UploadFile = File(...),
    report_id: str | None = Form(default=None),
) -> BatchAnalysisResponse:
    mm = _get_model_manager(request)
    if not mm.is_ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise ValueError("Uploaded CSV file is empty")

        frame = pd.read_csv(BytesIO(raw_bytes))
        detected_text_column = _detect_text_column(frame)
        working_frame = frame.copy()
        working_frame[detected_text_column] = (
            working_frame[detected_text_column].fillna("").astype(str).str.strip()
        )
        working_frame = working_frame[working_frame[detected_text_column] != ""]

        if working_frame.empty:
            raise ValueError("No non-empty text rows were found in the CSV")

        texts = working_frame[detected_text_column].tolist()
        predictions = mm.predict_batch(texts)

        rows = []
        for idx, (text, pred) in enumerate(zip(texts, predictions), start=1):
            label = str(pred.get("label", "unknown")).lower()
            if label == "positive":
                label = "bullish"
            elif label == "negative":
                label = "bearish"
            rows.append({
                "row_number": idx,
                "message": text,
                "predicted_label": label,
                "confidence": float(pred.get("confidence", 0.0)),
            })

        pred_frame = pd.DataFrame(rows)
        counts = pred_frame["predicted_label"].value_counts().to_dict()
        total = len(pred_frame)
        bullish_c = counts.get("bullish", 0)
        neutral_c = counts.get("neutral", 0)
        bearish_c = counts.get("bearish", 0)
        unknown_c = counts.get("unknown", 0)

        rid = report_id.strip() if report_id and report_id.strip() else f"FSR-{uuid4().hex[:10].upper()}"
        net_sent = round(((bullish_c - bearish_c) / total), 4) if total else 0.0
        avg_conf = round(float(pred_frame["confidence"].mean()), 4)

        net_label = "positive" if net_sent > 0.12 else ("negative" if net_sent < -0.12 else "mixed")

        summary = BatchAnalysisSummary(
            report_id=rid,
            detected_text_column=detected_text_column,
            total_rows=total,
            analyzed_rows=total,
            bullish_count=bullish_c,
            neutral_count=neutral_c,
            bearish_count=bearish_c,
            unknown_count=unknown_c,
            bullish_pct=round((bullish_c / total) * 100, 2) if total else 0.0,
            neutral_pct=round((neutral_c / total) * 100, 2) if total else 0.0,
            bearish_pct=round((bearish_c / total) * 100, 2) if total else 0.0,
            unknown_pct=round((unknown_c / total) * 100, 2) if total else 0.0,
            net_sentiment=net_sent,
            net_sentiment_label=net_label,
            average_confidence=avg_conf,
            report_pdf_url=f"/reports/{rid}.pdf",
        )

        return BatchAnalysisResponse(
            summary=summary,
            predictions=pred_frame.reset_index(drop=True).to_dict(orient="records"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("CSV analysis failed")
        raise HTTPException(status_code=500, detail=f"CSV analysis failed: {exc}") from exc
