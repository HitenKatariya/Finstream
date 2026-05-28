"""HTTP route definitions for the FinStream API."""

from io import BytesIO
import logging

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
from ..services.batch_service import analyze_csv_frame, build_report_context, generate_report_id, sanitize_report_filename
from ..services.report_service import get_report_path, create_pdf_report


router = APIRouter()
logger = logging.getLogger("finstream.api")


def _get_model_manager(request: Request):
    return request.app.state.model_manager


@router.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request) -> HealthResponse:
    """Return service and model readiness status."""

    model_manager = _get_model_manager(request)
    return HealthResponse(
        status="ok" if model_manager.is_ready else "degraded",
        model_loaded=model_manager.is_ready,
        device=model_manager.device,
        model_name=model_manager.model_name,
    )


@router.post("/predict", response_model=PredictResponse, tags=["prediction"])
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    """Run sentiment inference against the FinStream model."""

    model_manager = _get_model_manager(request)
    if not model_manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready",
        )

    try:
        result = await run_in_threadpool(model_manager.predict, payload.text)
        return PredictResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed",
        ) from exc


@router.post("/analyze-csv", response_model=BatchAnalysisResponse, tags=["batch-analysis"])
async def analyze_csv(
    request: Request,
    file: UploadFile = File(...),
    report_id: str | None = Form(default=None),
) -> BatchAnalysisResponse:
    """Analyze a CSV file, auto-detecting the message column and generating a PDF report."""

    model_manager = _get_model_manager(request)
    if not model_manager.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready",
        )

    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please upload a CSV file",
        )

    try:
        raw_bytes = await file.read()
        if not raw_bytes:
            raise ValueError("Uploaded CSV file is empty")

        frame = pd.read_csv(BytesIO(raw_bytes))
        analysis = await run_in_threadpool(analyze_csv_frame, frame, model_manager)

        generated_report_id = report_id.strip() if report_id and report_id.strip() else generate_report_id()
        safe_report_id = sanitize_report_filename(generated_report_id)
        report_context = build_report_context(safe_report_id, analysis["summary"])
        predictions_frame = analysis["predictions_frame"]
        pdf_path = await run_in_threadpool(create_pdf_report, safe_report_id, report_context, predictions_frame)

        summary = BatchAnalysisSummary(
            report_id=safe_report_id,
            detected_text_column=analysis["summary"]["detected_text_column"],
            total_rows=analysis["summary"]["total_rows"],
            analyzed_rows=analysis["summary"]["analyzed_rows"],
            bullish_count=analysis["summary"]["bullish_count"],
            neutral_count=analysis["summary"]["neutral_count"],
            bearish_count=analysis["summary"]["bearish_count"],
            unknown_count=analysis["summary"]["unknown_count"],
            bullish_pct=analysis["summary"]["bullish_pct"],
            neutral_pct=analysis["summary"]["neutral_pct"],
            bearish_pct=analysis["summary"]["bearish_pct"],
            unknown_pct=analysis["summary"]["unknown_pct"],
            net_sentiment=analysis["summary"]["net_sentiment"],
            net_sentiment_label=analysis["summary"]["net_sentiment_label"],
            average_confidence=analysis["summary"]["average_confidence"],
            report_pdf_url=f"/reports/{safe_report_id}.pdf",
        )

        item_payload = predictions_frame.reset_index(drop=True).to_dict(orient="records")

        return BatchAnalysisResponse(
            summary=summary,
            predictions=[
                {
                    "row_number": int(item["row_number"]),
                    "message": str(item["message"]),
                    "predicted_label": str(item["predicted_label"]),
                    "confidence": float(item["confidence"]),
                }
                for item in item_payload
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("CSV analysis failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CSV analysis failed: {exc}",
        ) from exc


@router.get("/reports/{report_id}.pdf", tags=["reports"])
async def download_report(report_id: str):
    """Download the generated PDF report for a report id."""

    pdf_path = get_report_path(sanitize_report_filename(report_id))
    if not pdf_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found",
        )

    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )
