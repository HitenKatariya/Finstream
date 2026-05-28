"""Batch CSV analysis helpers for the FinStream backend."""

from __future__ import annotations

from io import BytesIO
import logging
import re
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from .model_service import SentimentModelManager


logger = logging.getLogger("finstream.batch")

TEXT_COLUMN_HINTS = (
    "text",
    "message",
    "sentence",
    "content",
    "news",
    "headline",
    "comment",
    "description",
    "article",
    "body",
    "post",
)


def generate_report_id(prefix: str = "FSR") -> str:
    """Generate a compact report identifier."""

    return f"{prefix}-{uuid4().hex[:10].upper()}"


def _normalize_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column_name.lower())


def detect_text_column(frame: pd.DataFrame) -> str:
    """Detect the most likely column containing free-text messages."""

    if frame.empty:
        raise ValueError("CSV file is empty")

    normalized_columns = {column: _normalize_column_name(str(column)) for column in frame.columns}

    for column, normalized in normalized_columns.items():
        if normalized in TEXT_COLUMN_HINTS or any(hint in normalized for hint in TEXT_COLUMN_HINTS):
            return column

    object_columns = frame.select_dtypes(include=["object", "string"]).columns.tolist()
    if object_columns:
        scored_columns: list[tuple[float, str]] = []
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


def _sentiment_bucket(label: str) -> str:
    normalized = label.lower()
    if normalized == "bullish":
        return "bullish"
    if normalized == "bearish":
        return "bearish"
    if normalized == "neutral":
        return "neutral"
    return "unknown"


def _sentiment_label(net_sentiment: float) -> str:
    if net_sentiment > 0.12:
        return "positive"
    if net_sentiment < -0.12:
        return "negative"
    return "mixed"


def analyze_csv_frame(
    frame: pd.DataFrame,
    model_manager: SentimentModelManager,
) -> dict[str, Any]:
    """Run batch inference on a CSV frame and return summary data."""

    detected_text_column = detect_text_column(frame)
    working_frame = frame.copy()
    working_frame[detected_text_column] = working_frame[detected_text_column].fillna("").astype(str).str.strip()
    working_frame = working_frame[working_frame[detected_text_column] != ""]

    if working_frame.empty:
        raise ValueError("No non-empty text rows were found in the CSV")

    texts = working_frame[detected_text_column].tolist()
    predictions = model_manager.predict_batch(texts)

    rows: list[dict[str, Any]] = []
    for index, (text, prediction) in enumerate(zip(texts, predictions, strict=True), start=1):
        predicted_label = _sentiment_bucket(str(prediction.get("label", "unknown")))
        confidence = float(prediction.get("confidence", 0.0))
        rows.append(
            {
                "row_number": index,
                "message": text,
                "predicted_label": predicted_label,
                "confidence": confidence,
            }
        )

    prediction_frame = pd.DataFrame(rows)
    counts = prediction_frame["predicted_label"].value_counts().to_dict()
    total_rows = int(len(prediction_frame))
    bullish_count = int(counts.get("bullish", 0))
    neutral_count = int(counts.get("neutral", 0))
    bearish_count = int(counts.get("bearish", 0))
    unknown_count = int(counts.get("unknown", 0))

    bullish_pct = round((bullish_count / total_rows) * 100, 2) if total_rows else 0.0
    neutral_pct = round((neutral_count / total_rows) * 100, 2) if total_rows else 0.0
    bearish_pct = round((bearish_count / total_rows) * 100, 2) if total_rows else 0.0
    unknown_pct = round((unknown_count / total_rows) * 100, 2) if total_rows else 0.0

    net_sentiment = round(((bullish_count - bearish_count) / total_rows), 4) if total_rows else 0.0
    average_confidence = round(float(prediction_frame["confidence"].mean()), 4)

    summary = {
        "detected_text_column": detected_text_column,
        "total_rows": total_rows,
        "analyzed_rows": total_rows,
        "bullish_count": bullish_count,
        "neutral_count": neutral_count,
        "bearish_count": bearish_count,
        "unknown_count": unknown_count,
        "bullish_pct": bullish_pct,
        "neutral_pct": neutral_pct,
        "bearish_pct": bearish_pct,
        "unknown_pct": unknown_pct,
        "net_sentiment": net_sentiment,
        "net_sentiment_label": _sentiment_label(net_sentiment),
        "average_confidence": average_confidence,
    }

    return {
        "summary": summary,
        "predictions_frame": prediction_frame,
    }


def build_report_context(report_id: str, summary: dict[str, Any]) -> dict[str, Any]:
    """Build metadata for the PDF report and API response."""

    return {
        "report_id": report_id,
        "detected_text_column": summary["detected_text_column"],
        "total_rows": summary["total_rows"],
        "analyzed_rows": summary["analyzed_rows"],
        "bullish_count": summary["bullish_count"],
        "neutral_count": summary["neutral_count"],
        "bearish_count": summary["bearish_count"],
        "unknown_count": summary["unknown_count"],
        "bullish_pct": summary["bullish_pct"],
        "neutral_pct": summary["neutral_pct"],
        "bearish_pct": summary["bearish_pct"],
        "unknown_pct": summary["unknown_pct"],
        "net_sentiment": summary["net_sentiment"],
        "net_sentiment_label": summary["net_sentiment_label"],
        "average_confidence": summary["average_confidence"],
    }


def sanitize_report_filename(report_id: str) -> str:
    """Create a filesystem-safe PDF filename for a report id."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", report_id.strip())
