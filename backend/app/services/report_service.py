"""PDF report generation for FinStream CSV batch analyses."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Spacer,
    Paragraph,
    Table,
    TableStyle,
)


REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def get_report_path(report_id: str) -> Path:
    """Return the output path for a report PDF."""

    return REPORTS_DIR / f"{report_id}.pdf"


def create_pdf_report(
    report_id: str,
    report_context: dict[str, Any],
    predictions_frame,
) -> Path:
    """Create a clean PDF summary for the batch sentiment report."""

    output_path = get_report_path(report_id)
    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "FinStreamTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#0f172a"),
        fontSize=22,
        leading=26,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "FinStreamBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    )

    story = []
    story.append(Paragraph("FinStream Sentiment Batch Report", title_style))
    story.append(Paragraph(f"Report ID: <b>{report_id}</b>", body_style))
    story.append(Paragraph(f"Detected text column: <b>{report_context['detected_text_column']}</b>", body_style))
    story.append(Paragraph(f"Total analyzed rows: <b>{report_context['analyzed_rows']}</b>", body_style))
    story.append(Paragraph(f"Net sentiment: <b>{report_context['net_sentiment']:+.4f}</b> ({report_context['net_sentiment_label']})", body_style))
    story.append(Paragraph(f"Average confidence: <b>{report_context['average_confidence']:.4f}</b>", body_style))
    story.append(Spacer(1, 0.2 * inch))

    summary_table_data = [
        ["Metric", "Value"],
        ["Bullish", f"{report_context['bullish_count']} ({report_context['bullish_pct']:.2f}%)"],
        ["Neutral", f"{report_context['neutral_count']} ({report_context['neutral_pct']:.2f}%)"],
        ["Bearish", f"{report_context['bearish_count']} ({report_context['bearish_pct']:.2f}%)"],
        ["Unknown", f"{report_context['unknown_count']} ({report_context['unknown_pct']:.2f}%)"],
    ]
    summary_table = Table(summary_table_data, colWidths=[2.1 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0ea5e9")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#e2e8f0")]),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 0.25 * inch))

    story.append(Paragraph("Message-wise sentiment results", styles["Heading2"]))
    preview_rows = predictions_frame.head(20)
    table_data = [["#", "Message", "Label", "Confidence"]]
    for _, row in preview_rows.iterrows():
        message = str(row["message"])
        if len(message) > 95:
            message = message[:92] + "..."
        table_data.append(
            [
                str(int(row["row_number"])),
                message,
                str(row["predicted_label"]),
                f"{float(row['confidence']):.4f}",
            ]
        )

    prediction_table = Table(table_data, colWidths=[0.45 * inch, 3.45 * inch, 1.0 * inch, 1.0 * inch], repeatRows=1)
    prediction_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(prediction_table)
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Generated by the FinStream backend PDF reporting workflow.", body_style))

    document.build(story)
    return output_path