"""Evaluation script for the FinStream Hugging Face sentiment model.

This script loads the published model, runs inference on a custom set of
financial news samples, computes standard classification metrics, and writes
plots plus prediction logs to disk.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


MODEL_NAME = os.getenv("FINSTREAM_MODEL_NAME", "hitenvk22/FinStream-Sentiment")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
SAMPLE_BATCH_SIZE = int(os.getenv("FINSTREAM_EVAL_BATCH_SIZE", "8"))
LABEL_ORDER = ["bullish", "neutral", "bearish"]
LABEL_COLORS = {
    "bullish": "#22c55e",
    "neutral": "#f59e0b",
    "bearish": "#ef4444",
}

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
RESULTS_DIR = BASE_DIR / "results"
PREDICTIONS_CSV = RESULTS_DIR / "prediction_logs.csv"
METRICS_JSON = RESULTS_DIR / "metrics.json"
CLASSIFICATION_REPORT_TXT = RESULTS_DIR / "classification_report.txt"
CLASSIFICATION_REPORT_CSV = RESULTS_DIR / "classification_report.csv"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finstream.evaluation")


CUSTOM_FINANCIAL_NEWS_SAMPLES = [
    {
        "text": "Shares of the cloud software company rallied after quarterly revenue beat expectations and the guidance was raised.",
        "true_label": "bullish",
    },
    {
        "text": "The bank reported stronger net interest income and a surprise jump in deposits, pushing the stock higher in after-hours trading.",
        "true_label": "bullish",
    },
    {
        "text": "Investors bought the dip after the retailer announced an aggressive buyback and better-than-feared holiday sales.",
        "true_label": "bullish",
    },
    {
        "text": "The semiconductor maker posted record margins, and analysts upgraded the outlook following the earnings call.",
        "true_label": "bullish",
    },
    {
        "text": "The energy producer benefited from higher crude prices and said free cash flow will remain strong this quarter.",
        "true_label": "bullish",
    },
    {
        "text": "The fintech platform added new customers at a faster pace, while transaction volume climbed sharply year over year.",
        "true_label": "bullish",
    },
    {
        "text": "The airline cut fuel cost guidance and reported a smaller-than-expected loss, helping shares recover.",
        "true_label": "bullish",
    },
    {
        "text": "The EV manufacturer secured a major battery supply agreement, easing concerns about production capacity.",
        "true_label": "bullish",
    },
    {
        "text": "Markets sold off as inflation accelerated and traders priced in more aggressive rate hikes from the central bank.",
        "true_label": "bearish",
    },
    {
        "text": "The regional lender fell sharply after warning about rising loan losses and weaker credit quality.",
        "true_label": "bearish",
    },
    {
        "text": "The consumer brand missed estimates, cut full-year guidance, and warned that margins will remain under pressure.",
        "true_label": "bearish",
    },
    {
        "text": "Shares tumbled after the cybersecurity company disclosed a breach that could trigger costly remediation spending.",
        "true_label": "bearish",
    },
    {
        "text": "The shipping firm reported lower freight rates and warned that demand could soften through the rest of the year.",
        "true_label": "bearish",
    },
    {
        "text": "The retailer's turnaround stalled as same-store sales declined and inventory discounts weighed on profitability.",
        "true_label": "bearish",
    },
    {
        "text": "The biotech stock dropped after regulators requested additional trial data, delaying a key approval decision.",
        "true_label": "bearish",
    },
    {
        "text": "The industrial company announced layoffs and restructuring charges after demand slowed across core markets.",
        "true_label": "bearish",
    },
    {
        "text": "The central bank left rates unchanged and repeated that future moves will depend on incoming data.",
        "true_label": "neutral",
    },
    {
        "text": "The company said it completed a routine board review and maintained its prior full-year guidance without changes.",
        "true_label": "neutral",
    },
    {
        "text": "Analysts kept a hold rating on the stock after the quarterly update came in broadly in line with expectations.",
        "true_label": "neutral",
    },
    {
        "text": "The asset manager described trading conditions as stable while noting that asset inflows were mixed across product lines.",
        "true_label": "neutral",
    },
    {
        "text": "The conglomerate said it was monitoring macro conditions and would revisit capital allocation later this year.",
        "true_label": "neutral",
    },
    {
        "text": "The market digest noted unchanged analyst targets and no material updates after the investor conference.",
        "true_label": "neutral",
    },
    {
        "text": "The telecom operator reported steady subscriber growth, but management said the results were in line with prior quarters.",
        "true_label": "neutral",
    },
    {
        "text": "The company confirmed a planned leadership transition and said operations should continue without interruption.",
        "true_label": "neutral",
    },
]


def ensure_output_dirs() -> None:
    """Create the results and plots directories if they do not exist."""

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> tuple[str, int]:
    """Select GPU when available, otherwise fall back to CPU."""

    if torch.cuda.is_available():
        return "cuda", 0
    return "cpu", -1


def normalize_label(raw_label: str) -> str:
    """Map the model output to FinStream business labels."""

    normalized = raw_label.strip().lower()

    if normalized in {"positive", "bullish", "label_1", "1", "pos"}:
        return "bullish"
    if normalized in {"negative", "bearish", "label_0", "0", "neg"}:
        return "bearish"
    if normalized in {"neutral", "label_2", "2"}:
        return "neutral"
    if "pos" in normalized:
        return "bullish"
    if "neg" in normalized:
        return "bearish"
    if "neu" in normalized:
        return "neutral"

    return normalized


def load_model_pipeline() -> tuple[Any, str]:
    """Load the Hugging Face transformer pipeline for inference."""

    device_name, device_index = get_device()
    logger.info("Loading %s on %s", MODEL_NAME, device_name)

    tokenizer_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if HF_TOKEN:
        tokenizer_kwargs["token"] = HF_TOKEN
        model_kwargs["token"] = HF_TOKEN

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, **tokenizer_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, **model_kwargs)

    classifier = pipeline(
        task="sentiment-analysis",
        model=model,
        tokenizer=tokenizer,
        device=device_index,
        truncation=True,
    )
    logger.info("Model loaded successfully")
    return classifier, device_name


def predict_samples(classifier: Any, samples: list[dict[str, str]]) -> pd.DataFrame:
    """Run inference on the provided samples and return a tabular log."""

    texts = [sample["text"] for sample in samples]
    predictions = classifier(texts, batch_size=SAMPLE_BATCH_SIZE, truncation=True)

    if isinstance(predictions, dict):
        predictions = [predictions]

    rows: list[dict[str, Any]] = []
    for sample, prediction in zip(samples, predictions, strict=True):
        if isinstance(prediction, list):
            prediction = prediction[0]

        predicted_label = normalize_label(str(prediction.get("label", "unknown")))
        confidence = float(prediction.get("score", 0.0))

        rows.append(
            {
                "text": sample["text"],
                "true_label": sample["true_label"],
                "predicted_label": predicted_label,
                "confidence": confidence,
                "is_correct": sample["true_label"] == predicted_label,
            }
        )

    return pd.DataFrame(rows)


def compute_metrics(predictions_df: pd.DataFrame) -> tuple[dict[str, Any], str, pd.DataFrame, np.ndarray]:
    """Compute the required classification metrics and confusion matrix."""

    y_true = predictions_df["true_label"]
    y_pred = predictions_df["predicted_label"]

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    confusion = confusion_matrix(y_true, y_pred, labels=LABEL_ORDER)
    report_text = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        zero_division=0,
        digits=4,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=LABEL_ORDER,
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()

    metrics = {
        "model_name": MODEL_NAME,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sample_count": int(len(predictions_df)),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }

    return metrics, report_text, report_df, confusion


def save_metrics(metrics: dict[str, Any], report_text: str, report_df: pd.DataFrame) -> None:
    """Persist metrics and report artifacts to the results directory."""

    METRICS_JSON.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CLASSIFICATION_REPORT_TXT.write_text(report_text, encoding="utf-8")
    report_df.to_csv(CLASSIFICATION_REPORT_CSV, index=True)


def plot_confusion_matrix(confusion: np.ndarray) -> Path:
    """Save a confusion matrix heatmap."""

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
        cbar=False,
        ax=ax,
    )
    ax.set_title("Confusion Matrix Heatmap")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    fig.tight_layout()

    path = PLOTS_DIR / "confusion_matrix_heatmap.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_prediction_distribution(predictions_df: pd.DataFrame) -> Path:
    """Save a bar chart showing prediction counts by class."""

    order = LABEL_ORDER
    counts = predictions_df["predicted_label"].value_counts().reindex(order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        x=counts.index,
        y=counts.values,
        palette=[LABEL_COLORS[label] for label in counts.index],
        ax=ax,
    )
    ax.set_title("Prediction Distribution")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Count")
    for index, value in enumerate(counts.values):
        ax.text(index, value + 0.15, str(int(value)), ha="center", va="bottom", fontweight="bold")
    fig.tight_layout()

    path = PLOTS_DIR / "prediction_distribution.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_confidence_histogram(predictions_df: pd.DataFrame) -> Path:
    """Save a histogram of model confidence scores."""

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        predictions_df["confidence"],
        bins=10,
        kde=True,
        color="#38bdf8",
        ax=ax,
    )
    ax.set_title("Confidence Score Histogram")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Frequency")
    fig.tight_layout()

    path = PLOTS_DIR / "confidence_histogram.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sentiment_pie_chart(predictions_df: pd.DataFrame) -> Path:
    """Save a pie chart of predicted sentiment classes."""

    counts = predictions_df["predicted_label"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    active_counts = counts[counts > 0]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(
        active_counts.values,
        labels=active_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=[LABEL_COLORS[label] for label in active_counts.index],
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    ax.set_title("Sentiment Pie Chart")
    fig.tight_layout()

    path = PLOTS_DIR / "sentiment_pie_chart.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_plots(predictions_df: pd.DataFrame, confusion: np.ndarray) -> dict[str, str]:
    """Generate all requested plots and return their saved locations."""

    return {
        "confusion_matrix_heatmap": str(plot_confusion_matrix(confusion)),
        "prediction_distribution": str(plot_prediction_distribution(predictions_df)),
        "confidence_histogram": str(plot_confidence_histogram(predictions_df)),
        "sentiment_pie_chart": str(plot_sentiment_pie_chart(predictions_df)),
    }


def run_evaluation() -> dict[str, Any]:
    """Run the full FinStream evaluation workflow."""

    ensure_output_dirs()
    classifier, device_name = load_model_pipeline()
    predictions_df = predict_samples(classifier, CUSTOM_FINANCIAL_NEWS_SAMPLES)
    metrics, report_text, report_df, confusion = compute_metrics(predictions_df)
    metrics["device"] = device_name

    save_metrics(metrics, report_text, report_df)
    plot_paths = save_plots(predictions_df, confusion)
    predictions_df.to_csv(PREDICTIONS_CSV, index=False)

    logger.info("Evaluation complete")
    logger.info("Accuracy: %.4f | Precision: %.4f | Recall: %.4f | F1: %.4f", metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1_score"])
    logger.info("Metrics saved to %s", METRICS_JSON)
    logger.info("Prediction logs saved to %s", PREDICTIONS_CSV)

    return {
        "metrics": metrics,
        "classification_report": report_text,
        "classification_report_df": report_df,
        "predictions": predictions_df,
        "confusion_matrix": confusion,
        "plot_paths": plot_paths,
        "paths": {
            "metrics": str(METRICS_JSON),
            "classification_report_txt": str(CLASSIFICATION_REPORT_TXT),
            "classification_report_csv": str(CLASSIFICATION_REPORT_CSV),
            "predictions_csv": str(PREDICTIONS_CSV),
        },
    }


def main() -> None:
    """Console entrypoint for running the evaluation end-to-end."""

    results = run_evaluation()
    metrics = results["metrics"]

    print("FinStream evaluation completed")
    print(json.dumps(metrics, indent=2))
    print("Saved artifacts:")
    for name, path in results["paths"].items():
        print(f"- {name}: {path}")
    print("Plot artifacts:")
    for name, path in results["plot_paths"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
