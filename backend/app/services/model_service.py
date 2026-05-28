"""Hugging Face sentiment model loading and inference."""

from __future__ import annotations

import logging
from threading import Lock

from anyio import to_thread
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


logger = logging.getLogger("finstream.model")


def _normalize_label(raw_label: str) -> str:
    """Map common sentiment labels to FinStream business labels."""

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

    return normalized


class SentimentModelManager:
    """Owns model lifecycle and performs thread-safe inference."""

    def __init__(self, model_name: str, hf_token: str | None = None) -> None:
        self.model_name = model_name
        self.hf_token = hf_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device_index = 0 if torch.cuda.is_available() else -1
        self._pipeline = None
        self._lock = Lock()
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None and self._load_error is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def load_async(self) -> None:
        """Load the Hugging Face pipeline during application startup."""

        await to_thread.run_sync(self.load)

    def load(self) -> None:
        """Load the model and tokenizer once, even under concurrent startup."""

        if self._pipeline is not None:
            return

        with self._lock:
            if self._pipeline is not None:
                return

            try:
                logger.info("Loading model %s on %s", self.model_name, self.device)
                tokenizer_kwargs = {}
                model_kwargs = {}
                if self.hf_token:
                    tokenizer_kwargs["token"] = self.hf_token
                    model_kwargs["token"] = self.hf_token

                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    **tokenizer_kwargs,
                )
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    **model_kwargs,
                )
                self._pipeline = pipeline(
                    task="sentiment-analysis",
                    model=model,
                    tokenizer=tokenizer,
                    device=self._device_index,
                    truncation=True,
                )
                self._load_error = None
                logger.info("Model loaded successfully")
            except Exception as exc:
                self._load_error = str(exc)
                logger.exception("Failed to load sentiment model")

    def predict(self, text: str) -> dict[str, float | str]:
        """Run inference synchronously on a background thread."""

        if self._pipeline is None:
            raise RuntimeError("Model is not loaded")

        output = self._pipeline(text)
        prediction = output[0] if isinstance(output, list) else output

        label = prediction.get("label", "unknown")
        score = float(prediction.get("score", 0.0))

        return {"label": _normalize_label(label), "confidence": score}

    def predict_batch(self, texts: list[str]) -> list[dict[str, float | str]]:
        """Run inference for a batch of texts."""

        if self._pipeline is None:
            raise RuntimeError("Model is not loaded")

        output = self._pipeline(texts)
        if isinstance(output, dict):
            output = [output]

        results: list[dict[str, float | str]] = []
        for prediction in output:
            if isinstance(prediction, list):
                prediction = prediction[0]

            label = prediction.get("label", "unknown")
            score = float(prediction.get("score", 0.0))
            results.append({"label": _normalize_label(label), "confidence": score})

        return results
