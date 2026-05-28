from __future__ import annotations

import logging
import re
from threading import Lock

import torch
from anyio import to_thread
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


logger = logging.getLogger("finstream.model")

POSITIVE_WORDS = {
    "beat", "beats", "bullish", "climb", "climbs", "climbed",
    "gain", "gains", "gained", "growth", "higher",
    "improve", "improves", "improved", "improvement", "improvements",
    "outperform", "outperforms", "outperformed",
    "profit", "profits", "profitable", "profitability",
    "rally", "rallies", "rallied",
    "rise", "rises", "rose", "risen",
    "surge", "surges", "surged",
    "strong", "stronger", "strongly",
    "up", "uptick", "upside", "positive", "record",
    "boost", "boosts", "boosted",
    "upgrade", "upgrades", "upgraded",
    "exceed", "exceeds", "exceeded",
    "expand", "expands", "expanded", "expansion",
    "accelerate", "accelerates", "accelerated",
    "recover", "recovers", "recovered", "recovery",
    "rebound", "rebounds", "rebounded",
    "jump", "jumps", "jumped", "soar", "soars", "soared",
    "dividend", "dividends", "buyback", "buybacks",
    "upward", "uptrend", "bull", "upswing", "breakout",
    "optimistic", "optimism", "momentum",
    "upbeat", "win", "wins", "won", "success", "successful",
}

NEGATIVE_WORDS = {
    "bearish", "decline", "declines", "declined",
    "drop", "drops", "dropped",
    "fall", "falls", "fell", "fallen",
    "loss", "losses", "lost",
    "miss", "misses", "missed",
    "pressure", "pressures", "pressured",
    "risk", "risks", "risky",
    "selloff", "selloffs",
    "slump", "slumps", "slumped",
    "soft", "softer", "softness",
    "weak", "weaker", "weakness", "weaknesses", "weaken", "weakens", "weakened",
    "down", "downturn", "downturns", "downside", "downgrade",
    "negative",
    "cut", "cuts", "cutting",
    "lower", "lowers", "lowered",
    "reduce", "reduces", "reduced", "reduction",
    "layoff", "layoffs", "bankrupt", "bankruptcy", "debt",
    "default", "defaults",
    "delay", "delays", "delayed",
    "suspend", "suspends", "suspended", "suspension",
    "worst", "worse", "worsen", "worsens", "worsened",
    "volatile", "volatility",
    "uncertainty", "uncertain",
    "plunge", "plunges", "plunged",
    "tumble", "tumbles", "tumbled",
    "slide", "slides", "slid",
    "crash", "crashes", "crashed",
    "recession", "inflation", "inflationary",
    "underperform", "underperforms", "underperformed",
}


def _normalize_label(raw_label: str) -> str:
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
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = "cpu"
        self._device_index = -1
        self._pipeline = None
        self._load_error: str | None = None

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None and self._load_error is None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    async def load_async(self) -> None:
        await to_thread.run_sync(self.load)

    def load(self) -> None:
        if self._pipeline is not None:
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device_index = 0 if torch.cuda.is_available() else -1
            logger.info("Loading model %s on %s", self.model_name, self.device)

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )
            model.eval()

            self._pipeline = pipeline(
                task="sentiment-analysis",
                model=model,
                tokenizer=tokenizer,
                device=self._device_index,
                truncation=True,
                framework="pt",
            )
            self._load_error = None
            logger.info("Model loaded successfully on %s", self.device)
        except Exception as exc:
            self._load_error = str(exc)
            logger.exception("Failed to load sentiment model")

    @staticmethod
    def _stem(token: str) -> str:
        if len(token) <= 4:
            return token
        for suffix in ["ability", "abilities", "ification", "ifications",
                        "ization", "izations", "isation", "isations",
                        "ationally", "isation", "ization",
                        "iveness", "fulness", "iousness",
                        "ments", "ment", "ances", "ance",
                        "eness", "ness", "ship",
                        "able", "ably", "ible",
                        "ally", "wise", "like",
                        "ious", "eous", "uous",
                        "sion", "tion", "sions", "tions",
                        "ised", "ized", "ising", "izing",
                        "ative", "itive", "tive",
                        "less", "proof", "ward",
                        "ing", "ings",
                        "ed", "es", "er", "est", "ly"]:
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                return token[:-len(suffix)]
        return token

    def _rule_based_predict(self, text: str) -> dict[str, float | str]:
        tokens = re.findall(r"[a-zA-Z']+", text.lower())
        if not tokens:
            return {"label": "neutral", "confidence": 0.5}
        stemmed_tokens = [self._stem(t) for t in tokens]
        positive_hits = sum(
            1 for i, t in enumerate(tokens)
            if t in POSITIVE_WORDS or stemmed_tokens[i] in POSITIVE_WORDS
        )
        negative_hits = sum(
            1 for i, t in enumerate(tokens)
            if t in NEGATIVE_WORDS or stemmed_tokens[i] in NEGATIVE_WORDS
        )
        total_hits = positive_hits + negative_hits
        score = positive_hits - negative_hits
        if total_hits == 0:
            return {"label": "neutral", "confidence": 0.5}
        confidence = min(0.95, max(0.55, 0.55 + (abs(score) / total_hits) * 0.35))
        if score > 0:
            return {"label": "bullish", "confidence": round(confidence, 4)}
        if score < 0:
            return {"label": "bearish", "confidence": round(confidence, 4)}
        return {"label": "neutral", "confidence": round(0.5 + (positive_hits / total_hits) * 0.1, 4)}

    def predict(self, text: str) -> dict[str, float | str]:
        if self._pipeline is None:
            return self._rule_based_predict(text)
        with torch.no_grad():
            output = self._pipeline(text)
        prediction = output[0] if isinstance(output, list) else output
        return {
            "label": _normalize_label(prediction.get("label", "unknown")),
            "confidence": float(prediction.get("score", 0.0)),
        }

    def predict_batch(self, texts: list[str]) -> list[dict[str, float | str]]:
        if self._pipeline is None:
            return [self._rule_based_predict(text) for text in texts]
        with torch.no_grad():
            output = self._pipeline(texts)
        if isinstance(output, dict):
            output = [output]
        results = []
        for prediction in output:
            if isinstance(prediction, list):
                prediction = prediction[0]
            results.append({
                "label": _normalize_label(prediction.get("label", "unknown")),
                "confidence": float(prediction.get("score", 0.0)),
            })
        return results
