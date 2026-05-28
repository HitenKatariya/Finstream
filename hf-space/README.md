---
title: FinStream Sentiment API
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
---

# FinStream Sentiment API

GPU-accelerated FastAPI backend for FinStream financial sentiment analysis.

## API Endpoints

- `GET /health` - Service and model status
- `POST /predict` - Single text sentiment analysis
- `POST /analyze-csv` - Batch CSV analysis

## Model

- **Model**: hitenvk22/FinStream-Sentiment
- **Architecture**: distilroberta-base
- **Task**: 3-class financial sentiment (bullish, neutral, bearish)
