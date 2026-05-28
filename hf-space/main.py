from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from fastapi.exceptions import RequestValidationError

from app.api.routes import router as api_router
from app.services.model_service import SentimentModelManager


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finstream")

MODEL_NAME = os.getenv("MODEL_NAME", "hitenvk22/FinStream-Sentiment")


@asynccontextmanager
async def lifespan(app: FastAPI):
    mm = SentimentModelManager(model_name=MODEL_NAME)
    app.state.model_manager = mm
    await mm.load_async()
    logger.info("Device: %s | Ready: %s", mm.device, mm.is_ready)
    yield


app = FastAPI(
    title="FinStream Sentiment API",
    version="1.0.0",
    description="GPU-accelerated FinStream sentiment inference on Hugging Face Spaces",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    mm = getattr(app.state, "model_manager", None)
    return {
        "service": "FinStream Sentiment API",
        "version": "1.0.0",
        "mode": "transformers",
        "status": "running",
        "model": MODEL_NAME,
        "device": mm.device if mm else "unknown",
        "endpoints": {
            "predict": "/predict",
            "analyze_csv": "/analyze-csv",
            "health": "/health",
        },
    }


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning("Validation error on %s: %s", request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request payload", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
