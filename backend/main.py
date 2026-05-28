"""Application entrypoint for the FinStream FastAPI backend."""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request

from app.api.routes import router as api_router
from app.core.config import get_settings
from app.services.model_service import SentimentModelManager


settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finstream")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load shared resources before the API starts serving traffic."""

    model_manager = SentimentModelManager(
        model_name=settings.model_name,
        hf_token=settings.hf_token,
        backend=settings.model_backend,
    )
    app.state.model_manager = model_manager

    await model_manager.load_async()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Production FastAPI backend for FinStream sentiment inference. "
        "The API exposes a Hugging Face powered prediction endpoint."
    ),
    lifespan=lifespan,
)


@app.get("/", tags=["root"])
async def root_configuration():
    """Return a lightweight API configuration summary for the service root."""

    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "mode": settings.model_backend,
        "status": "running",
        "model": settings.model_name,
        "docs": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "health": "/health",
        },
        "endpoints": {
            "predict": "/predict",
            "analyze_csv": "/analyze-csv",
            "report_download": "/reports/{report_id}.pdf",
        },
    }

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials="*" not in settings.cors_origins,
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

