from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.middleware.error_handler import (
    http_exception_handler,
    validation_exception_handler,
    generic_exception_handler,
)
from app.models.ml_models import ModelManager
from app.services.prediction_service import PredictionService
from app.routes.router import register_routes


def create_app() -> FastAPI:
    """
    Application factory — creates and configures the FastAPI app.

    This is the single entry point for assembling all layers:
    config → logging → middleware → models → services → routes → static files.
    """
    settings = get_settings()

    # --- Logging ---
    setup_logging(settings.LOG_LEVEL)
    logger = get_logger(__name__)
    logger.info("Creating application: %s v%s", settings.APP_NAME, settings.VERSION)

    # --- FastAPI instance ---
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    # --- CORS ---
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --- Exception handlers ---
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    # --- ML models ---
    model_manager = ModelManager(settings.MODELS_DIR)
    model_manager.load_models()

    # --- Services ---
    prediction_service = PredictionService(model_manager)

    # --- Store in app state for dependency injection ---
    app.state.settings = settings
    app.state.model_manager = model_manager
    app.state.prediction_service = prediction_service

    # --- Routes ---
    register_routes(app)

    # --- Static files ---
    app.mount("/static", StaticFiles(directory="app/views/static"), name="static")

    logger.info("Application ready — listening on %s:%d", settings.HOST, settings.PORT)
    return app
