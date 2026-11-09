from fastapi import FastAPI
from app.controllers.api import home_controller as api_home
from app.controllers.api import prediction_controller as api_prediction
from app.controllers.api import dashboard_controller as api_dashboard
from app.controllers.web import dashboard_controller as web_dashboard


def register_routes(app: FastAPI) -> None:
    """Register all API and web route controllers."""

    # --- API routes (versioned) ---
    app.include_router(api_home.router, prefix="/api/v1")
    app.include_router(api_prediction.router, prefix="/api/v1")
    app.include_router(api_dashboard.router, prefix="/api/v1")

    # --- Web routes ---
    app.include_router(web_dashboard.router)
