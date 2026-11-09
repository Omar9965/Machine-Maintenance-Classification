from fastapi import APIRouter, HTTPException, Query, Request

from app.models.schemas import (
    BatchHistoryItem,
    DashboardConfig,
    DashboardSummary,
    SimulationStepResponse,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Dashboard"])


@router.post("/simulation/next", response_model=SimulationStepResponse)
async def simulation_next(
    request: Request,
    size: int | None = Query(default=None),
) -> SimulationStepResponse:
    """
    Run one simulation tick.

    ``size`` defaults to SIM_BATCH_SIZE; must satisfy 1 <= size <= SIM_BATCH_MAX.
    """
    settings = request.app.state.settings
    batch_size = settings.SIM_BATCH_SIZE if size is None else size
    if not 1 <= batch_size <= settings.SIM_BATCH_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"size must be between 1 and {settings.SIM_BATCH_MAX}, got {batch_size}",
        )
    try:
        simulation_service = request.app.state.simulation_service
        return simulation_service.step(batch_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulation/reset", response_model=DashboardSummary)
async def simulation_reset(request: Request) -> DashboardSummary:
    """Reset the replay source and monitoring state."""
    try:
        simulation_service = request.app.state.simulation_service
        return simulation_service.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/summary", response_model=DashboardSummary)
async def dashboard_summary(request: Request) -> DashboardSummary:
    """Cumulative dashboard analytics."""
    monitoring_service = request.app.state.monitoring_service
    return monitoring_service.summary()


@router.get("/dashboard/batches", response_model=list[BatchHistoryItem])
async def dashboard_batches(
    request: Request,
    include_rows: bool = Query(default=False),
) -> list[BatchHistoryItem]:
    """
    History of processed batches.

    With ``include_rows=true``, each item carries its full row list (used by
    the dashboard to rebuild state after a page reload).
    """
    monitoring_service = request.app.state.monitoring_service
    return monitoring_service.history(include_rows=include_rows)


@router.get("/dashboard/config", response_model=DashboardConfig)
async def dashboard_config(request: Request) -> DashboardConfig:
    """Configuration for the front end (threshold, poll interval, batch limits)."""
    settings = request.app.state.settings
    model_manager = request.app.state.model_manager
    return DashboardConfig(
        threshold=model_manager.threshold,
        poll_ms=settings.DASHBOARD_POLL_MS,
        batch_size_default=settings.SIM_BATCH_SIZE,
        batch_size_max=settings.SIM_BATCH_MAX,
    )
