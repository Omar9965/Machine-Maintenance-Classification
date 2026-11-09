import threading

from app.data_sources.csv_replay import DataSource
from app.services.monitoring_service import MonitoringService
from app.services.prediction_service import PredictionService
from app.models.schemas import (
    BatchResult,
    ConfusionCounts,
    DashboardSummary,
    SimulationStepResponse,
)
from app.core.logging import get_logger
from datetime import datetime, timezone

logger = get_logger(__name__)


class SimulationService:
    """
    Orchestrates one simulation tick: replay source → prediction → monitoring.

    The in-memory replay state lives in a single process, so ticks are
    serialized with a lock (sync endpoints run in the threadpool).
    """

    def __init__(
        self,
        data_source: DataSource,
        prediction_service: PredictionService,
        monitoring_service: MonitoringService,
    ):
        self._lock = threading.Lock()
        self._data_source = data_source
        self._prediction_service = prediction_service
        self._monitoring = monitoring_service

    def step(self, size: int) -> SimulationStepResponse:
        """
        Run one tick: pull the next batch, score it, record it.

        When the replay is exhausted, returns HTTP 200 semantics with an
        empty batch and ``replay_complete=True``.
        """
        with self._lock:
            df = self._data_source.next_batch(size)
            if df.empty:
                logger.info("Replay exhausted — no rows left to simulate")
                empty_batch = BatchResult(
                    batch_id=0,
                    created_at=datetime.now(timezone.utc),
                    size=0,
                    rows=[],
                    counts=ConfusionCounts(),
                )
                return SimulationStepResponse(
                    batch=empty_batch, summary=self._monitoring.summary()
                )

            scored = self._prediction_service.predict_batch(
                df, threshold=self._monitoring.threshold
            )
            batch = self._monitoring.record_batch(scored)
            summary = self._monitoring.summary()

            logger.info(
                "Simulation step: batch_id=%d, size=%d, remaining=%d",
                batch.batch_id,
                batch.size,
                self._data_source.remaining,
            )
            return SimulationStepResponse(batch=batch, summary=summary)

    def reset(self) -> DashboardSummary:
        """Reset the replay source and monitoring to a fresh state."""
        with self._lock:
            self._data_source.reset()
            summary = self._monitoring.reset()
        logger.info("Simulation reset — replay restarted from scratch")
        return summary
