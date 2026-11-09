import threading
from datetime import datetime, timezone

import pandas as pd

from app.models.schemas import (
    BatchHistoryItem,
    BatchResult,
    BatchRow,
    ConfusionCounts,
    DashboardSummary,
    Factor,
    ModeBreakdown,
)
from app.core.logging import get_logger

logger = get_logger(__name__)

FAILURE_MODE_FLAGS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
UNKNOWN_MODE = "UNKNOWN"


class MonitoringService:
    """
    In-memory monitoring store: batch history and cumulative analytics.

    Pure Python/pandas — no HTTP and no model access. Thread-safe
    (sync endpoints run in the threadpool; state is guarded by a lock).
    """

    def __init__(self, threshold: float, total_available: int):
        self._lock = threading.Lock()
        self._threshold = threshold
        self._total_available = total_available
        self._reset_state()

    def _reset_state(self) -> None:
        """Clear all stored batches and cumulative counters."""
        self._batches: list[BatchResult] = []
        self._counts = ConfusionCounts()
        self._mode_breakdown = {mode: ModeBreakdown() for mode in FAILURE_MODE_FLAGS}
        self._mode_breakdown[UNKNOWN_MODE] = ModeBreakdown()
        self._machines_scored = 0

    @property
    def threshold(self) -> float:
        """Decision threshold used to flag rows."""
        return self._threshold

    def record_batch(self, scored_rows: pd.DataFrame) -> BatchResult:
        """
        Record one scored batch and update cumulative analytics.

        Parameters
        ----------
        scored_rows : pd.DataFrame
            Output of ``PredictionService.predict_batch`` — must contain the
            replay columns (``UDI``, ``Product ID``, ``Type``, sensor columns,
            ``Machine failure``, ``TWF``... ``RNF``) plus ``score``,
            ``flagged`` and ``top_factors``.

        Returns
        -------
        BatchResult
            The stored batch with its per-row data and this batch's counts.
        """
        rows = [self._to_batch_row(record) for _, record in scored_rows.iterrows()]

        batch_counts = ConfusionCounts()
        for row in rows:
            if row.flagged and row.actual_failure:
                batch_counts.tp += 1
            elif row.flagged:
                batch_counts.fp += 1
            elif row.actual_failure:
                batch_counts.fn += 1
            else:
                batch_counts.tn += 1

        with self._lock:
            self._counts.tp += batch_counts.tp
            self._counts.fp += batch_counts.fp
            self._counts.fn += batch_counts.fn
            self._counts.tn += batch_counts.tn
            self._machines_scored += len(rows)

            for row in rows:
                if not row.actual_failure:
                    continue
                modes = row.failure_modes if row.failure_modes else [UNKNOWN_MODE]
                for mode in modes:
                    if row.flagged:
                        self._mode_breakdown[mode].caught += 1
                    else:
                        self._mode_breakdown[mode].missed += 1

            batch = BatchResult(
                batch_id=len(self._batches) + 1,
                created_at=datetime.now(timezone.utc),
                size=len(rows),
                rows=rows,
                counts=batch_counts,
            )
            self._batches.append(batch)

        logger.info(
            "Batch %d recorded: size=%d, tp=%d, fp=%d, fn=%d, tn=%d",
            batch.batch_id,
            batch.size,
            batch_counts.tp,
            batch_counts.fp,
            batch_counts.fn,
            batch_counts.tn,
        )
        return batch

    def summary(self) -> DashboardSummary:
        """Cumulative analytics across all recorded batches."""
        with self._lock:
            precision, recall, f1 = self._safe_metrics(self._counts)
            return DashboardSummary(
                threshold=self._threshold,
                batches_processed=len(self._batches),
                machines_scored=self._machines_scored,
                total_available=self._total_available,
                counts=self._counts.model_copy(),
                precision=precision,
                recall=recall,
                f1=f1,
                mode_breakdown={
                    mode: breakdown.model_copy()
                    for mode, breakdown in self._mode_breakdown.items()
                },
                replay_complete=self._machines_scored >= self._total_available,
            )

    def history(self, include_rows: bool = False) -> list[BatchHistoryItem]:
        """
        Compact history of all recorded batches.

        Parameters
        ----------
        include_rows : bool
            When True, each item also carries its full row list (used by the
            dashboard to rebuild its state after a page reload).
        """
        with self._lock:
            items = []
            for batch in self._batches:
                precision, recall, _ = self._safe_metrics(batch.counts)
                items.append(
                    BatchHistoryItem(
                        batch_id=batch.batch_id,
                        size=batch.size,
                        counts=batch.counts.model_copy(),
                        precision=precision,
                        recall=recall,
                        rows=batch.rows if include_rows else None,
                    )
                )
            return items

    def reset(self) -> DashboardSummary:
        """Clear all state and return the resulting (empty) summary."""
        with self._lock:
            self._reset_state()
        logger.info("Monitoring state reset")
        return self.summary()

    def _to_batch_row(self, record: pd.Series) -> BatchRow:
        """Convert one scored row into a BatchRow schema object."""
        actual_failure = bool(record["Machine failure"])
        failure_modes = [
            flag for flag in FAILURE_MODE_FLAGS if bool(record[flag])
        ]
        top_factors = []
        for factor in record.get("top_factors") or []:
            if isinstance(factor, Factor):
                top_factors.append(factor)
            else:
                top_factors.append(
                    Factor(feature=str(factor["feature"]), contribution=float(factor["contribution"]))
                )
        return BatchRow(
            udi=int(record["UDI"]),
            product_id=str(record["Product ID"]),
            type=str(record["Type"]),
            air_temperature_k=float(record["Air temperature [K]"]),
            process_temperature_k=float(record["Process temperature [K]"]),
            rotational_speed_rpm=float(record["Rotational speed [rpm]"]),
            torque_nm=float(record["Torque [Nm]"]),
            tool_wear_min=float(record["Tool wear [min]"]),
            score=float(record["score"]),
            flagged=bool(record["flagged"]),
            actual_failure=actual_failure,
            failure_modes=failure_modes if failure_modes else (["UNKNOWN"] if actual_failure else []),
            top_factors=top_factors,
        )

    @staticmethod
    def _safe_metrics(
        counts: ConfusionCounts,
    ) -> tuple[float | None, float | None, float | None]:
        """
        Precision / recall / F1 with zero-division guards.

        Precision is None when tp + fp == 0, recall is None when
        tp + fn == 0; F1 is None if either is None or both are 0.
        """
        precision = None
        if counts.tp + counts.fp > 0:
            precision = counts.tp / (counts.tp + counts.fp)
        recall = None
        if counts.tp + counts.fn > 0:
            recall = counts.tp / (counts.tp + counts.fn)
        f1 = None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1
