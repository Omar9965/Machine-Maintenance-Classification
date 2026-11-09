"""Tests for the MonitoringService — counts, metrics, None handling, modes."""

import pandas as pd
import pytest

from app.services.monitoring_service import MonitoringService

THRESHOLD = 0.8966
TOTAL = 2000


def make_rows(n=1, scores=None, failures=None, modes=None, flagged=None):
    """Build a scored frame shaped like PredictionService.predict_batch output."""
    scores = scores if scores is not None else [0.99] * n
    failures = failures if failures is not None else [0] * n
    modes = modes if modes is not None else [[] for _ in range(n)]
    flagged = flagged if flagged is not None else [s >= THRESHOLD for s in scores]
    return pd.DataFrame(
        {
            "UDI": range(1, n + 1),
            "Product ID": [f"M{i}" for i in range(n)],
            "Type": ["M"] * n,
            "Air temperature [K]": [302.0] * n,
            "Process temperature [K]": [311.0] * n,
            "Rotational speed [rpm]": [1500.0] * n,
            "Torque [Nm]": [45.0] * n,
            "Tool wear [min]": [10] * n,
            "score": scores,
            "flagged": flagged,
            "Machine failure": failures,
            "TWF": [int("TWF" in m) for m in modes],
            "HDF": [int("HDF" in m) for m in modes],
            "PWF": [int("PWF" in m) for m in modes],
            "OSF": [int("OSF" in m) for m in modes],
            "RNF": [int("RNF" in m) for m in modes],
            "top_factors": [
                [{"feature": "torque_nm", "contribution": 1.0}] for _ in range(n)
            ],
        }
    )


@pytest.fixture()
def service():
    return MonitoringService(threshold=THRESHOLD, total_available=TOTAL)


def test_initial_summary_is_empty(service):
    summary = service.summary()
    assert summary.batches_processed == 0
    assert summary.machines_scored == 0
    assert summary.total_available == TOTAL
    assert summary.counts.tp == 0
    assert summary.precision is None
    assert summary.recall is None
    assert summary.f1 is None
    assert summary.replay_complete is False
    assert set(summary.mode_breakdown.keys()) == {"TWF", "HDF", "PWF", "OSF", "RNF", "UNKNOWN"}


def test_batch_counts(service):
    batch = service.record_batch(make_rows(scores=[0.99, 0.10, 0.95, 0.20], failures=[1, 1, 0, 0]))
    assert batch.counts.tp == 1  # flagged + failure
    assert batch.counts.fn == 1  # not flagged + failure
    assert batch.counts.fp == 1  # flagged + no failure
    assert batch.counts.tn == 1  # not flagged + no failure
    assert batch.batch_id == 1
    assert batch.size == 4

    cum = service.summary().counts
    assert (cum.tp, cum.fp, cum.fn, cum.tn) == (1, 1, 1, 1)


def test_precision_recall_f1(service):
    # tp=2, fp=0 → precision 1.0; fn=1 → recall 2/3; f1 = 2*1*(2/3)/(1+2/3) = 0.8
    service.record_batch(make_rows(scores=[0.99, 0.98, 0.10], failures=[1, 1, 1]))
    summary = service.summary()
    assert summary.precision == pytest.approx(1.0)
    assert summary.recall == pytest.approx(2 / 3, rel=1e-6)
    assert summary.f1 == pytest.approx(0.8, rel=1e-6)


def test_precision_none_when_no_predictions(service):
    # All flagged=False → tp+fp == 0 → precision None
    service.record_batch(make_rows(scores=[0.10, 0.20], failures=[1, 0]))
    summary = service.summary()
    assert summary.precision is None
    assert summary.f1 is None
    assert summary.recall == 0.0  # one actual failure, missed


def test_recall_none_when_no_failures(service):
    service.record_batch(make_rows(scores=[0.10, 0.99], failures=[0, 0]))
    summary = service.summary()
    assert summary.recall is None
    assert summary.f1 is None
    assert summary.precision == 0.0  # one flagged, no failures


def test_f1_none_when_both_zero(service):
    service.record_batch(make_rows(scores=[0.10], failures=[0]))
    summary = service.summary()
    assert summary.precision is None
    assert summary.recall is None
    assert summary.f1 is None


def test_batch_history(service):
    service.record_batch(make_rows(n=3, scores=[0.9, 0.9, 0.1], failures=[1, 0, 0]))
    service.record_batch(make_rows(n=2, scores=[0.5], failures=[1]))
    items = service.history()
    assert [item.batch_id for item in items] == [1, 2]
    assert all(item.rows is None for item in items)
    assert items[0].size == 3
    with_rows = service.history(include_rows=True)
    assert with_rows[0].rows is not None and len(with_rows[0].rows) == 3


def test_unknown_mode(service):
    # Failure with no mode flag → UNKNOWN
    service.record_batch(make_rows(scores=[0.99], failures=[1]))
    assert service.summary().mode_breakdown["UNKNOWN"].missed == 0
    assert service.summary().mode_breakdown["UNKNOWN"].caught == 1
    # Missed failure with no mode flag
    service.record_batch(make_rows(scores=[0.10], failures=[1]))
    assert service.summary().mode_breakdown["UNKNOWN"].missed == 1


def test_mode_breakdown_caught_vs_missed(service):
    service.record_batch(
        make_rows(
            scores=[0.99, 0.10],
            failures=[1, 1],
            modes=[["OSF"], ["OSF", "HDF"]],
        )
    )
    breakdown = service.summary().mode_breakdown
    assert breakdown["OSF"].caught == 1
    assert breakdown["OSF"].missed == 1
    assert breakdown["HDF"].missed == 1
    assert breakdown["TWF"].caught == 0


def test_reset(service):
    service.record_batch(make_rows(n=2))
    summary = service.reset()
    assert summary.batches_processed == 0
    assert summary.machines_scored == 0
    assert summary.precision is None


def test_replay_complete(service):
    total = 2
    small = MonitoringService(threshold=THRESHOLD, total_available=total)
    small.record_batch(make_rows(n=2))
    assert small.summary().replay_complete is True
