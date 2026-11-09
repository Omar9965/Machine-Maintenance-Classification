"""Tests for the dashboard API: simulation, monitoring endpoints, web page."""

import pytest

from app.services.simulation_service import SimulationService
from app.services.monitoring_service import MonitoringService
from app.services.prediction_service import PredictionService
from app.data_sources.csv_replay import CsvReplaySource


def run_api(client, path, method="get", **kwargs):
    if method == "get":
        return client.get(path, **kwargs)
    return client.post(path, **kwargs)


class TestSimulationNext:
    def test_default_size(self, client, settings):
        response = client.post("/api/v1/simulation/next")
        assert response.status_code == 200
        data = response.json()
        assert data["batch"]["size"] == settings.SIM_BATCH_SIZE
        assert len(data["batch"]["rows"]) == settings.SIM_BATCH_SIZE
        assert sum(
            [data["batch"]["counts"][k] for k in ("tp", "fp", "fn", "tn")]
        ) == settings.SIM_BATCH_SIZE
        assert "summary" in data

    def test_custom_size(self, client):
        response = client.post("/api/v1/simulation/next?size=2")
        assert response.status_code == 200
        data = response.json()
        assert data["batch"]["size"] == 2
        assert len(data["batch"]["rows"]) == 2
        row = data["batch"]["rows"][0]
        expected_keys = {
            "udi", "product_id", "type", "air_temperature_k", "process_temperature_k",
            "rotational_speed_rpm", "torque_nm", "tool_wear_min", "score", "flagged",
            "actual_failure", "failure_modes", "top_factors",
        }
        assert set(row.keys()) == expected_keys
        assert isinstance(row["failure_modes"], list)

    def test_row_shape_of_batch(self, client):
        data = client.post("/api/v1/simulation/next?size=2").json()
        assert data["batch"]["batch_id"] == 1
        assert "created_at" in data["batch"]
        assert data["summary"]["batches_processed"] == 1
        assert data["summary"]["machines_scored"] == 2

    def test_size_zero_is_422(self, client):
        response = run_api(client, "/api/v1/simulation/next?size=0", method="post")
        assert response.status_code == 422

    def test_negative_size_is_422(self, client):
        response = run_api(client, "/api/v1/simulation/next?size=-5", method="post")
        assert response.status_code == 422

    def test_size_above_max_is_422(self, client, settings):
        response = run_api(
            client, f"/api/v1/simulation/next?size={settings.SIM_BATCH_MAX + 1}", method="post"
        )
        assert response.status_code == 422

    def test_sequential_batch_ids(self, client):
        first = client.post("/api/v1/simulation/next?size=3").json()
        second = client.post("/api/v1/simulation/next?size=3").json()
        assert first["batch"]["batch_id"] == 1
        assert second["batch"]["batch_id"] == 2

    def test_scores_returned_for_every_row(self, client):
        data = client.post("/api/v1/simulation/next?size=50", json={}).json()
        scores = [row["score"] for row in data["batch"]["rows"]]
        assert len(scores) == 50
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestDashboardEndpoints:
    def test_summary_empty(self, client):
        response = client.get("/api/v1/dashboard/summary")
        assert response.status_code == 200
        data = response.json()
        assert data["batches_processed"] == 0
        assert data["machines_scored"] == 0
        assert data["counts"] == {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        assert data["precision"] is None
        assert data["recall"] is None
        assert data["f1"] is None
        assert data["replay_complete"] is False
        assert data["total_available"] == 2000

    def test_summary_after_batch(self, client):
        client.post("/api/v1/simulation/next?size=10")
        data = client.get("/api/v1/dashboard/summary").json()
        assert data["machines_scored"] == 10
        assert data["batches_processed"] == 1
        assert data["threshold"] == pytest.approx(0.8966)

    def test_batches_without_rows(self, client):
        client.post("/api/v1/simulation/next?size=4")
        response = client.get("/api/v1/dashboard/batches")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["rows"] is None
        assert "precision" in data[0] and "recall" in data[0]

    def test_batches_with_rows(self, client):
        client.post("/api/v1/simulation/next?size=4")
        data = client.get("/api/v1/dashboard/batches?include_rows=true").json()
        assert len(data) == 1
        assert data[0]["rows"] is not None
        assert len(data[0]["rows"]) == 4

    def test_config(self, client, settings):
        response = client.get("/api/v1/dashboard/config")
        assert response.status_code == 200
        data = response.json()
        assert data["threshold"] == pytest.approx(0.8966)
        assert data["poll_ms"] == settings.DASHBOARD_POLL_MS
        assert data["batch_size_default"] == settings.SIM_BATCH_SIZE
        assert data["batch_size_max"] == settings.SIM_BATCH_MAX

    def test_reset(self, client):
        client.post("/api/v1/simulation/next?size=10")
        response = run_api(client, "/api/v1/simulation/reset", method="post")
        assert response.status_code == 200
        data = response.json()
        assert data["batches_processed"] == 0
        assert data["machines_scored"] == 0
        assert data["replay_complete"] is False

    def test_reset_restarts_replay_deterministically(self, client, settings):
        first = client.post("/api/v1/simulation/next?size=5").json()
        client.post("/api/v1/simulation/reset")
        second = client.post("/api/v1/simulation/next?size=5").json()
        first_udi = [r["udi"] for r in first["batch"]["rows"]]
        second_udi = [r["udi"] for r in second["batch"]["rows"]]
        assert first_udi == second_udi


class TestReplayComplete:
    def test_replay_exhaustion(self, client, settings):
        max_size = settings.SIM_BATCH_MAX
        while True:
            data = client.post(f"/api/v1/simulation/next?size={max_size}").json()
            if data["summary"]["replay_complete"]:
                break
        assert data["summary"]["machines_scored"] == 2000
        # Subsequent ticks return an empty batch
        data = client.post(f"/api/v1/simulation/next?size={max_size}").json()
        assert data["summary"]["replay_complete"] is True
        assert data["batch"]["size"] == 0
        assert data["batch"]["rows"] == []


class TestWebRoutes:
    def test_dashboard_page_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_dashboard_has_no_prediction_form(self, client):
        html = client.get("/").text.lower()
        assert "<form" not in html
        assert "submit" not in html
        assert "/predict" not in html

    def test_removed_get_predict_is_404(self, client):
        assert client.get("/predict").status_code == 404

    def test_removed_post_predict_is_404(self, client):
        response = client.post(
            "/predict",
            data={"Type": "M", "Air temperature [K]": "300.0"},
        )
        assert response.status_code == 404

    def test_dashboard_references_dashboard_assets(self, client):
        html = client.get("/").text
        assert "dashboard.js" in html
        assert "dashboard.html" not in html


# --- Golden test: full replay must equal direct scoring of the test set ---------

class TestGoldenReplay:
    def test_full_replay_matches_direct_scoring(
        self, model_manager, test_set_df, settings
    ):
        # --- Direct reference: score the entire test set in one pass ---
        INPUT_COLUMNS = [
            "Type",
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]
        engineered = test_set_df[INPUT_COLUMNS]
        from app.services.feature_engineering import add_engineered_features

        engineered = add_engineered_features(engineered)
        processed = model_manager.preprocessor.transform(engineered)
        scores = model_manager.model.predict_proba(processed)[:, 1]
        flagged = scores >= model_manager.threshold
        actual = test_set_df["Machine failure"].to_numpy() == 1

        direct = {
            "tp": int((flagged & actual).sum()),
            "fp": int((flagged & ~actual).sum()),
            "fn": int((~flagged & actual).sum()),
            "tn": int((~flagged & ~actual).sum()),
        }

        # --- Run the full replay through the SimulationService ---
        source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
        monitoring = MonitoringService(
            threshold=model_manager.threshold, total_available=source.total
        )
        simulation = SimulationService(
            source, PredictionService(model_manager), monitoring
        )
        while True:
            response = simulation.step(settings.SIM_BATCH_MAX)
            if response.summary.replay_complete:
                break

        summary = monitoring.summary()
        assert summary.machines_scored == 2000
        assert summary.counts.tp == direct["tp"]
        assert summary.counts.fp == direct["fp"]
        assert summary.counts.fn == direct["fn"]
        assert summary.counts.tn == direct["tn"]

        # --- Compare against the notebook reference numbers ---
        assert (summary.counts.tn, summary.counts.fp, summary.counts.fn, summary.counts.tp) == (
            1927, 5, 14, 54
        )

    def test_replay_is_reproducible(self, settings, model_manager):
        source = CsvReplaySource(settings.TEST_SET_PATH, settings.SIM_SEED)
        monitoring = MonitoringService(
            threshold=model_manager.threshold, total_available=source.total
        )
        simulation = SimulationService(
            source, PredictionService(model_manager), monitoring
        )
        summaries = []
        for _ in range(2):
            while True:
                response = simulation.step(100)
                if response.summary.replay_complete:
                    break
            summaries.append(monitoring.summary())
            simulation.reset()
        assert summaries[0].counts == summaries[1].counts
