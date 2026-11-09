import pytest
from fastapi.testclient import TestClient
from app import create_app


@pytest.fixture(scope="module")
def client():
    """Create a test client using the app factory."""
    app = create_app()
    with TestClient(app) as c:
        yield c


# --- API Tests ---

class TestAPIHealth:
    def test_health_check(self, client):
        response = client.get("/api/v1/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestAPIPrediction:
    VALID_PAYLOAD = {
        "Type": "M",
        "Air temperature [K]": 300.0,
        "Process temperature [K]": 310.0,
        "Rotational speed [rpm]": 1500.0,
        "Torque [Nm]": 40.0,
        "Tool wear [min]": 10.0,
    }

    def test_predict_valid_input(self, client):
        response = client.post("/api/v1/predict/xgboost", json=self.VALID_PAYLOAD)
        assert response.status_code == 200
        data = response.json()
        assert "Failure_prediction" in data
        assert "Failure_probability" in data
        assert isinstance(data["Failure_prediction"], bool)
        assert 0.0 <= data["Failure_probability"] <= 1.0

    def test_predict_invalid_input(self, client):
        response = client.post("/api/v1/predict/xgboost", json={"Type": "X"})
        assert response.status_code == 422

    def test_predict_missing_fields(self, client):
        response = client.post("/api/v1/predict/xgboost", json={})
        assert response.status_code == 422

    def test_predict_out_of_range(self, client):
        payload = self.VALID_PAYLOAD.copy()
        payload["Air temperature [K]"] = 999.0  # Way out of range
        response = client.post("/api/v1/predict/xgboost", json=payload)
        assert response.status_code == 422


# --- Web Tests ---

class TestWebDashboard:
    def test_dashboard_renders(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
