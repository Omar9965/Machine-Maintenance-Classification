import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app import create_app
from app.core.config import get_settings


@pytest.fixture(scope="session")
def settings():
    return get_settings()


@pytest.fixture(scope="session")
def model_manager():
    """Loaded ModelManager — real model artifacts (run from the repo root)."""
    from app.models.ml_models import ModelManager

    manager = ModelManager(get_settings().MODELS_DIR)
    manager.load_models()
    return manager


@pytest.fixture(scope="session")
def test_set_df() -> pd.DataFrame:
    """The held-out test split as a raw DataFrame (all 14 columns)."""
    return pd.read_csv(get_settings().TEST_SET_PATH)


@pytest.fixture()
def client():
    """Fresh app per test — each test starts with clean simulation state."""
    app = create_app()
    with TestClient(app) as c:
        yield c
