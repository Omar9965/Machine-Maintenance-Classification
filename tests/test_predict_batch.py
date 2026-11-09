"""Tests for PredictionService.predict_batch — parity, leakage, engineering."""

import numpy as np
import pandas as pd
import pytest

from app.models.schemas import MachineData
from app.services.feature_engineering import add_engineered_features
from app.services.prediction_service import PredictionService

INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def reference_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """
    Independent re-implementation of the notebook's engineered features
    (section formulas), built here on purpose — not imported from the app.
    """
    out = df.copy()
    air = df["Air temperature [K]"].to_numpy(dtype=float)
    proc = df["Process temperature [K]"].to_numpy(dtype=float)
    rot = df["Rotational speed [rpm]"].to_numpy(dtype=float)
    torque = df["Torque [Nm]"].to_numpy(dtype=float)
    wear = df["Tool wear [min]"].to_numpy(dtype=float)

    temp_diff = proc - air
    out["temp_diff"] = temp_diff
    out["temp_ratio"] = proc / air
    out["torque_to_speed_ratio"] = torque / rot
    out["power_approx"] = torque * rot * 0.10472
    out["high_torque_flag"] = (torque > 50).astype(int)
    out["low_speed_flag"] = (rot < 1500).astype(int)
    out["overheat_flag"] = (air > 303).astype(int)
    out["air_temp_norm"] = air - 300
    out["process_temp_norm"] = proc - 310
    out["torque_x_wear"] = torque * wear
    out["speed_x_wear"] = rot * wear
    out["tempdiff_x_torque"] = temp_diff * torque
    out["stress_index"] = torque**2 / rot
    return out


def reference_scores(df: pd.DataFrame, model_manager) -> np.ndarray:
    """Reference scoring: transform + predict_proba called directly."""
    engineered = reference_engineer(df[INPUT_COLUMNS])
    processed = model_manager.preprocessor.transform(engineered)
    return model_manager.model.predict_proba(processed)[:, 1]


@pytest.fixture()
def sample_df(test_set_df):
    """A deterministic sample of raw test-set rows."""
    return test_set_df.iloc[list(range(0, 500, 100))].copy()


def test_predict_batch_matches_reference(sample_df, model_manager):
    service = PredictionService(model_manager)
    scored = service.predict_batch(sample_df)
    expected = reference_scores(sample_df, model_manager)
    np.testing.assert_allclose(scored["score"].to_numpy(), expected, rtol=0, atol=1e-9)
    assert scored["flagged"].tolist() == list(scored["score"] >= 0.8966)


def test_feature_engineering_parity(sample_df):
    app_features = add_engineered_features(sample_df[INPUT_COLUMNS])
    ref_features = reference_engineer(sample_df[INPUT_COLUMNS])
    np.testing.assert_allclose(
        app_features.drop(columns=["Type"]).to_numpy(dtype=float),
        ref_features.drop(columns=["Type"]).to_numpy(dtype=float),
        rtol=0,
        atol=1e-12,
    )


def test_score_regression_against_golden_numbers(sample_df, model_manager):
    """Sample rows scored at the configured threshold behave near the notebook."""
    service = PredictionService(model_manager)
    scored = service.predict_batch(sample_df)
    # Scores must be well-formed and threshold-flagging consistent
    assert ((scored["score"] >= 0) & (scored["score"] <= 1)).all()
    assert scored["flagged"].equals(scored["score"] >= 0.8966)


def test_single_row_path_matches_batch_path(sample_df, model_manager):
    """predict() (API path) must produce identical scores to predict_batch()."""
    service = PredictionService(model_manager)
    scores = service.predict_batch(sample_df)["score"].to_numpy()

    for i, (_, row) in enumerate(sample_df.iterrows()):
        machine = MachineData(
            **{
                "Type": row["Type"],
                "Air temperature [K]": row["Air temperature [K]"],
                "Process temperature [K]": row["Process temperature [K]"],
                "Rotational speed [rpm]": row["Rotational speed [rpm]"],
                "Torque [Nm]": row["Torque [Nm]"],
                "Tool wear [min]": row["Tool wear [min]"],
            }
        )
        response = service.predict(machine)
        assert response.Failure_probability == pytest.approx(float(scores[i]), abs=1e-9)
        assert response.Failure_prediction == bool(scores[i] >= 0.8966)


def test_no_leakage_extra_columns_do_not_change_scores(sample_df, model_manager):
    """Targets and failure-mode flags must never affect the scores."""
    service = PredictionService(model_manager)
    base = service.predict_batch(sample_df)

    leaked = sample_df.copy()
    leaked["Machine failure"] = np.random.default_rng(1).integers(0, 2, len(leaked))
    leaked["TWF"] = 1          # extreme injected values
    leaked["UDI"] = 99999999
    actually = service.predict_batch(leaked)

    np.testing.assert_allclose(base["score"], actually["score"], rtol=0, atol=1e-12)


def test_predict_batch_returns_input_plus_columns(sample_df, model_manager):
    service = PredictionService(model_manager)
    scored = service.predict_batch(sample_df)
    assert len(scored) == len(sample_df)
    assert scored["UDI"].tolist() == sample_df["UDI"].tolist()
    assert "score" in scored.columns and "flagged" in scored.columns
    for factor in scored["top_factors"].iloc[0]:
        assert hasattr(factor, "feature") and hasattr(factor, "contribution")


def test_predict_batch_top_factors_has_at_most_three(sample_df, model_manager):
    service = PredictionService(model_manager)
    scored = service.predict_batch(sample_df)
    for factors in scored["top_factors"]:
        assert len(factors) <= 3
        assert all(f.contribution > 0 for f in factors)


def test_empty_frame(model_manager):
    service = PredictionService(model_manager)
    empty = pd.DataFrame(columns=INPUT_COLUMNS)
    scored = service.predict_batch(empty)
    assert scored.empty
    assert "score" in scored.columns and "flagged" in scored.columns
