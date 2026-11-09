from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class MachineData(BaseModel):
    """
    Input schema for machine sensor data used in predictive maintenance.
    Field aliases match the original dataset column names for API compatibility.
    """

    type: Literal["M", "L", "H"] = Field(
        alias="Type",
        description="Product quality type: L (Low), M (Medium), or H (High)",
    )
    air_temperature: float = Field(
        alias="Air temperature [K]",
        description="Air temperature in Kelvin, normalized around 300 K",
        ge=295.3,
        le=305.4,
    )
    process_temperature: float = Field(
        alias="Process temperature [K]",
        description="Process temperature in Kelvin, air temp + ~10 K",
        ge=305.7,
        le=313.8,
    )
    rotational_speed: float = Field(
        alias="Rotational speed [rpm]",
        description="Rotational speed in RPM, derived from 2860 W power",
        ge=1168.0,
        le=2886.0,
    )
    torque: float = Field(
        alias="Torque [Nm]",
        description="Torque in Nm, normally distributed around 40 Nm",
        ge=3.8,
        le=76.6,
    )
    tool_wear: float = Field(
        alias="Tool wear [min]",
        description="Tool wear in minutes, varies by quality type",
        ge=0.0,
        le=253.0,
    )

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "Type": "M",
                "Air temperature [K]": 300.0,
                "Process temperature [K]": 310.0,
                "Rotational speed [rpm]": 1500.0,
                "Torque [Nm]": 40.0,
                "Tool wear [min]": 10.0,
            }
        },
    }


class PredictionResponse(BaseModel):
    """Response schema for prediction results."""

    Failure_prediction: bool = Field(
        description="Whether the model predicts a machine failure"
    )
    Failure_probability: float = Field(
        description="Probability of machine failure (0.0 to 1.0)"
    )


class HealthResponse(BaseModel):
    """Response schema for health check endpoint."""

    message: str
    version: str


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str
    status_code: int


# --- Dashboard / simulation schemas ---


class Factor(BaseModel):
    """A single feature contribution pushing a row toward failure."""

    feature: str
    contribution: float


class BatchRow(BaseModel):
    """One scored machine row within a simulated batch."""

    udi: int = Field(description="Unique machine identifier (UDI)")
    product_id: str = Field(description="Product ID, e.g. L51991")
    type: str = Field(description="Product quality type: L, M or H")
    air_temperature_k: float
    process_temperature_k: float
    rotational_speed_rpm: float
    torque_nm: float
    tool_wear_min: float
    score: float = Field(description="Raw risk score (NOT a calibrated probability)")
    flagged: bool = Field(description="True when score >= threshold")
    actual_failure: bool = Field(description="Simulated outcome from the held-out test data")
    failure_modes: list[str] = Field(
        description="Ground-truth failure modes (TWF/HDF/PWF/OSF/RNF); ['UNKNOWN'] when a "
        "failure occurred but no mode flag was set; empty when the machine did not fail"
    )
    top_factors: list[Factor] = Field(default_factory=list)


class ConfusionCounts(BaseModel):
    """Cumulative confusion-matrix counts."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0


class BatchResult(BaseModel):
    """Result of one simulated batch."""

    batch_id: int
    created_at: datetime
    size: int
    rows: list[BatchRow]
    counts: ConfusionCounts


class ModeBreakdown(BaseModel):
    """Caught vs missed failures for one failure mode."""

    caught: int = 0
    missed: int = 0


class DashboardSummary(BaseModel):
    """Cumulative state of the monitoring simulation."""

    threshold: float
    batches_processed: int
    machines_scored: int
    total_available: int
    counts: ConfusionCounts
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    mode_breakdown: dict[str, ModeBreakdown]
    replay_complete: bool


class SimulationStepResponse(BaseModel):
    """Response for one simulation tick."""

    batch: BatchResult
    summary: DashboardSummary


class BatchHistoryItem(BaseModel):
    """A previously processed batch in compact form."""

    batch_id: int
    size: int
    counts: ConfusionCounts
    precision: float | None = None
    recall: float | None = None
    rows: list[BatchRow] | None = None


class DashboardConfig(BaseModel):
    """Configuration passed to the front end on load."""

    threshold: float
    poll_ms: int
    batch_size_default: int
    batch_size_max: int
