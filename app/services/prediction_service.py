import pandas as pd
from app.models.schemas import MachineData, PredictionResponse
from app.models.ml_models import ModelManager
from app.services.feature_engineering import add_engineered_features
from app.core.logging import get_logger

logger = get_logger(__name__)

INPUT_COLUMNS = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


class PredictionService:
    """
    Orchestrates the full prediction pipeline.

    Decoupled from HTTP — can be used from API controllers,
    web controllers, CLI tools, or scheduled jobs.
    """

    def __init__(self, model_manager: ModelManager):
        self._model_manager = model_manager

    def predict(self, data: MachineData) -> PredictionResponse:
        """
        Run a prediction for the given machine data.

        Thin wrapper over :meth:`predict_batch` so there is a single code path.
        """
        logger.info("Starting prediction for input: Type=%s", data.type)

        df = pd.DataFrame([data.model_dump(by_alias=True)])
        scored = self.predict_batch(df)

        failure_probability = float(scored["score"].iloc[0])
        failure_prediction = bool(scored["flagged"].iloc[0])

        logger.info(
            "Prediction complete: probability=%.4f, prediction=%s",
            failure_probability,
            failure_prediction,
        )

        return PredictionResponse(
            Failure_prediction=failure_prediction,
            Failure_probability=failure_probability,
        )

    def predict_batch(
        self, df: pd.DataFrame, threshold: float | None = None
    ) -> pd.DataFrame:
        """
        Score a batch of machines in one vectorized pass.

        Pipeline: select the six input columns (allow-list) → feature
        engineering → preprocessing → model inference → threshold.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least the six model input columns. Extra columns
            (e.g. ``UDI``, ``Machine failure``) are passed through untouched
            and never used as model inputs.
        threshold : float, optional
            Override the configured decision threshold. A row is ``flagged``
            when its score is greater than or equal to the threshold.

        Returns
        -------
        pd.DataFrame
            The input frame plus ``score`` (raw risk score, not a calibrated
            probability), ``flagged`` and ``top_factors`` columns.
        """
        if threshold is None:
            threshold = self._model_manager.threshold

        scored = df.copy()

        if scored.empty:
            scored["score"] = pd.Series(dtype=float)
            scored["flagged"] = pd.Series(dtype=bool)
            scored["top_factors"] = pd.Series(dtype=object)
            return scored

        # 1. Select the six model inputs by allow-list (never "drop the others":
        #    target and failure-flag columns must not leak into the model).
        X = scored[INPUT_COLUMNS]

        # 2. Add engineered features
        X_engineered = add_engineered_features(X)

        # 3. Apply preprocessing pipeline
        X_processed = self._model_manager.preprocessor.transform(X_engineered)

        # 4. Score (vectorized) and threshold
        scores = self._model_manager.score(X_processed)
        scored["score"] = scores
        scored["flagged"] = scores >= threshold

        # 5. Per-row top feature contributions
        scored["top_factors"] = self._model_manager.top_factors(X_processed)

        logger.info(
            "Batch prediction complete: rows=%d, flagged=%d",
            len(scored),
            int(scored["flagged"].sum()),
        )
        return scored
