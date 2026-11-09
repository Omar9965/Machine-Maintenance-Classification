import os
import joblib
import numpy as np
import xgboost

from app.core.logging import get_logger
from app.core.config import get_settings
from app.models.schemas import Factor

threshold = get_settings().THRESHOLD

logger = get_logger(__name__)


class ModelManager:
    """
    Manages the lifecycle of ML models and preprocessing pipelines.

    Loads models once at startup and provides access to them throughout
    the application via dependency injection.
    """

    def __init__(self, models_dir: str):
        self._models_dir = models_dir
        self._preprocessor = None
        self._model = None
        self._threshold: float = threshold

    def load_models(self) -> None:
        """Load all ML artifacts from disk."""
        preprocessor_path = os.path.join(self._models_dir, "preprocessor.pkl")
        model_path = os.path.join(self._models_dir, "xgb_model.pkl")

        logger.info("Loading preprocessor from %s", preprocessor_path)
        self._preprocessor = joblib.load(preprocessor_path)

        logger.info("Loading XGBoost model from %s", model_path)
        self._model = joblib.load(model_path)

        logger.info("All models loaded successfully")

    @property
    def preprocessor(self):
        """Sklearn preprocessing pipeline."""
        if self._preprocessor is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._preprocessor

    @property
    def model(self):
        """Trained XGBoost classifier."""
        if self._model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._model

    @property
    def threshold(self) -> float:
        """Tuned classification threshold."""
        return self._threshold

    def get_feature_names(self) -> list[str]:
        """
        Output feature names of the preprocessor with the ``num__`` / ``cat__``
        prefixes stripped.
        """
        names = self._preprocessor.get_feature_names_out().tolist()
        return [name.split("__", 1)[1] if "__" in name else name for name in names]

    def top_factors(self, X_processed, k: int = 3) -> list[list[Factor]]:
        """
        Compute per-row top-k positive feature contributions (SHAP values).

        Uses ``pred_contribs`` on the underlying booster; the bias term is
        dropped. Only the features contributing the most toward "failure"
        (largest positive contributions) are returned.

        Returns one ``list[Factor]`` per row; empty lists if the contribution
        values cannot be computed (logged as a warning, never raised).

        Parameters
        ----------
        X_processed : array or sparse matrix
            The preprocessed feature matrix (output of ``preprocessor.transform``).
        k : int
            Maximum number of factors to return per row.
        """
        try:
            booster = self._model.get_booster()
            contribs = booster.predict(xgboost.DMatrix(X_processed), pred_contribs=True)
        except Exception as exc:
            logger.warning("Could not compute feature contributions: %s", exc)
            return [[] for _ in range(X_processed.shape[0])]

        names = self.get_feature_names()
        results: list[list[Factor]] = []
        for row in contribs:
            values = np.asarray(row, dtype=float)[:-1]  # drop the bias term
            top_idx = np.argsort(values)[::-1][:k]
            results.append(
                [
                    Factor(feature=names[i], contribution=float(values[i]))
                    for i in top_idx
                    if values[i] > 0
                ]
            )
        return results

    def score(self, X_processed) -> np.ndarray:
        """
        Vectorized scoring: raw failure risk scores for the preprocessed matrix.

        The returned scores are NOT calibrated probabilities — the model was
        trained with a class-weight setting that inflates them.
        """
        return self._model.predict_proba(X_processed)[:, 1]
