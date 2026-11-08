import os
import joblib
from app.core.logging import get_logger
from app.core.config import get_settings

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
