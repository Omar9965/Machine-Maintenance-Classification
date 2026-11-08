from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    APP_NAME: str = "Machine Predictive Maintenance Classification API"
    VERSION: str = "1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    THRESHOLD: float = 0.8966

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    MODELS_DIR: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.MODELS_DIR:
            self.MODELS_DIR = os.path.join(self.BASE_DIR, "Models")


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — created once, reused everywhere."""
    return Settings()
