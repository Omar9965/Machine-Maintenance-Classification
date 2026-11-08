# AGENTS.md

FastAPI + XGBoost predictive-maintenance service (MVC, app-factory). API is under `/api/v1/`, plus a Jinja2 web UI.

## Commands
- Install: `pip install -r requirements.txt`
- Run: `uvicorn main:app --reload` (from repo root; ports/host from `.env`)
- Test all: `python -m pytest tests/ -v`
- Single test: `python -m pytest tests/test_api.py::TestAPIPrediction::test_predict_valid_input -v`
- Docker: `docker-compose build && docker-compose up`

There is no lint/typecheck/format tooling and no CI — do not invent commands.

## Environment gotchas
- Must run from the repo root: templates/static are registered with relative paths (`app/views/...` in `app/__init__.py:70`, `app/controllers/web/prediction_controller.py:7`).
- `myenv/` is a **Windows** venv (`Scripts/*.exe`) and cannot be used from WSL/Linux. Create a fresh Linux venv.
- Copy `.env.example` (not `env.example` as README says) to `.env`. `.env` is untracked.
- Dockerfile targets Python 3.12.10; `.dockerignore` excludes `Models/`, `Data/`, and `*.pkl`. A plain `docker run` of the built image fails at startup because models are missing. Models are only available via docker-compose's `.:/app` volume mount.

## Architecture (non-obvious)
- `main.py` → `app/__init__.py:create_app()` wires config → logging → error handlers → `ModelManager` → `PredictionService` → routes. Models are loaded eagerly at startup, so any `create_app()` call (including tests) requires `Models/*.pkl`.
- Request flow: controller → `PredictionService.predict` (`app/services/prediction_service.py`) → `add_engineered_features` → `preprocessor.transform` → `model.predict_proba` → threshold.
- Add services to `app.state` and read them in controllers via `request.app.state.*`; there is no FastAPI `Depends` wiring.
- `app/services/feature_engineering.py` adds 14 derived features. These must exactly match `Notebook/model.ipynb` training; changing them (or the input schema) requires retraining and replacing **both** `Models/preprocessor.pkl` and `Models/xgb_model.pkl`.
- Classification threshold is hardcoded `0.8966` in `app/models/ml_models.py:20`, not configurable via env.
- Input fields use dataset aliases with spaces (`"Air temperature [K]"`, etc.); `MachineData` allows both names. Web form and API both use the aliased names.
- `joblib` is imported directly (`app/models/ml_models.py`) but is only present transitively via scikit-learn; it is not pinned in `requirements.txt`.

## Tests
- Fixtures are defined inline in `tests/test_api.py`; there is no `tests/conftest.py` despite the README tree.
- Tests boot the real app and load real model artifacts (no mocking).
- Validation tests expect `422`; supported ranges live in `app/models/schemas.py`.
