# 🛠️ Machine Maintenance Classification

A **full-stack FastAPI application** for **monitoring machine failures** using XGBoost classification, built with a scalable **MVC architecture**. The primary experience is a **monitoring dashboard**: the server replays random batches of machines from a held-out test split (simulating incoming sensor data), scores each batch with the trained model, and the browser polls the server and renders live analytics.

> **Note on the data and scores.** The replay data is the held-out test split of the [AI4I 2020 Predictive Maintenance](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+synthetic+dataset) **synthetic dataset** — the "outcomes" shown on the dashboard are simulated ground truth from that test data. The model was trained with a class-weight setting (`scale_pos_weight`), so its `score` is **not a calibrated probability**; treat it as a risk score.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Monitoring Dashboard Endpoints](#-monitoring-dashboard-endpoints)
- [Prediction API](#-prediction-api-unchanged)
- [Dashboard Screenshot](#-dashboard-screenshot)
- [Testing](#-testing)

---

## 🚀 Project Overview

Predictive maintenance reduces unplanned downtime by anticipating failures before they happen. This application uses **FastAPI** and **XGBoost** to score machines and monitor how the model performs on a simulated stream.

Key components:

✅ MVC architecture with clear separation of concerns  
✅ App factory pattern for scalability and testability  
✅ Feature engineering pipeline (13 derived features)  
✅ Preprocessing pipelines (scaling, encoding)  
✅ Configurable classification threshold (default `0.8966`)  
✅ REST API with versioned endpoints (`/api/v1/`)  
✅ **Batch simulation**: seeded replay of the held-out test set, batch scoring, monitoring state  
✅ **Monitoring dashboard**: KPIs, flagged machines table, score histogram, precision/recall trend, failure-mode breakdown, client-side threshold exploration  
✅ Pydantic validation with type-safe configuration  
✅ Global error handling middleware  
✅ Centralized logging  
✅ Docker Compose configuration for local development

---

## ✨ Features

- **MVC Structure** — Models, Views, Controllers, and Services are cleanly separated.
- **App Factory** — `create_app()` assembles all layers, enabling multiple configurations and easy testing.
- **Monitoring Dashboard** — the only web page (`/`): the browser polls `POST /api/v1/simulation/next` and renders analytics; the view contains no prediction form or manual-input prediction.
- **Batch Simulation** — `CsvReplaySource` replays `Data/test_set.csv` (2,000 held-out rows) without replacement, deterministically (`SIM_SEED`); `SimulationService` orchestrates one tick; `MonitoringService` keeps cumulative metrics in memory (thread-safe, single worker).
- **Batch Prediction** — `PredictionService.predict_batch` scores a whole batch vectorized and computes per-row top-3 contributing factors (via XGBoost SHAP contributions).
- **Client-side threshold exploration** — the dashboard stores the raw `score` of every row, so the threshold slider recomputes flagged rows, KPIs, and charts entirely in the browser without a server call.
- **Pydantic Settings** — Type-safe, validated configuration from `.env` files with caching.
- **ModelManager** — Loads the trained model and preprocessor at application startup.
- **PredictionService** — Business logic decoupled from HTTP — reusable from API, simulation, or CLI.
- **Dark Glassmorphism UI** — Dark theme with animations and responsive design (Chart.js vendored locally).
- **Global Error Handling** — Consistent JSON error responses.
- **Docker Compose** — Runs the application with the project mounted into the container.

---

## 🏗️ Architecture

```
Browser (dashboard only, at /)
  │  polls POST /api/v1/simulation/next?size=N      (one simulated tick)
  │  reads GET  /api/v1/dashboard/{summary,batches,config}
  ▼
API Controller
  │
  ▼
SimulationService (one tick, serialized by a lock)
  ├── CsvReplaySource        → next batch of raw test-set rows (seeded shuffle)
  ├── PredictionService      → predict_batch
  │     ├── 6 input columns (allow-list)
  │     ├── Feature Engineering (13 features)
  │     ├── Preprocessing (preprocessor.pkl)
  │     └── XGBoost Inference (xgb_model.pkl) + top-3 contributions
  └── MonitoringService      → batch history, cumulative confusion counts,
                               mode breakdown  (in-memory, thread-safe)
        │
        ▼
  SimulationStepResponse { batch, summary }  → JSON → dashboard.js renders
```

The replay state lives in the server process memory: **run a single uvicorn worker** (`uvicorn main:app ...` — the default). Multi-worker deployment would split the state across processes and is not supported.

---

## 🗂️ Project Structure

```
Machine-Maintenance-Classification/
│
├── app/
│   ├── __init__.py                    # App factory (create_app) — wires services
│   │
│   ├── controllers/
│   │   ├── api/                       # JSON controllers (health, prediction, dashboard)
│   │   └── web/                       # Web controller — renders the dashboard
│   │
│   ├── core/
│   │   ├── config.py                  # Pydantic settings
│   │   └── logging.py                 # Centralized logging
│   │
│   ├── data_sources/
│   │   └── csv_replay.py              # DataSource protocol + CsvReplaySource
│   │
│   ├── middleware/
│   │   └── error_handler.py           # Global exception handlers
│   │
│   ├── models/
│   │   ├── schemas.py                 # Pydantic request/response schemas
│   │   └── ml_models.py               # ModelManager (load & serve models, top factors)
│   │
│   ├── routes/
│   │   └── router.py                  # Route registration
│   │
│   ├── services/
│   │   ├── prediction_service.py      # Single-row + batch prediction
│   │   ├── monitoring_service.py      # Batch history & cumulative analytics
│   │   ├── simulation_service.py      # One tick: replay → predict → monitor
│   │   └── feature_engineering.py     # 13 derived features
│   │
│   └── views/
│       ├── templates/
│       │   ├── base.html              # Base layout (branding only — no nav links)
│       │   └── dashboard.html         # The monitoring dashboard (only web page)
│       └── static/
│           ├── css/style.css          # Dark theme design system + dashboard styles
│           └── js/
│               ├── dashboard.js       # Polling, client-side metric recomputation
│               └── vendor/chart.umd.js  # Chart.js v4.4.4 (vendored)
│
├── Models/
│   ├── preprocessor.pkl               # Sklearn preprocessing pipeline
│   └── xgb_model.pkl                  # Trained XGBoost classifier
│
├── Data/
│   ├── ai4i2020.csv                   # AI4I 2020 dataset (training source)
│   └── test_set.csv                   # Held-out test split (2,000 rows) — replay source
│
├── Notebook/
│   └── model.ipynb                    # EDA & training notebook (source of the split)
│
├── tests/                             # Test suite (see Testing)
│
├── main.py                            # Entry point
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker build
├── docker-compose.yaml                # Docker orchestration
├── .env.example                       # Example environment variables
└── README.md
```

`Data/test_set.csv` is created by the notebook's `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)` — it contains exactly the 2,000 rows (68 actual failures) the model never saw during training, with all 14 original columns. Targets and failure-mode flags are **never** model inputs (the model receives only the six input columns by allow-list); they are exposed solely as `actual_failure` / `failure_modes` ground truth.

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/Omar9965/Machine-Maintenance-Classification.git
cd Machine-Maintenance-Classification
```

### 2️⃣ Create and activate a virtual environment

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Configure environment variables

Copy the example file to `.env` if you want to configure environment variables. On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

On macOS/Linux:

```bash
cp .env.example .env
```

The app loads these optional settings from the environment or `.env`:

| Variable            | Default                          | Description                                            |
| ------------------- | -------------------------------- | ------------------------------------------------------ |
| `APP_NAME`          | Machine Predictive Maintenance Classification API | Application display name             |
| `VERSION`           | `1.0`                            | Application/API version                                |
| `HOST`              | `0.0.0.0`                        | Configured server host                                 |
| `PORT`              | `8000`                           | Configured server port                                 |
| `LOG_LEVEL`         | `INFO`                           | Logging level                                          |
| `THRESHOLD`         | `0.8966`                         | Score cutoff for flagging a row (initial slider value) |
| `TEST_SET_PATH`     | `<project root>/Data/test_set.csv` | Replay source CSV                                    |
| `SIM_SEED`          | `42`                             | Shuffle seed for reproducible replays                  |
| `SIM_BATCH_SIZE`    | `50`                             | Default batch size                                     |
| `SIM_BATCH_MAX`     | `200`                            | Upper bound the API accepts for a batch                |
| `DASHBOARD_POLL_MS` | `5000`                           | Dashboard poll interval (milliseconds)                 |
| `MODELS_DIR`        | `<project root>/Models`          | Directory containing both model artifacts              |

`HOST` and `PORT` are settings; the local Uvicorn command below explicitly binds to `127.0.0.1:8000`.

The model files `Models/preprocessor.pkl` and `Models/xgb_model.pkl` must be present. The application loads both at startup, so the app and its tests will not start without them. `Data/test_set.csv` (the held-out test split) is also required by the simulation; it is included in the repository.

---

## ▶️ Usage

### Running locally

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

Access:

- 🌐 **Monitoring Dashboard**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — the only web page (no prediction form; `/predict` no longer exists)

> Use a single uvicorn worker. The simulation state (replay cursor, batch history, cumulative counts) is in-memory per process and guarded by a lock; multiple workers would each get their own disconnected state.

### Running with Docker Compose

```bash
docker compose build
docker compose up
```

Docker Compose reads `.env` and mounts the project directory into the container. Create `.env` from `.env.example` first. The Docker build excludes `Models/` and `*.pkl`, so running the image by itself without mounting or providing the model artifacts will fail during startup.

---

## 📡 Monitoring Dashboard Endpoints

All dashboard endpoints are versioned under `/api/v1/`.

| Method | Path                        | Description                                                          |
| ------ | --------------------------- | -------------------------------------------------------------------- |
| POST   | `/api/v1/simulation/next`   | Run one tick: replay a batch, score it, record it. `size` (optional) must be `1 ≤ size ≤ SIM_BATCH_MAX` (HTTP 422 otherwise); defaults to `SIM_BATCH_SIZE`. |
| POST   | `/api/v1/simulation/reset`  | Reset the replay source and monitoring state; returns the empty summary. |
| GET    | `/api/v1/dashboard/summary` | Cumulative analytics: confusion counts, precision, recall, F1, mode breakdown, `replay_complete`. |
| GET    | `/api/v1/dashboard/batches` | Batch history. Query `include_rows=true` also embeds each batch's rows. |
| GET    | `/api/v1/dashboard/config`  | Front-end configuration: threshold, poll interval, batch limits.     |

Example `POST /api/v1/simulation/next?size=2` response (shape only):

```json
{
  "batch": {
    "batch_id": 1,
    "created_at": "2026-01-01T12:00:00Z",
    "size": 2,
    "rows": [
      {
        "udi": 3812,
        "product_id": "L51991",
        "type": "L",
        "air_temperature_k": 302.1,
        "process_temperature_k": 311.4,
        "rotational_speed_rpm": 1325,
        "torque_nm": 61.2,
        "tool_wear_min": 218,
        "score": 0.97,
        "flagged": true,
        "actual_failure": true,
        "failure_modes": ["OSF"],
        "top_factors": [{ "feature": "torque_x_wear", "contribution": 3.1 }]
      }
    ],
    "counts": { "tp": 1, "fp": 0, "fn": 0, "tn": 1 }
  },
  "summary": {
    "threshold": 0.8966,
    "batches_processed": 1,
    "machines_scored": 2,
    "total_available": 2000,
    "counts": { "tp": 1, "fp": 0, "fn": 0, "tn": 1 },
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "mode_breakdown": { "OSF": { "caught": 1, "missed": 0 } },
    "replay_complete": false
  }
}
```

Metric rules: `precision` is `null` when `tp + fp == 0`; `recall` is `null` when `tp + fn == 0`; `f1` is `null` if either is `null` or both are zero. When the replay is exhausted, `POST /api/v1/simulation/next` still returns HTTP 200 with an empty batch and `replay_complete: true`; the dashboard stops polling and offers Restart.

## 🔮 Prediction API (unchanged)

### `GET /api/v1/`

Health check to confirm the API is live.

**Response:**

```json
{
  "message": "Welcome to Machine Predictive Maintenance Classification API",
  "version": "1.0"
}
```

### `POST /api/v1/predict/xgboost`

Accepts single-row machine sensor data, applies feature engineering and preprocessing, and returns:

- `Failure_prediction` — `true` if failure is predicted
- `Failure_probability` — risk score (0.0 – 1.0)

**Request:**

```json
POST /api/v1/predict/xgboost
Content-Type: application/json

{
    "Type": "M",
    "Air temperature [K]": 300.0,
    "Process temperature [K]": 310.0,
    "Rotational speed [rpm]": 1500.0,
    "Torque [Nm]": 40.0,
    "Tool wear [min]": 10.0
}
```

**Response:**

```json
{
  "Failure_prediction": false,
  "Failure_probability": 0.00011
}
```

### Input Schema

| Field                     | Type   | Range         | Description                   |
| ------------------------- | ------ | ------------- | ----------------------------- |
| `Type`                    | string | L, M, H       | Product quality type          |
| `Air temperature [K]`     | float  | 295.3 – 305.4 | Air temperature in Kelvin     |
| `Process temperature [K]` | float  | 305.7 – 313.8 | Process temperature in Kelvin |
| `Rotational speed [rpm]`  | float  | 1168 – 2886   | Rotational speed              |
| `Torque [Nm]`             | float  | 3.8 – 76.6    | Torque                        |
| `Tool wear [min]`         | float  | 0 – 253       | Tool wear time                |

---

## 📸 Dashboard Screenshot

*![Monitoring dashboard — placeholder. Add a screenshot of the running dashboard here.](docs/dashboard.png)*

The dashboard layout:

- **Controls**: Play/Pause, batch size, Reset, threshold slider (client-side recomputation).
- **KPI row**: machines scored, flagged in latest batch, flag rate, cumulative precision, cumulative recall.
- **Flagged machines table** (true positives vs. false alarms) next to the **risk score histogram** with the threshold line.
- **Precision/recall per batch** and **failure-mode breakdown** (caught vs missed, per TWF/HDF/PWF/OSF/RNF — plus `UNKNOWN` when a failure has no recorded mode).

---

## 🧪 Testing

Run the test suite with:

```bash
python -m pytest tests/ -v
```

Tests cover:

- ✅ API health check
- ✅ Prediction with valid input; validation rejection for invalid / missing / out-of-range input
- ✅ `CsvReplaySource` — determinism, no duplicate UDI, full coverage (batches sum to 2,000), exhaustion, reset reproducibility
- ✅ `MonitoringService` — confusion counts, precision/recall/F1 and null handling, mode breakdown incl. `UNKNOWN`, reset
- ✅ `predict_batch` — parity with an independent reference implementation (feature formulas + direct `predict_proba`), parity with the single-row API path, no-leakage checks (targets and failure flags never change scores)
- ✅ Dashboard API — `next` (default/custom size, size validation), summary, batches (with/without rows), config, reset, replay exhaustion, and the golden test: the full replay's cumulative confusion matrix equals direct scoring of the whole test set and the notebook reference `[[1927, 5], [14, 54]]`
- ✅ Web — dashboard renders with no prediction form; removed routes (`GET /predict`, `POST /predict`) return 404

Tests create the real application and load both model artifacts plus `Data/test_set.csv`, so run them from the repository root with those files available.
