# 🛠️ Machine Maintenance Classification

A **full-stack FastAPI application** for **predicting machine failure** using XGBoost classification, built with a scalable **MVC architecture**, feature engineering pipelines, and a modern dark-themed web UI.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Web Pages](#-web-pages)
- [Example Request](#-example-request)
- [Testing](#-testing)

---

## 🚀 Project Overview

Predictive maintenance reduces unplanned downtime by anticipating failures before they happen. This application uses **FastAPI** and **XGBoost** to predict machine failure from sensor and operational data.

Key components:

✅ MVC architecture with clear separation of concerns  
✅ App factory pattern for scalability and testability  
✅ Feature engineering pipeline (14 derived features)  
✅ Preprocessing pipelines (scaling, encoding)  
✅ Tuned classification threshold for precision  
✅ REST API with versioned endpoints (`/api/v1/`)  
✅ Web UI with prediction form and result visualization  
✅ Pydantic validation with type-safe configuration  
✅ Global error handling middleware  
✅ Centralized logging  
✅ Docker Compose configuration for local development

---

## ✨ Features

- **MVC Structure** — Models, Views, Controllers, and Services are cleanly separated.
- **App Factory** — `create_app()` assembles all layers, enabling multiple configurations and easy testing.
- **Dual Interface** — REST API for programmatic access + Web UI for interactive predictions.
- **Pydantic Settings** — Type-safe, validated configuration from `.env` files with caching.
- **ModelManager** — Loads the trained model and preprocessor at application startup.
- **PredictionService** — Business logic decoupled from HTTP — reusable from API, web, or CLI.
- **Modern Web UI** — Dark glassmorphism theme with animations, responsive design, and a probability gauge.
- **Global Error Handling** — Consistent JSON error responses for API, user-friendly error display for web.
- **Docker Compose** — Runs the application with the project mounted into the container.

---

## 🏗️ Architecture

```
Client Request
  ├── API: /api/v1/predict/xgboost (JSON)
  │     └── API Controller
  └── Web: /predict (Form)
        └── Web Controller
              │
              ▼
        PredictionService (Business Logic)
              │
              ├── Feature Engineering (14 features)
              ├── Preprocessing (preprocessor.pkl)
              └── XGBoost Inference (xgb_model.pkl)
                    │
                    ▼
              PredictionResponse
              ├── Failure_prediction: bool
              └── Failure_probability: float
```

---

## 🗂️ Project Structure

```
Machine-Maintenance-Classification/
│
├── app/
│   ├── __init__.py                    # App factory (create_app)
│   │
│   ├── controllers/                   # API and web controllers
│   │   ├── api/
│   │   └── web/
│   │
│   ├── core/
│   │   ├── config.py                  # Pydantic settings
│   │   └── logging.py                 # Centralized logging
│   │
│   ├── middleware/
│   │   └── error_handler.py           # Global exception handlers
│   │
│   ├── models/
│   │   ├── schemas.py                 # Pydantic request/response schemas
│   │   └── ml_models.py               # ModelManager (load & serve models)
│   │
│   ├── routes/
│   │   └── router.py                  # Route registration
│   │
│   ├── services/
│   │   ├── prediction_service.py      # Prediction orchestration
│   │   └── feature_engineering.py     # 14 derived features
│   │
│   └── views/
│   │   ├── templates/
│   │   │   ├── base.html              # Base layout with navigation
│   │   │   ├── index.html             # Landing page
│   │   │   ├── predict.html           # Prediction form
│   │   │   └── result.html            # Result with probability gauge
│   │   └── static/
│   │       ├── css/style.css          # Dark theme design system
│   │       └── js/main.js             # Client-side interactivity
│
├── Models/
│   ├── preprocessor.pkl               # Sklearn preprocessing pipeline
│   └── xgb_model.pkl                  # Trained XGBoost classifier
│
├── Data/
│   └── ai4i2020.csv                   # AI4I 2020 dataset
│
├── Notebook/
│   └── model.ipynb                    # EDA & training notebook
│
├── tests/
│   └── test_api.py                    # API and web endpoint tests
│
├── main.py                            # Entry point
├── requirements.txt                   # Python dependencies
├── Dockerfile                         # Docker build
├── docker-compose.yaml                # Docker orchestration
├── .env.example                       # Example environment variables
└── README.md
```

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

| Variable     | Default                                            | Description                                      |
| ------------ | -------------------------------------------------- | ------------------------------------------------ |
| `APP_NAME`   | Machine Predictive Maintenance Classification API | Application display name                         |
| `VERSION`    | `1.0`                                              | Application/API version                          |
| `HOST`       | `0.0.0.0`                                          | Configured server host                           |
| `PORT`       | `8000`                                             | Configured server port                           |
| `LOG_LEVEL`  | `INFO`                                             | Logging level                                    |
| `MODELS_DIR` | `<project root>/Models`                            | Directory containing both model artifacts       |

`HOST` and `PORT` are settings; the local Uvicorn command below explicitly binds to `127.0.0.1:8000`.

The model files `Models/preprocessor.pkl` and `Models/xgb_model.pkl` must be present. The application loads both at startup, so the app and its tests will not start without them.

---

## ▶️ Usage

### Running locally

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Access:

- 🌐 **Web UI**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 📋 **Prediction Form**: [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)
- **API health check**: [http://127.0.0.1:8000/api/v1/](http://127.0.0.1:8000/api/v1/)

Swagger, ReDoc, and the OpenAPI schema are disabled in the current app configuration.

### Running with Docker Compose

```bash
docker compose build
docker compose up
```

Docker Compose reads `.env` and mounts the project directory into the container. Create `.env` from `.env.example` first. The Docker build excludes `Models/` and `*.pkl`, so running the image by itself without mounting or providing the model artifacts will fail during startup.

---

## 📡 API Endpoints

All API endpoints are versioned under `/api/v1/`.

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

Accepts machine sensor data, applies feature engineering and preprocessing, and returns:

- `Failure_prediction` — `true` if failure is predicted
- `Failure_probability` — probability score (0.0 – 1.0)

---

## 🌐 Web Pages

| Path              | Description                                               |
| ----------------- | --------------------------------------------------------- |
| `/`               | Landing page with project overview and feature highlights |
| `/predict`        | Interactive form to input sensor data                     |
| `/predict` (POST) | Processes form and displays result with probability gauge |

---

## 📝 Example Request

### Request

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

### Response

Example output (the values depend on the input and model artifacts):

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

## 🧪 Testing

Run the test suite with:

```bash
python -m pytest tests/ -v
```

Tests cover:

- ✅ API health check
- ✅ Prediction with valid input
- ✅ Validation rejection for invalid / missing / out-of-range input
- ✅ Web page rendering (home, form, form submission)

Tests create the real application and load both model artifacts, so run them from the repository root with `Models/preprocessor.pkl` and `Models/xgb_model.pkl` available.

---
