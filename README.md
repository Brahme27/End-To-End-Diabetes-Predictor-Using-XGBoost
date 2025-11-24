# Diabetes Risk Predictor

## Overview
A comprehensive machine learning application designed to predict the risk of diabetes based on health metrics. This project integrates a robust **XGBoost** model with a **FastAPI** backend and an interactive **Streamlit** frontend, providing real-time predictions with **SHAP** (SHapley Additive exPlanations) for model interpretability.

## Features
- **Real-time Prediction**: Instant diabetes risk assessment using a trained XGBoost classifier.
- **Model Explainability**: Visual SHAP waterfall plots to understand which factors contributed most to the prediction.
- **Interactive UI**: Clean and intuitive web interface built with Streamlit.
- **RESTful API**: High-performance backend API powered by FastAPI.
- **Dockerized**: Fully containerized application for consistent deployment across environments.

## Tech Stack
- **Machine Learning**: XGBoost, Scikit-learn, SHAP, Pandas, NumPy
- **Backend**: FastAPI, Uvicorn, Pydantic
- **Frontend**: Streamlit, Plotly
- **DevOps & Tools**: Docker, Docker Compose, UV (Modern Python package installer)

## Quick Start (Docker)
The simplest way to run the application is using Docker Compose.

### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your machine.

### Steps
1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository-url>
    cd "Diabetes ML Project"
    ```

2.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the Application**:
    -   **Frontend Dashboard**: [http://localhost:8501](http://localhost:8501)
    -   **API Documentation (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)

## Local Development
If you prefer to run the services locally without Docker, ensure you have Python 3.10+ and `uv` installed.

1.  **Install `uv`**:
    ```bash
    # On Windows (PowerShell)
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    # On macOS/Linux
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Navigate to the source directory**:
    ```bash
    cd diabetes_predictor
    ```

3.  **Install Dependencies**:
    ```bash
    uv sync
    ```

4.  **Run the Backend**:
    ```bash
    uv run uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
    ```

5.  **Run the Frontend** (in a new terminal):
    ```bash
    uv run streamlit run frontend/streamlit_app.py
    ```

## Project Structure
```text
Diabetes ML Project/
├── diabetes_predictor/
│   ├── backend/            # FastAPI application logic
│   │   ├── app/            # API routes and main app
│   │   └── models/         # Trained ML models
│   ├── frontend/           # Streamlit user interface
│   ├── ml_pipeline/        # Scripts for data processing and training
│   ├── pyproject.toml      # Project dependencies
│   └── uv.lock             # Dependency lock file
├── Dockerfile              # Multi-stage Docker build definition
└── docker-compose.yml      # Docker Compose configuration
```

## API Endpoints
-   `GET /health`: Check API health status.
-   `POST /predict`: Submit health metrics to get a diabetes risk prediction and SHAP values.
-   