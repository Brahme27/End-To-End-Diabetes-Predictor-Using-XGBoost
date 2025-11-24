# Multi-stage Dockerfile for Diabetes ML Project
# This builds both backend (FastAPI) and frontend (Streamlit) services using uv

# ============================================
# Stage 1: Backend (FastAPI)
# ============================================
FROM python:3.10-slim AS backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY diabetes_predictor/pyproject.toml ./
COPY diabetes_predictor/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy backend application code
COPY diabetes_predictor/backend/app/ ./backend/app/
COPY diabetes_predictor/backend/models/ ./backend/models/

# Create non-root user for security
RUN useradd -m -u 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run FastAPI with uvicorn using uv run
CMD ["uv", "run", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# ============================================
# Stage 2: Frontend (Streamlit)
# ============================================
FROM python:3.10-slim AS frontend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy dependency files
COPY diabetes_predictor/pyproject.toml ./
COPY diabetes_predictor/uv.lock ./

# Install dependencies using uv
RUN uv sync --frozen --no-dev

# Copy frontend application files
COPY diabetes_predictor/frontend/streamlit_app.py ./frontend/

# Create non-root user
RUN useradd -m -u 1000 streamlituser && chown -R streamlituser:streamlituser /app

USER streamlituser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit using uv run
CMD ["uv", "run", "streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
