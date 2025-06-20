# Multi-stage Docker build for PhantomHunter production deployment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     PIP_NO_CACHE_DIR=1     PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y     build-essential     curl     git     && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r phantomhunter && useradd -r -g phantomhunter phantomhunter

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest black flake8 jupyter

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R phantomhunter:phantomhunter /app

USER phantomhunter

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY config.py utils.py dataset.py trainer.py __init__.py ./
COPY models/ ./models/
COPY production_utils.py api_server.py ./

# Create directories for logs and checkpoints
RUN mkdir -p /app/logs /app/checkpoints &&     chown -R phantomhunter:phantomhunter /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3     CMD curl -f http://localhost:8000/health || exit 1

USER phantomhunter

EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "api_server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker",      "--bind", "0.0.0.0:8000", "--timeout", "120", "--keepalive", "5",      "--max-requests", "1000", "--max-requests-jitter", "100",      "--access-logfile", "/app/logs/access.log",      "--error-logfile", "/app/logs/error.log",      "--log-level", "info"]

# GPU-enabled production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-production

# Install Python and system dependencies
RUN apt-get update && apt-get install -y     python3.11     python3-pip     curl     && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1     PYTHONDONTWRITEBYTECODE=1     CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN groupadd -r phantomhunter && useradd -r -g phantomhunter phantomhunter

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip3 install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY config.py utils.py dataset.py trainer.py __init__.py ./
COPY models/ ./models/
COPY production_utils.py api_server.py ./

# Create directories and set permissions
RUN mkdir -p /app/logs /app/checkpoints &&     chown -R phantomhunter:phantomhunter /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3     CMD curl -f http://localhost:8000/health || exit 1

USER phantomhunter

EXPOSE 8000

CMD ["gunicorn", "api_server:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker",      "--bind", "0.0.0.0:8000", "--timeout", "180", "--keepalive", "5",      "--max-requests", "500", "--access-logfile", "/app/logs/access.log",      "--error-logfile", "/app/logs/error.log", "--log-level", "info"]