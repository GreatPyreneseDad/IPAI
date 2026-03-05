# IPAI System Docker Configuration
# Multi-stage build for production deployment

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libpq-dev \
    gcc \
    g++ \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -r ipai && useradd -r -g ipai ipai

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application directory
WORKDIR /app

# Copy application code
COPY --chown=ipai:ipai . .

# Create necessary directories
RUN mkdir -p /app/logs /app/uploads /app/models \
    && chown -R ipai:ipai /app

# Switch to non-root user
USER ipai

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage
FROM builder as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    htop \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Set development environment
ENV ENVIRONMENT=development \
    DEBUG=true \
    RELOAD=true

# Create development user with sudo access
RUN apt-get update && apt-get install -y sudo \
    && echo "ipai ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Development command with hot reload
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "debug"]

# Testing stage
FROM development as testing

# Copy test configuration
COPY pytest.ini .
COPY tests/ tests/

# Set testing environment
ENV ENVIRONMENT=testing \
    TESTING=true

# Run tests by default
CMD ["pytest", "tests/", "-v", "--cov=src", "--cov-report=html", "--cov-report=term"]

# GPU-enabled stage (optional)
FROM nvidia/cuda:12.1-runtime-ubuntu22.04 as gpu-production

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install with GPU support
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy application
WORKDIR /app
COPY . .

# Create non-root user
RUN groupadd -r ipai && useradd -r -g ipai ipai
RUN chown -R ipai:ipai /app

USER ipai

# GPU-optimized command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]