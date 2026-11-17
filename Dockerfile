# Multi-stage lean serving Dockerfile
# Base: python:3.11-slim per CSL236 project specification
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

# Set working directory
WORKDIR /app

# Install minimal runtime dependencies (curl for container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install lean serving dependencies only (no MLflow, no pytest)
COPY requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-serve.txt

# Copy application code, model artifacts, data, and static UI assets
COPY app.py .
COPY model/ ./model/
COPY static/ ./static/
COPY data/ ./data/

# Create non-root user for container security
RUN addgroup --system appgroup && adduser --system --group appuser && \
    chown -R appuser:appgroup /app
USER appuser

# Expose HTTP port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production serving command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
