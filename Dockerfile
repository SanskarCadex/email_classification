# Use lightweight Python base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for the application
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    openssh-client \
    tesseract-ocr \
    tesseract-ocr-eng \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Environment variables (fallback defaults - actual values loaded from .env)
ENV PYTHONUNBUFFERED=1 \
    # Email configuration
    MAIL_SEND_ENABLED=False \
    FORCE_DRAFTS=True \
    ADD_EMAIL_FOOTER=true \
    YOUR_DOMAIN=abc-amega.com \
    # System configuration
    MS_GRAPH_TIMEOUT=60 \
    TIME_FILTER_HOURS=24 \
    EMAIL_FETCH_TOP=1000 \
    LOG_LEVEL=INFO \
    LOG_DIR=/app/logs \
    # SFTP configuration
    SFTP_ENABLED=True \
    SFTP_PORT=22

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Ensure logs folder exists with proper permissions
RUN mkdir -p /app/logs && chmod 755 /app/logs

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port for health checks and monitoring
EXPOSE 5000

# Health check optimized for 3-email automation
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Add labels for better container management
LABEL maintainer="ABC/AMEGA Development Team" \
      description="Email Classification System with 3-Email Automation" \
      version="1.1" \
      email-accounts="1" \
      batch-size="100"

# Start the application
CMD ["python", "main.py"]