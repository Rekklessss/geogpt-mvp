# Multi-stage build for GeoGPT-RAG
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    curl \
    wget \
    git \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies for NLTK
RUN python -c "import nltk; nltk.download('punkt')"

# Copy application code
COPY app/ ./app/
COPY start.sh ./

# Create necessary directories with proper permissions
RUN mkdir -p data/uploads split_chunks logs .cache/transformers .cache/huggingface .cache/torch

# Make start script executable
RUN chmod +x start.sh

# Set proper permissions for nobody user
RUN chown -R nobody:nogroup /app
USER nobody

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["./start.sh"]
