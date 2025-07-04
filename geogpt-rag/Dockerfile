# Multi-stage build for GeoGPT-RAG optimized for g5.xlarge
FROM nvidia/cuda:12.1-devel-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

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
    software-properties-common \
    ca-certificates \
    lsb-release \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt .

# Install Python dependencies with better error handling
RUN pip install --no-cache-dir --timeout 1000 -r requirements.txt

# Install additional NLTK data required by text splitter
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Copy application code
COPY app/ ./app/
COPY start.sh ./

# Create necessary directories with proper permissions
RUN mkdir -p data/uploads split_chunks logs .cache/transformers .cache/huggingface .cache/torch \
    && chmod -R 755 data split_chunks logs .cache

# Make start script executable
RUN chmod +x start.sh

# Create nobody user directories and set permissions
RUN chown -R nobody:nogroup /app \
    && chmod -R 755 /app

# Switch to nobody user for security
USER nobody

# Health check optimized for model loading time
HEALTHCHECK --interval=60s --timeout=45s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["./start.sh"]
