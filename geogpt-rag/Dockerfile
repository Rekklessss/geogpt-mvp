# Multi-stage build for GeoGPT-RAG optimized for g5.xlarge
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 AS base

# Set environment variables optimized for Ubuntu 24.04 Noble + CUDA 12.8
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    CUDA_VISIBLE_DEVICES=0 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 \
    GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 \
    GRPC_PYTHON_BUILD_EXT_COMPILER_JOBS=4

# Install system dependencies optimized for Ubuntu 24.04 Noble
RUN apt-get update && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
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
    cmake \
    ninja-build \
    libc-ares-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python (Ubuntu 24.04 uses Python 3.12)
RUN ln -sf /usr/bin/python3.12 /usr/bin/python \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3

# Skip pip upgrade - system packages are sufficient for Ubuntu 24.04
# RUN python -m pip install --break-system-packages --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY app/requirements.txt .

# Install Python dependencies optimized for Ubuntu 24.04 (prefer binary wheels)
RUN pip install --break-system-packages --no-cache-dir --timeout 1000 \
    --prefer-binary --only-binary=grpcio,protobuf,numpy,scipy \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -r requirements.txt

# Install additional NLTK data required by text splitter
RUN python -c "import nltk; nltk.download('punkt', download_dir='/usr/local/share/nltk_data')"

# Copy application code
COPY app/ ./app/
COPY start.sh ./

# Create necessary directories with proper permissions (including runtime directories)
RUN mkdir -p data/uploads split_chunks logs .cache/transformers .cache/huggingface .cache/torch

# Make start script executable
RUN chmod +x start.sh

# Create nobody user directories and set permissions
RUN chown -R nobody:nogroup /app \
    && chmod -R 755 /app \
    && chmod -R 777 data split_chunks logs .cache \
    && chmod 777 data/uploads

# Switch to nobody user for security
USER nobody

# Health check optimized for model loading time
HEALTHCHECK --interval=60s --timeout=45s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["./start.sh"]
