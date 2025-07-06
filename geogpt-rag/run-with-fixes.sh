#!/bin/bash

# GeoGPT-RAG Startup Script with Compatibility Fixes
# ===================================================

set -e

echo "🚀 Starting GeoGPT-RAG with compatibility fixes..."

# Function to print colored output
print_status() {
    local color=$1
    local message=$2
    case $color in
        "green") echo -e "\033[32m$message\033[0m" ;;
        "yellow") echo -e "\033[33m$message\033[0m" ;;
        "red") echo -e "\033[31m$message\033[0m" ;;
        *) echo "$message" ;;
    esac
}

# Apply compatibility fixes
print_status "yellow" "🔧 Applying compatibility fixes..."
if python3 embedding-compatibility-fix.py; then
    print_status "green" "✅ Compatibility fixes applied successfully"
else
    print_status "red" "❌ Some compatibility fixes failed, but continuing..."
fi

# Set additional environment variables for compatibility
export TF_CPP_MIN_LOG_LEVEL=2
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_LAUNCH_BLOCKING=0

# Reduce batch sizes to avoid OOM issues
export EMBEDDING_BATCH_SIZE=16
export RERANKING_BATCH_SIZE=16
export VEC_RECALL_NUM=64

print_status "green" "🔧 Environment variables configured"

# Check if Python path is set correctly
if [ -z "$PYTHONPATH" ]; then
    export PYTHONPATH=/app
    print_status "yellow" "⚠️  PYTHONPATH not set, defaulting to /app"
fi

# Start the application
print_status "green" "🚀 Starting GeoGPT-RAG application..."
cd /app

# Run the FastAPI application with Uvicorn
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info \
    --access-log \
    --use-colors 