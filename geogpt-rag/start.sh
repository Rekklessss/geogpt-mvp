#!/bin/bash

# GeoGPT-RAG Startup Script
set -e

echo "🚀 Starting GeoGPT-RAG Application..."

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    log "✅ CUDA available:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    log "⚠️  CUDA not available, using CPU"
fi

# Check Python environment
log "🐍 Python version: $(python --version)"
log "📦 Checking key dependencies..."

# Verify critical imports
python -c "
import torch
import transformers
import sentence_transformers
import pymilvus
import langchain
print('✅ All critical dependencies loaded successfully')
print(f'🔥 PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'🎯 CUDA device count: {torch.cuda.device_count()}')
    print(f'📱 Current CUDA device: {torch.cuda.current_device()}')
"

# Set default environment variables if not provided
export EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-"cuda"}
export RERANKING_DEVICE=${RERANKING_DEVICE:-"cuda"} 
export TEXT_SPLITTER_DEVICE=${TEXT_SPLITTER_DEVICE:-"cuda"}
export LOG_LEVEL=${LOG_LEVEL:-"INFO"}

# Ensure data directories exist
mkdir -p data/uploads split_chunks logs

log "🔧 Environment configuration:"
log "   - Embedding Device: $EMBEDDING_DEVICE"
log "   - Reranking Device: $RERANKING_DEVICE"
log "   - Text Splitter Device: $TEXT_SPLITTER_DEVICE"
log "   - Log Level: $LOG_LEVEL"
log "   - Milvus Collection: ${MILVUS_COLLECTION:-geodocs}"

# Pre-download models if not cached (optional optimization)
if [ "${PRELOAD_MODELS:-false}" = "true" ]; then
    log "📥 Pre-loading models..."
    python -c "
from app.models.embedding import EmbeddingModel
from app.models.reranker import RerankerModel
from app.config import EMBED_MODEL, RERANK_MODEL, BERT_PATH
from transformers import BertTokenizer, BertForNextSentencePrediction

print('Loading embedding model...')
EmbeddingModel(EMBED_MODEL)
print('Loading reranker model...')
RerankerModel(RERANK_MODEL)
print('Loading BERT for text splitting...')
BertTokenizer.from_pretrained(BERT_PATH)
BertForNextSentencePrediction.from_pretrained(BERT_PATH)
print('✅ All models loaded')
"
fi

# Health check function
health_check() {
    log "🔍 Performing startup health check..."
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            log "✅ Application is healthy!"
            return 0
        fi
        log "⏳ Waiting for application to start... ($i/30)"
        sleep 2
    done
    log "❌ Health check failed"
    return 1
}

# Start the application
log "🌟 Launching GeoGPT-RAG API server..."

# Use uvicorn with production settings
exec uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --access-log \
    --log-level info \
    --timeout-keep-alive 30 \
    --timeout-graceful-shutdown 10
