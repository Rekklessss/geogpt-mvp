#!/bin/bash

# =====================================================================================
# GeoGPT-RAG Deployment Verification Script
# Tests all components after fresh deployment
# =====================================================================================

log() {
    echo -e "\033[32m[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1\033[0m"
}

error() {
    echo -e "\033[31m[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1\033[0m"
}

echo "🔍 GeoGPT-RAG Deployment Verification"
echo "====================================="

# Test 1: Docker containers
log "📦 Checking Docker containers..."
if docker compose ps | grep -q "Up"; then
    log "✅ Containers are running"
    docker compose ps
else
    error "❌ Containers not running"
    exit 1
fi

# Test 2: CUDA access
log "🚀 Checking CUDA access..."
if docker exec geogpt-rag-geogpt-rag-1 nvidia-smi &>/dev/null; then
    log "✅ CUDA accessible"
    docker exec geogpt-rag-geogpt-rag-1 nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    error "❌ CUDA not accessible"
fi

# Test 3: Python/PyTorch CUDA
log "🐍 Checking PyTorch CUDA integration..."
docker exec geogpt-rag-geogpt-rag-1 python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

# Test 4: API Health
log "🌐 Testing API health..."
sleep 5  # Wait for startup
if curl -sf http://localhost:8000/health >/dev/null; then
    log "✅ API is healthy"
    curl -s http://localhost:8000/health | jq .
else
    error "❌ API health check failed"
fi

# Test 5: Model loading
log "🤖 Testing model endpoints..."

# Test embedding endpoint
log "Testing embeddings..."
curl -sf http://localhost:8000/embed/test >/dev/null && log "✅ Embeddings working" || error "❌ Embeddings failed"

# Test LLM endpoint
log "Testing LLM connection..."
curl -sf http://localhost:8000/llm/test >/dev/null && log "✅ LLM working" || error "❌ LLM failed"

# Test 6: Memory usage
log "💾 Checking memory usage..."
docker exec geogpt-rag-geogpt-rag-1 python -c "
import psutil
import torch
print(f'System RAM: {psutil.virtual_memory().percent}% used')
if torch.cuda.is_available():
    print(f'GPU memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f}GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"

# Test 7: Logs check
log "📋 Checking for errors in logs..."
if docker compose logs --tail=50 | grep -i "error\|failed\|exception" | grep -v "test"; then
    error "⚠️  Found errors in logs"
else
    log "✅ No critical errors in logs"
fi

# Final summary
echo
log "🎉 Verification completed!"
log "🌐 Access your API at:"
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com/ || echo "localhost")
log "   - API: http://$PUBLIC_IP:8000"
log "   - Docs: http://$PUBLIC_IP:8000/docs"
log "   - Health: http://$PUBLIC_IP:8000/health" 