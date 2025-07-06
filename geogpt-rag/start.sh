#!/bin/bash

# =====================================================================================
# GeoGPT-RAG Production Startup Script for g5.xlarge
# Optimized for AWS EC2 deployment with comprehensive validation
# =====================================================================================

set -e  # Exit on any error

echo "🚀 Starting GeoGPT-RAG Application..."

# Enhanced logging function with different levels
log() {
    echo -e "\033[32m[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1\033[0m"
}

warn() {
    echo -e "\033[33m[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1\033[0m"
}

error() {
    echo -e "\033[31m[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1\033[0m"
    exit 1
}

# Validate critical environment variables
validate_environment() {
    log "🔧 Validating environment configuration..."
    
    # Check for critical missing variables
    local missing_vars=()
    
    if [ -z "$ZILLIZ_URI" ] || [ "$ZILLIZ_URI" = "https://your-cluster.vectordb.zilliz.com:19530" ]; then
        missing_vars+=("ZILLIZ_URI")
    fi
    
    if [ -z "$ZILLIZ_TOKEN" ] || [ "$ZILLIZ_TOKEN" = "your_zilliz_token_here" ]; then
        missing_vars+=("ZILLIZ_TOKEN")
    fi
    
    if [ "$LLM_PROVIDER" = "sagemaker" ]; then
        if [ -z "$SAGEMAKER_ENDPOINT_NAME" ] || [ "$SAGEMAKER_ENDPOINT_NAME" = "your-geogpt-llm-endpoint-name" ]; then
            missing_vars+=("SAGEMAKER_ENDPOINT_NAME")
        fi
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        error "Missing critical environment variables: ${missing_vars[*]}"
    fi
    
    log "✅ Environment validation passed"
}

# Check GPU and CUDA availability
check_gpu() {
    log "🚀 Checking GPU availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        log "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        
        # Verify CUDA is accessible from Python
        python -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'✅ CUDA accessible: {gpu_name} ({gpu_memory:.1f}GB)')
else:
    print('⚠️  CUDA not accessible from Python')
" || warn "CUDA check failed, will use CPU fallback"
    else
        warn "⚠️  NVIDIA GPU not detected, using CPU mode"
        export EMBEDDING_DEVICE="cpu"
        export RERANKING_DEVICE="cpu"
        export TEXT_SPLITTER_DEVICE="cpu"
    fi
}

# Validate Python environment and dependencies
check_dependencies() {
    log "🐍 Checking Python environment..."
    python --version || error "Python not available"
    
    log "📦 Validating critical dependencies..."
    python -c "
import sys
required_packages = [
    'torch', 'transformers', 'sentence_transformers', 
    'pymilvus', 'langchain', 'fastapi', 'uvicorn',
    'pydantic', 'nltk', 'requests'
]

missing = []
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing.append(package)

if missing:
    print(f'❌ Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All critical dependencies available')
" || error "Dependency check failed"
}

# Set and validate environment variables
setup_environment() {
    log "⚙️ Setting up environment variables..."
    
    # Set default values for optional variables
    export EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-"cuda"}
    export RERANKING_DEVICE=${RERANKING_DEVICE:-"cuda"} 
    export TEXT_SPLITTER_DEVICE=${TEXT_SPLITTER_DEVICE:-"cuda"}
    export LOG_LEVEL=${LOG_LEVEL:-"INFO"}
    export MILVUS_COLLECTION=${MILVUS_COLLECTION:-"geodocs"}
    export LLM_PROVIDER=${LLM_PROVIDER:-"sagemaker"}
    
    # G5.xlarge optimizations
    export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"max_split_size_mb:2048,expandable_segments:True"}
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-"false"}
    export OMP_NUM_THREADS=${OMP_NUM_THREADS:-"4"}
    export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-"0"}
    
    # Performance optimizations for A10G
    export TORCH_CUDNN_V8_API_ENABLED=${TORCH_CUDNN_V8_API_ENABLED:-"1"}
    export TORCH_CUDNN_BENCHMARK=${TORCH_CUDNN_BENCHMARK:-"1"}
    
    log "🔧 Environment configuration:"
    log "   - LLM Provider: $LLM_PROVIDER"
    log "   - Embedding Device: $EMBEDDING_DEVICE"
    log "   - Reranking Device: $RERANKING_DEVICE"
    log "   - Text Splitter Device: $TEXT_SPLITTER_DEVICE"
    log "   - Log Level: $LOG_LEVEL"
    log "   - Collection: $MILVUS_COLLECTION"
}

# Ensure required directories exist
setup_directories() {
    log "📁 Verifying data directories..."
    
    # Create directories if they don't exist (permissions already set in Dockerfile)
    mkdir -p data/uploads split_chunks logs .cache/transformers .cache/huggingface .cache/torch
    
    # Note: Permissions are set in Dockerfile - don't try to chmod as nobody user
    ls -la data/ split_chunks/ logs/ .cache/ || log "⚠️  Some directories not found but will be created by application"
    log "✅ Directories verified"
}

# Download NLTK data if needed
setup_nltk() {
    log "📚 Setting up NLTK data..."
    python -c "
import nltk
import os
try:
    nltk.data.find('tokenizers/punkt')
    print('✅ NLTK punkt tokenizer available')
except LookupError:
    print('📥 Downloading NLTK punkt tokenizer...')
    nltk.download('punkt', download_dir=os.getenv('NLTK_DATA', '/usr/local/share/nltk_data'))
    print('✅ NLTK punkt tokenizer downloaded')
" || warn "NLTK setup had issues, continuing anyway"
}

# Test SageMaker connectivity if configured
test_sagemaker() {
    if [ "$LLM_PROVIDER" = "sagemaker" ] && [ -n "$SAGEMAKER_ENDPOINT_NAME" ]; then
        log "🔗 Testing SageMaker endpoint connectivity..."
        python -c "
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import os

try:
    session = boto3.Session()
    client = session.client('sagemaker-runtime', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    
    # Try to describe the endpoint (doesn't invoke it, just checks if it exists)
    sagemaker_client = session.client('sagemaker', region_name=os.getenv('AWS_REGION', 'us-east-1'))
    endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME')
    
    response = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    status = response['EndpointStatus']
    print(f'✅ SageMaker endpoint {endpoint_name} status: {status}')
    
    if status != 'InService':
        print(f'⚠️  Endpoint not in service, current status: {status}')
    
except NoCredentialsError:
    print('⚠️  AWS credentials not configured, will use runtime discovery')
except ClientError as e:
    print(f'⚠️  SageMaker endpoint check failed: {e}')
except Exception as e:
    print(f'⚠️  SageMaker connectivity test error: {e}')
" || warn "SageMaker connectivity test failed"
    else
        log "⚠️  SageMaker not configured or different LLM provider selected"
    fi
}

# Pre-load models with error handling
preload_models() {
    if [ "${PRELOAD_MODELS:-false}" = "true" ]; then
        log "📥 Pre-loading models (this may take 2-3 minutes)..."
        
        python -c "
import os
import sys
sys.path.insert(0, '/app')

try:
    from app.config import EMBED_MODEL, RERANK_MODEL, BERT_PATH
    
    print('🔄 Loading embedding model...')
    from app.models.embedding import EmbeddingModel
    embedding_model = EmbeddingModel(
        model_name=EMBED_MODEL,
        device=os.getenv('EMBEDDING_DEVICE', 'cuda'),
        fp16=os.getenv('EMBEDDING_FP16', 'true').lower() == 'true'
    )
    print('✅ Embedding model loaded')
    
    print('🔄 Loading reranker model...')
    from app.models.reranker import RerankerModel
    reranker_model = RerankerModel(
        model_name=RERANK_MODEL,
        device=os.getenv('RERANKING_DEVICE', 'cuda'),
        fp16=os.getenv('RERANKING_FP16', 'true').lower() == 'true'
    )
    print('✅ Reranker model loaded')
    
    print('🔄 Loading BERT for text splitting...')
    from transformers import BertTokenizer, BertForNextSentencePrediction
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    model = BertForNextSentencePrediction.from_pretrained(BERT_PATH)
    print('✅ BERT model loaded')
    
    print('🎉 All models pre-loaded successfully!')
    
except Exception as e:
    print(f'⚠️  Model preloading failed: {e}')
    print('⚠️  Models will be loaded on first request instead')
    import traceback
    traceback.print_exc()
" || warn "Model preloading failed, models will load on demand"
    else
        log "📦 Model preloading disabled, models will load on first request"
    fi
}

# Health check function with timeout
health_check() {
    log "🔍 Performing application health check..."
    local max_attempts=60  # 5 minutes for model loading
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            log "✅ Application is healthy!"
            return 0
        fi
        
        if [ $attempt -eq 1 ]; then
            log "⏳ Waiting for application to start (models may be loading)..."
        elif [ $((attempt % 10)) -eq 0 ]; then
            log "⏳ Still waiting for startup... ($attempt/$max_attempts)"
        fi
        
        sleep 5
        ((attempt++))
    done
    
    error "❌ Health check failed after $max_attempts attempts"
}

# Start uvicorn in background and monitor
start_application() {
    log "🌟 Launching GeoGPT-RAG API server..."
    
    # Start uvicorn with production settings optimized for g5.xlarge
    uvicorn app.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        --loop uvloop \
        --http httptools \
        --access-log \
        --log-level info \
        --timeout-keep-alive 65 \
        --timeout-graceful-shutdown 30 \
        --backlog 2048 \
        --limit-max-requests 1000 &
    
    local uvicorn_pid=$!
    log "🚀 Uvicorn started with PID: $uvicorn_pid"
    
    # Wait a moment for uvicorn to start
    sleep 10
    
    # Run health check
    if health_check; then
        log "🎉 GeoGPT-RAG is running successfully!"
        log "🌐 API available at http://localhost:8000"
        log "📚 Documentation at http://localhost:8000/docs"
        
        # Wait for uvicorn process
        wait $uvicorn_pid
    else
        kill $uvicorn_pid 2>/dev/null || true
        error "❌ Application failed to start properly"
    fi
}

# Main execution flow
main() {
    log "🚀 GeoGPT-RAG Production Startup"
    log "==============================="
    
    validate_environment
    check_gpu
    check_dependencies
    setup_environment
    setup_directories
    setup_nltk
    test_sagemaker
    preload_models
    start_application
}

# Handle signals gracefully
trap 'log "🛑 Received shutdown signal, stopping application..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@"
