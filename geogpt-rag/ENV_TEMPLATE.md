# Environment Variables Configuration

## Required Environment Variables

Copy these variables to your `.env` file and update with your actual values:

```bash
# =====================================================================================
# GeoGPT-RAG Environment Configuration
# =====================================================================================

# ─────────────────────────────────────────────────────────────────────────────────
# 🔥 REQUIRED CONFIGURATION (Must be set for production)
# ─────────────────────────────────────────────────────────────────────────────────

# Milvus/Zilliz Cloud Configuration
ZILLIZ_URI=https://your-cluster.vectordb.zilliz.com:19530
ZILLIZ_TOKEN=your_zilliz_token_here
MILVUS_COLLECTION=geodocs

# External LLM Configuration
LLM_URL=https://api.openai.com/v1
LLM_KEY=your_openai_api_key_here

# ─────────────────────────────────────────────────────────────────────────────────
# 🔧 MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────────

# Model Names (HuggingFace model paths)
EMBED_MODEL=GeoGPT-Research-Project/GeoEmbedding
RERANK_MODEL=GeoGPT-Research-Project/GeoReranker
BERT_PATH=bert-base-uncased

# ─────────────────────────────────────────────────────────────────────────────────
# 🚀 DEVICE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────────

# Device assignment (cuda/cpu)
EMBEDDING_DEVICE=cuda
RERANKING_DEVICE=cuda
TEXT_SPLITTER_DEVICE=cuda

# Precision settings (true/false)
EMBEDDING_FP16=true
RERANKING_FP16=true

# ─────────────────────────────────────────────────────────────────────────────────
# ⚡ PERFORMANCE TUNING
# ─────────────────────────────────────────────────────────────────────────────────

# Batch sizes (adjust based on GPU memory)
EMBEDDING_BATCH_SIZE=32
RERANKING_BATCH_SIZE=32

# Retrieval parameters
VEC_RECALL_NUM=128
TOP_K=3
SCORE_THRESHOLD=1.5
EXPAND_RANGE=1024
EXPAND_TIME_OUT=30

# Use metadata in reranking (true/false)
META=true

# ─────────────────────────────────────────────────────────────────────────────────
# 📝 TEXT PROCESSING
# ─────────────────────────────────────────────────────────────────────────────────

# Chunk size in tokens
CHUNK_MAX_SIZE=512

# Directory for storing text chunks
CHUNK_PATH_NAME=split_chunks

# ─────────────────────────────────────────────────────────────────────────────────
# 📊 LOGGING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────────

# Log level (DEBUG/INFO/WARNING/ERROR)
LOG_LEVEL=INFO

# Log file path
LOG_FILE=logs/rag.log

# ─────────────────────────────────────────────────────────────────────────────────
# ☁️ AWS CONFIGURATION (for CloudWatch logging)
# ─────────────────────────────────────────────────────────────────────────────────

# AWS region for CloudWatch logs
AWS_DEFAULT_REGION=us-east-1

# AWS credentials (optional - can use IAM roles instead)
# AWS_ACCESS_KEY_ID=your_aws_access_key
# AWS_SECRET_ACCESS_KEY=your_aws_secret_key

# ─────────────────────────────────────────────────────────────────────────────────
# 🎯 OPTIONAL OPTIMIZATIONS
# ─────────────────────────────────────────────────────────────────────────────────

# Pre-load models on startup (true/false) - speeds up first request
PRELOAD_MODELS=false

# GPU visibility (for multi-GPU systems)
CUDA_VISIBLE_DEVICES=0

# ─────────────────────────────────────────────────────────────────────────────────
# 🔧 ADVANCED CONFIGURATION (Optional)
# ─────────────────────────────────────────────────────────────────────────────────

# Custom RAG prompt template (use default if not set)
# Must include {search_results}, {cur_date}, and {question} placeholders
# RAG_PROMPT="Your custom prompt template here..."
```

## Variable Descriptions

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ZILLIZ_URI` | Zilliz Cloud endpoint URL | `https://your-cluster.vectordb.zilliz.com:19530` |
| `ZILLIZ_TOKEN` | Zilliz Cloud authentication token | `your_token_here` |
| `LLM_URL` | External LLM API endpoint | `https://api.openai.com/v1` |
| `LLM_KEY` | LLM API key | `sk-...` |

### Device Configuration

| Variable | Description | Values |
|----------|-------------|--------|
| `EMBEDDING_DEVICE` | Device for embedding model | `cuda` / `cpu` |
| `RERANKING_DEVICE` | Device for reranking model | `cuda` / `cpu` |
| `TEXT_SPLITTER_DEVICE` | Device for text splitter | `cuda` / `cpu` |
| `EMBEDDING_FP16` | Use FP16 for embeddings | `true` / `false` |
| `RERANKING_FP16` | Use FP16 for reranking | `true` / `false` |

### Performance Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_BATCH_SIZE` | Embedding batch size | `32` |
| `RERANKING_BATCH_SIZE` | Reranking batch size | `32` |
| `VEC_RECALL_NUM` | Vector search recall count | `128` |
| `TOP_K` | Final result count | `3` |
| `SCORE_THRESHOLD` | Score threshold for filtering | `1.5` |

## Configuration Examples

### CPU-Only Deployment
```bash
EMBEDDING_DEVICE=cpu
RERANKING_DEVICE=cpu
TEXT_SPLITTER_DEVICE=cpu
EMBEDDING_FP16=false
RERANKING_FP16=false
```

### Memory-Constrained GPU
```bash
EMBEDDING_BATCH_SIZE=16
RERANKING_BATCH_SIZE=16
CHUNK_MAX_SIZE=256
```

### High-Performance Setup
```bash
EMBEDDING_BATCH_SIZE=64
RERANKING_BATCH_SIZE=64
VEC_RECALL_NUM=256
PRELOAD_MODELS=true
``` 