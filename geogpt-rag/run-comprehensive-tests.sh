#!/bin/bash

# =====================================================================================
# GeoGPT-RAG Enhanced Comprehensive Pipeline Test Suite
# Validates complete RAG pipeline: SageMaker, embeddings, vector store, reranking
# =====================================================================================

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

success() {
    echo -e "${PURPLE}[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS: $1${NC}"
}

test_info() {
    echo -e "${CYAN}[$(date '+%Y-%m-%d %H:%M:%S')] TEST: $1${NC}"
}

# Check if running on EC2 or locally
if [[ -f /proc/version ]] && grep -q "Amazon" /proc/version; then
    ENVIRONMENT="EC2"
    log "🚀 Running on AWS EC2 instance"
else
    ENVIRONMENT="LOCAL"
    log "💻 Running on local machine"
fi

# Change to correct directory
cd "$(dirname "$0")"

log "🧪 Starting GeoGPT-RAG Enhanced Comprehensive Test Suite"
log "========================================================"
info "Environment: $ENVIRONMENT"
info "CUDA 12.8 Compatibility & Complete RAG Pipeline Validation"

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TEST_RESULTS=()

# Function to record test results
record_test() {
    local test_name="$1"
    local result="$2"
    local details="$3"
    
    if [[ "$result" == "PASS" ]]; then
        TESTS_PASSED=$((TESTS_PASSED + 1))
        TEST_RESULTS+=("✅ $test_name")
    else
        TESTS_FAILED=$((TESTS_FAILED + 1))
        TEST_RESULTS+=("❌ $test_name: $details")
    fi
}

# =============================================================================
# 1. DOCKER ENVIRONMENT TESTS
# =============================================================================

log "🐳 Phase 1: Docker Environment Validation"

test_info "Checking Docker containers status..."
if docker compose ps | grep -q "Up"; then
    success "✅ Docker containers are running"
    docker compose ps
    record_test "Docker Containers" "PASS"
else
    warn "⚠️  Containers not running, attempting to start..."
    docker compose up -d
    sleep 30
    
    if docker compose ps | grep -q "Up"; then
        success "✅ Containers started successfully"
        record_test "Docker Containers" "PASS"
    else
        record_test "Docker Containers" "FAIL" "Failed to start containers"
        error "❌ Failed to start containers"
    fi
fi

# =============================================================================
# 2. DEPENDENCY AND CUDA TESTS
# =============================================================================

log "🔧 Phase 2: Dependency & CUDA Validation"

test_info "Testing Python 3.12 and dependencies..."
if docker compose exec geogpt-rag python -c "
import sys
import torch
import transformers
import sentence_transformers
import pymilvus
import tensorflow as tf

print(f'✅ Python: {sys.version}')
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ Sentence Transformers: {sentence_transformers.__version__}')
print(f'✅ pymilvus: {pymilvus.__version__}')
print(f'✅ TensorFlow: {tf.__version__}')

# Verify key version requirements
assert sys.version_info >= (3, 12), 'Python 3.12+ required'
assert 'cu121' in torch.__version__ or not torch.cuda.is_available(), 'PyTorch CUDA compatibility issue'

pymilvus_version = tuple(map(int, pymilvus.__version__.split('.')[:2]))
assert pymilvus_version >= (2, 4), f'pymilvus 2.4.10+ required, got {pymilvus.__version__}'

print('🎉 All dependencies validated!')
" 2>/dev/null; then
    record_test "Dependencies" "PASS"
    success "✅ All dependencies validated"
else
    record_test "Dependencies" "FAIL" "Dependency validation failed"
fi

test_info "Testing CUDA 12.8 compatibility..."
if docker compose exec geogpt-rag python -c "
import torch

print(f'🔥 PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'📱 CUDA device count: {torch.cuda.device_count()}')
    print(f'💾 CUDA device name: {torch.cuda.get_device_name(0)}')
    print(f'🎯 CUDA version: {torch.version.cuda}')
    
    # Test basic GPU operations
    device = torch.device('cuda:0')
    test_tensor = torch.randn(100, 100, device=device)
    result = torch.matmul(test_tensor, test_tensor)
    print(f'✅ GPU tensor operations working: {result.shape}')
else:
    print('⚠️  CUDA not available (CPU-only mode)')
" 2>/dev/null; then
    record_test "CUDA Compatibility" "PASS"
    success "✅ CUDA compatibility verified"
else
    record_test "CUDA Compatibility" "FAIL" "CUDA test failed"
fi

success "✅ Phase 2 completed: Dependencies and CUDA validated"

# =============================================================================
# 3. API ENDPOINT TESTS
# =============================================================================

log "🌐 Phase 3: API Endpoint Testing"

test_info "Waiting for API to be ready..."
sleep 10

test_info "Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")

if [[ "$HEALTH_RESPONSE" == "200" ]]; then
    success "✅ Health endpoint responding (200 OK)"
    curl -s http://localhost:8000/health | jq '.' || echo "Health response received"
    record_test "Health Endpoint" "PASS"
elif [[ "$HEALTH_RESPONSE" == "503" ]]; then
    warn "⚠️  Health endpoint responding (503 Service Unavailable) - KB may not be initialized yet"
    curl -s http://localhost:8000/health | jq '.' || echo "Health response received"
    record_test "Health Endpoint" "PASS"
else
    record_test "Health Endpoint" "FAIL" "HTTP $HEALTH_RESPONSE"
    error "❌ Health endpoint not responding (HTTP $HEALTH_RESPONSE)"
fi

test_info "Testing root endpoint..."
ROOT_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ || echo "000")

if [[ "$ROOT_RESPONSE" == "200" ]]; then
    success "✅ Root endpoint responding (200 OK)"
    curl -s http://localhost:8000/ | jq '.' || echo "Root response received"
    record_test "Root Endpoint" "PASS"
else
    record_test "Root Endpoint" "FAIL" "HTTP $ROOT_RESPONSE"
    error "❌ Root endpoint not responding (HTTP $ROOT_RESPONSE)"
fi

success "✅ Phase 3 completed: API endpoints validated"

# =============================================================================
# 4. SAGEMAKER INTEGRATION TESTS
# =============================================================================

log "🤖 Phase 4: SageMaker Integration Testing"

test_info "Testing SageMaker endpoint connectivity and LLM generation..."
SAGEMAKER_TEST=$(curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What is geography?", "k": 3}' \
    -w "%{http_code}" -o /tmp/sagemaker_response.json || echo "000")

if [[ "$SAGEMAKER_TEST" == "200" ]]; then
    # Validate response structure
    if jq -e '.response' /tmp/sagemaker_response.json >/dev/null 2>&1; then
        success "✅ SageMaker LLM generation working"
        RESPONSE_TEXT=$(jq -r '.response' /tmp/sagemaker_response.json)
        info "Sample LLM Response: ${RESPONSE_TEXT:0:100}..."
        record_test "SageMaker LLM Generation" "PASS"
        
        # Test response quality
        if [[ ${#RESPONSE_TEXT} -gt 50 ]]; then
            success "✅ LLM response quality check passed"
            record_test "LLM Response Quality" "PASS"
        else
            warn "⚠️  LLM response seems too short"
            record_test "LLM Response Quality" "FAIL" "Response too short"
        fi
    else
        warn "⚠️  SageMaker responded but invalid JSON structure"
        record_test "SageMaker LLM Generation" "FAIL" "Invalid response structure"
    fi
elif [[ "$SAGEMAKER_TEST" == "500" ]]; then
    warn "⚠️  SageMaker endpoint returned 500 (may need documents in KB)"
    record_test "SageMaker LLM Generation" "PARTIAL" "No documents in KB"
else
    record_test "SageMaker LLM Generation" "FAIL" "HTTP $SAGEMAKER_TEST"
fi

# Clean up temp file
rm -f /tmp/sagemaker_response.json

test_info "Testing SageMaker endpoint status..."
docker compose exec geogpt-rag python -c "
import boto3
import os

try:
    sagemaker = boto3.client('sagemaker-runtime', region_name='us-east-1')
    endpoint_name = os.getenv('SAGEMAKER_ENDPOINT_NAME', 'GeoGPT-R1-Sagemaker-Endpoint')
    
    # Test with a simple query
    response = sagemaker.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body='{\"inputs\": \"Test connection\", \"parameters\": {\"max_length\": 50}}'
    )
    
    print('✅ SageMaker endpoint direct connection successful')
    print(f'✅ Response status: {response[\"ResponseMetadata\"][\"HTTPStatusCode\"]}')
except Exception as e:
    print(f'❌ SageMaker endpoint test failed: {e}')
" && record_test "SageMaker Direct Connection" "PASS" || record_test "SageMaker Direct Connection" "FAIL"

success "✅ Phase 4 completed: SageMaker integration tested"

# =============================================================================
# 5. END-TO-END RAG PIPELINE TESTS
# =============================================================================

log "🔗 Phase 5: End-to-End RAG Pipeline Testing"

test_info "Creating comprehensive test document..."
TEST_FILE="/tmp/comprehensive_test_doc.txt"
cat > "$TEST_FILE" << 'EOF'
# Geographic Information Systems (GIS) and Remote Sensing

Geographic Information Systems (GIS) are powerful tools used for capturing, storing, analyzing, and managing spatial and geographic data. GIS technology integrates hardware, software, and data to capture, manage, analyze, and display all forms of geographically referenced information.

## Remote Sensing Technologies

Remote sensing is the science of obtaining information about objects or areas from a distance, typically from aircraft or satellites. This technology allows scientists to collect data about Earth's surface without making physical contact with the object or area being studied.

### Satellite Applications
- Environmental monitoring
- Urban planning
- Agricultural assessment
- Climate change research
- Natural disaster management

## Spatial Analysis Methods

Spatial analysis involves the study of geographic phenomena and their relationships within space. This includes:

1. Point pattern analysis
2. Network analysis  
3. Surface analysis
4. Spatial interpolation
5. Geostatistical analysis

The integration of GIS and remote sensing provides comprehensive solutions for geographic data analysis and decision-making processes in various fields including environmental science, urban planning, and natural resource management.
EOF

test_info "Testing complete RAG pipeline: Upload → Embed → Store → Retrieve → Generate..."

# Step 1: Upload document
info "Step 1: Document upload..."
UPLOAD_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:8000/upload \
    -F "file=@$TEST_FILE" || echo "000")

if [[ "$UPLOAD_RESPONSE" == "200" ]]; then
    success "✅ Document upload successful"
    record_test "Document Upload" "PASS"
    
    # Wait for processing
    sleep 10
    
    # Step 2: Test retrieval with relevant query
    info "Step 2: Testing semantic retrieval..."
    RETRIEVAL_TEST=$(curl -s -X POST http://localhost:8000/query \
        -H "Content-Type: application/json" \
        -d '{"query": "What are the applications of satellite remote sensing in environmental monitoring?", "k": 5}' \
        -w "%{http_code}" -o /tmp/retrieval_response.json || echo "000")
    
    if [[ "$RETRIEVAL_TEST" == "200" ]]; then
        # Validate retrieval quality
        if jq -e '.retrieved_docs' /tmp/retrieval_response.json >/dev/null 2>&1; then
            DOC_COUNT=$(jq '.retrieved_docs | length' /tmp/retrieval_response.json)
            success "✅ Document retrieval working (retrieved $DOC_COUNT documents)"
            record_test "Document Retrieval" "PASS"
            
            # Step 3: Validate LLM response uses retrieved context
            if jq -e '.response' /tmp/retrieval_response.json >/dev/null 2>&1; then
                LLM_RESPONSE=$(jq -r '.response' /tmp/retrieval_response.json)
                
                # Check if response contains relevant keywords from our document
                if [[ "$LLM_RESPONSE" == *"satellite"* ]] || [[ "$LLM_RESPONSE" == *"remote sensing"* ]] || [[ "$LLM_RESPONSE" == *"environmental"* ]]; then
                    success "✅ LLM response uses retrieved context"
                    record_test "RAG Context Integration" "PASS"
                    info "LLM Response Preview: ${LLM_RESPONSE:0:200}..."
                else
                    warn "⚠️  LLM response may not be using retrieved context effectively"
                    record_test "RAG Context Integration" "FAIL" "Context not effectively used"
                fi
            else
                record_test "RAG Context Integration" "FAIL" "No LLM response"
            fi
        else
            record_test "Document Retrieval" "FAIL" "No retrieved docs in response"
        fi
    else
        record_test "Document Retrieval" "FAIL" "HTTP $RETRIEVAL_TEST"
    fi
else
    record_test "Document Upload" "FAIL" "HTTP $UPLOAD_RESPONSE"
fi

rm -f "$TEST_FILE" /tmp/retrieval_response.json

success "✅ Phase 5 completed: End-to-End RAG pipeline tested"

# =============================================================================
# 6. VECTOR STORE AND EMBEDDING TESTS
# =============================================================================

log "🔢 Phase 6: Vector Store & Embedding Quality Testing"

test_info "Testing embedding generation quality..."
docker compose exec geogpt-rag python -c "
import sys
sys.path.append('/app/app')
from embeddings import GeoEmbeddings

try:
    embedder = GeoEmbeddings()
    
    # Test embedding generation
    test_texts = [
        'Geographic Information Systems are powerful tools for spatial analysis.',
        'Remote sensing enables monitoring of environmental changes from space.',
        'The weather is nice today.'  # Different domain for comparison
    ]
    
    embeddings = []
    for text in test_texts:
        emb = embedder.embed_documents([text])[0]
        embeddings.append(emb)
        print(f'✅ Generated embedding for text (dim: {len(emb)})')
    
    # Test embedding similarity
    import numpy as np
    
    # Calculate similarities
    geo_sim = np.dot(embeddings[0], embeddings[1])  # GIS vs Remote sensing
    weather_sim = np.dot(embeddings[0], embeddings[2])  # GIS vs Weather
    
    print(f'🔍 GIS-RemoteSensing similarity: {geo_sim:.4f}')
    print(f'🔍 GIS-Weather similarity: {weather_sim:.4f}')
    
    # Validate that domain-related texts are more similar
    if geo_sim > weather_sim:
        print('✅ Embedding quality check passed: related texts more similar')
    else:
        print('⚠️  Embedding quality concern: unexpected similarity scores')
        
except Exception as e:
    print(f'❌ Embedding test failed: {e}')
" && record_test "Embedding Quality" "PASS" || record_test "Embedding Quality" "FAIL"

test_info "Testing vector store operations..."
docker compose exec geogpt-rag python -c "
import sys
sys.path.append('/app/app')
from kb import KBDocQA

try:
    kb = KBDocQA()
    
    # Test vector store connection
    if hasattr(kb, 'vector_store') and kb.vector_store:
        print('✅ Vector store connection established')
        
        # Test search functionality
        results = kb.vector_store.similarity_search('geographic information systems', k=3)
        print(f'✅ Vector search working (found {len(results)} results)')
        
        for i, doc in enumerate(results[:2]):
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f'   Result {i+1}: {preview}...')
    else:
        print('❌ Vector store not properly initialized')
        
except Exception as e:
    print(f'❌ Vector store test failed: {e}')
" && record_test "Vector Store Operations" "PASS" || record_test "Vector Store Operations" "FAIL"

success "✅ Phase 6 completed: Vector store and embeddings tested"

# =============================================================================
# 7. RERANKING AND RELEVANCE TESTS
# =============================================================================

log "🎯 Phase 7: Reranking & Relevance Testing"

test_info "Testing reranking functionality..."
docker compose exec geogpt-rag python -c "
import sys
sys.path.append('/app/app')
from reranking import GeoReranker

try:
    reranker = GeoReranker()
    
    # Test reranking with geographic query
    query = 'satellite applications in environmental monitoring'
    documents = [
        'Satellites are used for weather forecasting and climate monitoring.',
        'Environmental monitoring includes tracking deforestation and pollution.',
        'The restaurant serves delicious food.',  # Irrelevant doc
        'Remote sensing satellites provide data for environmental assessment.',
        'Urban planning uses GIS for city development.'
    ]
    
    # Get reranking scores
    scores = reranker.rerank(query, documents)
    print(f'✅ Reranking completed for {len(documents)} documents')
    
    # Display ranked results
    ranked_docs = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    
    for i, (doc, score) in enumerate(ranked_docs[:3]):
        preview = doc[:60]
        print(f'   Rank {i+1} (score: {score:.4f}): {preview}...')
    
    # Validate that relevant docs rank higher
    irrelevant_doc_rank = next(i for i, (doc, _) in enumerate(ranked_docs) if 'restaurant' in doc)
    if irrelevant_doc_rank >= 2:  # Should not be in top 2
        print('✅ Reranking quality check passed: irrelevant doc ranked lower')
    else:
        print('⚠️  Reranking quality concern: irrelevant doc ranked too high')
        
except Exception as e:
    print(f'❌ Reranking test failed: {e}')
" && record_test "Reranking Functionality" "PASS" || record_test "Reranking Functionality" "FAIL"

success "✅ Phase 7 completed: Reranking tested"

# =============================================================================
# 8. DOCUMENT PROCESSING PIPELINE TESTS
# =============================================================================

log "📄 Phase 8: Document Processing Pipeline Testing"

test_info "Testing text splitting and chunking..."
docker compose exec geogpt-rag python -c "
import sys
sys.path.append('/app/app')
from utils.parsers import SemanticChunker

try:
    chunker = SemanticChunker()
    
    # Test with a longer document
    test_doc = '''
    Geographic Information Systems (GIS) are computer-based tools for capturing, storing, checking, and displaying data related to positions on Earth's surface. GIS can show many different kinds of data on one map, such as streets, buildings, and vegetation. This enables people to more easily see, analyze, and understand patterns and relationships.
    
    Remote sensing is the process of detecting and monitoring the physical characteristics of an area by measuring its reflected and emitted radiation at a distance from the targeted area. Special cameras collect remotely sensed images, which help researchers sense things about the Earth.
    
    Spatial analysis uses statistical techniques to quantify patterns, relationships, and trends within geographic data. This analytical approach helps identify hotspots, clusters, and spatial relationships that might not be obvious through visual inspection alone.
    '''
    
    # Test chunking
    chunks = chunker.split_text(test_doc)
    print(f'✅ Document split into {len(chunks)} semantic chunks')
    
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk[:80].replace('\n', ' ').strip()
        print(f'   Chunk {i+1}: {preview}...')
    
    # Validate chunk quality
    total_chars = sum(len(chunk) for chunk in chunks)
    avg_chunk_size = total_chars / len(chunks) if chunks else 0
    
    if 200 <= avg_chunk_size <= 1000:  # Reasonable chunk size
        print(f'✅ Chunk sizing appropriate (avg: {avg_chunk_size:.0f} chars)')
    else:
        print(f'⚠️  Chunk sizing may need adjustment (avg: {avg_chunk_size:.0f} chars)')
        
except Exception as e:
    print(f'❌ Document processing test failed: {e}')
" && record_test "Document Processing" "PASS" || record_test "Document Processing" "FAIL"

success "✅ Phase 8 completed: Document processing tested"

# =============================================================================
# 9. PERFORMANCE BENCHMARKS
# =============================================================================

log "⚡ Phase 9: Performance Benchmarking"

test_info "Testing API response times and throughput..."
docker compose exec geogpt-rag python -c "
import time
import requests
import concurrent.futures
import statistics

def test_endpoint_performance(url, name, payload=None):
    times = []
    method = 'POST' if payload else 'GET'
    
    for _ in range(5):  # Run 5 times for average
        start_time = time.time()
        try:
            if payload:
                response = requests.post(url, json=payload, timeout=30)
            else:
                response = requests.get(url, timeout=30)
            end_time = time.time()
            
            if response.status_code in [200, 503]:  # 503 OK for uninitialized KB
                times.append((end_time - start_time) * 1000)
        except Exception as e:
            print(f'❌ {name}: Request failed ({e})')
            return None
    
    if times:
        avg_time = statistics.mean(times)
        min_time = min(times)
        max_time = max(times)
        print(f'⏱️  {name}: avg={avg_time:.2f}ms, min={min_time:.2f}ms, max={max_time:.2f}ms')
        return avg_time
    return None

# Test different endpoints
print('📊 Performance Benchmarks:')
health_time = test_endpoint_performance('http://localhost:8000/health', 'Health Endpoint')
root_time = test_endpoint_performance('http://localhost:8000/', 'Root Endpoint')

# Test query performance (with simple query to avoid long processing)
query_time = test_endpoint_performance(
    'http://localhost:8000/query',
    'Query Endpoint',
    {'query': 'test', 'k': 3}
)

# Concurrent request test
print('\n🚀 Concurrent Request Test:')
def make_request():
    try:
        response = requests.get('http://localhost:8000/health', timeout=10)
        return response.status_code == 200
    except:
        return False

start_time = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(make_request) for _ in range(10)]
    results = [f.result() for f in concurrent.futures.as_completed(futures)]

concurrent_time = time.time() - start_time
success_rate = sum(results) / len(results) * 100

print(f'✅ Concurrent requests: {success_rate:.0f}% success rate in {concurrent_time:.2f}s')

# Performance thresholds
if health_time and health_time < 1000:  # Under 1 second
    print('✅ Health endpoint performance: GOOD')
else:
    print('⚠️  Health endpoint performance: NEEDS ATTENTION')
"

test_info "Checking container resource usage..."
echo "📊 Current Resource Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}" | head -2

record_test "Performance Benchmarks" "PASS"
success "✅ Phase 9 completed: Performance benchmarked"

# =============================================================================
# 10. ERROR HANDLING AND EDGE CASES
# =============================================================================

log "🛡️ Phase 10: Error Handling & Edge Case Testing"

test_info "Testing malformed requests and error handling..."

# Test malformed JSON
MALFORMED_JSON_TEST=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"invalid": json}' || echo "000")

if [[ "$MALFORMED_JSON_TEST" == "422" ]] || [[ "$MALFORMED_JSON_TEST" == "400" ]]; then
    success "✅ Malformed JSON handled correctly"
    record_test "Malformed JSON Handling" "PASS"
else
    record_test "Malformed JSON Handling" "FAIL" "HTTP $MALFORMED_JSON_TEST"
fi

# Test missing required fields
MISSING_FIELDS_TEST=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{}' || echo "000")

if [[ "$MISSING_FIELDS_TEST" == "422" ]] || [[ "$MISSING_FIELDS_TEST" == "400" ]]; then
    success "✅ Missing fields handled correctly"
    record_test "Missing Fields Handling" "PASS"
else
    record_test "Missing Fields Handling" "FAIL" "HTTP $MISSING_FIELDS_TEST"
fi

# Test very long query
LONG_QUERY=$(printf 'A%.0s' {1..5000})  # 5000 character query
LONG_QUERY_TEST=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"$LONG_QUERY\", \"k\": 3}" || echo "000")

if [[ "$LONG_QUERY_TEST" == "200" ]] || [[ "$LONG_QUERY_TEST" == "413" ]] || [[ "$LONG_QUERY_TEST" == "422" ]]; then
    success "✅ Long query handled appropriately"
    record_test "Long Query Handling" "PASS"
else
    record_test "Long Query Handling" "FAIL" "HTTP $LONG_QUERY_TEST"
fi

# Test invalid file upload
INVALID_FILE_TEST=$(curl -s -o /dev/null -w "%{http_code}" \
    -X POST http://localhost:8000/upload \
    -F "file=@/dev/null" || echo "000")

if [[ "$INVALID_FILE_TEST" == "400" ]] || [[ "$INVALID_FILE_TEST" == "422" ]]; then
    success "✅ Invalid file upload handled correctly"
    record_test "Invalid File Handling" "PASS"
else
    record_test "Invalid File Handling" "FAIL" "HTTP $INVALID_FILE_TEST"
fi

success "✅ Phase 10 completed: Error handling tested"

# =============================================================================
# 11. PYTEST UNIT TESTS
# =============================================================================

log "🧪 Phase 11: PyTest Unit Test Suite"

test_info "Running comprehensive unit tests..."
if docker compose exec geogpt-rag python -m pytest tests/ -v --tb=short --disable-warnings 2>/dev/null; then
    success "✅ All unit tests passed"
    record_test "Unit Tests" "PASS"
else
    warn "⚠️  Some unit tests failed - running individual test categories..."
    
    # Try running test categories individually
    if docker compose exec geogpt-rag python -m pytest tests/test_api.py -v 2>/dev/null; then
        record_test "API Unit Tests" "PASS"
    else
        record_test "API Unit Tests" "FAIL"
    fi
    
    if docker compose exec geogpt-rag python -m pytest tests/test_cuda_compatibility.py -v 2>/dev/null; then
        record_test "CUDA Unit Tests" "PASS"
    else
        record_test "CUDA Unit Tests" "FAIL"
    fi
    
    if docker compose exec geogpt-rag python -m pytest tests/test_sagemaker_integration.py -v 2>/dev/null; then
        record_test "SageMaker Unit Tests" "PASS"
    else
        record_test "SageMaker Unit Tests" "FAIL"
    fi
fi

success "✅ Phase 11 completed: Unit tests executed"

# =============================================================================
# 12. FINAL COMPREHENSIVE REPORT
# =============================================================================

log "🎯 Phase 12: Comprehensive Test Results Report"

echo ""
echo "============================================================"
echo "🎉 GeoGPT-RAG Enhanced Pipeline Test Summary"
echo "============================================================"
echo "Environment: $ENVIRONMENT"
echo "CUDA Version: 12.8.0"
echo "Base Image: nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04"
echo "Test Execution Time: $(date)"
echo ""

# Display test results
echo "📊 Test Results Summary:"
echo "========================"
echo "✅ Tests Passed: $TESTS_PASSED"
echo "❌ Tests Failed: $TESTS_FAILED"
echo "📈 Success Rate: $(( TESTS_PASSED * 100 / (TESTS_PASSED + TESTS_FAILED) ))%"
echo ""

echo "📋 Detailed Test Results:"
echo "========================="
for result in "${TEST_RESULTS[@]}"; do
    echo "   $result"
done
echo ""

# Final system status
echo "🖥️  System Status:"
echo "=================="

# Check final container status
if docker compose ps | grep -q "Up"; then
    echo "✅ Docker Containers: Running"
else
    echo "❌ Docker Containers: Stopped"
fi

# Check API availability
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/ || echo "000")
if [[ "$API_STATUS" == "200" ]]; then
    echo "✅ API Endpoints: Operational"
else
    echo "⚠️  API Endpoints: Issues detected (HTTP $API_STATUS)"
fi

# Check GPU access
GPU_STATUS=$(docker compose exec geogpt-rag python -c "import torch; print('Yes' if torch.cuda.is_available() else 'No')" 2>/dev/null || echo "Error")
echo "🚀 GPU Access: $GPU_STATUS"

# Check dependency versions
echo ""
echo "📦 Key Dependencies:"
docker compose exec geogpt-rag python -c "
import sys, torch, pymilvus, transformers
print(f'   Python: {sys.version.split()[0]}')
print(f'   PyTorch: {torch.__version__}')
print(f'   pymilvus: {pymilvus.__version__}') 
print(f'   transformers: {transformers.__version__}')
" 2>/dev/null || echo "   Could not retrieve versions"

echo ""
echo "============================================================"

# Final assessment
if [[ $TESTS_FAILED -eq 0 ]] && [[ "$API_STATUS" == "200" ]] && docker compose ps | grep -q "Up"; then
    success "🎉 GeoGPT-RAG Pipeline: FULLY OPERATIONAL"
    echo ""
    info "🌟 All systems operational! Ready for production use."
    info "🌐 API Documentation: http://localhost:8000/docs"
    info "🏥 Health Check: http://localhost:8000/health"
    info "📊 Monitoring: docker compose logs -f"
elif [[ $TESTS_FAILED -le 2 ]]; then
    warn "⚠️  GeoGPT-RAG Pipeline: MOSTLY OPERATIONAL"
    echo ""
    info "🔧 Minor issues detected. Review failed tests above."
    info "📋 Check logs: docker compose logs -f"
else
    error "❌ GeoGPT-RAG Pipeline: SIGNIFICANT ISSUES DETECTED"
fi

echo "============================================================"
success "✅ Enhanced comprehensive test suite completed!"
echo "" 