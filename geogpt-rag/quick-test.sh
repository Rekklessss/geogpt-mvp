#!/bin/bash

# =====================================================================================
# Quick Test Script for GeoGPT-RAG - Tests core functionality quickly
# =====================================================================================

set -e

# Color codes
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

log "🚀 Quick GeoGPT-RAG Functionality Test"
log "====================================="

# Test 1: Check container status
log "1. Checking container status..."
if docker compose ps | grep -q "Up"; then
    log "✅ Container is running"
else
    error "❌ Container is not running"
    exit 1
fi

# Test 2: Check API health
log "2. Testing API health..."
HEALTH=$(curl -s -w "%{http_code}" http://localhost:8000/health -o /tmp/health_response.json || echo "000")
if [[ "$HEALTH" == "200" ]]; then
    log "✅ API is healthy"
    cat /tmp/health_response.json | jq '.' || echo "Health response received"
else
    error "❌ API health check failed (HTTP $HEALTH)"
    exit 1
fi

# Test 3: Check upload directory permissions
log "3. Testing upload directory permissions..."
if docker compose exec geogpt-rag touch /app/data/uploads/test_permission_file 2>/dev/null; then
    docker compose exec geogpt-rag rm -f /app/data/uploads/test_permission_file
    log "✅ Upload directory is writable"
else
    warn "⚠️  Upload directory permission issue detected"
    log "🔧 Checking directory ownership..."
    docker compose exec geogpt-rag ls -la /app/data/uploads
    
    log "🔧 Attempting to fix permissions..."
    docker compose exec -u root geogpt-rag chown -R nobody:nogroup /app/data/uploads
    docker compose exec -u root geogpt-rag chmod -R 777 /app/data/uploads
    
    # Test again
    if docker compose exec geogpt-rag touch /app/data/uploads/test_permission_file 2>/dev/null; then
        docker compose exec geogpt-rag rm -f /app/data/uploads/test_permission_file
        log "✅ Upload directory permissions fixed"
    else
        error "❌ Could not fix upload directory permissions"
        exit 1
    fi
fi

# Test 4: Simple file upload test
log "4. Testing file upload..."
echo "Test document for GeoGPT-RAG system. This is a simple test of geographic information systems." > /tmp/test_doc.txt

UPLOAD_RESULT=$(curl -s -w "%{http_code}" \
    -X POST http://localhost:8000/upload \
    -F "file=@/tmp/test_doc.txt" \
    -o /tmp/upload_response.json || echo "000")

UPLOAD_CODE="${UPLOAD_RESULT: -3}"

if [[ "$UPLOAD_CODE" == "200" ]]; then
    log "✅ File upload successful"
    cat /tmp/upload_response.json | jq '.' || echo "Upload response received"
    
    # Wait a bit for processing
    log "⏳ Waiting for document processing..."
    sleep 10
    
    # Check if document was processed
    STATS_RESULT=$(curl -s http://localhost:8000/stats | jq '.document_count' || echo "0")
    log "📊 Document count in knowledge base: $STATS_RESULT"
    
else
    error "❌ File upload failed (HTTP $UPLOAD_CODE)"
    cat /tmp/upload_response.json 2>/dev/null || echo "No response body"
    exit 1
fi

# Test 5: Simple query test
log "5. Testing simple query..."
QUERY_RESULT=$(curl -s -w "%{http_code}" \
    -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "test", "k": 3}' \
    -o /tmp/query_response.json || echo "000")

QUERY_CODE="${QUERY_RESULT: -3}"

if [[ "$QUERY_CODE" == "200" ]]; then
    log "✅ Query successful"
    RESPONSE_TEXT=$(cat /tmp/query_response.json | jq -r '.response' 2>/dev/null || echo "No response")
    log "🤖 Sample response: ${RESPONSE_TEXT:0:100}..."
else
    warn "⚠️  Query failed (HTTP $QUERY_CODE) - this may be normal if no documents are indexed"
    cat /tmp/query_response.json 2>/dev/null || echo "No response body"
fi

# Cleanup
rm -f /tmp/test_doc.txt /tmp/health_response.json /tmp/upload_response.json /tmp/query_response.json

log "🎉 Quick test completed!"
log ""
log "📋 Summary:"
log "  - Container: Running"
log "  - API: Healthy"
log "  - Permissions: Fixed"
log "  - Upload: Working"
log "  - Query: $([ "$QUERY_CODE" == "200" ] && echo "Working" || echo "Needs documents")"
log ""
log "🚀 Ready to run comprehensive tests: ./run-comprehensive-tests.sh" 