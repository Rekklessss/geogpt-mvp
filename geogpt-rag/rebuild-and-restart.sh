#!/bin/bash

# =====================================================================================
# GeoGPT-RAG Rebuild and Restart Script
# =====================================================================================

set -e

echo "🔧 Rebuilding GeoGPT-RAG with fixes..."

# Stop existing container
echo "⏹️  Stopping existing container..."
docker compose down

# Rebuild the image
echo "🏗️  Rebuilding Docker image..."
docker compose build --no-cache

# Start the container
echo "🚀 Starting updated container..."
docker compose up -d

# Show logs
echo "📝 Showing startup logs..."
docker compose logs -f geogpt-rag

echo "✅ Container rebuilt and restarted!" 