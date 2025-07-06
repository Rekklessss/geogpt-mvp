#!/bin/bash

# GeoGPT-RAG Host-Compatible Startup Script with Compatibility Fixes
# ==================================================================

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

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

print_status "green" "📁 Working directory: $SCRIPT_DIR"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_status "red" "❌ Error: docker-compose.yml not found. Please run this script from the geogpt-rag directory."
    exit 1
fi

# Check if .env file exists and load it
if [ -f ".env" ]; then
    print_status "green" "✅ Found .env file, loading environment variables"
    # Export variables from .env file (excluding comments and empty lines)
    export $(grep -v '^#' .env | grep -v '^$' | xargs)
else
    print_status "yellow" "⚠️  No .env file found, using docker-compose environment variables"
fi

# Stop any running containers first
print_status "yellow" "🛑 Stopping any running containers..."
docker compose down

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    print_status "red" "❌ Error: Docker is not running. Please start Docker first."
    exit 1
fi

# Build the image with the fixes
print_status "yellow" "🔨 Building Docker image with compatibility fixes..."
if docker compose build --no-cache; then
    print_status "green" "✅ Docker image built successfully"
else
    print_status "red" "❌ Docker build failed"
    exit 1
fi

# Apply the compatibility fixes inside the container
print_status "yellow" "🔧 Running compatibility fixes inside container..."
docker compose run --rm geogpt-rag python3 embedding-compatibility-fix.py || print_status "yellow" "⚠️  Some compatibility fixes failed, but continuing..."

# Start the application
print_status "green" "🚀 Starting GeoGPT-RAG application..."
if docker compose up -d; then
    print_status "green" "✅ Application started successfully"
else
    print_status "red" "❌ Failed to start application"
    exit 1
fi

# Wait a moment for the container to start
sleep 5

# Check container status
print_status "yellow" "📊 Checking container status..."
docker compose ps

# Test if the application is responding
print_status "yellow" "🏥 Testing application health..."
sleep 10  # Give it more time to start

# Function to test health endpoint
test_health() {
    local max_attempts=12  # 12 attempts * 10 seconds = 2 minutes
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -f http://localhost:8000/health > /dev/null 2>&1; then
            print_status "green" "✅ Application is healthy and responding!"
            curl http://localhost:8000/health | jq . 2>/dev/null || curl http://localhost:8000/health
            return 0
        else
            print_status "yellow" "⏳ Attempt $attempt/$max_attempts - waiting for application to start..."
            sleep 10
            ((attempt++))
        fi
    done
    
    print_status "red" "❌ Application health check failed after 2 minutes"
    print_status "yellow" "📋 Checking application logs:"
    docker compose logs --tail=20 geogpt-rag
    return 1
}

# Test application health
if test_health; then
    print_status "green" "🎉 GeoGPT-RAG started successfully with compatibility fixes!"
    print_status "green" "🌐 Application URL: http://localhost:8000"
    print_status "green" "📚 API Documentation: http://localhost:8000/docs"
    print_status "green" "📊 Health Check: http://localhost:8000/health"
    
    echo ""
    print_status "yellow" "📋 Useful commands:"
    echo "  - View logs: docker compose logs -f geogpt-rag"
    echo "  - Stop application: docker compose down"
    echo "  - Run tests: ./quick-test.sh"
    echo ""
else
    print_status "red" "❌ Application failed to start properly"
    print_status "yellow" "🔍 Check the logs with: docker compose logs geogpt-rag"
    exit 1
fi 