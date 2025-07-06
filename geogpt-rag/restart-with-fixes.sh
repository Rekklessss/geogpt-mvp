#!/bin/bash

# Simple GeoGPT-RAG Restart Script with Fixes
# ===========================================

set -e

echo "🔄 Restarting GeoGPT-RAG with compatibility fixes..."

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

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_status "red" "❌ Error: docker-compose.yml not found. Please run this script from the geogpt-rag directory."
    exit 1
fi

# Stop any running containers
print_status "yellow" "🛑 Stopping running containers..."
docker compose down

# Start the application with the existing fixes
print_status "green" "🚀 Starting application with compatibility fixes..."
docker compose up -d --build

# Wait for startup
print_status "yellow" "⏳ Waiting for application to start..."
sleep 15

# Check if it's running
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    print_status "green" "✅ Application restarted successfully!"
    print_status "green" "🌐 Access at: http://localhost:8000"
    print_status "green" "📚 Docs at: http://localhost:8000/docs"
    
    echo ""
    curl http://localhost:8000/health | jq . 2>/dev/null || curl http://localhost:8000/health
else
    print_status "yellow" "⏳ Application still starting up, check logs with:"
    print_status "yellow" "   docker compose logs -f geogpt-rag"
fi 