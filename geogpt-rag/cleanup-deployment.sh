#!/bin/bash

# =====================================================================================
# GeoGPT-RAG Complete Deployment Cleanup Script
# Removes all previous deployment artifacts for fresh start
# =====================================================================================

set -e

# Enhanced logging function
log() {
    echo -e "\033[32m[$(date '+%Y-%m-%d %H:%M:%S')] INFO: $1\033[0m"
}

warn() {
    echo -e "\033[33m[$(date '+%Y-%m-%d %H:%M:%S')] WARN: $1\033[0m"
}

error() {
    echo -e "\033[31m[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1\033[0m"
}

echo "🧹 GeoGPT-RAG Complete Cleanup"
echo "=============================="

# Confirm destructive operation
read -p "⚠️  This will remove ALL Docker containers, images, volumes, and cached files. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Cleanup cancelled"
    exit 0
fi

# Stop and remove all GeoGPT-RAG related containers
log "🛑 Stopping all GeoGPT-RAG containers..."
cd /opt/geogpt-rag/app/geogpt-rag 2>/dev/null || cd ~/geogpt-rag 2>/dev/null || true
docker compose down --remove-orphans --volumes 2>/dev/null || true
docker stop $(docker ps -q --filter "name=geogpt") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=geogpt") 2>/dev/null || true

# Remove all GeoGPT-RAG Docker images
log "🗑️  Removing GeoGPT-RAG Docker images..."
docker rmi $(docker images --filter "reference=*geogpt*" -q) 2>/dev/null || true
docker rmi $(docker images --filter "reference=geogpt-rag*" -q) 2>/dev/null || true

# Remove old CUDA images (force clean CUDA version change)
log "🗑️  Removing old CUDA Docker images..."
docker rmi $(docker images --filter "reference=nvidia/cuda:11.*" -q) 2>/dev/null || true
docker rmi $(docker images --filter "reference=nvidia/cuda:12.[0-7].*" -q) 2>/dev/null || true

# Clean Docker system (remove unused images, containers, networks, volumes)
log "🧽 Cleaning Docker system..."
docker system prune -af --volumes 2>/dev/null || true

# Remove application directories and cached files
log "📁 Removing application directories..."
sudo rm -rf /opt/geogpt-rag 2>/dev/null || true
sudo rm -rf ~/geogpt-rag-data 2>/dev/null || true
sudo rm -rf ~/.cache/huggingface 2>/dev/null || true
sudo rm -rf ~/.cache/torch 2>/dev/null || true
sudo rm -rf ~/.cache/transformers 2>/dev/null || true

# Remove systemd service if it exists
log "⚙️  Removing systemd service..."
sudo systemctl stop geogpt-rag 2>/dev/null || true
sudo systemctl disable geogpt-rag 2>/dev/null || true
sudo rm -f /etc/systemd/system/geogpt-rag.service 2>/dev/null || true
sudo systemctl daemon-reload 2>/dev/null || true

# Clear pip cache
log "🗑️  Clearing pip cache..."
python3 -m pip cache purge 2>/dev/null || true
python -m pip cache purge 2>/dev/null || true

# Remove any old Python environments
log "🐍 Removing old Python environments..."
sudo rm -rf /usr/local/lib/python*/site-packages/*geogpt* 2>/dev/null || true
sudo rm -rf /usr/local/lib/python*/site-packages/*torch* 2>/dev/null || true
sudo rm -rf /usr/local/lib/python*/site-packages/*transformers* 2>/dev/null || true

# Clear NLTK data
log "📚 Clearing NLTK data..."
sudo rm -rf /usr/local/share/nltk_data 2>/dev/null || true
sudo rm -rf /usr/share/nltk_data 2>/dev/null || true
sudo rm -rf ~/.nltk_data 2>/dev/null || true

# Reset Docker daemon (optional but recommended)
log "🔄 Restarting Docker daemon..."
sudo systemctl restart docker
sleep 5

# Verify cleanup
log "✅ Cleanup completed! Verifying..."

# Check Docker
log "📊 Docker status:"
echo "  - Containers: $(docker ps -aq | wc -l) running"
echo "  - Images: $(docker images -q | wc -l) total"
echo "  - Volumes: $(docker volume ls -q | wc -l) total"

# Check disk space freed
log "💾 Disk space:"
df -h / | tail -1 | awk '{print "  - Available: " $4 " (" $5 " used)"}'

# Verify NVIDIA Docker still works
log "🚀 Testing NVIDIA Docker..."
if docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 nvidia-smi &>/dev/null; then
    log "✅ NVIDIA Docker working correctly"
else
    warn "⚠️  NVIDIA Docker may need reconfiguration"
fi

echo
log "🎉 Complete cleanup finished!"
log "📋 Next steps:"
log "   1. Run: cd /tmp && wget https://raw.githubusercontent.com/Rekklessss/geogpt-mvp/main/geogpt-rag/deploy-ec2.sh"
log "   2. Run: chmod +x deploy-ec2.sh"
log "   3. Run: ./deploy-ec2.sh"
echo 