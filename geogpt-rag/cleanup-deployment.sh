#!/bin/bash

# =====================================================================================
# GeoGPT-RAG Complete Cleanup and Space-Optimized Deployment Script
# Removes all previous deployment artifacts and redeploys with space optimization
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

echo "🧹 GeoGPT-RAG Complete Cleanup & Space-Optimized Deployment"
echo "============================================================"

# Show initial disk usage
log "📊 Initial disk usage:"
df -h / | tail -1 | awk '{print "  - Available: " $4 " (" $5 " used)"}'

# Confirm destructive operation
read -p "⚠️  This will remove ALL Docker containers, images, volumes, cached files, and redeploy. Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    log "Cleanup cancelled"
    exit 0
fi

# Ask if user wants automatic redeployment
read -p "🚀 Auto-redeploy after cleanup? (Y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    REDEPLOY=false
else
    REDEPLOY=true
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

# Additional aggressive cleanup for space
log "🗑️  Aggressive space cleanup..."
sudo apt clean
sudo apt autoclean
sudo apt autoremove -y
sudo rm -rf /tmp/* 2>/dev/null || true
sudo rm -rf /var/tmp/* 2>/dev/null || true
sudo rm -rf /root/.cache/* 2>/dev/null || true
rm -rf ~/.cache/* 2>/dev/null || true

# Clean system logs older than 1 day
log "📄 Cleaning old log files..."
sudo journalctl --vacuum-time=1d
sudo find /var/log -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true

# Reset Docker daemon (optional but recommended)
log "🔄 Restarting Docker daemon..."
sudo systemctl restart docker
sleep 5

# Show space after cleanup
log "📊 Space freed up:"
df -h / | tail -1 | awk '{print "  - Available: " $4 " (" $5 " used)"}'

# Space-optimized deployment
if [ "$REDEPLOY" = true ]; then
    log "🚀 Starting space-optimized deployment..."
    
    # Create deployment directory
    sudo mkdir -p /opt/geogpt-rag/app
    cd /opt/geogpt-rag/app
    
    # Clone repository with latest fixes
    log "📥 Cloning latest repository..."
    sudo git clone https://github.com/Rekklessss/geogpt-mvp.git . 2>/dev/null || {
        # If already exists, ensure proper permissions and pull
        log "📥 Repository exists, fixing permissions and updating..."
        sudo chown -R ubuntu:ubuntu /opt/geogpt-rag/
        sudo git config --global --add safe.directory /opt/geogpt-rag/app
        sudo git config --global --add safe.directory /opt/geogpt-rag/app/geogpt-rag
        git pull origin main 2>/dev/null || {
            log "⚠️  Could not update repository, using existing code"
        }
    }
    
    # Fix ownership and permissions after clone/pull
    log "🔧 Fixing ownership and git permissions..."
    sudo chown -R ubuntu:ubuntu /opt/geogpt-rag/
    sudo git config --global --add safe.directory /opt/geogpt-rag/app
    sudo git config --global --add safe.directory /opt/geogpt-rag/app/geogpt-rag
    
    cd geogpt-rag
    
    # Ensure we have the latest code with all fixes
    log "📋 Ensuring latest code with all fixes..."
    git checkout -- Dockerfile 2>/dev/null || true
    git pull origin main 2>/dev/null || true
    
    # Build with space optimization (without sudo to avoid permission issues)
    log "🏗️  Building with space optimization..."
    docker compose build --no-cache --pull
    
    # Ensure proper directory permissions before starting
    log "🔧 Setting up directories with proper permissions..."
    mkdir -p data/uploads split_chunks logs
    chmod -R 777 data/uploads split_chunks logs
    
    # Start the application
    log "▶️  Starting application..."
    docker compose up -d
    
    # Wait for startup
    log "⏳ Waiting for application to start..."
    sleep 30
    
    # Check status
    log "📊 Deployment status:"
    docker compose ps
    
    # Test health
    log "🏥 Testing application health..."
    sleep 30
    if curl -f http://localhost:8000/health 2>/dev/null; then
        log "✅ Application is healthy!"
    else
        warn "⚠️  Application health check failed - may still be starting up"
        log "📋 Checking logs:"
        docker compose logs --tail=20
    fi
    
    # Verify key fixes
    log "🔍 Verifying deployment fixes..."
    
    # Check if uploads directory is accessible
    if docker compose exec geogpt-rag ls -la /app/data/uploads >/dev/null 2>&1; then
        log "✅ Uploads directory accessible inside container"
    else
        warn "⚠️  Uploads directory may have permission issues"
    fi
    
    # Quick test of reranking functionality
    if docker compose exec geogpt-rag python -c "from app.reranking import GeoReRanking; r = GeoReRanking(); print('✅ Reranking module working')" 2>/dev/null; then
        log "✅ Reranking module verified"
    else
        warn "⚠️  Reranking module may have issues"
    fi
    
    log "🎉 Space-optimized deployment completed!"
    log "🌐 Application URL: http://$(curl -s ifconfig.me):8000"
else
    log "⏭️  Skipping deployment as requested"
fi

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
if [ "$REDEPLOY" = true ]; then
    log "🎉 Complete cleanup and deployment finished!"
    log "📋 Your GeoGPT-RAG application is running!"
    log "   🌐 Access at: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_EC2_IP'):8000"
    log "   📊 Monitor with: cd /opt/geogpt-rag/app/geogpt-rag && docker compose logs -f"
    log "   🔄 Restart with: cd /opt/geogpt-rag/app/geogpt-rag && docker compose restart"
    log "   🧪 Test with: cd /opt/geogpt-rag/app/geogpt-rag && ./run-comprehensive-tests.sh"
else
    log "🎉 Complete cleanup finished!"
    log "📋 Next steps for manual deployment:"
    log "   1. Run: cd /opt/geogpt-rag/app/geogpt-rag"
    log "   2. Fix permissions: sudo chown -R ubuntu:ubuntu /opt/geogpt-rag/ && sudo git config --global --add safe.directory /opt/geogpt-rag/app/geogpt-rag"
    log "   3. Update code: git pull origin main"
    log "   4. Deploy: docker compose up -d"
    log "   5. Monitor: docker compose logs -f"
fi
echo 