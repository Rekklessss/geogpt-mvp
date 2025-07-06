#!/bin/bash

# =====================================================================================
# GeoGPT-RAG EC2 Fresh Deployment Script for g5.xlarge
# 
# Use this script for:
# - Fresh EC2 instance setup (new instance)
# - Installing Docker, NVIDIA toolkit from scratch
# 
# For existing deployments, use cleanup-deployment.sh instead
# =====================================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
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

# Check if running on correct instance
check_instance() {
    log "🔍 Verifying EC2 instance..."
    
    # Get instance metadata (support for both IMDSv1 and IMDSv2)
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "")
    
    if [ -n "$TOKEN" ]; then
        # Use IMDSv2 with token
        INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
        INSTANCE_TYPE=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
        PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
    else
        # Fallback to IMDSv1
        INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")
        INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null || echo "unknown")
        PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo "unknown")
    fi
    
    log "Instance ID: $INSTANCE_ID"
    log "Instance Type: $INSTANCE_TYPE"
    log "Public IP: $PUBLIC_IP"
    
    # Verify g5.xlarge instance
    if [[ "$INSTANCE_TYPE" != "g5.xlarge" ]]; then
        warn "This script is optimized for g5.xlarge instances. Current instance: $INSTANCE_TYPE"
        read -p "Continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            error "Deployment cancelled"
        fi
    fi
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        log "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        error "❌ NVIDIA GPU not found. Please install NVIDIA drivers."
    fi
}

# Install system dependencies
install_dependencies() {
    log "📦 Installing system dependencies..."
    
    sudo apt-get update
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release \
        software-properties-common \
        git \
        unzip \
        htop \
        tree \
        jq
    
    log "✅ System dependencies installed"
}

# Install Docker
install_docker() {
    log "🐳 Installing Docker..."
    
    if command -v docker &> /dev/null; then
        log "Docker already installed: $(docker --version)"
        return
    fi
    
    # Remove old Docker installations
    sudo apt-get remove -y docker docker-engine docker.io containerd runc 2>/dev/null || true
    
    # Add Docker repository
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    log "✅ Docker installed successfully"
}

# Install NVIDIA Container Toolkit
install_nvidia_container_toolkit() {
    log "🚀 Installing NVIDIA Container Toolkit..."
    
    if docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 nvidia-smi &>/dev/null; then
        log "NVIDIA Container Toolkit already configured"
        return
    fi
    
    # Add NVIDIA package repositories (fixed for Ubuntu 24.04 compatibility)
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    
    # Install the toolkit
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    
    # Configure Docker to use NVIDIA runtime
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    
    # Test GPU access
    if docker run --rm --gpus all nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04 nvidia-smi; then
        log "✅ NVIDIA Container Toolkit installed and configured"
    else
        error "❌ Failed to configure NVIDIA Container Toolkit"
    fi
}

# Setup application directories
setup_directories() {
    log "📁 Setting up application directories..."
    
    # Create base directory structure
    sudo mkdir -p /opt/geogpt-rag/{model_cache,huggingface_cache,data,logs,split_chunks}
    
    # Set basic permissions for the ubuntu user
    sudo chown -R $USER:docker /opt/geogpt-rag
    chmod -R 755 /opt/geogpt-rag
    
    # Create symlinks for easy access
    ln -sf /opt/geogpt-rag ~/geogpt-rag-data
    
    # Note: Container-specific permissions (data/uploads) are set in Dockerfile
    log "✅ Application directories created"
}

# Clone or update repository
setup_repository() {
    log "📥 Setting up GeoGPT-RAG repository..."
    
    APP_DIR="/opt/geogpt-rag/app"
    
    if [ -d "$APP_DIR/.git" ]; then
        log "Repository exists, updating..."
        cd "$APP_DIR"
        git pull origin main
    else
        log "Cloning repository..."
        sudo mkdir -p "$APP_DIR"
        sudo chown $USER:docker "$APP_DIR"
        
        # Clone the GeoGPT-RAG repository
        git clone https://github.com/Rekklessss/geogpt-mvp.git "$APP_DIR"
        cd "$APP_DIR/geogpt-rag"
    fi
    
    log "✅ Repository setup complete"
}

# Configure environment
setup_environment() {
    log "⚙️ Setting up environment configuration..."
    
    cd /opt/geogpt-rag/app/geogpt-rag
    
    # Copy production environment template
    if [ ! -f .env ]; then
        cp ec2-production.env .env
        log "📝 Created .env file from template"
        warn "⚠️  Please edit .env file with your actual configuration:"
        warn "   - SAGEMAKER_ENDPOINT_NAME"
        warn "   - ZILLIZ_URI and ZILLIZ_TOKEN"
        warn "   - AWS credentials (if not using IAM roles)"
        echo
        read -p "Press Enter to edit the environment file now..." -r
        nano .env
    else
        log "Environment file already exists"
    fi
    
    # Verify critical environment variables
    source .env
    
    if [ -z "$SAGEMAKER_ENDPOINT_NAME" ] || [ "$SAGEMAKER_ENDPOINT_NAME" = "your-geogpt-llm-endpoint-name" ]; then
        error "❌ Please configure SAGEMAKER_ENDPOINT_NAME in .env file"
    fi
    
    if [ -z "$ZILLIZ_URI" ] || [ "$ZILLIZ_URI" = "https://your-cluster.vectordb.zilliz.com:19530" ]; then
        error "❌ Please configure ZILLIZ_URI and ZILLIZ_TOKEN in .env file"
    fi
    
    log "✅ Environment configuration verified"
}

# Build and deploy application
deploy_application() {
    log "🚀 Building and deploying GeoGPT-RAG application..."
    
    cd /opt/geogpt-rag/app/geogpt-rag
    
    # Stop existing containers
    docker compose down 2>/dev/null || true
    
    # Build the application
    log "🔨 Building Docker image..."
    docker compose build --no-cache
    
    # Start the application
    log "🌟 Starting GeoGPT-RAG service..."
    docker compose up -d
    
    log "✅ Application deployment initiated"
}

# Monitor deployment
monitor_deployment() {
    log "👀 Monitoring deployment status..."
    
    cd /opt/geogpt-rag/app/geogpt-rag
    
    # Wait for container to start
    log "Waiting for container to start..."
    sleep 10
    
    # Show container status
    docker compose ps
    
    # Follow logs for initial startup
    log "📋 Showing startup logs (first 60 seconds)..."
    timeout 60 docker compose logs -f geogpt-rag || true
    
    # Check health
    log "🔍 Checking application health..."
    
    for i in {1..30}; do
        if curl -f http://localhost:8000/health &>/dev/null; then
            log "✅ Application is healthy!"
            break
        else
            log "⏳ Waiting for application to be ready... ($i/30)"
            sleep 10
        fi
    done
    
    # Final status check
    if curl -f http://localhost:8000/health &>/dev/null; then
        log "🎉 GeoGPT-RAG deployment successful!"
        # Get public IP for final URLs
        FINAL_PUBLIC_IP=$(if [ -n "$TOKEN" ]; then curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null; else curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null; fi || echo "$PUBLIC_IP")
        log "🌐 API available at: http://$FINAL_PUBLIC_IP:8000"
        log "📚 API Documentation: http://$FINAL_PUBLIC_IP:8000/docs"
    else
        error "❌ Application failed to start properly. Check logs with: docker compose logs"
    fi
}

# Test the deployment
test_deployment() {
    log "🧪 Running deployment tests..."
    
    # Test basic endpoints
    log "Testing health endpoint..."
    curl -f http://localhost:8000/health | jq .
    
    log "Testing LLM connection..."
    curl -f http://localhost:8000/llm/test | jq .
    
    log "Testing stats endpoint..."
    curl -f http://localhost:8000/stats | jq .
    
    log "✅ Basic tests passed"
}

# Setup systemd service for auto-restart
setup_service() {
    log "⚙️ Setting up systemd service..."
    
    cat << EOF | sudo tee /etc/systemd/system/geogpt-rag.service
[Unit]
Description=GeoGPT-RAG Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/geogpt-rag/app/geogpt-rag
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=600

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable geogpt-rag.service
    
    log "✅ Systemd service configured"
}

# Main deployment function
main() {
    log "🚀 Starting GeoGPT-RAG deployment on EC2 g5.xlarge"
    log "==============================================="
    
    check_instance
    install_dependencies
    install_docker
    install_nvidia_container_toolkit
    setup_directories
    setup_repository
    setup_environment
    deploy_application
    monitor_deployment
    test_deployment
    setup_service
    
    log "🎉 Deployment completed successfully!"
    log "==============================================="
    log "📋 Next Steps:"
    log "1. Test your SageMaker endpoint: curl http://localhost:8000/llm/test"
    log "2. Upload documents: curl -X POST http://localhost:8000/upload -F 'file=@your-doc.pdf'"
    log "3. Query the system: curl -X POST http://localhost:8000/query -H 'Content-Type: application/json' -d '{\"query\":\"your question\"}'"
    log "4. Monitor with: docker compose logs -f"
    log ""
    log "🌐 Access your API at: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
    log ""
    warn "📝 For future updates/redeployments, use: ./cleanup-deployment.sh"
    warn "   This script is only for fresh EC2 instance setup!"
}

# Run main function
main "$@" 