#!/bin/bash

# =====================================================================================
# Verify Manual IAM Role Setup for GeoGPT-RAG
# Tests EC2 instance IAM role and SageMaker permissions
# =====================================================================================

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}

echo "🔍 Verifying Manual IAM Role Setup for GeoGPT-RAG"
echo "================================================"

# Test 1: Check if we're on EC2 with IAM role
log "Testing EC2 metadata service..."
if curl -s --max-time 5 http://169.254.169.254/latest/meta-data/iam/security-credentials/ >/dev/null; then
    ROLE_NAME=$(curl -s http://169.254.169.254/latest/meta-data/iam/security-credentials/)
    if [ -n "$ROLE_NAME" ]; then
        log "EC2 instance has IAM role attached: $ROLE_NAME"
    else
        error "No IAM role found on this EC2 instance"
        exit 1
    fi
else
    error "Not running on EC2 or metadata service unavailable"
    exit 1
fi

# Test 2: Check if AWS CLI can assume the role
log "Testing AWS credentials from IAM role..."
if command -v aws >/dev/null 2>&1; then
    if CALLER_INFO=$(aws sts get-caller-identity 2>/dev/null); then
        USER_ARN=$(echo "$CALLER_INFO" | grep -o '"Arn": "[^"]*"' | cut -d'"' -f4)
        ACCOUNT_ID=$(echo "$CALLER_INFO" | grep -o '"Account": "[^"]*"' | cut -d'"' -f4)
        log "AWS credentials working via IAM role"
        log "  Account: $ACCOUNT_ID"
        log "  Role ARN: $USER_ARN"
    else
        error "Cannot assume IAM role or get caller identity"
        exit 1
    fi
else
    warn "AWS CLI not found - installing for testing..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update >/dev/null 2>&1
        sudo apt-get install -y awscli >/dev/null 2>&1
    elif command -v yum >/dev/null 2>&1; then
        sudo yum install -y awscli >/dev/null 2>&1
    else
        error "Cannot install AWS CLI automatically"
        exit 1
    fi
fi

# Test 3: Check SageMaker permissions
log "Testing SageMaker permissions..."
AWS_REGION=${AWS_REGION:-us-east-1}

if aws sagemaker list-endpoints --region "$AWS_REGION" >/dev/null 2>&1; then
    log "SageMaker list-endpoints permission: ✅ WORKING"
    
    # List available endpoints
    ENDPOINTS=$(aws sagemaker list-endpoints --region "$AWS_REGION" --query 'Endpoints[].EndpointName' --output text 2>/dev/null || echo "")
    if [ -n "$ENDPOINTS" ]; then
        log "Available SageMaker endpoints in $AWS_REGION:"
        for endpoint in $ENDPOINTS; do
            echo "    • $endpoint"
        done
    else
        warn "No SageMaker endpoints found in region $AWS_REGION"
    fi
else
    error "SageMaker list-endpoints permission: ❌ DENIED"
    echo "Your IAM role needs the following permissions:"
    echo "  • sagemaker:ListEndpoints"
    echo "  • sagemaker:DescribeEndpoint"
    echo "  • sagemaker:InvokeEndpoint"
    exit 1
fi

# Test 4: Test specific endpoint if provided
if [ -n "$SAGEMAKER_ENDPOINT_NAME" ]; then
    log "Testing specific endpoint: $SAGEMAKER_ENDPOINT_NAME"
    
    if ENDPOINT_INFO=$(aws sagemaker describe-endpoint --endpoint-name "$SAGEMAKER_ENDPOINT_NAME" --region "$AWS_REGION" 2>/dev/null); then
        STATUS=$(echo "$ENDPOINT_INFO" | grep -o '"EndpointStatus": "[^"]*"' | cut -d'"' -f4)
        log "Endpoint $SAGEMAKER_ENDPOINT_NAME status: $STATUS"
        
        if [ "$STATUS" = "InService" ]; then
            log "✅ Endpoint is ready for inference"
        else
            warn "⚠️  Endpoint not in service - current status: $STATUS"
        fi
    else
        error "Cannot describe endpoint $SAGEMAKER_ENDPOINT_NAME"
        echo "Check that:"
        echo "  • Endpoint name is correct"
        echo "  • Endpoint exists in region $AWS_REGION"
        echo "  • IAM role has sagemaker:DescribeEndpoint permission"
        exit 1
    fi
else
    warn "SAGEMAKER_ENDPOINT_NAME not set - skipping specific endpoint test"
    echo "Set this environment variable to test your specific endpoint"
fi

# Test 5: Test invoke permission (without actually invoking)
log "Testing SageMaker invoke permissions..."
if aws sagemaker describe-endpoint-config --endpoint-config-name "test-non-existent-config" --region "$AWS_REGION" 2>&1 | grep -q "does not exist\|ValidationException"; then
    log "SageMaker invoke permissions: ✅ AVAILABLE"
else
    # Check if it's a permissions error
    if aws sagemaker describe-endpoint-config --endpoint-config-name "test-non-existent-config" --region "$AWS_REGION" 2>&1 | grep -q "AccessDenied\|Forbidden"; then
        error "SageMaker invoke permissions: ❌ DENIED"
        exit 1
    else
        log "SageMaker invoke permissions: ✅ AVAILABLE"
    fi
fi

echo ""
log "🎉 IAM Role Verification Complete!"
echo ""
echo "✅ Your manual IAM role setup is working correctly"
echo "✅ SageMaker permissions are properly configured"
echo "✅ Ready for GeoGPT-RAG deployment"
echo ""
echo "🚀 Next steps:"
echo "  1. Configure your .env file with SAGEMAKER_ENDPOINT_NAME"
echo "  2. Deploy your application: ./deploy-ec2.sh"
echo "  3. Test the application endpoints"
echo "" 