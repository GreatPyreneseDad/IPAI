#!/bin/bash
# IPAI Deployment Script

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="production"
DEPLOY_TARGET="docker"
VERSION="latest"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--target)
            DEPLOY_TARGET="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo "  -e, --environment  Environment (development|staging|production)"
            echo "  -t, --target       Deploy target (docker|k8s)"
            echo "  -v, --version      Version tag"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}IPAI Deployment Script${NC}"
echo "Environment: $ENVIRONMENT"
echo "Deploy Target: $DEPLOY_TARGET"
echo "Version: $VERSION"
echo ""

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}Checking prerequisites...${NC}"
    
    if [[ "$DEPLOY_TARGET" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}Docker is not installed${NC}"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            echo -e "${RED}Docker Compose is not installed${NC}"
            exit 1
        fi
    elif [[ "$DEPLOY_TARGET" == "k8s" ]]; then
        if ! command -v kubectl &> /dev/null; then
            echo -e "${RED}kubectl is not installed${NC}"
            exit 1
        fi
    fi
    
    echo -e "${GREEN}Prerequisites check passed${NC}"
}

# Function to build images
build_images() {
    echo -e "${YELLOW}Building Docker images...${NC}"
    
    # Build backend
    echo "Building backend image..."
    docker build -t ipai/backend:$VERSION .
    
    # Build frontend
    echo "Building frontend image..."
    docker build -t ipai/frontend:$VERSION ./frontend
    
    echo -e "${GREEN}Images built successfully${NC}"
}

# Function to deploy with Docker Compose
deploy_docker() {
    echo -e "${YELLOW}Deploying with Docker Compose...${NC}"
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
    else
        docker-compose up -d
    fi
    
    echo -e "${GREEN}Docker deployment completed${NC}"
}

# Function to deploy to Kubernetes
deploy_k8s() {
    echo -e "${YELLOW}Deploying to Kubernetes...${NC}"
    
    # Apply namespace
    kubectl apply -f k8s/namespace.yaml
    
    # Create secrets (you should manage these securely)
    echo "Creating secrets..."
    kubectl create secret generic ipai-secrets \
        --from-literal=database-url=$DATABASE_URL \
        --from-literal=redis-url=$REDIS_URL \
        --from-literal=secret-key=$SECRET_KEY \
        --namespace=ipai \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply deployments
    kubectl apply -f k8s/backend-deployment.yaml
    kubectl apply -f k8s/frontend-deployment.yaml
    kubectl apply -f k8s/ingress.yaml
    
    echo -e "${GREEN}Kubernetes deployment completed${NC}"
}

# Function to run database migrations
run_migrations() {
    echo -e "${YELLOW}Running database migrations...${NC}"
    
    if [[ "$DEPLOY_TARGET" == "docker" ]]; then
        docker-compose exec backend python scripts/setup_database.py
    elif [[ "$DEPLOY_TARGET" == "k8s" ]]; then
        kubectl exec -it deployment/ipai-backend -n ipai -- python scripts/setup_database.py
    fi
    
    echo -e "${GREEN}Migrations completed${NC}"
}

# Function to perform health check
health_check() {
    echo -e "${YELLOW}Performing health check...${NC}"
    
    if [[ "$DEPLOY_TARGET" == "docker" ]]; then
        HEALTH_URL="http://localhost:8000/health"
    else
        HEALTH_URL="https://api.ipai.app/health"
    fi
    
    # Wait for service to be ready
    for i in {1..30}; do
        if curl -s $HEALTH_URL > /dev/null; then
            echo -e "${GREEN}Health check passed${NC}"
            return 0
        fi
        echo "Waiting for service to be ready... ($i/30)"
        sleep 2
    done
    
    echo -e "${RED}Health check failed${NC}"
    return 1
}

# Main deployment flow
main() {
    check_prerequisites
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        # Load production environment variables
        if [ -f .env.production ]; then
            export $(cat .env.production | grep -v '^#' | xargs)
        else
            echo -e "${RED}.env.production file not found${NC}"
            exit 1
        fi
    fi
    
    build_images
    
    if [[ "$DEPLOY_TARGET" == "docker" ]]; then
        deploy_docker
    elif [[ "$DEPLOY_TARGET" == "k8s" ]]; then
        deploy_k8s
    else
        echo -e "${RED}Unknown deploy target: $DEPLOY_TARGET${NC}"
        exit 1
    fi
    
    # Wait a bit for services to start
    sleep 10
    
    run_migrations
    health_check
    
    echo -e "${GREEN}Deployment completed successfully!${NC}"
}

# Run main function
main