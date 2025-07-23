# IPAI Deployment Guide

## Overview

This guide covers deploying IPAI in various environments, from local development to production cloud infrastructure.

## Deployment Options

### 1. Docker Compose (Recommended for Development)

#### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+
- 8GB RAM minimum
- 20GB disk space

#### Steps

1. **Clone and Configure**
```bash
git clone https://github.com/GreatPyreneseDad/IPAI.git
cd IPAI
cp .env.example .env
```

2. **Edit Environment Variables**
```bash
# Edit .env file with your settings
nano .env
```

Key variables to update:
- `SECRET_KEY`: Generate with `openssl rand -hex 32`
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string
- `OLLAMA_HOST`: Ollama server URL

3. **Start Services**
```bash
# Development
docker-compose up -d

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

4. **Initialize Database**
```bash
docker-compose exec backend python scripts/setup_database.py
```

5. **Verify Deployment**
```bash
# Check service health
curl http://localhost:8000/health

# View logs
docker-compose logs -f
```

### 2. Kubernetes Deployment

#### Prerequisites
- Kubernetes cluster (1.20+)
- kubectl configured
- Helm 3 (optional)
- Ingress controller
- cert-manager (for SSL)

#### Steps

1. **Create Namespace**
```bash
kubectl apply -f k8s/namespace.yaml
```

2. **Create Secrets**
```bash
# Create secret for sensitive data
kubectl create secret generic ipai-secrets \
  --from-literal=database-url="postgresql://user:pass@host/db" \
  --from-literal=redis-url="redis://redis:6379" \
  --from-literal=secret-key="your-secret-key" \
  --namespace=ipai
```

3. **Deploy Services**
```bash
# Deploy all components
kubectl apply -f k8s/

# Or use kustomize
kubectl apply -k k8s/
```

4. **Configure Ingress**
```bash
# Edit ingress.yaml with your domain
kubectl apply -f k8s/ingress.yaml
```

5. **Monitor Deployment**
```bash
# Check pod status
kubectl get pods -n ipai

# View logs
kubectl logs -f deployment/ipai-backend -n ipai
```

### 3. Cloud Platform Deployments

#### AWS ECS

1. **Build and Push Images**
```bash
# Build images
docker build -t ipai-backend .
docker build -t ipai-frontend ./frontend

# Tag for ECR
docker tag ipai-backend:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ipai-backend:latest
docker tag ipai-frontend:latest $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ipai-frontend:latest

# Push to ECR
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ipai-backend:latest
docker push $AWS_ACCOUNT.dkr.ecr.$REGION.amazonaws.com/ipai-frontend:latest
```

2. **Deploy with CloudFormation**
```bash
aws cloudformation create-stack \
  --stack-name ipai-production \
  --template-body file://aws/cloudformation.yaml \
  --parameters file://aws/parameters.json
```

#### Google Cloud Run

1. **Build and Push to GCR**
```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and push
docker build -t gcr.io/$PROJECT_ID/ipai-backend .
docker push gcr.io/$PROJECT_ID/ipai-backend
```

2. **Deploy to Cloud Run**
```bash
gcloud run deploy ipai-backend \
  --image gcr.io/$PROJECT_ID/ipai-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="DATABASE_URL=$DATABASE_URL,REDIS_URL=$REDIS_URL"
```

#### Azure Container Instances

```bash
# Create resource group
az group create --name ipai-rg --location eastus

# Create container instance
az container create \
  --resource-group ipai-rg \
  --name ipai-backend \
  --image ipai/backend:latest \
  --dns-name-label ipai-api \
  --ports 8000 \
  --environment-variables \
    DATABASE_URL=$DATABASE_URL \
    REDIS_URL=$REDIS_URL
```

## Production Configuration

### SSL/TLS Setup

1. **Let's Encrypt with Certbot**
```bash
# Install certbot
sudo apt-get install certbot python3-certbot-nginx

# Generate certificate
sudo certbot --nginx -d ipai.app -d www.ipai.app -d api.ipai.app
```

2. **Configure Nginx**
```nginx
server {
    listen 443 ssl http2;
    server_name api.ipai.app;
    
    ssl_certificate /etc/letsencrypt/live/ipai.app/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/ipai.app/privkey.pem;
    
    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Database Backup

1. **Automated Backups**
```bash
# Create backup script
cat > /opt/ipai/backup.sh << 'EOF'
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="ipai"

# Create backup
docker-compose exec -T postgres pg_dump -U ipai $DB_NAME | gzip > $BACKUP_DIR/ipai_$DATE.sql.gz

# Upload to S3
aws s3 cp $BACKUP_DIR/ipai_$DATE.sql.gz s3://ipai-backups/

# Keep only last 7 days locally
find $BACKUP_DIR -name "*.sql.gz" -mtime +7 -delete
EOF

# Add to crontab
0 2 * * * /opt/ipai/backup.sh
```

### Monitoring

1. **Prometheus Configuration**
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ipai-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
```

2. **Grafana Dashboard**
Import dashboard from `monitoring/grafana-dashboard.json`

### Security Hardening

1. **Environment Variables**
```bash
# Never commit .env files
echo ".env*" >> .gitignore

# Use secrets management
# AWS Secrets Manager
aws secretsmanager create-secret --name ipai-prod --secret-string file://secrets.json

# Kubernetes Secrets
kubectl create secret generic ipai-secrets --from-env-file=.env.production
```

2. **Network Security**
```bash
# Firewall rules
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable
```

3. **Container Security**
```dockerfile
# Run as non-root user
USER 1000:1000

# Security scanning
docker scan ipai-backend:latest
```

## Scaling

### Horizontal Scaling

1. **Docker Swarm**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ipai

# Scale service
docker service scale ipai_backend=5
```

2. **Kubernetes HPA**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ipai-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ipai-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Database Scaling

1. **Read Replicas**
```python
# Configure in settings
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'HOST': 'master.db.ipai.com',
        'NAME': 'ipai',
    },
    'replica': {
        'ENGINE': 'django.db.backends.postgresql',
        'HOST': 'replica.db.ipai.com',
        'NAME': 'ipai',
    }
}
```

2. **Connection Pooling**
```python
# PgBouncer configuration
[databases]
ipai = host=localhost port=5432 dbname=ipai

[pgbouncer]
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
```

## Troubleshooting

### Common Issues

1. **Database Connection Failed**
```bash
# Check connectivity
docker-compose exec backend pg_isready -h postgres

# Check credentials
docker-compose exec backend env | grep DATABASE

# View postgres logs
docker-compose logs postgres
```

2. **Redis Connection Failed**
```bash
# Test redis connection
docker-compose exec backend redis-cli -h redis ping

# Check redis logs
docker-compose logs redis
```

3. **Ollama Not Responding**
```bash
# Check ollama status
curl http://localhost:11434/api/tags

# Pull required model
docker-compose exec ollama ollama pull llama3.2
```

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

### Performance Tuning

1. **PostgreSQL**
```sql
-- Tune postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
work_mem = 4MB
max_connections = 200
```

2. **Gunicorn Workers**
```python
# gunicorn.conf.py
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
```

## Rollback Procedures

1. **Docker Compose**
```bash
# Tag current version
docker tag ipai-backend:latest ipai-backend:rollback

# Deploy previous version
docker-compose down
docker tag ipai-backend:v1.0.0 ipai-backend:latest
docker-compose up -d
```

2. **Kubernetes**
```bash
# Rollback deployment
kubectl rollout undo deployment/ipai-backend -n ipai

# Check rollout status
kubectl rollout status deployment/ipai-backend -n ipai
```

## Maintenance

### Regular Tasks

1. **Update Dependencies**
```bash
# Backend
pip install --upgrade -r requirements.txt

# Frontend
cd frontend && npm update
```

2. **Database Maintenance**
```sql
-- Vacuum and analyze
VACUUM ANALYZE;

-- Reindex
REINDEX DATABASE ipai;
```

3. **Log Rotation**
```bash
# Configure logrotate
cat > /etc/logrotate.d/ipai << EOF
/var/log/ipai/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
EOF
```