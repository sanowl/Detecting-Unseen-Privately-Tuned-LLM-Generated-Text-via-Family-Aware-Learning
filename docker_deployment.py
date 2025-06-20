"""
Docker Deployment Configuration for PhantomHunter
Includes Dockerfile, docker-compose, and deployment utilities
"""

DOCKERFILE_CONTENT = '''
# Multi-stage Docker build for PhantomHunter production deployment
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r phantomhunter && useradd -r -g phantomhunter phantomhunter

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
COPY requirements-prod.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-prod.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir pytest black flake8 jupyter

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R phantomhunter:phantomhunter /app

USER phantomhunter

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production

# Copy only necessary files
COPY config.py utils.py dataset.py trainer.py __init__.py ./
COPY models/ ./models/
COPY production_utils.py api_server.py ./

# Create directories for logs and checkpoints
RUN mkdir -p /app/logs /app/checkpoints && \
    chown -R phantomhunter:phantomhunter /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER phantomhunter

EXPOSE 8000

# Use gunicorn for production
CMD ["gunicorn", "api_server:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "120", "--keepalive", "5", \
     "--max-requests", "1000", "--max-requests-jitter", "100", \
     "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", \
     "--log-level", "info"]

# GPU-enabled production stage
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-production

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_VISIBLE_DEVICES=0

# Create non-root user
RUN groupadd -r phantomhunter && useradd -r -g phantomhunter phantomhunter

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt requirements-prod.txt ./
RUN pip3 install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY config.py utils.py dataset.py trainer.py __init__.py ./
COPY models/ ./models/
COPY production_utils.py api_server.py ./

# Create directories and set permissions
RUN mkdir -p /app/logs /app/checkpoints && \
    chown -R phantomhunter:phantomhunter /app

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

USER phantomhunter

EXPOSE 8000

CMD ["gunicorn", "api_server:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--timeout", "180", "--keepalive", "5", \
     "--max-requests", "500", "--access-logfile", "/app/logs/access.log", \
     "--error-logfile", "/app/logs/error.log", "--log-level", "info"]
'''

DOCKER_COMPOSE_CONTENT = '''
version: '3.8'

services:
  phantomhunter-api:
    build:
      context: .
      dockerfile: Dockerfile
      target: production
    ports:
      - "8000:8000"
    environment:
      - ENV=production
      - REDIS_URL=redis://redis:6379
      - DB_URL=postgresql://user:password@postgres:5432/phantomhunter
    volumes:
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
      - model_cache:/app/model_cache
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4.0'
        reservations:
          memory: 4G
          cpus: '2.0'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  phantomhunter-gpu:
    build:
      context: .
      dockerfile: Dockerfile
      target: gpu-production
    ports:
      - "8001:8000"
    environment:
      - ENV=production
      - CUDA_VISIBLE_DEVICES=0
      - REDIS_URL=redis://redis:6379
    volumes:
      - ./logs:/app/logs
      - ./checkpoints:/app/checkpoints
      - model_cache:/app/model_cache
    depends_on:
      - redis
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    profiles:
      - gpu

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru

  postgres:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=phantomhunter
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - phantomhunter-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    depends_on:
      - prometheus
    restart: unless-stopped

volumes:
  redis_data:
  postgres_data:
  prometheus_data:
  grafana_data:
  model_cache:
  nginx_logs:

networks:
  default:
    driver: bridge
'''

REQUIREMENTS_PROD_CONTENT = '''
# Production requirements for PhantomHunter
torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
datasets>=2.12.0
accelerate>=0.20.0

# API server
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
gunicorn>=21.0.0
pydantic>=2.0.0

# Authentication and security
pyjwt>=2.7.0
cryptography>=41.0.0
python-multipart>=0.0.6

# Rate limiting and caching
slowapi>=0.1.8
redis>=4.5.0
hiredis>=2.2.0

# Database
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
alembic>=1.11.0

# Monitoring and logging
prometheus-client>=0.17.0
structlog>=23.1.0
python-json-logger>=2.0.0

# Performance
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.3.0
pandas>=1.5.0

# NLP utilities
nltk>=3.8
spacy>=3.6.0
sentence-transformers>=2.2.0

# Async and concurrency
aioredis>=2.0.0
asyncpg>=0.28.0

# System monitoring
psutil>=5.9.0
GPUtil>=1.4.0

# Production utilities
python-dotenv>=1.0.0
click>=8.1.0
rich>=13.0.0
'''

NGINX_CONFIG = '''
events {
    worker_connections 1024;
}

http {
    upstream phantomhunter_backend {
        server phantomhunter-api:8000;
        # Add more servers for load balancing
        # server phantomhunter-api-2:8000;
    }

    upstream phantomhunter_gpu_backend {
        server phantomhunter-gpu:8000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=auth_limit:10m rate=1r/s;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Gzip compression
    gzip on;
    gzip_types text/plain application/json application/javascript text/css;

    # API server
    server {
        listen 80;
        server_name api.phantomhunter.example.com;

        # Redirect to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name api.phantomhunter.example.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
        ssl_prefer_server_ciphers off;

        # Request size limits
        client_max_body_size 10M;

        # API endpoints
        location /api/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://phantomhunter_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeouts
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # GPU endpoints (for heavy workloads)
        location /api/gpu/ {
            limit_req zone=api_limit burst=5 nodelay;
            
            proxy_pass http://phantomhunter_gpu_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Longer timeouts for GPU processing
            proxy_connect_timeout 120s;
            proxy_send_timeout 120s;
            proxy_read_timeout 300s;
        }

        # Authentication endpoints
        location /auth/ {
            limit_req zone=auth_limit burst=5 nodelay;
            
            proxy_pass http://phantomhunter_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://phantomhunter_backend;
            access_log off;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
'''

PROMETHEUS_CONFIG = '''
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'phantomhunter-api'
    static_configs:
      - targets: ['phantomhunter-api:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'nginx'
    static_configs:
      - targets: ['nginx:80']
    
  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
'''

KUBERNETES_DEPLOYMENT = '''
apiVersion: apps/v1
kind: Deployment
metadata:
  name: phantomhunter-api
  labels:
    app: phantomhunter-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: phantomhunter-api
  template:
    metadata:
      labels:
        app: phantomhunter-api
    spec:
      containers:
      - name: phantomhunter-api
        image: phantomhunter:latest
        ports:
        - containerPort: 8000
        env:
        - name: ENV
          value: "production"
        - name: REDIS_URL
          value: "redis://redis-service:6379"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-cache
          mountPath: /app/model_cache
        - name: logs
          mountPath: /app/logs
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: logs
        persistentVolumeClaim:
          claimName: logs-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: phantomhunter-service
spec:
  selector:
    app: phantomhunter-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: logs-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
'''

def create_deployment_files():
    """Create all deployment files"""
    files = {
        'Dockerfile': DOCKERFILE_CONTENT,
        'docker-compose.yml': DOCKER_COMPOSE_CONTENT,
        'requirements-prod.txt': REQUIREMENTS_PROD_CONTENT,
        'nginx.conf': NGINX_CONFIG,
        'prometheus.yml': PROMETHEUS_CONFIG,
        'k8s-deployment.yaml': KUBERNETES_DEPLOYMENT
    }
    
    for filename, content in files.items():
        with open(filename, 'w') as f:
            f.write(content.strip())
    
    print("✅ Created deployment files:")
    for filename in files.keys():
        print(f"  📄 {filename}")

def create_deployment_scripts():
    """Create deployment and management scripts"""
    
    deploy_script = '''#!/bin/bash
# PhantomHunter deployment script

set -e

echo "🚀 Starting PhantomHunter deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed"
    exit 1
fi

# Build and start services
echo "🔨 Building images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 30

# Check health
echo "🔍 Checking service health..."
curl -f http://localhost:8000/health || {
    echo "❌ Health check failed"
    docker-compose logs phantomhunter-api
    exit 1
}

echo "✅ PhantomHunter deployed successfully!"
echo "📊 API available at: http://localhost:8000"
echo "📈 Grafana dashboard: http://localhost:3000"
echo "🔍 Prometheus metrics: http://localhost:9090"
'''

    stop_script = '''#!/bin/bash
# Stop PhantomHunter services

echo "🛑 Stopping PhantomHunter services..."
docker-compose down

echo "🧹 Cleaning up..."
docker system prune -f

echo "✅ Services stopped"
'''

    logs_script = '''#!/bin/bash
# View PhantomHunter logs

if [ "$1" = "api" ]; then
    docker-compose logs -f phantomhunter-api
elif [ "$1" = "gpu" ]; then
    docker-compose logs -f phantomhunter-gpu
elif [ "$1" = "nginx" ]; then
    docker-compose logs -f nginx
else
    echo "Usage: ./logs.sh [api|gpu|nginx]"
    docker-compose logs -f
fi
'''

    scripts = {
        'deploy.sh': deploy_script,
        'stop.sh': stop_script,
        'logs.sh': logs_script
    }
    
    for filename, content in scripts.items():
        with open(filename, 'w') as f:
            f.write(content.strip())
        
        # Make executable
        import os
        os.chmod(filename, 0o755)
    
    print("\n✅ Created deployment scripts:")
    for filename in scripts.keys():
        print(f"  📜 {filename}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create":
        create_deployment_files()
        create_deployment_scripts()
        
        print("\n🎉 Deployment configuration created!")
        print("\nNext steps:")
        print("1. ./deploy.sh - Deploy the application")
        print("2. ./logs.sh api - View API logs")
        print("3. ./stop.sh - Stop all services")
    else:
        print("Usage: python docker_deployment.py create") 