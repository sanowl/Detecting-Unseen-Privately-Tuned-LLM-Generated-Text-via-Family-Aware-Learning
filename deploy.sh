#!/bin/bash
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