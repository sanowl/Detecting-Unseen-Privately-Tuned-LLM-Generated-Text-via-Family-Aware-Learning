#!/bin/bash
# Stop PhantomHunter services

echo "🛑 Stopping PhantomHunter services..."
docker-compose down

echo "🧹 Cleaning up..."
docker system prune -f

echo "✅ Services stopped"