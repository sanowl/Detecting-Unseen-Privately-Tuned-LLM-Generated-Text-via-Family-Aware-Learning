#!/bin/bash
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