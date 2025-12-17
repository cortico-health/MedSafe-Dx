#!/bin/bash
#
# Serve the leaderboard HTML using Docker
#

PORT=${1:-18080}

echo "========================================"
echo "Starting Leaderboard Server"
echo "========================================"
echo ""

cd "$(dirname "$0")/.."

echo "Server running at: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

export WEB_PORT=$PORT
docker compose up web --build
