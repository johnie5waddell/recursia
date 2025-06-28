#!/bin/bash
# Minimal API server startup script
# No dashboards, no visualization, just REST API

echo "Starting minimal Recursia API server..."

# Kill any existing API server processes
pkill -f "api_server.py" || true
pkill -f "scripts/backend/run_api_server.py" || true
pkill -f "uvicorn" || true

# Wait for processes to die
sleep 1

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Run the minimal API server directly
python scripts/backend/run_api_server.py