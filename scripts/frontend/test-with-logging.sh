#!/bin/bash
# Test script with comprehensive logging

echo "=== RECURSIA FRONTEND TEST WITH LOGGING ==="
echo "This will start both API server and frontend with detailed logging"
echo

# Kill any existing processes
echo "Killing existing processes..."
pkill -f "api_server.py" || true
pkill -f "scripts/backend/run_api_server.py" || true
pkill -f "uvicorn" || true
pkill -f "vite" || true
pkill -f "npm run dev" || true
sleep 2

# Start API server
echo
echo "Starting minimal API server..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
python scripts/backend/run_api_server.py > frontend/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Wait for API to be ready
echo "Waiting for API server to be ready..."
sleep 3

# Check if API is responding
echo "Testing API connection..."
curl -s http://localhost:8080/health || echo "API not responding yet..."

# Start frontend
echo
echo "Starting frontend with logging..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "Frontend started with PID: $FRONTEND_PID"

echo
echo "=== SERVICES RUNNING ==="
echo "API Server PID: $API_PID"
echo "Frontend PID: $FRONTEND_PID"
echo
echo "Logs:"
echo "  API: frontend/api.log"
echo "  Frontend: Check the console output"
echo
echo "Open http://localhost:5173 in your browser"
echo "Press Ctrl+C to stop all services"
echo

# Function to cleanup on exit
cleanup() {
    echo
    echo "Stopping services..."
    kill $API_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    pkill -f "api_server.py" || true
    pkill -f "uvicorn" || true
    pkill -f "vite" || true
    echo "Services stopped."
}

# Set trap to cleanup on exit
trap cleanup EXIT

# Keep script running
wait