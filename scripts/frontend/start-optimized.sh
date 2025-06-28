#!/bin/bash

echo "Starting Recursia with Memory Optimizations..."
echo "============================================"

# Kill any existing processes
echo "Stopping any existing servers..."
pkill -f "python.*api_server" 2>/dev/null
pkill -f "node.*vite" 2>/dev/null
sleep 2

# Get the project root (three levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Start API server
echo "Starting API server on port 8080..."
cd "$PROJECT_ROOT"

# Detect and activate virtual environment - cross-platform approach
VENV_DIR="${RECURSIA_VENV_DIR:-venv}"

# Try different activation scripts based on platform
if [ -d "$VENV_DIR/bin" ]; then
    source "$VENV_DIR/bin/activate"
elif [ -d "$VENV_DIR/Scripts" ]; then
    source "$VENV_DIR/Scripts/activate"
elif [ -d "venv/bin" ]; then
    source "venv/bin/activate"
elif [ -d "venv/Scripts" ]; then
    source "venv/Scripts/activate"
fi
python -m src.api_server > api_server.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

# Wait for API to be ready
echo "Waiting for API server to start..."
sleep 3

# Start frontend
echo "Starting frontend dev server..."
cd "$PROJECT_ROOT/frontend"
npm run dev > dev_server.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend server started with PID: $FRONTEND_PID"

# Wait for frontend to be ready
echo "Waiting for frontend to start..."
sleep 3

# Display status
echo ""
echo "============================================"
echo "Servers are running!"
echo "Frontend: http://localhost:3000"
echo "API: http://localhost:8080"
echo ""
echo "Memory optimizations applied:"
echo "- Dynamic grid sizing based on available memory"
echo "- Optimized SimulationHarness with lazy loading"
echo "- Cached potential calculations"
echo "- Memory monitoring with auto-cleanup"
echo ""
echo "To stop servers: kill $API_PID $FRONTEND_PID"
echo "Or run: pkill -f 'python.*api_server|node.*vite'"
echo ""
echo "Check logs:"
echo "- API: tail -f api_server.log"
echo "- Frontend: tail -f dev_server.log"
echo "============================================"