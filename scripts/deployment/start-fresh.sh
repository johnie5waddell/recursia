#!/bin/bash

echo "=========================================="
echo "Starting Recursia - Fresh Start"
echo "=========================================="

# Kill ALL existing processes
echo "Killing all existing processes..."
pkill -f "python.*recursia" 2>/dev/null
pkill -f "python.*api_server" 2>/dev/null
pkill -f "python.*dashboard" 2>/dev/null
pkill -f "node.*vite" 2>/dev/null
sleep 3

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Start API server
echo -e "${YELLOW}Starting API server...${NC}"
# Get the project root (three levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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

# Start API server with the fixed code
echo -e "${GREEN}Starting API server on port 8080...${NC}"
python -m src.api_server > api_fresh.log 2>&1 &
API_PID=$!
echo -e "${GREEN}API server started with PID: $API_PID${NC}"

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API server...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:8080/api/metrics > /dev/null 2>&1; then
        echo -e "${GREEN}API server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Start frontend
echo -e "${YELLOW}Starting frontend...${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev > dev_fresh.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend started with PID: $FRONTEND_PID${NC}"

# Wait a moment
sleep 3

echo ""
echo -e "${GREEN}=========================================="
echo "Recursia is running!"
echo "=========================================="
echo -e "Frontend: ${NC}http://localhost:3000"
echo -e "${GREEN}API:      ${NC}http://localhost:8080"
echo ""
echo -e "${YELLOW}Key improvements applied:${NC}"
echo "✓ Grid size: 6³ = 216 cells (was 12³ = 1,728)"
echo "✓ Async engine initialization"
echo "✓ Minimal initial state"
echo "✓ Fixed API server OSHMetrics.information error"
echo ""
echo -e "${YELLOW}Monitor logs:${NC}"
echo "tail -f api_fresh.log"
echo "tail -f dev_fresh.log"
echo ""
echo -e "${YELLOW}To stop:${NC}"
echo "kill $API_PID $FRONTEND_PID"
echo "=========================================="