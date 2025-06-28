#!/bin/bash

echo "=========================================="
echo "Starting Recursia Production Environment"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Kill any existing processes
echo -e "${YELLOW}Cleaning up existing processes...${NC}"
pkill -f "python.*api_server" 2>/dev/null
pkill -f "node.*vite" 2>/dev/null
sleep 2

# Check Python environment
echo -e "${YELLOW}Checking Python environment...${NC}"
# Get the project root (three levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ ! -d "$PROJECT_ROOT/venv_linux" ]; then
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Creating virtual environment..."
    cd "$PROJECT_ROOT"
    python3 -m venv venv_linux
    source venv_linux/bin/activate
    pip install -r requirements.txt
else
    cd "$PROJECT_ROOT"
    source venv_linux/bin/activate
fi

# Start API server with fixed code
echo -e "${YELLOW}Starting API server on port 8080...${NC}"
python -m src.api_server > api_server_production.log 2>&1 &
API_PID=$!
echo -e "${GREEN}API server started with PID: $API_PID${NC}"

# Wait for API to be ready
echo -e "${YELLOW}Waiting for API server to initialize...${NC}"
for i in {1..10}; do
    if curl -s http://localhost:8080/api/metrics > /dev/null 2>&1; then
        echo -e "${GREEN}API server is ready!${NC}"
        break
    fi
    echo -n "."
    sleep 1
done
echo ""

# Check if API is responding
if ! curl -s http://localhost:8080/api/metrics > /dev/null 2>&1; then
    echo -e "${RED}WARNING: API server may not be responding correctly${NC}"
    echo "Check api_server_production.log for errors"
fi

# Start frontend
echo -e "${YELLOW}Starting frontend dev server...${NC}"
cd "$PROJECT_ROOT/frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi

npm run dev > dev_server_production.log 2>&1 &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend server started with PID: $FRONTEND_PID${NC}"

# Wait for frontend to be ready
echo -e "${YELLOW}Waiting for frontend to start...${NC}"
sleep 5

# Display status
echo ""
echo -e "${GREEN}=========================================="
echo -e "Recursia is running!"
echo -e "=========================================="
echo -e "Frontend: ${NC}http://localhost:3000"
echo -e "${GREEN}API:      ${NC}http://localhost:8080"
echo -e "${GREEN}API Docs: ${NC}http://localhost:8080/docs"
echo ""
echo -e "${YELLOW}Process IDs:${NC}"
echo "- API Server: $API_PID"
echo "- Frontend:   $FRONTEND_PID"
echo ""
echo -e "${YELLOW}Logs:${NC}"
echo "- API: tail -f $PROJECT_ROOT/api_server_production.log"
echo "- Frontend: tail -f $PROJECT_ROOT/frontend/dev_server_production.log"
echo ""
echo -e "${YELLOW}To stop all services:${NC}"
echo "kill $API_PID $FRONTEND_PID"
echo "Or run: pkill -f 'python.*api_server|node.*vite'"
echo ""
echo -e "${GREEN}Memory optimizations applied:${NC}"
echo "✓ Dynamic grid sizing (4³-12³ cells based on memory)"
echo "✓ Optimized SimulationHarness with lazy loading"
echo "✓ Memory monitoring with auto-cleanup at 90% usage"
echo "✓ Cached potential calculations"
echo "✓ API server OSHMetrics.information fix applied"
echo ""
echo -e "${GREEN}=========================================="