#!/bin/bash
# Comprehensive startup script for Recursia
# Backend: Port 8080
# Frontend: Port 5173

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${GREEN}Starting Recursia Development Environment${NC}"
echo "=========================================="
echo "Backend API: http://localhost:8080"
echo "Frontend UI: http://localhost:5173"
echo "=========================================="

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}Warning: Port $port is already in use${NC}"
        echo "Process using port $port:"
        lsof -i:$port | grep LISTEN
        return 1
    fi
    return 0
}

# Function to kill process on port
kill_port() {
    local port=$1
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}Killing process on port $port${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Check Python virtual environment
if [ ! -d "venv_linux" ]; then
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv_linux
fi

# Activate virtual environment
echo "Activating Python virtual environment..."
source venv_linux/bin/activate

# Install/update Python dependencies
echo "Checking Python dependencies..."
pip install -q -r requirements.txt 2>/dev/null || {
    echo -e "${YELLOW}Installing Python dependencies...${NC}"
    pip install -r requirements.txt
}

# Check Node.js dependencies
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    cd frontend
    npm install
    cd ..
fi

# Kill any existing processes on our ports
echo -e "\n${YELLOW}Checking ports...${NC}"
kill_port 8080
kill_port 5173

# Start backend API server
echo -e "\n${GREEN}Starting Backend API Server on port 8080...${NC}"
export BACKEND_PORT=8080
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Start backend in background
python -m src.api.unified_api_server &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "Waiting for backend to start..."
for i in {1..30}; do
    if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Backend failed to start after 30 seconds${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo

# Run backend tests
echo -e "\n${YELLOW}Running backend API tests...${NC}"
python test_backend_api.py || {
    echo -e "${RED}Backend tests failed!${NC}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
}

# Start frontend
echo -e "\n${GREEN}Starting Frontend on port 5173...${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend to be ready
echo "Waiting for frontend to start..."
for i in {1..30}; do
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend is ready!${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}✗ Frontend failed to start after 30 seconds${NC}"
        kill $BACKEND_PID 2>/dev/null
        kill $FRONTEND_PID 2>/dev/null
        exit 1
    fi
    sleep 1
    echo -n "."
done
echo

# Save PIDs for stop script
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

echo -e "\n${GREEN}=========================================="
echo "✅ Recursia is running!"
echo "=========================================="
echo "Backend API: http://localhost:8080"
echo "Frontend UI: http://localhost:5173"
echo "=========================================="
echo -e "${NC}"
echo "Press Ctrl+C to stop all services"
echo

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    rm -f .backend.pid .frontend.pid
    echo -e "${GREEN}Services stopped.${NC}"
    exit 0
}

# Set up trap for cleanup
trap cleanup INT TERM

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID