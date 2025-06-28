#!/bin/bash
# Stop all Recursia services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping Recursia services...${NC}"

# Function to kill process on port
kill_port() {
    local port=$1
    local name=$2
    if lsof -i:$port > /dev/null 2>&1; then
        echo -e "${YELLOW}Stopping $name on port $port${NC}"
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
        sleep 1
    else
        echo "$name not running on port $port"
    fi
}

# Kill services by port
kill_port 8080 "Backend API"
kill_port 5173 "Frontend"

# Also try to kill by PID files if they exist
if [ -f .backend.pid ]; then
    PID=$(cat .backend.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping backend process $PID"
        kill $PID 2>/dev/null || true
    fi
    rm -f .backend.pid
fi

if [ -f .frontend.pid ]; then
    PID=$(cat .frontend.pid)
    if kill -0 $PID 2>/dev/null; then
        echo "Stopping frontend process $PID"
        kill $PID 2>/dev/null || true
    fi
    rm -f .frontend.pid
fi

echo -e "${GREEN}All services stopped.${NC}"