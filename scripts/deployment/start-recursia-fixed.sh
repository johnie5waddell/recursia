#!/bin/bash

# Fixed startup script with proper process management
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Lock file to prevent multiple instances
LOCK_FILE="/tmp/recursia.lock"
BACKEND_PID_FILE="/tmp/recursia_backend.pid"
FRONTEND_PID_FILE="/tmp/recursia_frontend.pid"

# Check if already running
if [ -f "$LOCK_FILE" ]; then
    # Check if the PID in the lock file is still running
    if [ -f "$BACKEND_PID_FILE" ]; then
        OLD_PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo -e "${RED}Recursia is already running (Backend PID: $OLD_PID)${NC}"
            echo -e "${YELLOW}Use ./scripts/deployment/stop-recursia.sh to stop it first${NC}"
            exit 1
        fi
    fi
    # Stale lock file, remove it
    rm -f "$LOCK_FILE" "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
fi

# Create lock file
touch "$LOCK_FILE"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down Recursia...${NC}"
    
    # Kill backend
    if [ -f "$BACKEND_PID_FILE" ]; then
        BACKEND_PID=$(cat "$BACKEND_PID_FILE")
        if kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo -e "${CYAN}Stopping backend (PID: $BACKEND_PID)...${NC}"
            kill -TERM "$BACKEND_PID" 2>/dev/null || true
            sleep 2
            kill -KILL "$BACKEND_PID" 2>/dev/null || true
        fi
    fi
    
    # Kill frontend
    if [ -f "$FRONTEND_PID_FILE" ]; then
        FRONTEND_PID=$(cat "$FRONTEND_PID_FILE")
        if kill -0 "$FRONTEND_PID" 2>/dev/null; then
            echo -e "${CYAN}Stopping frontend (PID: $FRONTEND_PID)...${NC}"
            kill -TERM "$FRONTEND_PID" 2>/dev/null || true
            sleep 2
            kill -KILL "$FRONTEND_PID" 2>/dev/null || true
        fi
    fi
    
    # Remove lock files
    rm -f "$LOCK_FILE" "$BACKEND_PID_FILE" "$FRONTEND_PID_FILE"
    
    echo -e "${GREEN}Shutdown complete${NC}"
}

# Set up cleanup on exit
trap cleanup EXIT INT TERM

# Change to project root
cd "$(dirname "$0")/../.."

echo -e "${GREEN}=== RECURSIA STARTUP (FIXED) ===${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed${NC}"
    exit 1
fi

# Activate virtual environment
if [ -d "venv_linux" ]; then
    echo -e "${CYAN}Activating virtual environment...${NC}"
    source venv_linux/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}No virtual environment found${NC}"
    exit 1
fi

# Start backend
echo -e "${CYAN}Starting backend API server...${NC}"
cd "$(pwd)"
python scripts/backend/run_api_server.py > logs/backend_$(date +%Y%m%d_%H%M%S).log 2>&1 &
BACKEND_PID=$!
echo "$BACKEND_PID" > "$BACKEND_PID_FILE"
echo -e "${GREEN}Backend started with PID: $BACKEND_PID${NC}"

# Wait for backend to be ready
echo -e "${YELLOW}Waiting for backend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8080/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready!${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo

# Start frontend
echo -e "${CYAN}Starting frontend...${NC}"
cd frontend
npm run dev > ../logs/frontend_$(date +%Y%m%d_%H%M%S).log 2>&1 &
FRONTEND_PID=$!
echo "$FRONTEND_PID" > "$FRONTEND_PID_FILE"
cd ..
echo -e "${GREEN}Frontend started with PID: $FRONTEND_PID${NC}"

# Wait for frontend
echo -e "${YELLOW}Waiting for frontend to be ready...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend is ready!${NC}"
        break
    fi
    sleep 1
    echo -n "."
done
echo

echo -e "${GREEN}===================================================="
echo -e "🧠 Recursia Quantum OSH Computing Platform"
echo -e "===================================================="
echo -e "${CYAN}Frontend:${NC} http://localhost:5173"
echo -e "${CYAN}API Server:${NC} http://localhost:8080"
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo -e "${GREEN}====================================================${NC}"

# Simple monitoring - just check if processes are alive
while true; do
    # Check backend
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
        echo -e "\n${RED}Backend crashed!${NC}"
        break
    fi
    
    # Check frontend
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo -e "\n${RED}Frontend crashed!${NC}"
        break
    fi
    
    sleep 5
done

echo -e "${RED}One or more services crashed. Check the logs for details.${NC}"
exit 1