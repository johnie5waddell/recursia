#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Recursia with Full Iteration Support${NC}"
echo "=================================================="

# Function to check if a process is running
check_process() {
    if lsof -i:$1 >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Kill any existing processes
echo -e "\n${YELLOW}🧹 Cleaning up existing processes...${NC}"
./stop-all.sh 2>/dev/null || true
sleep 2

# Start the backend
echo -e "\n${GREEN}🔧 Starting backend API server...${NC}"
cd "$(dirname "$0")"
source venv_linux/bin/activate 2>/dev/null || source venv/bin/activate 2>/dev/null || {
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Please run: python3 -m venv venv_linux && source venv_linux/bin/activate && pip install -r requirements.txt"
    exit 1
}

# Start backend in background
python -m src.api.unified_api_server > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend to start...${NC}"
for i in {1..30}; do
    if check_process 8080; then
        echo -e "${GREEN}✅ Backend is running on port 8080${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend failed to start!${NC}"
        echo "Check logs/backend.log for errors"
        exit 1
    fi
    sleep 1
done

# Test backend health
echo -e "\n${YELLOW}🏥 Checking backend health...${NC}"
if curl -s http://localhost:8080/api/health | grep -q "healthy"; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
else
    echo -e "${RED}❌ Backend health check failed!${NC}"
    exit 1
fi

# Quick API test
echo -e "\n${YELLOW}🧪 Running quick API test...${NC}"
python test_api_quick.py

# Start frontend
echo -e "\n${GREEN}🎨 Starting frontend...${NC}"
cd frontend
npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

# Wait for frontend
echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
for i in {1..30}; do
    if check_process 5173; then
        echo -e "${GREEN}✅ Frontend is running on port 5173${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Frontend failed to start!${NC}"
        echo "Check logs/frontend.log for errors"
        exit 1
    fi
    sleep 1
done

# Summary
echo -e "\n${GREEN}🎉 Recursia is ready!${NC}"
echo "=================================================="
echo -e "${GREEN}Frontend:${NC} http://localhost:5173"
echo -e "${GREEN}Backend API:${NC} http://localhost:8080"
echo -e "${GREEN}API Docs:${NC} http://localhost:8080/docs"
echo ""
echo -e "${YELLOW}📋 Iteration Feature Test Instructions:${NC}"
echo "1. Open http://localhost:5173 in your browser"
echo "2. Switch to 'Program Execution' mode"
echo "3. Click on 'Quantum Programs Library' tab"
echo "4. Click the ▶️ button on any program (e.g., 'Hello World')"
echo "5. Select number of iterations (try 1, 10, 100)"
echo "6. Click 'Execute' and watch the OSH Execution Log"
echo ""
echo -e "${YELLOW}🧪 To run comprehensive tests:${NC}"
echo "   python test_full_iteration_flow.py"
echo ""
echo -e "${YELLOW}📊 Logs:${NC}"
echo "   Backend: tail -f logs/backend.log"
echo "   Frontend: tail -f logs/frontend.log"
echo ""
echo -e "${YELLOW}🛑 To stop:${NC} ./stop-all.sh"
echo ""

# Keep script running and show logs
echo -e "${GREEN}📺 Showing backend logs (Ctrl+C to exit)...${NC}"
echo "=================================================="
tail -f logs/backend.log