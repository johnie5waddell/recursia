#!/bin/bash
set -e

# Recursia Complete Startup Script with Testing
# This script ensures the entire system is working correctly

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        🚀 Recursia Quantum Programming Environment 🚀        ║${NC}"
echo -e "${BLUE}║                   Complete Startup & Test                     ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if a process is running
check_process() {
    if lsof -i:$1 >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local port=$1
    local name=$2
    local max_attempts=30
    
    echo -e "${YELLOW}⏳ Waiting for $name on port $port...${NC}"
    for i in $(seq 1 $max_attempts); do
        if check_process $port; then
            echo -e "${GREEN}✅ $name is ready!${NC}"
            return 0
        fi
        sleep 1
    done
    echo -e "${RED}❌ $name failed to start on port $port${NC}"
    return 1
}

# Step 1: Clean up existing processes
echo -e "\n${YELLOW}🧹 Step 1: Cleaning up existing processes...${NC}"
./stop-all.sh 2>/dev/null || true
sleep 2

# Step 2: Check environment
echo -e "\n${YELLOW}🔍 Step 2: Checking environment...${NC}"
cd "$(dirname "$0")"

# Check for virtual environment
if [ -d "venv_linux" ]; then
    source venv_linux/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    echo -e "${RED}❌ No virtual environment found!${NC}"
    echo "Please create one with: python3 -m venv venv_linux"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
echo -e "   Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Check key dependencies
echo -e "   Checking dependencies..."
python -c "import fastapi" && echo -e "   ${GREEN}✓${NC} FastAPI" || echo -e "   ${RED}✗${NC} FastAPI"
python -c "import numpy" && echo -e "   ${GREEN}✓${NC} NumPy" || echo -e "   ${RED}✗${NC} NumPy"
python -c "import websocket" && echo -e "   ${GREEN}✓${NC} WebSocket" || echo -e "   ${RED}✗${NC} WebSocket"

# Step 3: Start Backend
echo -e "\n${YELLOW}🔧 Step 3: Starting Backend API Server...${NC}"
mkdir -p logs
python -m src.api.unified_api_server > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo -e "   Backend PID: ${GREEN}$BACKEND_PID${NC}"

# Wait for backend
if ! wait_for_service 8080 "Backend API"; then
    echo -e "${RED}Failed to start backend. Check logs/backend.log${NC}"
    tail -20 logs/backend.log
    exit 1
fi

# Step 4: Test Backend Health
echo -e "\n${YELLOW}🏥 Step 4: Testing Backend Health...${NC}"
HEALTH_RESPONSE=$(curl -s http://localhost:8080/api/health || echo '{}')
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo -e "${GREEN}✅ Backend is healthy${NC}"
    echo "$HEALTH_RESPONSE" | python -m json.tool | head -10
else
    echo -e "${RED}❌ Backend health check failed${NC}"
    exit 1
fi

# Step 5: Quick API Test
echo -e "\n${YELLOW}🧪 Step 5: Running Quick API Test...${NC}"
python test_api_quick.py
API_TEST_RESULT=$?

if [ $API_TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✅ API test passed${NC}"
else
    echo -e "${RED}❌ API test failed${NC}"
    echo "Check the output above for details"
fi

# Step 5.5: Check ML Models
echo -e "\n${YELLOW}🤖 Step 5.5: Checking ML Models...${NC}"
if [ -f "scripts/deployment/check_and_train_ml_models.sh" ]; then
    source scripts/deployment/check_and_train_ml_models.sh
    missing=$(check_ml_models)
    
    if [ "$missing" -gt 0 ]; then
        echo -e "${YELLOW}   Missing $missing ML decoder models${NC}"
        train_ml_models
    else
        echo -e "${GREEN}✅ All ML decoder models are present${NC}"
    fi
else
    echo -e "${YELLOW}⚠ ML model check script not found, skipping${NC}"
fi

# Step 6: Start Frontend
echo -e "\n${YELLOW}🎨 Step 6: Starting Frontend...${NC}"
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}   Installing frontend dependencies...${NC}"
    npm install
fi

npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..
echo -e "   Frontend PID: ${GREEN}$FRONTEND_PID${NC}"

# Wait for frontend
if ! wait_for_service 5173 "Frontend"; then
    echo -e "${RED}Failed to start frontend. Check logs/frontend.log${NC}"
    tail -20 logs/frontend.log
    exit 1
fi

# Step 7: Final System Check
echo -e "\n${YELLOW}🔍 Step 7: Final System Check...${NC}"
sleep 3  # Give everything time to stabilize

# Check all services
echo -e "   Service Status:"
if check_process 8080; then
    echo -e "   ${GREEN}✓${NC} Backend API: http://localhost:8080"
else
    echo -e "   ${RED}✗${NC} Backend API"
fi

if check_process 5173; then
    echo -e "   ${GREEN}✓${NC} Frontend: http://localhost:5173"
else
    echo -e "   ${RED}✗${NC} Frontend"
fi

# Step 8: Display Usage Instructions
echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    🎉 Recursia is Ready! 🎉                  ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${GREEN}🌐 Access Points:${NC}"
echo -e "   Frontend UI: ${BLUE}http://localhost:5173${NC}"
echo -e "   Backend API: ${BLUE}http://localhost:8080${NC}"
echo -e "   API Documentation: ${BLUE}http://localhost:8080/docs${NC}"
echo -e "   Test Page: ${BLUE}http://localhost:5173/test-iterations.html${NC}"

echo -e "\n${GREEN}📋 How to Test Execution:${NC}"
echo -e "   1. Open ${BLUE}http://localhost:5173${NC} in your browser"
echo -e "   2. Switch to ${YELLOW}'Program Execution'${NC} mode (top tabs)"
echo -e "   3. Open ${YELLOW}'Quantum Programs Library'${NC} tab"
echo -e "   4. Find ${YELLOW}'Hello World Simple'${NC} program"
echo -e "   5. Click the ${GREEN}▶️${NC} play button to execute"
echo -e "   6. Choose number of iterations (1-1,000,000,000)"
echo -e "   7. Click ${GREEN}'Execute'${NC} and watch the execution log"

echo -e "\n${GREEN}🧪 Additional Testing:${NC}"
echo -e "   Run comprehensive tests: ${YELLOW}python test_full_iteration_flow.py${NC}"
echo -e "   Test frontend execution: ${YELLOW}python test_frontend_execution.py${NC}"
echo -e "   Visual test page: ${BLUE}http://localhost:5173/test-iterations.html${NC}"

echo -e "\n${GREEN}📊 Monitoring:${NC}"
echo -e "   Backend logs: ${YELLOW}tail -f logs/backend.log${NC}"
echo -e "   Frontend logs: ${YELLOW}tail -f logs/frontend.log${NC}"
echo -e "   Both logs: ${YELLOW}tail -f logs/*.log${NC}"

echo -e "\n${GREEN}🛑 To Stop Everything:${NC}"
echo -e "   Run: ${YELLOW}./stop-all.sh${NC}"

echo -e "\n${YELLOW}💡 Troubleshooting:${NC}"
echo -e "   - If execution fails, check browser console (F12)"
echo -e "   - Ensure ports 8080 and 5173 are not blocked"
echo -e "   - Check WebSocket connection in Network tab"
echo -e "   - Verify API health: ${BLUE}http://localhost:8080/api/health${NC}"

# Write PIDs to file for easy cleanup
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

# Optional: Show initial logs
echo -e "\n${YELLOW}📺 Initial Backend Logs:${NC}"
echo "────────────────────────────────────────"
tail -20 logs/backend.log | grep -v "DEBUG" || true
echo "────────────────────────────────────────"

echo -e "\n${GREEN}✨ Startup complete! The system is ready for use.${NC}"
echo -e "\n${YELLOW}Press Ctrl+C to stop watching logs...${NC}"

# Keep running and show logs
trap 'echo -e "\n${YELLOW}Stopping services...${NC}"; ./stop-all.sh; exit 0' INT
tail -f logs/backend.log | grep -v "DEBUG"