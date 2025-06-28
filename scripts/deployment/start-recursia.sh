#!/bin/bash
# Comprehensive Recursia Startup Script - Enterprise Grade
# Handles both venv and non-venv environments, multiple backend options,
# and ensures proper port management with automatic cleanup

echo "=== RECURSIA STARTUP ==="
echo "Starting Recursia API server and frontend dashboard..."
echo

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration - Use environment variables with sensible defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

# Port configuration with environment variable support
BACKEND_PORTS=(${RECURSIA_BACKEND_PORT:-8080} 8081 8082 8083)
FRONTEND_PORTS=(${RECURSIA_FRONTEND_PORT:-5173} 5174 5175 3000)
BACKEND_PORT=""
FRONTEND_PORT=""

# Cross-platform temp directory configuration
if [[ -n "${TMPDIR:-}" ]]; then
    TEMP_DIR="${TMPDIR%/}/recursia"
elif [[ -n "${TMP:-}" ]]; then
    TEMP_DIR="${TMP%/}/recursia"
elif [[ -w "/tmp" ]]; then
    TEMP_DIR="/tmp/recursia"
else
    TEMP_DIR="$PROJECT_ROOT/tmp"
fi

# PID directory configuration
PID_DIR="$TEMP_DIR/pids"
BACKEND_PID_FILE="$PID_DIR/backend.pid"
FRONTEND_PID_FILE="$PID_DIR/frontend.pid"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
            echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
            return 0
        else
            echo -e "${RED}✗ Python 3.8+ required, found $PYTHON_VERSION${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Python 3 not found${NC}"
        return 1
    fi
}

# Function to check Node.js version
check_node_version() {
    if command_exists node; then
        NODE_VERSION=$(node --version | sed 's/v//')
        NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
        
        if [ "$NODE_MAJOR" -ge 16 ]; then
            echo -e "${GREEN}✓ Node.js $NODE_VERSION found${NC}"
            return 0
        else
            echo -e "${RED}✗ Node.js 16+ required, found $NODE_VERSION${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Node.js not found${NC}"
        return 1
    fi
}

# Enhanced function to get PID from port - works across platforms
get_pid_from_port() {
    local port=$1
    local pid=""
    
    # Try multiple methods to find process using port
    if command_exists lsof; then
        pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    elif command_exists netstat; then
        # For systems with netstat
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            pid=$(sudo netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 || true)
        else
            # macOS netstat doesn't show PIDs
            pid=""
        fi
    elif command_exists ss; then
        # For newer Linux systems
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -o 'pid=[0-9]*' | cut -d'=' -f2 || true)
    fi
    
    # Fallback: use fuser if available
    if [[ -z "$pid" ]] && command_exists fuser; then
        pid=$(fuser "$port/tcp" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' || true)
    fi
    
    echo "$pid"
}

# Robust function to kill process on port
kill_port_process() {
    local port=$1
    local service_name=$2
    echo -e "${YELLOW}Checking port $port for $service_name...${NC}"
    
    # Get PIDs using the port
    local pids=$(get_pid_from_port "$port")
    
    if [[ -n "$pids" ]]; then
        echo -e "${YELLOW}Port $port is in use. Attempting to free it...${NC}"
        
        # Kill each PID
        for pid in $pids; do
            echo -e "${YELLOW}Killing process $pid on port $port${NC}"
            kill -TERM "$pid" 2>/dev/null || true
            
            # Give it time to terminate gracefully
            sleep 2
            
            # Force kill if still running
            if kill -0 "$pid" 2>/dev/null; then
                echo -e "${YELLOW}Force killing process $pid${NC}"
                kill -KILL "$pid" 2>/dev/null || true
            fi
        done
        
        # Additional cleanup for stubborn processes
        if command_exists fuser; then
            fuser -k "$port/tcp" 2>/dev/null || true
        fi
        
        # Wait a moment for port to be released
        sleep 1
        
        # Verify port is free
        local remaining_pids=$(get_pid_from_port "$port")
        if [[ -z "$remaining_pids" ]]; then
            echo -e "${GREEN}✓ Port $port is now free${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to free port $port${NC}"
            return 1
        fi
    else
        echo -e "${GREEN}✓ Port $port is available${NC}"
        return 0
    fi
}

# Function to find available port from list
find_available_port() {
    local port_array=("$@")
    local service_name=$1
    shift  # Remove service name from array
    local available_port=""
    
    for port in "${port_array[@]}"; do
        if kill_port_process "$port" "$service_name" >/dev/null 2>&1; then
            available_port=$port
            break
        fi
    done
    
    echo "$available_port"
}

# Function to install system dependencies on different platforms
install_system_deps() {
    echo -e "${YELLOW}Installing system dependencies...${NC}"
    
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command_exists apt; then
            sudo apt update
            sudo apt install -y python3 python3-pip python3-venv nodejs npm
        elif command_exists yum; then
            sudo yum install -y python3 python3-pip nodejs npm
        elif command_exists dnf; then
            sudo dnf install -y python3 python3-pip nodejs npm
        elif command_exists pacman; then
            sudo pacman -S python python-pip nodejs npm
        else
            echo -e "${RED}Unsupported Linux distribution. Please install Python 3.8+ and Node.js 16+ manually.${NC}"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command_exists brew; then
            brew install python3 node
        else
            echo -e "${RED}Homebrew not found. Please install Python 3.8+ and Node.js 16+ manually.${NC}"
            echo -e "${CYAN}Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
            exit 1
        fi
    else
        echo -e "${RED}Unsupported operating system: $OSTYPE${NC}"
        exit 1
    fi
}

# Function to cleanup old log files
cleanup_old_logs() {
    echo -e "${BLUE}Cleaning up old log files...${NC}"
    
    # Create logs directory if it doesn't exist
    mkdir -p logs 2>/dev/null
    
    # Maximum number of log files to keep per type
    local MAX_LOGS=20
    
    # Function to cleanup specific log pattern
    cleanup_log_pattern() {
        local pattern=$1
        local description=$2
        
        # Count matching files
        local count=$(ls -1 logs/${pattern}*.log 2>/dev/null | wc -l)
        
        if [ "$count" -gt "$MAX_LOGS" ]; then
            local to_remove=$((count - MAX_LOGS))
            echo -e "${YELLOW}Found $count $description logs, removing oldest $to_remove...${NC}"
            
            # Remove oldest files (sorted by modification time)
            ls -1t logs/${pattern}*.log 2>/dev/null | tail -n +$((MAX_LOGS + 1)) | xargs rm -f
            
            echo -e "${GREEN}✓ Cleaned up $to_remove old $description logs${NC}"
        else
            echo -e "${GREEN}✓ $description logs within limit ($count/$MAX_LOGS)${NC}"
        fi
    }
    
    # Cleanup different log types
    cleanup_log_pattern "backend_" "backend"
    cleanup_log_pattern "frontend_" "frontend"
    cleanup_log_pattern "recursia_comprehensive_" "comprehensive"
    
    # Cleanup main recursia.log if it's too large (> 50MB)
    if [ -f "logs/recursia.log" ]; then
        local log_size=$(stat -f%z "logs/recursia.log" 2>/dev/null || stat -c%s "logs/recursia.log" 2>/dev/null || echo "0")
        local max_size=$((50 * 1024 * 1024))  # 50MB in bytes
        
        if [ "$log_size" -gt "$max_size" ]; then
            echo -e "${YELLOW}Main recursia.log is large ($(($log_size / 1024 / 1024))MB), archiving...${NC}"
            
            # Archive the old log with timestamp
            local archive_name="logs/recursia_archived_$(date +%Y%m%d_%H%M%S).log"
            mv logs/recursia.log "$archive_name"
            
            # Compress the archived log
            if command_exists gzip; then
                gzip "$archive_name"
                echo -e "${GREEN}✓ Archived main log to ${archive_name}.gz${NC}"
            else
                echo -e "${GREEN}✓ Archived main log to $archive_name${NC}"
            fi
            
            # Create new empty log
            touch logs/recursia.log
        else
            echo -e "${GREEN}✓ Main recursia.log size OK ($(($log_size / 1024))KB)${NC}"
        fi
    fi
    
    # Remove empty log files
    find logs -name "*.log" -size 0 -delete 2>/dev/null || true
    
    # Remove very old archived logs (older than 30 days)
    find logs -name "*archived*.log*" -mtime +30 -delete 2>/dev/null || true
    find logs -name "*validation*.log" -mtime +7 -delete 2>/dev/null || true
    
    echo -e "${GREEN}✓ Log cleanup complete${NC}"
}

# Cleanup old logs first
cleanup_old_logs

# Check system requirements
echo -e "${BLUE}Checking system requirements...${NC}"

PYTHON_OK=false
NODE_OK=false

if check_python_version; then
    PYTHON_OK=true
fi

if check_node_version; then
    NODE_OK=true
fi

# Install missing dependencies
if [ "$PYTHON_OK" = false ] || [ "$NODE_OK" = false ]; then
    echo -e "${YELLOW}Some dependencies are missing. Attempting to install...${NC}"
    install_system_deps
    
    # Re-check after installation
    if ! check_python_version || ! check_node_version; then
        echo -e "${RED}Failed to install required dependencies. Please install manually:${NC}"
        echo -e "${CYAN}  - Python 3.8+: https://www.python.org/downloads/${NC}"
        echo -e "${CYAN}  - Node.js 16+: https://nodejs.org/en/download/${NC}"
        exit 1
    fi
fi

# Determine Python environment strategy
echo -e "${BLUE}Setting up Python environment...${NC}"

PYTHON_CMD=""
PIP_CMD=""
USE_VENV=false

# Check for virtual environment - Use single cross-platform venv name
VENV_DIR="${RECURSIA_VENV_DIR:-venv}"

# First, check if we're already in a virtual environment
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    echo -e "${GREEN}✓ Already in virtual environment: $VIRTUAL_ENV${NC}"
    PYTHON_CMD="python"
    PIP_CMD="pip"
    USE_VENV=false
elif [ -d "$VENV_DIR" ]; then
    # Virtual environment exists, use it
    echo -e "${YELLOW}Activating virtual environment: $VENV_DIR${NC}"
    source "$VENV_DIR/bin/activate"
    PYTHON_CMD="python"
    PIP_CMD="pip"
    USE_VENV=true
else
    # No virtual environment, check if we should create one or use system Python
    echo -e "${YELLOW}No virtual environment found.${NC}"
    
    # Check if pip packages are installed system-wide
    if python3 -c "import fastapi" 2>/dev/null; then
        echo -e "${GREEN}✓ Using system Python with installed packages${NC}"
        PYTHON_CMD="python3"
        PIP_CMD="pip3"
        USE_VENV=false
    else
        # Create virtual environment
        echo -e "${YELLOW}Creating virtual environment...${NC}"
        python3 -m venv "$VENV_DIR"
        source "$VENV_DIR/bin/activate"
        PYTHON_CMD="python"
        PIP_CMD="pip"
        USE_VENV=true
    fi
fi

# Install Python dependencies
echo -e "${BLUE}Checking Python dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    # Check if key packages are installed
    PACKAGES_OK=true
    for package in fastapi uvicorn numpy; do
        if ! $PYTHON_CMD -c "import $package" 2>/dev/null; then
            PACKAGES_OK=false
            break
        fi
    done
    
    if [ "$PACKAGES_OK" = false ]; then
        echo -e "${YELLOW}Installing Python dependencies...${NC}"
        $PIP_CMD install --upgrade pip
        $PIP_CMD install -r requirements.txt
        $PIP_CMD install -e .
    else
        echo -e "${GREEN}✓ Python dependencies already installed${NC}"
    fi
else
    echo -e "${RED}requirements.txt not found. Make sure you're in the Recursia root directory.${NC}"
    exit 1
fi

# Set up frontend
echo -e "${BLUE}Setting up frontend...${NC}"
cd frontend

if [ ! -f "package.json" ]; then
    echo -e "${RED}Frontend package.json not found. Make sure you're in the correct directory.${NC}"
    exit 1
fi

if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install
else
    echo -e "${GREEN}✓ Frontend dependencies already installed${NC}"
fi

# Go back to root directory
cd ..

# Check and train ML models if needed
echo -e "${BLUE}Checking ML decoder models...${NC}"
if [ -f "scripts/deployment/check_and_train_ml_models.sh" ]; then
    source scripts/deployment/check_and_train_ml_models.sh
    missing=$(check_ml_models)
    
    if [ "$missing" -gt 0 ]; then
        echo -e "${YELLOW}Missing $missing ML decoder models${NC}"
        train_ml_models
    else
        echo -e "${GREEN}✓ All ML decoder models are present${NC}"
    fi
else
    echo -e "${YELLOW}⚠ ML model check script not found, skipping ML model verification${NC}"
fi

# Find available ports
echo -e "${BLUE}Finding available ports...${NC}"

# Clean up backend ports
for port in "${BACKEND_PORTS[@]}"; do
    if kill_port_process "$port" "backend"; then
        BACKEND_PORT=$port
        break
    fi
done

if [[ -z "$BACKEND_PORT" ]]; then
    echo -e "${RED}No available backend ports found. All ports ${BACKEND_PORTS[@]} are in use.${NC}"
    exit 1
fi

# Clean up frontend ports
for port in "${FRONTEND_PORTS[@]}"; do
    if kill_port_process "$port" "frontend"; then
        FRONTEND_PORT=$port
        break
    fi
done

if [[ -z "$FRONTEND_PORT" ]]; then
    echo -e "${RED}No available frontend ports found. All ports ${FRONTEND_PORTS[@]} are in use.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Selected Backend Port: $BACKEND_PORT${NC}"
echo -e "${GREEN}✓ Selected Frontend Port: $FRONTEND_PORT${NC}"

# Kill any remaining processes that might interfere
echo -e "${YELLOW}Cleaning up any stale processes...${NC}"
pkill -f "api_server_enhanced" 2>/dev/null || true
pkill -f "uvicorn.*$BACKEND_PORT" 2>/dev/null || true
pkill -f "vite.*$FRONTEND_PORT" 2>/dev/null || true
sleep 2

# Function to start backend server with multiple fallback options
start_backend() {
    echo -e "${CYAN}Starting API server on port $BACKEND_PORT...${NC}"
    
    # Export port for the API server to use
    export BACKEND_PORT
    
    # Create log file for this session
    local BACKEND_LOG="logs/backend_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs 2>/dev/null
    
    # Start unified API server
    if [ -f "src/api/unified_api_server.py" ]; then
        echo -e "${CYAN}Starting Unified API server (centralized runtime)...${NC}"
        $PYTHON_CMD -m src.api.unified_api_server > "$BACKEND_LOG" 2>&1 &
        BACKEND_PID=$!
    elif [ -f "scripts/backend/run_api_server.py" ]; then
        echo -e "${CYAN}Starting API server via run script...${NC}"
        $PYTHON_CMD scripts/backend/run_api_server.py > "$BACKEND_LOG" 2>&1 &
        BACKEND_PID=$!
    elif [ -f "src/api_server.py" ]; then
        echo -e "${CYAN}Starting standard API server...${NC}"
        $PYTHON_CMD src/api_server.py > "$BACKEND_LOG" 2>&1 &
        BACKEND_PID=$!
    else
        echo -e "${RED}No API server script found!${NC}"
        exit 1
    fi
    
    echo "API Server PID: $BACKEND_PID"
    echo "API Server Log: $BACKEND_LOG"
    
    # Wait for backend to be ready - check multiple endpoints
    echo -e "${YELLOW}Waiting for API server to start...${NC}"
    RETRY=0
    MAX_RETRY=40  # Increased timeout
    API_READY=false
    
    while [ $RETRY -lt $MAX_RETRY ]; do
        # Check if process is still running
        if ! kill -0 $BACKEND_PID 2>/dev/null; then
            echo -e "\n${RED}API server process died. Last 20 lines of log:${NC}"
            tail -20 "$BACKEND_LOG"
            exit 1
        fi
        
        # Try multiple endpoints to determine if server is ready
        if curl -s -f "http://localhost:$BACKEND_PORT/health" > /dev/null 2>&1 || \
           curl -s -f "http://localhost:$BACKEND_PORT/api/health" > /dev/null 2>&1 || \
           curl -s -f "http://localhost:$BACKEND_PORT/docs" > /dev/null 2>&1 || \
           curl -s -f "http://localhost:$BACKEND_PORT/" > /dev/null 2>&1; then
            API_READY=true
            echo -e "\n${GREEN}✓ API server is ready!${NC}"
            break
        fi
        
        # Check if the server is listening on the port
        if get_pid_from_port "$BACKEND_PORT" | grep -q "$BACKEND_PID"; then
            # Server is listening, might just be initializing
            if [ $RETRY -eq 10 ]; then
                echo -e "\n${YELLOW}Server is listening on port $BACKEND_PORT but not responding to health checks yet...${NC}"
            fi
        fi
        
        sleep 1
        RETRY=$((RETRY + 1))
        echo -n "."
    done
    echo
    
    # Final check - if server is listening on port, consider it successful
    if [ "$API_READY" != "true" ]; then
        if get_pid_from_port "$BACKEND_PORT" | grep -q "$BACKEND_PID"; then
            echo -e "${YELLOW}⚠ API server is running on port $BACKEND_PORT but health endpoints are not responding.${NC}"
            echo -e "${YELLOW}This may be normal if the server uses different endpoints.${NC}"
            echo -e "${GREEN}Continuing with startup...${NC}"
        else
            echo -e "${RED}API server failed to start. Last 20 lines of log:${NC}"
            tail -20 "$BACKEND_LOG"
            exit 1
        fi
    fi
}

# Function to start frontend server
start_frontend() {
    echo -e "${CYAN}Starting frontend server on port $FRONTEND_PORT...${NC}"
    cd frontend
    
    # Create log file for this session
    local FRONTEND_LOG="../logs/frontend_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p ../logs 2>/dev/null
    
    # Export port for vite to use if not default
    if [ "$FRONTEND_PORT" != "5173" ]; then
        export VITE_PORT=$FRONTEND_PORT
        npm run dev -- --port $FRONTEND_PORT > "$FRONTEND_LOG" 2>&1 &
    else
        npm run dev > "$FRONTEND_LOG" 2>&1 &
    fi
    
    FRONTEND_PID=$!
    echo "Frontend PID: $FRONTEND_PID"
    echo "Frontend Log: $FRONTEND_LOG"
    cd ..
    
    # Wait for frontend to be ready
    echo -e "${YELLOW}Waiting for frontend server to start...${NC}"
    local RETRY=0
    local MAX_RETRY=30
    
    while [ $RETRY -lt $MAX_RETRY ]; do
        # Check if process is still running
        if ! kill -0 $FRONTEND_PID 2>/dev/null; then
            echo -e "\n${RED}Frontend server process died. Last 20 lines of log:${NC}"
            tail -20 "$FRONTEND_LOG"
            exit 1
        fi
        
        # Check if server is ready
        if curl -s "http://localhost:$FRONTEND_PORT" > /dev/null 2>&1; then
            echo -e "\n${GREEN}✓ Frontend server is ready!${NC}"
            break
        fi
        
        sleep 1
        RETRY=$((RETRY + 1))
        echo -n "."
    done
    echo
    
    if [ $RETRY -eq $MAX_RETRY ]; then
        echo -e "${RED}Frontend server failed to start. Last 20 lines of log:${NC}"
        tail -20 "$FRONTEND_LOG"
        exit 1
    fi
}

# Function to open browser - Cross-platform with graceful fallback
open_browser() {
    sleep 3
    local URL="http://localhost:$FRONTEND_PORT"
    
    # Skip browser opening if RECURSIA_NO_BROWSER is set or running in headless environment
    if [[ -n "${RECURSIA_NO_BROWSER:-}" ]] || [[ -n "${CI:-}" ]] || [[ -n "${HEADLESS:-}" ]]; then
        echo -e "${CYAN}Browser opening skipped (headless mode)${NC}"
        echo -e "${GREEN}Recursia Studio available at: $URL${NC}"
        return 0
    fi
    
    echo -e "${GREEN}Opening Recursia Studio in browser...${NC}"
    echo -e "${CYAN}URL: $URL${NC}"
    
    # Try cross-platform browser opening with fallback
    local opened=false
    
    # Try Python webbrowser module (most reliable cross-platform)
    if command_exists python3 && ! $opened; then
        if python3 -c "import webbrowser; webbrowser.open('$URL')" 2>/dev/null; then
            opened=true
        fi
    fi
    
    # Platform-specific fallbacks
    if ! $opened; then
        if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "linux"* ]]; then
            for cmd in xdg-open gnome-open kde-open; do
                if command_exists "$cmd"; then
                    "$cmd" "$URL" 2>/dev/null && opened=true && break
                fi
            done
        elif [[ "$OSTYPE" == "darwin"* ]]; then
            if command_exists open; then
                open "$URL" 2>/dev/null && opened=true
            fi
        elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
            if command_exists start; then
                start "$URL" 2>/dev/null && opened=true
            elif command_exists cmd; then
                cmd /c start "$URL" 2>/dev/null && opened=true
            fi
        fi
    fi
    
    if ! $opened; then
        echo -e "${YELLOW}Could not open browser automatically${NC}"
        echo -e "${CYAN}Please manually open: $URL${NC}"
    fi
}

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Shutting down servers...${NC}"
    
    # Kill backend
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
        
        # Also kill by port in case PID tracking failed
        local port_pid=$(get_pid_from_port "$BACKEND_PORT")
        if [[ -n "$port_pid" ]]; then
            kill $port_pid 2>/dev/null || true
        fi
    fi
    
    # Kill frontend
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
        
        # Also kill by port in case PID tracking failed
        local port_pid=$(get_pid_from_port "$FRONTEND_PORT")
        if [[ -n "$port_pid" ]]; then
            kill $port_pid 2>/dev/null || true
        fi
    fi
    
    # Kill any remaining processes
    pkill -f "vite.*$FRONTEND_PORT" 2>/dev/null || true
    pkill -f "api_server.*$BACKEND_PORT" 2>/dev/null || true
    pkill -f "uvicorn.*$BACKEND_PORT" 2>/dev/null || true
    
    # Deactivate virtual environment if we activated it
    if [ "$USE_VENV" = true ]; then
        deactivate 2>/dev/null || true
    fi
    
    echo -e "${GREEN}Recursia shutdown complete.${NC}"
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM EXIT

# Create necessary directories
mkdir -p "$PID_DIR" 2>/dev/null || {
    echo -e "${YELLOW}Warning: Could not create PID directory $PID_DIR${NC}"
    # Fallback to project directory
    PID_DIR="$PROJECT_ROOT"
    BACKEND_PID_FILE="$PID_DIR/backend.pid"
    FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
}

# Start servers
echo -e "${GREEN}🌟 All dependencies ready! Starting Recursia...${NC}"
echo ""

# Start backend first
start_backend

# Small delay to ensure backend is stable
sleep 2

# Start frontend
start_frontend

# Open browser
if [ "${NO_BROWSER:-0}" != "1" ]; then
    open_browser &
fi

echo -e "${GREEN}=================================================="
echo -e "🧠 Recursia Quantum OSH Computing Platform"
echo -e "=================================================="
echo -e "${CYAN}Frontend: ${NC}http://localhost:$FRONTEND_PORT"
echo -e "${CYAN}API Server: ${NC}http://localhost:$BACKEND_PORT"
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo -e "${GREEN}=================================================="

# Monitor servers and restart if they crash
# Keep track of restart attempts to prevent loops
BACKEND_RESTART_COUNT=0
FRONTEND_RESTART_COUNT=0
MAX_RESTART_ATTEMPTS=3

while true; do
    # Check backend
    if [ ! -z "$BACKEND_PID" ] && ! kill -0 $BACKEND_PID 2>/dev/null; then
        # Backend process ended - check if it's a normal shutdown or crash
        echo -e "${YELLOW}Backend process ended (PID: $BACKEND_PID)${NC}"
        
        # Check if a new process is already using the port (might be a reload)
        current_port_pid=$(get_pid_from_port "$BACKEND_PORT")
        if [[ -n "$current_port_pid" ]]; then
            echo -e "${YELLOW}Port $BACKEND_PORT is already in use by PID $current_port_pid${NC}"
            # Update our PID tracking
            BACKEND_PID=$current_port_pid
            BACKEND_RESTART_COUNT=0
            echo -e "${GREEN}Updated backend PID to $BACKEND_PID${NC}"
        else
            # Port is free, so backend really crashed
            BACKEND_RESTART_COUNT=$((BACKEND_RESTART_COUNT + 1))
            
            if [ $BACKEND_RESTART_COUNT -gt $MAX_RESTART_ATTEMPTS ]; then
                echo -e "${RED}Backend crashed too many times ($BACKEND_RESTART_COUNT). Giving up.${NC}"
                echo -e "${RED}Check the logs for errors.${NC}"
                exit 1
            fi
            
            echo -e "${RED}Backend crashed! Restart attempt $BACKEND_RESTART_COUNT/$MAX_RESTART_ATTEMPTS...${NC}"
            
            # Wait a bit before restarting to avoid rapid loops
            sleep 2
            
            # Restart backend
            start_backend
        fi
    fi
    
    # Check frontend
    if [ ! -z "$FRONTEND_PID" ] && ! kill -0 $FRONTEND_PID 2>/dev/null; then
        # Frontend process ended
        echo -e "${YELLOW}Frontend process ended (PID: $FRONTEND_PID)${NC}"
        
        # Check if a new process is already using the port
        current_port_pid=$(get_pid_from_port "$FRONTEND_PORT")
        if [[ -n "$current_port_pid" ]]; then
            echo -e "${YELLOW}Port $FRONTEND_PORT is already in use by PID $current_port_pid${NC}"
            # Update our PID tracking
            FRONTEND_PID=$current_port_pid
            FRONTEND_RESTART_COUNT=0
            echo -e "${GREEN}Updated frontend PID to $FRONTEND_PID${NC}"
        else
            # Port is free, so frontend really crashed
            FRONTEND_RESTART_COUNT=$((FRONTEND_RESTART_COUNT + 1))
            
            if [ $FRONTEND_RESTART_COUNT -gt $MAX_RESTART_ATTEMPTS ]; then
                echo -e "${RED}Frontend crashed too many times ($FRONTEND_RESTART_COUNT). Giving up.${NC}"
                echo -e "${RED}Check the logs for errors.${NC}"
                exit 1
            fi
            
            echo -e "${RED}Frontend crashed! Restart attempt $FRONTEND_RESTART_COUNT/$MAX_RESTART_ATTEMPTS...${NC}"
            
            # Wait a bit before restarting
            sleep 2
            
            # Restart frontend
            start_frontend
        fi
    fi
    
    # Check every 5 seconds
    sleep 5
done