#!/bin/bash
# Recursia Platform Stop Script - Enterprise Grade
# Ensures clean shutdown of all Recursia services

set -euo pipefail

# ===========================
# CONFIGURATION
# ===========================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Cross-platform temp directory configuration
if [[ -n "${TMPDIR:-}" ]]; then
    readonly PID_DIR="${TMPDIR%/}/recursia/pids"
elif [[ -n "${TMP:-}" ]]; then
    readonly PID_DIR="${TMP%/}/recursia/pids"
elif [[ -w "/tmp" ]]; then
    readonly PID_DIR="/tmp/recursia/pids"
else
    readonly PID_DIR="$(pwd)/tmp/pids"
fi
readonly BACKEND_PID_FILE="$PID_DIR/backend.pid"
readonly FRONTEND_PID_FILE="$PID_DIR/frontend.pid"
readonly BACKEND_PORT=8080
readonly FRONTEND_PORT=5173

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m'

# ===========================
# LOGGING
# ===========================
log() {
    local level=$1
    shift
    local message="$@"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        "INFO")
            echo -e "${BLUE}[$timestamp]${NC} ${CYAN}INFO:${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${BLUE}[$timestamp]${NC} ${GREEN}SUCCESS:${NC} $message"
            ;;
        "WARNING")
            echo -e "${BLUE}[$timestamp]${NC} ${YELLOW}WARNING:${NC} $message"
            ;;
        "ERROR")
            echo -e "${BLUE}[$timestamp]${NC} ${RED}ERROR:${NC} $message"
            ;;
    esac
}

# ===========================
# UTILITY FUNCTIONS
# ===========================

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get process ID from port
get_pid_from_port() {
    local port=$1
    local pid=""
    
    if command_exists lsof; then
        pid=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    elif command_exists netstat; then
        pid=$(netstat -tlnp 2>/dev/null | grep ":$port " | awk '{print $7}' | cut -d'/' -f1 || true)
    elif command_exists ss; then
        pid=$(ss -tlnp 2>/dev/null | grep ":$port " | awk '{print $6}' | grep -o 'pid=[0-9]*' | cut -d'=' -f2 || true)
    fi
    
    echo "$pid"
}

# Kill process tree
kill_process_tree() {
    local pid=$1
    local signal="${2:-TERM}"
    
    if [[ -z "$pid" ]]; then
        return 0
    fi
    
    # Get all child processes
    local children=$(pgrep -P "$pid" 2>/dev/null || true)
    
    # Kill children first
    for child in $children; do
        kill_process_tree "$child" "$signal"
    done
    
    # Kill the parent process
    if kill -0 "$pid" 2>/dev/null; then
        kill "-$signal" "$pid" 2>/dev/null || true
        
        # Wait for process to terminate
        local timeout=10
        while [[ $timeout -gt 0 ]] && kill -0 "$pid" 2>/dev/null; do
            sleep 0.5
            ((timeout--))
        done
        
        # Force kill if still running
        if kill -0 "$pid" 2>/dev/null; then
            log "WARNING" "Process $pid didn't terminate gracefully, forcing kill"
            kill -9 "$pid" 2>/dev/null || true
        fi
    fi
}

# ===========================
# SHUTDOWN FUNCTIONS
# ===========================

stop_backend() {
    log "INFO" "Stopping backend API server..."
    
    local stopped=false
    
    # Try PID file first
    if [[ -f "$BACKEND_PID_FILE" ]]; then
        local pid=$(cat "$BACKEND_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Stopping backend process (PID: $pid)"
            kill_process_tree "$pid"
            stopped=true
        fi
        rm -f "$BACKEND_PID_FILE"
    fi
    
    # Check port
    local port_pids=$(get_pid_from_port "$BACKEND_PORT")
    if [[ -n "$port_pids" ]]; then
        for pid in $port_pids; do
            log "INFO" "Stopping process $pid on backend port $BACKEND_PORT"
            kill_process_tree "$pid"
            stopped=true
        done
    fi
    
    # Pattern-based cleanup
    local patterns=(
        "api_server_enhanced"
        "uvicorn.*8080"
        "python.*api_server"
    )
    
    for pattern in "${patterns[@]}"; do
        local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            log "INFO" "Stopping processes matching: $pattern"
            for pid in $pids; do
                kill_process_tree "$pid"
                stopped=true
            done
        fi
    done
    
    if [[ "$stopped" == "true" ]]; then
        log "SUCCESS" "Backend server stopped"
    else
        log "INFO" "No backend server processes found"
    fi
}

stop_frontend() {
    log "INFO" "Stopping frontend development server..."
    
    local stopped=false
    
    # Try PID file first
    if [[ -f "$FRONTEND_PID_FILE" ]]; then
        local pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            log "INFO" "Stopping frontend process (PID: $pid)"
            kill_process_tree "$pid"
            stopped=true
        fi
        rm -f "$FRONTEND_PID_FILE"
    fi
    
    # Check port
    local port_pids=$(get_pid_from_port "$FRONTEND_PORT")
    if [[ -n "$port_pids" ]]; then
        for pid in $port_pids; do
            log "INFO" "Stopping process $pid on frontend port $FRONTEND_PORT"
            kill_process_tree "$pid"
            stopped=true
        done
    fi
    
    # Pattern-based cleanup
    local patterns=(
        "vite.*5173"
        "node.*frontend.*dev"
        "npm.*run.*dev"
    )
    
    for pattern in "${patterns[@]}"; do
        local pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [[ -n "$pids" ]]; then
            log "INFO" "Stopping processes matching: $pattern"
            for pid in $pids; do
                kill_process_tree "$pid"
                stopped=true
            done
        fi
    done
    
    if [[ "$stopped" == "true" ]]; then
        log "SUCCESS" "Frontend server stopped"
    else
        log "INFO" "No frontend server processes found"
    fi
}

verify_shutdown() {
    log "INFO" "Verifying shutdown..."
    
    local all_stopped=true
    
    # Check backend port
    if [[ -n "$(get_pid_from_port "$BACKEND_PORT")" ]]; then
        log "ERROR" "Backend port $BACKEND_PORT is still in use"
        all_stopped=false
    fi
    
    # Check frontend port
    if [[ -n "$(get_pid_from_port "$FRONTEND_PORT")" ]]; then
        log "ERROR" "Frontend port $FRONTEND_PORT is still in use"
        all_stopped=false
    fi
    
    # Check for any remaining processes
    local patterns=(
        "api_server_enhanced"
        "uvicorn.*8080"
        "vite.*5173"
    )
    
    for pattern in "${patterns[@]}"; do
        if pgrep -f "$pattern" >/dev/null 2>&1; then
            log "ERROR" "Processes still running matching: $pattern"
            all_stopped=false
        fi
    done
    
    if [[ "$all_stopped" == "true" ]]; then
        log "SUCCESS" "All Recursia services stopped successfully"
        return 0
    else
        log "ERROR" "Some services failed to stop properly"
        return 1
    fi
}

# ===========================
# MAIN EXECUTION
# ===========================

main() {
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║          ${YELLOW}RECURSIA PLATFORM SHUTDOWN${CYAN}                     ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo
    
    # Stop services
    stop_backend
    stop_frontend
    
    # Clean up PID directory if empty
    if [[ -d "$PID_DIR" ]] && [[ -z "$(ls -A "$PID_DIR")" ]]; then
        rmdir "$PID_DIR"
    fi
    
    # Verify shutdown
    sleep 2
    if verify_shutdown; then
        echo
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║         RECURSIA PLATFORM SHUTDOWN COMPLETE              ║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
        exit 0
    else
        echo
        echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║      WARNING: Some processes may still be running        ║${NC}"
        echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
        exit 1
    fi
}

# Run main
main "$@"