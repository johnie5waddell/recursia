#!/bin/bash
# Recursia Platform Status Script - Enterprise Grade
# Shows detailed status of all Recursia services

set -euo pipefail

# Configuration
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
readonly BACKEND_URL="http://localhost:$BACKEND_PORT"
readonly FRONTEND_URL="http://localhost:$FRONTEND_PORT"

# Color codes
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly MAGENTA='\033[0;35m'
readonly NC='\033[0m'

# Status icons
readonly CHECK_MARK="✓"
readonly CROSS_MARK="✗"
readonly WARNING_SIGN="⚠"

# ===========================
# UTILITY FUNCTIONS
# ===========================

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

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

format_uptime() {
    local pid=$1
    if [[ -z "$pid" ]]; then
        echo "N/A"
        return
    fi
    
    local uptime_seconds=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ -z "$uptime_seconds" ]] || [[ "$uptime_seconds" == "0" ]]; then
        echo "N/A"
        return
    fi
    
    local days=$((uptime_seconds / 86400))
    local hours=$(((uptime_seconds % 86400) / 3600))
    local minutes=$(((uptime_seconds % 3600) / 60))
    local seconds=$((uptime_seconds % 60))
    
    local result=""
    [[ $days -gt 0 ]] && result="${days}d "
    [[ $hours -gt 0 ]] && result="${result}${hours}h "
    [[ $minutes -gt 0 ]] && result="${result}${minutes}m "
    result="${result}${seconds}s"
    
    echo "$result"
}

get_memory_usage() {
    local pid=$1
    if [[ -z "$pid" ]]; then
        echo "N/A"
        return
    fi
    
    local mem_kb=$(ps -o vsz= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ -z "$mem_kb" ]] || [[ "$mem_kb" == "0" ]]; then
        echo "N/A"
        return
    fi
    
    local mem_mb=$((mem_kb / 1024))
    echo "${mem_mb}MB"
}

get_cpu_usage() {
    local pid=$1
    if [[ -z "$pid" ]]; then
        echo "N/A"
        return
    fi
    
    local cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0")
    if [[ -z "$cpu" ]]; then
        echo "N/A"
        return
    fi
    
    echo "${cpu}%"
}

check_http_endpoint() {
    local url=$1
    local timeout=5
    
    if curl -s -f -m "$timeout" "$url" >/dev/null 2>&1; then
        echo "OK"
        return 0
    else
        echo "FAILED"
        return 1
    fi
}

# ===========================
# STATUS CHECK FUNCTIONS
# ===========================

check_backend_status() {
    echo -e "${CYAN}Backend API Server:${NC}"
    echo "────────────────────────────────────────"
    
    local backend_pid=""
    local status="${RED}${CROSS_MARK} STOPPED${NC}"
    local pid_source="none"
    
    # Check PID file
    if [[ -f "$BACKEND_PID_FILE" ]]; then
        backend_pid=$(cat "$BACKEND_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
            status="${GREEN}${CHECK_MARK} RUNNING${NC}"
            pid_source="pid_file"
        fi
    fi
    
    # Check port if PID not found
    if [[ -z "$backend_pid" ]] || ! kill -0 "$backend_pid" 2>/dev/null; then
        backend_pid=$(get_pid_from_port "$BACKEND_PORT")
        if [[ -n "$backend_pid" ]]; then
            status="${YELLOW}${WARNING_SIGN} RUNNING${NC} (no PID file)"
            pid_source="port"
        fi
    fi
    
    echo -e "  Status: $status"
    echo -e "  Port: ${CYAN}$BACKEND_PORT${NC}"
    
    if [[ -n "$backend_pid" ]] && kill -0 "$backend_pid" 2>/dev/null; then
        echo -e "  PID: ${MAGENTA}$backend_pid${NC} (from $pid_source)"
        echo -e "  Uptime: $(format_uptime "$backend_pid")"
        echo -e "  Memory: $(get_memory_usage "$backend_pid")"
        echo -e "  CPU: $(get_cpu_usage "$backend_pid")"
        
        # Check API endpoints
        echo -e "  API Endpoints:"
        echo -n "    - /api/metrics: "
        if check_http_endpoint "$BACKEND_URL/api/metrics"; then
            echo -e "${GREEN}${CHECK_MARK} OK${NC}"
        else
            echo -e "${RED}${CROSS_MARK} FAILED${NC}"
        fi
        
        echo -n "    - /api/execute: "
        if curl -s -X POST "$BACKEND_URL/api/execute" -H "Content-Type: application/json" -d '{}' >/dev/null 2>&1; then
            echo -e "${GREEN}${CHECK_MARK} OK${NC}"
        else
            echo -e "${RED}${CROSS_MARK} FAILED${NC}"
        fi
    else
        echo -e "  ${RED}Service is not running${NC}"
    fi
    
    echo
}

check_frontend_status() {
    echo -e "${CYAN}Frontend Development Server:${NC}"
    echo "────────────────────────────────────────"
    
    local frontend_pid=""
    local status="${RED}${CROSS_MARK} STOPPED${NC}"
    local pid_source="none"
    
    # Check PID file
    if [[ -f "$FRONTEND_PID_FILE" ]]; then
        frontend_pid=$(cat "$FRONTEND_PID_FILE" 2>/dev/null || echo "")
        if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
            status="${GREEN}${CHECK_MARK} RUNNING${NC}"
            pid_source="pid_file"
        fi
    fi
    
    # Check port if PID not found
    if [[ -z "$frontend_pid" ]] || ! kill -0 "$frontend_pid" 2>/dev/null; then
        frontend_pid=$(get_pid_from_port "$FRONTEND_PORT")
        if [[ -n "$frontend_pid" ]]; then
            status="${YELLOW}${WARNING_SIGN} RUNNING${NC} (no PID file)"
            pid_source="port"
        fi
    fi
    
    echo -e "  Status: $status"
    echo -e "  Port: ${CYAN}$FRONTEND_PORT${NC}"
    
    if [[ -n "$frontend_pid" ]] && kill -0 "$frontend_pid" 2>/dev/null; then
        echo -e "  PID: ${MAGENTA}$frontend_pid${NC} (from $pid_source)"
        echo -e "  Uptime: $(format_uptime "$frontend_pid")"
        echo -e "  Memory: $(get_memory_usage "$frontend_pid")"
        echo -e "  CPU: $(get_cpu_usage "$frontend_pid")"
        
        # Check frontend accessibility
        echo -n "  Web UI: "
        if check_http_endpoint "$FRONTEND_URL"; then
            echo -e "${GREEN}${CHECK_MARK} ACCESSIBLE${NC}"
        else
            echo -e "${RED}${CROSS_MARK} NOT ACCESSIBLE${NC}"
        fi
    else
        echo -e "  ${RED}Service is not running${NC}"
    fi
    
    echo
}

check_system_resources() {
    echo -e "${CYAN}System Resources:${NC}"
    echo "────────────────────────────────────────"
    
    # Memory
    if command_exists free; then
        local total_mem=$(free -m | awk 'NR==2{print $2}')
        local used_mem=$(free -m | awk 'NR==2{print $3}')
        local free_mem=$(free -m | awk 'NR==2{print $4}')
        local mem_percent=$((used_mem * 100 / total_mem))
        
        echo -e "  Memory: ${used_mem}MB / ${total_mem}MB (${mem_percent}% used)"
        
        if [[ $mem_percent -gt 90 ]]; then
            echo -e "  ${RED}${WARNING_SIGN} Memory usage is high!${NC}"
        fi
    fi
    
    # CPU
    if command_exists uptime; then
        local load_avg=$(uptime | awk -F'load average:' '{print $2}' | xargs)
        echo -e "  Load Average: $load_avg"
    fi
    
    # Disk
    local project_disk=$(df -h "$SCRIPT_DIR" | awk 'NR==2{print $5}' | sed 's/%//')
    echo -e "  Project Disk Usage: ${project_disk}%"
    
    if [[ $project_disk -gt 90 ]]; then
        echo -e "  ${RED}${WARNING_SIGN} Disk usage is high!${NC}"
    fi
    
    echo
}

check_logs() {
    echo -e "${CYAN}Recent Log Activity:${NC}"
    echo "────────────────────────────────────────"
    
    local log_dir="$SCRIPT_DIR/logs"
    if [[ -d "$log_dir" ]]; then
        # Find most recent backend log
        local backend_log=$(find "$log_dir" -name "backend_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [[ -n "$backend_log" ]] && [[ -f "$backend_log" ]]; then
            echo -e "  Backend Log: ${BLUE}$backend_log${NC}"
            if [[ -f "$backend_log" ]]; then
                local last_lines=$(tail -3 "$backend_log" 2>/dev/null | sed 's/^/    /')
                if [[ -n "$last_lines" ]]; then
                    echo -e "  Last entries:"
                    echo "$last_lines"
                fi
            fi
        fi
        
        echo
        
        # Find most recent frontend log
        local frontend_log=$(find "$log_dir" -name "frontend_*.log" -type f -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
        if [[ -n "$frontend_log" ]] && [[ -f "$frontend_log" ]]; then
            echo -e "  Frontend Log: ${BLUE}$frontend_log${NC}"
            if [[ -f "$frontend_log" ]]; then
                local last_lines=$(tail -3 "$frontend_log" 2>/dev/null | sed 's/^/    /')
                if [[ -n "$last_lines" ]]; then
                    echo -e "  Last entries:"
                    echo "$last_lines"
                fi
            fi
        fi
    else
        echo -e "  ${YELLOW}No log directory found${NC}"
    fi
    
    echo
}

# ===========================
# MAIN EXECUTION
# ===========================

main() {
    clear
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          ${CYAN}RECURSIA PLATFORM STATUS${GREEN}                       ║${NC}"
    echo -e "${GREEN}║          $(date '+%Y-%m-%d %H:%M:%S')                          ${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo
    
    check_backend_status
    check_frontend_status
    check_system_resources
    check_logs
    
    # Summary
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                      SUMMARY                             ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════╣${NC}"
    
    local backend_running=false
    local frontend_running=false
    
    if [[ -n "$(get_pid_from_port "$BACKEND_PORT")" ]]; then
        backend_running=true
    fi
    
    if [[ -n "$(get_pid_from_port "$FRONTEND_PORT")" ]]; then
        frontend_running=true
    fi
    
    if [[ "$backend_running" == "true" ]] && [[ "$frontend_running" == "true" ]]; then
        echo -e "${GREEN}║ ${CHECK_MARK} All services are running                              ║${NC}"
        echo -e "${GREEN}║                                                          ║${NC}"
        echo -e "${GREEN}║ Access the platform at:                                  ║${NC}"
        echo -e "${GREEN}║   ${CYAN}$FRONTEND_URL${GREEN}                           ║${NC}"
    elif [[ "$backend_running" == "true" ]] || [[ "$frontend_running" == "true" ]]; then
        echo -e "${YELLOW}║ ${WARNING_SIGN} Some services are not running                         ║${NC}"
        echo -e "${YELLOW}║                                                          ║${NC}"
        echo -e "${YELLOW}║ Run ./start-recursia.sh to start all services           ║${NC}"
    else
        echo -e "${RED}║ ${CROSS_MARK} No services are running                               ║${NC}"
        echo -e "${RED}║                                                          ║${NC}"
        echo -e "${RED}║ Run ./start-recursia.sh to start the platform           ║${NC}"
    fi
    
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
}

# Run main
main "$@"