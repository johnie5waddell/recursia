#!/bin/bash
# Recursia Platform Restart Script - Enterprise Grade
# Performs clean restart of all services

set -euo pipefail

# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly CYAN='\033[0;36m'
readonly YELLOW='\033[1;33m'
readonly NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          ${YELLOW}RECURSIA PLATFORM RESTART${CYAN}                      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo

# Stop existing services
"$SCRIPT_DIR/stop-recursia.sh"

echo
echo "Waiting for services to fully terminate..."
sleep 3

# Start services
"$SCRIPT_DIR/start-recursia.sh" "$@"