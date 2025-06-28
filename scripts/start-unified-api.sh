#!/bin/bash

# Start Unified API Server
# Uses centralized runtime instance for all operations

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Activate virtual environment if it exists
if [ -d "$PROJECT_ROOT/venv_linux" ]; then
    source "$PROJECT_ROOT/venv_linux/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    source "$PROJECT_ROOT/venv/bin/activate"
fi

# Set Python path
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Start the unified API server
echo "Starting Unified Recursia API Server..."
echo "This server uses the centralized runtime instance"
echo "All subsystems are accessed through the runtime"
echo ""

cd "$PROJECT_ROOT"
python -m src.api.unified_api_server "$@"