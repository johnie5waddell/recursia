#!/bin/bash
# Start API server for testing

# Get the project root (two levels up from this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect virtual environment
if [ -d "venv_linux/bin" ]; then
    source venv_linux/bin/activate
elif [ -d "venv_mac/bin" ]; then
    source venv_mac/bin/activate
elif [ -d "venv/bin" ]; then
    source venv/bin/activate
elif [ -d "venv_windows/Scripts" ]; then
    source venv_windows/Scripts/activate
fi

echo "Starting Recursia API Server for testing..."
echo "Press Ctrl+C to stop"
echo ""

# Run with debug mode for better error messages
python3 src/api_server.py --port 5000 --debug