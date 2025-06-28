#!/bin/bash
# Script to properly restart Recursia services

echo "=== Restarting Recursia Services ==="
echo

# Kill all existing processes
echo "Stopping existing processes..."
pkill -f "api_server"
pkill -f "cli_dashboard"
pkill -f "recursia"
sleep 2

# Clear Python cache
echo "Clearing Python cache..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Activate virtual environment
echo "Activating virtual environment..."
source venv_linux/bin/activate

# Start API server
echo "Starting API server..."
nohup python -m src.api_server_enhanced > api_server_new.log 2>&1 &
echo "API server started with PID $!"

# Give it time to start
sleep 5

# Start dashboard
echo "Starting dashboard..."
python src/cli_dashboard.py &
echo "Dashboard started with PID $!"

echo
echo "=== Services Restarted ==="
echo
echo "API server log: api_server_new.log"
echo "Dashboard available at: http://localhost:8000"
echo
echo "To stop services, run:"
echo "  pkill -f api_server"
echo "  pkill -f cli_dashboard"