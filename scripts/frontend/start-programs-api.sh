#!/bin/bash

# Start the program loader API server

echo "Starting Recursia Program Loader API..."

# Navigate to project root
cd "$(dirname "$0")/../.."

# Activate virtual environment if it exists
if [ -d "venv_linux" ]; then
    source venv_linux/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the API server
echo "Starting API server on port 5000..."
python src/api/program_loader_api.py