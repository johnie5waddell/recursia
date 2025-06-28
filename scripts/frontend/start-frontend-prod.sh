#!/bin/bash

# Production-ready frontend start script
# Ensures all environment variables are properly set

echo "Starting Recursia Frontend in production mode..."

# Navigate to frontend directory
cd "$(dirname "$0")/../../frontend"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Build for production
echo "Building frontend for production..."
npm run build

# Check if build was successful
if [ ! -d "dist" ]; then
    echo "Build failed! Check for errors above."
    exit 1
fi

# Start preview server
echo "Starting production preview server..."
echo "Access the application at http://localhost:4173"
npm run preview -- --host 0.0.0.0 --port 4173