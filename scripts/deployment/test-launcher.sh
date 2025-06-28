#!/bin/bash

# Quick test script for auto-launcher functionality
echo "🧪 Testing Recursia Auto-Launcher Components..."

# Test Python availability
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3 not found"
fi

# Test Node.js availability
if command -v node >/dev/null 2>&1; then
    NODE_VERSION=$(node --version)
    echo "✅ Node.js $NODE_VERSION found"
else
    echo "❌ Node.js not found"
fi

# Test npm availability
if command -v npm >/dev/null 2>&1; then
    NPM_VERSION=$(npm --version)
    echo "✅ npm $NPM_VERSION found"
else
    echo "❌ npm not found"
fi

# Check virtual environment
if [ -d "venv_linux" ]; then
    echo "✅ Python virtual environment exists"
else
    echo "⚠️  Python virtual environment not found"
fi

# Check frontend dependencies
if [ -d "frontend/node_modules" ]; then
    echo "✅ Frontend dependencies installed"
else
    echo "⚠️  Frontend dependencies not installed"
fi

# Check key files
if [ -f "requirements.txt" ]; then
    echo "✅ requirements.txt found"
else
    echo "❌ requirements.txt missing"
fi

if [ -f "frontend/package.json" ]; then
    echo "✅ Frontend package.json found"
else
    echo "❌ Frontend package.json missing"
fi

if [ -f "src/cli_dashboard.py" ]; then
    echo "✅ Backend server found"
else
    echo "❌ Backend server missing"
fi

echo ""
echo "🚀 Auto-launcher test complete!"
echo "Ready to run: ./start-recursia.sh"