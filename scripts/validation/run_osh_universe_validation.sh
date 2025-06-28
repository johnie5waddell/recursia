#!/bin/bash
# OSH Universe Validation Runner
# ==============================
# Production-ready script to execute the 100,000 iteration OSH validation
# with proper environment setup, error handling, and monitoring.

set -euo pipefail  # Exit on error, undefined vars, pipe failures

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}OSH Universe Validation Runner${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

# Function to print colored messages
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
log_info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.8"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    log_error "Python 3.8+ is required. Found: $PYTHON_VERSION"
    exit 1
fi
log_info "Python version OK: $PYTHON_VERSION"

# Activate virtual environment
if [ -d "$PROJECT_ROOT/venv_linux" ]; then
    log_info "Activating Linux virtual environment..."
    source "$PROJECT_ROOT/venv_linux/bin/activate"
elif [ -d "$PROJECT_ROOT/venv" ]; then
    log_info "Activating virtual environment..."
    source "$PROJECT_ROOT/venv/bin/activate"
else
    log_error "No virtual environment found. Please run setup first."
    exit 1
fi

# Check required dependencies
log_info "Checking dependencies..."
python3 -c "import numpy, pandas, scipy, matplotlib" 2>/dev/null || {
    log_error "Missing required dependencies. Installing..."
    pip install numpy pandas scipy matplotlib seaborn
}

# Create output directories
OUTPUT_DIR="$PROJECT_ROOT/experiments/osh_validation_results"
mkdir -p "$OUTPUT_DIR"
log_info "Output directory: $OUTPUT_DIR"

# Check if validation program exists
VALIDATION_PROGRAM="$PROJECT_ROOT/validation_programs/osh_universe_validation.recursia"
if [ ! -f "$VALIDATION_PROGRAM" ]; then
    log_error "Validation program not found: $VALIDATION_PROGRAM"
    exit 1
fi
log_info "Validation program found"

# Set environment variables for optimal performance
export PYTHONUNBUFFERED=1  # Unbuffered output
export OMP_NUM_THREADS=4   # OpenMP threads for numpy
export NUMEXPR_NUM_THREADS=4  # NumExpr threads

# Resource monitoring setup
log_info "Setting up resource monitoring..."

# Function to monitor system resources
monitor_resources() {
    local PID=$1
    local MONITOR_FILE="$OUTPUT_DIR/resource_usage.log"
    
    echo "Timestamp,CPU%,Memory(MB),Status" > "$MONITOR_FILE"
    
    while kill -0 $PID 2>/dev/null; do
        if command -v ps >/dev/null 2>&1; then
            # Get CPU and memory usage
            STATS=$(ps -p $PID -o %cpu,rss --no-headers 2>/dev/null || echo "0 0")
            CPU=$(echo $STATS | awk '{print $1}')
            MEM_KB=$(echo $STATS | awk '{print $2}')
            MEM_MB=$((MEM_KB / 1024))
            
            echo "$(date +%Y-%m-%d_%H:%M:%S),$CPU,$MEM_MB,Running" >> "$MONITOR_FILE"
        fi
        sleep 5
    done
    
    echo "$(date +%Y-%m-%d_%H:%M:%S),0,0,Completed" >> "$MONITOR_FILE"
}

# Run validation with monitoring
log_info "Starting OSH universe validation..."
log_info "This will run 100,000 iterations and may take several minutes."
echo ""

# Start timer
START_TIME=$(date +%s)

# Run validation
cd "$PROJECT_ROOT"
python3 experiments/run_osh_universe_validation.py &
VALIDATION_PID=$!

# Start resource monitor in background
monitor_resources $VALIDATION_PID &
MONITOR_PID=$!

# Wait for validation to complete
wait $VALIDATION_PID
VALIDATION_EXIT_CODE=$?

# Stop resource monitor
kill $MONITOR_PID 2>/dev/null || true

# Calculate execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
if [ $VALIDATION_EXIT_CODE -eq 0 ]; then
    log_info "Validation completed successfully!"
    log_info "Execution time: ${MINUTES}m ${SECONDS}s"
    
    # List output files
    echo ""
    log_info "Generated files:"
    ls -la "$OUTPUT_DIR"/*.* 2>/dev/null | while read line; do
        echo "  $line"
    done
    
    # Show summary if report exists
    REPORT_FILE="$OUTPUT_DIR/osh_validation_report.txt"
    if [ -f "$REPORT_FILE" ]; then
        echo ""
        log_info "Validation summary:"
        grep -A 20 "FINAL VERDICT" "$REPORT_FILE" | head -25
    fi
    
    # Display visualization if in graphical environment
    if [ -n "${DISPLAY:-}" ] && command -v xdg-open >/dev/null 2>&1; then
        IMAGE_FILE="$OUTPUT_DIR/osh_validation_results.png"
        if [ -f "$IMAGE_FILE" ]; then
            log_info "Opening visualization..."
            xdg-open "$IMAGE_FILE" 2>/dev/null &
        fi
    fi
else
    log_error "Validation failed with exit code: $VALIDATION_EXIT_CODE"
    log_error "Check the log files in $OUTPUT_DIR for details"
    exit 1
fi

echo ""
echo -e "${BLUE}================================${NC}"
echo -e "${GREEN}Validation pipeline complete!${NC}"
echo -e "${BLUE}================================${NC}"