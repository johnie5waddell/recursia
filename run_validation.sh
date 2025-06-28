#!/bin/bash
#
# OSH Validation Suite Runner
# Easy-to-use wrapper for running validation experiments
#

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
ITERATIONS=10
PARALLEL=1
PROGRAM="quantum_programs/validation/osh_validation_optimized.recursia"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--iterations)
            ITERATIONS="$2"
            shift 2
            ;;
        -p|--parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --program)
            PROGRAM="$2"
            shift 2
            ;;
        -h|--help)
            echo "OSH Validation Suite Runner"
            echo ""
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  -i, --iterations NUM    Number of iterations (default: 10)"
            echo "  -p, --parallel NUM      Number of parallel workers (default: 1)"
            echo "  --program PATH          Path to validation program"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Quick test with 5 iterations"
            echo "  $0 -i 5"
            echo ""
            echo "  # Full validation with 100 iterations using 4 workers"
            echo "  $0 -i 100 -p 4"
            echo ""
            echo "  # Test advanced validation program"
            echo "  $0 --program quantum_programs/validation/osh_validation_advanced.recursia -i 20"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not activated${NC}"
    echo -e "${CYAN}Attempting to activate venv...${NC}"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        echo -e "${GREEN}✓ Virtual environment activated${NC}"
    else
        echo -e "${RED}Error: venv not found. Please create it first:${NC}"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r requirements.txt"
        exit 1
    fi
fi

# Display configuration
echo -e "${PURPLE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║          OSH Validation Suite - Enhanced UX              ║${NC}"
echo -e "${PURPLE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo -e "  • Iterations: ${GREEN}$ITERATIONS${NC}"
echo -e "  • Parallel Workers: ${GREEN}$PARALLEL${NC}"
echo -e "  • Program: ${GREEN}$(basename $PROGRAM)${NC}"
echo ""

# Run the validation suite
python experiments/osh_validation_suite_enhanced.py \
    --iterations $ITERATIONS \
    --parallel $PARALLEL \
    --program "$PROGRAM"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                  Validation Complete!                     ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
else
    echo ""
    echo -e "${RED}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║                   Validation Failed                       ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════╝${NC}"
fi