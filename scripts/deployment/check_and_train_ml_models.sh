#!/bin/bash
# Check and train ML models if needed

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to check if ML models exist
check_ml_models() {
    local models_dir="src/quantum/decoders/models"
    local required_models=(
        "surface_d3_model.pt"
        "surface_d5_model.pt"
        "surface_d7_model.pt"
        "steane_d3_model.pt"
        "shor_d3_model.pt"
    )
    
    # Check if models directory exists
    if [ ! -d "$models_dir" ]; then
        mkdir -p "$models_dir"
    fi
    
    # Check for each required model
    local missing_models=()
    for model in "${required_models[@]}"; do
        if [ ! -f "$models_dir/$model" ]; then
            missing_models+=("$model")
        fi
    done
    
    # Return number of missing models
    echo ${#missing_models[@]}
}

# Function to train ML models
train_ml_models() {
    echo -e "${CYAN}Training ML decoders for quantum error correction...${NC}"
    echo -e "${YELLOW}This may take 5-10 minutes on first run.${NC}"
    
    # Activate virtual environment if not already active
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        if [ -f "venv/bin/activate" ]; then
            source venv/bin/activate
        elif [ -f ".venv/bin/activate" ]; then
            source .venv/bin/activate
        fi
    fi
    
    # Find Python command
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Python not found!${NC}"
        return 1
    fi
    
    # Check if PyTorch is installed
    if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
        echo -e "${YELLOW}Installing PyTorch for ML decoder training...${NC}"
        $PYTHON_CMD -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Run the training script
    echo -e "${BLUE}Starting ML decoder training...${NC}"
    if [ -f "scripts/validation/train_ml_decoders.py" ]; then
        $PYTHON_CMD scripts/validation/train_ml_decoders.py
        
        # Check if training was successful
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ ML decoders trained successfully!${NC}"
            
            # Verify models were created
            local missing=$(check_ml_models)
            if [ "$missing" -eq 0 ]; then
                echo -e "${GREEN}✓ All required ML models are present${NC}"
                return 0
            else
                echo -e "${YELLOW}⚠ Some models may be missing, but training completed${NC}"
                return 0
            fi
        else
            echo -e "${RED}✗ ML decoder training failed${NC}"
            echo -e "${YELLOW}The system will work but with reduced QEC performance${NC}"
            return 1
        fi
    else
        echo -e "${RED}ML decoder training script not found!${NC}"
        return 1
    fi
}

# Main execution
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    # Script is being executed directly
    missing=$(check_ml_models)
    
    if [ "$missing" -gt 0 ]; then
        echo -e "${YELLOW}Missing $missing ML decoder models${NC}"
        train_ml_models
    else
        echo -e "${GREEN}✓ All ML decoder models are present${NC}"
    fi
fi

# Export functions for use in other scripts
export -f check_ml_models
export -f train_ml_models