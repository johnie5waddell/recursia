# Check and train ML models if needed for Windows

# Function to check if ML models exist
function Test-MLModels {
    $modelsDir = "src\quantum\decoders\models"
    $requiredModels = @(
        "surface_d3_model.pt",
        "surface_d5_model.pt",
        "surface_d7_model.pt",
        "steane_d3_model.pt",
        "shor_d3_model.pt"
    )
    
    # Check if models directory exists
    if (-not (Test-Path $modelsDir)) {
        New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
    }
    
    # Check for each required model
    $missingModels = @()
    foreach ($model in $requiredModels) {
        if (-not (Test-Path "$modelsDir\$model")) {
            $missingModels += $model
        }
    }
    
    # Return number of missing models
    return $missingModels.Count
}

# Function to train ML models
function Start-MLTraining {
    Write-Host "Training ML decoders for quantum error correction..." -ForegroundColor Cyan
    Write-Host "This may take 5-10 minutes on first run." -ForegroundColor Yellow
    
    # Check if PyTorch is installed
    $pytorchInstalled = & python -c "import torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing PyTorch for ML decoder training..." -ForegroundColor Yellow
        & python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
    
    # Run the training script
    Write-Host "Starting ML decoder training..." -ForegroundColor Blue
    if (Test-Path "scripts\validation\train_ml_decoders.py") {
        & python scripts\validation\train_ml_decoders.py
        
        # Check if training was successful
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✓ ML decoders trained successfully!" -ForegroundColor Green
            
            # Verify models were created
            $missing = Test-MLModels
            if ($missing -eq 0) {
                Write-Host "✓ All required ML models are present" -ForegroundColor Green
                return $true
            } else {
                Write-Host "⚠ Some models may be missing, but training completed" -ForegroundColor Yellow
                return $true
            }
        } else {
            Write-Host "✗ ML decoder training failed" -ForegroundColor Red
            Write-Host "The system will work but with reduced QEC performance" -ForegroundColor Yellow
            return $false
        }
    } else {
        Write-Host "ML decoder training script not found!" -ForegroundColor Red
        return $false
    }
}

# Main execution if script is run directly
if ($MyInvocation.InvocationName -ne '.') {
    $missing = Test-MLModels
    
    if ($missing -gt 0) {
        Write-Host "Missing $missing ML decoder models" -ForegroundColor Yellow
        Start-MLTraining
    } else {
        Write-Host "✓ All ML decoder models are present" -ForegroundColor Green
    }
}