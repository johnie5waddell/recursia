# Recursia Startup Script for Windows PowerShell - FIXED VERSION
# Addresses common Windows startup issues

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "     Recursia Quantum OSH Computing Platform" -ForegroundColor Cyan
Write-Host "==================================================`n" -ForegroundColor Cyan

# Helper function to test if a command exists
function Test-Command {
    param($CommandName)
    try {
        if (Get-Command $CommandName -ErrorAction Stop) {
            return $true
        }
    }
    catch {
        return $false
    }
}

# Check if we're in the right directory
if (-not (Test-Path "src/recursia.py")) {
    Write-Host "Error: Not in Recursia root directory!" -ForegroundColor Red
    Write-Host "Please cd to the Recursia folder and try again." -ForegroundColor Yellow
    exit 1
}

# Kill any existing processes on our ports
Write-Host "Checking for existing processes..." -ForegroundColor Blue
$ports = @(8080, 5173)
foreach ($port in $ports) {
    try {
        $connection = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($connection) {
            $pid = $connection.OwningProcess
            $processName = (Get-Process -Id $pid -ErrorAction SilentlyContinue).ProcessName
            Write-Host "Killing $processName (PID: $pid) on port $port" -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    }
    catch {
        # Port is free, continue
    }
}

# Check Python
Write-Host "Checking Python..." -ForegroundColor Blue
$pythonCmd = $null

if (Test-Command python) {
    $version = & python --version 2>&1
    if ($version -match "Python 3\.([89]|1[0-9])") {
        $pythonCmd = "python"
        Write-Host "Found: $version" -ForegroundColor Green
    }
}

if (-not $pythonCmd) {
    if (Test-Command python3) {
        $version = & python3 --version 2>&1
        if ($version -match "Python 3\.([89]|1[0-9])") {
            $pythonCmd = "python3"
            Write-Host "Found: $version" -ForegroundColor Green
        }
    }
}

if (-not $pythonCmd) {
    Write-Host "Python 3.8+ not found! Please install Python from python.org" -ForegroundColor Red
    exit 1
}

# Check Node.js
Write-Host "`nChecking Node.js..." -ForegroundColor Blue
if (Test-Command node) {
    $nodeVersion = & node --version 2>&1
    Write-Host "Found Node.js: $nodeVersion" -ForegroundColor Green
}
else {
    Write-Host "Node.js not found! Please install from nodejs.org" -ForegroundColor Red
    exit 1
}

# Set up virtual environment
Write-Host "`nSetting up Python environment..." -ForegroundColor Blue
$venvDir = "venv"

if (-not (Test-Path $venvDir)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    & $pythonCmd -m venv $venvDir
}

# Activate virtual environment
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Host "Virtual environment activated" -ForegroundColor Green
}

# Update pip first to avoid issues
Write-Host "`nUpdating pip..." -ForegroundColor Blue
& $pythonCmd -m pip install --upgrade pip --quiet

# Install Python dependencies
Write-Host "Checking Python dependencies..." -ForegroundColor Blue
if (Test-Path "requirements.txt") {
    $packagesOk = $false
    try {
        & $pythonCmd -c "import fastapi, uvicorn, numpy" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $packagesOk = $true
        }
    }
    catch {
        $packagesOk = $false
    }
    
    if (-not $packagesOk) {
        Write-Host "Installing Python packages..." -ForegroundColor Yellow
        & $pythonCmd -m pip install -r requirements.txt
        # Don't install in editable mode on Windows, it can cause issues
        # & $pythonCmd -m pip install -e .
    }
    else {
        Write-Host "Python packages already installed" -ForegroundColor Green
    }
}

# Install frontend dependencies
Write-Host "`nSetting up frontend..." -ForegroundColor Blue
Push-Location frontend
if (-not (Test-Path "node_modules")) {
    Write-Host "Installing frontend packages..." -ForegroundColor Yellow
    npm install
}
else {
    Write-Host "Frontend packages already installed" -ForegroundColor Green
}
Pop-Location

# Check ML models
Write-Host "`nChecking ML models..." -ForegroundColor Blue
$mlModelsPath = "src\quantum\decoders\models"
$requiredModels = @(
    "surface_d3_model.pt",
    "surface_d5_model.pt", 
    "surface_d7_model.pt",
    "steane_d3_model.pt",
    "shor_d3_model.pt"
)

$missingModels = 0
foreach ($model in $requiredModels) {
    if (-not (Test-Path (Join-Path $mlModelsPath $model))) {
        $missingModels++
    }
}

if ($missingModels -gt 0) {
    Write-Host "Missing $missingModels ML models. Training..." -ForegroundColor Yellow
    Write-Host "This will take 5-10 minutes on first run." -ForegroundColor Yellow
    
    # Check PyTorch
    try {
        & $pythonCmd -c "import torch" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing PyTorch..." -ForegroundColor Yellow
            & $pythonCmd -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        }
    }
    catch {
        Write-Host "Installing PyTorch..." -ForegroundColor Yellow
        & $pythonCmd -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
    
    # Train models
    if (Test-Path "scripts\validation\train_ml_decoders.py") {
        & $pythonCmd scripts\validation\train_ml_decoders.py
    }
}
else {
    Write-Host "All ML models present" -ForegroundColor Green
}

# Create backend batch file to ensure proper environment
$backendBat = @"
@echo off
cd /d "$((Get-Location).Path)"
call venv\Scripts\activate.bat
python -m uvicorn src.api.unified_api_server:app --host 127.0.0.1 --port 8080 --reload
"@
$backendBat | Out-File -FilePath "start-backend.bat" -Encoding ASCII

# Create frontend batch file
$frontendBat = @"
@echo off
cd /d "$((Get-Location).Path)\frontend"
call npm run dev
"@
$frontendBat | Out-File -FilePath "start-frontend.bat" -Encoding ASCII

# Start servers
Write-Host "`nStarting servers..." -ForegroundColor Cyan

# Start backend using cmd to ensure proper environment
Write-Host "Starting backend server on port 8080..." -ForegroundColor Blue
$backend = Start-Process -FilePath "cmd.exe" -ArgumentList "/c start-backend.bat" -PassThru -WindowStyle Minimized

# Wait for backend to start
Write-Host "Waiting for backend to initialize..." -ForegroundColor Yellow
$backendReady = $false
$attempts = 0
$maxAttempts = 30

while (-not $backendReady -and $attempts -lt $maxAttempts) {
    Start-Sleep -Seconds 1
    $attempts++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $backendReady = $true
            Write-Host "Backend is ready!" -ForegroundColor Green
        }
    }
    catch {
        Write-Host "." -NoNewline
    }
}

if (-not $backendReady) {
    Write-Host "`nBackend failed to start after $maxAttempts seconds!" -ForegroundColor Red
    Write-Host "Check if another process is using port 8080" -ForegroundColor Yellow
    
    # Clean up
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    Remove-Item "start-backend.bat" -ErrorAction SilentlyContinue
    Remove-Item "start-frontend.bat" -ErrorAction SilentlyContinue
    exit 1
}

# Start frontend
Write-Host "`nStarting frontend server on port 5173..." -ForegroundColor Blue
$frontend = Start-Process -FilePath "cmd.exe" -ArgumentList "/c start-frontend.bat" -PassThru -WindowStyle Minimized

# Wait for frontend to start
Write-Host "Waiting for frontend to initialize..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if frontend started
$frontendReady = $false
$attempts = 0

while (-not $frontendReady -and $attempts -lt 30) {
    Start-Sleep -Seconds 1
    $attempts++
    
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:5173" -TimeoutSec 1 -ErrorAction SilentlyContinue
        if ($response.StatusCode -eq 200) {
            $frontendReady = $true
            Write-Host "Frontend is ready!" -ForegroundColor Green
        }
    }
    catch {
        # Frontend might return different status codes during startup
        if ($_.Exception.Response.StatusCode.Value__ -in @(200, 404)) {
            $frontendReady = $true
            Write-Host "Frontend is ready!" -ForegroundColor Green
        }
        else {
            Write-Host "." -NoNewline
        }
    }
}

# Open browser
if ($frontendReady) {
    Write-Host "`nOpening browser..." -ForegroundColor Green
    Start-Process "http://localhost:5173"
}

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "Recursia is running!" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8080" -ForegroundColor Cyan
Write-Host "`nPress Ctrl+C to stop all servers" -ForegroundColor Yellow
Write-Host "==================================================" -Green

# Monitor servers
$stopRequested = $false
$lastCheck = Get-Date

try {
    # Register Ctrl+C handler
    [Console]::TreatControlCAsInput = $true
    
    while (-not $stopRequested) {
        if ([Console]::KeyAvailable) {
            $key = [Console]::ReadKey($true)
            if ($key.Key -eq "C" -and $key.Modifiers -eq "Control") {
                $stopRequested = $true
                Write-Host "`nStop requested..." -ForegroundColor Yellow
            }
        }
        
        # Check server status every 5 seconds
        if ((Get-Date) -gt $lastCheck.AddSeconds(5)) {
            $lastCheck = Get-Date
            
            if ($backend.HasExited) {
                Write-Host "`nBackend server stopped unexpectedly!" -ForegroundColor Red
                $stopRequested = $true
            }
            
            if ($frontend.HasExited) {
                Write-Host "`nFrontend server stopped unexpectedly!" -ForegroundColor Red
                $stopRequested = $true
            }
        }
        
        Start-Sleep -Milliseconds 100
    }
}
finally {
    [Console]::TreatControlCAsInput = $false
    
    Write-Host "`nShutting down..." -ForegroundColor Yellow
    
    # Kill processes
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
        # Also kill any child python processes
        Get-Process python -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -like "*uvicorn*"
        } | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    
    if (-not $frontend.HasExited) {
        Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
        # Also kill any child node processes
        Get-Process node -ErrorAction SilentlyContinue | Where-Object {
            $_.CommandLine -like "*vite*" -or $_.CommandLine -like "*dev*"
        } | Stop-Process -Force -ErrorAction SilentlyContinue
    }
    
    # Clean up batch files
    Remove-Item "start-backend.bat" -ErrorAction SilentlyContinue
    Remove-Item "start-frontend.bat" -ErrorAction SilentlyContinue
    
    Write-Host "Shutdown complete." -ForegroundColor Green
}