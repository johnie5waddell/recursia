# Recursia Startup Script for Windows PowerShell
# Complete self-contained script with PID management

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

# Helper function to kill processes by port
function Stop-ProcessByPort {
    param($Port)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $pids = $connections.OwningProcess | Sort-Object -Unique
            foreach ($pid in $pids) {
                if ($pid -gt 0) {
                    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($process) {
                        Write-Host "Stopping $($process.ProcessName) (PID: $pid) on port $Port" -ForegroundColor Yellow
                        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    }
                }
            }
            Start-Sleep -Seconds 2
        }
    }
    catch {
        # Port is free or error accessing
    }
}

# Helper function to save PID
function Save-PID {
    param($ProcessId, $Type)
    $pidFile = "$Type.pid"
    $ProcessId | Out-File -FilePath $pidFile -Force
    Write-Host "Saved $Type PID: $ProcessId to $pidFile" -ForegroundColor DarkGray
}

# Helper function to read saved PID
function Get-SavedPID {
    param($Type)
    $pidFile = "$Type.pid"
    if (Test-Path $pidFile) {
        $pid = Get-Content $pidFile -ErrorAction SilentlyContinue
        if ($pid) {
            return [int]$pid
        }
    }
    return $null
}

# Helper function to check if saved process is still running
function Test-SavedProcess {
    param($Type)
    $pid = Get-SavedPID $Type
    if ($pid) {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            return $true
        }
    }
    return $false
}

# Helper function to stop saved process
function Stop-SavedProcess {
    param($Type)
    $savedPid = Get-SavedPID $Type
    if ($savedPid) {
        $process = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping saved $Type process (PID: $savedPid)" -ForegroundColor Yellow
            Stop-Process -Id $savedPid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
    }
    # Remove PID file
    $pidFile = "$Type.pid"
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# Check if we're in the right directory
if (-not (Test-Path "src/recursia.py")) {
    Write-Host "Error: Not in Recursia root directory!" -ForegroundColor Red
    Write-Host "Please cd to the Recursia folder and try again." -ForegroundColor Yellow
    exit 1
}

# Clean up any existing Recursia servers
Write-Host "Checking for existing Recursia servers..." -ForegroundColor Blue

# Stop saved processes first
Stop-SavedProcess "backend"
Stop-SavedProcess "frontend"

# Also stop any processes on our ports
Stop-ProcessByPort 8080
Stop-ProcessByPort 5173

# Clean up any orphaned Python/Node processes
Get-Process python, python3 -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*unified_api_server*"
} | ForEach-Object {
    Write-Host "Stopping orphaned Python process (PID: $($_.Id))" -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Get-Process node -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*vite*" -or $_.CommandLine -like "*dev*"
} | ForEach-Object {
    Write-Host "Stopping orphaned Node process (PID: $($_.Id))" -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

# Check Python
Write-Host "`nChecking Python..." -ForegroundColor Blue
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
    # Update pythonCmd to use venv python
    $pythonCmd = Join-Path $venvDir "Scripts\python.exe"
}

# Install Python dependencies
Write-Host "`nChecking Python dependencies..." -ForegroundColor Blue
if (Test-Path "requirements.txt") {
    $packagesOk = $false
    try {
        # Use full path to python in venv
        $venvPython = Join-Path $venvDir "Scripts\python.exe"
        if (Test-Path $venvPython) {
            & $venvPython -c "import fastapi, uvicorn, numpy, websockets, pydantic" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $packagesOk = $true
            }
        }
    }
    catch {
        $packagesOk = $false
    }
    
    if (-not $packagesOk) {
        Write-Host "Installing Python packages..." -ForegroundColor Yellow
        & $pythonCmd -m pip install --upgrade pip
        & $pythonCmd -m pip install -r requirements.txt
        & $pythonCmd -m pip install -e .
        
        # Windows sometimes needs these reinstalled/upgraded
        Write-Host "Ensuring Windows-specific dependencies..." -ForegroundColor Yellow
        & $pythonCmd -m pip install --force-reinstall fastapi uvicorn[standard] websockets
        
        # Upgrade to latest versions to fix Windows signal handling issues
        Write-Host "Upgrading to latest versions for Windows compatibility..." -ForegroundColor Yellow
        & $pythonCmd -m pip install -U "uvicorn[standard]" "fastapi[standard]" starlette
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

# Ensure pywin32 is installed on Windows for proper signal handling
if ($IsWindows -or $env:OS -match "Windows") {
    Write-Host "`nChecking Windows-specific dependencies..." -ForegroundColor Blue
    try {
        & $pythonCmd -c "import win32api" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Installing pywin32 for proper Ctrl+C handling..." -ForegroundColor Yellow
            & $pythonCmd -m pip install pywin32
        }
    }
    catch {
        Write-Host "Installing pywin32 for proper Ctrl+C handling..." -ForegroundColor Yellow
        & $pythonCmd -m pip install pywin32
    }
}

# Start servers using the Python launcher
Write-Host "`nStarting servers..." -ForegroundColor Cyan

# Set project root
$projectRoot = Join-Path $PSScriptRoot "..\..\" | Resolve-Path
$env:PYTHONPATH = $projectRoot

# Use the Python launcher for proper signal handling
$launcherPath = Join-Path $PSScriptRoot "recursia-launcher.py"

Write-Host "Launching Recursia with enterprise-grade process management..." -ForegroundColor Green
Write-Host "Ctrl+C will now work properly to stop all servers!" -ForegroundColor Cyan

# Change to project root and run launcher
Push-Location $projectRoot
try {
    # The launcher handles everything including monitoring
    & $pythonCmd $launcherPath
}
finally {
    Pop-Location
    Write-Host "`nRecursia has been shut down." -ForegroundColor Green
}
