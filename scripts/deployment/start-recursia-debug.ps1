# Recursia Startup Script for Windows PowerShell - DEBUG VERSION
# This version includes extensive error checking and logging

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "     Recursia Quantum OSH Computing Platform" -ForegroundColor Cyan
Write-Host "     DEBUG VERSION - Verbose Logging Enabled" -ForegroundColor Yellow
Write-Host "==================================================`n" -ForegroundColor Cyan

# Create a log file
$logFile = "recursia-startup-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
function Write-Log {
    param($Message, $Color = "White")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$timestamp - $Message" | Out-File -Append $logFile
    Write-Host $Message -ForegroundColor $Color
}

Write-Log "Starting Recursia startup script..."

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
    Write-Log "Error: Not in Recursia root directory!" "Red"
    Write-Log "Current directory: $(Get-Location)" "Yellow"
    exit 1
}

# Check Python
Write-Log "Checking Python..." "Blue"
$pythonCmd = $null

if (Test-Command python) {
    $version = & python --version 2>&1
    if ($version -match "Python 3\.([89]|1[0-9])") {
        $pythonCmd = "python"
        Write-Log "Found: $version" "Green"
    }
}

if (-not $pythonCmd) {
    Write-Log "Python 3.8+ not found! Please install Python from python.org" "Red"
    exit 1
}

# Check Node.js
Write-Log "`nChecking Node.js..." "Blue"
if (Test-Command node) {
    $nodeVersion = & node --version 2>&1
    Write-Log "Found Node.js: $nodeVersion" "Green"
}
else {
    Write-Log "Node.js not found! Please install from nodejs.org" "Red"
    exit 1
}

# Check if ports are already in use
Write-Log "`nChecking port availability..." "Blue"
$backendPort = 8080
$frontendPort = 5173

function Test-Port {
    param($Port)
    try {
        $tcpConnection = New-Object System.Net.Sockets.TcpClient
        $tcpConnection.Connect("127.0.0.1", $Port)
        $tcpConnection.Close()
        return $true  # Port is in use
    }
    catch {
        return $false  # Port is available
    }
}

if (Test-Port $backendPort) {
    Write-Log "WARNING: Port $backendPort is already in use!" "Yellow"
    Write-Log "Attempting to find what's using it..." "Yellow"
    
    try {
        $process = Get-NetTCPConnection -LocalPort $backendPort -State Listen -ErrorAction SilentlyContinue
        if ($process) {
            $pid = $process.OwningProcess
            $processName = (Get-Process -Id $pid).ProcessName
            Write-Log "Port $backendPort is used by: $processName (PID: $pid)" "Yellow"
            
            $response = Read-Host "Kill the existing process? (y/n)"
            if ($response -eq 'y') {
                Stop-Process -Id $pid -Force
                Write-Log "Killed process $pid" "Green"
                Start-Sleep -Seconds 2
            }
            else {
                Write-Log "Cannot continue with port $backendPort in use" "Red"
                exit 1
            }
        }
    }
    catch {
        Write-Log "Could not determine what's using port $backendPort" "Yellow"
    }
}

if (Test-Port $frontendPort) {
    Write-Log "WARNING: Port $frontendPort is already in use!" "Yellow"
    # Similar check for frontend port...
}

# Set up virtual environment
Write-Log "`nSetting up Python environment..." "Blue"
$venvDir = "venv"

if (-not (Test-Path $venvDir)) {
    Write-Log "Creating virtual environment..." "Yellow"
    & $pythonCmd -m venv $venvDir
}

# Activate virtual environment
$activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
if (Test-Path $activateScript) {
    & $activateScript
    Write-Log "Virtual environment activated" "Green"
}

# Install Python dependencies
Write-Log "`nChecking Python dependencies..." "Blue"
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
        Write-Log "Installing Python packages..." "Yellow"
        & $pythonCmd -m pip install --upgrade pip
        & $pythonCmd -m pip install -r requirements.txt
        & $pythonCmd -m pip install -e .
    }
    else {
        Write-Log "Python packages already installed" "Green"
    }
}

# Install frontend dependencies
Write-Log "`nSetting up frontend..." "Blue"
Push-Location frontend
if (-not (Test-Path "node_modules")) {
    Write-Log "Installing frontend packages..." "Yellow"
    npm install
}
else {
    Write-Log "Frontend packages already installed" "Green"
}
Pop-Location

# Check ML models
Write-Log "`nChecking ML models..." "Blue"
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
    Write-Log "Missing $missingModels ML models. Training..." "Yellow"
    Write-Log "This will take 5-10 minutes on first run." "Yellow"
    
    # Check PyTorch
    try {
        & $pythonCmd -c "import torch" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Log "Installing PyTorch..." "Yellow"
            & $pythonCmd -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        }
    }
    catch {
        Write-Log "Installing PyTorch..." "Yellow"
        & $pythonCmd -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
    
    # Train models
    if (Test-Path "scripts\validation\train_ml_decoders.py") {
        & $pythonCmd scripts\validation\train_ml_decoders.py
    }
}
else {
    Write-Log "All ML models present" "Green"
}

# Create backend startup script with error handling
$backendScript = @"
import sys
import traceback

try:
    print("Starting backend server...", flush=True)
    sys.path.insert(0, '.')
    from src.api.unified_api_server import app
    import uvicorn
    
    print(f"Python path: {sys.path}", flush=True)
    print("Starting uvicorn...", flush=True)
    
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
except Exception as e:
    print(f"Backend error: {str(e)}", flush=True)
    print(traceback.format_exc(), flush=True)
    sys.exit(1)
"@

$backendScript | Out-File -FilePath "temp_backend_start.py" -Encoding UTF8

# Start servers with better error handling
Write-Log "`nStarting servers..." "Cyan"

# Start backend with output redirection
Write-Log "Starting backend server on port 8080..." "Blue"
$backendOutput = "backend-output.log"
$backendError = "backend-error.log"

$backendInfo = New-Object System.Diagnostics.ProcessStartInfo
$backendInfo.FileName = $pythonCmd
$backendInfo.Arguments = "temp_backend_start.py"
$backendInfo.UseShellExecute = $false
$backendInfo.RedirectStandardOutput = $true
$backendInfo.RedirectStandardError = $true
$backendInfo.CreateNoWindow = $true

$backend = New-Object System.Diagnostics.Process
$backend.StartInfo = $backendInfo

# Set up event handlers for output
$outputHandler = {
    if ($EventArgs.Data) {
        Add-Content -Path $backendOutput -Value $EventArgs.Data
    }
}
$errorHandler = {
    if ($EventArgs.Data) {
        Add-Content -Path $backendError -Value $EventArgs.Data
        Write-Log "Backend Error: $($EventArgs.Data)" "Red"
    }
}

Register-ObjectEvent -InputObject $backend -EventName OutputDataReceived -Action $outputHandler | Out-Null
Register-ObjectEvent -InputObject $backend -EventName ErrorDataReceived -Action $errorHandler | Out-Null

$backend.Start() | Out-Null
$backend.BeginOutputReadLine()
$backend.BeginErrorReadLine()

Write-Log "Backend process started with PID: $($backend.Id)" "Green"

# Wait a moment for backend to start
Write-Log "Waiting for backend to initialize..." "Yellow"
Start-Sleep -Seconds 5

# Check if backend is still running
if ($backend.HasExited) {
    Write-Log "Backend crashed immediately!" "Red"
    Write-Log "Exit code: $($backend.ExitCode)" "Red"
    
    if (Test-Path $backendError) {
        Write-Log "`nBackend error log:" "Red"
        Get-Content $backendError | ForEach-Object { Write-Log $_ "Red" }
    }
    
    if (Test-Path $backendOutput) {
        Write-Log "`nBackend output log:" "Yellow"
        Get-Content $backendOutput | ForEach-Object { Write-Log $_ "Yellow" }
    }
    
    exit 1
}

# Test backend health
Write-Log "Testing backend health..." "Blue"
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 5
    if ($response.StatusCode -eq 200) {
        Write-Log "Backend is healthy!" "Green"
    }
}
catch {
    Write-Log "Backend health check failed: $_" "Yellow"
}

# Start frontend with better error handling
Write-Log "Starting frontend server on port 5173..." "Blue"
Push-Location frontend

# Create a frontend startup script
$frontendScript = @"
console.log('Starting frontend development server...');
const { spawn } = require('child_process');
const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';

const child = spawn(npm, ['run', 'dev'], {
    stdio: 'pipe',
    shell: true
});

child.stdout.on('data', (data) => {
    console.log(`Frontend: ${data}`);
});

child.stderr.on('data', (data) => {
    console.error(`Frontend Error: ${data}`);
});

child.on('close', (code) => {
    console.log(`Frontend process exited with code ${code}`);
    process.exit(code);
});
"@

$frontendScript | Out-File -FilePath "temp_frontend_start.js" -Encoding UTF8

$frontendOutput = "..\frontend-output.log"
$frontendError = "..\frontend-error.log"

$frontendInfo = New-Object System.Diagnostics.ProcessStartInfo
$frontendInfo.FileName = "node"
$frontendInfo.Arguments = "temp_frontend_start.js"
$frontendInfo.UseShellExecute = $false
$frontendInfo.RedirectStandardOutput = $true
$frontendInfo.RedirectStandardError = $true
$frontendInfo.CreateNoWindow = $true
$frontendInfo.WorkingDirectory = (Get-Location).Path

$frontend = New-Object System.Diagnostics.Process
$frontend.StartInfo = $frontendInfo

# Set up event handlers for frontend output
$frontendOutputHandler = {
    if ($EventArgs.Data) {
        Add-Content -Path $frontendOutput -Value $EventArgs.Data
    }
}
$frontendErrorHandler = {
    if ($EventArgs.Data) {
        Add-Content -Path $frontendError -Value $EventArgs.Data
        Write-Log "Frontend Error: $($EventArgs.Data)" "Red"
    }
}

Register-ObjectEvent -InputObject $frontend -EventName OutputDataReceived -Action $frontendOutputHandler | Out-Null
Register-ObjectEvent -InputObject $frontend -EventName ErrorDataReceived -Action $frontendErrorHandler | Out-Null

$frontend.Start() | Out-Null
$frontend.BeginOutputReadLine()
$frontend.BeginErrorReadLine()

Write-Log "Frontend process started with PID: $($frontend.Id)" "Green"

Pop-Location

# Wait and check frontend
Write-Log "Waiting for frontend to initialize..." "Yellow"
Start-Sleep -Seconds 10

# Check if frontend is still running
if ($frontend.HasExited) {
    Write-Log "Frontend crashed immediately!" "Red"
    Write-Log "Exit code: $($frontend.ExitCode)" "Red"
    
    if (Test-Path $frontendError) {
        Write-Log "`nFrontend error log:" "Red"
        Get-Content $frontendError | ForEach-Object { Write-Log $_ "Red" }
    }
    
    if (Test-Path $frontendOutput) {
        Write-Log "`nFrontend output log:" "Yellow"
        Get-Content $frontendOutput | ForEach-Object { Write-Log $_ "Yellow" }
    }
    
    # Don't exit, backend might still be running
}

# Open browser
if (-not $frontend.HasExited) {
    Write-Log "`nOpening browser..." "Green"
    Start-Process "http://localhost:5173"
}

Write-Log "`n==================================================" "Green"
Write-Log "Recursia startup complete!" "Green"
Write-Log "Frontend: http://localhost:5173" "Cyan"
Write-Log "Backend:  http://localhost:8080" "Cyan"
Write-Log "`nPress Ctrl+C to stop all servers" "Yellow"
Write-Log "Check log file: $logFile" "Yellow"
Write-Log "==================================================" "Green"

# Monitor servers with better error handling
$checkInterval = 2  # seconds
$errorCount = 0
$maxErrors = 5

try {
    while ($true) {
        Start-Sleep -Seconds $checkInterval
        
        $backendRunning = !$backend.HasExited
        $frontendRunning = !$frontend.HasExited
        
        if (-not $backendRunning) {
            Write-Log "`nBackend server stopped! Exit code: $($backend.ExitCode)" "Red"
            $errorCount++
            
            # Check logs
            if (Test-Path $backendError) {
                Write-Log "Last backend errors:" "Red"
                Get-Content $backendError -Tail 10 | ForEach-Object { Write-Log $_ "Red" }
            }
        }
        
        if (-not $frontendRunning) {
            Write-Log "`nFrontend server stopped! Exit code: $($frontend.ExitCode)" "Red"
            $errorCount++
            
            # Check logs
            if (Test-Path $frontendError) {
                Write-Log "Last frontend errors:" "Red"
                Get-Content $frontendError -Tail 10 | ForEach-Object { Write-Log $_ "Red" }
            }
        }
        
        if ($errorCount -ge $maxErrors -or (-not $backendRunning -and -not $frontendRunning)) {
            Write-Log "`nToo many errors or both servers stopped. Shutting down..." "Red"
            break
        }
        
        # Restart servers if needed
        if (-not $backendRunning -and $errorCount -lt $maxErrors) {
            Write-Log "Attempting to restart backend..." "Yellow"
            # Restart logic here if desired
        }
    }
}
finally {
    Write-Log "`nShutting down..." "Yellow"
    
    # Clean up processes
    if (-not $backend.HasExited) {
        Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue
    }
    if (-not $frontend.HasExited) {
        Stop-Process -Id $frontend.Id -Force -ErrorAction SilentlyContinue
    }
    
    # Clean up temp files
    if (Test-Path "temp_backend_start.py") {
        Remove-Item "temp_backend_start.py"
    }
    if (Test-Path "frontend\temp_frontend_start.js") {
        Remove-Item "frontend\temp_frontend_start.js"
    }
    
    Write-Log "Shutdown complete." "Green"
    Write-Log "Full log saved to: $logFile" "Yellow"
}