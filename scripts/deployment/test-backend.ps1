# Simple backend test script
Write-Host "Testing backend startup directly..." -ForegroundColor Cyan

# Find Python in venv
$venvPython = "venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Virtual environment not found!" -ForegroundColor Red
    exit 1
}

Write-Host "Using Python: $venvPython" -ForegroundColor Green

# Set PYTHONPATH
$env:PYTHONPATH = Get-Location

# Check if pywin32 is installed for proper Ctrl+C
Write-Host "`nChecking Windows signal handling..." -ForegroundColor Blue
try {
    & $venvPython -c "import win32api" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing pywin32 for proper Ctrl+C handling..." -ForegroundColor Yellow
        & $venvPython -m pip install pywin32
    }
}
catch {
    Write-Host "Installing pywin32 for proper Ctrl+C handling..." -ForegroundColor Yellow
    & $venvPython -m pip install pywin32
}

# Upgrade to latest versions to fix Windows signal handling issues
Write-Host "`nUpgrading packages for Windows compatibility..." -ForegroundColor Blue
& $venvPython -m pip install -U "uvicorn[standard]" "fastapi[standard]" starlette

Write-Host "`nTesting imports..." -ForegroundColor Blue
& $venvPython -c "import sys; print('Python path:'); [print(f'  {p}') for p in sys.path]"

# First check if port is in use
Write-Host "`nChecking if port 8080 is available..." -ForegroundColor Blue
$portInUse = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($portInUse) {
    Write-Host "WARNING: Port 8080 is already in use!" -ForegroundColor Red
    Write-Host "Attempting to stop the process using it..." -ForegroundColor Yellow
    $portInUse | ForEach-Object {
        Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

# Use the launcher for proper signal handling
$launcherPath = Join-Path $PSScriptRoot "recursia-launcher.py"
if (Test-Path $launcherPath) {
    Write-Host "`nUsing Python launcher for proper Ctrl+C handling..." -ForegroundColor Green
    Write-Host "Starting backend server with frontend..." -ForegroundColor Blue
    & $venvPython $launcherPath
}
else {
    # Fallback to direct execution if launcher doesn't exist
    Write-Host "`nStarting backend server directly..." -ForegroundColor Blue
    Write-Host "Server is running at http://localhost:8080" -ForegroundColor Green
    Write-Host "Press Ctrl+C to stop (or close window if Ctrl+C doesn't work)" -ForegroundColor Yellow
    
    # Run the server directly
    & $venvPython src/api/unified_api_server.py
}