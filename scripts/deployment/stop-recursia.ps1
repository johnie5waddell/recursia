# Recursia Stop Script for Windows PowerShell
# Stops all Recursia servers using saved PIDs

Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "     Stopping Recursia Servers" -ForegroundColor Cyan
Write-Host "==================================================`n" -ForegroundColor Cyan

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

# Helper function to stop saved process
function Stop-SavedProcess {
    param($Type)
    $pid = Get-SavedPID $Type
    if ($pid) {
        $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Stopping $Type process (PID: $pid)" -ForegroundColor Yellow
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
            Write-Host "$Type stopped successfully" -ForegroundColor Green
        }
        else {
            Write-Host "$Type process (PID: $pid) not found - may have already stopped" -ForegroundColor DarkGray
        }
    }
    else {
        Write-Host "No saved PID found for $Type" -ForegroundColor DarkGray
    }
    
    # Remove PID file
    $pidFile = "$Type.pid"
    if (Test-Path $pidFile) {
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
}

# Helper function to kill processes by port
function Stop-ProcessByPort {
    param($Port, $Type)
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $pids = $connections.OwningProcess | Sort-Object -Unique
            foreach ($pid in $pids) {
                if ($pid -gt 0) {
                    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($process) {
                        Write-Host "Stopping $Type on port $Port (PID: $pid)" -ForegroundColor Yellow
                        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
                    }
                }
            }
        }
    }
    catch {
        # Port is free or error accessing
    }
}

# Stop saved processes first
Write-Host "Stopping saved Recursia processes..." -ForegroundColor Blue
Stop-SavedProcess "backend"
Stop-SavedProcess "frontend"

# Also check ports in case PIDs were lost
Write-Host "`nChecking for processes on Recursia ports..." -ForegroundColor Blue
Stop-ProcessByPort 8080 "backend"
Stop-ProcessByPort 5173 "frontend"

# Clean up any orphaned Python/Node processes
Write-Host "`nCleaning up orphaned processes..." -ForegroundColor Blue

$pythonProcesses = Get-Process python, python3 -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*unified_api_server*"
}

if ($pythonProcesses) {
    foreach ($proc in $pythonProcesses) {
        Write-Host "Stopping orphaned Python process (PID: $($proc.Id))" -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

$nodeProcesses = Get-Process node -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*vite*" -or $_.CommandLine -like "*dev*"
}

if ($nodeProcesses) {
    foreach ($proc in $nodeProcesses) {
        Write-Host "Stopping orphaned Node process (PID: $($proc.Id))" -ForegroundColor Yellow
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

# Clean up PID files if they still exist
Write-Host "`nCleaning up PID files..." -ForegroundColor Blue
Remove-Item "backend.pid" -Force -ErrorAction SilentlyContinue
Remove-Item "frontend.pid" -Force -ErrorAction SilentlyContinue

Write-Host "`n==================================================" -ForegroundColor Green
Write-Host "All Recursia servers stopped" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green