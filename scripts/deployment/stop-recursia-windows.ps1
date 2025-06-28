# Stop Recursia servers on Windows
Write-Host "`n==================================================" -ForegroundColor Red
Write-Host "     Stopping Recursia Servers" -ForegroundColor Red
Write-Host "==================================================`n" -ForegroundColor Red

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

# Stop backend
$backendPid = Get-SavedPID "backend"
if ($backendPid) {
    Write-Host "Stopping backend (PID: $backendPid)..." -ForegroundColor Yellow
    Stop-Process -Id $backendPid -Force -ErrorAction SilentlyContinue
    Remove-Item "backend.pid" -Force -ErrorAction SilentlyContinue
}

# Stop frontend
$frontendPid = Get-SavedPID "frontend"
if ($frontendPid) {
    Write-Host "Stopping frontend (PID: $frontendPid)..." -ForegroundColor Yellow
    Stop-Process -Id $frontendPid -Force -ErrorAction SilentlyContinue
    Remove-Item "frontend.pid" -Force -ErrorAction SilentlyContinue
}

# Also stop any processes on our ports
Write-Host "`nCleaning up any remaining processes..." -ForegroundColor Blue

# Stop processes on port 8080
$port8080 = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
if ($port8080) {
    $port8080 | ForEach-Object {
        if ($_.OwningProcess -gt 0) {
            Write-Host "Stopping process on port 8080 (PID: $($_.OwningProcess))..." -ForegroundColor Yellow
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

# Stop processes on port 5173
$port5173 = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue
if ($port5173) {
    $port5173 | ForEach-Object {
        if ($_.OwningProcess -gt 0) {
            Write-Host "Stopping process on port 5173 (PID: $($_.OwningProcess))..." -ForegroundColor Yellow
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

# Clean up Python/Node processes
Get-Process python, python3 -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*unified_api_server*" -or $_.CommandLine -like "*uvicorn*"
} | ForEach-Object {
    Write-Host "Stopping Python process (PID: $($_.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Get-Process node -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*vite*" -or $_.CommandLine -like "*dev*"
} | ForEach-Object {
    Write-Host "Stopping Node process (PID: $($_.Id))..." -ForegroundColor Yellow
    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
}

Write-Host "`nAll Recursia servers stopped!" -ForegroundColor Green
Write-Host "You can now run start-recursia.ps1 again." -ForegroundColor Cyan