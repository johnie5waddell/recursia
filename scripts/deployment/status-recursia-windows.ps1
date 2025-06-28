# Check Recursia server status on Windows
Write-Host "`n==================================================" -ForegroundColor Cyan
Write-Host "     Recursia Server Status" -ForegroundColor Cyan
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

# Check backend
Write-Host "Backend Server:" -ForegroundColor Blue
$backendPid = Get-SavedPID "backend"
if ($backendPid) {
    $process = Get-Process -Id $backendPid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "  ✓ Running (PID: $backendPid)" -ForegroundColor Green
        
        # Test API endpoint
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8080/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($response.StatusCode -eq 200) {
                Write-Host "  ✓ API responding on port 8080" -ForegroundColor Green
            }
        }
        catch {
            Write-Host "  ✗ API not responding" -ForegroundColor Red
        }
    }
    else {
        Write-Host "  ✗ Process not found (PID: $backendPid)" -ForegroundColor Red
    }
}
else {
    Write-Host "  ✗ Not running (no PID file)" -ForegroundColor Red
}

# Check frontend
Write-Host "`nFrontend Server:" -ForegroundColor Blue
$frontendPid = Get-SavedPID "frontend"
if ($frontendPid) {
    $process = Get-Process -Id $frontendPid -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "  ✓ Running (PID: $frontendPid)" -ForegroundColor Green
        Write-Host "  ✓ UI available at http://localhost:5173" -ForegroundColor Green
    }
    else {
        Write-Host "  ✗ Process not found (PID: $frontendPid)" -ForegroundColor Red
    }
}
else {
    Write-Host "  ✗ Not running (no PID file)" -ForegroundColor Red
}

# Check ports
Write-Host "`nPort Status:" -ForegroundColor Blue
$port8080 = Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue
$port5173 = Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue

if ($port8080) {
    Write-Host "  ✓ Port 8080 (Backend) is in use" -ForegroundColor Green
}
else {
    Write-Host "  ✗ Port 8080 (Backend) is free" -ForegroundColor Yellow
}

if ($port5173) {
    Write-Host "  ✓ Port 5173 (Frontend) is in use" -ForegroundColor Green
}
else {
    Write-Host "  ✗ Port 5173 (Frontend) is free" -ForegroundColor Yellow
}

Write-Host "`n==================================================" -ForegroundColor Cyan