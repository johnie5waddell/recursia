# Recursia Status Script for Windows PowerShell
# Shows status of all Recursia servers

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

# Helper function to check process status
function Get-ProcessStatus {
    param($Type, $Port)
    
    $status = @{
        Type = $Type
        Port = $Port
        Status = "Not Running"
        PID = $null
        Details = ""
    }
    
    # Check saved PID first
    $savedPid = Get-SavedPID $Type
    if ($savedPid) {
        $process = Get-Process -Id $savedPid -ErrorAction SilentlyContinue
        if ($process) {
            $status.Status = "Running (from PID file)"
            $status.PID = $savedPid
            $status.Details = "Process: $($process.ProcessName)"
        }
    }
    
    # Also check port
    try {
        $connections = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($connections) {
            $portPid = $connections[0].OwningProcess
            if ($portPid -gt 0) {
                $portProcess = Get-Process -Id $portPid -ErrorAction SilentlyContinue
                if ($portProcess) {
                    if ($savedPid -and $savedPid -eq $portPid) {
                        # Already found via PID file
                    }
                    elseif ($savedPid -and $savedPid -ne $portPid) {
                        $status.Status = "Running (PID mismatch!)"
                        $status.PID = "$savedPid (saved) / $portPid (actual)"
                        $status.Details = "Process: $($portProcess.ProcessName) - PID file out of sync!"
                    }
                    else {
                        $status.Status = "Running (no PID file)"
                        $status.PID = $portPid
                        $status.Details = "Process: $($portProcess.ProcessName)"
                    }
                }
            }
        }
    }
    catch {
        # Error checking port
    }
    
    # Check service health if running
    if ($status.Status -like "Running*") {
        if ($Type -eq "backend") {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:$Port/health" -TimeoutSec 2 -ErrorAction SilentlyContinue
                if ($response.StatusCode -eq 200) {
                    $status.Details += " - Health: OK"
                }
            }
            catch {
                $status.Details += " - Health: Not responding"
            }
        }
        elseif ($Type -eq "frontend") {
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:$Port" -TimeoutSec 2 -ErrorAction SilentlyContinue
                if ($response.StatusCode -in @(200, 304)) {
                    $status.Details += " - Serving"
                }
            }
            catch {
                $status.Details += " - Not responding"
            }
        }
    }
    
    return $status
}

# Check backend status
$backendStatus = Get-ProcessStatus "backend" 8080
$frontendStatus = Get-ProcessStatus "frontend" 5173

# Display status
Write-Host "Backend Server (Port 8080):" -ForegroundColor White
if ($backendStatus.Status -like "*Running*") {
    Write-Host "  Status: $($backendStatus.Status)" -ForegroundColor Green
}
else {
    Write-Host "  Status: $($backendStatus.Status)" -ForegroundColor Red
}
if ($backendStatus.PID) {
    Write-Host "  PID: $($backendStatus.PID)" -ForegroundColor Gray
}
if ($backendStatus.Details) {
    Write-Host "  Details: $($backendStatus.Details)" -ForegroundColor Gray
}

Write-Host "`nFrontend Server (Port 5173):" -ForegroundColor White
if ($frontendStatus.Status -like "*Running*") {
    Write-Host "  Status: $($frontendStatus.Status)" -ForegroundColor Green
}
else {
    Write-Host "  Status: $($frontendStatus.Status)" -ForegroundColor Red
}
if ($frontendStatus.PID) {
    Write-Host "  PID: $($frontendStatus.PID)" -ForegroundColor Gray
}
if ($frontendStatus.Details) {
    Write-Host "  Details: $($frontendStatus.Details)" -ForegroundColor Gray
}

# Check for orphaned processes
Write-Host "`nOrphaned Processes:" -ForegroundColor White

$orphanedPython = Get-Process python, python3 -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*unified_api_server*"
} | Where-Object {
    $_.Id -ne $backendStatus.PID
}

$orphanedNode = Get-Process node -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*vite*" -or $_.CommandLine -like "*dev*"
} | Where-Object {
    $_.Id -ne $frontendStatus.PID
}

if ($orphanedPython -or $orphanedNode) {
    if ($orphanedPython) {
        foreach ($proc in $orphanedPython) {
            Write-Host "  Python process (PID: $($proc.Id)) - possibly from previous run" -ForegroundColor Yellow
        }
    }
    if ($orphanedNode) {
        foreach ($proc in $orphanedNode) {
            Write-Host "  Node process (PID: $($proc.Id)) - possibly from previous run" -ForegroundColor Yellow
        }
    }
    Write-Host "`n  Run .\scripts\deployment\stop-recursia.ps1 to clean up" -ForegroundColor Yellow
}
else {
    Write-Host "  None found" -ForegroundColor Green
}

# Summary
Write-Host "`n==================================================" -ForegroundColor Cyan
if ($backendStatus.Status -like "*Running*" -and $frontendStatus.Status -like "*Running*") {
    Write-Host "Recursia is running!" -ForegroundColor Green
    Write-Host "Frontend: http://localhost:5173" -ForegroundColor Cyan
    Write-Host "Backend:  http://localhost:8080" -ForegroundColor Cyan
}
elseif ($backendStatus.Status -like "*Running*" -or $frontendStatus.Status -like "*Running*") {
    Write-Host "Recursia is partially running" -ForegroundColor Yellow
    Write-Host "Run .\scripts\deployment\stop-recursia.ps1 then .\scripts\deployment\start-recursia.ps1" -ForegroundColor Yellow
}
else {
    Write-Host "Recursia is not running" -ForegroundColor Red
    Write-Host "Run .\scripts\deployment\start-recursia.ps1 to start" -ForegroundColor Yellow
}
Write-Host "==================================================`n" -ForegroundColor Cyan