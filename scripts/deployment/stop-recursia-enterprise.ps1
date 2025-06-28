# Recursia Enterprise Stop Script for Windows PowerShell
# Production-quality script for gracefully stopping all Recursia services

#Requires -Version 5.1
#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [Parameter()]
    [switch]$Force,
    
    [Parameter()]
    [int]$GracefulTimeout = 10,
    
    [Parameter()]
    [switch]$CleanLogs
)

# Script configuration
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Global variables
$script:LogDirectory = Join-Path $PSScriptRoot "logs"
$script:PidDirectory = Join-Path $PSScriptRoot "pids"
$script:LogFile = Join-Path $LogDirectory "recursia-stop-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"

#region Logging Functions
function Initialize-Logging {
    if (-not (Test-Path $LogDirectory)) {
        New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    }
    
    "Recursia Stop Script Log - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $LogFile
    "Force Mode: $Force" | Add-Content $LogFile
    "=" * 80 | Add-Content $LogFile
}

function Write-Log {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Message,
        
        [Parameter()]
        [ValidateSet('Info', 'Warning', 'Error', 'Success', 'Debug')]
        [string]$Level = 'Info'
    )
    
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Write to log file
    $logEntry | Add-Content $LogFile
    
    # Write to console
    $color = switch ($Level) {
        'Info'    { 'White' }
        'Warning' { 'Yellow' }
        'Error'   { 'Red' }
        'Success' { 'Green' }
        'Debug'   { 'Gray' }
    }
    Write-Host $Message -ForegroundColor $color
}
#endregion

#region Process Discovery Functions
function Get-SavedProcessInfo {
    [CmdletBinding()]
    param()
    
    $processes = @()
    
    if (Test-Path $PidDirectory) {
        Get-ChildItem -Path $PidDirectory -Filter "*.pid" | ForEach-Object {
            $serviceName = $_.BaseName
            $pidContent = Get-Content $_.FullName -ErrorAction SilentlyContinue
            
            if ($pidContent) {
                try {
                    $pid = [int]$pidContent
                    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    
                    if ($process) {
                        $processes += @{
                            Name = $serviceName
                            ProcessId = $pid
                            Process = $process
                            Source = 'PidFile'
                        }
                        Write-Log "Found $serviceName process from PID file: $pid" -Level Debug
                    }
                    else {
                        Write-Log "PID file for $serviceName references non-existent process: $pid" -Level Warning
                        # Clean up stale PID file
                        Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue
                    }
                }
                catch {
                    Write-Log "Invalid PID file for $serviceName : $_" -Level Warning
                }
            }
        }
    }
    
    return $processes
}

function Get-ProcessesByPort {
    [CmdletBinding()]
    param()
    
    $processes = @()
    $ports = @(8080, 5173)  # Default Recursia ports
    
    foreach ($port in $ports) {
        try {
            # Method 1: Get-NetTCPConnection
            $connections = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
            
            foreach ($conn in $connections) {
                if ($conn.OwningProcess -gt 0) {
                    $process = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                    if ($process) {
                        $serviceName = if ($port -eq 8080) { 'backend' } else { 'frontend' }
                        
                        $processes += @{
                            Name = $serviceName
                            ProcessId = $conn.OwningProcess
                            Process = $process
                            Port = $port
                            Source = 'Port'
                        }
                        Write-Log "Found $serviceName process on port $port : PID $($conn.OwningProcess)" -Level Debug
                    }
                }
            }
        }
        catch {
            Write-Log "Error checking port $port : $_" -Level Warning
            
            # Fallback to netstat
            try {
                $netstatOutput = & netstat -ano | Select-String ":$port\s+.*LISTENING\s+(\d+)$"
                if ($netstatOutput) {
                    $pid = [int]$netstatOutput.Matches[0].Groups[1].Value
                    $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                    if ($process) {
                        $serviceName = if ($port -eq 8080) { 'backend' } else { 'frontend' }
                        
                        $processes += @{
                            Name = $serviceName
                            ProcessId = $pid
                            Process = $process
                            Port = $port
                            Source = 'Netstat'
                        }
                        Write-Log "Found $serviceName process on port $port via netstat: PID $pid" -Level Debug
                    }
                }
            }
            catch {
                Write-Log "Netstat fallback failed for port $port : $_" -Level Warning
            }
        }
    }
    
    return $processes
}

function Get-OrphanedProcesses {
    [CmdletBinding()]
    param(
        [Parameter()]
        [array]$ExcludePids = @()
    )
    
    $orphaned = @()
    
    # Check for Python processes running Recursia
    Get-Process python, python3 -ErrorAction SilentlyContinue | Where-Object {
        $_.Id -notin $ExcludePids
    } | ForEach-Object {
        try {
            # Get command line using WMI
            $wmi = Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue
            $commandLine = $wmi.CommandLine
            
            if ($commandLine -match "uvicorn|unified_api_server|recursia") {
                $orphaned += @{
                    Name = 'python-orphaned'
                    ProcessId = $_.Id
                    Process = $_
                    CommandLine = $commandLine
                    Source = 'Orphaned'
                }
                Write-Log "Found orphaned Python process: PID $($_.Id)" -Level Debug
            }
        }
        catch {
            # Check process name as fallback
            if ($_.ProcessName -match "python") {
                $orphaned += @{
                    Name = 'python-possible'
                    ProcessId = $_.Id
                    Process = $_
                    Source = 'Orphaned'
                }
            }
        }
    }
    
    # Check for Node processes
    Get-Process node -ErrorAction SilentlyContinue | Where-Object {
        $_.Id -notin $ExcludePids
    } | ForEach-Object {
        try {
            $wmi = Get-WmiObject Win32_Process -Filter "ProcessId = $($_.Id)" -ErrorAction SilentlyContinue
            $commandLine = $wmi.CommandLine
            
            if ($commandLine -match "vite|dev|recursia") {
                $orphaned += @{
                    Name = 'node-orphaned'
                    ProcessId = $_.Id
                    Process = $_
                    CommandLine = $commandLine
                    Source = 'Orphaned'
                }
                Write-Log "Found orphaned Node process: PID $($_.Id)" -Level Debug
            }
        }
        catch {
            # Add as possible orphan
            $orphaned += @{
                Name = 'node-possible'
                ProcessId = $_.Id
                Process = $_
                Source = 'Orphaned'
            }
        }
    }
    
    return $orphaned
}
#endregion

#region Process Termination Functions
function Stop-ProcessGracefully {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable]$ProcessInfo,
        
        [Parameter()]
        [int]$Timeout = 10
    )
    
    $process = $ProcessInfo.Process
    $name = $ProcessInfo.Name
    $pid = $ProcessInfo.ProcessId
    
    Write-Log "Attempting graceful shutdown of $name (PID: $pid)..." -Level Info
    
    try {
        # Try to close main window first (works for GUI apps)
        if ($process.MainWindowHandle -ne [IntPtr]::Zero) {
            $process.CloseMainWindow() | Out-Null
            $stopped = $process.WaitForExit($Timeout * 1000)
            
            if ($stopped) {
                Write-Log "$name stopped gracefully via window close" -Level Success
                return $true
            }
        }
        
        # Send SIGTERM equivalent (graceful termination)
        # For console apps, we'll use GenerateConsoleCtrlEvent if possible
        if ($name -match "backend|python") {
            # For Python/uvicorn, send Ctrl+C signal
            try {
                # This requires the process to have a console
                $result = [System.Diagnostics.Process]::GetProcessById($pid).CloseMainWindow()
                Start-Sleep -Seconds 2
                
                if (-not (Get-Process -Id $pid -ErrorAction SilentlyContinue)) {
                    Write-Log "$name stopped gracefully" -Level Success
                    return $true
                }
            }
            catch {
                Write-Log "Could not send graceful signal to $name : $_" -Level Debug
            }
        }
        
        # Final attempt - use Stop-Process with confirmation
        if (-not $Force) {
            $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
            Write-Host "Waiting for $name to stop gracefully..." -NoNewline
            
            while ($stopwatch.Elapsed.TotalSeconds -lt $Timeout) {
                if (-not (Get-Process -Id $pid -ErrorAction SilentlyContinue)) {
                    Write-Host " Done!" -ForegroundColor Green
                    Write-Log "$name stopped gracefully" -Level Success
                    return $true
                }
                Write-Host "." -NoNewline
                Start-Sleep -Milliseconds 500
            }
            Write-Host " Timeout!" -ForegroundColor Yellow
        }
        
        return $false
    }
    catch {
        Write-Log "Error during graceful shutdown of $name : $_" -Level Warning
        return $false
    }
}

function Stop-ProcessForcefully {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable]$ProcessInfo
    )
    
    $name = $ProcessInfo.Name
    $pid = $ProcessInfo.ProcessId
    
    Write-Log "Force stopping $name (PID: $pid)..." -Level Warning
    
    try {
        Stop-Process -Id $pid -Force -ErrorAction Stop
        
        # Wait briefly to ensure process is terminated
        $maxWait = 50  # 5 seconds
        $waited = 0
        
        while ((Get-Process -Id $pid -ErrorAction SilentlyContinue) -and $waited -lt $maxWait) {
            Start-Sleep -Milliseconds 100
            $waited++
        }
        
        if (-not (Get-Process -Id $pid -ErrorAction SilentlyContinue)) {
            Write-Log "$name force stopped successfully" -Level Success
            return $true
        }
        else {
            Write-Log "$name could not be stopped!" -Level Error
            return $false
        }
    }
    catch {
        if ($_.Exception.Message -match "Cannot find a process") {
            Write-Log "$name was already stopped" -Level Info
            return $true
        }
        else {
            Write-Log "Error force stopping $name : $_" -Level Error
            return $false
        }
    }
}
#endregion

#region Cleanup Functions
function Remove-PidFiles {
    [CmdletBinding()]
    param()
    
    if (Test-Path $PidDirectory) {
        Write-Log "Cleaning up PID files..." -Level Info
        
        Get-ChildItem -Path $PidDirectory -Filter "*.pid" | ForEach-Object {
            try {
                Remove-Item $_.FullName -Force
                Write-Log "Removed PID file: $($_.Name)" -Level Debug
            }
            catch {
                Write-Log "Could not remove PID file $($_.Name): $_" -Level Warning
            }
        }
    }
}

function Clean-Logs {
    [CmdletBinding()]
    param()
    
    Write-Log "Cleaning old log files..." -Level Info
    
    # Keep logs from last 7 days
    $cutoffDate = (Get-Date).AddDays(-7)
    
    Get-ChildItem -Path $LogDirectory -Filter "recursia-*.log" | Where-Object {
        $_.LastWriteTime -lt $cutoffDate
    } | ForEach-Object {
        try {
            Remove-Item $_.FullName -Force
            Write-Log "Removed old log file: $($_.Name)" -Level Debug
        }
        catch {
            Write-Log "Could not remove log file $($_.Name): $_" -Level Warning
        }
    }
}
#endregion

#region Main Function
function Stop-RecursiaServices {
    [CmdletBinding()]
    param()
    
    try {
        Initialize-Logging
        
        Write-Host "`n==================================================" -ForegroundColor Cyan
        Write-Host "     Stopping Recursia Services" -ForegroundColor Cyan
        Write-Host "==================================================`n" -ForegroundColor Cyan
        
        # Collect all Recursia processes
        Write-Log "Discovering Recursia processes..." -Level Info
        
        $savedProcesses = Get-SavedProcessInfo
        $portProcesses = Get-ProcessesByPort
        
        # Merge process lists, avoiding duplicates
        $allProcesses = @{}
        
        foreach ($proc in ($savedProcesses + $portProcesses)) {
            $key = $proc.ProcessId
            if (-not $allProcesses.ContainsKey($key)) {
                $allProcesses[$key] = $proc
            }
        }
        
        # Get orphaned processes
        $knownPids = $allProcesses.Keys
        $orphanedProcesses = Get-OrphanedProcesses -ExcludePids $knownPids
        
        # Display summary
        $totalProcesses = $allProcesses.Count + $orphanedProcesses.Count
        
        if ($totalProcesses -eq 0) {
            Write-Log "No Recursia processes found running" -Level Success
            Remove-PidFiles
            return
        }
        
        Write-Log "Found $totalProcesses Recursia process(es)" -Level Info
        
        # Stop main processes
        foreach ($proc in $allProcesses.Values) {
            Write-Log "Processing $($proc.Name) (PID: $($proc.ProcessId), Source: $($proc.Source))" -Level Info
            
            if (-not $Force) {
                $stopped = Stop-ProcessGracefully -ProcessInfo $proc -Timeout $GracefulTimeout
                
                if (-not $stopped) {
                    Write-Log "Graceful shutdown failed, using force stop..." -Level Warning
                    Stop-ProcessForcefully -ProcessInfo $proc | Out-Null
                }
            }
            else {
                Stop-ProcessForcefully -ProcessInfo $proc | Out-Null
            }
        }
        
        # Handle orphaned processes
        if ($orphanedProcesses.Count -gt 0) {
            Write-Log "`nHandling $($orphanedProcesses.Count) orphaned process(es)..." -Level Warning
            
            foreach ($proc in $orphanedProcesses) {
                if ($proc.CommandLine) {
                    Write-Log "Orphaned: $($proc.CommandLine)" -Level Debug
                }
                
                if ($Force -or $proc.Name -notmatch "possible") {
                    Stop-ProcessForcefully -ProcessInfo $proc | Out-Null
                }
                else {
                    $response = Read-Host "Stop possible orphaned $($proc.Name) process (PID: $($proc.ProcessId))? (y/n)"
                    if ($response -eq 'y') {
                        Stop-ProcessForcefully -ProcessInfo $proc | Out-Null
                    }
                }
            }
        }
        
        # Clean up
        Remove-PidFiles
        
        if ($CleanLogs) {
            Clean-Logs
        }
        
        # Final verification
        Write-Log "`nVerifying all processes stopped..." -Level Info
        
        $remainingBackend = Get-ProcessesByPort | Where-Object { $_.Port -eq 8080 }
        $remainingFrontend = Get-ProcessesByPort | Where-Object { $_.Port -eq 5173 }
        
        if ($remainingBackend -or $remainingFrontend) {
            Write-Log "WARNING: Some processes may still be running" -Level Warning
            
            if ($remainingBackend) {
                Write-Log "Port 8080 still in use" -Level Warning
            }
            if ($remainingFrontend) {
                Write-Log "Port 5173 still in use" -Level Warning
            }
        }
        else {
            Write-Log "All Recursia services stopped successfully" -Level Success
        }
        
        Write-Host "`n==================================================" -ForegroundColor Green
        Write-Host "Recursia services stopped" -ForegroundColor Green
        Write-Host "Log file: $LogFile" -ForegroundColor Gray
        Write-Host "==================================================" -ForegroundColor Green
    }
    catch {
        Write-Log "Fatal error: $_" -Level Error
        Write-Log $_.ScriptStackTrace -Level Error
        throw
    }
}
#endregion

# Main entry point
if ($MyInvocation.InvocationName -ne '.') {
    Stop-RecursiaServices
}