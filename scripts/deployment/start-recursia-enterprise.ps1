# Recursia Enterprise Startup Script for Windows PowerShell
# Production-quality script with comprehensive error handling and recovery

#Requires -Version 5.1
#Requires -RunAsAdministrator

[CmdletBinding()]
param(
    [Parameter()]
    [ValidateSet('Development', 'Production', 'Debug')]
    [string]$Environment = 'Development',
    
    [Parameter()]
    [int]$BackendPort = 8080,
    
    [Parameter()]
    [int]$FrontendPort = 5173,
    
    [Parameter()]
    [switch]$SkipDependencyCheck,
    
    [Parameter()]
    [switch]$AutoRestart,
    
    [Parameter()]
    [int]$MaxRestartAttempts = 3
)

# Script configuration
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

# Global variables
$script:LogDirectory = Join-Path $PSScriptRoot "logs"
$script:PidDirectory = Join-Path $PSScriptRoot "pids"
$script:LogFile = Join-Path $LogDirectory "recursia-startup-$(Get-Date -Format 'yyyyMMdd-HHmmss').log"
$script:ProcessMonitors = @{}
$script:ShutdownRequested = $false

#region Initialization
function Initialize-Environment {
    [CmdletBinding()]
    param()
    
    # Create necessary directories
    @($LogDirectory, $PidDirectory) | ForEach-Object {
        if (-not (Test-Path $_)) {
            New-Item -ItemType Directory -Path $_ -Force | Out-Null
        }
    }
    
    # Initialize log file
    "Recursia Enterprise Startup Log - $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File $LogFile
    "Environment: $Environment" | Add-Content $LogFile
    "Backend Port: $BackendPort" | Add-Content $LogFile
    "Frontend Port: $FrontendPort" | Add-Content $LogFile
    "=" * 80 | Add-Content $LogFile
}

function Write-Log {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory, Position = 0)]
        [string]$Message,
        
        [Parameter()]
        [ValidateSet('Info', 'Warning', 'Error', 'Success', 'Debug')]
        [string]$Level = 'Info',
        
        [Parameter()]
        [switch]$NoConsole
    )
    
    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff'
    $logEntry = "[$timestamp] [$Level] $Message"
    
    # Write to log file
    $logEntry | Add-Content $LogFile
    
    # Write to console unless suppressed
    if (-not $NoConsole) {
        $color = switch ($Level) {
            'Info'    { 'White' }
            'Warning' { 'Yellow' }
            'Error'   { 'Red' }
            'Success' { 'Green' }
            'Debug'   { 'Gray' }
        }
        Write-Host $Message -ForegroundColor $color
    }
}
#endregion

#region Process Management
function Get-ProcessUsingPort {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$Port
    )
    
    try {
        # Use multiple methods to find process
        
        # Method 1: Get-NetTCPConnection (requires elevated privileges)
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if ($connection) {
            $pid = $connection.OwningProcess
            if ($pid -gt 0) {
                $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                if ($process) {
                    return @{
                        ProcessId = $pid
                        ProcessName = $process.ProcessName
                        CommandLine = $process.CommandLine
                        Method = 'Get-NetTCPConnection'
                    }
                }
            }
        }
        
        # Method 2: netstat parsing (fallback)
        $netstatOutput = & netstat -ano | Select-String ":$Port\s+.*LISTENING\s+(\d+)$"
        if ($netstatOutput) {
            $pid = [int]$netstatOutput.Matches[0].Groups[1].Value
            if ($pid -gt 0) {
                $process = Get-Process -Id $pid -ErrorAction SilentlyContinue
                if ($process) {
                    return @{
                        ProcessId = $pid
                        ProcessName = $process.ProcessName
                        CommandLine = $null
                        Method = 'netstat'
                    }
                }
            }
        }
        
        # Method 3: Test-NetConnection (last resort)
        if (Test-NetConnection -ComputerName localhost -Port $Port -InformationLevel Quiet -WarningAction SilentlyContinue) {
            return @{
                ProcessId = $null
                ProcessName = 'Unknown'
                CommandLine = $null
                Method = 'Test-NetConnection'
            }
        }
    }
    catch {
        Write-Log "Error checking port $Port : $_" -Level Warning
    }
    
    return $null
}

function Stop-ProcessUsingPort {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [int]$Port,
        
        [Parameter()]
        [switch]$Force
    )
    
    $processInfo = Get-ProcessUsingPort -Port $Port
    if ($processInfo) {
        Write-Log "Port $Port is in use by $($processInfo.ProcessName) (PID: $($processInfo.ProcessId))" -Level Warning
        
        if ($processInfo.ProcessId) {
            if ($Force) {
                Write-Log "Force stopping process $($processInfo.ProcessId)..." -Level Warning
                Stop-Process -Id $processInfo.ProcessId -Force -ErrorAction SilentlyContinue
                Start-Sleep -Milliseconds 500
                return $true
            }
            else {
                $response = Read-Host "Kill process $($processInfo.ProcessName) using port $Port? (y/n)"
                if ($response -eq 'y') {
                    Stop-Process -Id $processInfo.ProcessId -Force -ErrorAction SilentlyContinue
                    Start-Sleep -Milliseconds 500
                    return $true
                }
            }
        }
        else {
            Write-Log "Unable to determine process ID for port $Port" -Level Error
        }
    }
    
    return $false
}

class ProcessManager {
    [string]$Name
    [string]$Type
    [System.Diagnostics.Process]$Process
    [int]$Port
    [string]$PidFile
    [int]$RestartCount = 0
    [datetime]$LastRestart = [datetime]::MinValue
    [System.Text.StringBuilder]$OutputBuffer
    [System.Text.StringBuilder]$ErrorBuffer
    
    ProcessManager([string]$name, [string]$type, [int]$port) {
        $this.Name = $name
        $this.Type = $type
        $this.Port = $port
        $this.PidFile = Join-Path $script:PidDirectory "$name.pid"
        $this.OutputBuffer = [System.Text.StringBuilder]::new()
        $this.ErrorBuffer = [System.Text.StringBuilder]::new()
    }
    
    [void]SavePid() {
        if ($this.Process -and -not $this.Process.HasExited) {
            $this.Process.Id | Out-File -FilePath $this.PidFile -Force
            Write-Log "Saved $($this.Name) PID: $($this.Process.Id)" -Level Debug
        }
    }
    
    [int]LoadPid() {
        if (Test-Path $this.PidFile) {
            $pid = Get-Content $this.PidFile -ErrorAction SilentlyContinue
            if ($pid) {
                return [int]$pid
            }
        }
        return 0
    }
    
    [void]ClearPid() {
        if (Test-Path $this.PidFile) {
            Remove-Item $this.PidFile -Force -ErrorAction SilentlyContinue
        }
    }
    
    [bool]IsRunning() {
        return $this.Process -and -not $this.Process.HasExited
    }
    
    [void]Stop() {
        if ($this.IsRunning()) {
            Write-Log "Stopping $($this.Name) (PID: $($this.Process.Id))..." -Level Info
            
            # Try graceful shutdown first
            $this.Process.CloseMainWindow() | Out-Null
            $stopped = $this.Process.WaitForExit(5000)
            
            if (-not $stopped) {
                Write-Log "$($this.Name) did not stop gracefully, forcing..." -Level Warning
                $this.Process.Kill()
                $this.Process.WaitForExit(2000)
            }
            
            Write-Log "$($this.Name) stopped" -Level Success
        }
        
        $this.ClearPid()
    }
    
    [string]GetRecentOutput() {
        return $this.OutputBuffer.ToString()
    }
    
    [string]GetRecentErrors() {
        return $this.ErrorBuffer.ToString()
    }
}
#endregion

#region Dependency Checking
function Test-PythonEnvironment {
    [CmdletBinding()]
    param()
    
    Write-Log "Checking Python environment..." -Level Info
    
    # Find Python command
    $pythonCommands = @('python', 'python3', 'py')
    $pythonCmd = $null
    
    foreach ($cmd in $pythonCommands) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            $version = & $cmd --version 2>&1
            if ($version -match "Python 3\.([89]|1[0-9])") {
                $pythonCmd = $cmd
                Write-Log "Found Python: $version" -Level Success
                break
            }
        }
    }
    
    if (-not $pythonCmd) {
        throw "Python 3.8+ not found. Please install from python.org"
    }
    
    # Check virtual environment
    $venvPath = Join-Path $PSScriptRoot ".." ".." "venv"
    $venvActivate = Join-Path $venvPath "Scripts" "Activate.ps1"
    
    if (-not (Test-Path $venvActivate)) {
        Write-Log "Creating virtual environment..." -Level Info
        & $pythonCmd -m venv $venvPath
    }
    
    # Return Python executable path in venv
    $venvPython = Join-Path $venvPath "Scripts" "python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    else {
        return $pythonCmd
    }
}

function Test-NodeEnvironment {
    [CmdletBinding()]
    param()
    
    Write-Log "Checking Node.js environment..." -Level Info
    
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
        throw "Node.js not found. Please install from nodejs.org"
    }
    
    $nodeVersion = & node --version 2>&1
    Write-Log "Found Node.js: $nodeVersion" -Level Success
    
    # Check npm
    if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
        throw "npm not found. Please reinstall Node.js"
    }
    
    return $true
}

function Install-Dependencies {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PythonPath
    )
    
    $projectRoot = Join-Path $PSScriptRoot ".." ".."
    
    # Install Python dependencies
    Write-Log "Checking Python dependencies..." -Level Info
    $requirementsPath = Join-Path $projectRoot "requirements.txt"
    
    if (Test-Path $requirementsPath) {
        # Check if packages are already installed
        $packagesInstalled = & $PythonPath -c "
try:
    import fastapi, uvicorn, numpy, torch
    print('OK')
except ImportError:
    print('MISSING')
" 2>&1
        
        if ($packagesInstalled -ne 'OK') {
            Write-Log "Installing Python packages..." -Level Info
            & $PythonPath -m pip install --upgrade pip
            & $PythonPath -m pip install -r $requirementsPath
            
            # Install PyTorch if needed
            & $PythonPath -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        }
        else {
            Write-Log "Python packages already installed" -Level Success
        }
    }
    
    # Install frontend dependencies
    Write-Log "Checking frontend dependencies..." -Level Info
    $frontendPath = Join-Path $projectRoot "frontend"
    
    if (Test-Path $frontendPath) {
        Push-Location $frontendPath
        try {
            if (-not (Test-Path "node_modules")) {
                Write-Log "Installing frontend packages..." -Level Info
                & npm install
            }
            else {
                Write-Log "Frontend packages already installed" -Level Success
            }
        }
        finally {
            Pop-Location
        }
    }
}
#endregion

#region Server Management
function Start-BackendServer {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$PythonPath,
        
        [Parameter(Mandatory)]
        [ProcessManager]$Manager
    )
    
    Write-Log "Starting backend server on port $($Manager.Port)..." -Level Info
    
    $projectRoot = Join-Path $PSScriptRoot ".." ".."
    
    # Create process start info
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $PythonPath
    $psi.Arguments = "-u -m uvicorn src.api.unified_api_server:app --host 127.0.0.1 --port $($Manager.Port) --reload"
    $psi.WorkingDirectory = $projectRoot
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    
    # Set environment variables
    $psi.EnvironmentVariables["PYTHONPATH"] = $projectRoot
    $psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1"
    
    # Create and start process
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    
    # Set up output handlers
    $outputHandler = {
        param($sender, $args)
        if ($args.Data) {
            $Manager.OutputBuffer.AppendLine($args.Data)
            if ($Manager.OutputBuffer.Length -gt 10000) {
                $Manager.OutputBuffer.Remove(0, 5000)
            }
            
            if ($Environment -eq 'Debug') {
                Write-Log "[Backend] $($args.Data)" -Level Debug -NoConsole
            }
        }
    }.GetNewClosure()
    
    $errorHandler = {
        param($sender, $args)
        if ($args.Data) {
            $Manager.ErrorBuffer.AppendLine($args.Data)
            if ($Manager.ErrorBuffer.Length -gt 10000) {
                $Manager.ErrorBuffer.Remove(0, 5000)
            }
            
            Write-Log "[Backend Error] $($args.Data)" -Level Error
        }
    }.GetNewClosure()
    
    $process.add_OutputDataReceived($outputHandler)
    $process.add_ErrorDataReceived($errorHandler)
    
    # Start process
    if ($process.Start()) {
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        
        $Manager.Process = $process
        $Manager.SavePid()
        
        Write-Log "Backend process started with PID: $($process.Id)" -Level Success
        
        # Wait for backend to be ready
        $ready = Wait-ForService -Url "http://localhost:$($Manager.Port)/health" -Timeout 30
        
        if ($ready) {
            Write-Log "Backend server is ready!" -Level Success
            return $true
        }
        else {
            Write-Log "Backend server failed to become ready" -Level Error
            $Manager.Stop()
            return $false
        }
    }
    else {
        Write-Log "Failed to start backend process" -Level Error
        return $false
    }
}

function Start-FrontendServer {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [ProcessManager]$Manager
    )
    
    Write-Log "Starting frontend server on port $($Manager.Port)..." -Level Info
    
    $projectRoot = Join-Path $PSScriptRoot ".." ".."
    $frontendPath = Join-Path $projectRoot "frontend"
    
    # Determine npm command for Windows
    $npmCmd = if ($env:OS -match "Windows" -or $IsWindows) { "npm.cmd" } else { "npm" }
    
    # Create process start info
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $npmCmd
    $psi.Arguments = "run dev"
    $psi.WorkingDirectory = $frontendPath
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true
    
    # Set Vite port
    $psi.EnvironmentVariables["VITE_PORT"] = $Manager.Port.ToString()
    
    # Create and start process
    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    
    # Set up output handlers
    $outputHandler = {
        param($sender, $args)
        if ($args.Data) {
            $Manager.OutputBuffer.AppendLine($args.Data)
            if ($Manager.OutputBuffer.Length -gt 10000) {
                $Manager.OutputBuffer.Remove(0, 5000)
            }
            
            if ($Environment -eq 'Debug') {
                Write-Log "[Frontend] $($args.Data)" -Level Debug -NoConsole
            }
        }
    }.GetNewClosure()
    
    $errorHandler = {
        param($sender, $args)
        if ($args.Data) {
            $Manager.ErrorBuffer.AppendLine($args.Data)
            if ($Manager.ErrorBuffer.Length -gt 10000) {
                $Manager.ErrorBuffer.Remove(0, 5000)
            }
            
            # Vite outputs some info to stderr, not all are errors
            if ($args.Data -match "error|fail|exception" -and $args.Data -notmatch "warn") {
                Write-Log "[Frontend Error] $($args.Data)" -Level Error
            }
        }
    }.GetNewClosure()
    
    $process.add_OutputDataReceived($outputHandler)
    $process.add_ErrorDataReceived($errorHandler)
    
    # Start process
    if ($process.Start()) {
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        
        $Manager.Process = $process
        $Manager.SavePid()
        
        Write-Log "Frontend process started with PID: $($process.Id)" -Level Success
        
        # Wait for frontend to be ready
        $ready = Wait-ForService -Url "http://localhost:$($Manager.Port)" -Timeout 30
        
        if ($ready) {
            Write-Log "Frontend server is ready!" -Level Success
            return $true
        }
        else {
            Write-Log "Frontend server failed to become ready" -Level Error
            $Manager.Stop()
            return $false
        }
    }
    else {
        Write-Log "Failed to start frontend process" -Level Error
        return $false
    }
}

function Wait-ForService {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [string]$Url,
        
        [Parameter()]
        [int]$Timeout = 30
    )
    
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    
    while ($stopwatch.Elapsed.TotalSeconds -lt $Timeout) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Start-Sleep -Milliseconds 500
    }
    
    return $false
}
#endregion

#region Main Script Logic
function Start-Recursia {
    [CmdletBinding()]
    param()
    
    try {
        # Initialize environment
        Initialize-Environment
        
        Write-Host "`n==================================================" -ForegroundColor Cyan
        Write-Host "     Recursia Quantum OSH Computing Platform" -ForegroundColor Cyan
        Write-Host "     Enterprise Edition - $Environment Mode" -ForegroundColor Cyan
        Write-Host "==================================================`n" -ForegroundColor Cyan
        
        # Check if we're in the right directory
        $projectRoot = Join-Path $PSScriptRoot ".." ".."
        if (-not (Test-Path (Join-Path $projectRoot "src" "recursia.py"))) {
            throw "Not in Recursia project directory. Expected to find src/recursia.py"
        }
        
        # Clean up existing processes
        Write-Log "Checking for existing processes..." -Level Info
        
        # Check and clean ports
        @($BackendPort, $FrontendPort) | ForEach-Object {
            $processInfo = Get-ProcessUsingPort -Port $_
            if ($processInfo) {
                Stop-ProcessUsingPort -Port $_ -Force:($Environment -eq 'Production')
            }
        }
        
        # Set up environment
        $pythonPath = Test-PythonEnvironment
        Test-NodeEnvironment
        
        if (-not $SkipDependencyCheck) {
            Install-Dependencies -PythonPath $pythonPath
        }
        
        # Create process managers
        $script:ProcessMonitors['backend'] = [ProcessManager]::new('backend', 'python', $BackendPort)
        $script:ProcessMonitors['frontend'] = [ProcessManager]::new('frontend', 'node', $FrontendPort)
        
        # Start servers
        $backendStarted = Start-BackendServer -PythonPath $pythonPath -Manager $ProcessMonitors['backend']
        if (-not $backendStarted) {
            throw "Failed to start backend server"
        }
        
        $frontendStarted = Start-FrontendServer -Manager $ProcessMonitors['frontend']
        if (-not $frontendStarted) {
            Write-Log "Frontend failed to start, but backend is running" -Level Warning
        }
        
        # Open browser if in development mode
        if ($Environment -eq 'Development' -and $frontendStarted) {
            Write-Log "Opening browser..." -Level Info
            Start-Process "http://localhost:$FrontendPort"
        }
        
        Write-Host "`n==================================================" -ForegroundColor Green
        Write-Host "Recursia is running!" -ForegroundColor Green
        Write-Host "Frontend: http://localhost:$FrontendPort" -ForegroundColor Cyan
        Write-Host "Backend:  http://localhost:$BackendPort" -ForegroundColor Cyan
        Write-Host "Log file: $LogFile" -ForegroundColor Gray
        Write-Host "`nPress Ctrl+C to stop all servers" -ForegroundColor Yellow
        Write-Host "==================================================" -ForegroundColor Green
        
        # Monitor servers
        Start-ServerMonitoring
    }
    catch {
        Write-Log "Fatal error: $_" -Level Error
        Write-Log $_.ScriptStackTrace -Level Error
        
        # Clean up
        Stop-AllServers
        
        throw
    }
}

function Start-ServerMonitoring {
    [CmdletBinding()]
    param()
    
    # Set up Ctrl+C handler
    [Console]::TreatControlCAsInput = $true
    $script:ShutdownRequested = $false
    
    try {
        while (-not $script:ShutdownRequested) {
            # Check for Ctrl+C
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true)
                if ($key.Key -eq "C" -and $key.Modifiers -eq "Control") {
                    Write-Log "`nShutdown requested..." -Level Warning
                    $script:ShutdownRequested = $true
                    break
                }
            }
            
            # Check server health
            foreach ($name in $ProcessMonitors.Keys) {
                $monitor = $ProcessMonitors[$name]
                
                if (-not $monitor.IsRunning()) {
                    Write-Log "$name server has stopped (exit code: $($monitor.Process.ExitCode))" -Level Error
                    
                    # Log recent output/errors
                    $recentErrors = $monitor.GetRecentErrors()
                    if ($recentErrors) {
                        Write-Log "Recent $name errors:`n$recentErrors" -Level Error
                    }
                    
                    # Auto-restart if enabled
                    if ($AutoRestart -and $monitor.RestartCount -lt $MaxRestartAttempts) {
                        $timeSinceLastRestart = [DateTime]::Now - $monitor.LastRestart
                        
                        # Implement exponential backoff
                        $backoffSeconds = [Math]::Pow(2, $monitor.RestartCount) * 5
                        
                        if ($timeSinceLastRestart.TotalSeconds -gt $backoffSeconds) {
                            Write-Log "Attempting to restart $name (attempt $($monitor.RestartCount + 1)/$MaxRestartAttempts)..." -Level Warning
                            
                            $monitor.RestartCount++
                            $monitor.LastRestart = [DateTime]::Now
                            
                            # Restart the appropriate server
                            if ($name -eq 'backend') {
                                $pythonPath = Test-PythonEnvironment
                                Start-BackendServer -PythonPath $pythonPath -Manager $monitor
                            }
                            else {
                                Start-FrontendServer -Manager $monitor
                            }
                        }
                    }
                    else {
                        Write-Log "$name will not be restarted (restart limit reached or disabled)" -Level Warning
                    }
                }
            }
            
            # Brief pause
            Start-Sleep -Milliseconds 500
        }
    }
    finally {
        [Console]::TreatControlCAsInput = $false
        Stop-AllServers
    }
}

function Stop-AllServers {
    [CmdletBinding()]
    param()
    
    Write-Log "Stopping all servers..." -Level Info
    
    foreach ($name in $ProcessMonitors.Keys) {
        $monitor = $ProcessMonitors[$name]
        $monitor.Stop()
    }
    
    # Clean up any orphaned processes
    Get-Process python, python3 -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*unified_api_server*"
    } | ForEach-Object {
        Write-Log "Stopping orphaned Python process (PID: $($_.Id))" -Level Warning
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    
    Get-Process node -ErrorAction SilentlyContinue | Where-Object {
        $_.CommandLine -like "*vite*" -or $_.CommandLine -like "*dev*"
    } | ForEach-Object {
        Write-Log "Stopping orphaned Node process (PID: $($_.Id))" -Level Warning
        Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
    }
    
    Write-Log "All servers stopped" -Level Success
}
#endregion

# Main entry point
if ($MyInvocation.InvocationName -ne '.') {
    Start-Recursia
}