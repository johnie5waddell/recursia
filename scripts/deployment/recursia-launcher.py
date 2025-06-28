#!/usr/bin/env python3
"""
Enterprise-grade launcher for Recursia that handles signals properly on Windows
"""
import os
import sys
import signal
import subprocess
import platform
import time
from pathlib import Path

class RecursiaLauncher:
    def __init__(self):
        self.processes = {}
        self.running = True
        self.setup_signal_handlers()
        
    def setup_signal_handlers(self):
        """Setup proper signal handling for clean shutdown"""
        # Handle Ctrl+C properly on all platforms
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Windows-specific signal handling
        if platform.system() == 'Windows':
            signal.signal(signal.SIGBREAK, self.signal_handler)
    
    def windows_handler(self, event):
        """Windows-specific console event handler"""
        self.shutdown()
        return True
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print("\n\n🛑 Shutdown signal received. Stopping servers gracefully...")
        self.shutdown()
        
    def start_backend(self):
        """Start the backend server"""
        print("🚀 Starting backend server on port 8080...")
        
        # Set PYTHONPATH to project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent.parent.parent)
        
        # Start backend with platform-specific handling
        if platform.system() == 'Windows':
            # Windows needs special process group handling for signals
            self.processes['backend'] = subprocess.Popen(
                [sys.executable, 'src/api/unified_api_server.py'],
                env=env,
                cwd=Path(__file__).parent.parent.parent,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
            )
        else:
            # Keep existing behavior for Unix systems (Ubuntu/Mac)
            self.processes['backend'] = subprocess.Popen(
                [sys.executable, 'src/api/unified_api_server.py'],
                env=env,
                cwd=Path(__file__).parent.parent.parent
            )
        
        # Save PID for external management
        with open('backend.pid', 'w') as f:
            f.write(str(self.processes['backend'].pid))
            
        # Wait a moment for backend to start
        print("  Waiting for backend to initialize...")
        time.sleep(5)  # Give it more time
        
        # Check if process is still running
        if self.processes['backend'] and self.processes['backend'].poll() is None:
            print("✅ Backend is ready!")
            return True
        else:
            print("⚠️  Backend failed to start!")
            return False
        
    def start_frontend(self):
        """Start the frontend server"""
        print("🚀 Starting frontend server on port 5173...")
        
        # Determine npm command based on platform
        npm_cmd = 'npm.cmd' if platform.system() == 'Windows' else 'npm'
        
        # Start frontend
        frontend_dir = Path(__file__).parent.parent.parent / 'frontend'
        self.processes['frontend'] = subprocess.Popen(
            [npm_cmd, 'run', 'dev'],
            cwd=frontend_dir,
            shell=False  # Important for signal handling
        )
        
        # Save PID
        with open('frontend.pid', 'w') as f:
            f.write(str(self.processes['frontend'].pid))
            
        print("✅ Frontend server started!")
        return True
        
    def shutdown(self):
        """Shutdown all processes gracefully"""
        self.running = False
        
        for name, process in self.processes.items():
            if process and process.poll() is None:
                print(f"  Stopping {name} (PID: {process.pid})...")
                
                # Try graceful shutdown first
                if platform.system() == 'Windows':
                    # Windows: send CTRL_C_EVENT to the process group
                    try:
                        # First try CTRL_C_EVENT
                        process.send_signal(signal.CTRL_C_EVENT)
                    except:
                        # If that fails, try CTRL_BREAK_EVENT
                        try:
                            process.send_signal(signal.CTRL_BREAK_EVENT)
                        except:
                            # Last resort: terminate
                            process.terminate()
                else:
                    # Unix systems: normal terminate
                    process.terminate()
                
                # Give it 5 seconds to shut down gracefully
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"  Force killing {name}...")
                    process.kill()
                    
        # Clean up PID files
        for pid_file in ['backend.pid', 'frontend.pid']:
            if os.path.exists(pid_file):
                os.remove(pid_file)
                
        print("\n✅ All servers stopped successfully!")
        sys.exit(0)
        
    def run(self):
        """Main run loop"""
        # Kill any existing processes on our ports
        self.cleanup_ports()
        
        # Start servers
        if not self.start_backend():
            self.shutdown()
            return
            
        if not self.start_frontend():
            self.shutdown()
            return
            
        # Open browser
        print("\n🌐 Opening browser at http://localhost:5173")
        import webbrowser
        webbrowser.open('http://localhost:5173')
        
        print("\n" + "="*50)
        print("✅ Recursia is running!")
        print("Frontend: http://localhost:5173")
        print("Backend:  http://localhost:8080")
        print("\n🛑 Press Ctrl+C to stop all servers")
        print("="*50 + "\n")
        
        # Monitor processes
        try:
            while self.running:
                # Check if processes are still alive
                for name, process in self.processes.items():
                    if process and process.poll() is not None:
                        print(f"\n⚠️  {name} crashed! Exit code: {process.returncode}")
                        self.shutdown()
                        return
                        
                time.sleep(1)
                
        except KeyboardInterrupt:
            # This should be caught by our signal handler
            pass
            
    def cleanup_ports(self):
        """Clean up processes using our ports"""
        if platform.system() == 'Windows':
            print("🧹 Cleaning up existing processes...")
            
            # Use netstat to find processes
            for port in [8080, 5173]:
                try:
                    result = subprocess.run(
                        ['netstat', '-ano', '-p', 'tcp'],
                        capture_output=True,
                        text=True
                    )
                    
                    for line in result.stdout.split('\n'):
                        if f':{port}' in line and 'LISTENING' in line:
                            # Extract PID from last column
                            parts = line.split()
                            if parts:
                                pid = int(parts[-1])
                                if pid > 0:
                                    print(f"  Killing process on port {port} (PID: {pid})")
                                    try:
                                        subprocess.run(['taskkill', '/F', '/PID', str(pid)], 
                                                     capture_output=True)
                                    except:
                                        pass
                except:
                    pass
                    
            time.sleep(2)  # Give processes time to die

if __name__ == '__main__':
    launcher = RecursiaLauncher()
    launcher.run()