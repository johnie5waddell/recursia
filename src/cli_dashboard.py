"""
Enhanced CLI Dashboard Integration for Recursia
Integrates the full 3D visualization and web dashboard capabilities with the CLI.
"""

import sys
import logging
import asyncio
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
import time
import signal
import webbrowser
from dataclasses import dataclass

# Core Recursia imports
from src.core.interpreter import RecursiaInterpreter
from src.core.runtime import RecursiaRuntime
from src.core.data_classes import DashboardConfiguration
from src.visualization.dashboard import Dashboard, DashboardState, create_dashboard_with_defaults
from src.visualization.web_dashboard import (
    WebDashboard, WebDashboardConfig, create_web_dashboard, 
    create_integrated_dashboard, WEB_AVAILABLE
)
# Terminal dashboard removed

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CLIDashboardConfig:
    """Configuration for CLI dashboard integration."""
    auto_launch_browser: bool = True
    dashboard_host: str = "localhost"
    dashboard_port: int = 8080
    enable_web_dashboard: bool = True
    enable_3d_visualization: bool = True
    enable_real_time_updates: bool = True
    auto_save_experiments: bool = True
    visualization_export_path: str = "./visualizations"
    experiment_storage_path: str = "./experiments"
    scientific_reporting: bool = True
    log_level: str = "INFO"


class CLIDashboardIntegration:
    """
    Integrates Recursia's full dashboard capabilities with the CLI.
    Provides enterprise-grade visualization, experiment management, and real-time monitoring.
    """
    
    def __init__(self, config: CLIDashboardConfig = None):
        self.config = config or CLIDashboardConfig()
        
        # Core components
        self.core_dashboard: Optional[Dashboard] = None
        self.web_dashboard: Optional[WebDashboard] = None
        self.terminal_dashboard = None  # Disabled - was causing console spam
        self.runtime: Optional[RecursiaRuntime] = None
        self.interpreter: Optional[RecursiaInterpreter] = None
        
        # State tracking
        self.is_running = False
        self.dashboard_thread: Optional[threading.Thread] = None
        self.current_execution: Optional[Dict[str, Any]] = None
        
        # Setup logging
        self._setup_logging()
        
        logger.info("CLI Dashboard Integration initialized")
    
    def _setup_logging(self):
        """Setup enhanced logging for dashboard integration."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler for dashboard logs
        log_file = Path("./logs/dashboard.log")
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors to console
        console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        
        # Configure loggers
        for logger_name in ['src.visualization', 'src.cli_dashboard']:
            log = logging.getLogger(logger_name)
            log.setLevel(log_level)
            log.addHandler(file_handler)
            log.addHandler(console_handler)
    
    def initialize_dashboards(self, runtime: RecursiaRuntime = None, 
                            interpreter: RecursiaInterpreter = None) -> bool:
        """Initialize core and web dashboards with system integration."""
        try:
            self.runtime = runtime
            self.interpreter = interpreter
            
            # Create dashboard state with system references
            dashboard_state = DashboardState(
                runtime=runtime,
                interpreter=interpreter,
                execution_context=getattr(interpreter, 'execution_context', None) if interpreter else None,
                config=DashboardConfiguration(
                    real_time_updates=self.config.enable_real_time_updates,
                    enable_3d_visualization=self.config.enable_3d_visualization,
                    update_interval=0.5,
                    history_window=1000
                )
            )
            
            # Initialize physics and quantum system references if available
            if runtime:
                dashboard_state.physics_engine = getattr(runtime, 'physics_engine', None)
                dashboard_state.field_dynamics = getattr(runtime, 'field_dynamics', None)
                dashboard_state.memory_field_physics = getattr(runtime, 'memory_field', None)
                dashboard_state.coherence_manager = getattr(runtime, 'coherence_manager', None)
                dashboard_state.entanglement_manager = getattr(runtime, 'entanglement_manager', None)
                dashboard_state.observer_dynamics = getattr(runtime, 'observer_dynamics', None)
                dashboard_state.recursive_mechanics = getattr(runtime, 'recursive_mechanics', None)
                dashboard_state.quantum_backend = getattr(runtime, 'quantum_backend', None)
                dashboard_state.state_registry = getattr(runtime, 'state_registry', None)
                dashboard_state.observer_registry = getattr(runtime, 'observer_registry', None)
                dashboard_state.event_system = getattr(runtime, 'event_system', None)
                dashboard_state.memory_manager = getattr(runtime, 'memory_manager', None)
            
            # Create core dashboard
            self.core_dashboard = Dashboard(dashboard_state)
            
            # Create web dashboard if available and enabled
            if WEB_AVAILABLE and self.config.enable_web_dashboard:
                web_config = WebDashboardConfig(
                    host=self.config.dashboard_host,
                    port=self.config.dashboard_port,
                    enable_3d_rendering=self.config.enable_3d_visualization,
                    real_time_updates=self.config.enable_real_time_updates,
                    experiment_storage_path=self.config.experiment_storage_path,
                    export_path=self.config.visualization_export_path,
                    enable_experiment_builder=True,
                    enable_scientific_reporting=self.config.scientific_reporting
                )
                
                self.web_dashboard = create_web_dashboard(web_config)
                self.web_dashboard.set_dashboard(self.core_dashboard)
                
                logger.info("Web dashboard initialized successfully")
            else:
                if not WEB_AVAILABLE:
                    logger.warning("Web dashboard dependencies not available")
                    logger.info("Using core dashboard only - no terminal dashboard")
                    # Do NOT create terminal dashboard - it's causing console spam
                else:
                    logger.info("Using core dashboard only")
            
            return True
            
        except Exception as e:
            logger.error(f"Dashboard initialization failed: {e}")
            return False
    
    def start_dashboards(self) -> bool:
        """Start dashboard services."""
        try:
            if not self.core_dashboard:
                logger.error("Dashboards not initialized")
                return False
            
            # Start core dashboard
            self.core_dashboard.start()
            logger.info("Core dashboard started")
            
            # Start web dashboard in separate thread if available
            if self.web_dashboard:
                def run_web_dashboard():
                    try:
                        logger.info(f"Starting web dashboard on {self.config.dashboard_host}:{self.config.dashboard_port}")
                        self.web_dashboard.run(
                            host=self.config.dashboard_host,
                            port=self.config.dashboard_port
                        )
                    except Exception as e:
                        logger.error(f"Web dashboard failed: {e}")
                
                self.dashboard_thread = threading.Thread(target=run_web_dashboard, daemon=True)
                self.dashboard_thread.start()
                
                # Give web server time to start
                time.sleep(2.0)
                
                # Auto-launch browser if enabled
                if self.config.auto_launch_browser:
                    self._launch_browser()
            elif self.terminal_dashboard:
                # Do NOT start terminal dashboard - it's causing console spam
                logger.info("Terminal dashboard disabled to prevent console spam")
            
            self.is_running = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start dashboards: {e}")
            return False
    
    def _launch_browser(self):
        """Launch browser to dashboard URL."""
        try:
            dashboard_url = f"http://{self.config.dashboard_host}:{self.config.dashboard_port}"
            logger.info(f"Opening dashboard in browser: {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            logger.warning(f"Failed to open browser: {e}")
    
    def stop_dashboards(self):
        """Stop dashboard services."""
        try:
            self.is_running = False
            
            if self.core_dashboard:
                self.core_dashboard.stop()
                logger.info("Core dashboard stopped")
            
            if self.terminal_dashboard:
                self.terminal_dashboard.stop()
                logger.info("Terminal dashboard stopped")
            
            # Web dashboard will stop when the main thread exits
            if self.dashboard_thread and self.dashboard_thread.is_alive():
                logger.info("Web dashboard will stop when process exits")
            
        except Exception as e:
            logger.error(f"Error stopping dashboards: {e}")
    
    def update_execution_context(self, execution_data: Dict[str, Any]):
        """Update dashboards with current execution context."""
        if not self.core_dashboard:
            return
        
        try:
            self.current_execution = execution_data
            self.core_dashboard.update(execution_data)
            
            # Log significant events
            if 'quantum_operations' in execution_data:
                ops_count = execution_data['quantum_operations']
                logger.debug(f"Execution context updated: {ops_count} quantum operations")
                
        except Exception as e:
            logger.warning(f"Failed to update execution context: {e}")
    
    def render_visualization(self, width: int = 1920, height: int = 1080) -> Dict[str, Any]:
        """Render current visualization."""
        if not self.core_dashboard:
            return {"success": False, "error": "Dashboard not initialized"}
        
        try:
            result = self.core_dashboard.render_dashboard(width=width, height=height)
            
            # Save visualization if path configured
            if result.get("success") and result.get("image_data"):
                self._save_visualization(result["image_data"])
            
            return result
            
        except Exception as e:
            logger.error(f"Visualization render failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _save_visualization(self, image_data: bytes):
        """Save visualization to file."""
        try:
            viz_path = Path(self.config.visualization_export_path)
            viz_path.mkdir(exist_ok=True, parents=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = viz_path / f"visualization_{timestamp}.png"
            
            with open(filename, 'wb') as f:
                f.write(image_data)
            
            logger.info(f"Visualization saved to: {filename}")
            
        except Exception as e:
            logger.warning(f"Failed to save visualization: {e}")
    
    def export_scientific_report(self, format: str = "json") -> Dict[str, Any]:
        """Export comprehensive scientific report."""
        if not self.core_dashboard:
            return {"success": False, "error": "Dashboard not initialized"}
        
        try:
            result = self.core_dashboard.export_scientific_report(format=format)
            
            if result.get("success"):
                logger.info(f"Scientific report exported: {result.get('filename')}")
            
            return result
            
        except Exception as e:
            logger.error(f"Report export failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_dashboard_url(self) -> Optional[str]:
        """Get the dashboard URL if web dashboard is active."""
        if self.web_dashboard:
            return f"http://{self.config.dashboard_host}:{self.config.dashboard_port}"
        return None
    
    def is_terminal_mode(self) -> bool:
        """Check if running in terminal mode."""
        return self.terminal_dashboard is not None and self.web_dashboard is None
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health information."""
        if not self.core_dashboard:
            return {"overall_health": 0.0, "error": "Dashboard not initialized"}
        
        return self.core_dashboard.get_system_health()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.core_dashboard:
            return {"error": "Dashboard not initialized"}
        
        return self.core_dashboard.current_metrics.to_dict()
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List available experiments."""
        if not self.web_dashboard:
            return []
        
        try:
            experiments = self.web_dashboard.experiment_manager.list_experiments(include_templates=True)
            return [exp.__dict__ for exp in experiments]
        except Exception as e:
            logger.error(f"Failed to list experiments: {e}")
            return []
    
    def create_experiment_from_execution(self, name: str, code: str, 
                                       execution_result: Dict[str, Any]) -> Optional[str]:
        """Create experiment from successful execution."""
        if not self.web_dashboard:
            logger.warning("Web dashboard not available for experiment creation")
            return None
        
        try:
            description = f"Experiment generated from CLI execution at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            # Extract parameters from execution result
            parameters = {
                "execution_time": execution_result.get("execution_time", 0.0),
                "quantum_operations": execution_result.get("quantum_operations", 0),
                "success": execution_result.get("success", False)
            }
            
            exp_id = self.web_dashboard.experiment_manager.create_experiment(
                name=name,
                description=description,
                code=code,
                parameters=parameters,
                author="cli_user",
                tags=["cli-generated", "auto-saved"]
            )
            
            logger.info(f"Experiment created from execution: {exp_id}")
            return exp_id
            
        except Exception as e:
            logger.error(f"Failed to create experiment: {e}")
            return None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_dashboards()


def create_cli_dashboard_integration(config: CLIDashboardConfig = None) -> CLIDashboardIntegration:
    """Create CLI dashboard integration."""
    return CLIDashboardIntegration(config=config)


def setup_signal_handlers(dashboard_integration: CLIDashboardIntegration):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger.info("Received shutdown signal, stopping dashboards...")
        dashboard_integration.stop_dashboards()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


# Enhanced CLI integration functions
def run_with_dashboard(recursia_file: str, config: CLIDashboardConfig = None, 
                      **execution_kwargs) -> Dict[str, Any]:
    """
    Run Recursia file with full dashboard integration.
    
    Args:
        recursia_file: Path to Recursia file to execute
        config: Dashboard configuration
        **execution_kwargs: Additional execution parameters
    
    Returns:
        Execution result with dashboard information
    """
    dashboard_config = config or CLIDashboardConfig()
    
    with create_cli_dashboard_integration(dashboard_config) as dashboard_integration:
        try:
            # Initialize runtime and interpreter
            from src.core.interpreter import RecursiaInterpreter
            from src.core.runtime import RecursiaRuntime
            
            interpreter = RecursiaInterpreter()
            runtime = RecursiaRuntime()
            
            # Initialize dashboards with system integration
            if not dashboard_integration.initialize_dashboards(runtime, interpreter):
                logger.error("Failed to initialize dashboards")
                return {"success": False, "error": "Dashboard initialization failed"}
            
            # Start dashboard services
            if not dashboard_integration.start_dashboards():
                logger.error("Failed to start dashboards")
                return {"success": False, "error": "Dashboard startup failed"}
            
            # Setup signal handlers
            setup_signal_handlers(dashboard_integration)
            
            # Execute the Recursia file
            logger.info(f"Executing Recursia file: {recursia_file}")
            start_time = time.time()
            
            # This would integrate with your existing execution pipeline
            execution_result = interpreter.run_file(recursia_file, **execution_kwargs)
            
            execution_time = time.time() - start_time
            
            # Update dashboard with execution results
            execution_data = {
                "file": recursia_file,
                "execution_time": execution_time,
                "success": execution_result.get("success", False),
                "quantum_operations": execution_result.get("quantum_operations", 0),
                "timestamp": time.time()
            }
            
            dashboard_integration.update_execution_context(execution_data)
            
            # Auto-save experiment if enabled and execution successful
            if (dashboard_config.auto_save_experiments and 
                execution_result.get("success") and
                dashboard_integration.web_dashboard):
                
                with open(recursia_file, 'r') as f:
                    code = f.read()
                
                exp_name = f"Auto-saved: {Path(recursia_file).stem}"
                exp_id = dashboard_integration.create_experiment_from_execution(
                    exp_name, code, execution_result
                )
                execution_result["experiment_id"] = exp_id
            
            # Add dashboard information to result
            dashboard_url = dashboard_integration.get_dashboard_url()
            if dashboard_url:
                execution_result["dashboard_url"] = dashboard_url
                logger.info(f"Dashboard available at: {dashboard_url}")
            
            execution_result["dashboard_integration"] = {
                "core_dashboard_active": dashboard_integration.core_dashboard is not None,
                "web_dashboard_active": dashboard_integration.web_dashboard is not None,
                "visualization_path": dashboard_config.visualization_export_path
            }
            
            # Keep dashboard running for a while to allow interaction
            if dashboard_integration.web_dashboard:
                logger.info("Dashboard is running. Press Ctrl+C to exit.")
                try:
                    # Keep the main thread alive
                    while dashboard_integration.is_running:
                        time.sleep(1.0)
                        
                        # Update dashboard periodically
                        dashboard_integration.update_execution_context(execution_data)
                        
                except KeyboardInterrupt:
                    logger.info("Dashboard shutdown requested")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Execution with dashboard failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "dashboard_integration": {
                    "core_dashboard_active": False,
                    "web_dashboard_active": False
                }
            }


def launch_dashboard_only(config: CLIDashboardConfig = None) -> bool:
    """
    Launch dashboard without executing any Recursia code.
    Useful for exploring experiments and visualization.
    """
    dashboard_config = config or CLIDashboardConfig()
    
    with create_cli_dashboard_integration(dashboard_config) as dashboard_integration:
        try:
            # Initialize dashboards without runtime (standalone mode)
            if not dashboard_integration.initialize_dashboards():
                logger.error("Failed to initialize dashboards")
                return False
            
            # Start dashboard services
            if not dashboard_integration.start_dashboards():
                logger.error("Failed to start dashboards")
                return False
            
            dashboard_url = dashboard_integration.get_dashboard_url()
            if dashboard_url:
                print(f"\n🚀 Recursia Dashboard is running at: {dashboard_url}")
                print("🎯 Features available:")
                print("   • Interactive 3D quantum visualization")
                print("   • Experiment builder and library")
                print("   • Real-time OSH metrics monitoring")
                print("   • Scientific report generation")
                print("   • WebSocket real-time updates")
                print("\n💡 Press Ctrl+C to stop the dashboard\n")
            elif dashboard_integration.is_terminal_mode():
                # Terminal dashboard disabled - skip terminal mode entirely
                print("\n📊 Core Dashboard running (backend only - terminal dashboard disabled)")
                print("💡 Press Ctrl+C to stop\n")
            else:
                print("\n📊 Core Dashboard running (backend only)")
                print("💡 Press Ctrl+C to stop\n")
            
            # Setup signal handlers
            setup_signal_handlers(dashboard_integration)
            
            # Keep dashboard running (for web/core modes)
            try:
                while dashboard_integration.is_running:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("\n🛑 Dashboard shutdown requested")
            
            return True
            
        except Exception as e:
            logger.error(f"Dashboard launch failed: {e}")
            return False


# Export main components
__all__ = [
    'CLIDashboardIntegration',
    'CLIDashboardConfig', 
    'create_cli_dashboard_integration',
    'run_with_dashboard',
    'launch_dashboard_only'
]

if __name__ == "__main__":
    """Main entry point for the CLI dashboard server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recursia CLI Dashboard Server')
    parser.add_argument('--port', type=int, default=8000, help='Server port (default: 8000)')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    parser.add_argument('--no-browser', action='store_true', help='Do not open browser automatically')
    
    args = parser.parse_args()
    
    # Create configuration
    config = CLIDashboardConfig(
        auto_launch_browser=not args.no_browser,
        dashboard_host=args.host,
        dashboard_port=args.port,
        enable_web_dashboard=True,
        enable_3d_visualization=True,
        enable_real_time_updates=True
    )
    
    print(f"🚀 Starting Recursia Backend Server on {args.host}:{args.port}")
    
    # Launch dashboard server
    success = launch_dashboard_only(config)
    
    if not success:
        print("❌ Failed to start dashboard server")
        sys.exit(1)
    else:
        print("✅ Dashboard server started successfully")