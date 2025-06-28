#!/usr/bin/env python3
"""
Recursia CLI Entry Point

The main command-line interface for the Recursia quantum simulation language.
This module handles CLI argument parsing, configuration management, and high-level
command orchestration while delegating actual functionality to appropriate subsystems.

Usage:
    recursia run <file> [--target=<target>] [--config=<config>] [--output=<output>] [--verbose] [--debug]
    recursia compile <file> [--target=<target>] [--config=<config>] [--output=<output>] [--optimization=<level>]
    recursia repl [--config=<config>] [--hardware] [--verbose]
    recursia validate-osh <file> [--config=<config>] [--report=<format>] [--strict]
    recursia dashboard [--config=<config>] [--port=<port>] [--host=<host>] [--export=<format>]
    recursia hardware [list|connect|disconnect|status] [--provider=<provider>] [--device=<device>]
    recursia report <type> [--input=<input>] [--output=<output>] [--format=<format>] [--config=<config>]
    recursia config [get|set|reset] [<key>] [<value>] [--global] [--local]
    recursia version
    recursia --help

Commands:
    run             Execute a Recursia program file
    compile         Compile a Recursia program without execution
    repl            Start interactive REPL environment
    validate-osh    Validate OSH alignment and generate analysis
    dashboard       Launch real-time visualization dashboard
    hardware        Manage quantum hardware connections
    report          Generate scientific reports and analyses
    config          Manage configuration settings
    version         Show version information

Options:
    --target=<target>       Compilation target [default: quantum_simulator]
    --config=<config>       Configuration file path
    --output=<output>       Output file or directory
    --optimization=<level>  Optimization level (0-3) [default: 1]
    --report=<format>       Report format (pdf, html, json) [default: pdf]
    --format=<format>       Output format [default: json]
    --port=<port>           Dashboard port [default: 8080]
    --host=<host>           Dashboard host [default: localhost]
    --provider=<provider>   Hardware provider (ibm, rigetti, google, ionq)
    --device=<device>       Specific hardware device
    --verbose               Enable verbose output
    --debug                 Enable debug mode
    --strict                Enable strict validation mode
    --hardware              Enable hardware backend in REPL
    --global                Apply to global configuration
    --local                 Apply to local configuration
    -h --help               Show this help message

Examples:
    recursia run quantum_program.recursia
    recursia compile program.recursia --target=ibm_quantum --optimization=2
    recursia repl --hardware --verbose
    recursia validate-osh experiment.recursia --report=pdf --strict
    recursia dashboard --port=8080 --export=pdf
    recursia hardware connect --provider=ibm --device=ibmq_lima
    recursia report comprehensive --input=results.json --format=pdf
"""

import sys
import os
import traceback
import signal
import atexit
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

# RecursiaInterpreter removed - using bytecode system
from src.core.runtime import RecursiaRuntime
from src.core.compiler import RecursiaCompiler
from src.visualization.dashboard import Dashboard

try:
    from docopt import docopt
except ImportError:
    print("Error: docopt is required. Install with: pip install docopt", file=sys.stderr)
    sys.exit(1)

# Import Recursia core utilities
from src.core.utils import (
    global_error_manager, 
    global_config_manager, 
    performance_profiler,
    colorize_text,
    VisualizationHelper
)

# Version information
__version__ = "2.1.0"
__author__ = "Recursia Development Team"
__license__ = "MIT"

class RecursiaCLI:
    """
    Main CLI controller that orchestrates high-level Recursia operations.
    Delegates actual functionality to appropriate subsystems without overlapping.
    """
    
    def __init__(self):
        self.config = {}
        self.runtime = None
        self.interpreter = None  # Unified interpreter
        self.compiler = None
        self.dashboard = None
        self.logger = None
        self.performance_enabled = False
        self.debug_mode = False
        self.verbose_mode = False
        
        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self, debug: bool = False, verbose: bool = False) -> None:
        """Setup global logging configuration."""
        self.debug_mode = debug
        self.verbose_mode = verbose
        
        log_level = logging.DEBUG if debug else (logging.INFO if verbose else logging.WARNING)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('recursia.log', mode='a')
            ]
        )
        
        self.logger = logging.getLogger('recursia.cli')
        self.logger.info(f"Recursia CLI v{__version__} initialized")
        
        if debug:
            self.logger.debug("Debug mode enabled")
        if verbose:
            self.logger.info("Verbose mode enabled")
    
    def _load_configuration(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load and validate configuration."""
        try:
            if config_path:
                self.config = global_config_manager.load_config(config_path)
            else:
                self.config = global_config_manager.get_default_config()
            
            # Validate configuration
            self._validate_configuration()
            
            if self.logger:
                self.logger.info(f"Configuration loaded: {len(self.config)} sections")
            
            return self.config
            
        except Exception as e:
            error_msg = f"Configuration loading failed: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}", file=sys.stderr)
            raise
    
    def _validate_configuration(self) -> None:
        """Validate loaded configuration for completeness and correctness."""
        required_sections = ['system', 'quantum', 'runtime', 'compiler', 'visualization']
        
        for section in required_sections:
            if section not in self.config:
                self.config[section] = {}
                if self.logger:
                    self.logger.warning(f"Missing configuration section '{section}', using defaults")
        
        # Validate critical settings have valid values
        system_config = self.config.get('system', {})
        if 'max_qubits' in system_config:
            max_qubits = system_config['max_qubits']
            if not isinstance(max_qubits, int) or max_qubits < 1:
                system_config['max_qubits'] = 25
                if self.logger:
                    self.logger.warning("Invalid max_qubits value, defaulting to 25")
    
    def _signal_handler(self, signum, frame):
        """Handle system signals gracefully."""
        signal_name = signal.Signals(signum).name
        if self.logger:
            self.logger.info(f"Received signal {signal_name}, shutting down gracefully")
        else:
            print(f"\nReceived {signal_name}, shutting down gracefully...")
        
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self) -> None:
        """Cleanup resources and subsystems."""
        try:
            if self.dashboard:
                self.dashboard.cleanup()
                self.dashboard = None
            
            if self.runtime:
                self.runtime.cleanup()
                self.runtime = None
            
            # Interpreter removed - using compiler directly
            if self.compiler:
                self.compiler = None
            
            if self.performance_enabled:
                performance_stats = performance_profiler.get_timer_summary()
                if self.logger and performance_stats:
                    self.logger.info(f"Performance summary: {performance_stats}")
            
            if self.logger:
                self.logger.info("Recursia CLI shutdown complete")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during cleanup: {str(e)}")
            else:
                print(f"Cleanup error: {str(e)}", file=sys.stderr)
    
    def _get_runtime(self) -> 'RecursiaRuntime':
        """Get or create runtime instance."""
        if not self.runtime:
            try:
                from src.core.runtime import RecursiaRuntime
                self.runtime = RecursiaRuntime(self.config)
                if self.logger:
                    self.logger.info("Runtime initialized")
            except Exception as e:
                error_msg = f"Runtime initialization failed: {str(e)}"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        return self.runtime
    
    def _get_interpreter(self):
        """Get or create unified interpreter instance."""
        if not self.interpreter:
            try:
                # Import bytecode components
                from src.core.direct_parser import DirectParser
                from src.core.bytecode_vm import RecursiaVM
                from src.core.interpreter import CompilationResult, ExecutionResult
                
                runtime = self._get_runtime()
                
                # Create a unified interpreter wrapper
                class UnifiedInterpreter:
                    def __init__(self, runtime):
                        self.runtime = runtime
                        self.parser = DirectParser()
                        self.vm = None
                        
                    def interpret_file(self, filepath, runtime=None):
                        """Interpret a file and return compilation and execution results."""
                        try:
                            with open(filepath, 'r') as f:
                                code = f.read()
                            
                            # Compile
                            compilation_result = self.compile(code)
                            if not compilation_result.success:
                                return compilation_result, ExecutionResult(
                                    success=False,
                                    error="Compilation failed",
                                    output=compilation_result.errors
                                )
                            
                            # Execute
                            self.vm = RecursiaVM(self.runtime)
                            vm_result = self.vm.execute(compilation_result.bytecode_module)
                            
                            # Convert VM result to ExecutionResult
                            execution_result = ExecutionResult(
                                success=vm_result.success if hasattr(vm_result, 'success') else True,
                                output=vm_result.output if hasattr(vm_result, 'output') else [],
                                metrics={
                                    'integrated_information': getattr(vm_result, 'integrated_information', 0.0),
                                    'kolmogorov_complexity': getattr(vm_result, 'kolmogorov_complexity', 0.0),
                                    'entropy_flux': getattr(vm_result, 'entropy_flux', 0.0)
                                },
                                execution_time=getattr(vm_result, 'execution_time', 0.0)
                            )
                            
                            return compilation_result, execution_result
                            
                        except Exception as e:
                            return CompilationResult(
                                success=False,
                                errors=[str(e)]
                            ), ExecutionResult(
                                success=False,
                                error=str(e)
                            )
                    
                    def compile(self, code):
                        """Compile code to bytecode."""
                        try:
                            bytecode_module = self.parser.parse(code)
                            return CompilationResult(
                                success=True,
                                bytecode_module=bytecode_module,
                                errors=[],
                                warnings=[]
                            )
                        except Exception as e:
                            return CompilationResult(
                                success=False,
                                bytecode_module=None,
                                errors=[str(e)],
                                warnings=[]
                            )
                
                self.interpreter = UnifiedInterpreter(runtime)
                if self.logger:
                    self.logger.info("Unified interpreter initialized")
                    
            except Exception as e:
                error_msg = f"Interpreter initialization failed: {str(e)}"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        return self.interpreter

    def _get_dashboard(self) -> 'Dashboard':
        """Get or create dashboard instance."""
        if not self.dashboard:
            try:
                from src.visualization.dashboard import create_dashboard
                runtime = self._get_runtime()
                self.dashboard = create_dashboard(runtime=runtime, config=self.config)
                if self.logger:
                    self.logger.info("Dashboard initialized")
            except Exception as e:
                error_msg = f"Dashboard initialization failed: {str(e)}"
                if self.logger:
                    self.logger.error(error_msg)
                raise RuntimeError(error_msg)
        return self.dashboard
    
    def cmd_run(self, args: Dict[str, Any]) -> int:
        """Execute a Recursia program file."""
        file_path = args['<file>']
        target = args.get('--target', 'quantum_simulator')
        output_path = args.get('--output')
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if self.performance_enabled:
                performance_profiler.start_timer("file_execution")
            
            interpreter = self._get_interpreter()
            runtime = self._get_runtime()
            
            # Execute the file - interpret_file returns (CompilationResult, ExecutionResult)
            compilation_result, execution_result = interpreter.interpret_file(file_path, runtime)
            
            if self.performance_enabled:
                performance_profiler.stop_timer("file_execution")
            
            # Check compilation first
            if not compilation_result.success:
                print(colorize_text("✗ Compilation failed", "red"))
                for error in compilation_result.errors:
                    print(f"Error: {error}")
                return 1
            
            # Check execution
            if execution_result.success:
                # Display program output first
                if execution_result.result and isinstance(execution_result.result, dict):
                    # Get output from the execution result
                    output = execution_result.result.get('output', [])
                    if output:
                        print(colorize_text("Program Output:", "cyan"))
                        print("-" * 40)
                        for line in output:
                            print(line)
                        print("-" * 40)
                
                print(colorize_text("✓ Execution completed successfully", "green"))
                
                if self.verbose_mode:
                    print(f"Compilation time: {compilation_result.compilation_time:.3f}s")
                    print(f"Execution time: {execution_result.execution_time:.3f}s")
                    
                    if execution_result.runtime_stats:
                        stats = execution_result.runtime_stats
                        print(f"Runtime statistics available: {len(stats)} entries")
                        
                    # Show measurements if available
                    if execution_result.result and isinstance(execution_result.result, dict):
                        measurements = execution_result.result.get('measurements', [])
                        if measurements:
                            print(f"\nMeasurement Results:")
                            for i, m in enumerate(measurements):
                                if isinstance(m, dict):
                                    print(f"  [{i}] State: {m.get('state', 'unknown')}, "
                                          f"Qubit: {m.get('qubit', 'all')}, "
                                          f"Result: {m.get('result', 'N/A')}")
                                else:
                                    print(f"  [{i}] {m}")
                        
                        # Show metrics if available
                        metrics = execution_result.result.get('metrics', {})
                        if metrics:
                            print(f"\nQuantum Metrics:")
                            print(f"  RSP (Recursive Substrate Potential): {metrics.get('rsp', 0):.6f}")
                            print(f"  Average Coherence: {metrics.get('average_coherence', 0):.6f}")
                            print(f"  Average Entropy: {metrics.get('average_entropy', 0):.6f}")
                            print(f"  Total Gates Applied: {metrics.get('total_gates', 0)}")
                            print(f"  Total Measurements: {metrics.get('measurement_count', 0)}")
                
                # Save output if requested and result is available
                if output_path and execution_result.result:
                    self._save_output(str(execution_result.result), output_path)
                
                return 0
            else:
                print(colorize_text("✗ Execution failed", "red"))
                for error in execution_result.errors:
                    print(f"Error: {error}")
                return 1
                
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
        
    def cmd_compile(self, args: Dict[str, Any]) -> int:
        """Compile a Recursia program without execution."""
        file_path = args['<file>']
        target = args.get('--target', 'quantum_simulator')
        output_path = args.get('--output')
        optimization = int(args.get('--optimization', 1))
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if self.performance_enabled:
                performance_profiler.start_timer("compilation")
            
            interpreter = self._get_interpreter()
            
            # Update interpreter configuration
            interpreter.optimization_level = optimization
            interpreter.target_backend = target
            
            # Compile the file
            result = interpreter.compile_file(file_path)
            
            if self.performance_enabled:
                performance_profiler.stop_timer("compilation")
            
            if result.success:
                print(colorize_text("✓ Compilation completed successfully", "green"))
                
                if self.verbose_mode:
                    print(f"Compilation time: {result.compilation_time:.3f}s")
                    print(f"Optimization level: {optimization}")
                    print(f"Target: {target}")
                    if result.source_hash:
                        print(f"Source hash: {result.source_hash[:8]}...")
                
                # Save compiled output
                if output_path and result.code:
                    self._save_output(result.code, output_path)
                elif self.verbose_mode and result.code:
                    print("\nCompiled code preview:")
                    lines = result.code.split('\n')
                    preview_lines = lines[:10] if len(lines) > 10 else lines
                    for line in preview_lines:
                        print(f"  {line}")
                    if len(lines) > 10:
                        print(f"  ... ({len(lines) - 10} more lines)")
                
                return 0
            else:
                print(colorize_text("✗ Compilation failed", "red"))
                for error in result.errors:
                    print(f"Error: {error}")
                return 1
                
        except Exception as e:
            error_msg = f"Compilation failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
        
    def cmd_repl(self, args: Dict[str, Any]) -> int:
        """Start interactive REPL environment."""
        use_hardware = args.get('--hardware', False)
        
        try:
            print(colorize_text(f"Recursia REPL v{__version__}", "cyan"))
            print("Type 'help' for available commands, 'exit' to quit")
            
            interpreter = self._get_interpreter()
            
            # Configure hardware if requested
            if use_hardware:
                runtime = self._get_runtime()
                if runtime.connect_hardware():
                    print(colorize_text("✓ Hardware backend connected", "green"))
                else:
                    print(colorize_text("⚠ Hardware connection failed, using simulator", "yellow"))
            
            # Start REPL
            from src.core.repl import RecursiaREPL
            repl = RecursiaREPL(interpreter)
            repl.cmdloop()
            
            return 0
            
        except KeyboardInterrupt:
            print(colorize_text("\nREPL interrupted by user", "yellow"))
            return 0
        except Exception as e:
            error_msg = f"REPL failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
    
    def cmd_validate_osh(self, args: Dict[str, Any]) -> int:
        """Validate OSH alignment and generate analysis."""
        file_path = args['<file>']
        report_format = args.get('--report', 'pdf')
        strict_mode = args.get('--strict', False)
        
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            print(colorize_text("Running OSH validation analysis...", "cyan"))
            
            interpreter = self._get_interpreter()
            runtime = self._get_runtime()
            
            # Execute with OSH validation enabled
            if hasattr(runtime, 'config'):
                runtime.config['osh_validation_enabled'] = True
                runtime.config['strict_osh_mode'] = strict_mode
            
            compilation_result, execution_result = interpreter.interpret_file(file_path, runtime)
            
            # Check compilation first
            if not compilation_result.success:
                print(colorize_text("✗ OSH validation failed - compilation errors", "red"))
                for error in compilation_result.errors:
                    print(f"Compilation Error: {error}")
                return 1
            
            if execution_result.success:
                # Get OSH metrics from runtime
                try:
                    osh_metrics = runtime.get_current_metrics()
                    validation_score = (osh_metrics.coherence + (1.0 - osh_metrics.entropy) + 
                                    (1.0 - osh_metrics.strain)) / 3.0
                    
                    print(colorize_text("✓ OSH validation completed", "green"))
                    print(f"Validation Score: {validation_score:.3f}")
                    print(f"Coherence: {osh_metrics.coherence:.3f}")
                    print(f"Entropy: {osh_metrics.entropy:.3f}")
                    print(f"Strain: {osh_metrics.strain:.3f}")
                    print(f"RSP: {osh_metrics.rsp:.3f}")
                    
                    # Generate report if requested
                    if report_format != 'none':
                        try:
                            dashboard = self._get_dashboard()
                            report_file = f"osh_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{report_format}"
                            
                            # Create a simple report
                            report_content = f"""OSH Validation Report
    Generated: {datetime.now().isoformat()}
    File: {file_path}

    Validation Score: {validation_score:.3f}
    Coherence: {osh_metrics.coherence:.3f}
    Entropy: {osh_metrics.entropy:.3f}
    Strain: {osh_metrics.strain:.3f}
    RSP: {osh_metrics.rsp:.3f}

    Validation Result: {'PASSED' if validation_score >= 0.7 else 'FAILED'}
    """
                            self._save_output(report_content, report_file)
                            print(f"Report saved: {report_file}")
                        except Exception as e:
                            self.logger.warning(f"Report generation failed: {e}")
                    
                    return 0 if validation_score >= 0.7 else 1
                except Exception as e:
                    print(colorize_text(f"✗ Failed to get OSH metrics: {e}", "red"))
                    return 1
            else:
                print(colorize_text("✗ OSH validation failed", "red"))
                for error in execution_result.errors:
                    print(f"Error: {error}")
                return 1
                
        except Exception as e:
            error_msg = f"OSH validation failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
        
    def cmd_dashboard(self, args: Dict[str, Any]) -> int:
        """Launch enterprise-grade 3D visualization dashboard with experiment builder."""
        port = int(args.get('--port', 8080))
        host = args.get('--host', 'localhost')
        export_format = args.get('--export')
        
        try:
            print(colorize_text("🚀 Starting Recursia Enterprise Dashboard...", "cyan"))
            print(colorize_text("🎯 Features: 3D Visualization • Experiment Builder • Real-time OSH Monitoring", "cyan"))
            
            # Import enhanced dashboard integration
            try:
                from src.cli_dashboard import launch_dashboard_only, CLIDashboardConfig
                
                # Create enhanced dashboard configuration
                config = CLIDashboardConfig(
                    dashboard_host=host,
                    dashboard_port=port,
                    auto_launch_browser=True,
                    enable_web_dashboard=True,
                    enable_3d_visualization=True,
                    enable_real_time_updates=True,
                    scientific_reporting=True,
                    auto_save_experiments=True,
                    log_level="DEBUG" if self.debug_mode else "INFO"
                )
                
                if export_format:
                    config.scientific_reporting = True
                
                # Launch full enterprise dashboard system
                success = launch_dashboard_only(config)
                
                if success:
                    print(colorize_text("✅ Enterprise Dashboard launched successfully", "green"))
                    return 0
                else:
                    print(colorize_text("❌ Dashboard launch failed", "red"))
                    return 1
                    
            except ImportError as e:
                if "Web dashboard dependencies not available" in str(e):
                    print(colorize_text("⚠️  Web dashboard dependencies not installed.", "yellow"))
                    print(colorize_text("💡 Install with: pip install fastapi uvicorn jinja2 websockets", "cyan"))
                    print(colorize_text("🔄 Falling back to basic dashboard...", "yellow"))
                    
                    # Fallback to basic dashboard
                    dashboard = self._get_dashboard()
                    
                    # Configure basic dashboard
                    dashboard.config.real_time_updates = True
                    if hasattr(dashboard.config, 'enable_streaming'):
                        dashboard.config.enable_streaming = True
                    
                    if export_format and hasattr(dashboard.config, 'enable_export'):
                        dashboard.config.enable_export = True
                        dashboard.config.export_format = export_format
                    
                    # Start basic dashboard
                    if hasattr(dashboard, 'start_server'):
                        dashboard.start_server(host=host, port=port)
                        print(colorize_text(f"✓ Basic Dashboard running at http://{host}:{port}", "green"))
                    else:
                        dashboard.start()
                        print(colorize_text("📊 Core Dashboard started (visualization backend only)", "green"))
                        print(colorize_text(f"🌐 No web interface available - install dependencies for full features", "yellow"))
                    
                    print("Press Ctrl+C to stop")
                    
                    # Keep running until interrupted
                    try:
                        while True:
                            import time
                            time.sleep(1)
                    except KeyboardInterrupt:
                        print(colorize_text("\nDashboard stopped by user", "yellow"))
                        dashboard.stop()
                    
                    return 0
                else:
                    raise e
            
        except Exception as e:
            error_msg = f"Dashboard initialization failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
    
    def cmd_hardware(self, args: Dict[str, Any]) -> int:
        """Manage quantum hardware connections."""
        action = args.get('list') or args.get('connect') or args.get('disconnect') or args.get('status')
        if not action:
            action = 'list'  # Default action
        
        provider = args.get('--provider')
        device = args.get('--device')
        
        try:
            runtime = self._get_runtime()
            
            if action == 'list':
                providers = runtime.list_hardware_providers()
                print(colorize_text("Available hardware providers:", "cyan"))
                for p in providers:
                    devices = runtime.list_hardware_devices(p)
                    print(f"  {p}: {', '.join(devices) if devices else 'No devices available'}")
            
            elif action == 'connect':
                if not provider:
                    print(colorize_text("✗ Provider required for connection", "red"))
                    return 1
                
                result = runtime.connect_hardware(provider=provider, device=device)
                if result:
                    print(colorize_text(f"✓ Connected to {provider}" + (f":{device}" if device else ""), "green"))
                else:
                    print(colorize_text(f"✗ Failed to connect to {provider}", "red"))
                    return 1
            
            elif action == 'disconnect':
                runtime.disconnect_hardware()
                print(colorize_text("✓ Hardware disconnected", "green"))
            
            elif action == 'status':
                status = runtime.get_hardware_status()
                if status.get('connected', False):
                    print(colorize_text("✓ Hardware connected", "green"))
                    print(f"Provider: {status.get('provider', 'Unknown')}")
                    print(f"Device: {status.get('device', 'Unknown')}")
                    print(f"Qubits: {status.get('qubits', 'Unknown')}")
                else:
                    print(colorize_text("✗ No hardware connected", "yellow"))
            
            return 0
            
        except Exception as e:
            error_msg = f"Hardware operation failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
    
    def cmd_report(self, args: Dict[str, Any]) -> int:
        """Generate scientific reports and analyses."""
        report_type = args['<type>']
        input_path = args.get('--input')
        output_path = args.get('--output')
        report_format = args.get('--format', 'pdf')
        
        try:
            dashboard = self._get_dashboard()
            
            # Generate report based on type
            if report_type == 'comprehensive':
                report_file = dashboard.export_scientific_report(
                    filename=output_path or f"comprehensive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=report_format,
                    report_type='comprehensive'
                )
            elif report_type == 'summary':
                report_file = dashboard.export_scientific_report(
                    filename=output_path or f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=report_format,
                    report_type='summary'
                )
            elif report_type == 'osh_validation':
                report_file = dashboard.export_scientific_report(
                    filename=output_path or f"osh_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    format=report_format,
                    report_type='osh_validation'
                )
            else:
                print(colorize_text(f"✗ Unknown report type: {report_type}", "red"))
                print("Available types: comprehensive, summary, osh_validation")
                return 1
            
            print(colorize_text(f"✓ Report generated: {report_file}", "green"))
            return 0
            
        except Exception as e:
            error_msg = f"Report generation failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
    
    def cmd_config(self, args: Dict[str, Any]) -> int:
        """Manage configuration settings."""
        action = 'get'  # Default action
        if args.get('set'):
            action = 'set'
        elif args.get('reset'):
            action = 'reset'
        
        key = args.get('<key>')
        value = args.get('<value>')
        is_global = args.get('--global', False)
        
        try:
            if action == 'get':
                if key:
                    config_value = global_config_manager.get_nested(key)
                    print(f"{key}: {config_value}")
                else:
                    # Show all configuration
                    config = global_config_manager.get_all()
                    for section, settings in config.items():
                        print(f"[{section}]")
                        for k, v in settings.items():
                            print(f"  {k}: {v}")
                        print()
            
            elif action == 'set':
                if not key or value is None:
                    print(colorize_text("✗ Key and value required for set operation", "red"))
                    return 1
                
                global_config_manager.set_nested(key, value, global_scope=is_global)
                print(colorize_text(f"✓ Configuration updated: {key} = {value}", "green"))
            
            elif action == 'reset':
                if key:
                    global_config_manager.reset_key(key, global_scope=is_global)
                    print(colorize_text(f"✓ Configuration reset: {key}", "green"))
                else:
                    global_config_manager.reset_all(global_scope=is_global)
                    print(colorize_text("✓ All configuration reset to defaults", "green"))
            
            return 0
            
        except Exception as e:
            error_msg = f"Configuration operation failed: {str(e)}"
            self.logger.error(error_msg)
            print(colorize_text(f"✗ {error_msg}", "red"))
            if self.debug_mode:
                traceback.print_exc()
            return 1
    
    def cmd_version(self, args: Dict[str, Any]) -> int:
        """Show version information."""
        print(f"Recursia v{__version__}")
        print(f"Author: {__author__}")
        print(f"License: {__license__}")
        
        if self.verbose_mode:
            # Show component versions
            try:
                import numpy as np
                print(f"NumPy: {np.__version__}")
            except ImportError:
                pass
            
            try:
                import matplotlib
                print(f"Matplotlib: {matplotlib.__version__}")
            except ImportError:
                pass
            
            try:
                import scipy
                print(f"SciPy: {scipy.__version__}")
            except ImportError:
                pass
        
        return 0
    
    def _save_output(self, content: str, output_path: str) -> None:
        """Save content to output file."""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            if self.logger:
                self.logger.info(f"Output saved to: {output_path}")
        except Exception as e:
            error_msg = f"Failed to save output to {output_path}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            print(colorize_text(f"⚠ {error_msg}", "yellow"))
    
    def run(self, argv: Optional[List[str]] = None) -> int:
        """Main CLI entry point."""
        try:
            # Parse command line arguments
            args = docopt(__doc__, argv=argv, version=f"Recursia v{__version__}")
            
            # Setup logging based on flags
            debug = args.get('--debug', False)
            verbose = args.get('--verbose', False)
            self._setup_logging(debug=debug, verbose=verbose)
            
            # Enable performance profiling if debug mode
            if debug:
                self.performance_enabled = True
                performance_profiler.start_timer("total_execution")
            
            # Load configuration
            config_path = args.get('--config')
            self._load_configuration(config_path)
            
            # Route to appropriate command handler
            if args['run']:
                return self.cmd_run(args)
            elif args['compile']:
                return self.cmd_compile(args)
            elif args['repl']:
                return self.cmd_repl(args)
            elif args['validate-osh']:
                return self.cmd_validate_osh(args)
            elif args['dashboard']:
                print(colorize_text("❌ Dashboard command is deprecated. Please use the frontend dashboard instead.", "red"))
                print(colorize_text("   Run 'cd frontend && npm run dev' to start the React dashboard.", "yellow"))
                return
            elif args['hardware']:
                return self.cmd_hardware(args)
            elif args['report']:
                return self.cmd_report(args)
            elif args['config']:
                return self.cmd_config(args)
            elif args['version']:
                return self.cmd_version(args)
            else:
                print(colorize_text("✗ Unknown command", "red"))
                return 1
                
        except KeyboardInterrupt:
            print(colorize_text("\nOperation interrupted by user", "yellow"))
            return 130
        except Exception as e:
            error_msg = f"CLI error: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(f"Error: {error_msg}", file=sys.stderr)
            
            if self.debug_mode:
                traceback.print_exc()
            return 1
        finally:
            if self.performance_enabled:
                performance_profiler.stop_timer("total_execution")
            self.cleanup()


def main() -> int:
    """Main entry point for the Recursia CLI."""
    cli = RecursiaCLI()
    return cli.run()


if __name__ == '__main__':
    sys.exit(main())