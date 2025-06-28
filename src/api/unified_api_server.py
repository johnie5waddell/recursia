#!/usr/bin/env python3
"""
Unified Recursia Backend API Server
Uses centralized runtime instance for all operations
"""

import asyncio
import json
import logging
import math
import time
# numpy imported at function level for performance
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Core Recursia imports
from src.core.runtime import RecursiaRuntime, get_global_runtime, set_global_runtime, create_optimized_runtime
from src.core.compiler import RecursiaCompiler
from src.core.direct_parser import DirectParser
from src.core.bytecode_vm import RecursiaVM
from src.core.data_classes import VMExecutionResult, OSHMetrics
from src.engines.DynamicUniverseEngine import DynamicUniverseEngine, UNIVERSE_MODES
# OSH calculations now unified in VM - no external calculator needed
# VM calculations are done IN THE VM during execution

# Request/Response Models
class ExecuteRequest(BaseModel):
    """Request model for code execution."""
    code: str = Field(..., description="Recursia code to execute")
    options: Dict[str, Any] = Field(default_factory=dict, description="Execution options")
    iterations: int = Field(default=1, description="Number of iterations to run")

class CompileRequest(BaseModel):
    """Request model for code compilation."""
    code: str = Field(..., description="Recursia code to compile")
    target: str = Field(default="quantum_simulator", description="Compilation target")

class QuantumStateResponse(BaseModel):
    """Response model for quantum state data."""
    states: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())

class MetricsResponse(BaseModel):
    """Response model for metrics data including Theory of Everything."""
    # Core OSH metrics
    rsp: float = Field(default=0.0)
    coherence: float = Field(default=0.0)
    entropy: float = Field(default=1.0)
    information: float = Field(default=0.0)
    
    # Additional metrics
    strain: float = Field(default=0.0)
    phi: float = Field(default=0.0)
    emergence_index: float = Field(default=0.0)
    field_energy: float = Field(default=0.0)
    temporal_stability: float = Field(default=0.5)
    observer_influence: float = Field(default=0.0)
    memory_field_coupling: float = Field(default=0.0)
    
    # System status
    observer_count: int = Field(default=0)
    state_count: int = Field(default=0)
    recursion_depth: int = Field(default=0)
    depth: int = Field(default=0)
    
    # Performance metrics
    fps: float = Field(default=60.0)
    error: float = Field(default=0.001)
    quantum_volume: float = Field(default=0.0)
    
    # UI-specific metrics
    memory_strain: float = Field(default=0.0)
    observer_focus: float = Field(default=0.0)
    focus: float = Field(default=0.0)
    
    # Time derivatives
    drsp_dt: float = Field(default=0.0)
    di_dt: float = Field(default=0.0)
    dc_dt: float = Field(default=0.0)
    de_dt: float = Field(default=0.0)
    acceleration: float = Field(default=0.0)
    
    # Theory of Everything metrics
    gravitational_anomaly: float = Field(default=0.0)
    information_curvature: float = Field(default=0.0)
    consciousness_probability: float = Field(default=0.0)
    consciousness_threshold_exceeded: bool = Field(default=False)
    collapse_probability: float = Field(default=0.0)
    electromagnetic_coupling: float = Field(default=0.0073)
    weak_coupling: float = Field(default=0.03)
    strong_coupling: float = Field(default=1.0)
    gravitational_coupling: float = Field(default=6.67e-11)
    metric_fluctuations: float = Field(default=0.0)
    holographic_entropy: float = Field(default=0.0)
    emergence_scale: float = Field(default=0.0)
    information_density: float = Field(default=0.0)
    complexity_density: float = Field(default=0.0)
    
    # Memory fragments - removed
    # memory_fragments: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Resources
    resources: Dict[str, Any] = Field(default_factory=lambda: {
        "memory": 0.5,
        "cpu": 0.3,
        "gpu": 0.0,
        "healthy": True,
        "throttleLevel": 0.0
    })
    
    # Timestamp
    timestamp: float = Field(default_factory=lambda: time.time())

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomCORSMiddleware(BaseHTTPMiddleware):
    """
    Custom CORS middleware to ensure proper headers on all responses
    """
    async def dispatch(self, request: Request, call_next):
        # Handle preflight OPTIONS requests
        if request.method == "OPTIONS":
            response = Response(content="", status_code=200)
        else:
            response = await call_next(request)
        
        # Add CORS headers to all responses
        origin = request.headers.get("origin", "*")
        
        # List of allowed origins
        allowed_origins = [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://localhost:3001",
            "http://localhost:3002",
            "http://localhost:5174"
        ]
        
        # Check if origin is allowed
        if origin in allowed_origins or origin.startswith("http://localhost:") or origin.startswith("http://127.0.0.1:"):
            response.headers["Access-Control-Allow-Origin"] = origin
        else:
            response.headers["Access-Control-Allow-Origin"] = "http://localhost:5173"
        
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept, Origin"
        response.headers["Access-Control-Expose-Headers"] = "Content-Length, Content-Type"
        response.headers["Access-Control-Max-Age"] = "3600"
        
        return response


class UnifiedAPIServer:
    """
    Unified API server that uses the centralized runtime instance.
    All subsystems are accessed through the runtime, eliminating duplicate instances.
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the unified API server."""
        self.debug = debug
        self.app = self._create_app()
        
        # Use singleton runtime instance
        self.runtime = None
        self.interpreter = None
        self.compiler = None
        self.metrics_calculator = None
        # Initialize VM calculator for Theory of Everything metrics
        # VM calculations are done IN THE VM, not here
        
        # Connection tracking - use thread-safe manager
        from src.api.websocket_manager import get_websocket_manager, WebSocketMessageValidator
        self.websocket_manager = get_websocket_manager()
        self.message_validator = WebSocketMessageValidator()
        self.is_simulation_paused = False  # Track pause state
        
        # Metrics tracking
        self.last_metrics = None
        self._metrics_history = []
        
        self._setup_routes()
        self._setup_exception_handlers()
        
        # Initialize components on startup
        self.app.add_event_handler("startup", self._startup_event)
        self.app.add_event_handler("shutdown", self._shutdown_event)
    
    async def _startup_event(self):
        """Initialize components on server startup."""
        try:
            # Get or create singleton runtime
            self.runtime = get_global_runtime()
            if not self.runtime:
                logger.info("Creating new optimized runtime instance")
                self.runtime = create_optimized_runtime({
                    'use_unified_executor': True,  # Use unified executor for consistent state management
                    'enable_visualization': False,  # API server doesn't need visualization
                    'enable_event_logging': True,
                    'thread_pool_size': 4,
                    'enable_performance_optimizer': True,
                    'parallel_operations_enabled': True,
                    'quantum_operation_cache_size': 1000
                })
                # Set as global runtime
                set_global_runtime(self.runtime)
            else:
                logger.info("Using existing runtime instance")
            
            # Create bytecode interpreter that uses the runtime
            # DirectParser and RecursiaVM already imported at module level
            
            # Define local CompilationResult for DirectParser
            from dataclasses import dataclass
            from typing import Any, List, Optional
            
            @dataclass
            class CompilationResult:
                success: bool
                bytecode_module: Optional[Any] = None
                errors: List[str] = None
                warnings: List[str] = None
                
                def __post_init__(self):
                    if self.errors is None:
                        self.errors = []
                    if self.warnings is None:
                        self.warnings = []
            
            # Create a minimal interpreter wrapper for API compatibility
            class DirectParserWrapper:
                def __init__(self, runtime):
                    self.runtime = runtime
                    self.parser = DirectParser()
                    
                def compile_source(self, source: str):
                    """Compile source using DirectParser."""
                    try:
                        module = self.parser.parse(source)
                        return CompilationResult(
                            success=True,
                            bytecode_module=module,
                            errors=[],
                            warnings=[]
                        )
                    except Exception as e:
                        logger.error(f"Compilation error: {str(e)}")
                        return CompilationResult(
                            success=False,
                            bytecode_module=None,
                            errors=[str(e)],
                            warnings=[]
                        )
                
                def execute_bytecode(self, bytecode_module, runtime, context, timeout=None):
                    """Execute bytecode using RecursiaVM."""
                    self.vm = RecursiaVM(runtime)
                    
                    # Set up measurement callback if we have a websocket connection
                    if hasattr(runtime, '_measurement_log_callback'):
                        # Store the callback in runtime for VM to use
                        runtime._measurement_callback = runtime._measurement_log_callback
                    
                    return self.vm.execute(bytecode_module)
                
                def execute_code(self, compilation_result, runtime, options=None):
                    """Execute compiled bytecode."""
                    start_time = time.time()
                    logger.info(f"[DirectParserWrapper] Starting execute_code")
                    
                    if not compilation_result.success:
                        raise RuntimeError(f"Cannot execute failed compilation: {compilation_result.errors}")
                    
                    logger.info(f"[DirectParserWrapper] Executing bytecode module")
                    
                    # Execute bytecode - VM result contains the output
                    vm_result = self.execute_bytecode(
                        compilation_result.bytecode_module,
                        runtime,
                        runtime.execution_context,  # Use runtime's execution context
                        timeout=options.get('timeout') if options else None
                    )
                    
                    execution_time = time.time() - start_time
                    logger.info(f"[DirectParserWrapper] VM execution completed in {execution_time:.3f}s, success={vm_result.success}")
                    
                    # Get OSHMetrics from execution context
                    if runtime.execution_context and isinstance(runtime.execution_context.current_metrics, OSHMetrics):
                        metrics = runtime.execution_context.current_metrics
                        # Update with VM result values
                        metrics.information_density = vm_result.integrated_information
                        metrics.kolmogorov_complexity = vm_result.kolmogorov_complexity
                        metrics.entanglement_entropy = vm_result.entropy_flux
                        logger.debug(f"[DirectParserWrapper] Setting metrics.rsp from vm_result: {vm_result.recursive_simulation_potential}")
                        metrics.rsp = vm_result.recursive_simulation_potential
                        metrics.phi = vm_result.phi
                        metrics.coherence = vm_result.coherence
                        metrics.memory_strain = vm_result.memory_strain
                        metrics.gravitational_anomaly = vm_result.gravitational_anomaly
                        metrics.conservation_violation = vm_result.conservation_violation
                        
                        # Update derived fields
                        metrics.entropy = 1.0 - vm_result.coherence if vm_result.coherence > 0 else 1.0
                        metrics.strain = vm_result.memory_strain
                        metrics.emergence_index = vm_result.phi / 15.0 if vm_result.phi > 0 else 0.0
                        metrics.temporal_stability = 0.95
                        metrics.information_curvature = 0.001
                        metrics.conservation_law = 1.0 - vm_result.conservation_violation
                        
                        # Update aliases
                        metrics.information = metrics.information_density
                        metrics.integrated_information = metrics.information_density
                        metrics.complexity = metrics.kolmogorov_complexity
                        metrics.depth = metrics.recursive_depth
                        metrics.focus = metrics.observer_focus
                        
                        # Update measurement count
                        if hasattr(runtime, 'measurement_results'):
                            metrics.measurement_count = len(runtime.measurement_results)
                        # Also check VM measurements if available
                        if hasattr(self, 'vm') and hasattr(self.vm, 'measurements'):
                            metrics.measurement_count = len(self.vm.measurements)
                        
                        metrics.timestamp = time.time()
                    else:
                        # Create new OSHMetrics if needed
                        metrics = OSHMetrics(
                            information_density=vm_result.integrated_information,
                            kolmogorov_complexity=vm_result.kolmogorov_complexity,
                            entanglement_entropy=vm_result.entropy_flux,
                            rsp=vm_result.recursive_simulation_potential,
                            phi=vm_result.phi,
                            coherence=vm_result.coherence,
                            memory_strain=vm_result.memory_strain,
                            gravitational_anomaly=vm_result.gravitational_anomaly,
                            conservation_violation=vm_result.conservation_violation,
                            entropy=1.0 - vm_result.coherence if vm_result.coherence > 0 else 1.0,
                            strain=vm_result.memory_strain,
                            emergence_index=vm_result.phi / 15.0 if vm_result.phi > 0 else 0.0,
                            temporal_stability=0.95,
                            information_curvature=0.001,
                            timestamp=time.time()
                        )
                        # Update aliases
                        metrics.information = metrics.information_density
                        metrics.integrated_information = metrics.information_density
                        metrics.complexity = metrics.kolmogorov_complexity
                        metrics.conservation_law = 1.0 - metrics.conservation_violation
                        metrics.depth = metrics.recursive_depth
                        metrics.focus = metrics.observer_focus
                        
                        runtime.execution_context.current_metrics = metrics
                    
                    # Update observer and state counts from runtime
                    if self.runtime:
                        if hasattr(self.runtime, 'observer_registry') and self.runtime.observer_registry:
                            metrics.observer_count = len(self.runtime.observer_registry.observers)
                            metrics.observer_influence = metrics.observer_count * 0.1  # Simple influence calculation
                            
                            # Calculate average observer focus
                            total_focus = 0.0
                            count = 0
                            for observer_name in self.runtime.observer_registry.observers:
                                props = self.runtime.observer_registry.observer_properties.get(observer_name, {})
                                observer_focus = props.get('observer_focus', 0.0)
                                
                                # observer_focus might be a numeric value or a string (target state)
                                if isinstance(observer_focus, (int, float)):
                                    total_focus += observer_focus
                                    count += 1
                                elif props.get('focus_strength', 0.0) > 0:
                                    # Use focus_strength if observer_focus is a target state
                                    total_focus += props.get('focus_strength', 0.5)
                                    count += 1
                            
                            avg_focus = total_focus / count if count > 0 else 0.0
                            metrics.observer_focus = avg_focus
                            metrics.focus = avg_focus
                            
                        if hasattr(self.runtime, 'state_registry') and self.runtime.state_registry:
                            metrics.state_count = len(self.runtime.state_registry.states)
                    
                    # Get measurements directly from VM
                    measurements = []
                    if hasattr(self, 'vm') and hasattr(self.vm, 'measurements'):
                        measurements = self.vm.measurements
                    
                    return {
                        'success': vm_result.success,
                        'output': vm_result.output,  # Get output from VM result
                        'metrics': metrics,
                        'measurements': measurements
                    }
            
            self.interpreter = DirectParserWrapper(self.runtime)
            
            # Create compiler
            self.compiler = RecursiaCompiler()
            
            # All metrics calculations now happen in the VM via UnifiedVMCalculations
            # No external metrics calculator needed - metrics are stored in execution_context.current_metrics
            
            logger.info("Unified API server initialized with central runtime")
            
            # Initialize default metrics in execution context
            self._initialize_default_metrics()
            
            # Initialize default memory regions for visualization
            self._initialize_default_memory_regions()
            
            # Initialize dynamic universe engine
            self.universe_engine = DynamicUniverseEngine(self.runtime, mode="standard")
            self.universe_task = None
            
            # Auto-start and watchdog disabled - causing infinite loops
            # asyncio.create_task(self._auto_start_universe())
            # asyncio.create_task(self._universe_watchdog())
            
            # Start background task for broadcasting metrics
            # Re-enabled with memory leak fixes
            self._metrics_task = asyncio.create_task(self._metrics_update_loop())
            
        except Exception as e:
            logger.error(f"Failed to initialize unified API server: {e}")
            raise
    
    async def _metrics_update_loop(self):
        """Background task to broadcast metrics updates with memory leak prevention."""
        logger.info("Starting metrics update loop with memory leak fixes")
        
        # Track last sent metrics to avoid sending duplicates
        last_sent_metrics = None
        update_count = 0
        
        while True:
            try:
                # Only broadcast if we have runtime (connections not required for universe mode)
                if self.runtime and self.runtime.execution_context:
                    connection_count = await self.websocket_manager.get_connection_count()
                    logger.debug(f"Metrics update loop: {connection_count} active connections")
                    # Get metrics from VM execution context ONLY
                    current_metrics = self.runtime.execution_context.current_metrics
                    
                    # Debug universe metrics every 5 updates for better visibility
                    if update_count % 5 == 0 and self.universe_engine:
                        # Get actual metrics being sent
                        if isinstance(current_metrics, OSHMetrics):
                            logger.info(f"[Metrics Loop] Universe metrics - Running: {self.universe_engine.is_running}, " +
                                       f"Time: {self.universe_engine.universe_time:.2f}, Iter: {self.universe_engine.iteration_count}, " +
                                       f"Phi: {current_metrics.phi:.4f}, RSP: {current_metrics.rsp:.2f}, " +
                                       f"Coherence: {current_metrics.coherence:.4f}, " +
                                       f"Observers: {current_metrics.observer_count}, States: {current_metrics.state_count}")
                    
                    # Ensure we have an OSHMetrics object
                    if not isinstance(current_metrics, OSHMetrics):
                        current_metrics = OSHMetrics()
                        self.runtime.execution_context.current_metrics = current_metrics
                    
                    # Update counts and universe data directly on the OSHMetrics object
                    # Only update counts from registries if in universe mode
                    # In program mode, preserve the counts set during execution
                    if self.runtime and self.universe_engine and self.universe_engine.is_running:
                        # Universe mode - update from registries
                        if hasattr(self.runtime, 'observer_registry') and self.runtime.observer_registry:
                            observer_count = len(self.runtime.observer_registry.observers)
                            current_metrics.observer_count = observer_count
                            # Debug log when observer count changes or is 0 after iterations
                            if update_count % 10 == 0 or (observer_count == 0 and self.universe_engine.iteration_count > 1):
                                logger.info(f"[Metrics Loop] Observer registry count: {observer_count}, " +
                                          f"Iteration: {self.universe_engine.iteration_count}")
                        else:
                            current_metrics.observer_count = 0
                            
                        if hasattr(self.runtime, 'state_registry') and self.runtime.state_registry:
                            current_metrics.state_count = len(self.runtime.state_registry.states)
                        else:
                            current_metrics.state_count = 0
                    # In program mode, counts are preserved from execution results
                            
                        # Update measurement count from runtime and VM
                        measurement_count = 0
                        if hasattr(self.runtime, 'measurement_results'):
                            measurement_count = len(self.runtime.measurement_results)
                        # Also check VM measurements from universe engine
                        if self.universe_engine and hasattr(self.universe_engine, 'vm') and hasattr(self.universe_engine.vm, 'measurements'):
                            measurement_count = max(measurement_count, len(self.universe_engine.vm.measurements))
                        current_metrics.measurement_count = measurement_count
                        
                        # Update entanglement count from execution context
                        if hasattr(self.runtime, 'execution_context') and self.runtime.execution_context:
                            if hasattr(self.runtime.execution_context, 'statistics'):
                                current_metrics.num_entanglements = self.runtime.execution_context.statistics.get('entanglement_count', 0)
                            else:
                                current_metrics.num_entanglements = 0
                        else:
                            current_metrics.num_entanglements = 0
                    
                    # Update universe metrics
                    self._update_universe_metrics(current_metrics)
                    
                    # Convert to dict only for JSON serialization
                    metrics = self._serialize_metrics(current_metrics)
                    
                    # Debug logging for universe mode
                    if self.universe_engine and self.universe_engine.is_running:
                        if update_count % 10 == 0:  # Log every 10 updates
                            logger.info(f"[Metrics Loop] 🌌 Universe Update #{update_count}: " +
                                       f"Time: {metrics.get('universe_time', 0):.2f}, " +
                                       f"Iteration: {metrics.get('iteration_count', 0)}, " +
                                       f"PHI: {metrics.get('phi', 0):.4f}, " +
                                       f"RSP: {metrics.get('rsp', 0):.2f}, " +
                                       f"States: {metrics.get('state_count', 0)}, " +
                                       f"Observers: {metrics.get('observer_count', 0)}")
                    
                    # Only send if metrics have changed significantly
                    should_send = False
                    if last_sent_metrics is None:
                        should_send = True
                    else:
                        # For universe mode, always send updates when universe is running
                        if metrics.get('universe_running', False):
                            should_send = True
                            if update_count % 5 == 0:  # Log every 5th update for better debugging
                                logger.info(f"[Metrics Loop] Sending universe update #{update_count}: " +
                                           f"Time: {metrics.get('universe_time', 0):.2f}, Iter: {metrics.get('iteration_count', 0)}, " +
                                           f"Phi: {metrics.get('phi', 0):.4f}, RSP: {metrics.get('rsp', 0):.2f}, " +
                                           f"Observers: {metrics.get('observer_count', 0)}, States: {metrics.get('state_count', 0)}")
                        else:
                            # Check for significant changes (> 0.01 for key metrics)
                            for key in ['rsp', 'coherence', 'entropy', 'phi', 'universe_time', 'iteration_count', 'measurement_count', 'state_count', 'observer_count']:
                                if key in metrics and key in last_sent_metrics:
                                    # For counters, any change is significant
                                    if key in ['universe_time', 'iteration_count', 'measurement_count', 'state_count', 'observer_count']:
                                        if metrics[key] != last_sent_metrics[key]:
                                            should_send = True
                                            # Debug log count changes
                                            if key in ['state_count', 'observer_count'] and update_count % 10 == 0:
                                                logger.info(f"[Metrics Loop] {key} changed: {last_sent_metrics.get(key, 0)} -> {metrics[key]}")
                                            break
                                    # For other metrics, check threshold
                                    elif abs(metrics[key] - last_sent_metrics[key]) > 0.01:
                                        should_send = True
                                        break
                    
                    if should_send:
                        # Limit memory fragments to prevent large payloads
                        # Memory fragments removed
                        # if 'memory_fragments' in metrics and len(metrics['memory_fragments']) > 10:
                        #     metrics['memory_fragments'] = metrics['memory_fragments'][:10]
                        
                        # Broadcast to all connected clients (only if we have connections)
                        connection_count = await self.websocket_manager.get_connection_count()
                        if connection_count > 0:
                            successful_sends = await self.websocket_manager.broadcast({
                                "type": "metrics_update",
                                "data": metrics
                            })
                            
                            if successful_sends < connection_count:
                                logger.info(f"Metrics broadcast: {successful_sends}/{connection_count} successful")
                        
                        last_sent_metrics = metrics.copy()
                        update_count += 1
                        
                        # Log periodically for debugging
                        if update_count % 100 == 0:
                            logger.info(f"[Metrics Loop] Sent {update_count} updates to {connection_count} clients, " +
                                      f"Last metrics - Phi: {metrics.get('phi', 0):.4f}, RSP: {metrics.get('rsp', 0):.2f}")
                
                # Update every 100ms (10Hz) to match universe evolution rate
                await asyncio.sleep(0.1)
                
                # Periodic garbage collection to prevent memory buildup
                if update_count % 1000 == 0:
                    import gc
                    gc.collect()
                    logger.debug("Performed garbage collection in metrics loop")
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(1.0)  # Longer delay on error
    
    async def _shutdown_event(self):
        """Cleanup on server shutdown."""
        logger.info("Shutting down unified API server")
        
        # Stop universe simulation
        if hasattr(self, 'universe_engine') and self.universe_engine:
            await self.universe_engine.stop()
        
        # Cancel metrics task if running
        if hasattr(self, '_metrics_task') and self._metrics_task:
            self._metrics_task.cancel()
            try:
                await self._metrics_task
            except asyncio.CancelledError:
                pass
        
        # Close all WebSocket connections - handled by websocket_manager
        
        # Don't cleanup runtime - it's shared
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Recursia Unified API",
            description="Unified backend API for Recursia Quantum Programming Language",
            version="3.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Add custom CORS middleware first (before CORSMiddleware)
        app.add_middleware(CustomCORSMiddleware)
        
        # Configure standard CORS middleware as fallback
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[
                "http://localhost:3000",
                "http://localhost:3001", 
                "http://localhost:3002",
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://localhost:5174",
                "*"  # Allow all origins in development
            ],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=["*"],
            max_age=3600
        )
        
        return app
    
    def _setup_routes(self):
        """Setup API routes."""
        
        @self.app.get("/")
        async def root():
            """Root endpoint."""
            return {
                "name": "Recursia Unified API",
                "version": "3.0.0",
                "runtime_active": self.runtime is not None,
                "endpoints": {
                    "execute": "/api/execute",
                    "compile": "/api/compile",
                    "states": "/api/states",
                    "metrics": "/api/metrics",
                    "health": "/api/health",
                    "websocket": "/ws"
                }
            }
        
        @self.app.get("/api/health")
        async def health():
            """Health check endpoint."""
            if not self.runtime:
                raise HTTPException(status_code=503, detail="Runtime not initialized")
            
            return {
                "status": "healthy",
                "runtime_active": True,
                "subsystems": {
                    "state_registry": self.runtime.state_registry is not None,
                    "observer_registry": self.runtime.observer_registry is not None,
                    "physics_engine": self.runtime.physics_engine is not None,
                    "event_system": self.runtime.event_system is not None,
                    "memory_manager": self.runtime.memory_manager is not None
                },
                "connections": await self.websocket_manager.get_connection_count(),
                "uptime": time.time() - self.runtime.start_time if self.runtime else 0
            }
        
        @self.app.post("/api/execute")
        async def execute(request: ExecuteRequest):
            """Execute Recursia code using unified runtime with comprehensive logging."""
            import gc
            import psutil
            import os
            import traceback
            
            # Enhanced memory and performance tracking
            process = psutil.Process(os.getpid())
            execution_start_time = time.time()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            objects_before = len(gc.get_objects())
            
            # Log request ID for tracking
            request_id = f"REQ-{int(time.time()*1000)}"
            logger.info(f"[{request_id}] ====== NEW EXECUTION REQUEST ======")
            logger.info(f"[{request_id}] Code length: {len(request.code)} chars")
            logger.info(f"[{request_id}] Iterations: {request.iterations}")
            logger.info(f"[{request_id}] Options: {request.options}")
            logger.info(f"[{request_id}] Memory before: {mem_before:.2f}MB")
            logger.info(f"[{request_id}] Objects before: {objects_before}")
            logger.debug(f"[{request_id}] Code preview: {request.code[:100]}...")
            
            # Extract timeout from request
            execution_timeout = request.options.get('timeout', 120.0)
            logger.info(f"[{request_id}] Execution timeout: {execution_timeout}s")
            
            if not self.runtime or not self.interpreter:
                logger.error(f"[{request_id}] Runtime or interpreter not initialized")
                raise HTTPException(status_code=503, detail="Runtime not initialized")
            
            try:
                
                # Compile the code first
                logger.info(f"[{request_id}] Starting compilation")
                compilation_start = time.time()
                compilation_result = self.interpreter.compile_source(request.code)
                compilation_time = time.time() - compilation_start
                
                logger.info(f"[{request_id}] Compilation completed in {compilation_time:.3f}s")
                logger.info(f"[{request_id}] Compilation success: {compilation_result.success}")
                
                if not compilation_result.success:
                    logger.error(f"[{request_id}] Compilation failed: {compilation_result.errors}")
                    return {
                        "success": False,
                        "error": "Compilation failed",
                        "errors": compilation_result.errors,
                        "metrics": self._get_default_metrics()
                    }
                
                # Collect results from all iterations
                all_results = []
                all_outputs = []
                all_metrics = []
                all_states = []
                all_observers = []
                all_measurements = []
                execution_times = []
                execution_errors = []  # Track errors for each iteration
                
                logger.info(f"[{request_id}] Starting {request.iterations} iterations")
                
                for iteration in range(request.iterations):
                    iteration_start = time.time()
                    iter_mem_before = process.memory_info().rss / 1024 / 1024
                    iter_objects_before = len(gc.get_objects())
                    
                    logger.info(f"[{request_id}] === ITERATION {iteration + 1}/{request.iterations} ===")
                    logger.info(f"[{request_id}] Iteration memory before: {iter_mem_before:.2f}MB")
                    logger.info(f"[{request_id}] Iteration objects before: {iter_objects_before}")
                    
                    # Reset runtime for each iteration to ensure independence
                    logger.info(f"[{request_id}] Resetting runtime for iteration {iteration + 1}")
                    reset_start = time.time()
                    reset_success = self.runtime.reset_simulation()
                    reset_time = time.time() - reset_start
                    
                    logger.info(f"[{request_id}] Runtime reset completed in {reset_time:.3f}s, success: {reset_success}")
                    if not reset_success:
                        logger.warning(f"[{request_id}] Runtime reset failed for iteration {iteration + 1}")
                    
                    # Force garbage collection after reset
                    gc_start = time.time()
                    collected = gc.collect()
                    gc_time = time.time() - gc_start
                    logger.info(f"[API EXECUTE] Garbage collection: {collected} objects in {gc_time:.3f}s")
                    
                    # Execute using runtime
                    try:
                        logger.info(f"[{request_id}] Executing iteration {iteration + 1}")
                        execution_start = time.time()
                        
                        # Update runtime config with the timeout for UnifiedExecutor
                        if hasattr(self.runtime, 'config') and isinstance(self.runtime.config, dict):
                            self.runtime.config['max_execution_time'] = execution_timeout
                        
                        try:
                            logger.info(f"[{request_id}] Starting asyncio.wait_for with timeout={execution_timeout}s")
                            
                            # Create a future for timeout tracking
                            start_exec = time.time()
                            
                            # Execute bytecode directly
                            result = await asyncio.wait_for(
                                asyncio.to_thread(
                                    self.interpreter.execute_code,
                                    compilation_result,  # Pass the compilation result containing bytecode
                                    self.runtime,
                                    request.options
                                ),
                                timeout=execution_timeout
                            )
                            
                            actual_exec_time = time.time() - start_exec
                            logger.info(f"[{request_id}] Execution completed successfully in {actual_exec_time:.3f}s")
                        except asyncio.TimeoutError:
                            actual_time = time.time() - start_exec
                            error_msg = f"Execution timeout in iteration {iteration + 1} (exceeded {execution_timeout}s after {actual_time:.3f}s)"
                            logger.error(f"[{request_id}] {error_msg}")
                            execution_errors.append(error_msg)
                            
                            # For single iteration, return timeout error immediately
                            if request.iterations == 1:
                                return {
                                    "success": False,
                                    "error": f"Execution timed out after {execution_timeout} seconds",
                                    "errors": [error_msg],
                                    "metrics": self._get_default_metrics()
                                }
                            # For multiple iterations, continue with others
                            continue
                        
                        execution_time = time.time() - execution_start
                        iter_mem_after = process.memory_info().rss / 1024 / 1024
                        iter_objects_after = len(gc.get_objects())
                        iter_memory_growth = iter_mem_after - iter_mem_before
                        iter_object_growth = iter_objects_after - iter_objects_before
                        
                        logger.info(f"[{request_id}] Iteration {iteration + 1} execution completed in {execution_time:.3f}s")
                        logger.info(f"[{request_id}] Iteration memory after: {iter_mem_after:.2f}MB (growth: {iter_memory_growth:.2f}MB)")
                        logger.info(f"[{request_id}] Iteration objects after: {iter_objects_after} (growth: {iter_object_growth})")
                        logger.debug(f"[{request_id}] Iteration {iteration + 1} result: {result.get('success', False) if isinstance(result, dict) else str(result)[:100]}")
                        
                        if iter_memory_growth > 10:
                            logger.warning(f"[API EXECUTE] HIGH MEMORY GROWTH in iteration {iteration + 1}: {iter_memory_growth:.2f}MB")
                    except Exception as exec_error:
                        # Get detailed error information
                        import traceback
                        error_detail = str(exec_error)
                        if not error_detail:
                            error_detail = type(exec_error).__name__
                        
                        # Check if it's a RuntimeError with embedded message
                        if isinstance(exec_error, RuntimeError) and "Execution failed:" in error_detail:
                            # Extract the actual error from RuntimeError
                            error_parts = error_detail.split("Execution failed:", 1)
                            if len(error_parts) > 1:
                                error_detail = error_parts[1].strip()
                        
                        error_msg = f"Execution error in iteration {iteration + 1}: {error_detail}"
                        logger.error(error_msg, exc_info=True)
                        
                        # Get full traceback for debugging
                        tb = traceback.format_exc()
                        logger.error(f"Full traceback:\n{tb}")
                        
                        execution_errors.append(error_msg)
                        
                        # For single iteration, return error immediately
                        if request.iterations == 1:
                            return {
                                "success": False,
                                "error": error_detail,
                                "errors": [error_msg],
                                "metrics": self._get_default_metrics()
                            }
                        # For multiple iterations, continue with others
                        continue
                    
                    # Get states from runtime's state registry
                    post_start = time.time()
                    states = {}
                    if self.runtime.state_registry:
                        states = self.runtime.state_registry.get_all_states()
                    
                    # Get observers from runtime's observer registry  
                    observers = {}
                    if self.runtime.observer_registry:
                        observers = self.runtime.observer_registry.get_all_observers()
                    
                    # Extract output and measurements from result if available
                    output = []
                    measurements = []
                    if isinstance(result, dict):
                        output = result.get('output', [])
                        measurements = result.get('measurements', [])
                    
                    # SINGLE SOURCE OF TRUTH: Use metrics from execution result ONLY
                    # The VM calculates ALL metrics - NO FALLBACK TO RUNTIME METRICS
                    if isinstance(result, dict) and 'metrics' in result:
                        metrics_obj = result['metrics']
                        if isinstance(metrics_obj, OSHMetrics):
                            # We have an OSHMetrics object - counts will be updated in _serialize_metrics
                            self._update_universe_metrics(metrics_obj)
                            metrics = self._serialize_metrics(metrics_obj)
                            logger.info(f"[{request_id}] Using OSHMetrics from VM execution result - RSP: {metrics['rsp']}, " +
                                      f"Observers: {metrics['observer_count']}, States: {metrics['state_count']}, " +
                                      f"Measurements: {metrics['measurement_count']}")
                        else:
                            # Legacy dict format - should not happen with current code
                            metrics = metrics_obj
                            logger.warning(f"[{request_id}] Got dict metrics instead of OSHMetrics object")
                        
                        # Add calculation metadata
                        if hasattr(self.interpreter, 'vm') and hasattr(self.interpreter.vm, 'metrics_engine'):
                            try:
                                # state_name may not be defined - use a default
                                state_name = next(iter(states.keys())) if states else None
                                if state_name:
                                    calc_info = self.interpreter.vm.metrics_engine.calculator.get_calculation_info(state_name)
                                    metrics['calculation_metadata'] = calc_info
                                    metrics['phi_algorithm_threshold'] = self.interpreter.vm.metrics_engine.calculator.get_phi_algorithm_threshold()
                            except Exception as e:
                                logger.warning(f"Could not get calculation metadata: {e}")
                    else:
                        # No fallback - just use default metrics
                        logger.warning(f"[{request_id}] Execution result missing metrics, using defaults")
                        metrics = self._get_default_metrics()
                    
                    post_time = time.time() - post_start
                    logger.info(f"[{request_id}] Post-execution processing took {post_time:.3f}s")
                    
                    # Collect iteration results
                    all_results.append(result)
                    all_outputs.append(output)
                    all_metrics.append(metrics)
                    all_states.append(states)
                    all_observers.append(observers)
                    all_measurements.append(measurements)
                    execution_times.append(time.time() - iteration_start)
                
                # Aggregate results for multiple iterations
                if request.iterations > 1:
                    # Calculate statistical metrics
                    aggregated_metrics = self._aggregate_metrics(all_metrics)
                    
                    # Update execution context metrics with the aggregated results
                    # This ensures the header displays the correct counts in program mode
                    if self.runtime and self.runtime.execution_context:
                        current_metrics = self.runtime.execution_context.current_metrics
                        if isinstance(current_metrics, OSHMetrics):
                            # Update counts from aggregated metrics (uses mean values)
                            current_metrics.state_count = int(aggregated_metrics.get('state_count', 0))
                            current_metrics.observer_count = int(aggregated_metrics.get('observer_count', 0))
                            current_metrics.measurement_count = int(aggregated_metrics.get('measurement_count', 0))
                            logger.info(f"Updated execution context metrics (multi-iteration) - States: {current_metrics.state_count}, " +
                                      f"Observers: {current_metrics.observer_count}, Measurements: {current_metrics.measurement_count}")
                    
                    # Broadcast aggregated metrics
                    async def broadcast_aggregated_metrics():
                        count = await self.websocket_manager.get_connection_count()
                        if count > 0:
                            await self.websocket_manager.broadcast({
                                "type": "metrics_update",
                                "data": aggregated_metrics
                            })
                            logger.info(f"Broadcasting aggregated metrics update to {count} WebSocket clients")
                    
                    asyncio.create_task(broadcast_aggregated_metrics())
                    
                    return {
                        "success": True,
                        "iterations": request.iterations,
                        "results": all_results,
                        "outputs": all_outputs,
                        "metrics": aggregated_metrics,
                        "metrics_per_iteration": all_metrics,
                        "states": all_states[-1] if all_states else {},  # Return last iteration's states
                        "observers": all_observers[-1] if all_observers else {},  # Return last iteration's observers
                        "measurements": all_measurements[-1] if all_measurements else [],  # Return last iteration's measurements
                        "execution_times": execution_times,
                        "total_execution_time": sum(execution_times)
                    }
                else:
                    # Single iteration - return as before
                    if not all_results:
                        # This should not happen as single iteration errors are handled above
                        error_detail = execution_errors[0] if execution_errors else "Execution failed - no results returned"
                        return {
                            "success": False,
                            "error": error_detail,
                            "errors": execution_errors,
                            "metrics": self._get_default_metrics()
                        }
                    
                    # IMPORTANT: Return ONLY the execution result to avoid duplication
                    # The UnifiedExecutor already provides all needed metrics
                    result = all_results[0] if all_results else {}
                    
                    # If we have a proper result from UnifiedExecutor, return it directly
                    # It already contains all metrics in the correct structure
                    if isinstance(result, dict) and 'metrics' in result:
                        logger.info("Returning UnifiedExecutor result directly (single source of truth)")
                        # Just add states and observers at the top level for backward compatibility
                        result['states'] = all_states[0] if all_states else {}
                        result['observers'] = all_observers[0] if all_observers else {}
                        
                        # Ensure metrics are serialized if they're an OSHMetrics object
                        if isinstance(result['metrics'], OSHMetrics):
                            result['metrics'] = self._serialize_metrics(result['metrics'])
                        
                        # Add calculation metadata if available
                        if hasattr(self.interpreter, 'vm') and hasattr(self.interpreter.vm, 'metrics_engine'):
                            # Get the primary state name from states
                            primary_state = None
                            if all_states and all_states[0]:
                                state_names = list(all_states[0].keys())
                                if state_names:
                                    primary_state = state_names[0]
                            
                            if primary_state:
                                calc_info = self.interpreter.vm.metrics_engine.calculator.get_calculation_info(primary_state)
                                result['metrics']['calculation_metadata'] = calc_info
                                result['metrics']['phi_algorithm_threshold'] = self.interpreter.vm.metrics_engine.calculator.get_phi_algorithm_threshold()
                        
                        # Broadcast metrics to all WebSocket clients after successful execution
                        if 'metrics' in result and isinstance(result['metrics'], dict):
                            # Update execution context metrics with the execution results
                            # This ensures the header displays the correct counts in program mode
                            if self.runtime and self.runtime.execution_context:
                                current_metrics = self.runtime.execution_context.current_metrics
                                if isinstance(current_metrics, OSHMetrics):
                                    # Update counts from execution result
                                    current_metrics.state_count = result['metrics'].get('state_count', 0)
                                    current_metrics.observer_count = result['metrics'].get('observer_count', 0)
                                    current_metrics.measurement_count = result['metrics'].get('measurement_count', 0)
                                    logger.info(f"Updated execution context metrics - States: {current_metrics.state_count}, " +
                                              f"Observers: {current_metrics.observer_count}, Measurements: {current_metrics.measurement_count}")
                            
                            # Create task for async broadcast
                            async def broadcast_metrics():
                                count = await self.websocket_manager.get_connection_count()
                                if count > 0:
                                    await self.websocket_manager.broadcast({
                                        "type": "metrics_update",
                                        "data": result['metrics']
                                    })
                                    logger.info(f"Broadcasting metrics update to {count} WebSocket clients")
                            
                            asyncio.create_task(broadcast_metrics())
                        
                        # Ensure everything is JSON serializable
                        return self._ensure_json_serializable(result)
                    else:
                        # Fallback for old-style results
                        logger.warning("Result missing metrics, building response manually")
                        
                        # Build the fallback result
                        fallback_result = {
                            "success": True,
                            "output": all_outputs[0] if all_outputs else [],
                            "metrics": all_metrics[0] if all_metrics else self._get_default_metrics(),
                            "states": all_states[0] if all_states else {},
                            "observers": all_observers[0] if all_observers else {},
                            "measurements": all_measurements[0] if all_measurements else [],
                            "execution_time": execution_times[0] if execution_times else 0
                        }
                        
                        # Update execution context metrics even in fallback case
                        if self.runtime and self.runtime.execution_context:
                            current_metrics = self.runtime.execution_context.current_metrics
                            if isinstance(current_metrics, OSHMetrics) and isinstance(fallback_result['metrics'], dict):
                                # Update counts from fallback metrics
                                current_metrics.state_count = fallback_result['metrics'].get('state_count', 0)
                                current_metrics.observer_count = fallback_result['metrics'].get('observer_count', 0)
                                current_metrics.measurement_count = fallback_result['metrics'].get('measurement_count', 0)
                        
                        # Broadcast fallback metrics
                        if isinstance(fallback_result['metrics'], dict):
                            async def broadcast_fallback_metrics():
                                count = await self.websocket_manager.get_connection_count()
                                if count > 0:
                                    await self.websocket_manager.broadcast({
                                        "type": "metrics_update",
                                        "data": fallback_result['metrics']
                                    })
                                    logger.info(f"Broadcasting fallback metrics update to {count} WebSocket clients")
                            
                            asyncio.create_task(broadcast_fallback_metrics())
                        
                        return self._ensure_json_serializable(fallback_result)
                
            except Exception as e:
                total_time = time.time() - execution_start_time
                logger.error(f"[{request_id}] Execution error after {total_time:.3f}s: {e}")
                return {
                    "success": False,
                    "error": str(e),
                    "metrics": self._get_default_metrics()
                }
            finally:
                total_elapsed = time.time() - execution_start_time
                logger.info(f"[{request_id}] ====== REQUEST COMPLETE in {total_elapsed:.3f}s ======")
                # Enhanced memory cleanup and tracking
                try:
                    # Force runtime reset to clean up states/observers
                    if self.runtime:
                        self.runtime.reset_simulation()
                    
                    # Clear any cached metrics
                    if hasattr(self, 'last_metrics'):
                        self.last_metrics = None
                    
                    # Multiple GC passes to handle circular references
                    for i in range(3):
                        collected = gc.collect()
                        if collected == 0:
                            break
                        logger.debug(f"GC pass {i+1}: collected {collected} objects")
                    
                    # Memory tracking
                    mem_after = process.memory_info().rss / 1024 / 1024  # MB
                    mem_growth = mem_after - mem_before
                    
                    logger.info(f"Memory after execution: {mem_after:.2f}MB (growth: {mem_growth:.2f}MB)")
                    
                    if mem_growth > 50:  # Lowered threshold to 50MB
                        logger.warning(f"MEMORY GROWTH WARNING: Single execution grew memory by {mem_growth:.2f}MB!")
                        
                        # Log detailed state for debugging
                        if self.runtime:
                            if self.runtime.state_registry:
                                state_count = len(self.runtime.state_registry.states)
                                logger.warning(f"Current state count: {state_count}")
                            
                            if self.runtime.observer_registry:
                                observer_count = len(self.runtime.observer_registry.observers)
                                logger.warning(f"Current observer count: {observer_count}")
                            
                            # Log memory manager stats if available
                            if hasattr(self.runtime, 'memory_manager') and self.runtime.memory_manager:
                                mem_usage = self.runtime.memory_manager.get_memory_usage()
                                logger.warning(f"Memory manager stats: {mem_usage}")
                        
                        # Log garbage collection stats
                        gc_stats = gc.get_stats()
                        logger.warning(f"GC stats: {gc_stats}")
                        
                        # Check for potential memory leaks
                        if mem_growth > 100:
                            logger.error(f"SEVERE MEMORY LEAK: {mem_growth:.2f}MB growth detected!")
                            # Consider triggering memory dump or additional diagnostics here
                        
                except Exception as cleanup_error:
                    logger.error(f"Error during memory cleanup: {cleanup_error}")
        
        @self.app.get("/api/states")
        async def get_states():
            """Get quantum states from runtime's state registry."""
            if not self.runtime or not self.runtime.state_registry:
                return {"states": {}}
            
            states = self.runtime.state_registry.get_all_states()
            return {"states": states, "timestamp": time.time()}
        
        @self.app.get("/api/metrics", response_model=None)
        async def get_metrics():
            """Get current metrics from VM execution context."""
            if not self.runtime or not self.runtime.execution_context:
                # Return default OSHMetrics object serialized
                logger.warning("[GET /api/metrics] No runtime or execution context - returning defaults")
                metrics = OSHMetrics()
                self._update_universe_metrics(metrics)
                return self._serialize_metrics(metrics)
            
            # Get metrics directly from execution context (set by VM during execution)
            current_metrics = self.runtime.execution_context.current_metrics
            
            # Handle case where current_metrics might not be initialized
            if not current_metrics or not isinstance(current_metrics, OSHMetrics):
                logger.warning("[GET /api/metrics] No current_metrics in execution context - creating new")
                current_metrics = OSHMetrics()
                self.runtime.execution_context.current_metrics = current_metrics
            
            # Log the actual values before serialization
            logger.info(f"[GET /api/metrics] Current metrics - RSP: {current_metrics.rsp}, Coherence: {current_metrics.coherence}, Universe running: {getattr(current_metrics, 'universe_running', False)}")
            
            # Counts will be updated in _serialize_metrics to ensure consistency
            
            # Update universe metrics directly on the object
            self._update_universe_metrics(current_metrics)
            
            # Serialize OSHMetrics object to dict for JSON response
            return self._serialize_metrics(current_metrics)
        
        @self.app.get("/api/observers")
        async def get_observers():
            """Get observers from runtime's observer registry."""
            if not self.runtime or not self.runtime.observer_registry:
                return {"observers": [], "timestamp": time.time()}
            
            # Get observers from registry
            observers_data = self.runtime.observer_registry.get_all_observers()
            
            # Handle both list and dict formats
            observers_list = []
            if isinstance(observers_data, dict):
                # Convert dict format to list
                for name, observer_data in observers_data.items():
                    observer_info = {
                        "name": name,
                        "type": observer_data.get("type", "unknown"),
                        "focus": observer_data.get("focus", ""),
                        "self_awareness": observer_data.get("self_awareness", 0),
                        "consciousness_probability": observer_data.get("consciousness_probability", 0)
                    }
                    observers_list.append(observer_info)
            elif isinstance(observers_data, list):
                # Already in list format
                observers_list = observers_data
            else:
                logger.warning(f"Unexpected observer data format: {type(observers_data)}")
                observers_list = []
            
            return {"observers": observers_list, "timestamp": time.time()}
        
        @self.app.post("/universe/start")
        async def start_universe(request: Dict[str, Any] = {}):
            """Start the dynamic universe evolution."""
            try:
                mode = request.get('mode', 'standard')
                logger.info(f"Starting universe in {mode} mode")
                
                if self.universe_engine and self.universe_engine.is_running:
                    return {"success": False, "message": "Universe already running"}
                
                # Start the universe
                await self.universe_engine.start()
                
                return {
                    "success": True,
                    "message": f"Universe started in {mode} mode",
                    "mode": mode,
                    "is_running": self.universe_engine.is_running
                }
            except Exception as e:
                logger.error(f"Error starting universe: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.post("/universe/stop")
        async def stop_universe():
            """Stop the dynamic universe evolution."""
            try:
                logger.info("Stopping universe")
                
                if not self.universe_engine or not self.universe_engine.is_running:
                    return {"success": False, "message": "Universe not running"}
                
                # Stop the universe
                await self.universe_engine.stop()
                
                return {
                    "success": True,
                    "message": "Universe stopped",
                    "is_running": self.universe_engine.is_running
                }
            except Exception as e:
                logger.error(f"Error stopping universe: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.get("/universe/status")
        async def universe_status():
            """Get universe status."""
            if not self.universe_engine:
                return {"is_running": False, "message": "Universe engine not initialized"}
            
            return {
                "is_running": self.universe_engine.is_running,
                "is_paused": self.universe_engine.is_paused,
                "mode": self.universe_engine.mode.name if self.universe_engine.mode else "unknown",
                "iteration_count": self.universe_engine.iteration_count,
                "universe_time": self.universe_engine.universe_time,
                "states": len(self.universe_engine.states),
                "observers": len(self.universe_engine.observers),
                "parameters": {
                    "evolution_rate": self.universe_engine.mode.evolution_rate,
                    "chaos_factor": self.universe_engine.mode.chaos_factor,
                    "interaction_strength": self.universe_engine.mode.interaction_strength,
                    "observer_influence": self.universe_engine.mode.observer_influence
                }
            }
        
        @self.app.post("/universe/pause")
        async def pause_universe():
            """Pause the dynamic universe evolution."""
            try:
                logger.info("Pausing universe")
                
                if not self.universe_engine or not self.universe_engine.is_running:
                    return {"success": False, "message": "Universe not running"}
                
                await self.universe_engine.pause()
                
                return {
                    "success": True,
                    "message": "Universe paused",
                    "is_paused": self.universe_engine.is_paused
                }
            except Exception as e:
                logger.error(f"Error pausing universe: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.post("/universe/resume")
        async def resume_universe():
            """Resume the dynamic universe evolution."""
            try:
                logger.info("Resuming universe")
                
                if not self.universe_engine or not self.universe_engine.is_running:
                    return {"success": False, "message": "Universe not running"}
                
                await self.universe_engine.resume()
                
                return {
                    "success": True,
                    "message": "Universe resumed",
                    "is_paused": self.universe_engine.is_paused
                }
            except Exception as e:
                logger.error(f"Error resuming universe: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.post("/universe/parameters")
        async def update_universe_parameters(request: Dict[str, Any]):
            """Update universe simulation parameters."""
            try:
                logger.info(f"Updating universe parameters: {request}")
                
                if not self.universe_engine:
                    return {"success": False, "message": "Universe engine not initialized"}
                
                self.universe_engine.update_parameters(request)
                
                return {
                    "success": True,
                    "message": "Parameters updated",
                    "mode": self.universe_engine.mode.name if self.universe_engine.mode else "unknown"
                }
            except Exception as e:
                logger.error(f"Error updating universe parameters: {e}")
                return {"success": False, "error": str(e)}
        
        @self.app.get("/api/load-program")
        async def load_program(path: str):
            """Load a Recursia program from disk."""
            try:
                # Resolve path relative to project root
                project_root = Path(__file__).parent.parent.parent
                file_path = project_root / path
                
                if not file_path.exists():
                    raise HTTPException(status_code=404, detail=f"Program not found: {path}")
                
                if not file_path.suffix == '.recursia':
                    raise HTTPException(status_code=400, detail="File must have .recursia extension")
                
                content = file_path.read_text()
                return {
                    "content": content,
                    "path": str(path)
                }
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error loading program: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/list-programs")
        async def list_programs():
            """List available Recursia programs dynamically from filesystem."""
            try:
                project_root = Path(__file__).parent.parent.parent
                programs = []
                
                # Search for .recursia files in common locations
                search_dirs = ["quantum_programs", "experiments", "validation_programs"]
                
                for dir_name in search_dirs:
                    dir_path = project_root / dir_name
                    if dir_path.exists():
                        for file_path in dir_path.rglob("*.recursia"):
                            if file_path.is_file():
                                rel_path = file_path.relative_to(project_root)
                                
                                # Determine subcategory from path
                                path_parts = rel_path.parts
                                subcategory = path_parts[1] if len(path_parts) > 2 else None
                                
                                # Extract info from filename and path
                                name = file_path.stem.replace('_', ' ').title()
                                
                                # Determine difficulty based on keywords
                                filename_lower = file_path.stem.lower()
                                if any(word in filename_lower for word in ['simple', 'basic', 'hello', 'test']):
                                    difficulty = 'beginner'
                                elif any(word in filename_lower for word in ['advanced', 'complex', 'expert']):
                                    difficulty = 'advanced'
                                elif any(word in filename_lower for word in ['intermediate', 'demo']):
                                    difficulty = 'intermediate'
                                else:
                                    difficulty = 'intermediate'
                                
                                # Read first few lines to get description
                                description = f"Quantum program: {name}"
                                try:
                                    with open(file_path, 'r') as f:
                                        lines = f.readlines()[:10]
                                        for line in lines:
                                            if line.strip().startswith('#') and not line.strip().startswith('##'):
                                                desc_line = line.strip('#').strip()
                                                if desc_line and len(desc_line) > 5:
                                                    description = desc_line
                                                    break
                                except:
                                    pass
                                
                                # Generate tags from name and path
                                tags = []
                                if subcategory:
                                    tags.append(subcategory)
                                tags.extend([w for w in filename_lower.split('_') if len(w) > 2])
                                
                                programs.append({
                                    "id": f"{dir_name}-{file_path.stem}",
                                    "path": str(rel_path),
                                    "name": name,
                                    "category": dir_name,
                                    "subcategory": subcategory,
                                    "difficulty": difficulty,
                                    "description": description,
                                    "tags": tags,
                                    "status": "working"  # Assume all existing files work
                                })
                
                # Sort by category, then name
                programs.sort(key=lambda p: (p['category'], p['name']))
                
                return {"programs": programs, "count": len(programs)}
            except Exception as e:
                logger.error(f"Error listing programs: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/compile")
        async def compile_code(request: CompileRequest):
            """Compile Recursia code."""
            if not self.compiler:
                raise HTTPException(status_code=503, detail="Compiler not initialized")
            
            try:
                result = self.compiler.compile(request.code, request.target)
                return {
                    "success": True,
                    "compiled_code": result,
                    "target": request.target
                }
            except Exception as e:
                logger.error(f"Compilation error: {e}")
                return {
                    "success": False,
                    "error": str(e)
                }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket for real-time updates from runtime."""
            logger.info(f"[WebSocket] New connection attempt from {websocket.client}")
            try:
                await websocket.accept()
                logger.info(f"[WebSocket] Connection accepted")
            except Exception as e:
                logger.error(f"[WebSocket] Failed to accept connection: {e}")
                raise
                
            connection_id = await self.websocket_manager.connect(websocket)
            
            logger.info(f"[WebSocket] Connection {connection_id} established successfully")
            
            try:
                # Send initial state
                await websocket.send_json({
                    "type": "connection",
                    "data": {
                        "id": connection_id,
                        "message": "Connected to Recursia Unified API"
                    }
                })
                
                # Listen for messages and send updates
                while True:
                    # Receive message
                    try:
                        # Use shorter timeout for universe mode to enable faster updates
                        timeout = 0.1 if (self.universe_engine and self.universe_engine.is_running) else 1.0
                        message = await asyncio.wait_for(
                            websocket.receive_text(),
                            timeout=timeout
                        )
                        data = self.message_validator.validate_message(message)
                        if data is None:
                            await websocket.send_json({
                                "type": "error",
                                "data": {"message": "Invalid message format"}
                            })
                            continue
                        logger.info(f"[WebSocket] Received message: {data.get('type', 'unknown')}")
                        
                        # Handle different message types
                        if data.get("type") == "ping":
                            await websocket.send_json({"type": "pong"})
                        
                        elif data.get("type") == "get_metrics":
                            # Get current metrics from execution context or create default
                            if self.runtime and self.runtime.execution_context:
                                current_metrics = self.runtime.execution_context.current_metrics
                                if not isinstance(current_metrics, OSHMetrics):
                                    current_metrics = OSHMetrics()
                                    self.runtime.execution_context.current_metrics = current_metrics
                                # Update counts and universe data
                                if self.runtime.observer_registry:
                                    current_metrics.observer_count = len(self.runtime.observer_registry.observers)
                                if self.runtime.state_registry:
                                    current_metrics.state_count = len(self.runtime.state_registry.states)
                                if hasattr(self.runtime, 'measurement_results'):
                                    current_metrics.measurement_count = len(self.runtime.measurement_results)
                                self._update_universe_metrics(current_metrics)
                                # Serialize only for JSON transmission
                                metrics = self._serialize_metrics(current_metrics)
                            else:
                                metrics = self._get_default_metrics()
                            await websocket.send_json({
                                "type": "metrics",
                                "data": metrics
                            })
                        
                        elif data.get("type") == "get_states":
                            states = {}
                            if self.runtime and self.runtime.state_registry:
                                states = self.runtime.state_registry.get_all_states()
                            await websocket.send_json({
                                "type": "states",
                                "data": states
                            })
                        
                        elif data.get("type") == "pause_simulation":
                            self.is_simulation_paused = True
                            logger.info("Simulation paused via WebSocket")
                            
                            # Also pause universe if running
                            if self.universe_engine and self.universe_engine.is_running:
                                await self.universe_engine.pause()
                                logger.info("Universe paused via WebSocket")
                            
                            # Broadcast pause to all connections
                            await self.websocket_manager.broadcast({
                                "type": "simulation_paused",
                                "data": {"paused": True}
                            })
                        
                        elif data.get("type") == "resume_simulation":
                            self.is_simulation_paused = False
                            logger.info("Simulation resumed via WebSocket")
                            
                            # Also resume universe if running
                            if self.universe_engine and self.universe_engine.is_running:
                                await self.universe_engine.resume()
                                logger.info("Universe resumed via WebSocket")
                            
                            # Broadcast resume to all connections
                            await self.websocket_manager.broadcast({
                                "type": "simulation_resumed",
                                "data": {"paused": False}
                            })
                        
                        elif data.get("type") == "update_universe_params":
                            params = data.get("data", {}).get("params", {})
                            logger.info(f"Updating universe parameters via WebSocket: {params}")
                            
                            if self.universe_engine:
                                try:
                                    self.universe_engine.update_parameters(params)
                                    await websocket.send_json({
                                        "type": "universe_params_updated",
                                        "data": {"success": True, "params": params}
                                    })
                                except Exception as e:
                                    logger.error(f"Error updating universe parameters: {e}")
                                    await websocket.send_json({
                                        "type": "error",
                                        "data": {"message": f"Failed to update parameters: {str(e)}"}
                                    })
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "data": {"message": "Universe engine not available"}
                                })
                        
                        elif data.get("type") == "seek_simulation":
                            seek_time = data.get("time", 0)
                            logger.info(f"Simulation seek to {seek_time} via WebSocket")
                            # This would require implementing state restoration
                            # For now, just acknowledge the request
                            await websocket.send_json({
                                "type": "seek_acknowledged",
                                "data": {"time": seek_time}
                            })
                        
                        elif data.get("type") == "start_universe":
                            mode = data.get("data", {}).get("mode", "standard")
                            
                            # Call start_universe_simulation and wait for it
                            await self.start_universe_simulation(mode)
                            
                            # Wait for universe to actually start running
                            if self.universe_engine:
                                # Wait up to 2 seconds for universe to start
                                start_wait_time = 0
                                while not self.universe_engine.is_running and start_wait_time < 2.0:
                                    await asyncio.sleep(0.1)
                                    start_wait_time += 0.1
                                
                                # Wait for at least one iteration to complete
                                if self.universe_engine.is_running:
                                    iteration_wait_time = 0
                                    while self.universe_engine.iteration_count == 0 and iteration_wait_time < 1.0:
                                        await asyncio.sleep(0.1)
                                        iteration_wait_time += 0.1
                                
                                # Now send the metrics update
                                metrics = self._get_current_metrics()
                                await websocket.send_json({
                                    "type": "metrics_update",
                                    "data": metrics
                                })
                                
                                # Also send a specific universe_started message
                                await websocket.send_json({
                                    "type": "universe_started",
                                    "data": {
                                        "mode": mode,
                                        "running": self.universe_engine.is_running,
                                        "iteration_count": self.universe_engine.iteration_count,
                                        "universe_time": self.universe_engine.universe_time
                                    }
                                })
                            
                        elif data.get("type") == "stop_universe":
                            await self.stop_universe_simulation()
                        
                        elif data.get("type") == "get_universe_stats":
                            stats = self.universe_engine.get_evolution_stats() if self.universe_engine else {}
                            await websocket.send_json({
                                "type": "universe_stats",
                                "data": stats
                            })
                        
                        elif data.get("type") == "set_universe_mode":
                            mode = data.get("data", {}).get("mode", "standard")
                            logger.info(f"Setting universe mode to: {mode}")
                            if self.universe_engine:
                                try:
                                    self.universe_engine.set_mode(mode)
                                    await websocket.send_json({
                                        "type": "universe_mode_changed",
                                        "data": {"mode": mode, "success": True}
                                    })
                                except Exception as e:
                                    logger.error(f"Error setting universe mode: {e}")
                                    await websocket.send_json({
                                        "type": "error",
                                        "data": {"message": f"Failed to set universe mode: {str(e)}"}
                                    })
                            else:
                                await websocket.send_json({
                                    "type": "error",
                                    "data": {"message": "Universe engine not available"}
                                })
                            
                    except asyncio.TimeoutError:
                        # Send periodic updates for universe simulation
                        if not hasattr(websocket, '_last_update_time'):
                            websocket._last_update_time = 0
                        
                        current_time = time.time()
                        
                        # Determine update frequency based on mode
                        universe_running = self.universe_engine and self.universe_engine.is_running
                        
                        # Log universe state periodically for debugging
                        if current_time % 5 < 0.1:  # Every 5 seconds
                            logger.debug(f"[WebSocket] Periodic check - Universe running: {universe_running}, " +
                                       f"iteration: {self.universe_engine.iteration_count if self.universe_engine else 0}")
                        
                        if universe_running:
                            # Universe mode: high frequency updates (10 updates/second)
                            update_interval = 0.1  # 100ms
                        else:
                            # Program mode: lower frequency (1 update/second)
                            update_interval = 1.0
                        
                        # Check if it's time to update
                        if current_time - websocket._last_update_time >= update_interval and self.runtime and not self.is_simulation_paused:
                            websocket._last_update_time = current_time
                            # Get current metrics from execution context
                            if self.runtime.execution_context:
                                current_metrics = self.runtime.execution_context.current_metrics
                                if isinstance(current_metrics, OSHMetrics):
                                    # CRITICAL: Update timestamp to current time for fresh metrics
                                    current_metrics.timestamp = current_time
                                    
                                    # Update counts directly before serialization
                                    observer_count = 0
                                    state_count = 0
                                    
                                    if self.runtime.observer_registry:
                                        observer_count = len(self.runtime.observer_registry.observers)
                                    if self.runtime.state_registry:
                                        state_count = len(self.runtime.state_registry.states)
                                    
                                    # CRITICAL: If universe is running but has no states/observers, create them
                                    if universe_running and (state_count == 0 or observer_count == 0):
                                        logger.error(f"[WebSocket] Universe has no states/observers! States: {state_count}, Observers: {observer_count}")
                                        # Force creation by creating minimal quantum system
                                        if state_count == 0 and self.runtime.state_registry:
                                            try:
                                                # Create a universe state
                                                self.runtime.state_registry.create_state('universe', {
                                                    'state_qubits': 8,
                                                    'state_coherence': 0.95,
                                                    'state_entropy': 0.05
                                                })
                                                state_count = 1
                                                logger.info("[WebSocket] Created universe state")
                                            except Exception as e:
                                                logger.error(f"[WebSocket] Failed to create universe state: {e}")
                                        
                                        if observer_count == 0 and self.runtime.observer_registry:
                                            try:
                                                # Create a primary observer
                                                self.runtime.observer_registry.create_observer('primary_observer', {
                                                    'observer_type': 'conscious',
                                                    'observer_focus': 0.8,
                                                    'observer_self_awareness': 0.95
                                                })
                                                observer_count = 1
                                                logger.info("[WebSocket] Created primary observer")
                                            except Exception as e:
                                                logger.error(f"[WebSocket] Failed to create observer: {e}")
                                    
                                    current_metrics.observer_count = observer_count
                                    current_metrics.state_count = state_count
                                    
                                    # Count measurements from both runtime and VM
                                    measurement_count = 0
                                    if hasattr(self.runtime, 'measurement_results'):
                                        measurement_count = len(self.runtime.measurement_results)
                                    # Also check VM measurements
                                    if self.universe_engine and hasattr(self.universe_engine, 'vm') and hasattr(self.universe_engine.vm, 'measurements'):
                                        measurement_count = max(measurement_count, len(self.universe_engine.vm.measurements))
                                    current_metrics.measurement_count = measurement_count
                                    
                                    self._update_universe_metrics(current_metrics)
                                    # Serialize only for JSON transmission
                                    metrics = self._serialize_metrics(current_metrics)
                                    
                                    # CRITICAL: Ensure universe evolution data is included
                                    if universe_running and self.universe_engine:
                                        metrics['universe_time'] = self.universe_engine.universe_time
                                        metrics['iteration_count'] = self.universe_engine.iteration_count
                                        metrics['universe_running'] = True
                                    # Debug log the actual values being sent
                                    elif current_time % 5 < update_interval:  # Log every 5 seconds for program mode
                                        logger.debug(f"[WebSocket] Program metrics - observers: {metrics.get('observer_count', 0)}, states: {metrics.get('state_count', 0)}")
                            else:
                                # No current metrics in execution context, build from registries
                                logger.warning("[WebSocket] ⚠️ No current metrics in execution context, building default metrics")
                                current_metrics = OSHMetrics()
                                # Ensure we have proper VM calculations if available
                                if self.universe_engine and hasattr(self.universe_engine, 'vm') and self.universe_engine.vm:
                                    try:
                                        # Try to compute metrics directly if we have states
                                        if self.runtime and self.runtime.state_registry and len(self.runtime.state_registry.states) > 0:
                                            primary_state = list(self.runtime.state_registry.states.keys())[0]
                                            vm_metrics = self.universe_engine.vm._compute_metrics_directly(primary_state, None)
                                            if vm_metrics:
                                                current_metrics.phi = vm_metrics.get('phi', 0.0)
                                                current_metrics.rsp = vm_metrics.get('rsp', 0.0)
                                                current_metrics.integrated_information = vm_metrics.get('integrated_information', 0.0)
                                                current_metrics.information_density = vm_metrics.get('integrated_information', 0.0)
                                                logger.info(f"[WebSocket] Computed metrics from VM - phi={current_metrics.phi}, rsp={current_metrics.rsp}")
                                    except Exception as e:
                                        logger.error(f"[WebSocket] Failed to compute metrics from VM: {e}")
                                
                                # Update universe data
                                self._update_universe_metrics(current_metrics)
                                metrics = self._serialize_metrics(current_metrics)
                                
                                # CRITICAL: Ensure universe evolution data is included
                                if universe_running and self.universe_engine:
                                    metrics['universe_time'] = self.universe_engine.universe_time
                                    metrics['iteration_count'] = self.universe_engine.iteration_count
                                    metrics['universe_running'] = True
                            # Add pause state to metrics
                            metrics['is_paused'] = self.is_simulation_paused
                            
                            # Only send if we have valid metrics
                            if metrics and isinstance(metrics, dict):
                                await websocket.send_json({
                                    "type": "metrics_update",
                                    "data": metrics
                                })
                                
                                # Log first few updates for universe mode to debug
                                if universe_running and self.universe_engine.iteration_count < 5:
                                    logger.info(f"[WebSocket] Sent universe update #{self.universe_engine.iteration_count}")
                            else:
                                logger.warning("[WebSocket] Skipping update - invalid metrics")
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket connection {connection_id} disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                await self.websocket_manager.disconnect(connection_id)
    
    # _broadcast_message removed - using websocket_manager.broadcast() instead
    
    
    
    def _get_resource_metrics(self) -> Dict[str, Any]:
        """Get current resource usage metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            # Get memory usage
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Get CPU usage (non-blocking)
            cpu_percent = process.cpu_percent(interval=0)
            
            # Determine health status
            healthy = memory_percent < 80 and cpu_percent < 90
            throttle_level = max(0, min(1, (memory_percent - 50) / 50))
            
            return {
                "memory": memory_percent / 100.0,  # Convert to 0-1 range
                "cpu": cpu_percent / 100.0,        # Convert to 0-1 range
                "gpu": 0.0,                        # GPU monitoring not implemented
                "healthy": healthy,
                "throttleLevel": throttle_level
            }
        except:
            # Fallback if psutil not available or error
            return {
                "memory": 0.5,
                "cpu": 0.3,
                "gpu": 0.0,
                "healthy": True,
                "throttleLevel": 0.0
            }
    
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics from execution context or build from registries."""
        if self.runtime and self.runtime.execution_context and self.runtime.execution_context.current_metrics:
            current_metrics = self.runtime.execution_context.current_metrics
            if isinstance(current_metrics, OSHMetrics):
                self._update_universe_metrics(current_metrics)
                return self._serialize_metrics(current_metrics)
        
        # Fallback to default metrics
        return self._get_default_metrics()
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Get default metrics structure including Theory of Everything."""
        return {
            # Core OSH metrics
            "rsp": 0.0,
            "coherence": 0.0,
            "entropy": 1.0,
            "information": 0.0,
            "strain": 0.0,
            "phi": 0.0,
            "emergence_index": 0.0,
            "field_energy": 0.0,
            "temporal_stability": 0.5,
            "observer_influence": 0.0,
            "memory_field_coupling": 0.0,
            
            # Calculation metadata
            "calculation_metadata": {
                "phi_algorithm": "unknown",
                "phi_algorithm_description": "No calculation performed",
                "algorithm_stats": {},
                "is_approximation": False
            },
            "phi_algorithm_threshold": 8,
            
            # System status
            "observer_count": 0,
            "state_count": 0,
            "recursion_depth": 0,
            "depth": 0,
            "error": 0.001,
            "fps": 60.0,
            "quantum_volume": 0.0,
            "memory_strain": 0.0,
            "observer_focus": 0.0,
            "focus": 0.0,
            
            # Time derivatives
            "drsp_dt": 0.0,
            "di_dt": 0.0,
            "dc_dt": 0.0,
            "de_dt": 0.0,
            "acceleration": 0.0,
            
            # Theory of Everything metrics
            "gravitational_anomaly": 0.0,
            "information_curvature": 0.0,
            "consciousness_probability": 0.0,
            "consciousness_threshold_exceeded": False,
            "collapse_probability": 0.0,
            "electromagnetic_coupling": 0.0073,  # Fine structure constant
            "weak_coupling": 0.03,
            "strong_coupling": 1.0,
            "gravitational_coupling": 6.67e-11,
            "metric_fluctuations": 0.0,
            "holographic_entropy": 0.0,
            "emergence_scale": 0.0,
            "information_density": 0.0,
            "complexity_density": 0.0,
            
            # Additional metrics
            # "memory_fragments": [],  # Removed
            "timestamp": time.time(),
            
            # Resources info
            "resources": self._get_resource_metrics()
        }
    
            
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple iterations."""
        if not metrics_list:
            return self._get_default_metrics()
        
        # Metrics to aggregate
        metric_keys = ['rsp', 'coherence', 'entropy', 'information', 'strain', 'phi', 
                      'emergence_index', 'field_energy', 'temporal_stability',
                      'measurement_count', 'state_count', 'observer_count',
                      'observer_influence', 'memory_field_coupling', 'error',
                      'quantum_volume', 'drsp_dt', 'di_dt', 'dc_dt', 'de_dt', 
                      'acceleration']
        
        aggregated = {
            'iterations': len(metrics_list),
            'timestamp': time.time()
        }
        
        # Calculate mean, std, min, max for each metric
        for key in metric_keys:
            values = [m.get(key, 0) for m in metrics_list if key in m]
            if values:
                import numpy as np
                aggregated[f'{key}_mean'] = float(np.mean(values))
                aggregated[f'{key}_std'] = float(np.std(values)) if len(values) > 1 else 0.0
                aggregated[f'{key}_min'] = float(np.min(values))
                aggregated[f'{key}_max'] = float(np.max(values))
                # Also include the mean as the main value for backwards compatibility
                aggregated[key] = aggregated[f'{key}_mean']
        
        # Add any non-numeric fields from the last iteration
        last_metrics = metrics_list[-1] if metrics_list else {}
        # aggregated['memory_fragments'] = last_metrics.get('memory_fragments', [])  # Removed
        aggregated['fps'] = last_metrics.get('fps', 60.0)
        aggregated['depth'] = last_metrics.get('depth', 0)
        aggregated['recursion_depth'] = last_metrics.get('recursion_depth', 0)
        aggregated['memory_strain'] = aggregated.get('strain', 0.0)
        aggregated['observer_focus'] = aggregated.get('observer_influence', 0.0)
        aggregated['focus'] = aggregated.get('observer_influence', 0.0)
        
        return self._ensure_json_serializable(aggregated)
    
    def _calculate_average_observer_focus(self) -> float:
        """Calculate average observer focus from all observers."""
        if not self.runtime or not hasattr(self.runtime, 'observer_registry') or not self.runtime.observer_registry:
            return 0.0
        
        total_focus = 0.0
        count = 0
        
        # Get all observer properties
        for observer_name in self.runtime.observer_registry.observers:
            props = self.runtime.observer_registry.observer_properties.get(observer_name, {})
            observer_focus = props.get('observer_focus', 0.0)
            
            # observer_focus might be a numeric value or a string (target state)
            # If numeric, use it; if string, use the focus_strength instead
            if isinstance(observer_focus, (int, float)):
                total_focus += observer_focus
                count += 1
            elif props.get('focus_strength', 0.0) > 0:
                # Use focus_strength if observer_focus is a target state
                total_focus += props.get('focus_strength', 0.5)
                count += 1
        
        # Return average focus, or 0 if no observers
        return total_focus / count if count > 0 else 0.0
    
    def _update_universe_metrics(self, metrics: OSHMetrics) -> None:
        """Update OSHMetrics object with dynamic universe data."""
        if self.universe_engine:
            metrics.universe_time = self.universe_engine.universe_time
            metrics.iteration_count = self.universe_engine.iteration_count
            # Get entanglement count from execution context
            if self.runtime.execution_context and hasattr(self.runtime.execution_context, 'statistics'):
                metrics.num_entanglements = self.runtime.execution_context.statistics.get('entanglement_count', 0)
            else:
                metrics.num_entanglements = 0
            metrics.universe_mode = self.universe_engine.mode.name
            metrics.universe_running = self.universe_engine.is_running
            
        else:
            # Default values when no universe engine
            logger.warning("[_update_universe_metrics] No universe engine available!")
            metrics.universe_time = 0.0
            metrics.iteration_count = 0
            metrics.num_entanglements = 0
            metrics.universe_mode = 'standard'
            metrics.universe_running = False
        
        # Update aliases and computed fields
        # Only update if the target field is not already set
        if metrics.depth == 0 and metrics.recursive_depth > 0:
            metrics.depth = metrics.recursive_depth
        if metrics.focus == 0 and metrics.observer_focus > 0:
            metrics.focus = metrics.observer_focus
        if metrics.complexity == 0 and metrics.kolmogorov_complexity > 0:
            metrics.complexity = metrics.kolmogorov_complexity
        # Don't overwrite integrated_information if it's already set!
        if metrics.information == 0 and metrics.information_density > 0:
            metrics.information = metrics.information_density
        if metrics.emergence_scale == 0 and metrics.emergence_index > 0:
            metrics.emergence_scale = metrics.emergence_index
        if metrics.complexity_density == 0 and metrics.kolmogorov_complexity > 0:
            metrics.complexity_density = metrics.kolmogorov_complexity
        if metrics.holographic_entropy == 0 and metrics.entanglement_entropy > 0:
            metrics.holographic_entropy = metrics.entanglement_entropy
        
        # Calculate consciousness probability
        if metrics.phi > 0:
            metrics.consciousness_probability = metrics.phi / 15.0
            metrics.consciousness_threshold_exceeded = metrics.phi > 5.0
        
        # Set resources if not already set
        if metrics.resources is None:
            metrics.resources = self._get_resource_metrics()
    
    def _serialize_metrics(self, metrics: OSHMetrics) -> Dict[str, Any]:
        """Serialize OSHMetrics object to dict for JSON transmission.
        
        This is the ONLY place where we convert OSHMetrics to dict,
        and only for JSON serialization over WebSocket/HTTP.
        """
        # Always update counts from registries
        if self.runtime:
            if hasattr(self.runtime, 'observer_registry') and self.runtime.observer_registry:
                observer_count = len(self.runtime.observer_registry.observers)
                metrics.observer_count = observer_count
                # Debug log every so often
                if hasattr(self, '_serialize_call_count'):
                    self._serialize_call_count += 1
                else:
                    self._serialize_call_count = 1
                if self._serialize_call_count % 10 == 0:  # Log every 10th call
                    logger.info(f"[_serialize_metrics] Call #{self._serialize_call_count} - " +
                              f"Phi: {metrics.phi:.4f}, RSP: {metrics.rsp:.2f}, " +
                              f"Observer count: {observer_count}, State count: {metrics.state_count}, " +
                              f"Registry sizes - observers: {len(self.runtime.observer_registry.observers) if self.runtime.observer_registry else 0}, " +
                              f"states: {len(self.runtime.state_registry.states) if self.runtime.state_registry else 0}")
            else:
                metrics.observer_count = 0
                
            if hasattr(self.runtime, 'state_registry') and self.runtime.state_registry:
                state_count = len(self.runtime.state_registry.states)
                metrics.state_count = state_count
            else:
                metrics.state_count = 0
                
            if hasattr(self.runtime, 'measurement_results'):
                metrics.measurement_count = len(self.runtime.measurement_results)
            else:
                metrics.measurement_count = 0
            
            # Update entanglement count from execution context statistics
            if hasattr(self.runtime, 'execution_context') and self.runtime.execution_context:
                if hasattr(self.runtime.execution_context, 'statistics'):
                    metrics.num_entanglements = self.runtime.execution_context.statistics.get('entanglement_count', 0)
                else:
                    metrics.num_entanglements = 0
            else:
                metrics.num_entanglements = 0
            
            # Populate memory fragments from memory field physics
            if hasattr(self.runtime, 'memory_field_physics') and self.runtime.memory_field_physics:
                try:
                    # Only update memory fragments periodically to reduce load
                    current_time = time.time()
                    if not hasattr(self, '_last_memory_fragment_update'):
                        self._last_memory_fragment_update = 0
                    
                    # Memory fragments removed
                    # if current_time - self._last_memory_fragment_update > 2.0:
                    #     memory_fragments = self.runtime.memory_field_physics.get_memory_fragments()
                    #     if memory_fragments:
                    #         metrics.memory_fragments = memory_fragments[:8]  # Reduced limit to 8 fragments
                    #         self._last_memory_fragment_update = current_time
                    #     else:
                    #         metrics.memory_fragments = []
                    # else:
                    #     # Keep existing fragments between updates
                    #     metrics.memory_fragments = getattr(metrics, 'memory_fragments', [])
                except Exception as e:
                    logger.warning(f"[_serialize_metrics] Failed to get memory fragments: {e}")
        
        # Calculate derivatives from recent metrics
        if hasattr(self.runtime, 'execution_context') and hasattr(self.runtime.execution_context, 'current_metrics'):
            current_metrics = self.runtime.execution_context.current_metrics
            if isinstance(current_metrics, OSHMetrics):
                # Simple derivative calculation using stored previous values
                if not hasattr(self, '_previous_metrics'):
                    self._previous_metrics = {
                        'rsp': current_metrics.rsp,
                        'information': current_metrics.information_density,
                        'coherence': current_metrics.coherence,
                        'entropy': current_metrics.entropy,
                        'timestamp': time.time()
                    }
                
                # Calculate time delta
                current_time = time.time()
                dt = current_time - self._previous_metrics['timestamp']
                
                if dt > 0.01:  # Only calculate if enough time has passed
                    # Calculate derivatives
                    metrics.drsp_dt = (current_metrics.rsp - self._previous_metrics['rsp']) / dt
                    metrics.di_dt = (current_metrics.information_density - self._previous_metrics['information']) / dt
                    metrics.dc_dt = (current_metrics.coherence - self._previous_metrics['coherence']) / dt
                    metrics.de_dt = (current_metrics.entropy - self._previous_metrics['entropy']) / dt
                    
                    # Calculate acceleration (second derivative of RSP)
                    if hasattr(self, '_previous_drsp_dt'):
                        metrics.acceleration = (metrics.drsp_dt - self._previous_drsp_dt) / dt
                        self._previous_drsp_dt = metrics.drsp_dt
                    else:
                        self._previous_drsp_dt = metrics.drsp_dt
                    
                    # Update previous values
                    self._previous_metrics = {
                        'rsp': current_metrics.rsp,
                        'information': current_metrics.information_density,
                        'coherence': current_metrics.coherence,
                        'entropy': current_metrics.entropy,
                        'timestamp': current_time
                    }
        
        # Calculate derivatives if we have history
            if len(self._metrics_history) > 0:
                prev_metrics = self._metrics_history[-1]
                current_time = time.time()
                dt = current_time - prev_metrics.get('timestamp', current_time)
                if dt > 0.01:  # Only calculate if enough time has passed
                    metrics.drsp_dt = (metrics.rsp - prev_metrics.get('rsp', 0)) / dt
                    metrics.di_dt = (metrics.integrated_information - prev_metrics.get('integrated_information', 0)) / dt
                    metrics.dc_dt = (metrics.kolmogorov_complexity - prev_metrics.get('kolmogorov_complexity', 1)) / dt
                    metrics.de_dt = (metrics.entropy_flux - prev_metrics.get('entropy_flux', 0)) / dt
                    
                    # Log derivative calculations periodically
                    if self._serialize_call_count % 20 == 0:
                        logger.info(f"[Derivatives] dt={dt:.3f}s, drsp_dt={metrics.drsp_dt:.6f}, " +
                                   f"di_dt={metrics.di_dt:.6f}, dc_dt={metrics.dc_dt:.6f}, de_dt={metrics.de_dt:.6f}")
                    
                    if len(self._metrics_history) > 1:
                        prev_prev_metrics = self._metrics_history[-2]
                        prev_drsp_dt = (prev_metrics.get('rsp', 0) - prev_prev_metrics.get('rsp', 0)) / dt
                        metrics.acceleration = (metrics.drsp_dt - prev_drsp_dt) / dt
            
            # Store current metrics for next calculation
            self._metrics_history.append({
                'rsp': metrics.rsp,
                'integrated_information': metrics.integrated_information,
                'kolmogorov_complexity': metrics.kolmogorov_complexity,
                'entropy_flux': metrics.entropy_flux,
                'timestamp': time.time()  # Always use current time
            })
            
            # Keep only last 10 entries
            if len(self._metrics_history) > 10:
                self._metrics_history = self._metrics_history[-10:]
        else:
            self._metrics_history = []
        
        result = {
            # Core OSH metrics
            'rsp': metrics.rsp,
            'coherence': metrics.coherence,
            'entropy': metrics.entropy,
            'information': metrics.information_density,
            
            # Additional metrics
            'strain': metrics.strain,
            'phi': metrics.phi,
            'emergence_index': metrics.emergence_index,
            'field_energy': metrics.field_energy,
            'temporal_stability': metrics.temporal_stability,
            'observer_influence': metrics.observer_influence,
            'memory_field_coupling': metrics.memory_field_coupling,
            
            # Dynamic universe fields
            'universe_time': metrics.universe_time,
            'iteration_count': metrics.iteration_count,
            'num_entanglements': metrics.num_entanglements,
            'universe_mode': metrics.universe_mode,
            'universe_running': metrics.universe_running,
            
            # System metrics
            'observer_count': metrics.observer_count,
            'state_count': metrics.state_count,
            'recursion_depth': metrics.recursive_depth,
            'measurement_count': metrics.measurement_count,
            
            # Performance metrics
            'fps': metrics.fps,
            'error': metrics.error,
            'quantum_volume': metrics.quantum_volume,
            
            # UI-specific metrics and aliases
            'observer_focus': metrics.observer_focus,
            'focus': metrics.focus,
            'depth': metrics.depth,
            'memory_strain': metrics.memory_strain,
            
            # Advanced metrics
            'information_curvature': metrics.information_curvature,
            'integrated_information': metrics.integrated_information,
            'complexity': metrics.complexity,
            'entropy_flux': metrics.entropy_flux,
            'conservation_law': 1.0 - metrics.conservation_violation,
            
            # Time derivatives - group them for frontend
            'derivatives': {
                'drsp_dt': metrics.drsp_dt,
                'di_dt': metrics.di_dt,
                'dc_dt': metrics.dc_dt,
                'de_dt': metrics.de_dt,
                'acceleration': metrics.acceleration
            },
            # Also include them at top level for backward compatibility
            'drsp_dt': metrics.drsp_dt,
            'di_dt': metrics.di_dt,
            'dc_dt': metrics.dc_dt,
            'de_dt': metrics.de_dt,
            'acceleration': metrics.acceleration,
            
            # Theory of Everything metrics
            'gravitational_anomaly': metrics.gravitational_anomaly,
            'consciousness_probability': metrics.consciousness_probability,
            'consciousness_threshold_exceeded': metrics.consciousness_threshold_exceeded,
            'collapse_probability': metrics.collapse_probability,
            'electromagnetic_coupling': metrics.electromagnetic_coupling,
            'weak_coupling': metrics.weak_coupling,
            'strong_coupling': metrics.strong_coupling,
            'metric_fluctuations': metrics.metric_fluctuations,
            'holographic_entropy': metrics.holographic_entropy,
            'emergence_scale': metrics.emergence_scale,
            'complexity_density': metrics.complexity_density,
            
            # Resources and metadata
            'resources': metrics.resources or self._get_resource_metrics(),
            # 'memory_fragments': metrics.memory_fragments[:10] if metrics.memory_fragments else [],  # Removed
            'timestamp': time.time()  # Always use current time for WebSocket updates
        }
        
        # Ensure all values are JSON serializable (handle numpy types)
        return self._ensure_json_serializable(result)
    
    def _ensure_json_serializable(self, obj: Any) -> Any:
        """Ensure object is JSON serializable."""
        if isinstance(obj, dict):
            return {k: self._ensure_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_json_serializable(v) for v in obj]
        # Check for numpy types
        try:
            import numpy as np
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        # Handle OSHMetrics object
        if hasattr(obj, '__dict__') and obj.__class__.__name__ == 'OSHMetrics':
            return self._serialize_metrics(obj)
        return obj
    
    def _initialize_default_metrics(self):
        """Initialize execution context with default non-zero metrics."""
        try:
            if self.runtime and self.runtime.execution_context:
                # Create default OSHMetrics with realistic initial values
                from src.core.data_classes import OSHMetrics
                default_metrics = OSHMetrics(
                    information_density=1.0,
                    kolmogorov_complexity=10.0,
                    entanglement_entropy=0.1,
                    rsp=1.0,  # Start with RSP of 1
                    phi=0.5,
                    coherence=0.95,
                    memory_strain=0.05,
                    entropy=0.05,
                    strain=0.05,
                    emergence_index=0.1,
                    information_curvature=0.001,
                    temporal_stability=0.95,
                    consciousness_field=0.1,
                    observer_influence=0.05,
                    recursive_depth=1,
                    memory_field_coupling=0.1
                )
                self.runtime.execution_context.current_metrics = default_metrics
                logger.info("Initialized default metrics in execution context")
        except Exception as e:
            logger.error(f"Failed to initialize default metrics: {e}")
    
    def _initialize_default_memory_regions(self):
        """Initialize default memory regions for visualization."""
        try:
            if self.runtime and hasattr(self.runtime, 'memory_field_physics') and self.runtime.memory_field_physics:
                logger.info("Initializing default memory regions")
                
                # Create some default memory regions with different characteristics
                default_regions = [
                    ("core_memory", 0.1, 0.95, 0.05),  # Low strain, high coherence
                    ("working_memory", 0.3, 0.8, 0.15),  # Medium strain, good coherence
                    ("quantum_buffer", 0.2, 0.9, 0.1),   # Low-medium strain, high coherence
                    ("observer_cache", 0.25, 0.85, 0.12), # Medium strain, good coherence
                    ("entanglement_store", 0.15, 0.92, 0.08), # Low strain, very high coherence
                ]
                
                for region_name, strain, coherence, entropy in default_regions:
                    try:
                        self.runtime.memory_field_physics.register_memory_region(
                            region_name,
                            initial_strain=strain,
                            initial_coherence=coherence,
                            initial_entropy=entropy,
                            metadata={
                                'type': 'default',
                                'created_at': time.time(),
                                'description': f'Default {region_name} region'
                            }
                        )
                        logger.info(f"Created default memory region: {region_name}")
                    except Exception as e:
                        logger.warning(f"Failed to create memory region {region_name}: {e}")
                
                # Connect some regions to create a network
                try:
                    self.runtime.memory_field_physics.connect_regions("core_memory", "working_memory", 0.7)
                    self.runtime.memory_field_physics.connect_regions("working_memory", "quantum_buffer", 0.5)
                    self.runtime.memory_field_physics.connect_regions("quantum_buffer", "observer_cache", 0.6)
                    self.runtime.memory_field_physics.connect_regions("observer_cache", "entanglement_store", 0.4)
                    self.runtime.memory_field_physics.connect_regions("core_memory", "entanglement_store", 0.3)
                    logger.info("Connected default memory regions")
                except Exception as e:
                    logger.warning(f"Failed to connect memory regions: {e}")
                    
                logger.info("Default memory regions initialized successfully")
            else:
                logger.info("Memory field physics not available - skipping default memory regions")
        except Exception as e:
            logger.error(f"Failed to initialize default memory regions: {e}")
    
    
    def _setup_exception_handlers(self):
        """Setup exception handlers."""
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions."""
            logger.error(f"Unhandled exception: {exc}", exc_info=True)
            
            error_response = {
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if self.debug else "An unexpected error occurred",
                "type": type(exc).__name__,
                "path": str(request.url.path)
            }
            
            return JSONResponse(
                status_code=500,
                content=error_response
            )
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions with proper CORS headers."""
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )
        
        @self.app.exception_handler(RequestValidationError)
        async def validation_exception_handler(request: Request, exc: RequestValidationError):
            """Handle validation errors."""
            return JSONResponse(
                status_code=422,
                content={"detail": exc.errors(), "body": exc.body}
            )
    
    async def _auto_start_universe(self):
        """Auto-start universe simulation for demonstration purposes."""
        try:
            # Wait for server to be fully initialized
            await asyncio.sleep(3.0)
            
            # Check if universe is already running
            if self.universe_engine and self.universe_engine.is_running:
                logger.info("[AUTO-START] Universe simulation already running")
                return
            
            # Start the universe simulation directly without going through the WebSocket handler
            logger.info("[AUTO-START] Starting universe simulation...")
            
            # Direct universe start without the WebSocket layer
            if self.universe_engine:
                await self.universe_engine.start()
                logger.info("[AUTO-START] ✅ Universe simulation started successfully")
                
                # Verify it's running
                await asyncio.sleep(0.5)
                if self.universe_engine.is_running:
                    logger.info(f"[AUTO-START] ✅ Universe confirmed running - mode: {self.universe_engine.mode.name}")
                else:
                    logger.warning("[AUTO-START] ❌ Universe failed to start")
            else:
                logger.error("[AUTO-START] ❌ Universe engine not available")
                
        except Exception as e:
            logger.error(f"[AUTO-START] Failed to auto-start universe simulation: {e}")
            import traceback
            traceback.print_exc()

    async def _universe_watchdog(self):
        """Monitor universe simulation and restart if it stops unexpectedly."""
        logger.info("[WATCHDOG] Universe watchdog started")
        await asyncio.sleep(5.0)  # Wait for initial startup
        
        last_iteration_count = 0
        stall_count = 0
        
        while True:
            try:
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
                if not self.universe_engine:
                    continue
                
                current_iterations = self.universe_engine.iteration_count
                is_running = self.universe_engine.is_running
                
                # Check if universe stopped running
                if not is_running and current_iterations > 0:
                    logger.warning(f"[WATCHDOG] Universe stopped after {current_iterations} iterations - restarting")
                    await self.start_universe_simulation("standard")
                    last_iteration_count = 0
                    stall_count = 0
                    continue
                
                # Check if universe is stalled (no progress)
                if is_running:
                    if current_iterations == last_iteration_count:
                        stall_count += 1
                        logger.warning(f"[WATCHDOG] Universe stalled at iteration {current_iterations} (stall count: {stall_count})")
                        
                        if stall_count >= 3:  # 30 seconds of no progress
                            logger.error(f"[WATCHDOG] Universe completely stalled - restarting")
                            await self.start_universe_simulation("standard")
                            stall_count = 0
                    else:
                        stall_count = 0
                        logger.debug(f"[WATCHDOG] Universe healthy - iteration {current_iterations}")
                    
                    last_iteration_count = current_iterations
                
            except Exception as e:
                logger.error(f"[WATCHDOG] Error in universe watchdog: {e}")

    async def start_universe_simulation(self, mode: str = "standard"):
        """Start the dynamic universe simulation."""
        try:
            logger.info(f"[API] start_universe_simulation called with mode: {mode}")
            logger.info(f"[API] Universe engine exists: {self.universe_engine is not None}")
            
            # Check if universe is already running in the same mode
            if (self.universe_engine and self.universe_engine.is_running and 
                self.universe_engine.mode.name.lower().replace(' ', '_') == mode):
                logger.info(f"[API] Universe already running in {mode} mode, skipping restart")
                return
            
            # Stop existing universe if running
            if self.universe_task and not self.universe_task.done():
                logger.info(f"[API] Stopping existing universe before starting {mode} mode")
                await self.universe_engine.stop()
                self.universe_task.cancel()
                try:
                    await self.universe_task
                except asyncio.CancelledError:
                    pass
            
            # Set new mode if specified
            if mode in UNIVERSE_MODES:
                logger.info(f"[API] Setting universe mode to: {mode}")
                self.universe_engine.set_mode(mode)
            
            # Start universe evolution
            logger.info(f"[API] Starting universe evolution in {mode} mode")
            logger.info(f"[API] Universe engine is_running before start: {self.universe_engine.is_running}")
            await self.universe_engine.start()
            logger.info(f"[API] Universe engine is_running after start: {self.universe_engine.is_running}")
            logger.info(f"[API] Universe engine evolution_task: {self.universe_engine.evolution_task}")
            
            logger.info(f"[API] Universe simulation started successfully in {mode} mode")
            
            # Give the universe engine a moment to execute its first iteration
            # This ensures states and observers are created before we report back
            await asyncio.sleep(0.2)
            
            # Critical check - verify universe is actually running
            logger.info(f"[API] After sleep - Universe engine is_running: {self.universe_engine.is_running}")
            logger.info(f"[API] After sleep - Universe iteration count: {self.universe_engine.iteration_count}")
            
            # Force a metrics update to ensure frontend gets the state
            if self.runtime and self.runtime.execution_context:
                current_metrics = self.runtime.execution_context.current_metrics
                if isinstance(current_metrics, OSHMetrics):
                    self._update_universe_metrics(current_metrics)
                    logger.info(f"[API] Forced metrics update - universe_running in metrics: {current_metrics.universe_running}")
            
            # Broadcast universe start to all connections
            await self.websocket_manager.broadcast({
                "type": "universe_started",
                "data": {
                    "mode": mode,
                    "modes_available": list(UNIVERSE_MODES.keys()),
                    "stats": self.universe_engine.get_evolution_stats()
                }
            })
            
            
        except Exception as e:
            logger.error(f"Error starting universe simulation: {e}", exc_info=True)
    
    async def stop_universe_simulation(self):
        """Stop the dynamic universe simulation."""
        try:
            if self.universe_engine:
                await self.universe_engine.stop()
                logger.info("Stopped dynamic universe simulation")
                
                # Broadcast universe stop
                await self.websocket_manager.broadcast({
                    "type": "universe_stopped",
                    "data": {"stopped": True}
                })
        except Exception as e:
            logger.error(f"Error stopping universe simulation: {e}")
    
    async def _update_loop(self):
        """Placeholder for the old update loop - now using _metrics_update_loop instead."""
        # This function is no longer used but kept for backward compatibility
        # All metrics updates are handled by _metrics_update_loop
        pass
    
    # _broadcast_message has been removed - using websocket_manager.broadcast() instead
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the unified API server."""
        logger.info(f"Starting Unified Recursia API server on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)




# Create module-level app instance for uvicorn
# This allows running with: uvicorn src.api.unified_api_server:app
_server_instance = UnifiedAPIServer(debug=False)
app = _server_instance.app

if __name__ == "__main__":
    import os
    import sys
    
    # Get port from environment variable or command line
    port = 8080  # DEFAULT PORT IS 8080
    
    # Check environment variable first (set by startup script)
    if 'BACKEND_PORT' in os.environ:
        try:
            port = int(os.environ['BACKEND_PORT'])
            logger.info(f"Using port from BACKEND_PORT environment variable: {port}")
        except ValueError:
            logger.warning(f"Invalid BACKEND_PORT value: {os.environ['BACKEND_PORT']}, using default")
    
    # Check command line arguments
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
            logger.info(f"Using port from command line argument: {port}")
        except ValueError:
            logger.warning(f"Invalid port argument: {sys.argv[1]}, using default")
    
    # Create and run server with explicit timeout configuration
    server = UnifiedAPIServer(debug=True)
    
    # Configure uvicorn with proper timeouts to prevent 2-minute cutoff
    import uvicorn
    config = uvicorn.Config(
        app=server.app,
        host="0.0.0.0",
        port=port,
        # Increase timeout to prevent 2-minute cutoff
        timeout_keep_alive=300,  # 5 minutes keep-alive
        # Add other timeout configurations
        ws_ping_timeout=60,  # WebSocket ping timeout
        ws_ping_interval=30,  # WebSocket ping interval
        # Prevent automatic worker restarts
        reload=False,
        # Increase limits - set to None for unlimited in development
        limit_max_requests=None,
        # Log level
        log_level="info"
    )
    server_instance = uvicorn.Server(config)
    server_instance.run()