# dashboard.py - Recursia Central Scientific Dashboard Engine
"""
Central visualization and monitoring dashboard for Recursia simulations.
Consumes data from runtime and provides real-time scientific visualization,
OSH metrics monitoring, and comprehensive reporting capabilities.
"""

import threading
import time
import json
import base64
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from collections import deque
import numpy as np
from datetime import datetime
import io
import os

# Import visualization and rendering components
from src.visualization.field_panel import FieldPanel
from src.visualization.observer_panel import ObserverPanel  
from src.visualization.render_utils import render_dashboard
from src.visualization.simulation_panel import SimulationPanel
from src.visualization.quantum_renderer import QuantumRenderer
from src.visualization.coherence_renderer import AdvancedCoherenceRenderer
from src.visualization.render_physics import PhysicsRenderer

from src.visualization.setup_utils import (
    setup_advanced_export_system, setup_real_time_streaming,
    setup_accessibility_features, setup_performance_optimization,
    ExportConfiguration, StreamingConfiguration, AccessibilityConfiguration,
    PerformanceConfiguration
)

# Import data classes and utilities
from src.core.data_classes import (
    OSHMetrics, SystemHealthProfile, ComprehensiveMetrics,
    DashboardConfiguration, VisualizationConfig
)
from src.core.utils import (
    global_error_manager, performance_profiler, 
    visualization_helper, colorize_text
)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class DashboardState:
    """Container for all system references used in the Dashboard."""
    # Core system references (consumed, not created)
    runtime: Optional[Any] = None
    interpreter: Optional[Any] = None
    execution_context: Optional[Any] = None
    
    # Physics subsystem references
    physics_engine: Optional[Any] = None
    field_dynamics: Optional[Any] = None
    memory_field_physics: Optional[Any] = None
    coherence_manager: Optional[Any] = None
    entanglement_manager: Optional[Any] = None
    observer_dynamics: Optional[Any] = None
    recursive_mechanics: Optional[Any] = None
    
    # Quantum subsystem references
    quantum_backend: Optional[Any] = None
    gate_operations: Optional[Any] = None
    measurement_operations: Optional[Any] = None
    state_registry: Optional[Any] = None
    observer_registry: Optional[Any] = None
    
    # Infrastructure references
    event_system: Optional[Any] = None
    memory_manager: Optional[Any] = None
    
    # Visualization components (created by dashboard)
    field_panel: Optional[FieldPanel] = None
    observer_panel: Optional[ObserverPanel] = None
    simulation_panel: Optional[SimulationPanel] = None
    quantum_renderer: Optional[QuantumRenderer] = None
    coherence_renderer: Optional[AdvancedCoherenceRenderer] = None
    physics_renderer: Optional[PhysicsRenderer] = None
    
    # Analytics and reporting
    phenomena_detector: Optional[Any] = None
    report_builder: Optional[Any] = None
    performance_profiler: Optional[Any] = None
    
    # Configuration
    config: Optional[DashboardConfiguration] = None


@dataclass
class DashboardMetrics:
    """Structured real-time OSH telemetry."""
    timestamp: float = field(default_factory=time.time)
    
    # Core OSH Metrics
    coherence: float = 0.0
    entropy: float = 0.0
    strain: float = 0.0
    rsp: float = 0.0
    phi: float = 0.0
    emergence_index: float = 0.0
    consciousness_quotient: float = 0.0
    kolmogorov_complexity: float = 0.0
    information_curvature: float = 0.0
    temporal_stability: float = 0.0
    
    # System counts
    quantum_states_count: int = 0
    observer_count: int = 0
    active_observers: int = 0
    memory_regions: int = 0
    field_count: int = 0
    recursion_depth: int = 0
    
    # Performance metrics
    render_fps: float = 0.0
    update_fps: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    
    # Event counts
    measurement_count: int = 0
    collapse_events: int = 0
    teleportation_events: int = 0
    entanglement_events: int = 0
    
    # Alerts and phenomena
    critical_alerts: List[str] = field(default_factory=list)
    emergent_phenomena: List[str] = field(default_factory=list)
    phenomena_strength: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class Dashboard:
    """
    Central scientific dashboard for Recursia simulations.
    Provides real-time visualization, monitoring, and reporting capabilities.
    """
    
    def __init__(self, dashboard_state: Optional[DashboardState] = None, **kwargs):
        """Initialize dashboard with system references."""
        self.state = dashboard_state or DashboardState()
        
        # Update state with any provided kwargs
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        
        # Configuration
        self.config = self.state.config or DashboardConfiguration()
        
        # Metrics and history
        self.current_metrics = DashboardMetrics()
        self.metrics_history = deque(maxlen=self.config.history_window)
        self.last_update_time = time.time()
        
        # Threading and control
        self._running = False
        self._paused = False
        self._update_thread = None
        self._update_lock = threading.RLock()
        
        # Performance tracking
        self.render_times = deque(maxlen=100)
        self.update_times = deque(maxlen=100)
        self.frame_count = 0
        
        # Export and streaming
        self.export_system = None
        self.streaming_server = None
        self.export_config = ExportConfiguration()
        self.streaming_config = StreamingConfiguration()
        
        # Event tracking
        self.event_counts = {}
        self.last_event_time = time.time()
        
        # Initialize visualization components
        self._initialize_visualization_components()
        
        # Setup export and streaming systems
        self._setup_export_system()
        self._setup_streaming_system()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        logger.info("Dashboard initialized successfully")
    
    def _initialize_visualization_components(self):
        """Initialize visualization panels and renderers."""
        try:
            # Create field panel
            if not self.state.field_panel:
                self.state.field_panel = FieldPanel(
                    field_dynamics=self.state.field_dynamics,
                    memory_field=self.state.memory_field_physics,
                    coherence_manager=self.state.coherence_manager,
                    recursive_mechanics=self.state.recursive_mechanics,
                    quantum_renderer=self.state.quantum_renderer,
                    coherence_renderer=self.state.coherence_renderer,
                    config=self.config.__dict__
                )
            
            # Create observer panel
            if not self.state.observer_panel:
                self.state.observer_panel = ObserverPanel(
                    observer_dynamics=self.state.observer_dynamics,
                    recursive_mechanics=self.state.recursive_mechanics,
                    quantum_renderer=self.state.quantum_renderer,
                    coherence_renderer=self.state.coherence_renderer,
                    event_system=self.state.event_system,
                    coherence_manager=self.state.coherence_manager,
                    entanglement_manager=self.state.entanglement_manager,
                    config=self.config.__dict__
                )
            
            # Create simulation panel
            if not self.state.simulation_panel:
                self.state.simulation_panel = SimulationPanel(
                    interpreter=self.state.interpreter,
                    execution_context=self.state.execution_context,
                    event_system=self.state.event_system,
                    memory_field=self.state.memory_field_physics,
                    recursive_mechanics=self.state.recursive_mechanics,
                    quantum_renderer=self.state.quantum_renderer,
                    coherence_renderer=self.state.coherence_renderer,
                    config=self.config.__dict__
                )
            
            # Create quantum renderer
            if not self.state.quantum_renderer:
                self.state.quantum_renderer = QuantumRenderer(
                    coherence_manager=self.state.coherence_manager,
                    entanglement_manager=self.state.entanglement_manager,
                    event_system=self.state.event_system,
                    state_registry=self.state.state_registry,
                    config=self.config.__dict__
                )
            
            # Create coherence renderer
            if not self.state.coherence_renderer:
                self.state.coherence_renderer = AdvancedCoherenceRenderer(
                    coherence_manager=self.state.coherence_manager,
                    memory_field=self.state.memory_field_physics,
                    recursive_mechanics=self.state.recursive_mechanics,
                    event_system=self.state.event_system,
                    config=self.config.__dict__
                )
            
            # Create physics renderer
            if not self.state.physics_renderer:
                self.state.physics_renderer = PhysicsRenderer(
                    quantum_renderer=self.state.quantum_renderer,
                    field_panel=self.state.field_panel,
                    field_dynamics=self.state.field_dynamics,
                    observer_panel=self.state.observer_panel,
                    observer_dynamics=self.state.observer_dynamics,
                    memory_field=self.state.memory_field_physics,
                    coherence_renderer=self.state.coherence_renderer,
                    entanglement_manager=self.state.entanglement_manager,
                    current_metrics=self.current_metrics,
                    metrics_history=list(self.metrics_history)
                )
            
        except Exception as e:
            logger.error(f"Failed to initialize visualization components: {e}")
            global_error_manager.error(f"Visualization init failed: {e}", "dashboard")
    
    def _setup_export_system(self):
        """Setup export system for reports and data."""
        try:
            self.export_system = setup_advanced_export_system()
            logger.info("Export system initialized")
        except Exception as e:
            logger.warning(f"Export system initialization failed: {e}")
            self.export_system = {}
    
    def _setup_streaming_system(self):
        """Setup real-time streaming system."""
        if self.config.real_time_updates and self.streaming_config.enabled:
            try:
                streaming_result = setup_real_time_streaming(self.streaming_config)
                self.streaming_server = streaming_result.get('server')
                logger.info("Streaming system initialized")
            except Exception as e:
                logger.warning(f"Streaming system initialization failed: {e}")
    
    def _setup_event_handlers(self):
        """Setup event handlers for system monitoring."""
        if not self.state.event_system:
            return
        
        try:
            # Register event handlers
            event_handlers = {
                'state_creation_event': self._handle_state_creation,
                'measurement_event': self._handle_measurement,
                'collapse_event': self._handle_collapse,
                'teleportation_event': self._handle_teleportation,
                'entanglement_creation_event': self._handle_entanglement,
                'emergent_phenomena_event': self._handle_emergent_phenomena,
                'memory_strain_threshold_event': self._handle_memory_strain,
                'observer_consensus_event': self._handle_observer_consensus,
                'coherence_change_event': self._handle_coherence_change,
                'recursive_boundary_event': self._handle_recursive_boundary
            }
            
            for event_type, handler in event_handlers.items():
                self.state.event_system.add_listener(event_type, handler)
            
            logger.info("Event handlers registered")
            
        except Exception as e:
            logger.warning(f"Event handler setup failed: {e}")
    
    def start(self):
        """Start the dashboard update loop."""
        if self._running:
            return
        
        self._running = True
        self._paused = False
        
        if self.config.real_time_updates:
            self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
            self._update_thread.start()
        
        logger.info("Dashboard started")
    
    def stop(self):
        """Stop the dashboard and cleanup resources."""
        self._running = False
        
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=1.0)
        
        self._cleanup_resources()
        logger.info("Dashboard stopped")
    
    def pause(self):
        """Pause dashboard updates."""
        self._paused = True
        logger.info("Dashboard paused")
    
    def resume(self):
        """Resume dashboard updates."""
        self._paused = False
        logger.info("Dashboard resumed")
    
    def _update_loop(self):
        """Main update loop for real-time monitoring."""
        while self._running:
            if not self._paused:
                try:
                    start_time = time.time()
                    
                    # Update metrics from runtime
                    self.update()
                    
                    # Track update performance
                    update_time = time.time() - start_time
                    self.update_times.append(update_time)
                    
                    # Broadcast updates if streaming enabled
                    if self.streaming_server:
                        self._broadcast_update()
                    
                except Exception as e:
                    logger.error(f"Dashboard update loop error: {e}")
                    global_error_manager.error(f"Update loop error: {e}", "dashboard")
            
            time.sleep(self.config.update_interval)
    
    def update(self, simulation_data: Optional[Dict[str, Any]] = None):
        """Update dashboard with latest data from runtime."""
        with self._update_lock:
            try:
                # Calculate OSH metrics from runtime
                self._calculate_osh_metrics()
                
                # Update system counts
                self._update_system_counts()
                
                # Update performance metrics
                self._update_performance_metrics()
                
                # Update event counts
                self._update_event_counts()
                
                # Update panels with simulation data
                self._update_panels(simulation_data)
                
                # Store metrics in history
                self.current_metrics.timestamp = time.time()
                self.metrics_history.append(DashboardMetrics(**asdict(self.current_metrics)))
                
                self.last_update_time = time.time()
                
            except Exception as e:
                logger.error(f"Dashboard update failed: {e}")
                global_error_manager.error(f"Update failed: {e}", "dashboard")
    
    def _calculate_osh_metrics(self):
        """Calculate OSH metrics from runtime subsystems."""
        try:
            # Get metrics from runtime if available
            if self.state.runtime and hasattr(self.state.runtime, 'get_current_metrics'):
                runtime_metrics = self.state.runtime.get_current_metrics()
                if runtime_metrics:
                    self.current_metrics.coherence = getattr(runtime_metrics, 'coherence', 0.0)
                    self.current_metrics.entropy = getattr(runtime_metrics, 'entropy', 0.0)
                    self.current_metrics.strain = getattr(runtime_metrics, 'strain', 0.0)
                    self.current_metrics.rsp = getattr(runtime_metrics, 'rsp', 0.0)
                    self.current_metrics.phi = getattr(runtime_metrics, 'phi', 0.0)
                    self.current_metrics.emergence_index = getattr(runtime_metrics, 'emergence_index', 0.0)
                    self.current_metrics.consciousness_quotient = getattr(runtime_metrics, 'consciousness_quotient', 0.0)
                    self.current_metrics.kolmogorov_complexity = getattr(runtime_metrics, 'kolmogorov_complexity', 0.0)
                    self.current_metrics.information_curvature = getattr(runtime_metrics, 'information_curvature', 0.0)
                    self.current_metrics.temporal_stability = getattr(runtime_metrics, 'temporal_stability', 0.0)
            
            # Calculate temporal stability from history
            if len(self.metrics_history) > 10:
                recent_coherence = [m.coherence for m in list(self.metrics_history)[-10:]]
                coherence_std = np.std(recent_coherence) if recent_coherence else 0.0
                self.current_metrics.temporal_stability = max(0.0, 1.0 - coherence_std)
            
        except Exception as e:
            logger.warning(f"OSH metrics calculation failed: {e}")
    
    def _update_system_counts(self):
        """Update system component counts."""
        try:
            # Quantum states count
            if self.state.state_registry:
                self.current_metrics.quantum_states_count = len(
                    getattr(self.state.state_registry, 'states', {})
                )
            
            # Observer counts
            if self.state.observer_registry:
                observers = getattr(self.state.observer_registry, 'observers', {})
                self.current_metrics.observer_count = len(observers)
                # Count active observers (assuming active phase indicates activity)
                self.current_metrics.active_observers = sum(
                    1 for obs in observers.values() 
                    if getattr(obs, 'phase', 'passive') != 'passive'
                )
            
            # Memory regions
            if self.state.memory_field_physics:
                memory_regions = getattr(self.state.memory_field_physics, 'memory_regions', {})
                self.current_metrics.memory_regions = len(memory_regions)
            
            # Field count
            if self.state.field_dynamics:
                active_fields = getattr(self.state.field_dynamics, 'active_fields', {})
                self.current_metrics.field_count = len(active_fields)
            
            # Recursion depth
            if self.state.recursive_mechanics:
                recursion_levels = getattr(self.state.recursive_mechanics, 'recursion_levels', {})
                self.current_metrics.recursion_depth = max(recursion_levels.values(), default=0)
            
        except Exception as e:
            logger.warning(f"System counts update failed: {e}")
    
    def _update_performance_metrics(self):
        """Update performance metrics."""
        try:
            # Calculate FPS
            current_time = time.time()
            time_window = 5.0  # 5 second window
            
            # Render FPS
            recent_renders = [t for t in self.render_times if current_time - t < time_window]
            self.current_metrics.render_fps = len(recent_renders) / time_window if recent_renders else 0.0
            
            # Update FPS
            recent_updates = [t for t in self.update_times if current_time - t < time_window]
            self.current_metrics.update_fps = len(recent_updates) / time_window if recent_updates else 0.0
            
            # System resource usage (simplified)
            try:
                import psutil
                process = psutil.Process()
                self.current_metrics.cpu_usage = process.cpu_percent()
                memory_info = process.memory_info()
                self.current_metrics.memory_usage = memory_info.rss / (1024 * 1024)  # MB
            except ImportError:
                # Fallback if psutil not available
                self.current_metrics.cpu_usage = 0.0
                self.current_metrics.memory_usage = 0.0
            
        except Exception as e:
            logger.warning(f"Performance metrics update failed: {e}")
    
    def _update_event_counts(self):
        """Update event counters."""
        try:
            # Reset counters
            self.current_metrics.measurement_count = self.event_counts.get('measurement_event', 0)
            self.current_metrics.collapse_events = self.event_counts.get('collapse_event', 0)
            self.current_metrics.teleportation_events = self.event_counts.get('teleportation_event', 0)
            self.current_metrics.entanglement_events = self.event_counts.get('entanglement_creation_event', 0)
            
        except Exception as e:
            logger.warning(f"Event counts update failed: {e}")
    
    def _update_panels(self, simulation_data: Optional[Dict[str, Any]]):
        """Update visualization panels with simulation data."""
        try:
            # Prepare simulation data dict
            if simulation_data is None:
                simulation_data = {
                    'current_metrics': self.current_metrics,
                    'metrics_history': list(self.metrics_history),
                    'timestamp': time.time()
                }
            
            # Update field panel
            if self.state.field_panel:
                self.state.field_panel.update(simulation_data)
            
            # Update observer panel
            if self.state.observer_panel:
                self.state.observer_panel.update(simulation_data)
            
            # Update simulation panel
            if self.state.simulation_panel:
                self.state.simulation_panel.update(simulation_data)
            
        except Exception as e:
            logger.warning(f"Panel update failed: {e}")
    
    def render_dashboard(self, width: int = 1920, height: int = 1080) -> Dict[str, Any]:
        """Render complete dashboard visualization."""
        start_time = time.time()
        
        try:
            # Use render_dashboard from render_utils
            result = render_dashboard(
                width=width,
                height=height,
                quantum_renderer=self.state.quantum_renderer,
                field_panel=self.state.field_panel,
                observer_panel=self.state.observer_panel,
                simulation_panel=self.state.simulation_panel,
                memory_field=self.state.memory_field_physics,
                coherence_renderer=self.state.coherence_renderer,
                entanglement_manager=self.state.entanglement_manager,
                coherence_manager=self.state.coherence_manager,
                current_metrics=self.current_metrics,
                metrics_history=list(self.metrics_history),
                phenomena_detector=self.state.phenomena_detector,
                performance_tracker=self
            )
            
            # Track render performance
            render_time = time.time() - start_time
            self.render_times.append(time.time())
            
            # Add dashboard-specific metadata
            result.update({
                'dashboard_render_time': render_time,
                'dashboard_metrics': self.current_metrics.to_dict(),
                'system_health': self.get_system_health(),
                'dashboard_config': asdict(self.config)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Dashboard render failed: {e}")
            global_error_manager.error(f"Render failed: {e}", "dashboard")
            
            return {
                'success': False,
                'error': str(e),
                'image_data': None,
                'render_time': time.time() - start_time
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health profile."""
        try:
            health_profile = {
                'overall_health': 0.0,
                'component_health': {},
                'performance_metrics': {
                    'render_fps': self.current_metrics.render_fps,
                    'update_fps': self.current_metrics.update_fps,
                    'cpu_usage': self.current_metrics.cpu_usage,
                    'memory_usage': self.current_metrics.memory_usage
                },
                'resource_utilization': {
                    'memory_mb': self.current_metrics.memory_usage,
                    'cpu_percent': self.current_metrics.cpu_usage
                },
                'stability_indicators': {
                    'coherence': self.current_metrics.coherence,
                    'entropy': self.current_metrics.entropy,
                    'strain': self.current_metrics.strain,
                    'temporal_stability': self.current_metrics.temporal_stability
                },
                'alerts': list(self.current_metrics.critical_alerts),
                'recommendations': [],
                'critical_issues': [],
                'health_trend': 'stable',
                'predictive_alerts': [],
                'timestamp': datetime.now()
            }
            
            # Calculate component health scores
            health_profile['component_health'] = {
                'coherence': min(1.0, self.current_metrics.coherence),
                'entropy': max(0.0, 1.0 - self.current_metrics.entropy),
                'strain': max(0.0, 1.0 - self.current_metrics.strain),
                'performance': min(1.0, self.current_metrics.render_fps / 30.0),
                'temporal_stability': self.current_metrics.temporal_stability
            }
            
            # Calculate overall health
            health_values = list(health_profile['component_health'].values())
            health_profile['overall_health'] = np.mean(health_values) if health_values else 0.0
            
            # Generate recommendations
            if self.current_metrics.entropy > 0.7:
                health_profile['recommendations'].append("Consider entropy reduction measures")
            if self.current_metrics.strain > 0.8:
                health_profile['recommendations'].append("High strain detected - apply defragmentation")
            if self.current_metrics.render_fps < 10:
                health_profile['recommendations'].append("Low render performance - optimize visualization")
            
            return health_profile
            
        except Exception as e:
            logger.warning(f"System health calculation failed: {e}")
            return {'overall_health': 0.0, 'error': str(e)}
    
    def export_scientific_report(self, filename: Optional[str] = None, format: str = "pdf") -> Dict[str, Any]:
        """Export comprehensive scientific report."""
        try:
            if not self.export_system:
                return {'success': False, 'error': 'Export system not available'}
            
            # Prepare report data
            report_data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'dashboard_version': '1.0.0',
                    'export_format': format
                },
                'current_metrics': self.current_metrics.to_dict(),
                'metrics_history': [m.to_dict() for m in self.metrics_history],
                'system_health': self.get_system_health(),
                'performance_data': self.get_dashboard_statistics(),
                'configuration': asdict(self.config)
            }
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"recursia_dashboard_report_{timestamp}.{format}"
            
            # Use export system to generate report
            if 'create_scientific_report' in self.export_system:
                result = self.export_system['create_scientific_report'](
                    data=report_data,
                    filename=filename,
                    format=format,
                    report_type='comprehensive'
                )
                return result
            else:
                # Fallback to simple export
                return self._export_simple_report(report_data, filename, format)
            
        except Exception as e:
            logger.error(f"Scientific report export failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _export_simple_report(self, data: Dict[str, Any], filename: str, format: str) -> Dict[str, Any]:
        """Simple fallback export functionality."""
        try:
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return {'success': True, 'filename': filename}
            else:
                # For other formats, just save as JSON
                json_filename = filename.rsplit('.', 1)[0] + '.json'
                with open(json_filename, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                return {'success': True, 'filename': json_filename, 'note': f'Exported as JSON instead of {format}'}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dashboard statistics."""
        return {
            'uptime': time.time() - (self.last_update_time if hasattr(self, 'last_update_time') else time.time()),
            'metrics_history_count': len(self.metrics_history),
            'event_counts': dict(self.event_counts),
            'render_performance': {
                'avg_render_time': np.mean(self.render_times) if self.render_times else 0.0,
                'max_render_time': max(self.render_times) if self.render_times else 0.0,
                'render_fps': self.current_metrics.render_fps
            },
            'update_performance': {
                'avg_update_time': np.mean(self.update_times) if self.update_times else 0.0,
                'max_update_time': max(self.update_times) if self.update_times else 0.0,
                'update_fps': self.current_metrics.update_fps
            },
            'system_status': {
                'running': self._running,
                'paused': self._paused,
                'thread_active': self._update_thread.is_alive() if self._update_thread else False
            }
        }
    
    def _broadcast_update(self):
        """Broadcast update to streaming clients."""
        if not self.streaming_server:
            return
        
        try:
            update_data = {
                'type': 'metrics_update',
                'timestamp': time.time(),
                'metrics': self.current_metrics.to_dict(),
                'system_health': self.get_system_health()
            }
            
            # Broadcast via streaming server
            if hasattr(self.streaming_server, 'broadcast'):
                self.streaming_server.broadcast(json.dumps(update_data, default=str))
            
        except Exception as e:
            logger.warning(f"Broadcast update failed: {e}")
    
    def _cleanup_resources(self):
        """Cleanup dashboard resources."""
        try:
            # Stop streaming server
            if self.streaming_server and hasattr(self.streaming_server, 'shutdown'):
                self.streaming_server.shutdown()
            
            # Cleanup visualization components
            for component in [self.state.field_panel, self.state.observer_panel, 
                            self.state.simulation_panel, self.state.quantum_renderer,
                            self.state.coherence_renderer, self.state.physics_renderer]:
                if component and hasattr(component, 'cleanup'):
                    component.cleanup()
            
            logger.info("Dashboard resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Resource cleanup failed: {e}")
    
    # Event handlers
    def _handle_state_creation(self, event_data: Dict[str, Any]):
        """Handle state creation event."""
        self.event_counts['state_creation_event'] = self.event_counts.get('state_creation_event', 0) + 1
    
    def _handle_measurement(self, event_data: Dict[str, Any]):
        """Handle measurement event."""
        self.event_counts['measurement_event'] = self.event_counts.get('measurement_event', 0) + 1
    
    def _handle_collapse(self, event_data: Dict[str, Any]):
        """Handle collapse event."""
        self.event_counts['collapse_event'] = self.event_counts.get('collapse_event', 0) + 1
    
    def _handle_teleportation(self, event_data: Dict[str, Any]):
        """Handle teleportation event."""
        self.event_counts['teleportation_event'] = self.event_counts.get('teleportation_event', 0) + 1
    
    def _handle_entanglement(self, event_data: Dict[str, Any]):
        """Handle entanglement creation event."""
        self.event_counts['entanglement_creation_event'] = self.event_counts.get('entanglement_creation_event', 0) + 1
    
    def _handle_emergent_phenomena(self, event_data: Dict[str, Any]):
        """Handle emergent phenomena event."""
        phenomena = event_data.get('phenomena', [])
        if isinstance(phenomena, str):
            phenomena = [phenomena]
        
        for phenomenon in phenomena:
            if phenomenon not in self.current_metrics.emergent_phenomena:
                self.current_metrics.emergent_phenomena.append(phenomenon)
        
        # Update phenomena strength
        strength = event_data.get('strength', 0.0)
        if strength > self.current_metrics.phenomena_strength:
            self.current_metrics.phenomena_strength = strength
    
    def _handle_memory_strain(self, event_data: Dict[str, Any]):
        """Handle memory strain threshold event."""
        strain_level = event_data.get('strain', 0.0)
        region = event_data.get('region', 'unknown')
        
        alert = f"Critical memory strain in region {region}: {strain_level:.2f}"
        if alert not in self.current_metrics.critical_alerts:
            self.current_metrics.critical_alerts.append(alert)
    
    def _handle_observer_consensus(self, event_data: Dict[str, Any]):
        """Handle observer consensus event."""
        self.event_counts['observer_consensus_event'] = self.event_counts.get('observer_consensus_event', 0) + 1
    
    def _handle_coherence_change(self, event_data: Dict[str, Any]):
        """Handle coherence change event."""
        self.event_counts['coherence_change_event'] = self.event_counts.get('coherence_change_event', 0) + 1
    
    def _handle_recursive_boundary(self, event_data: Dict[str, Any]):
        """Handle recursive boundary event."""
        self.event_counts['recursive_boundary_event'] = self.event_counts.get('recursive_boundary_event', 0) + 1
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.stop()
        except:
            pass


# Global dashboard instance
_global_dashboard = None


def create_dashboard(dashboard_state: Optional[DashboardState] = None, **kwargs) -> Dashboard:
    """Create a new dashboard instance."""
    return Dashboard(dashboard_state=dashboard_state, **kwargs)


def create_dashboard_with_defaults(**kwargs) -> Dashboard:
    """Create dashboard with default configuration."""
    config = DashboardConfiguration()
    state = DashboardState(config=config)
    return Dashboard(dashboard_state=state, **kwargs)


def set_global_dashboard(dashboard: Dashboard):
    """Set the global dashboard instance."""
    global _global_dashboard
    _global_dashboard = dashboard


def get_global_dashboard() -> Optional[Dashboard]:
    """Get the global dashboard instance."""
    return _global_dashboard


# Export main components
__all__ = [
    'Dashboard',
    'DashboardState', 
    'DashboardMetrics',
    'create_dashboard',
    'create_dashboard_with_defaults',
    'set_global_dashboard',
    'get_global_dashboard'
]