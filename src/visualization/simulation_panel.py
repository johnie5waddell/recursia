"""
Recursia Simulation Control & OSH Monitoring Panel

This module implements the SimulationPanel class, providing comprehensive
simulation control, monitoring, and visualization capabilities aligned with
the Organic Simulation Hypothesis (OSH) framework.

Features:
- Real-time simulation control (start, pause, stop, step)
- OSH metrics visualization (coherence, entropy, strain, RSP)
- Memory usage and performance monitoring
- Event logging and analysis
- Recursive boundary visualization
- System health diagnostics
- Scientific-grade charting and export

Author: Johnie Waddell
"""

import time
import logging
import threading
from collections import deque
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import base64
import io
import json
import traceback

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats, signal
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Set scientific visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

@dataclass
class SimulationMetrics:
    """Container for simulation performance and state metrics"""
    simulation_time: float = 0.0
    real_time: float = 0.0
    memory_usage: float = 0.0
    average_coherence: float = 0.0
    average_entropy: float = 0.0
    average_strain: float = 0.0
    observer_interactions: int = 0
    field_strain: float = 0.0
    recursive_depth: int = 0
    rsp: float = 0.0
    consciousness_quotient: float = 0.0
    phenomena_count: int = 0
    boundary_crossings: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class OSHFieldMetrics:
    """OSH-specific field calculations and metrics"""
    coherence_field: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)))
    entropy_field: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)))
    strain_field: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)))
    rsp_field: np.ndarray = field(default_factory=lambda: np.zeros((10, 10)))
    information_curvature: float = 0.0
    field_energy: float = 0.0
    emergence_index: float = 0.0
    critical_regions: List[Tuple[int, int]] = field(default_factory=list)

class SimulationPanel:
    """
    Advanced simulation control and monitoring panel with OSH integration.
    
    Provides comprehensive visualization and control of Recursia simulations
    including real-time metrics, OSH field analysis, and scientific diagnostics.
    """
    
    def __init__(self, 
                 interpreter=None,
                 execution_context=None,
                 event_system=None,
                 memory_field=None,
                 recursive_mechanics=None,
                 quantum_renderer=None,
                 coherence_renderer=None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the simulation panel with integrated subsystems.
        
        Args:
            interpreter: RecursiaInterpreter instance
            execution_context: ExecutionContext for simulation state
            event_system: EventSystem for real-time events
            memory_field: MemoryFieldPhysics for field dynamics
            recursive_mechanics: RecursiveMechanics for recursion analysis
            quantum_renderer: QuantumRenderer for quantum visualizations
            coherence_renderer: CoherenceRenderer for OSH visualizations
            config: Configuration dictionary
        """
        self.interpreter = interpreter
        self.execution_context = execution_context
        self.event_system = event_system
        self.memory_field = memory_field
        self.recursive_mechanics = recursive_mechanics
        self.quantum_renderer = quantum_renderer
        self.coherence_renderer = coherence_renderer
        
        # Configuration
        self.config = config or {}
        self.max_series_length = self.config.get('max_series_length', 1000)
        self.update_interval = self.config.get('update_interval', 0.1)
        self.scientific_mode = self.config.get('scientific_mode', True)
        self.enable_caching = self.config.get('enable_caching', True)
        
        # Visualization state
        self.current_visualization = "simulation_overview"
        self.visualization_cache = {} if self.enable_caching else None
        
        # Time series data storage
        self.time_series = {
            'simulation_time': deque(maxlen=self.max_series_length),
            'real_time': deque(maxlen=self.max_series_length),
            'memory_usage': deque(maxlen=self.max_series_length),
            'average_coherence': deque(maxlen=self.max_series_length),
            'average_entropy': deque(maxlen=self.max_series_length),
            'average_strain': deque(maxlen=self.max_series_length),
            'observer_interactions': deque(maxlen=self.max_series_length),
            'field_strain': deque(maxlen=self.max_series_length),
            'recursive_depth': deque(maxlen=self.max_series_length),
            'rsp': deque(maxlen=self.max_series_length),
            'consciousness_quotient': deque(maxlen=self.max_series_length),
            'phenomena_count': deque(maxlen=self.max_series_length),
            'boundary_crossings': deque(maxlen=self.max_series_length),
            'timestamps': deque(maxlen=self.max_series_length)
        }
        
        # Event tracking
        self.event_log = deque(maxlen=1000)
        self.event_categories = {
            'quantum': ['state_creation', 'measurement', 'entanglement', 'teleportation'],
            'observer': ['observation', 'phase_change', 'consensus'],
            'memory': ['strain_threshold', 'defragmentation', 'coherence_wave'],
            'recursive': ['boundary_crossing', 'depth_change', 'system_creation'],
            'simulation': ['start', 'pause', 'stop', 'step', 'error']
        }
        
        # OSH metrics tracking
        self.current_osh_metrics = OSHFieldMetrics()
        self.osh_history = deque(maxlen=self.max_series_length)
        
        # Performance tracking
        self.performance_metrics = {
            'render_times': deque(maxlen=100),
            'update_times': deque(maxlen=100),
            'frame_rate': 0.0,
            'avg_render_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize visualization components
        self._initialize_components()
        
        # Register event listeners
        self._register_event_listeners()
    
    def _initialize_components(self):
        """Initialize visualization components and themes"""
        # OSH color scheme
        self.osh_colors = {
            'coherence': '#00D4FF',    # Cyan
            'entropy': '#FF6B35',      # Orange-red
            'strain': '#FF1744',       # Red
            'rsp': '#76FF03',          # Bright green
            'consciousness': '#E91E63', # Pink
            'emergence': '#9C27B0',    # Purple
            'background': '#0D1117',   # Dark background
            'text': '#F0F6FC',         # Light text
            'grid': '#21262D'          # Grid color
        }
        
        # Scientific colormaps
        self.scientific_cmaps = {
            'coherence': 'viridis',
            'entropy': 'plasma',
            'strain': 'hot',
            'rsp': 'coolwarm',
            'emergence': 'magma'
        }
        
        # Matplotlib theme setup
        plt.rcParams.update({
            'figure.facecolor': self.osh_colors['background'],
            'axes.facecolor': self.osh_colors['background'],
            'axes.edgecolor': self.osh_colors['text'],
            'axes.labelcolor': self.osh_colors['text'],
            'text.color': self.osh_colors['text'],
            'xtick.color': self.osh_colors['text'],
            'ytick.color': self.osh_colors['text'],
            'grid.color': self.osh_colors['grid'],
            'figure.figsize': (12, 8),
            'figure.dpi': 100 if not self.scientific_mode else 150,
            'font.size': 10 if not self.scientific_mode else 12,
            'axes.titlesize': 14 if not self.scientific_mode else 16,
            'axes.labelsize': 12 if not self.scientific_mode else 14
        })
    
    def _register_event_listeners(self):
        """Register event listeners for real-time updates"""
        if not self.event_system:
            return
            
        # Simulation events
        simulation_events = [
            'simulation_start', 'simulation_pause', 'simulation_stop',
            'simulation_step', 'simulation_error'
        ]
        
        # OSH events
        osh_events = [
            'coherence_change', 'entropy_increase', 'strain_threshold',
            'observer_consensus', 'recursive_boundary', 'emergence_event'
        ]
        
        # Register all event handlers
        for event in simulation_events + osh_events:
            try:
                self.event_system.add_listener(event, self._handle_event)
            except Exception as e:
                self.logger.warning(f"Could not register listener for {event}: {e}")
    
    def _handle_event(self, event_data: Dict[str, Any]):
        """Handle incoming simulation events"""
        with self._lock:
            try:
                # Add to event log
                event_entry = {
                    'timestamp': time.time(),
                    'type': event_data.get('type', 'unknown'),
                    'data': event_data.get('data', {}),
                    'category': self._categorize_event(event_data.get('type', ''))
                }
                self.event_log.append(event_entry)
                
                # Update metrics based on event
                self._update_metrics_from_event(event_data)
                
            except Exception as e:
                self.logger.error(f"Error handling event: {e}")
    
    def _categorize_event(self, event_type: str) -> str:
        """Categorize event type for visualization"""
        for category, events in self.event_categories.items():
            if any(event_type.startswith(event) for event in events):
                return category
        return 'other'
    
    def _update_metrics_from_event(self, event_data: Dict[str, Any]):
        """Update internal metrics based on event data"""
        event_type = event_data.get('type', '')
        data = event_data.get('data', {})
        
        # Update specific metrics based on event type
        if 'coherence' in event_type:
            self._update_time_series()
        elif 'observer' in event_type:
            # Handle observer-related events
            pass
        elif 'recursive' in event_type:
            # Handle recursive events
            pass
    
    def select_visualization(self, visualization_type: str) -> bool:
        """
        Select visualization type for the panel.
        
        Args:
            visualization_type: Type of visualization to display
            
        Returns:
            bool: Success status
        """
        valid_types = [
            'simulation_overview', 'memory_usage', 'time_evolution',
            'event_log', 'osh_system_state', 'execution_context',
            'system_snapshots', 'recursive_boundaries'
        ]
        
        if visualization_type not in valid_types:
            self.logger.warning(f"Invalid visualization type: {visualization_type}")
            return False
        
        self.current_visualization = visualization_type
        
        # Clear cache for new visualization
        if self.enable_caching and self.visualization_cache:
            self.visualization_cache.clear()
        
        return True
    
    def start_simulation(self) -> Dict[str, Any]:
        """Start or resume the simulation"""
        try:
            if self.execution_context:
                if hasattr(self.execution_context, 'state') and self.execution_context.state == 'PAUSED':
                    self.execution_context.resume_execution()
                    message = "Simulation resumed"
                else:
                    self.execution_context.start_execution()
                    message = "Simulation started"
                
                return {
                    'success': True,
                    'message': message,
                    'state': getattr(self.execution_context, 'state', 'RUNNING'),
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': "No execution context available",
                    'state': 'ERROR'
                }
        except Exception as e:
            self.logger.error(f"Error starting simulation: {e}")
            return {
                'success': False,
                'message': f"Error starting simulation: {str(e)}",
                'state': 'ERROR'
            }
    
    def pause_simulation(self) -> Dict[str, Any]:
        """Pause the simulation"""
        try:
            if self.execution_context:
                self.execution_context.pause_execution()
                return {
                    'success': True,
                    'message': "Simulation paused",
                    'state': 'PAUSED',
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': "No execution context available",
                    'state': 'ERROR'
                }
        except Exception as e:
            self.logger.error(f"Error pausing simulation: {e}")
            return {
                'success': False,
                'message': f"Error pausing simulation: {str(e)}",
                'state': 'ERROR'
            }
    
    def stop_simulation(self) -> Dict[str, Any]:
        """Stop the simulation"""
        try:
            if self.execution_context:
                self.execution_context.complete_execution()
                elapsed = getattr(self.execution_context, 'elapsed_time', 0.0)
                return {
                    'success': True,
                    'message': f"Simulation completed in {elapsed:.2f} seconds",
                    'state': 'COMPLETED',
                    'elapsed_time': elapsed,
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': "No execution context available",
                    'state': 'ERROR'
                }
        except Exception as e:
            self.logger.error(f"Error stopping simulation: {e}")
            return {
                'success': False,
                'message': f"Error stopping simulation: {str(e)}",
                'state': 'ERROR'
            }
    
    def step_simulation(self, time_step: Optional[float] = None) -> Dict[str, Any]:
        """Advance simulation by one step"""
        try:
            if self.execution_context:
                # Use default time step if not specified
                if time_step is None:
                    time_step = getattr(self.execution_context, 'time_step', 0.01)
                
                # Advance simulation time
                current_time = getattr(self.execution_context, 'simulation_time', 0.0)
                new_time = current_time + time_step
                
                if hasattr(self.execution_context, 'advance_simulation_time'):
                    self.execution_context.advance_simulation_time(time_step)
                
                # Update metrics
                self._update_time_series()
                
                return {
                    'success': True,
                    'message': f"Simulation stepped by {time_step:.4f}",
                    'time_step': time_step,
                    'simulation_time': new_time,
                    'timestamp': time.time()
                }
            else:
                return {
                    'success': False,
                    'message': "No execution context available",
                    'state': 'ERROR'
                }
        except Exception as e:
            self.logger.error(f"Error stepping simulation: {e}")
            return {
                'success': False,
                'message': f"Error stepping simulation: {str(e)}",
                'state': 'ERROR'
            }
    
    def update(self, simulation_data: Dict[str, Any]) -> bool:
        """
        Update panel with new simulation data.
        
        Args:
            simulation_data: Dictionary containing simulation state data
            
        Returns:
            bool: Success status
        """
        with self._lock:
            try:
                start_time = time.time()
                
                # Extract and update time series data
                self._extract_simulation_metrics(simulation_data)
                
                # Update OSH field calculations
                self._update_osh_fields()
                
                # Update performance metrics
                update_time = time.time() - start_time
                self.performance_metrics['update_times'].append(update_time)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error updating simulation panel: {e}")
                return False
    
    def _extract_simulation_metrics(self, simulation_data: Dict[str, Any]):
        """Extract metrics from simulation data"""
        current_time = time.time()
        
        # Create metrics object
        metrics = SimulationMetrics(timestamp=current_time)
        
        # Extract simulation time
        if self.execution_context:
            metrics.simulation_time = getattr(self.execution_context, 'simulation_time', 0.0)
            metrics.real_time = getattr(self.execution_context, 'elapsed_time', 0.0)
        
        # Extract memory usage
        if 'memory' in simulation_data:
            memory_data = simulation_data['memory']
            metrics.memory_usage = self._extract_numeric_value(
                memory_data.get('total_usage', 0))
        
        # Extract coherence data
        if 'coherence' in simulation_data:
            coherence_data = simulation_data['coherence']
            metrics.average_coherence = self._extract_numeric_value(
                coherence_data.get('average', 0))
        
        # Extract entropy data
        if 'entropy' in simulation_data:
            entropy_data = simulation_data['entropy']
            metrics.average_entropy = self._extract_numeric_value(
                entropy_data.get('average', 0))
        
        # Extract strain data
        if 'strain' in simulation_data:
            strain_data = simulation_data['strain']
            metrics.average_strain = self._extract_numeric_value(
                strain_data.get('average', 0))
            metrics.field_strain = self._extract_numeric_value(
                strain_data.get('field_strain', 0))
        
        # Extract observer data
        if 'observers' in simulation_data:
            observer_data = simulation_data['observers']
            metrics.observer_interactions = self._extract_numeric_value(
                observer_data.get('interactions', 0))
        
        # Extract recursive data
        if 'recursive' in simulation_data:
            recursive_data = simulation_data['recursive']
            metrics.recursive_depth = self._extract_numeric_value(
                recursive_data.get('depth', 0))
            metrics.boundary_crossings = self._extract_numeric_value(
                recursive_data.get('boundary_crossings', 0))
        
        # Calculate RSP
        metrics.rsp = self._calculate_rsp(
            metrics.average_coherence,
            metrics.average_entropy,
            metrics.average_strain
        )
        
        # Calculate consciousness quotient
        metrics.consciousness_quotient = self._calculate_consciousness_quotient(metrics)
        
        # Extract phenomena count
        if 'phenomena' in simulation_data:
            phenomena_data = simulation_data['phenomena']
            metrics.phenomena_count = len(phenomena_data.get('active', []))
        
        # Add to time series
        self._add_metrics_to_time_series(metrics)
    
    def _extract_numeric_value(self, value: Any) -> float:
        """Extract numeric value from various data types"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, dict):
            # Try common keys
            for key in ['value', 'average', 'mean', 'current']:
                if key in value:
                    return self._extract_numeric_value(value[key])
            return 0.0
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            # Use first numeric value or average
            try:
                numeric_values = [self._extract_numeric_value(v) for v in value]
                return np.mean(numeric_values) if numeric_values else 0.0
            except:
                return 0.0
        elif isinstance(value, str):
            try:
                return float(value)
            except:
                return 0.0
        else:
            return 0.0
    
    def _calculate_rsp(self, coherence: float, entropy: float, strain: float) -> float:
        """
        Calculate Recursive Simulation Potential (RSP).
        
        RSP = (Coherence × (1 - Entropy)) / (Strain + ε)
        """
        epsilon = 1e-6
        numerator = coherence * (1.0 - entropy)
        denominator = strain + epsilon
        return numerator / denominator
    
    def _calculate_consciousness_quotient(self, metrics: SimulationMetrics) -> float:
        """Calculate consciousness quotient based on OSH principles"""
        # Base consciousness on coherence, complexity, and integration
        base_consciousness = metrics.average_coherence * (1.0 - metrics.average_entropy)
        
        # Enhance with recursive depth and RSP
        recursive_factor = np.log(1 + metrics.recursive_depth) / 10.0
        rsp_factor = np.tanh(metrics.rsp / 10.0)
        
        # Observer interaction factor
        observer_factor = np.tanh(metrics.observer_interactions / 100.0)
        
        consciousness = base_consciousness * (1 + recursive_factor + rsp_factor + observer_factor)
        
        return np.clip(consciousness, 0.0, 1.0)
    
    def _add_metrics_to_time_series(self, metrics: SimulationMetrics):
        """Add metrics to time series data"""
        self.time_series['simulation_time'].append(metrics.simulation_time)
        self.time_series['real_time'].append(metrics.real_time)
        self.time_series['memory_usage'].append(metrics.memory_usage)
        self.time_series['average_coherence'].append(metrics.average_coherence)
        self.time_series['average_entropy'].append(metrics.average_entropy)
        self.time_series['average_strain'].append(metrics.average_strain)
        self.time_series['observer_interactions'].append(metrics.observer_interactions)
        self.time_series['field_strain'].append(metrics.field_strain)
        self.time_series['recursive_depth'].append(metrics.recursive_depth)
        self.time_series['rsp'].append(metrics.rsp)
        self.time_series['consciousness_quotient'].append(metrics.consciousness_quotient)
        self.time_series['phenomena_count'].append(metrics.phenomena_count)
        self.time_series['boundary_crossings'].append(metrics.boundary_crossings)
        self.time_series['timestamps'].append(metrics.timestamp)
    
    def _update_time_series(self):
        """Update time series with current values"""
        # Get current values from various sources
        current_values = self._get_current_osh_values()
        
        # Create metrics object and add to series
        metrics = SimulationMetrics(
            simulation_time=current_values.get('simulation_time', 0.0),
            average_coherence=current_values.get('coherence', 0.0),
            average_entropy=current_values.get('entropy', 0.0),
            average_strain=current_values.get('strain', 0.0),
            memory_usage=current_values.get('memory_usage', 0.0),
            observer_interactions=current_values.get('observer_interactions', 0),
            recursive_depth=current_values.get('recursive_depth', 0),
            boundary_crossings=current_values.get('boundary_crossings', 0)
        )
        
        # Calculate derived metrics
        metrics.rsp = self._calculate_rsp(
            metrics.average_coherence,
            metrics.average_entropy,
            metrics.average_strain
        )
        metrics.consciousness_quotient = self._calculate_consciousness_quotient(metrics)
        
        self._add_metrics_to_time_series(metrics)
    
    def _get_current_osh_values(self) -> Dict[str, float]:
        """Get current OSH values from system components"""
        values = {}
        
        # Execution context values
        if self.execution_context:
            values['simulation_time'] = getattr(self.execution_context, 'simulation_time', 0.0)
            values['memory_usage'] = getattr(self.execution_context, 'memory_usage', 0.0)
        
        # Memory field values
        if self.memory_field:
            try:
                field_stats = self.memory_field.get_field_statistics()
                values['coherence'] = field_stats.get('avg_coherence', 0.0)
                values['entropy'] = field_stats.get('avg_entropy', 0.0)
                values['strain'] = field_stats.get('avg_strain', 0.0)
            except:
                values.update({'coherence': 0.0, 'entropy': 0.0, 'strain': 0.0})
        
        # Recursive mechanics values
        if self.recursive_mechanics:
            try:
                recursive_stats = self.recursive_mechanics.get_system_statistics()
                values['recursive_depth'] = recursive_stats.get('max_depth', 0)
                values['boundary_crossings'] = recursive_stats.get('boundary_crossings', 0)
            except:
                values.update({'recursive_depth': 0, 'boundary_crossings': 0})
        
        # Default values
        for key in ['simulation_time', 'coherence', 'entropy', 'strain', 'memory_usage',
                   'observer_interactions', 'recursive_depth', 'boundary_crossings']:
            if key not in values:
                values[key] = 0.0
        
        return values
    
    def _update_osh_fields(self):
        """Update OSH field calculations"""
        try:
            # Calculate 2D field representations
            field_size = 20
            
            # Get current values
            current_values = self._get_current_osh_values()
            
            # Create field grids with some spatial variation
            x = np.linspace(-1, 1, field_size)
            y = np.linspace(-1, 1, field_size)
            X, Y = np.meshgrid(x, y)
            
            # Base field calculations
            coherence_base = current_values['coherence']
            entropy_base = current_values['entropy']
            strain_base = current_values['strain']
            
            # Add spatial variations
            coherence_field = coherence_base * (1 + 0.2 * np.sin(3 * X) * np.cos(3 * Y))
            entropy_field = entropy_base * (1 + 0.3 * np.cos(2 * X) * np.sin(2 * Y))
            strain_field = strain_base * (1 + 0.25 * np.sin(4 * X + Y) * np.cos(2 * Y))
            
            # Calculate RSP field using correct OSH formula
            from src.visualization.osh_formula_utils import calculate_rsp_visualization
            rsp_field = calculate_rsp_visualization(coherence_field, entropy_field, strain_field)
            
            # Update field metrics
            self.current_osh_metrics.coherence_field = np.clip(coherence_field, 0, 1)
            self.current_osh_metrics.entropy_field = np.clip(entropy_field, 0, 1)
            self.current_osh_metrics.strain_field = np.clip(strain_field, 0, 1)
            self.current_osh_metrics.rsp_field = np.clip(rsp_field, 0, 10)
            
            # Calculate derived metrics
            self.current_osh_metrics.information_curvature = self._calculate_information_curvature()
            self.current_osh_metrics.field_energy = np.sum(coherence_field**2)
            self.current_osh_metrics.emergence_index = self._calculate_emergence_index()
            
            # Find critical regions (high strain, low coherence)
            critical_mask = (strain_field > 0.7) & (coherence_field < 0.3)
            critical_indices = np.where(critical_mask)
            self.current_osh_metrics.critical_regions = list(zip(critical_indices[0], critical_indices[1]))
            
            # Add to history
            self.osh_history.append({
                'timestamp': time.time(),
                'coherence_mean': np.mean(coherence_field),
                'entropy_mean': np.mean(entropy_field),
                'strain_mean': np.mean(strain_field),
                'rsp_mean': np.mean(rsp_field),
                'information_curvature': self.current_osh_metrics.information_curvature,
                'emergence_index': self.current_osh_metrics.emergence_index
            })
            
        except Exception as e:
            self.logger.error(f"Error updating OSH fields: {e}")
    
    def _calculate_information_curvature(self) -> float:
        """Calculate information geometry curvature"""
        try:
            # Use coherence field for curvature calculation
            field = self.current_osh_metrics.coherence_field
            
            # Calculate second derivatives (discrete Laplacian)
            laplacian = np.zeros_like(field)
            laplacian[1:-1, 1:-1] = (
                field[2:, 1:-1] + field[:-2, 1:-1] + 
                field[1:-1, 2:] + field[1:-1, :-2] - 
                4 * field[1:-1, 1:-1]
            )
            
            # Return RMS curvature
            return np.sqrt(np.mean(laplacian**2))
            
        except Exception as e:
            self.logger.error(f"Error calculating information curvature: {e}")
            return 0.0
    
    def _calculate_emergence_index(self) -> float:
        """Calculate emergence index based on field complexity"""
        try:
            # Use multiple fields for emergence calculation
            coherence = self.current_osh_metrics.coherence_field
            entropy = self.current_osh_metrics.entropy_field
            strain = self.current_osh_metrics.strain_field
            
            # Calculate spatial variance (complexity measure)
            coherence_var = np.var(coherence)
            entropy_var = np.var(entropy)
            strain_var = np.var(strain)
            
            # Calculate correlation between fields
            coherence_flat = coherence.flatten()
            entropy_flat = entropy.flatten()
            correlation = np.corrcoef(coherence_flat, entropy_flat)[0, 1]
            
            # Emergence index combines complexity and correlation
            complexity = (coherence_var + entropy_var + strain_var) / 3.0
            emergence = complexity * (1 - abs(correlation))
            
            return np.clip(emergence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating emergence index: {e}")
            return 0.0
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics"""
        try:
            stats = {}
            
            # Basic simulation state
            if self.execution_context:
                stats['simulation_state'] = getattr(self.execution_context, 'state', 'UNKNOWN')
                stats['simulation_time'] = getattr(self.execution_context, 'simulation_time', 0.0)
                stats['elapsed_seconds'] = getattr(self.execution_context, 'elapsed_time', 0.0)
                stats['elapsed_time'] = self._format_time(stats['elapsed_seconds'])
            else:
                stats.update({
                    'simulation_state': 'NO_CONTEXT',
                    'simulation_time': 0.0,
                    'elapsed_seconds': 0.0,
                    'elapsed_time': '00:00:00'
                })
            
            # Context statistics
            if self.execution_context and hasattr(self.execution_context, 'get_statistics'):
                stats['context_stats'] = self.execution_context.get_statistics()
            else:
                stats['context_stats'] = {}
            
            # Memory statistics
            stats['memory_stats'] = self._get_memory_statistics()
            
            # Event statistics
            stats['event_stats'] = self._get_event_statistics()
            
            # OSH metrics
            stats['osh_metrics'] = self._get_current_osh_summary()
            
            # Performance statistics
            stats['performance'] = self._get_performance_statistics()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting simulation statistics: {e}")
            return {'error': str(e)}
    
    def _format_time(self, seconds: float) -> str:
        """Format time in HH:MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        stats = {'pools': {}, 'total_usage': 0.0}
        
        try:
            if self.execution_context and hasattr(self.execution_context, 'memory_manager'):
                memory_manager = self.execution_context.memory_manager
                if hasattr(memory_manager, 'get_memory_usage'):
                    memory_stats = memory_manager.get_memory_usage()
                    stats.update(memory_stats)
            
            # Get memory field statistics
            if self.memory_field:
                field_stats = self.memory_field.get_field_statistics()
                stats['field_stats'] = field_stats
                
        except Exception as e:
            self.logger.error(f"Error getting memory statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _get_event_statistics(self) -> Dict[str, Any]:
        """Get event statistics"""
        stats = {
            'total_events': len(self.event_log),
            'events_by_category': {},
            'recent_events': 0
        }
        
        try:
            # Count events by category
            for category in self.event_categories.keys():
                count = sum(1 for event in self.event_log if event['category'] == category)
                stats['events_by_category'][category] = count
            
            # Count recent events (last 60 seconds)
            current_time = time.time()
            recent_threshold = current_time - 60.0
            stats['recent_events'] = sum(
                1 for event in self.event_log 
                if event['timestamp'] > recent_threshold
            )
            
        except Exception as e:
            self.logger.error(f"Error getting event statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def _get_current_osh_summary(self) -> Dict[str, Any]:
        """Get current OSH metrics summary"""
        try:
            summary = {}
            
            # Current field averages
            if hasattr(self.current_osh_metrics, 'coherence_field'):
                summary['coherence'] = float(np.mean(self.current_osh_metrics.coherence_field))
                summary['entropy'] = float(np.mean(self.current_osh_metrics.entropy_field))
                summary['strain'] = float(np.mean(self.current_osh_metrics.strain_field))
                summary['rsp'] = float(np.mean(self.current_osh_metrics.rsp_field))
            else:
                summary.update({'coherence': 0.0, 'entropy': 0.0, 'strain': 0.0, 'rsp': 0.0})
            
            # Derived metrics
            summary['information_curvature'] = self.current_osh_metrics.information_curvature
            summary['field_energy'] = self.current_osh_metrics.field_energy
            summary['emergence_index'] = self.current_osh_metrics.emergence_index
            summary['critical_regions'] = len(self.current_osh_metrics.critical_regions)
            
            # Classification
            summary['rsp_classification'] = self._classify_rsp_level(summary['rsp'])
            summary['coherence_classification'] = self._classify_coherence_level(summary['coherence'])
            summary['emergence_classification'] = self._classify_emergence_level(summary['emergence_index'])
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting OSH summary: {e}")
            return {'error': str(e)}
    
    def _classify_rsp_level(self, rsp: float) -> str:
        """Classify RSP level"""
        if rsp < 0.1:
            return "critical"
        elif rsp < 0.5:
            return "low"
        elif rsp < 2.0:
            return "moderate"
        elif rsp < 5.0:
            return "high"
        else:
            return "exceptional"
    
    def _classify_coherence_level(self, coherence: float) -> str:
        """Classify coherence level"""
        if coherence < 0.2:
            return "very_low"
        elif coherence < 0.4:
            return "low"
        elif coherence < 0.6:
            return "moderate"
        elif coherence < 0.8:
            return "high"
        else:
            return "very_high"
    
    def _classify_emergence_level(self, emergence: float) -> str:
        """Classify emergence level"""
        if emergence < 0.1:
            return "minimal"
        elif emergence < 0.3:
            return "weak"
        elif emergence < 0.6:
            return "moderate"
        elif emergence < 0.8:
            return "strong"
        else:
            return "exceptional"
    
    def _get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        
        try:
            # Render performance
            render_times = list(self.performance_metrics['render_times'])
            if render_times:
                stats['avg_render_time'] = np.mean(render_times)
                stats['max_render_time'] = np.max(render_times)
                stats['min_render_time'] = np.min(render_times)
            else:
                stats.update({
                    'avg_render_time': 0.0,
                    'max_render_time': 0.0,
                    'min_render_time': 0.0
                })
            
            # Update performance
            update_times = list(self.performance_metrics['update_times'])
            if update_times:
                stats['avg_update_time'] = np.mean(update_times)
                stats['update_frequency'] = len(update_times) / max(60.0, time.time() - update_times[0] if update_times else 60.0)
            else:
                stats.update({
                    'avg_update_time': 0.0,
                    'update_frequency': 0.0
                })
            
            # Frame rate estimation
            if render_times:
                stats['estimated_fps'] = 1.0 / max(np.mean(render_times), 0.001)
            else:
                stats['estimated_fps'] = 0.0
            
        except Exception as e:
            self.logger.error(f"Error getting performance statistics: {e}")
            stats['error'] = str(e)
        
        return stats
    
    def get_simulation_snapshot(self) -> Dict[str, Any]:
        """Get comprehensive simulation snapshot"""
        try:
            snapshot = {}
            
            # Basic state
            if self.execution_context:
                if hasattr(self.execution_context, 'get_context_snapshot'):
                    snapshot.update(self.execution_context.get_context_snapshot())
                else:
                    # Manual snapshot
                    snapshot.update({
                        'state': getattr(self.execution_context, 'state', 'UNKNOWN'),
                        'simulation_time': getattr(self.execution_context, 'simulation_time', 0.0),
                        'memory_usage': getattr(self.execution_context, 'memory_usage', 0.0),
                        'active_observer': getattr(self.execution_context, 'active_observer', None),
                        'current_scope': getattr(self.execution_context, 'current_scope', None)
                    })
            
            # Add OSH metrics
            snapshot['osh_metrics'] = self._get_current_osh_summary()
            
            # Add time series summary
            snapshot['time_series_summary'] = self._get_time_series_summary()
            
            # Add event log summary
            snapshot['event_summary'] = self._get_event_statistics()
            
            # Add timestamp
            snapshot['snapshot_timestamp'] = time.time()
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Error creating simulation snapshot: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _get_time_series_summary(self) -> Dict[str, Any]:
        """Get time series data summary"""
        summary = {}
        
        try:
            for key, series in self.time_series.items():
                if series and key != 'timestamps':
                    data = list(series)
                    if data:
                        summary[key] = {
                            'current': data[-1],
                            'mean': np.mean(data),
                            'std': np.std(data),
                            'min': np.min(data),
                            'max': np.max(data),
                            'count': len(data)
                        }
                        
                        # Add trend analysis
                        if len(data) > 1:
                            slope, _, _, _, _ = stats.linregress(range(len(data)), data)
                            summary[key]['trend'] = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'
                            summary[key]['slope'] = slope
            
        except Exception as e:
            self.logger.error(f"Error getting time series summary: {e}")
            summary['error'] = str(e)
        
        return summary
    
    def render_panel(self, width: int = 1600, height: int = 900) -> Dict[str, Any]:
        """
        Render the simulation panel.
        
        Args:
            width: Panel width in pixels
            height: Panel height in pixels
            
        Returns:
            Dict containing render results and metadata
        """
        start_time = time.time()
        
        try:
            # Check cache first
            if self.enable_caching and self.visualization_cache:
                cache_key = f"{self.current_visualization}_{width}_{height}_{hash(str(self._get_current_osh_summary()))}"
                if cache_key in self.visualization_cache:
                    cached_result = self.visualization_cache[cache_key].copy()
                    cached_result['from_cache'] = True
                    return cached_result
            
            # Route to appropriate renderer
            result = self._route_visualization(width, height)
            
            # Add performance metrics
            render_time = time.time() - start_time
            self.performance_metrics['render_times'].append(render_time)
            result['render_time'] = render_time
            
            # Cache result
            if self.enable_caching and self.visualization_cache and result.get('success', False):
                cache_key = f"{self.current_visualization}_{width}_{height}_{hash(str(self._get_current_osh_summary()))}"
                result_copy = result.copy()
                result_copy.pop('render_time', None)  # Don't cache timing info
                self.visualization_cache[cache_key] = result_copy
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error rendering simulation panel: {e}")
            return {
                'success': False,
                'error': str(e),
                'visualization': self.current_visualization,
                'render_time': time.time() - start_time,
                'traceback': traceback.format_exc()
            }
    
    def _route_visualization(self, width: int, height: int) -> Dict[str, Any]:
        """Route to appropriate visualization renderer"""
        renderers = {
            'simulation_overview': self._render_simulation_overview,
            'memory_usage': self._render_memory_usage,
            'time_evolution': self._render_time_evolution,
            'event_log': self._render_event_log,
            'osh_system_state': self._render_osh_system_state,
            'execution_context': self._render_execution_context,
            'system_snapshots': self._render_system_snapshots,
            'recursive_boundaries': self._render_recursive_boundaries
        }
        
        renderer = renderers.get(self.current_visualization, self._render_simulation_overview)
        return renderer(width, height)
    
    def _render_simulation_overview(self, width: int, height: int) -> Dict[str, Any]:
        """Render comprehensive simulation overview"""
        try:
            fig = plt.figure(figsize=(width/100, height/100), facecolor=self.osh_colors['background'])
            gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Get current statistics
            stats = self.get_simulation_statistics()
            
            # State indicator (top-left)
            ax1 = fig.add_subplot(gs[0, 0])
            self._render_state_indicator(ax1, stats)
            
            # Memory usage (top-right)
            ax2 = fig.add_subplot(gs[0, 1])
            self._render_memory_overview(ax2, stats.get('memory_stats', {}))
            
            # Time evolution (bottom-left)
            ax3 = fig.add_subplot(gs[1, 0])
            self._render_time_overview(ax3)
            
            # OSH radar (bottom-right)
            ax4 = fig.add_subplot(gs[1, 1])
            self._render_osh_radar(ax4, stats.get('osh_metrics', {}))
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'simulation_overview',
                'statistics': stats
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering simulation overview: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'simulation_overview'
            }
    
    def _render_state_indicator(self, ax, stats: Dict[str, Any]):
        """Render simulation state indicator"""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        
        # State colors
        state_colors = {
            'RUNNING': self.osh_colors['coherence'],
            'PAUSED': '#FFA500',
            'COMPLETED': self.osh_colors['rsp'],
            'ERROR': self.osh_colors['strain'],
            'UNKNOWN': '#808080'
        }
        
        state = stats.get('simulation_state', 'UNKNOWN')
        color = state_colors.get(state, '#808080')
        
        # Draw state circle
        circle = Circle((0.5, 0.7), 0.15, color=color, alpha=0.8)
        ax.add_patch(circle)
        
        # State text
        ax.text(0.5, 0.7, state, ha='center', va='center', 
               fontsize=12, fontweight='bold', color='white')
        
        # Simulation time
        sim_time = stats.get('simulation_time', 0.0)
        ax.text(0.5, 0.4, f"Time: {sim_time:.3f}s", ha='center', va='center',
               fontsize=10, color=self.osh_colors['text'])
        
        # Elapsed time
        elapsed = stats.get('elapsed_time', '00:00:00')
        ax.text(0.5, 0.2, f"Elapsed: {elapsed}", ha='center', va='center',
               fontsize=10, color=self.osh_colors['text'])
        
        ax.set_title("Simulation State", color=self.osh_colors['text'], fontsize=14, fontweight='bold')
        ax.axis('off')
    
    def _render_memory_overview(self, ax, memory_stats: Dict[str, Any]):
        """Render memory usage overview"""
        # Extract memory data
        pools = memory_stats.get('pools', {})
        if not pools:
            # Create sample data
            pools = {
                'standard': {'used': 1024, 'total': 2048},
                'quantum': {'used': 512, 'total': 1024},
                'observer': {'used': 256, 'total': 512},
                'temporary': {'used': 128, 'total': 256}
            }
        
        # Prepare data
        pool_names = list(pools.keys())
        used_values = [pools[name].get('used', 0) for name in pool_names]
        total_values = [pools[name].get('total', 1) for name in pool_names]
        usage_percentages = [used/total*100 if total > 0 else 0 
                           for used, total in zip(used_values, total_values)]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(pool_names))
        bars = ax.barh(y_pos, usage_percentages, color=self.osh_colors['coherence'], alpha=0.7)
        
        # Color code based on usage
        for i, (bar, usage) in enumerate(zip(bars, usage_percentages)):
            if usage > 80:
                bar.set_color(self.osh_colors['strain'])
            elif usage > 60:
                bar.set_color('#FFA500')
            else:
                bar.set_color(self.osh_colors['coherence'])
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pool_names, color=self.osh_colors['text'])
        ax.set_xlabel('Usage %', color=self.osh_colors['text'])
        ax.set_title('Memory Usage', color=self.osh_colors['text'], fontsize=14, fontweight='bold')
        ax.set_xlim(0, 100)
        
        # Add value labels
        for i, (bar, usage, used, total) in enumerate(zip(bars, usage_percentages, used_values, total_values)):
            ax.text(usage + 2, i, f'{usage:.1f}% ({used}/{total})', 
                   va='center', color=self.osh_colors['text'], fontsize=9)
    
    def _render_time_overview(self, ax):
        """Render time series overview"""
        # Get recent data (last 20 points)
        n_points = min(20, len(self.time_series['timestamps']))
        if n_points < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   color=self.osh_colors['text'], transform=ax.transAxes)
            ax.set_title('Time Evolution', color=self.osh_colors['text'], fontsize=14, fontweight='bold')
            return
        
        # Extract data
        timestamps = list(self.time_series['timestamps'])[-n_points:]
        coherence = list(self.time_series['average_coherence'])[-n_points:]
        entropy = list(self.time_series['average_entropy'])[-n_points:]
        strain = list(self.time_series['average_strain'])[-n_points:]
        
        # Convert to relative time
        if timestamps:
            base_time = timestamps[0]
            rel_times = [(t - base_time) for t in timestamps]
        else:
            rel_times = list(range(n_points))
        
        # Plot lines
        ax.plot(rel_times, coherence, color=self.osh_colors['coherence'], 
               label='Coherence', linewidth=2, marker='o', markersize=3)
        ax.plot(rel_times, entropy, color=self.osh_colors['entropy'], 
               label='Entropy', linewidth=2, marker='s', markersize=3)
        ax.plot(rel_times, strain, color=self.osh_colors['strain'], 
               label='Strain', linewidth=2, marker='^', markersize=3)
        
        ax.set_xlabel('Time (s)', color=self.osh_colors['text'])
        ax.set_ylabel('Value', color=self.osh_colors['text'])
        ax.set_title('OSH Metrics Evolution', color=self.osh_colors['text'], fontsize=14, fontweight='bold')
        ax.legend(facecolor=self.osh_colors['background'], edgecolor=self.osh_colors['text'])
        ax.grid(True, alpha=0.3, color=self.osh_colors['grid'])
    
    def _render_osh_radar(self, ax, osh_metrics: Dict[str, Any]):
        """Render OSH metrics radar chart"""
        # Metrics for radar chart
        metrics = {
            'Coherence': osh_metrics.get('coherence', 0.0),
            'Order': 1.0 - osh_metrics.get('entropy', 1.0),  # Order = 1 - Entropy
            'Stability': 1.0 - osh_metrics.get('strain', 1.0),  # Stability = 1 - Strain
            'RSP': min(osh_metrics.get('rsp', 0.0) / 5.0, 1.0),  # Normalize RSP
            'Emergence': osh_metrics.get('emergence_index', 0.0),
            'Complexity': min(osh_metrics.get('information_curvature', 0.0) * 10, 1.0)
        }
        
        # Setup radar chart
        categories = list(metrics.keys())
        values = list(metrics.values())
        
        # Number of variables
        N = len(categories)
        
        # Angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Values
        values += values[:1]  # Complete the circle
        
        # Plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles, values, 'o-', linewidth=2, color=self.osh_colors['coherence'])
        ax.fill(angles, values, alpha=0.25, color=self.osh_colors['coherence'])
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color=self.osh_colors['text'])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], 
                          color=self.osh_colors['text'], fontsize=8)
        ax.grid(True, alpha=0.3, color=self.osh_colors['grid'])
        
        ax.set_title('OSH Metrics Radar', color=self.osh_colors['text'], 
                    fontsize=14, fontweight='bold', pad=20)
    
    def _render_memory_usage(self, width: int, height: int) -> Dict[str, Any]:
        """Render detailed memory usage visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100), 
                                         facecolor=self.osh_colors['background'])
            
            # Get memory statistics
            memory_stats = self._get_memory_statistics()
            pools = memory_stats.get('pools', {})
            
            if not pools:
                # Create sample data for demonstration
                pools = {
                    'standard': {'used': 1024, 'capacity': 2048, 'allocations': 45},
                    'quantum': {'used': 1536, 'capacity': 2048, 'allocations': 23},
                    'observer': {'used': 768, 'capacity': 1024, 'allocations': 12},
                    'temporary': {'used': 256, 'capacity': 512, 'allocations': 67},
                    'field': {'used': 512, 'capacity': 1024, 'allocations': 8}
                }
            
            # Pool usage chart (top)
            pool_names = list(pools.keys())
            used_values = [pools[name].get('used', 0) for name in pool_names]
            capacity_values = [pools[name].get('capacity', 1) for name in pool_names]
            
            x_pos = np.arange(len(pool_names))
            width_bar = 0.35
            
            bars1 = ax1.bar(x_pos - width_bar/2, used_values, width_bar, 
                           label='Used', color=self.osh_colors['coherence'], alpha=0.8)
            bars2 = ax1.bar(x_pos + width_bar/2, capacity_values, width_bar,
                           label='Capacity', color=self.osh_colors['text'], alpha=0.3)
            
            ax1.set_xlabel('Memory Pools', color=self.osh_colors['text'])
            ax1.set_ylabel('Memory (MB)', color=self.osh_colors['text'])
            ax1.set_title('Memory Pool Usage', color=self.osh_colors['text'], fontsize=16, fontweight='bold')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(pool_names, rotation=45, ha='right', color=self.osh_colors['text'])
            ax1.legend(facecolor=self.osh_colors['background'], edgecolor=self.osh_colors['text'])
            ax1.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            
            # Add usage percentage labels
            for i, (used, capacity) in enumerate(zip(used_values, capacity_values)):
                if capacity > 0:
                    percentage = (used / capacity) * 100
                    ax1.text(i, used + capacity * 0.02, f'{percentage:.1f}%', 
                            ha='center', va='bottom', color=self.osh_colors['text'], fontweight='bold')
            
            # Allocation activity chart (bottom)
            allocations = [pools[name].get('allocations', 0) for name in pool_names]
            colors = [self._get_activity_color(alloc) for alloc in allocations]
            
            bars3 = ax2.bar(pool_names, allocations, color=colors, alpha=0.8)
            ax2.set_xlabel('Memory Pools', color=self.osh_colors['text'])
            ax2.set_ylabel('Active Allocations', color=self.osh_colors['text'])
            ax2.set_title('Memory Allocation Activity', color=self.osh_colors['text'], fontsize=16, fontweight='bold')
            ax2.tick_params(axis='x', rotation=45, colors=self.osh_colors['text'])
            ax2.tick_params(axis='y', colors=self.osh_colors['text'])
            ax2.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            
            # Add value labels
            for bar, alloc in zip(bars3, allocations):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{alloc}', ha='center', va='bottom', 
                        color=self.osh_colors['text'], fontweight='bold')
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'memory_usage',
                'statistics': memory_stats
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering memory usage: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'memory_usage'
            }
    
    def _get_activity_color(self, allocation_count: int) -> str:
        """Get color based on allocation activity level"""
        if allocation_count > 50:
            return self.osh_colors['strain']  # High activity - red
        elif allocation_count > 25:
            return '#FFA500'  # Medium activity - orange
        elif allocation_count > 10:
            return '#FFFF00'  # Low-medium activity - yellow
        else:
            return self.osh_colors['coherence']  # Low activity - cyan
    
    def _render_time_evolution(self, width: int, height: int) -> Dict[str, Any]:
        """Render comprehensive time evolution visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(width/100, height/100), 
                                   facecolor=self.osh_colors['background'])
            fig.suptitle('OSH Metrics Time Evolution', color=self.osh_colors['text'], 
                        fontsize=18, fontweight='bold')
            
            # Get time series data
            n_points = len(self.time_series['timestamps'])
            if n_points < 2:
                for ax in axes.flat:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                           color=self.osh_colors['text'], transform=ax.transAxes)
                    ax.set_facecolor(self.osh_colors['background'])
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                           bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close(fig)
                return {
                    'success': True,
                    'image_data': f"data:image/png;base64,{image_data}",
                    'visualization': 'time_evolution',
                    'message': 'Insufficient data for time evolution'
                }
            
            # Extract time series data
            timestamps = list(self.time_series['timestamps'])
            base_time = timestamps[0] if timestamps else 0
            rel_times = [(t - base_time) for t in timestamps]
            
            coherence_data = list(self.time_series['average_coherence'])
            entropy_data = list(self.time_series['average_entropy'])
            strain_data = list(self.time_series['average_strain'])
            memory_data = list(self.time_series['memory_usage'])
            
            # Coherence evolution (top-left)
            ax1 = axes[0, 0]
            ax1.plot(rel_times, coherence_data, color=self.osh_colors['coherence'], 
                    linewidth=2, marker='o', markersize=4, label='Coherence')
            ax1.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Critical')
            ax1.set_xlabel('Time (s)', color=self.osh_colors['text'])
            ax1.set_ylabel('Coherence', color=self.osh_colors['text'])
            ax1.set_title('Coherence Evolution', color=self.osh_colors['text'], fontweight='bold')
            ax1.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            ax1.legend(facecolor=self.osh_colors['background'])
            ax1.set_ylim(0, 1)
            
            # Entropy evolution (top-right)
            ax2 = axes[0, 1]
            ax2.plot(rel_times, entropy_data, color=self.osh_colors['entropy'], 
                    linewidth=2, marker='s', markersize=4, label='Entropy')
            ax2.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='Warning')
            ax2.set_xlabel('Time (s)', color=self.osh_colors['text'])
            ax2.set_ylabel('Entropy', color=self.osh_colors['text'])
            ax2.set_title('Entropy Evolution', color=self.osh_colors['text'], fontweight='bold')
            ax2.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            ax2.legend(facecolor=self.osh_colors['background'])
            ax2.set_ylim(0, 1)
            
            # Strain evolution (bottom-left)
            ax3 = axes[1, 0]
            ax3.plot(rel_times, strain_data, color=self.osh_colors['strain'], 
                    linewidth=2, marker='^', markersize=4, label='Memory Strain')
            ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Critical')
            ax3.set_xlabel('Time (s)', color=self.osh_colors['text'])
            ax3.set_ylabel('Strain', color=self.osh_colors['text'])
            ax3.set_title('Memory Strain Evolution', color=self.osh_colors['text'], fontweight='bold')
            ax3.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            ax3.legend(facecolor=self.osh_colors['background'])
            ax3.set_ylim(0, 1)
            
            # Memory usage evolution (bottom-right)
            ax4 = axes[1, 1]
            memory_mb = [m / (1024*1024) if m > 1024 else m for m in memory_data]  # Convert to MB if needed
            ax4.plot(rel_times, memory_mb, color='#FFD700', 
                    linewidth=2, marker='d', markersize=4, label='Memory Usage')
            ax4.set_xlabel('Time (s)', color=self.osh_colors['text'])
            ax4.set_ylabel('Memory (MB)', color=self.osh_colors['text'])
            ax4.set_title('Memory Usage Evolution', color=self.osh_colors['text'], fontweight='bold')
            ax4.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            ax4.legend(facecolor=self.osh_colors['background'])
            
            # Style all axes
            for ax in axes.flat:
                ax.set_facecolor(self.osh_colors['background'])
                ax.tick_params(colors=self.osh_colors['text'])
                ax.spines['bottom'].set_color(self.osh_colors['text'])
                ax.spines['top'].set_color(self.osh_colors['text'])
                ax.spines['right'].set_color(self.osh_colors['text'])
                ax.spines['left'].set_color(self.osh_colors['text'])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'time_evolution',
                'statistics': self._get_time_series_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering time evolution: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'time_evolution'
            }
    
    def _render_event_log(self, width: int, height: int) -> Dict[str, Any]:
        """Render event log visualization"""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(width/100, height/100),
                                         facecolor=self.osh_colors['background'])
            
            # Get event data
            events = list(self.event_log)
            if not events:
                # Create sample events for demonstration
                current_time = time.time()
                events = [
                    {'timestamp': current_time - 30, 'category': 'quantum', 'type': 'state_creation'},
                    {'timestamp': current_time - 25, 'category': 'observer', 'type': 'observation'},
                    {'timestamp': current_time - 20, 'category': 'memory', 'type': 'strain_threshold'},
                    {'timestamp': current_time - 15, 'category': 'recursive', 'type': 'boundary_crossing'},
                    {'timestamp': current_time - 10, 'category': 'simulation', 'type': 'step'},
                    {'timestamp': current_time - 5, 'category': 'quantum', 'type': 'measurement'},
                ]
            
            # Event timeline (top)
            if events:
                # Prepare data
                categories = list(self.event_categories.keys())
                colors = [self.osh_colors['coherence'], self.osh_colors['entropy'], 
                         self.osh_colors['strain'], self.osh_colors['rsp'], 
                         self.osh_colors['consciousness']][:len(categories)]
                
                category_colors = dict(zip(categories, colors))
                
                # Extract data for recent events (last 100)
                recent_events = events[-100:] if len(events) > 100 else events
                timestamps = [e['timestamp'] for e in recent_events]
                event_categories = [e['category'] for e in recent_events]
                
                # Convert to relative time
                if timestamps:
                    base_time = min(timestamps)
                    rel_times = [(t - base_time) for t in timestamps]
                else:
                    rel_times = []
                
                # Create scatter plot
                for i, (cat, color) in enumerate(category_colors.items()):
                    cat_times = [t for t, c in zip(rel_times, event_categories) if c == cat]
                    cat_y = [i] * len(cat_times)
                    
                    if cat_times:
                        ax1.scatter(cat_times, cat_y, c=color, alpha=0.7, s=50, label=cat.title())
                
                ax1.set_xlabel('Time (s)', color=self.osh_colors['text'])
                ax1.set_ylabel('Event Category', color=self.osh_colors['text'])
                ax1.set_title('Event Timeline', color=self.osh_colors['text'], fontsize=16, fontweight='bold')
                ax1.set_yticks(range(len(categories)))
                ax1.set_yticklabels([cat.title() for cat in categories], color=self.osh_colors['text'])
                ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', 
                          facecolor=self.osh_colors['background'], edgecolor=self.osh_colors['text'])
                ax1.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            
            # Event count by category (bottom)
            event_counts = {}
            for category in self.event_categories.keys():
                count = sum(1 for e in events if e.get('category') == category)
                event_counts[category] = count
            
            if event_counts:
                categories = list(event_counts.keys())
                counts = list(event_counts.values())
                colors = [category_colors.get(cat, '#808080') for cat in categories]
                
                bars = ax2.bar(categories, counts, color=colors, alpha=0.8)
                ax2.set_xlabel('Event Category', color=self.osh_colors['text'])
                ax2.set_ylabel('Event Count', color=self.osh_colors['text'])
                ax2.set_title('Event Distribution', color=self.osh_colors['text'], fontsize=16, fontweight='bold')
                ax2.tick_params(axis='x', rotation=45, colors=self.osh_colors['text'])
                ax2.tick_params(axis='y', colors=self.osh_colors['text'])
                ax2.grid(True, alpha=0.3, color=self.osh_colors['grid'])
                
                # Add count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{count}', ha='center', va='bottom', 
                            color=self.osh_colors['text'], fontweight='bold')
            
            # Style axes
            for ax in [ax1, ax2]:
                ax.set_facecolor(self.osh_colors['background'])
                ax.tick_params(colors=self.osh_colors['text'])
                for spine in ax.spines.values():
                    spine.set_color(self.osh_colors['text'])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'event_log',
                'event_stats': self._get_event_statistics()
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering event log: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'event_log'
            }
    
    def _render_osh_system_state(self, width: int, height: int) -> Dict[str, Any]:
        """Render comprehensive OSH system state visualization"""
        try:
            # Try to use advanced coherence renderer if available
            if self.coherence_renderer and hasattr(self.coherence_renderer, 'render_osh_substrate_comprehensive'):
                try:
                    result = self.coherence_renderer.render_osh_substrate_comprehensive(
                        width=width, height=height
                    )
                    if result.get('success', False):
                        result['visualization'] = 'osh_system_state'
                        return result
                except Exception as e:
                    self.logger.warning(f"Advanced OSH renderer failed, using fallback: {e}")
            
            # Fallback to manual OSH visualization
            fig, axes = plt.subplots(2, 2, figsize=(width/100, height/100),
                                   facecolor=self.osh_colors['background'])
            fig.suptitle('OSH System State', color=self.osh_colors['text'], 
                        fontsize=18, fontweight='bold')
            
            # Update OSH fields
            self._update_osh_fields()
            
            # Coherence field (top-left)
            ax1 = axes[0, 0]
            coherence_field = self.current_osh_metrics.coherence_field
            im1 = ax1.imshow(coherence_field, cmap=self.scientific_cmaps['coherence'], 
                           aspect='equal', vmin=0, vmax=1)
            ax1.set_title('Coherence Field', color=self.osh_colors['text'], fontweight='bold')
            ax1.set_xlabel('Spatial X', color=self.osh_colors['text'])
            ax1.set_ylabel('Spatial Y', color=self.osh_colors['text'])
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.ax.tick_params(colors=self.osh_colors['text'])
            
            # Entropy field (top-right)
            ax2 = axes[0, 1]
            entropy_field = self.current_osh_metrics.entropy_field
            im2 = ax2.imshow(entropy_field, cmap=self.scientific_cmaps['entropy'], 
                           aspect='equal', vmin=0, vmax=1)
            ax2.set_title('Entropy Field', color=self.osh_colors['text'], fontweight='bold')
            ax2.set_xlabel('Spatial X', color=self.osh_colors['text'])
            ax2.set_ylabel('Spatial Y', color=self.osh_colors['text'])
            cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(colors=self.osh_colors['text'])
            
            # Strain field (bottom-left)
            ax3 = axes[1, 0]
            strain_field = self.current_osh_metrics.strain_field
            im3 = ax3.imshow(strain_field, cmap=self.scientific_cmaps['strain'], 
                           aspect='equal', vmin=0, vmax=1)
            ax3.set_title('Strain Field', color=self.osh_colors['text'], fontweight='bold')
            ax3.set_xlabel('Spatial X', color=self.osh_colors['text'])
            ax3.set_ylabel('Spatial Y', color=self.osh_colors['text'])
            cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(colors=self.osh_colors['text'])
            
            # RSP field (bottom-right)
            ax4 = axes[1, 1]
            rsp_field = self.current_osh_metrics.rsp_field
            im4 = ax4.imshow(rsp_field, cmap=self.scientific_cmaps['rsp'], 
                           aspect='equal', vmin=0, vmax=np.max(rsp_field))
            ax4.set_title('RSP Field', color=self.osh_colors['text'], fontweight='bold')
            ax4.set_xlabel('Spatial X', color=self.osh_colors['text'])
            ax4.set_ylabel('Spatial Y', color=self.osh_colors['text'])
            cbar4 = plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.ax.tick_params(colors=self.osh_colors['text'])
            
            # Add critical regions overlay
            for ax in axes.flat:
                for (y, x) in self.current_osh_metrics.critical_regions:
                    circle = Circle((x, y), 0.5, fill=False, color='red', linewidth=2)
                    ax.add_patch(circle)
            
            # Style all axes
            for ax in axes.flat:
                ax.set_facecolor(self.osh_colors['background'])
                ax.tick_params(colors=self.osh_colors['text'])
                for spine in ax.spines.values():
                    spine.set_color(self.osh_colors['text'])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'osh_system_state',
                'osh_metrics': self._get_current_osh_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering OSH system state: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'osh_system_state'
            }
    
    def _render_execution_context(self, width: int, height: int) -> Dict[str, Any]:
        """Render execution context information"""
        try:
            # Get execution context snapshot
            snapshot = self.get_simulation_snapshot()
            
            # Create text-based visualization
            fig, ax = plt.subplots(figsize=(width/100, height/100), 
                                 facecolor=self.osh_colors['background'])
            
            # Format context information
            context_text = self._format_execution_context(snapshot)
            
            # Display text
            ax.text(0.05, 0.95, context_text, transform=ax.transAxes, 
                   fontsize=10, color=self.osh_colors['text'], 
                   verticalalignment='top', fontfamily='monospace')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('Execution Context', color=self.osh_colors['text'], 
                        fontsize=16, fontweight='bold')
            ax.axis('off')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'execution_context',
                'message': context_text
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering execution context: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'execution_context'
            }
    
    def _format_execution_context(self, snapshot: Dict[str, Any]) -> str:
        """Format execution context information as text"""
        lines = []
        lines.append("=== RECURSIA EXECUTION CONTEXT ===\n")
        
        # Basic state
        lines.append(f"State: {snapshot.get('state', 'UNKNOWN')}")
        lines.append(f"Simulation Time: {snapshot.get('simulation_time', 0.0):.6f}s")
        lines.append(f"Memory Usage: {snapshot.get('memory_usage', 0.0):.2f} MB")
        lines.append("")
        
        # OSH metrics
        osh_metrics = snapshot.get('osh_metrics', {})
        if osh_metrics:
            lines.append("=== OSH METRICS ===")
            lines.append(f"Coherence: {osh_metrics.get('coherence', 0.0):.4f}")
            lines.append(f"Entropy: {osh_metrics.get('entropy', 0.0):.4f}")
            lines.append(f"Strain: {osh_metrics.get('strain', 0.0):.4f}")
            lines.append(f"RSP: {osh_metrics.get('rsp', 0.0):.4f}")
            lines.append(f"RSP Classification: {osh_metrics.get('rsp_classification', 'unknown')}")
            lines.append(f"Information Curvature: {osh_metrics.get('information_curvature', 0.0):.6f}")
            lines.append(f"Emergence Index: {osh_metrics.get('emergence_index', 0.0):.4f}")
            lines.append("")
        
        # Time series summary
        ts_summary = snapshot.get('time_series_summary', {})
        if ts_summary:
            lines.append("=== TIME SERIES SUMMARY ===")
            for metric, data in ts_summary.items():
                if isinstance(data, dict) and 'current' in data:
                    lines.append(f"{metric}: {data['current']:.4f} "
                               f"(avg: {data.get('mean', 0.0):.4f}, "
                               f"trend: {data.get('trend', 'unknown')})")
            lines.append("")
        
        # Event summary
        event_summary = snapshot.get('event_summary', {})
        if event_summary:
            lines.append("=== EVENT SUMMARY ===")
            lines.append(f"Total Events: {event_summary.get('total_events', 0)}")
            lines.append(f"Recent Events: {event_summary.get('recent_events', 0)}")
            
            events_by_category = event_summary.get('events_by_category', {})
            for category, count in events_by_category.items():
                lines.append(f"  {category.title()}: {count}")
            lines.append("")
        
        # Additional context
        if 'active_observer' in snapshot:
            lines.append(f"Active Observer: {snapshot.get('active_observer', 'None')}")
        if 'current_scope' in snapshot:
            lines.append(f"Current Scope: {snapshot.get('current_scope', 'global')}")
        
        lines.append(f"\nSnapshot Time: {datetime.fromtimestamp(snapshot.get('snapshot_timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')}")
        
        return "\n".join(lines)
    
    def _render_system_snapshots(self, width: int, height: int) -> Dict[str, Any]:
        """Render system snapshots timeline"""
        try:
            fig, ax = plt.subplots(figsize=(width/100, height/100),
                                 facecolor=self.osh_colors['background'])
            
            # Generate snapshot timeline visualization
            # For now, show OSH history as snapshots
            history = list(self.osh_history)
            
            if not history:
                ax.text(0.5, 0.5, 'No snapshot data available', ha='center', va='center',
                       color=self.osh_colors['text'], transform=ax.transAxes, fontsize=14)
                ax.set_title('System Snapshots', color=self.osh_colors['text'], 
                            fontsize=16, fontweight='bold')
                ax.axis('off')
            else:
                # Extract data
                timestamps = [h['timestamp'] for h in history]
                coherence_values = [h['coherence_mean'] for h in history]
                entropy_values = [h['entropy_mean'] for h in history]
                rsp_values = [h['rsp_mean'] for h in history]
                
                # Convert to relative time
                if timestamps:
                    base_time = min(timestamps)
                    rel_times = [(t - base_time) for t in timestamps]
                else:
                    rel_times = list(range(len(history)))
                
                # Create timeline plot
                ax.plot(rel_times, coherence_values, 'o-', color=self.osh_colors['coherence'], 
                       label='Coherence', linewidth=2, markersize=6)
                ax.plot(rel_times, entropy_values, 's-', color=self.osh_colors['entropy'], 
                       label='Entropy', linewidth=2, markersize=6)
                
                # Secondary axis for RSP
                ax2 = ax.twinx()
                ax2.plot(rel_times, rsp_values, '^-', color=self.osh_colors['rsp'], 
                        label='RSP', linewidth=2, markersize=6)
                ax2.set_ylabel('RSP', color=self.osh_colors['text'])
                ax2.tick_params(colors=self.osh_colors['text'])
                
                # Style primary axis
                ax.set_xlabel('Time (s)', color=self.osh_colors['text'])
                ax.set_ylabel('Coherence / Entropy', color=self.osh_colors['text'])
                ax.set_title('System Snapshots Timeline', color=self.osh_colors['text'], 
                            fontsize=16, fontweight='bold')
                ax.legend(loc='upper left', facecolor=self.osh_colors['background'], 
                         edgecolor=self.osh_colors['text'])
                ax2.legend(loc='upper right', facecolor=self.osh_colors['background'], 
                          edgecolor=self.osh_colors['text'])
                ax.grid(True, alpha=0.3, color=self.osh_colors['grid'])
                ax.set_ylim(0, 1)
                ax2.set_ylim(0, max(rsp_values) * 1.1 if rsp_values else 1)
            
            # Style axes
            ax.set_facecolor(self.osh_colors['background'])
            ax.tick_params(colors=self.osh_colors['text'])
            for spine in ax.spines.values():
                spine.set_color(self.osh_colors['text'])
            
            if 'ax2' in locals():
                for spine in ax2.spines.values():
                    spine.set_color(self.osh_colors['text'])
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'system_snapshots',
                'snapshot_count': len(history)
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering system snapshots: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'system_snapshots'
            }
    
    def _render_recursive_boundaries(self, width: int, height: int) -> Dict[str, Any]:
        """Render recursive boundary visualization"""
        try:
            fig, ax = plt.subplots(figsize=(width/100, height/100),
                                 facecolor=self.osh_colors['background'])
            
            # Get recursive mechanics data
            if self.recursive_mechanics and hasattr(self.recursive_mechanics, 'get_system_statistics'):
                try:
                    recursive_stats = self.recursive_mechanics.get_system_statistics()
                    self._render_recursive_hierarchy(ax, recursive_stats)
                except Exception as e:
                    self.logger.warning(f"Could not get recursive statistics: {e}")
                    self._render_recursive_placeholder(ax)
            else:
                self._render_recursive_placeholder(ax)
            
            ax.set_title('Recursive System Boundaries', color=self.osh_colors['text'], 
                        fontsize=16, fontweight='bold')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'recursive_boundaries'
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering recursive boundaries: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'recursive_boundaries'
            }
    
    def _render_recursive_hierarchy(self, ax, recursive_stats: Dict[str, Any]):
        """Render recursive system hierarchy"""
        # Extract hierarchy information
        max_depth = recursive_stats.get('max_depth', 3)
        system_count = recursive_stats.get('system_count', 5)
        boundary_count = recursive_stats.get('boundary_count', 8)
        
        # Create hierarchical visualization
        ax.set_xlim(0, 10)
        ax.set_ylim(0, max_depth + 1)
        
        # Draw recursive levels
        level_colors = [self.osh_colors['coherence'], self.osh_colors['entropy'], 
                       self.osh_colors['strain'], self.osh_colors['rsp']]
        
        for level in range(max_depth + 1):
            # Draw level background
            rect = Rectangle((0, level), 10, 0.8, 
                           facecolor=level_colors[level % len(level_colors)], 
                           alpha=0.2, edgecolor=self.osh_colors['text'])
            ax.add_patch(rect)
            
            # Draw systems in level
            systems_in_level = max(1, system_count // (level + 1))
            for i in range(systems_in_level):
                x_pos = 1 + i * (8 / max(1, systems_in_level - 1)) if systems_in_level > 1 else 5
                circle = Circle((x_pos, level + 0.4), 0.3, 
                              facecolor=level_colors[level % len(level_colors)], 
                              edgecolor=self.osh_colors['text'], linewidth=2)
                ax.add_patch(circle)
                
                # System label
                ax.text(x_pos, level + 0.4, f'S{level}{i}', ha='center', va='center',
                       color=self.osh_colors['text'], fontsize=8, fontweight='bold')
                
                # Draw connections to parent level
                if level > 0:
                    parent_x = 5  # Simplified - connect to center of parent level
                    ax.plot([x_pos, parent_x], [level + 0.1, level - 0.1], 
                           color=self.osh_colors['text'], linestyle='--', alpha=0.7)
        
        # Draw boundary indicators
        for i in range(min(boundary_count, 5)):
            boundary_y = i * (max_depth / 5) if max_depth > 0 else 0
            ax.axhline(y=boundary_y + 0.8, color=self.osh_colors['strain'], 
                      linestyle=':', linewidth=2, alpha=0.8)
            ax.text(9.5, boundary_y + 0.8, 'B', ha='center', va='center',
                   color=self.osh_colors['strain'], fontweight='bold')
        
        # Labels and styling
        ax.set_xlabel('System Hierarchy', color=self.osh_colors['text'])
        ax.set_ylabel('Recursive Depth', color=self.osh_colors['text'])
        ax.tick_params(colors=self.osh_colors['text'])
        ax.set_facecolor(self.osh_colors['background'])
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', 
                      markerfacecolor=self.osh_colors['coherence'], markersize=10, label='Systems'),
            plt.Line2D([0], [0], color=self.osh_colors['text'], linestyle='--', label='Connections'),
            plt.Line2D([0], [0], color=self.osh_colors['strain'], linestyle=':', 
                      linewidth=2, label='Boundaries')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 facecolor=self.osh_colors['background'], edgecolor=self.osh_colors['text'])
        
        for spine in ax.spines.values():
            spine.set_color(self.osh_colors['text'])
    
    def _render_recursive_placeholder(self, ax):
        """Render placeholder for recursive boundaries when no data available"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5)
        
        # Create sample recursive structure
        levels = 4
        colors = [self.osh_colors['coherence'], self.osh_colors['entropy'], 
                 self.osh_colors['strain'], self.osh_colors['rsp']]
        
        for level in range(levels):
            # Level background
            rect = Rectangle((0, level), 10, 0.8, 
                           facecolor=colors[level], alpha=0.2, 
                           edgecolor=self.osh_colors['text'])
            ax.add_patch(rect)
            
            # Sample systems
            for i in range(3 - level):
                x_pos = 2 + i * 3
                circle = Circle((x_pos, level + 0.4), 0.2, 
                              facecolor=colors[level], alpha=0.8,
                              edgecolor=self.osh_colors['text'])
                ax.add_patch(circle)
        
        # Sample boundaries
        for i in range(3):
            ax.axhline(y=i + 0.8, color=self.osh_colors['strain'], 
                      linestyle=':', linewidth=2, alpha=0.6)
        
        ax.text(5, 2, 'No Recursive Data\n(Sample Structure)', 
               ha='center', va='center', color=self.osh_colors['text'], 
               fontsize=12, style='italic')
        
        ax.set_xlabel('System Hierarchy', color=self.osh_colors['text'])
        ax.set_ylabel('Recursive Depth', color=self.osh_colors['text'])
        ax.tick_params(colors=self.osh_colors['text'])
        ax.set_facecolor(self.osh_colors['background'])
        
        for spine in ax.spines.values():
            spine.set_color(self.osh_colors['text'])
    
    def render_osh_timeline(self, width: int, height: int) -> Dict[str, Any]:
        """
        Render specialized OSH timeline visualization.
        
        Args:
            width: Timeline width in pixels
            height: Timeline height in pixels
            
        Returns:
            Dict containing render results and OSH analysis
        """
        try:
            fig, axes = plt.subplots(3, 1, figsize=(width/100, height/100),
                                   facecolor=self.osh_colors['background'])
            fig.suptitle('OSH Temporal Analysis', color=self.osh_colors['text'], 
                        fontsize=18, fontweight='bold')
            
            # Get time series data
            n_points = len(self.time_series['timestamps'])
            if n_points < 2:
                for ax in axes:
                    ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                           color=self.osh_colors['text'], transform=ax.transAxes)
                plt.tight_layout()
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                           bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode()
                plt.close(fig)
                return {
                    'success': True,
                    'image_data': f"data:image/png;base64,{image_data}",
                    'message': 'Insufficient data for OSH timeline'
                }
            
            # Extract data
            timestamps = list(self.time_series['timestamps'])
            base_time = timestamps[0]
            rel_times = [(t - base_time) for t in timestamps]
            
            coherence_data = list(self.time_series['average_coherence'])
            entropy_data = list(self.time_series['average_entropy'])
            strain_data = list(self.time_series['average_strain'])
            rsp_data = list(self.time_series['rsp'])
            
            # OSH Primary Metrics (top)
            ax1 = axes[0]
            ax1.plot(rel_times, coherence_data, color=self.osh_colors['coherence'], 
                    linewidth=3, label='Coherence', alpha=0.9)
            ax1.plot(rel_times, entropy_data, color=self.osh_colors['entropy'], 
                    linewidth=3, label='Entropy', alpha=0.9)
            ax1.plot(rel_times, strain_data, color=self.osh_colors['strain'], 
                    linewidth=3, label='Strain', alpha=0.9)
            
            # Add threshold lines
            ax1.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Critical Threshold')
            ax1.axhline(y=0.8, color=self.osh_colors['strain'], linestyle=':', alpha=0.7, label='Danger Zone')
            
            # Highlight critical regions
            for i, (c, e, s) in enumerate(zip(coherence_data, entropy_data, strain_data)):
                if s > 0.8 or c < 0.2:  # Critical conditions
                    ax1.axvspan(rel_times[max(0, i-1)], rel_times[min(len(rel_times)-1, i+1)], 
                               color='red', alpha=0.2)
            
            ax1.set_ylabel('OSH Metrics', color=self.osh_colors['text'])
            ax1.set_title('Primary OSH Indicators', color=self.osh_colors['text'], fontweight='bold')
            ax1.legend(loc='upper right', facecolor=self.osh_colors['background'], 
                      edgecolor=self.osh_colors['text'])
            ax1.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            ax1.set_ylim(0, 1)
            
            # RSP Evolution (middle)
            ax2 = axes[1]
            ax2.plot(rel_times, rsp_data, color=self.osh_colors['rsp'], 
                    linewidth=3, marker='o', markersize=4, label='RSP')
            
            # RSP classification regions
            ax2.axhspan(0, 0.5, color='red', alpha=0.1, label='Critical/Low')
            ax2.axhspan(0.5, 2.0, color='yellow', alpha=0.1, label='Moderate')
            ax2.axhspan(2.0, 5.0, color='green', alpha=0.1, label='High')
            
            # Add RSP trend analysis
            if len(rsp_data) > 5:
                # Calculate moving average
                window_size = min(5, len(rsp_data) // 3)
                rsp_smooth = np.convolve(rsp_data, np.ones(window_size)/window_size, mode='valid')
                smooth_times = rel_times[window_size-1:]
                ax2.plot(smooth_times, rsp_smooth, color='white', linewidth=2, 
                        linestyle='--', alpha=0.8, label='Trend')
            
            ax2.set_ylabel('RSP Value', color=self.osh_colors['text'])
            ax2.set_title('Recursive Simulation Potential', color=self.osh_colors['text'], fontweight='bold')
            ax2.legend(loc='upper right', facecolor=self.osh_colors['background'], 
                      edgecolor=self.osh_colors['text'])
            ax2.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            
            # Emergence Analysis (bottom)
            ax3 = axes[2]
            
            # Calculate emergence indicators
            consciousness_data = list(self.time_series['consciousness_quotient'])
            emergence_events = []
            stability_scores = []
            
            for i in range(len(coherence_data)):
                # Emergence event detection
                if (i > 0 and coherence_data[i] > coherence_data[i-1] + 0.1 and 
                    entropy_data[i] < entropy_data[i-1] - 0.05):
                    emergence_events.append((rel_times[i], 1.0))
                else:
                    emergence_events.append((rel_times[i], 0.0))
                
                # Stability score
                if i >= 2:
                    coherence_stability = 1.0 - abs(coherence_data[i] - coherence_data[i-1])
                    entropy_stability = 1.0 - abs(entropy_data[i] - entropy_data[i-1])
                    stability = (coherence_stability + entropy_stability) / 2.0
                    stability_scores.append(stability)
                else:
                    stability_scores.append(1.0)
            
            # Plot emergence and stability
            emergence_times, emergence_values = zip(*emergence_events)
            ax3.fill_between(emergence_times, emergence_values, alpha=0.3, 
                           color=self.osh_colors['emergence'], label='Emergence Events')
            ax3.plot(rel_times, stability_scores, color='white', linewidth=2, 
                    label='System Stability')
            ax3.plot(rel_times, consciousness_data, color=self.osh_colors['consciousness'], 
                    linewidth=2, label='Consciousness Quotient')
            
            ax3.set_xlabel('Time (s)', color=self.osh_colors['text'])
            ax3.set_ylabel('Emergence Indicators', color=self.osh_colors['text'])
            ax3.set_title('Consciousness and Emergence Analysis', color=self.osh_colors['text'], fontweight='bold')
            ax3.legend(loc='upper right', facecolor=self.osh_colors['background'], 
                      edgecolor=self.osh_colors['text'])
            ax3.grid(True, alpha=0.3, color=self.osh_colors['grid'])
            ax3.set_ylim(0, 1)
            
            # Style all axes
            for ax in axes:
                ax.set_facecolor(self.osh_colors['background'])
                ax.tick_params(colors=self.osh_colors['text'])
                for spine in ax.spines.values():
                    spine.set_color(self.osh_colors['text'])
            
            plt.tight_layout()
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', facecolor=self.osh_colors['background'], 
                       bbox_inches='tight', dpi=150 if self.scientific_mode else 100)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            plt.close(fig)
            
            # Calculate analysis metrics
            analysis_metrics = {
                'stability_score': np.mean(stability_scores) if stability_scores else 0.0,
                'emergence_events': sum(emergence_values),
                'rsp_trend': 'increasing' if len(rsp_data) > 1 and rsp_data[-1] > rsp_data[0] else 'decreasing',
                'coherence_trend': 'increasing' if len(coherence_data) > 1 and coherence_data[-1] > coherence_data[0] else 'decreasing',
                'critical_periods': sum(1 for s in strain_data if s > 0.8),
                'consciousness_peak': max(consciousness_data) if consciousness_data else 0.0
            }
            
            return {
                'success': True,
                'image_data': f"data:image/png;base64,{image_data}",
                'visualization': 'osh_timeline',
                'analysis_metrics': analysis_metrics,
                'osh_summary': self._get_current_osh_summary()
            }
            
        except Exception as e:
            self.logger.error(f"Error rendering OSH timeline: {e}")
            plt.close('all')
            return {
                'success': False,
                'error': str(e),
                'visualization': 'osh_timeline'
            }
    
    def export_simulation_data(self, format: str = 'json') -> Dict[str, Any]:
        """
        Export simulation data in specified format.
        
        Args:
            format: Export format ('json', 'csv', 'xlsx')
            
        Returns:
            Dict containing export results
        """
        try:
            # Collect all data
            export_data = {
                'metadata': {
                    'export_time': datetime.now().isoformat(),
                    'simulation_panel_version': '1.0.0',
                    'osh_framework_version': '1.0.0'
                },
                'simulation_statistics': self.get_simulation_statistics(),
                'time_series': {key: list(series) for key, series in self.time_series.items()},
                'event_log': list(self.event_log),
                'osh_metrics': self._get_current_osh_summary(),
                'osh_history': list(self.osh_history),
                'performance_metrics': {
                    key: list(value) if hasattr(value, '__iter__') and not isinstance(value, str) else value
                    for key, value in self.performance_metrics.items()
                }
            }
            
            if format.lower() == 'json':
                return {
                    'success': True,
                    'format': 'json',
                    'data': export_data,
                    'filename': f"recursia_simulation_{int(time.time())}.json"
                }
            
            elif format.lower() == 'csv':
                # Convert time series to CSV format
                csv_data = []
                timestamps = export_data['time_series']['timestamps']
                
                for i, timestamp in enumerate(timestamps):
                    row = {'timestamp': timestamp}
                    for key, series in export_data['time_series'].items():
                        if key != 'timestamps' and i < len(series):
                            row[key] = series[i]
                    csv_data.append(row)
                
                return {
                    'success': True,
                    'format': 'csv',
                    'data': csv_data,
                    'metadata': export_data['metadata'],
                    'filename': f"recursia_simulation_{int(time.time())}.csv"
                }
            
            else:
                return {
                    'success': False,
                    'error': f"Unsupported export format: {format}",
                    'supported_formats': ['json', 'csv']
                }
                
        except Exception as e:
            self.logger.error(f"Error exporting simulation data: {e}")
            return {
                'success': False,
                'error': str(e),
                'format': format
            }
    
    def reset_panel(self) -> bool:
        """Reset panel state and clear all data"""
        try:
            with self._lock:
                # Clear time series
                for series in self.time_series.values():
                    series.clear()
                
                # Clear event log
                self.event_log.clear()
                
                # Clear OSH history
                self.osh_history.clear()
                
                # Reset OSH metrics
                self.current_osh_metrics = OSHFieldMetrics()
                
                # Clear performance metrics
                for key in self.performance_metrics:
                    if hasattr(self.performance_metrics[key], 'clear'):
                        self.performance_metrics[key].clear()
                    else:
                        self.performance_metrics[key] = 0.0
                
                # Clear cache
                if self.enable_caching and self.visualization_cache:
                    self.visualization_cache.clear()
                
                self.logger.info("Simulation panel reset successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error resetting simulation panel: {e}")
            return False
    
    def get_panel_statistics(self) -> Dict[str, Any]:
        """Get comprehensive panel statistics"""
        try:
            return {
                'panel_type': 'SimulationPanel',
                'current_visualization': self.current_visualization,
                'data_points': {
                    'time_series_length': len(self.time_series['timestamps']),
                    'event_log_length': len(self.event_log),
                    'osh_history_length': len(self.osh_history)
                },
                'performance': self._get_performance_statistics(),
                'configuration': {
                    'max_series_length': self.max_series_length,
                    'update_interval': self.update_interval,
                    'scientific_mode': self.scientific_mode,
                    'enable_caching': self.enable_caching
                },
                'cache_info': {
                    'cache_enabled': self.enable_caching,
                    'cache_size': len(self.visualization_cache) if self.visualization_cache else 0
                },
                'osh_classification': {
                    'rsp_level': self._classify_rsp_level(self._get_current_osh_summary().get('rsp', 0.0)),
                    'coherence_level': self._classify_coherence_level(self._get_current_osh_summary().get('coherence', 0.0)),
                    'emergence_level': self._classify_emergence_level(self._get_current_osh_summary().get('emergence_index', 0.0))
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting panel statistics: {e}")
            return {'error': str(e)}


def create_simulation_panel(interpreter=None, execution_context=None, event_system=None,
                          memory_field=None, recursive_mechanics=None, quantum_renderer=None,
                          coherence_renderer=None, config=None) -> SimulationPanel:
    """
    Factory function to create a fully configured SimulationPanel.
    
    Args:
        interpreter: RecursiaInterpreter instance
        execution_context: ExecutionContext for simulation state
        event_system: EventSystem for real-time events
        memory_field: MemoryFieldPhysics for field dynamics
        recursive_mechanics: RecursiveMechanics for recursion analysis
        quantum_renderer: QuantumRenderer for quantum visualizations
        coherence_renderer: CoherenceRenderer for OSH visualizations
        config: Configuration dictionary
        
    Returns:
        Configured SimulationPanel instance
    """
    # Default configuration
    default_config = {
        'max_series_length': 2000,
        'update_interval': 0.05,  # 20 FPS
        'scientific_mode': True,
        'enable_caching': True,
        'enable_real_time': True,
        'log_level': 'INFO'
    }
    
    # Merge with provided config
    final_config = default_config.copy()
    if config:
        final_config.update(config)
    
    # Create and return panel
    panel = SimulationPanel(
        interpreter=interpreter,
        execution_context=execution_context,
        event_system=event_system,
        memory_field=memory_field,
        recursive_mechanics=recursive_mechanics,
        quantum_renderer=quantum_renderer,
        coherence_renderer=coherence_renderer,
        config=final_config
    )
    
    return panel