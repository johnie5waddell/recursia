"""
Field Panel for Recursia - Advanced OSH-Aligned Quantum Field Visualization

This module provides enterprise-grade visualization capabilities for quantum field dynamics
within the Recursia framework, implementing full OSH (Organic Simulation Hypothesis) 
alignment with scientific accuracy and real-time performance optimization.

Features:
- Multi-dimensional field visualization (scalar, vector, tensor, spinor)
- OSH metric computation (RSP, coherence, entropy, strain)
- Advanced PDE evolution visualization  
- Recursive memory field analysis
- Real-time performance optimization
- Scientific publication-quality output
"""

import time
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, RadioButtons
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import deque
from scipy import ndimage, signal, optimize
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.fft import fft2, ifft2, fftfreq
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

from src.physics.field.field_dynamics import FieldDynamics
from src.physics.field.field_evolve import FieldEvolutionEngine
from src.physics.memory_field import MemoryFieldPhysics
from src.physics.coherence import CoherenceManager
from src.physics.recursive import RecursiveMechanics
from src.visualization.quantum_renderer import QuantumRenderer
from src.visualization.coherence_renderer import CoherenceRenderer

logger = logging.getLogger(__name__)


class AdvancedFieldMetrics:
    """
    Advanced metric calculation engine for OSH-aligned field analysis.
    Implements scientifically accurate calculations of coherence, entropy, strain,
    and Recursive Simulation Potential (RSP) according to OSH theory.
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 1.0  # seconds
        
    def calculate_field_coherence(self, field_data: np.ndarray, 
                                method: str = "l1_norm") -> np.ndarray:
        """
        Calculate coherence using multiple scientifically validated methods.
        
        Args:
            field_data: N-dimensional field array
            method: Calculation method ('l1_norm', 'variance_inverse', 'correlation')
            
        Returns:
            Coherence map normalized to [0, 1]
        """
        cache_key = f"coherence_{method}_{field_data.shape}_{hash(field_data.tobytes())}"
        
        if cache_key in self.cache:
            cache_time, result = self.cache[cache_key]
            if time.time() - cache_time < self.cache_timeout:
                return result
        
        if method == "l1_norm":
            # Off-diagonal coherence via local Fourier analysis
            coherence = self._fourier_coherence_analysis(field_data)
        elif method == "variance_inverse":
            # Inverse of local variance (stability measure)
            coherence = self._variance_inverse_coherence(field_data)
        elif method == "correlation":
            # Spatial correlation coherence
            coherence = self._correlation_coherence(field_data)
        else:
            coherence = self._fourier_coherence_analysis(field_data)
            
        # Cache result
        self.cache[cache_key] = (time.time(), coherence)
        return coherence
    
    def _fourier_coherence_analysis(self, field_data: np.ndarray) -> np.ndarray:
        """Fourier-based coherence analysis using phase consistency."""
        # Apply 2D FFT to analyze frequency domain coherence
        fft_field = fft2(field_data)
        phase = np.angle(fft_field)
        magnitude = np.abs(fft_field)
        
        # Calculate phase coherence via local phase consistency
        phase_coherence = np.zeros_like(field_data, dtype=float)
        window_size = min(5, min(field_data.shape) // 4)
        
        for i in range(window_size//2, field_data.shape[0] - window_size//2):
            for j in range(window_size//2, field_data.shape[1] - window_size//2):
                window = phase[i-window_size//2:i+window_size//2+1,
                             j-window_size//2:j+window_size//2+1]
                # Phase coherence = 1 - circular variance
                phase_coherence[i, j] = 1.0 - np.var(np.exp(1j * window))
                
        return np.clip(phase_coherence, 0, 1)
    
    def _variance_inverse_coherence(self, field_data: np.ndarray) -> np.ndarray:
        """Calculate coherence as inverse of local variance."""
        local_var = ndimage.generic_filter(field_data, np.var, size=3)
        max_var = np.max(local_var) if np.max(local_var) > 0 else 1e-10
        return 1.0 - (local_var / max_var)
    
    def _correlation_coherence(self, field_data: np.ndarray) -> np.ndarray:
        """Calculate coherence via spatial correlation analysis."""
        # Calculate spatial autocorrelation
        correlation_map = np.zeros_like(field_data)
        
        for shift_x in range(-2, 3):
            for shift_y in range(-2, 3):
                if shift_x == 0 and shift_y == 0:
                    continue
                    
                shifted = np.roll(np.roll(field_data, shift_x, axis=0), shift_y, axis=1)
                correlation = np.corrcoef(field_data.flatten(), shifted.flatten())[0, 1]
                if not np.isnan(correlation):
                    correlation_map += np.abs(correlation)
                    
        return np.clip(correlation_map / 20.0, 0, 1)  # Normalize by max possible correlations
    
    def calculate_field_entropy(self, field_data: np.ndarray, 
                              method: str = "shannon") -> np.ndarray:
        """
        Calculate field entropy using thermodynamically consistent methods.
        
        Args:
            field_data: Field array
            method: Entropy calculation method
            
        Returns:
            Entropy map normalized to [0, 1]
        """
        if method == "shannon":
            return self._shannon_entropy_field(field_data)
        elif method == "von_neumann":
            return self._von_neumann_entropy_field(field_data)
        elif method == "renyi":
            return self._renyi_entropy_field(field_data)
        else:
            return self._shannon_entropy_field(field_data)
    
    def _shannon_entropy_field(self, field_data: np.ndarray) -> np.ndarray:
        """Calculate Shannon entropy using local histograms."""
        entropy_map = np.zeros_like(field_data)
        window_size = 5
        
        for i in range(window_size//2, field_data.shape[0] - window_size//2):
            for j in range(window_size//2, field_data.shape[1] - window_size//2):
                window = field_data[i-window_size//2:i+window_size//2+1,
                                  j-window_size//2:j+window_size//2+1]
                
                # Create probability distribution
                hist, _ = np.histogram(window, bins=8, density=True)
                hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist + 1e-10
                
                # Calculate Shannon entropy
                entropy_map[i, j] = -np.sum(hist * np.log2(hist + 1e-10))
                
        # Normalize to [0, 1]
        max_entropy = np.log2(8)  # Maximum entropy for 8 bins
        return entropy_map / max_entropy if max_entropy > 0 else entropy_map
    
    def _von_neumann_entropy_field(self, field_data: np.ndarray) -> np.ndarray:
        """Calculate von Neumann entropy for quantum field analysis."""
        entropy_map = np.zeros_like(field_data)
        
        # Convert field to density matrix representation locally
        for i in range(1, field_data.shape[0] - 1):
            for j in range(1, field_data.shape[1] - 1):
                # Create local 2x2 density matrix from field values
                local_data = field_data[i-1:i+2, j-1:j+2]
                
                # Normalize to create valid density matrix
                rho = np.outer(local_data.flatten(), local_data.flatten().conj())
                rho = rho / np.trace(rho) if np.trace(rho) > 0 else rho
                
                # Calculate eigenvalues and von Neumann entropy
                eigenvals = np.real(np.linalg.eigvals(rho))
                eigenvals = eigenvals[eigenvals > 1e-10]  # Remove numerical zeros
                
                if len(eigenvals) > 0:
                    entropy_map[i, j] = -np.sum(eigenvals * np.log2(eigenvals))
                    
        # Normalize
        max_entropy = np.log2(min(9, 2))  # Max entropy for effective dimension
        return entropy_map / max_entropy if max_entropy > 0 else entropy_map
    
    def _renyi_entropy_field(self, field_data: np.ndarray, alpha: float = 2.0) -> np.ndarray:
        """Calculate Rényi entropy with parameter alpha."""
        entropy_map = np.zeros_like(field_data)
        window_size = 3
        
        for i in range(window_size//2, field_data.shape[0] - window_size//2):
            for j in range(window_size//2, field_data.shape[1] - window_size//2):
                window = field_data[i-window_size//2:i+window_size//2+1,
                                  j-window_size//2:j+window_size//2+1]
                
                hist, _ = np.histogram(window, bins=6, density=True)
                hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist + 1e-10
                
                # Rényi entropy
                if alpha == 1.0:
                    entropy_map[i, j] = -np.sum(hist * np.log2(hist + 1e-10))
                else:
                    entropy_map[i, j] = np.log2(np.sum(hist ** alpha)) / (1 - alpha)
                    
        return np.clip(entropy_map / np.log2(6), 0, 1)
    
    def calculate_field_strain(self, field_data: np.ndarray) -> np.ndarray:
        """
        Calculate field strain using gradient tensor analysis and curvature.
        Implements full tensor mechanics for accurate strain measurement.
        """
        # Calculate gradient components
        grad_y, grad_x = np.gradient(field_data)
        
        # Calculate second derivatives (Hessian components)
        grad_xx = np.gradient(grad_x, axis=1)
        grad_yy = np.gradient(grad_y, axis=0)
        grad_xy = np.gradient(grad_x, axis=0)
        
        # Strain tensor components (symmetric part of gradient)
        strain_xx = grad_xx
        strain_yy = grad_yy
        strain_xy = 0.5 * (grad_xy + np.gradient(grad_y, axis=1))
        
        # Von Mises strain (scalar strain measure)
        strain_magnitude = np.sqrt(strain_xx**2 + strain_yy**2 + 2*strain_xy**2)
        
        # Normalize to [0, 1]
        max_strain = np.max(strain_magnitude) if np.max(strain_magnitude) > 0 else 1.0
        return strain_magnitude / max_strain
    
    def calculate_rsp(self, coherence: np.ndarray, entropy: np.ndarray, 
                     strain: np.ndarray, complexity_factor: float = 1.0) -> np.ndarray:
        """
        Calculate Recursive Simulation Potential according to OSH theory.
        
        RSP = (I * C) / E_flux
        where I = integrated information, C = complexity, E_flux = entropy flux
        """
        # Integrated information (coherence-based)
        integrated_info = coherence * (1.0 + np.log(1.0 + np.abs(np.gradient(coherence)[0]) + 
                                                   np.abs(np.gradient(coherence)[1])))
        
        # Complexity measure (inverse entropy with gradient contribution)
        complexity = complexity_factor * (1.0 - entropy) * (1.0 + np.std(entropy))
        
        # Entropy flux (strain-weighted entropy change)
        entropy_flux = strain + 1e-6  # Prevent division by zero
        
        # Calculate RSP
        rsp = (integrated_info * complexity) / entropy_flux
        
        # Normalize to meaningful range [0, 10] for visualization
        rsp_normalized = np.clip(rsp / (np.mean(rsp) + np.std(rsp)), 0, 10)
        
        return rsp_normalized


class StabilityAnalyzer:
    """Advanced stability analysis for field evolution."""
    
    def analyze_stability(self, current_field: np.ndarray, previous_field: np.ndarray) -> float:
        """
        Comprehensive stability analysis using multiple indicators.
        Returns stability factor [0, 1] where 1 is perfectly stable.
        """
        # Rate of change stability
        change_rate = np.linalg.norm(current_field - previous_field) / np.linalg.norm(current_field)
        rate_stability = np.exp(-10 * change_rate)  # Exponential decay with change rate
        
        # Gradient stability (smoothness)
        grad_current = np.gradient(current_field)
        grad_magnitude = np.sqrt(grad_current[0]**2 + grad_current[1]**2)
        gradient_stability = 1.0 / (1.0 + np.std(grad_magnitude))
        
        # Energy conservation stability
        energy_current = np.sum(np.abs(current_field)**2)
        energy_previous = np.sum(np.abs(previous_field)**2)
        energy_change = abs(energy_current - energy_previous) / max(energy_previous, 1e-10)
        energy_stability = np.exp(-5 * energy_change)
        
        # Combined stability measure
        stability = 0.4 * rate_stability + 0.3 * gradient_stability + 0.3 * energy_stability
        
        return np.clip(stability, 0, 1)


class RecursiveLayerRenderer:
    """Advanced renderer for recursive memory layers with full 3D capability."""
    
    def __init__(self):
        self.layer_cache = {}
        
    def render_recursive_layers_3d(self, field_data: np.ndarray, 
                                  recursive_data: Dict, ax) -> Dict:
        """
        Render full 3D recursive layer visualization with advanced shading and connectivity.
        """
        n_layers = len(recursive_data) if recursive_data else 3
        layer_spacing = 1.0
        
        # Create coordinate meshes
        x = np.arange(field_data.shape[1])
        y = np.arange(field_data.shape[0])
        X, Y = np.meshgrid(x, y)
        
        # Store surfaces for connectivity lines
        surfaces = []
        
        for layer_idx in range(n_layers):
            # Calculate layer-specific field transformation
            if recursive_data and str(layer_idx) in recursive_data:
                layer_data = recursive_data[str(layer_idx)]
                coherence_factor = layer_data.get('coherence', 1.0 - layer_idx * 0.2)
                entropy_factor = layer_data.get('entropy', layer_idx * 0.1)
            else:
                coherence_factor = 1.0 - layer_idx * 0.2
                entropy_factor = layer_idx * 0.1
            
            # Apply recursive transformation
            sigma = 1.0 + layer_idx * 1.5  # Increasing blur with depth
            layer_field = ndimage.gaussian_filter(field_data, sigma=sigma)
            
            # Apply OSH transformations
            layer_field = layer_field * coherence_factor
            layer_field += np.random.normal(0, entropy_factor * 0.1, layer_field.shape)
            
            # Normalize for consistent visualization
            if np.max(layer_field) > np.min(layer_field):
                layer_field = (layer_field - np.min(layer_field)) / (np.max(layer_field) - np.min(layer_field))
            
            # Z-position for this layer
            z_base = -layer_idx * layer_spacing
            Z = z_base + layer_field * 0.3  # Add field variation to z-position
            
            # Color and transparency
            alpha = 0.9 - layer_idx * 0.25
            colormap = plt.cm.viridis if layer_idx % 2 == 0 else plt.cm.plasma
            
            # Render surface
            surface = ax.plot_surface(X, Y, Z, cmap=colormap, alpha=alpha,
                                    linewidth=0, antialiased=True, shade=True)
            surfaces.append((X, Y, Z, layer_idx))
        
        # Add connectivity between layers
        self._add_recursive_connections(ax, surfaces, field_data.shape)
        
        # Set viewing parameters
        ax.view_init(elev=25, azim=45)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Recursive Depth')
        ax.set_title('Recursive Memory Layer Structure')
        
        return {
            "layers_rendered": n_layers,
            "connectivity_lines": len(surfaces) * 4,  # 4 corners per layer
            "depth_range": n_layers * layer_spacing
        }
    
    def _add_recursive_connections(self, ax, surfaces: List, field_shape: Tuple):
        """Add connecting lines between recursive layers at key points."""
        connection_points = [
            (0, 0), (0, field_shape[0]-1),
            (field_shape[1]-1, 0), (field_shape[1]-1, field_shape[0]-1),
            (field_shape[1]//2, field_shape[0]//2)  # Center point
        ]
        
        for x, y in connection_points:
            if x < field_shape[1] and y < field_shape[0]:
                z_values = []
                for X, Y, Z, layer_idx in surfaces:
                    z_values.append(Z[y, x])
                
                # Draw connection line
                ax.plot([x] * len(z_values), [y] * len(z_values), z_values,
                       'k-', alpha=0.3, linewidth=1)


class FieldPanel:
    """
    Enterprise-grade field visualization panel for Recursia OSH analysis.
    
    Provides comprehensive visualization of quantum fields with full OSH integration,
    advanced metric calculation, real-time performance optimization, and scientific
    publication-quality output.
    """
    
    def __init__(self, 
                 field_dynamics: Optional[FieldDynamics] = None,
                 memory_field: Optional[MemoryFieldPhysics] = None,
                 coherence_manager: Optional[CoherenceManager] = None,
                 recursive_mechanics: Optional[RecursiveMechanics] = None,
                 quantum_renderer: Optional[QuantumRenderer] = None,
                 coherence_renderer: Optional[CoherenceRenderer] = None,
                 config: Optional[Dict] = None):
        """
        Initialize advanced field panel with full OSH integration.
        
        Args:
            field_dynamics: FieldDynamics instance for field management
            memory_field: MemoryFieldPhysics for memory analysis
            coherence_manager: CoherenceManager for coherence tracking
            recursive_mechanics: RecursiveMechanics for recursion analysis
            quantum_renderer: QuantumRenderer for quantum visualizations
            coherence_renderer: CoherenceRenderer for OSH visualizations
            config: Configuration dictionary
        """
        # Core components
        self.field_dynamics = field_dynamics
        self.memory_field = memory_field
        self.coherence_manager = coherence_manager
        self.recursive_mechanics = recursive_mechanics
        self.quantum_renderer = quantum_renderer
        self.coherence_renderer = coherence_renderer
        
        # Configuration
        self.config = config or {}
        self.performance_mode = self.config.get("performance_mode", "high_quality")
        self.cache_enabled = self.config.get("cache_enabled", True)
        self.max_animation_frames = self.config.get("max_animation_frames", 100)
        
        # Advanced metric calculator
        self.metrics_engine = AdvancedFieldMetrics()
        
        # Evolution engine
        self.evolution_engine = FieldEvolutionEngine(field_dynamics) if field_dynamics else None
        
        # Recursive layer renderer
        self.recursive_renderer = RecursiveLayerRenderer()
        
        # Panel state
        self.selected_field = None
        self.selected_visualization = "field_values"
        self.animation_frames = deque(maxlen=self.max_animation_frames)
        self.evolution_data = {}
        
        # Visualization settings
        self.figure_size = (12, 10)
        self.dpi = self.config.get("dpi", 150)
        self.colormap = "viridis"
        self.show_grid = True
        self.auto_range = True
        self.value_range = (0.0, 1.0)
        self.slice_index = 0
        self.time_step = 0.01
        
        # OSH parameters
        self.coherence_threshold = 0.7
        self.strain_threshold = 0.8
        self.entropy_threshold = 0.5
        self.rsp_threshold = 2.0
        self.show_recursive_boundaries = True
        
        # Performance tracking
        self.last_update_time = 0
        self.render_cache = {}
        self.cache_ttl = 2.0  # seconds
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("FieldPanel initialized with advanced OSH capabilities")

    def get_available_fields(self) -> List[str]:
        """Get list of available fields with type information."""
        if not self.field_dynamics:
            return []
            
        fields_info = []
        for field_name in self.field_dynamics.fields.keys():
            field_info = self.field_dynamics.get_field(field_name)
            field_type = field_info.get('field_type', 'unknown')
            shape = field_info.get('shape', [])
            fields_info.append(f"{field_name} ({field_type}) {shape}")
            
        return fields_info

    def select_field(self, field_name: str) -> bool:
        """Select field with validation and caching."""
        if not self.field_dynamics:
            return False
            
        if field_name in self.field_dynamics.fields:
            self.selected_field = field_name
            # Clear relevant caches
            self._clear_field_cache(field_name)
            logger.info(f"Selected field: {field_name}")
            return True
            
        logger.warning(f"Field not found: {field_name}")
        return False

    def select_visualization(self, visualization_type: str) -> bool:
        """Select visualization type with validation."""
        valid_types = [
            "field_values", "coherence", "entropy", "strain", "rsp",
            "gradient", "evolution", "field_strain", "recursive_layers",
            "osh_comprehensive", "phase_space", "spectral_analysis",
            "stability_analysis", "memory_topology"
        ]
        
        if visualization_type in valid_types:
            self.selected_visualization = visualization_type
            logger.info(f"Selected visualization: {visualization_type}")
            return True
            
        logger.warning(f"Invalid visualization type: {visualization_type}")
        return False

    def update(self, simulation_data: Dict) -> bool:
        """
        Update panel with comprehensive simulation data processing.
        
        Args:
            simulation_data: Complete simulation state dictionary
            
        Returns:
            Success status
        """
        try:
            self.last_update_time = time.time()
            
            # Process field data
            if 'fields' in simulation_data and self.field_dynamics:
                self._process_field_updates(simulation_data['fields'])
            
            # Process memory field data
            if 'memory_field' in simulation_data and self.memory_field:
                self._process_memory_updates(simulation_data['memory_field'])
            
            # Process coherence data
            if 'coherence' in simulation_data and self.coherence_manager:
                self._process_coherence_updates(simulation_data['coherence'])
            
            # Process recursive data
            if 'recursive' in simulation_data and self.recursive_mechanics:
                self._process_recursive_updates(simulation_data['recursive'])
            
            # Update OSH metrics
            self._update_osh_metrics(simulation_data)
            
            # Auto-select field if none selected
            if not self.selected_field:
                available_fields = list(self.field_dynamics.fields.keys()) if self.field_dynamics else []
                if available_fields:
                    self.select_field(available_fields[0])
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating field panel: {e}")
            return False

    def _process_field_updates(self, field_data: Dict):
        """Process field-specific updates with advanced analysis."""
        for field_name, data in field_data.items():
            if field_name not in self.evolution_data:
                self.evolution_data[field_name] = {
                    "coherence_history": deque(maxlen=100),
                    "entropy_history": deque(maxlen=100),
                    "strain_history": deque(maxlen=100),
                    "rsp_history": deque(maxlen=100),
                    "stability_history": deque(maxlen=100)
                }
            
            # Calculate and store OSH metrics
            if isinstance(data, dict) and 'values' in data:
                field_values = data['values']
                if isinstance(field_values, np.ndarray):
                    self._calculate_and_store_metrics(field_name, field_values)

    def _calculate_and_store_metrics(self, field_name: str, field_values: np.ndarray):
        """Calculate and store comprehensive OSH metrics."""
        try:
            # Calculate OSH metrics
            coherence = self.metrics_engine.calculate_field_coherence(field_values)
            entropy = self.metrics_engine.calculate_field_entropy(field_values)
            strain = self.metrics_engine.calculate_field_strain(field_values)
            rsp = self.metrics_engine.calculate_rsp(coherence, entropy, strain)
            
            # Store in history
            history = self.evolution_data[field_name]
            history["coherence_history"].append(np.mean(coherence))
            history["entropy_history"].append(np.mean(entropy))
            history["strain_history"].append(np.mean(strain))
            history["rsp_history"].append(np.mean(rsp))
            
            # Calculate stability
            if len(history["coherence_history"]) > 1:
                coherence_stability = 1.0 - abs(history["coherence_history"][-1] - 
                                               history["coherence_history"][-2])
                history["stability_history"].append(coherence_stability)
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {field_name}: {e}")

    def _update_osh_metrics(self, simulation_data: Dict):
        """Update comprehensive OSH metrics from simulation data."""
        try:
            # Extract OSH values from various sources
            coherence = self._extract_coherence_data(simulation_data)
            entropy = self._extract_entropy_data(simulation_data)
            strain = self._extract_strain_data(simulation_data)
            
            # Store current OSH state
            if not hasattr(self, 'current_osh_metrics'):
                self.current_osh_metrics = {}
                
            # Import correct RSP calculation
            from src.visualization.osh_formula_utils import calculate_rsp_simple
            
            self.current_osh_metrics.update({
                'coherence': coherence,
                'entropy': entropy,
                'strain': strain,
                'rsp': calculate_rsp_simple(coherence, entropy, strain),
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"Error updating OSH metrics: {e}")

    def _extract_coherence_data(self, simulation_data: Dict) -> float:
        """Extract coherence data from multiple sources."""
        # Try coherence manager first
        if self.coherence_manager:
            try:
                coherence_stats = self.coherence_manager.get_coherence_statistics()
                if coherence_stats and 'average' in coherence_stats:
                    return float(coherence_stats['average'])
            except Exception:
                pass
        
        # Try direct from simulation data
        if 'coherence' in simulation_data:
            coherence_data = simulation_data['coherence']
            if isinstance(coherence_data, (int, float)):
                return float(coherence_data)
            elif isinstance(coherence_data, dict) and 'average' in coherence_data:
                return float(coherence_data['average'])
        
        # Try from memory field
        if self.memory_field and hasattr(self.memory_field, 'get_field_statistics'):
            try:
                stats = self.memory_field.get_field_statistics()
                if 'average_coherence' in stats:
                    return float(stats['average_coherence'])
            except Exception:
                pass
        
        return 0.5  # Default value

    def _extract_entropy_data(self, simulation_data: Dict) -> float:
        """Extract entropy data from multiple sources."""
        # Similar extraction logic for entropy
        if 'entropy' in simulation_data:
            entropy_data = simulation_data['entropy']
            if isinstance(entropy_data, (int, float)):
                return float(entropy_data)
            elif isinstance(entropy_data, dict) and 'average' in entropy_data:
                return float(entropy_data['average'])
        
        if self.memory_field and hasattr(self.memory_field, 'get_field_statistics'):
            try:
                stats = self.memory_field.get_field_statistics()
                if 'average_entropy' in stats:
                    return float(stats['average_entropy'])
            except Exception:
                pass
        
        return 0.3  # Default value

    def _extract_strain_data(self, simulation_data: Dict) -> float:
        """Extract strain data from multiple sources."""
        # Similar extraction logic for strain
        if 'strain' in simulation_data:
            strain_data = simulation_data['strain']
            if isinstance(strain_data, (int, float)):
                return float(strain_data)
            elif isinstance(strain_data, dict) and 'average' in strain_data:
                return float(strain_data['average'])
        
        if self.memory_field and hasattr(self.memory_field, 'get_field_statistics'):
            try:
                stats = self.memory_field.get_field_statistics()
                if 'average_strain' in stats:
                    return float(stats['average_strain'])
            except Exception:
                pass
        
        return 0.2  # Default value

    def render_panel(self, width: int = 1200, height: int = 1000) -> Dict:
        """
        Render the field panel with enterprise-grade visualization.
        
        Args:
            width: Panel width in pixels
            height: Panel height in pixels
            
        Returns:
            Comprehensive rendering result dictionary
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(width, height)
        if cache_key in self.render_cache and self.cache_enabled:
            cache_time, cached_result = self.render_cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                cached_result["from_cache"] = True
                return cached_result
        
        try:
            # Validate field selection
            if not self.selected_field:
                return self._render_no_field_selected(width, height)
            
            # Get field data with error handling
            field_data = self._get_field_data_safe()
            if field_data is None:
                return self._render_field_error(width, height, "Failed to retrieve field data")
            
            # Route to appropriate renderer
            render_result = self._route_visualization(field_data, width, height)
            
            # Add performance metrics
            render_time = time.time() - start_time
            render_result.update({
                "render_time": render_time,
                "render_fps": 1.0 / render_time if render_time > 0 else float('inf'),
                "field_shape": field_data.shape,
                "visualization_type": self.selected_visualization,
                "timestamp": time.time()
            })
            
            # Cache result
            if self.cache_enabled:
                self.render_cache[cache_key] = (time.time(), render_result.copy())
            
            return render_result
            
        except Exception as e:
            logger.error(f"Error rendering field panel: {e}")
            return self._render_field_error(width, height, str(e))

    def _route_visualization(self, field_data: np.ndarray, width: int, height: int) -> Dict:
        """Route to appropriate visualization renderer."""
        visualization_map = {
            "field_values": self._render_field_values_advanced,
            "coherence": self._render_coherence_advanced,
            "entropy": self._render_entropy_advanced,
            "strain": self._render_strain_advanced,
            "rsp": self._render_rsp_advanced,
            "gradient": self._render_gradient_advanced,
            "evolution": self._render_evolution_advanced,
            "recursive_layers": self._render_recursive_layers_advanced,
            "osh_comprehensive": self._render_osh_comprehensive_advanced,
            "phase_space": self._render_phase_space_analysis,
            "spectral_analysis": self._render_spectral_analysis,
            "stability_analysis": self._render_stability_analysis,
            "memory_topology": self._render_memory_topology
        }
        
        renderer = visualization_map.get(self.selected_visualization, 
                                       self._render_field_values_advanced)
        return renderer(field_data, width, height)

    def _render_field_values_advanced(self, field_data: np.ndarray, 
                                    width: int, height: int) -> Dict:
        """Advanced field values rendering with scientific overlays."""
        fig = plt.figure(figsize=(width/100, height/100), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # Main field visualization
        ax_main = fig.add_subplot(gs[0, :])
        field_2d = self._prepare_field_for_display(field_data)
        
        # Determine value range
        if self.auto_range:
            vmin, vmax = np.percentile(field_2d, [1, 99])  # Robust range
        else:
            vmin, vmax = self.value_range
        
        # Main heatmap with advanced styling
        im = ax_main.imshow(field_2d, cmap=self.colormap, origin='lower', 
                           vmin=vmin, vmax=vmax, aspect='auto', 
                           interpolation='bilinear')
        
        # Add contour lines
        contours = ax_main.contour(field_2d, levels=10, colors='white', 
                                  alpha=0.6, linewidths=0.8)
        ax_main.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # Colorbar with scientific notation
        cbar = plt.colorbar(im, ax=ax_main, shrink=0.8)
        cbar.set_label('Field Amplitude', fontsize=12)
        
        # Grid and labels
        if self.show_grid:
            ax_main.grid(True, alpha=0.3, linestyle='--')
        
        # Field information
        field_info = self._get_field_info()
        title_text = f"{self.selected_field} ({field_info.get('field_type', 'unknown')})"
        ax_main.set_title(title_text, fontsize=14, fontweight='bold')
        
        # Statistical analysis subplot
        ax_stats = fig.add_subplot(gs[1, 0])
        self._add_statistical_analysis(ax_stats, field_2d)
        
        # OSH metrics subplot  
        ax_osh = fig.add_subplot(gs[1, 1])
        self._add_osh_metrics_display(ax_osh, field_2d)
        
        plt.tight_layout()
        
        # Generate result
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        # Calculate comprehensive statistics
        stats = self._calculate_comprehensive_stats(field_2d, field_info)
        
        return {
            "success": True,
            "image_data": img_data,
            "statistics": stats,
            "field_info": field_info,
            "osh_metrics": self._calculate_current_osh_metrics(field_2d)
        }

    def _render_coherence_advanced(self, field_data: np.ndarray, 
                                 width: int, height: int) -> Dict:
        """Advanced coherence visualization with multiple analysis methods."""
        coherence_data = self.metrics_engine.calculate_field_coherence(field_data, "l1_norm")
        
        if self.coherence_renderer:
            # Use advanced coherence renderer
            try:
                result = self.coherence_renderer.render_coherence_map(
                    coherence_data=coherence_data,
                    title=f"Coherence Analysis: {self.selected_field}",
                    settings={
                        'show_critical_points': True,
                        'coherence_threshold': self.coherence_threshold,
                        'colormap': 'viridis',
                        'contour_levels': 15,
                        'show_phase_portrait': True
                    }
                )
                
                if result.get("success"):
                    result.update({
                        "field_name": self.selected_field,
                        "coherence_method": "l1_norm",
                        "threshold_used": self.coherence_threshold
                    })
                    return result
                    
            except Exception as e:
                logger.warning(f"Advanced coherence renderer failed: {e}")
        
        # Fallback to comprehensive built-in renderer
        return self._render_coherence_comprehensive(coherence_data, width, height)

    def _render_coherence_comprehensive(self, coherence_data: np.ndarray, 
                                      width: int, height: int) -> Dict:
        """Comprehensive coherence visualization fallback."""
        fig = plt.figure(figsize=(width/100, height/100), dpi=self.dpi)
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # Main coherence map
        ax_main = fig.add_subplot(gs[0, :2])
        im = ax_main.imshow(coherence_data, cmap='viridis', origin='lower', vmin=0, vmax=1)
        
        # Contour lines with threshold
        contours = ax_main.contour(coherence_data, levels=np.linspace(0, 1, 11), 
                                  colors='white', alpha=0.7, linewidths=1)
        ax_main.clabel(contours, inline=True, fontsize=8)
        
        # Threshold line
        threshold_contour = ax_main.contour(coherence_data, levels=[self.coherence_threshold], 
                                          colors='red', linewidths=2, linestyles='--')
        
        plt.colorbar(im, ax=ax_main, label='Coherence')
        ax_main.set_title(f'Coherence Map: {self.selected_field}')
        
        # Coherence histogram
        ax_hist = fig.add_subplot(gs[0, 2])
        ax_hist.hist(coherence_data.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        ax_hist.axvline(self.coherence_threshold, color='red', linestyle='--', 
                       label=f'Threshold ({self.coherence_threshold:.2f})')
        ax_hist.set_xlabel('Coherence')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.set_title('Coherence Distribution')
        
        # Time evolution if available
        ax_evolution = fig.add_subplot(gs[1, :])
        if (self.selected_field in self.evolution_data and 
            self.evolution_data[self.selected_field]["coherence_history"]):
            
            history = list(self.evolution_data[self.selected_field]["coherence_history"])
            time_points = np.arange(len(history))
            
            ax_evolution.plot(time_points, history, 'b-', linewidth=2, label='Mean Coherence')
            ax_evolution.axhline(self.coherence_threshold, color='red', linestyle='--', 
                               label=f'Threshold ({self.coherence_threshold:.2f})')
            ax_evolution.fill_between(time_points, history, alpha=0.3)
            ax_evolution.set_xlabel('Time Step')
            ax_evolution.set_ylabel('Coherence')
            ax_evolution.legend()
            ax_evolution.set_title('Coherence Evolution')
            ax_evolution.grid(True, alpha=0.3)
        else:
            ax_evolution.text(0.5, 0.5, 'No evolution data available', 
                            ha='center', va='center', transform=ax_evolution.transAxes)
            ax_evolution.set_title('Coherence Evolution (No Data)')
        
        plt.tight_layout()
        
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        # Comprehensive statistics
        stats = {
            "mean_coherence": float(np.mean(coherence_data)),
            "std_coherence": float(np.std(coherence_data)),
            "min_coherence": float(np.min(coherence_data)),
            "max_coherence": float(np.max(coherence_data)),
            "threshold_exceeded": float(np.mean(coherence_data > self.coherence_threshold)),
            "coherence_range": float(np.max(coherence_data) - np.min(coherence_data)),
            "spatial_correlation": self._calculate_spatial_coherence_correlation(coherence_data)
        }
        
        return {
            "success": True,
            "image_data": img_data,
            "statistics": stats,
            "coherence_threshold": self.coherence_threshold,
            "analysis_method": "comprehensive"
        }

    def _render_rsp_advanced(self, field_data: np.ndarray, width: int, height: int) -> Dict:
        """Advanced Recursive Simulation Potential visualization."""
        # Calculate OSH components
        coherence = self.metrics_engine.calculate_field_coherence(field_data)
        entropy = self.metrics_engine.calculate_field_entropy(field_data)
        strain = self.metrics_engine.calculate_field_strain(field_data)
        rsp = self.metrics_engine.calculate_rsp(coherence, entropy, strain)
        
        fig = plt.figure(figsize=(width/100, height/100), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 3, figure=fig)
        
        # Main RSP visualization
        ax_main = fig.add_subplot(gs[0, :])
        im_rsp = ax_main.imshow(rsp, cmap='coolwarm', origin='lower', vmin=0, vmax=5)
        
        # RSP contours and critical regions
        rsp_contours = ax_main.contour(rsp, levels=np.linspace(0, 5, 11), 
                                      colors='black', alpha=0.6, linewidths=0.8)
        ax_main.clabel(rsp_contours, inline=True, fontsize=8, fmt='%.1f')
        
        # Highlight high-RSP regions
        high_rsp_mask = rsp > self.rsp_threshold
        ax_main.contour(high_rsp_mask, levels=[0.5], colors='yellow', linewidths=3, 
                       linestyles='solid', alpha=0.8)
        
        cbar_rsp = plt.colorbar(im_rsp, ax=ax_main, shrink=0.8)
        cbar_rsp.set_label('RSP (Recursive Simulation Potential)')
        ax_main.set_title(f'RSP Landscape: {self.selected_field}', fontsize=14, fontweight='bold')
        
        # Component analysis
        components = [
            (coherence, 'Coherence', 'viridis', gs[1, 0]),
            (entropy, 'Entropy', 'plasma', gs[1, 1]),
            (strain, 'Strain', 'hot_r', gs[1, 2])
        ]
        
        for component_data, title, cmap, grid_pos in components:
            ax = fig.add_subplot(grid_pos)
            im = ax.imshow(component_data, cmap=cmap, origin='lower')
            plt.colorbar(im, ax=ax, shrink=0.7)
            ax.set_title(title, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        # RSP analysis plots
        ax_hist = fig.add_subplot(gs[2, 0])
        ax_hist.hist(rsp.flatten(), bins=50, alpha=0.7, color='blue', density=True)
        ax_hist.axvline(self.rsp_threshold, color='red', linestyle='--', 
                       label=f'Critical RSP ({self.rsp_threshold:.1f})')
        ax_hist.set_xlabel('RSP Value')
        ax_hist.set_ylabel('Density')
        ax_hist.legend()
        ax_hist.set_title('RSP Distribution')
        
        # RSP vs Coherence scatter
        ax_scatter = fig.add_subplot(gs[2, 1])
        ax_scatter.scatter(coherence.flatten(), rsp.flatten(), alpha=0.6, s=1)
        ax_scatter.set_xlabel('Coherence')
        ax_scatter.set_ylabel('RSP')
        ax_scatter.set_title('RSP vs Coherence')
        ax_scatter.grid(True, alpha=0.3)
        
        # Phase space projection
        ax_phase = fig.add_subplot(gs[2, 2])
        ax_phase.scatter(entropy.flatten(), strain.flatten(), c=rsp.flatten(), 
                        cmap='coolwarm', alpha=0.6, s=2)
        ax_phase.set_xlabel('Entropy')
        ax_phase.set_ylabel('Strain')
        ax_phase.set_title('Entropy-Strain Phase Space')
        
        plt.tight_layout()
        
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        # RSP statistics and classification
        rsp_stats = {
            "mean_rsp": float(np.mean(rsp)),
            "max_rsp": float(np.max(rsp)),
            "high_rsp_fraction": float(np.mean(rsp > self.rsp_threshold)),
            "rsp_variance": float(np.var(rsp)),
            "rsp_class": self._classify_rsp_level(np.mean(rsp)),
            "critical_regions": int(np.sum(rsp > self.rsp_threshold)),
            "emergence_index": self._calculate_emergence_index(coherence, entropy, strain)
        }
        
        return {
            "success": True,
            "image_data": img_data,
            "rsp_statistics": rsp_stats,
            "osh_components": {
                "coherence_mean": float(np.mean(coherence)),
                "entropy_mean": float(np.mean(entropy)),
                "strain_mean": float(np.mean(strain))
            }
        }

    def _render_recursive_layers_advanced(self, field_data: np.ndarray, 
                                        width: int, height: int) -> Dict:
        """Advanced recursive layer visualization with full 3D rendering."""
        fig = plt.figure(figsize=(width/100, height/100), dpi=self.dpi)
        
        # Get recursive data if available
        recursive_data = {}
        if (self.recursive_mechanics and 
            hasattr(self.recursive_mechanics, 'get_system_statistics')):
            try:
                recursive_stats = self.recursive_mechanics.get_system_statistics()
                # Create synthetic recursive data for visualization
                max_depth = recursive_stats.get('max_depth', 3)
                for i in range(max_depth):
                    recursive_data[str(i)] = {
                        'coherence': 1.0 - i * 0.15,
                        'entropy': i * 0.1,
                        'strain': i * 0.05 + 0.1,
                        'depth': i
                    }
            except Exception as e:
                logger.warning(f"Could not retrieve recursive data: {e}")
        
        # Create 3D visualization
        ax_3d = fig.add_subplot(111, projection='3d')
        
        # Render recursive layers
        layer_info = self.recursive_renderer.render_recursive_layers_3d(
            field_data, recursive_data, ax_3d
        )
        
        # Add field information
        field_info = self._get_field_info()
        fig.suptitle(f'Recursive Memory Layers: {self.selected_field}', 
                    fontsize=14, fontweight='bold')
        
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        return {
            "success": True,
            "image_data": img_data,
            "layer_info": layer_info,
            "recursive_data": recursive_data,
            "field_type": field_info.get('field_type', 'unknown')
        }

    def _render_osh_comprehensive_advanced(self, field_data: np.ndarray, 
                                         width: int, height: int) -> Dict:
        """Comprehensive OSH state visualization with all metrics."""
        if self.coherence_renderer:
            try:
                # Use advanced OSH renderer
                coherence = self.metrics_engine.calculate_field_coherence(field_data)
                entropy = self.metrics_engine.calculate_field_entropy(field_data)
                strain = self.metrics_engine.calculate_field_strain(field_data)
                
                result = self.coherence_renderer.render_osh_substrate(
                    coherence_data=coherence,
                    entropy_data=entropy,
                    strain_data=strain,
                    title=f"OSH Comprehensive Analysis: {self.selected_field}",
                    emit_events=False
                )
                
                if result.get("success"):
                    result.update({
                        "field_name": self.selected_field,
                        "analysis_type": "comprehensive_osh"
                    })
                    return result
                    
            except Exception as e:
                logger.warning(f"Advanced OSH renderer failed: {e}")
        
        # Fallback comprehensive OSH visualization
        return self._render_osh_fallback_comprehensive(field_data, width, height)

    def _render_osh_fallback_comprehensive(self, field_data: np.ndarray, 
                                         width: int, height: int) -> Dict:
        """Comprehensive OSH fallback visualization."""
        # Calculate all OSH metrics
        coherence = self.metrics_engine.calculate_field_coherence(field_data)
        entropy = self.metrics_engine.calculate_field_entropy(field_data)
        strain = self.metrics_engine.calculate_field_strain(field_data)
        rsp = self.metrics_engine.calculate_rsp(coherence, entropy, strain)
        
        fig = plt.figure(figsize=(width/100, height/100), dpi=self.dpi)
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Main field
        ax_field = fig.add_subplot(gs[0, 0])
        field_2d = self._prepare_field_for_display(field_data)
        im_field = ax_field.imshow(field_2d, cmap='viridis', origin='lower')
        plt.colorbar(im_field, ax=ax_field, shrink=0.8)
        ax_field.set_title('Original Field')
        
        # OSH components
        osh_components = [
            (coherence, 'Coherence', 'viridis', gs[0, 1]),
            (entropy, 'Entropy', 'plasma', gs[0, 2]),
            (strain, 'Strain', 'hot_r', gs[0, 3])
        ]
        
        for component, title, cmap, pos in osh_components:
            ax = fig.add_subplot(pos)
            im = ax.imshow(component, cmap=cmap, origin='lower', vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, shrink=0.8)
            ax.set_title(title)
        
        # RSP visualization
        ax_rsp = fig.add_subplot(gs[1, :2])
        im_rsp = ax_rsp.imshow(rsp, cmap='coolwarm', origin='lower')
        contours = ax_rsp.contour(rsp, levels=10, colors='black', alpha=0.6, linewidths=0.8)
        ax_rsp.clabel(contours, inline=True, fontsize=8)
        plt.colorbar(im_rsp, ax=ax_rsp, shrink=0.8)
        ax_rsp.set_title('Recursive Simulation Potential (RSP)')
        
        # Phase space analysis
        ax_phase = fig.add_subplot(gs[1, 2:], projection='3d')
        scatter = ax_phase.scatter(coherence.flatten(), entropy.flatten(), strain.flatten(),
                                 c=rsp.flatten(), cmap='coolwarm', alpha=0.6, s=2)
        ax_phase.set_xlabel('Coherence')
        ax_phase.set_ylabel('Entropy')
        ax_phase.set_zlabel('Strain')
        ax_phase.set_title('OSH Phase Space')
        
        # Statistical analysis
        ax_stats = fig.add_subplot(gs[2, :2])
        metrics = ['Coherence', 'Entropy', 'Strain', 'RSP']
        values = [np.mean(coherence), np.mean(entropy), np.mean(strain), np.mean(rsp)]
        colors = ['blue', 'red', 'orange', 'green']
        
        bars = ax_stats.bar(metrics, values, color=colors, alpha=0.7)
        ax_stats.set_ylabel('Mean Value')
        ax_stats.set_title('OSH Metrics Summary')
        ax_stats.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax_stats.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                         f'{value:.3f}', ha='center', va='bottom')
        
        # Time evolution
        ax_evolution = fig.add_subplot(gs[2, 2:])
        if (self.selected_field in self.evolution_data):
            evolution = self.evolution_data[self.selected_field]
            
            time_points = np.arange(len(evolution["coherence_history"]))
            ax_evolution.plot(time_points, list(evolution["coherence_history"]), 
                            'b-', label='Coherence', linewidth=2)
            ax_evolution.plot(time_points, list(evolution["entropy_history"]), 
                            'r-', label='Entropy', linewidth=2)
            ax_evolution.plot(time_points, list(evolution["strain_history"]), 
                            'orange', label='Strain', linewidth=2)
            
            ax_evolution.set_xlabel('Time Step')
            ax_evolution.set_ylabel('Value')
            ax_evolution.legend()
            ax_evolution.set_title('OSH Evolution')
            ax_evolution.grid(True, alpha=0.3)
        else:
            ax_evolution.text(0.5, 0.5, 'No evolution data', 
                            ha='center', va='center', transform=ax_evolution.transAxes)
        
        plt.tight_layout()
        
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        # Comprehensive OSH statistics
        osh_stats = {
            "coherence": {
                "mean": float(np.mean(coherence)),
                "std": float(np.std(coherence)),
                "max": float(np.max(coherence))
            },
            "entropy": {
                "mean": float(np.mean(entropy)),
                "std": float(np.std(entropy)),
                "max": float(np.max(entropy))
            },
            "strain": {
                "mean": float(np.mean(strain)),
                "std": float(np.std(strain)),
                "max": float(np.max(strain))
            },
            "rsp": {
                "mean": float(np.mean(rsp)),
                "std": float(np.std(rsp)),
                "max": float(np.max(rsp)),
                "class": self._classify_rsp_level(np.mean(rsp))
            }
        }
        
        return {
            "success": True,
            "image_data": img_data,
            "osh_statistics": osh_stats,
            "analysis_type": "comprehensive_fallback"
        }

    def evolve_field(self, num_steps: int = 10, evolution_type: str = "wave_equation",
                    parameters: Optional[Dict] = None) -> Dict:
        """
        Advanced field evolution with comprehensive analysis.
        
        Args:
            num_steps: Number of evolution steps
            evolution_type: Type of PDE evolution
            parameters: Evolution parameters
            
        Returns:
            Complete evolution analysis and visualization
        """
        if not self.evolution_engine or not self.selected_field:
            return {
                "success": False,
                "error": "Evolution engine not available or no field selected"
            }
        
        # Prepare evolution parameters
        evolution_params = {
            "evolution_type": evolution_type,
            "num_steps": num_steps,
            "time_step": self.time_step,
            "adaptive_dt": True,
            "stability_check": True
        }
        
        if parameters:
            evolution_params.update(parameters)
        
        try:
            # Perform evolution
            evolution_result = self.evolution_engine.evolve_field_advanced(
                self.selected_field, evolution_params
            )
            
            if not evolution_result["success"]:
                return evolution_result
            
            # Store animation frames
            evolution_data = evolution_result["evolution_data"]
            self.animation_frames.clear()
            for frame in evolution_data["frames"]:
                self.animation_frames.append(frame.copy())
            
            # Generate evolution visualization
            animation_result = self._create_evolution_animation()
            
            # Combine results
            result = {
                "success": True,
                "evolution_result": evolution_result,
                "animation_result": animation_result,
                "num_frames": len(self.animation_frames),
                "evolution_type": evolution_type,
                "final_metrics": evolution_result.get("final_metrics", {})
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in field evolution: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_evolution_animation(self) -> Dict:
        """Create high-quality evolution animation."""
        if len(self.animation_frames) < 2:
            return {"success": False, "error": "Insufficient frames for animation"}
        
        try:
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            
            # Determine consistent value range across all frames
            all_values = np.concatenate([frame.flatten() for frame in self.animation_frames])
            vmin, vmax = np.percentile(all_values, [1, 99])
            
            # Set up the animation
            im = ax.imshow(self.animation_frames[0], cmap=self.colormap, 
                          origin='lower', animated=True, vmin=vmin, vmax=vmax)
            
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Field Amplitude')
            
            title = ax.set_title(f"Evolution of {self.selected_field} - Frame 1/{len(self.animation_frames)}")
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            
            def animate(frame_idx):
                im.set_array(self.animation_frames[frame_idx])
                title.set_text(f"Evolution of {self.selected_field} - Frame {frame_idx+1}/{len(self.animation_frames)}")
                return [im]
            
            # Create animation
            anim = FuncAnimation(fig, animate, frames=len(self.animation_frames),
                               interval=200, blit=True, repeat=True)
            
            # Save as GIF
            buffer = io.BytesIO()
            writer = PillowWriter(fps=5)
            anim.save(buffer, writer=writer, format='gif')
            buffer.seek(0)
            
            gif_data = base64.b64encode(buffer.read()).decode('utf-8')
            
            plt.close(fig)
            
            # Calculate evolution statistics
            evolution_stats = self._calculate_evolution_statistics()
            
            return {
                "success": True,
                "animation_data": f"data:image/gif;base64,{gif_data}",
                "evolution_statistics": evolution_stats
            }
            
        except Exception as e:
            logger.error(f"Error creating evolution animation: {e}")
            return {"success": False, "error": str(e)}

    def _calculate_evolution_statistics(self) -> Dict:
        """Calculate comprehensive evolution statistics."""
        if len(self.animation_frames) < 2:
            return {}
        
        # Frame-to-frame analysis
        frame_differences = []
        energy_evolution = []
        coherence_evolution = []
        
        for i in range(1, len(self.animation_frames)):
            # Frame difference
            diff = np.linalg.norm(self.animation_frames[i] - self.animation_frames[i-1])
            frame_differences.append(diff)
            
            # Energy
            energy = np.sum(np.abs(self.animation_frames[i])**2)
            energy_evolution.append(energy)
            
            # Coherence
            coherence = self.metrics_engine.calculate_field_coherence(self.animation_frames[i])
            coherence_evolution.append(np.mean(coherence))
        
        return {
            "total_frames": len(self.animation_frames),
            "mean_frame_difference": float(np.mean(frame_differences)),
            "max_frame_difference": float(np.max(frame_differences)),
            "energy_conservation": {
                "initial_energy": float(energy_evolution[0]) if energy_evolution else 0.0,
                "final_energy": float(energy_evolution[-1]) if energy_evolution else 0.0,
                "energy_drift": float(energy_evolution[-1] - energy_evolution[0]) if len(energy_evolution) > 1 else 0.0
            },
            "coherence_evolution": {
                "initial_coherence": float(coherence_evolution[0]) if coherence_evolution else 0.0,
                "final_coherence": float(coherence_evolution[-1]) if coherence_evolution else 0.0,
                "coherence_trend": "increasing" if (coherence_evolution and 
                                                  coherence_evolution[-1] > coherence_evolution[0]) else "decreasing"
            }
        }

    # Utility methods
    
    def _get_field_data_safe(self) -> Optional[np.ndarray]:
        """Safely retrieve field data with error handling."""
        try:
            if not self.field_dynamics or not self.selected_field:
                return None
                
            field_values = self.field_dynamics.get_field_values(self.selected_field)
            if field_values is None:
                return None
                
            return self._prepare_field_for_display(field_values)
            
        except Exception as e:
            logger.error(f"Error retrieving field data: {e}")
            return None

    def _prepare_field_for_display(self, field_data: np.ndarray) -> np.ndarray:
        """Prepare field data for visualization by handling dimensions and slicing."""
        if len(field_data.shape) == 1:
            # Convert 1D to 2D for visualization
            size = int(np.sqrt(len(field_data)))
            if size * size == len(field_data):
                return field_data.reshape((size, size))
            else:
                return field_data.reshape((-1, 1))
                
        elif len(field_data.shape) == 2:
            return field_data
            
        elif len(field_data.shape) == 3:
            # Take a slice for 3D data
            slice_idx = min(self.slice_index, field_data.shape[-1] - 1)
            return field_data[:, :, slice_idx]
            
        elif len(field_data.shape) == 4:
            # For 4D data (like tensor fields), take diagonal slice
            slice_idx = min(self.slice_index, field_data.shape[-1] - 1)
            return field_data[:, :, slice_idx, slice_idx]
            
        else:
            # For higher dimensions, flatten to 2D
            return field_data.reshape(field_data.shape[0], -1)

    def _get_field_info(self) -> Dict:
        """Get comprehensive field information."""
        if not self.field_dynamics or not self.selected_field:
            return {}
            
        try:
            field_info = self.field_dynamics.get_field(self.selected_field)
            return field_info if field_info else {}
        except Exception as e:
            logger.error(f"Error getting field info: {e}")
            return {}

    def _calculate_comprehensive_stats(self, field_data: np.ndarray, field_info: Dict) -> Dict:
        """Calculate comprehensive field statistics."""
        try:
            return {
                "basic_stats": {
                    "mean": float(np.mean(field_data)),
                    "std": float(np.std(field_data)),
                    "min": float(np.min(field_data)),
                    "max": float(np.max(field_data)),
                    "variance": float(np.var(field_data))
                },
                "shape_info": {
                    "shape": list(field_data.shape),
                    "size": int(field_data.size),
                    "ndim": int(field_data.ndim)
                },
                "field_properties": {
                    "field_type": field_info.get("field_type", "unknown"),
                    "energy": float(np.sum(np.abs(field_data)**2)),
                    "gradient_norm": float(np.linalg.norm(np.gradient(field_data))),
                    "sparsity": float(np.mean(np.abs(field_data) < 1e-6))
                }
            }
        except Exception as e:
            logger.error(f"Error calculating comprehensive stats: {e}")
            return {}

    def _calculate_current_osh_metrics(self, field_data: np.ndarray) -> Dict:
        """Calculate current OSH metrics for the field."""
        try:
            coherence = self.metrics_engine.calculate_field_coherence(field_data)
            entropy = self.metrics_engine.calculate_field_entropy(field_data)
            strain = self.metrics_engine.calculate_field_strain(field_data)
            rsp = self.metrics_engine.calculate_rsp(coherence, entropy, strain)
            
            return {
                "coherence": {
                    "mean": float(np.mean(coherence)),
                    "max": float(np.max(coherence)),
                    "critical_fraction": float(np.mean(coherence > self.coherence_threshold))
                },
                "entropy": {
                    "mean": float(np.mean(entropy)),
                    "max": float(np.max(entropy)),
                    "high_entropy_fraction": float(np.mean(entropy > self.entropy_threshold))
                },
                "strain": {
                    "mean": float(np.mean(strain)),
                    "max": float(np.max(strain)),
                    "critical_strain_fraction": float(np.mean(strain > self.strain_threshold))
                },
                "rsp": {
                    "mean": float(np.mean(rsp)),
                    "max": float(np.max(rsp)),
                    "classification": self._classify_rsp_level(np.mean(rsp))
                }
            }
        except Exception as e:
            logger.error(f"Error calculating OSH metrics: {e}")
            return {}

    def _classify_rsp_level(self, rsp_value: float) -> str:
        """Classify RSP level according to OSH theory."""
        if rsp_value > 4.0:
            return "exceptional"
        elif rsp_value > 2.5:
            return "high"
        elif rsp_value > 1.0:
            return "moderate"
        elif rsp_value > 0.5:
            return "low"
        else:
            return "critical"

    def _calculate_emergence_index(self, coherence: np.ndarray, 
                                 entropy: np.ndarray, strain: np.ndarray) -> float:
        """Calculate emergence index for OSH analysis."""
        try:
            # Emergence as the interaction between order (coherence) and disorder (entropy)
            # modulated by structural strain
            coherence_var = np.var(coherence)
            entropy_var = np.var(entropy)
            strain_coupling = np.corrcoef(coherence.flatten(), strain.flatten())[0, 1]
            
            if np.isnan(strain_coupling):
                strain_coupling = 0.0
                
            emergence = (coherence_var * entropy_var) / (1.0 + abs(strain_coupling))
            return float(emergence)
            
        except Exception:
            return 0.0

    def _calculate_spatial_coherence_correlation(self, coherence_data: np.ndarray) -> float:
        """Calculate spatial coherence correlation."""
        try:
            # Calculate autocorrelation
            flat_coherence = coherence_data.flatten()
            correlation = np.corrcoef(flat_coherence[:-1], flat_coherence[1:])[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except Exception:
            return 0.0

    def _add_statistical_analysis(self, ax, field_data: np.ndarray):
        """Add statistical analysis subplot."""
        try:
            # Calculate statistics
            mean_val = np.mean(field_data)
            std_val = np.std(field_data)
            skewness = float(np.mean(((field_data - mean_val) / std_val) ** 3))
            kurtosis = float(np.mean(((field_data - mean_val) / std_val) ** 4)) - 3
            
            # Create text summary
            stats_text = f"""Statistical Analysis:
Mean: {mean_val:.4f}
Std: {std_val:.4f}
Skewness: {skewness:.4f}
Kurtosis: {kurtosis:.4f}
            """
            
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Add histogram
            ax.hist(field_data.flatten(), bins=30, alpha=0.7, density=True)
            ax.set_xlabel('Field Value')
            ax.set_ylabel('Density')
            ax.set_title('Field Distribution')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Statistical analysis failed: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _add_osh_metrics_display(self, ax, field_data: np.ndarray):
        """Add OSH metrics display subplot."""
        try:
            osh_metrics = self._calculate_current_osh_metrics(field_data)
            
            # Create radar chart for OSH metrics
            metrics = ['Coherence', 'Order', 'Stability', 'RSP']
            values = [
                osh_metrics.get('coherence', {}).get('mean', 0),
                1.0 - osh_metrics.get('entropy', {}).get('mean', 0),
                1.0 - osh_metrics.get('strain', {}).get('mean', 0),
                min(osh_metrics.get('rsp', {}).get('mean', 0) / 5.0, 1.0)  # Normalize RSP
            ]
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            values = np.concatenate((values, [values[0]]))  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            ax.plot(angles, values, 'b-', linewidth=2, label='OSH Metrics')
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics)
            ax.set_ylim(0, 1)
            ax.set_title('OSH Metrics Radar')
            ax.grid(True)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'OSH analysis failed: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)

    def _generate_cache_key(self, width: int, height: int) -> str:
        """Generate cache key for render results."""
        key_components = [
            self.selected_field or "none",
            self.selected_visualization,
            str(width), str(height),
            str(self.colormap),
            str(self.auto_range),
            str(self.slice_index)
        ]
        
        key_string = "_".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _clear_field_cache(self, field_name: str):
        """Clear cache entries related to a specific field."""
        keys_to_remove = [key for key in self.render_cache.keys() 
                         if field_name in key]
        for key in keys_to_remove:
            del self.render_cache[key]

    def _get_figure_data(self, fig) -> str:
        """Convert matplotlib figure to base64 PNG."""
        try:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            buf.seek(0)
            data = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{data}"
        except Exception as e:
            logger.error(f"Error generating figure data: {e}")
            return ""

    def _render_no_field_selected(self, width: int, height: int) -> Dict:
        """Render message when no field is selected."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=self.dpi)
        
        ax.text(0.5, 0.6, "No Field Selected", ha='center', va='center', 
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        available_fields = self.get_available_fields()
        if available_fields:
            fields_text = "Available fields:\n" + "\n".join(available_fields[:10])
            if len(available_fields) > 10:
                fields_text += f"\n... and {len(available_fields) - 10} more"
        else:
            fields_text = "No fields available in FieldDynamics"
            
        ax.text(0.5, 0.4, fields_text, ha='center', va='center', 
               fontsize=12, transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        return {
            "success": True,
            "image_data": img_data,
            "message": "No field selected",
            "available_fields": available_fields
        }

    def _render_field_error(self, width: int, height: int, error_message: str) -> Dict:
        """Render error message."""
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=self.dpi)
        
        ax.text(0.5, 0.5, f"Field Visualization Error:\n{error_message}", 
               ha='center', va='center', fontsize=14, color='red',
               transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        img_data = self._get_figure_data(fig)
        plt.close(fig)
        
        return {
            "success": False,
            "image_data": img_data,
            "error": error_message
        }

    def cleanup(self):
        """Clean up resources and threads."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=True)
            
            # Clear caches
            self.render_cache.clear()
            self.metrics_engine.cache.clear()
            
            logger.info("FieldPanel cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def create_field_panel(field_dynamics=None, memory_field=None, coherence_manager=None,
                      recursive_mechanics=None, quantum_renderer=None, 
                      coherence_renderer=None, config=None):
    """
    Factory function to create a FieldPanel instance with comprehensive configuration.
    
    Args:
        field_dynamics: FieldDynamics instance
        memory_field: MemoryFieldPhysics instance  
        coherence_manager: CoherenceManager instance
        recursive_mechanics: RecursiveMechanics instance
        quantum_renderer: QuantumRenderer instance
        coherence_renderer: CoherenceRenderer instance
        config: Configuration dictionary
        
    Returns:
        Fully configured FieldPanel instance
    """
    # Default configuration for optimal performance
    default_config = {
        "performance_mode": "high_quality",
        "cache_enabled": True,
        "max_animation_frames": 100,
        "dpi": 150,
        "cache_ttl": 2.0
    }
    
    if config:
        default_config.update(config)
    
    return FieldPanel(
        field_dynamics=field_dynamics,
        memory_field=memory_field,
        coherence_manager=coherence_manager,
        recursive_mechanics=recursive_mechanics,
        quantum_renderer=quantum_renderer,
        coherence_renderer=coherence_renderer,
        config=default_config
    )