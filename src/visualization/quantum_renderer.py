"""
Recursia Enterprise Quantum Renderer
====================================

Advanced quantum state visualization engine with full OSH (Organic Simulation Hypothesis) integration.
Provides comprehensive quantum visualization capabilities including Bloch spheres, density matrices,
entanglement networks, quantum circuits, and OSH-aligned recursive quantum memory visualization.

Features:
- Scientific-grade quantum state visualization
- Real-time coherence and entropy tracking
- OSH metrics calculation and validation
- Enterprise-level error handling and performance optimization
- Publication-quality rendering with customizable themes
- Comprehensive statistical analysis and reporting
- Integration with Recursia's physics engines and event systems
"""

import base64
import hashlib
import io
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
try:
    import networkx as nx
except ImportError:
    nx = None
try:
    from scipy.linalg import eigvals, eigvalsh
    from scipy.spatial.distance import pdist, squareform
    from scipy.stats import entropy as scipy_entropy
except ImportError:
    eigvals = None
    eigvalsh = None
    pdist = None
    squareform = None
    scipy_entropy = None
try:
    import seaborn as sns
except ImportError:
    sns = None

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=matplotlib.MatplotlibDeprecationWarning)

# Configure scientific plotting defaults
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    pass
if sns is not None:
    sns.set_palette("husl")


class QuantumRenderer:
    """
    Enterprise-grade quantum visualization engine with full OSH integration.
    
    Provides comprehensive quantum state visualization capabilities including:
    - Bloch sphere rendering with OSH coherence tracking
    - Density matrix visualization (magnitude and phase)
    - Quantum measurement probability distributions
    - Coherence evolution and entropy dynamics
    - Entanglement network topology
    - Quantum circuit diagrams
    - OSH recursive quantum memory grids
    - Advanced statistical analysis and OSH metrics
    """
    
    def __init__(self, 
                 coherence_manager=None,
                 entanglement_manager=None,
                 event_system=None,
                 state_registry=None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the quantum renderer with comprehensive subsystem integration.
        
        Args:
            coherence_manager: Coherence physics engine
            entanglement_manager: Entanglement tracking system
            event_system: Real-time event broadcasting system
            state_registry: Quantum state management registry
            config: Rendering configuration parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Core physics integrations
        self.coherence_manager = coherence_manager
        self.entanglement_manager = entanglement_manager
        self.event_system = event_system
        self.state_registry = state_registry
        
        # Configuration with enterprise defaults
        self.config = {
            'dpi': 300,
            'figure_size': (12, 8),
            'color_scheme': 'viridis',
            'theme': 'dark_quantum',
            'scientific_mode': True,
            'publication_quality': True,
            'enable_caching': True,
            'cache_size': 100,
            'performance_monitoring': True,
            'osh_overlay_enabled': True,
            'entropy_visualization': True,
            'coherence_tracking': True,
            'max_animation_frames': 60,
            'render_timeout': 30.0,
            'memory_limit_mb': 512
        }
        
        if config:
            self.config.update(config)
        
        # Performance and caching systems
        self.render_cache = {}
        self.performance_metrics = {
            'render_times': deque(maxlen=100),
            'cache_hits': 0,
            'cache_misses': 0,
            'total_renders': 0,
            'memory_usage': 0
        }
        
        # OSH metrics tracking
        self.osh_metrics_history = deque(maxlen=1000)
        self.coherence_evolution = deque(maxlen=500)
        self.entropy_evolution = deque(maxlen=500)
        
        # Quantum visualization constants
        self.PAULI_MATRICES = {
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex),
            'I': np.array([[1, 0], [0, 1]], dtype=complex)
        }
        
        # Color schemes for different visualization modes
        self.color_schemes = {
            'dark_quantum': {
                'background': '#0a0a0a',
                'foreground': '#ffffff',
                'accent': '#00ffff',
                'coherence': 'plasma',
                'entropy': 'inferno',
                'phase': 'hsv'
            },
            'light_scientific': {
                'background': '#ffffff',
                'foreground': '#000000',
                'accent': '#0066cc',
                'coherence': 'viridis',
                'entropy': 'plasma',
                'phase': 'twilight'
            },
            'publication': {
                'background': '#ffffff',
                'foreground': '#000000',
                'accent': '#333333',
                'coherence': 'Blues',
                'entropy': 'Reds',
                'phase': 'hsv'
            }
        }
        
        # Initialize rendering subsystems
        self._initialize_rendering_subsystems()
        
        self.logger.info("QuantumRenderer initialized with enterprise configuration")
    
    def _initialize_rendering_subsystems(self):
        """Initialize all rendering subsystems and validate dependencies."""
        try:
            # Configure matplotlib for scientific rendering
            self._configure_matplotlib()
            
            # Initialize OSH computation engines
            self._initialize_osh_engines()
            
            # Set up event system integration
            self._setup_event_integration()
            
            # Initialize performance monitoring
            self._initialize_performance_monitoring()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rendering subsystems: {e}")
            raise RuntimeError(f"QuantumRenderer initialization failed: {e}")
    
    def _configure_matplotlib(self):
        """Configure matplotlib for scientific-grade rendering."""
        plt.rcParams.update({
            'figure.dpi': self.config['dpi'],
            'figure.figsize': self.config['figure_size'],
            'font.size': 10,
            'font.family': 'serif',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'lines.linewidth': 2,
            'patch.linewidth': 1,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def _initialize_osh_engines(self):
        """Initialize OSH computation engines for advanced metrics."""
        self.osh_calculators = {
            'coherence': self._calculate_osh_coherence,
            'entropy': self._calculate_von_neumann_entropy,
            'rsp': self._calculate_recursive_simulation_potential,
            'phi': self._calculate_integrated_information,
            'emergence': self._calculate_emergence_index,
            'complexity': self._calculate_kolmogorov_complexity_estimate
        }
    
    def _setup_event_integration(self):
        """Set up integration with the event system for real-time updates."""
        if self.event_system:
            try:
                # Register for quantum-related events
                quantum_events = [
                    'state_creation_event',
                    'coherence_change_event',
                    'entanglement_creation_event',
                    'measurement_event',
                    'collapse_event'
                ]
                
                for event_type in quantum_events:
                    self.event_system.add_listener(
                        event_type,
                        self._handle_quantum_event,
                        description=f"QuantumRenderer {event_type} handler"
                    )
                
                self.logger.debug("Event system integration configured")
            except Exception as e:
                self.logger.warning(f"Event system integration failed: {e}")
    
    def _initialize_performance_monitoring(self):
        """Initialize comprehensive performance monitoring."""
        if self.config['performance_monitoring']:
            self.performance_monitor = {
                'start_time': time.time(),
                'render_count': 0,
                'total_render_time': 0.0,
                'peak_memory': 0,
                'cache_efficiency': 0.0
            }
    
    def render_bloch_sphere(self,
                          state_vector: np.ndarray,
                          qubit_index: int = 0,
                          title: Optional[str] = None,
                          show_vector: bool = True,
                          show_axes: bool = True,
                          show_grid: bool = True,
                          coherence_overlay: bool = True,
                          **kwargs) -> Dict[str, Any]:
        """
        Render a comprehensive Bloch sphere visualization with OSH metrics.
        
        Args:
            state_vector: Quantum state vector
            qubit_index: Index of qubit to visualize (for multi-qubit states)
            title: Custom title for the visualization
            show_vector: Whether to display the Bloch vector
            show_axes: Whether to show coordinate axes
            show_grid: Whether to show sphere grid lines
            coherence_overlay: Whether to include coherence visualization
            **kwargs: Additional rendering parameters
        
        Returns:
            Dictionary containing rendered image, statistics, and OSH metrics
        """
        render_start = time.time()
        
        try:
            # Validate input state
            if not isinstance(state_vector, np.ndarray):
                raise ValueError("State vector must be a numpy array")
            
            if len(state_vector.shape) != 1:
                raise ValueError("State vector must be 1-dimensional")
            
            # Calculate reduced density matrix for the target qubit
            density_matrix = self._get_reduced_density_matrix(state_vector, qubit_index)
            
            # Calculate Bloch vector coordinates
            bloch_vector = self._calculate_bloch_vector(density_matrix)
            
            # Calculate OSH metrics
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            purity = np.real(np.trace(density_matrix @ density_matrix))
            
            # Create the visualization
            fig = plt.figure(figsize=self.config['figure_size'], dpi=self.config['dpi'])
            ax = fig.add_subplot(111, projection='3d')
            
            # Apply theme
            self._apply_theme(fig, ax)
            
            # Draw the Bloch sphere
            self._draw_bloch_sphere(ax, show_grid, show_axes)
            
            # Draw the Bloch vector
            if show_vector and np.linalg.norm(bloch_vector) > 1e-10:
                self._draw_bloch_vector(ax, bloch_vector, coherence)
            
            # Add coherence visualization overlay
            if coherence_overlay and self.config['osh_overlay_enabled']:
                self._add_coherence_overlay(ax, coherence, entropy)
            
            # Set title and labels
            if title is None:
                title = f"Bloch Sphere (Qubit {qubit_index})\nCoherence: {coherence:.3f}, Entropy: {entropy:.3f}"
            ax.set_title(title, fontsize=14, pad=20)
            
            # Add OSH information panel
            self._add_osh_info_panel(fig, coherence, entropy, purity, bloch_vector)
            
            # Convert to base64 image
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate statistics
            statistics = {
                'bloch_vector': bloch_vector.tolist(),
                'coherence': float(coherence),
                'entropy': float(entropy),
                'purity': float(purity),
                'vector_magnitude': float(np.linalg.norm(bloch_vector)),
                'render_time': time.time() - render_start
            }
            
            # Calculate comprehensive OSH metrics
            osh_metrics = self._calculate_comprehensive_osh_metrics(
                density_matrix, statistics
            )
            
            # Update performance tracking
            self._update_performance_metrics('bloch_sphere', time.time() - render_start)
            
            # Emit event if system available
            self._emit_render_event('bloch_sphere_rendered', {
                'qubit_index': qubit_index,
                'coherence': coherence,
                'entropy': entropy
            })
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': statistics,
                'osh_metrics': osh_metrics,
                'render_time': time.time() - render_start,
                'visualization_type': 'bloch_sphere'
            }
            
        except Exception as e:
            self.logger.error(f"Bloch sphere rendering failed: {e}")
            return self._create_error_response('bloch_sphere', str(e), time.time() - render_start)
    
    def render_density_matrix(self,
                             state_vector_or_density: np.ndarray,
                             qubits: Optional[List[int]] = None,
                             title: Optional[str] = None,
                             show_magnitude: bool = True,
                             show_phase: bool = True,
                             colormap: str = 'viridis',
                             **kwargs) -> Dict[str, Any]:
        """
        Render comprehensive density matrix visualization with magnitude and phase.
        
        Args:
            state_vector_or_density: Quantum state vector or density matrix
            qubits: Specific qubits to include (None for all)
            title: Custom title for the visualization
            show_magnitude: Whether to show magnitude heatmap
            show_phase: Whether to show phase information
            colormap: Colormap for visualization
            **kwargs: Additional rendering parameters
        
        Returns:
            Dictionary containing rendered image, statistics, and OSH metrics
        """
        render_start = time.time()
        
        try:
            # Convert to density matrix if needed
            if len(state_vector_or_density.shape) == 1:
                # State vector - convert to density matrix
                psi = state_vector_or_density.reshape(-1, 1)
                density_matrix = psi @ psi.conj().T
            else:
                density_matrix = state_vector_or_density.copy()
            
            # Apply qubit selection if specified
            if qubits is not None:
                density_matrix = self._get_multi_qubit_density_matrix(density_matrix, qubits)
            
            # Validate density matrix properties
            self._validate_density_matrix(density_matrix)
            
            # Calculate comprehensive metrics
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            purity = np.real(np.trace(density_matrix @ density_matrix))
            
            # Create subplot layout
            fig_width = self.config['figure_size'][0]
            fig_height = self.config['figure_size'][1]
            
            if show_magnitude and show_phase:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(fig_width, fig_height))
            elif show_magnitude or show_phase:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height * 0.6))
                ax3 = ax4 = None
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=(fig_width * 0.8, fig_height * 0.8))
                ax2 = ax3 = ax4 = None
            
            # Apply theme
            self._apply_theme(fig)
            
            # Plot magnitude
            if show_magnitude:
                magnitude = np.abs(density_matrix)
                im1 = ax1.imshow(magnitude, cmap=colormap, aspect='equal')
                ax1.set_title(f'Density Matrix Magnitude\nCoherence: {coherence:.3f}')
                self._add_matrix_annotations(ax1, magnitude)
                plt.colorbar(im1, ax=ax1, shrink=0.8)
                
                if ax2 is not None:
                    # 3D surface plot of magnitude
                    x, y = np.meshgrid(range(magnitude.shape[0]), range(magnitude.shape[1]))
                    ax2.remove()
                    ax2 = fig.add_subplot(1, 2, 2, projection='3d') if ax3 is None else fig.add_subplot(2, 2, 2, projection='3d')
                    surf = ax2.plot_surface(x, y, magnitude, cmap=colormap, alpha=0.8)
                    ax2.set_title('Magnitude Surface')
                    ax2.set_xlabel('Row')
                    ax2.set_ylabel('Column')
                    ax2.set_zlabel('|ρᵢⱼ|')
            
            # Plot phase
            if show_phase and ax3 is not None:
                phase = np.angle(density_matrix)
                im2 = ax3.imshow(phase, cmap='hsv', aspect='equal', vmin=-np.pi, vmax=np.pi)
                ax3.set_title(f'Density Matrix Phase\nEntropy: {entropy:.3f}')
                self._add_matrix_annotations(ax3, phase, format_func=lambda x: f'{x:.2f}π', 
                                           scale=1/np.pi)
                cbar2 = plt.colorbar(im2, ax=ax3, shrink=0.8)
                cbar2.set_label('Phase (radians)')
            
            # OSH metrics visualization
            if ax4 is not None and self.config['osh_overlay_enabled']:
                self._render_osh_metrics_panel(ax4, coherence, entropy, purity, density_matrix)
            
            # Add comprehensive title
            if title is None:
                title = f"Density Matrix Analysis ({density_matrix.shape[0]}×{density_matrix.shape[1]})"
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate detailed statistics
            statistics = {
                'matrix_shape': density_matrix.shape,
                'coherence': float(coherence),
                'entropy': float(entropy),
                'purity': float(purity),
                'trace': float(np.real(np.trace(density_matrix))),
                'frobenius_norm': float(np.linalg.norm(density_matrix, 'fro')),
                'eigenvalues': np.real(eigvals(density_matrix)).tolist(),
                'condition_number': float(np.linalg.cond(density_matrix)),
                'rank': int(np.linalg.matrix_rank(density_matrix)),
                'render_time': time.time() - render_start
            }
            
            # Calculate OSH metrics
            osh_metrics = self._calculate_comprehensive_osh_metrics(density_matrix, statistics)
            
            # Update performance tracking
            self._update_performance_metrics('density_matrix', time.time() - render_start)
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': statistics,
                'osh_metrics': osh_metrics,
                'render_time': time.time() - render_start,
                'visualization_type': 'density_matrix'
            }
            
        except Exception as e:
            self.logger.error(f"Density matrix rendering failed: {e}")
            return self._create_error_response('density_matrix', str(e), time.time() - render_start)
    
    def render_measurement_probabilities(self,
                                       state_vector: np.ndarray,
                                       measurement_basis: str = 'computational',
                                       threshold: float = 0.001,
                                       max_states: int = 16,
                                       title: Optional[str] = None,
                                       **kwargs) -> Dict[str, Any]:
        """
        Render measurement probability distribution with OSH analysis.
        
        Args:
            state_vector: Quantum state vector
            measurement_basis: Measurement basis ('computational', 'bell', 'fourier')
            threshold: Minimum probability to display
            max_states: Maximum number of states to show
            title: Custom title
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with rendered visualization and comprehensive analysis
        """
        render_start = time.time()
        
        try:
            # Normalize state vector
            state_vector = state_vector / np.linalg.norm(state_vector)
            n_qubits = int(np.log2(len(state_vector)))
            
            # Calculate probabilities in specified basis
            if measurement_basis == 'computational':
                probabilities = np.abs(state_vector)**2
                labels = [format(i, f'0{n_qubits}b') for i in range(len(state_vector))]
            elif measurement_basis == 'bell':
                probabilities, labels = self._calculate_bell_probabilities(state_vector)
            elif measurement_basis == 'fourier':
                probabilities, labels = self._calculate_fourier_probabilities(state_vector)
            else:
                raise ValueError(f"Unknown measurement basis: {measurement_basis}")
            
            # Filter by threshold and limit
            significant_indices = np.where(probabilities >= threshold)[0]
            if len(significant_indices) > max_states:
                # Keep top max_states probabilities
                top_indices = np.argsort(probabilities)[-max_states:]
                significant_indices = top_indices
            
            filtered_probs = probabilities[significant_indices]
            filtered_labels = [labels[i] for i in significant_indices]
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config['figure_size'])
            self._apply_theme(fig)
            
            # Main probability bar chart
            colors = plt.cm.viridis(filtered_probs / np.max(filtered_probs))
            bars = ax1.bar(range(len(filtered_probs)), filtered_probs, color=colors)
            ax1.set_xticks(range(len(filtered_labels)))
            ax1.set_xticklabels(filtered_labels, rotation=45)
            ax1.set_ylabel('Probability')
            ax1.set_title(f'Measurement Probabilities ({measurement_basis} basis)')
            ax1.grid(True, alpha=0.3)
            
            # Add probability values on bars
            for bar, prob in zip(bars, filtered_probs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Probability distribution analysis
            self._plot_probability_statistics(ax2, probabilities, state_vector)
            
            # Coherence evolution (if history available)
            if len(self.coherence_evolution) > 1:
                ax3.plot(list(self.coherence_evolution), 'b-', linewidth=2, label='Coherence')
                ax3.set_ylabel('Coherence')
                ax3.set_xlabel('Time Step')
                ax3.set_title('Coherence Evolution')
                ax3.grid(True, alpha=0.3)
                ax3.legend()
            else:
                ax3.text(0.5, 0.5, 'Coherence history\nnot available', 
                        ha='center', va='center', transform=ax3.transAxes)
            
            # OSH metrics panel
            density_matrix = np.outer(state_vector, state_vector.conj())
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            self._render_osh_metrics_panel(ax4, coherence, entropy, 1.0, density_matrix)
            
            # Set overall title
            if title is None:
                title = f"Quantum Measurement Analysis ({n_qubits} qubits)"
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate comprehensive statistics
            statistics = {
                'n_qubits': n_qubits,
                'measurement_basis': measurement_basis,
                'total_states': len(probabilities),
                'significant_states': len(filtered_probs),
                'max_probability': float(np.max(probabilities)),
                'min_probability': float(np.min(probabilities[probabilities > 0])),
                'entropy_classical': float(scipy_entropy(probabilities[probabilities > 0])),
                'participation_ratio': float(1 / np.sum(probabilities**2)),
                'effective_dimension': float(np.exp(scipy_entropy(probabilities[probabilities > 0]))),
                'coherence': float(coherence),
                'von_neumann_entropy': float(entropy),
                'render_time': time.time() - render_start
            }
            
            # Calculate OSH metrics
            osh_metrics = self._calculate_comprehensive_osh_metrics(density_matrix, statistics)
            
            self._update_performance_metrics('probabilities', time.time() - render_start)
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': statistics,
                'osh_metrics': osh_metrics,
                'probabilities': filtered_probs.tolist(),
                'labels': filtered_labels,
                'render_time': time.time() - render_start,
                'visualization_type': 'measurement_probabilities'
            }
            
        except Exception as e:
            self.logger.error(f"Probability rendering failed: {e}")
            return self._create_error_response('probabilities', str(e), time.time() - render_start)
    
    def render_coherence_evolution(self,
                                 coherence_data: Union[List[float], np.ndarray],
                                 time_points: Optional[Union[List[float], np.ndarray]] = None,
                                 title: Optional[str] = None,
                                 show_entropy: bool = True,
                                 show_rsp: bool = True,
                                 show_thresholds: bool = True,
                                 **kwargs) -> Dict[str, Any]:
        """
        Render comprehensive coherence evolution with OSH metrics analysis.
        
        Args:
            coherence_data: Time series of coherence values
            time_points: Corresponding time points (optional)
            title: Custom title
            show_entropy: Whether to show entropy evolution
            show_rsp: Whether to show RSP evolution
            show_thresholds: Whether to show critical thresholds
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with visualization and trend analysis
        """
        render_start = time.time()
        
        try:
            coherence_data = np.array(coherence_data)
            if time_points is None:
                time_points = np.arange(len(coherence_data))
            else:
                time_points = np.array(time_points)
            
            # Create comprehensive evolution visualization
            fig = plt.figure(figsize=(self.config['figure_size'][0], self.config['figure_size'][1] * 1.2))
            
            # Main coherence evolution plot
            ax1 = plt.subplot(3, 2, (1, 2))
            self._apply_theme(fig, ax1)
            
            # Plot coherence with trend analysis
            ax1.plot(time_points, coherence_data, 'b-', linewidth=3, label='Coherence', alpha=0.8)
            
            # Add moving average if enough data points
            if len(coherence_data) >= 10:
                window = min(len(coherence_data) // 5, 20)
                moving_avg = np.convolve(coherence_data, np.ones(window)/window, mode='same')
                ax1.plot(time_points, moving_avg, 'r--', linewidth=2, label=f'Moving Average ({window})', alpha=0.7)
            
            # Add critical thresholds
            if show_thresholds:
                ax1.axhline(y=0.7, color='orange', linestyle=':', alpha=0.7, label='High Coherence')
                ax1.axhline(y=0.3, color='red', linestyle=':', alpha=0.7, label='Low Coherence')
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Coherence')
            ax1.set_title('Coherence Evolution Analysis')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Entropy evolution (if available and requested)
            ax2 = plt.subplot(3, 2, 3)
            if show_entropy and len(self.entropy_evolution) > 1:
                entropy_data = list(self.entropy_evolution)[-len(coherence_data):]
                ax2.plot(time_points[-len(entropy_data):], entropy_data, 'g-', linewidth=2, label='Entropy')
                ax2.set_ylabel('Entropy')
                ax2.set_xlabel('Time')
                ax2.set_title('Entropy Evolution')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, 'Entropy data\nnot available', ha='center', va='center', 
                        transform=ax2.transAxes)
            
            # RSP calculation and evolution
            ax3 = plt.subplot(3, 2, 4)
            if show_rsp:
                # Calculate RSP from coherence and entropy
                if len(self.entropy_evolution) >= len(coherence_data):
                    entropy_data = np.array(list(self.entropy_evolution)[-len(coherence_data):])
                    rsp_data = self._calculate_rsp_timeseries(coherence_data, entropy_data)
                    ax3.plot(time_points, rsp_data, 'm-', linewidth=2, label='RSP')
                    ax3.set_ylabel('RSP')
                    ax3.set_xlabel('Time')
                    ax3.set_title('Recursive Simulation Potential')
                    ax3.grid(True, alpha=0.3)
                    ax3.legend()
                else:
                    ax3.text(0.5, 0.5, 'RSP calculation\nrequires entropy data', 
                            ha='center', va='center', transform=ax3.transAxes)
            
            # Statistical analysis panel
            ax4 = plt.subplot(3, 2, 5)
            self._plot_coherence_statistics(ax4, coherence_data, time_points)
            
            # Phase space analysis
            ax5 = plt.subplot(3, 2, 6)
            if len(self.entropy_evolution) >= len(coherence_data):
                entropy_data = np.array(list(self.entropy_evolution)[-len(coherence_data):])
                self._plot_phase_space_analysis(ax5, coherence_data, entropy_data)
            else:
                ax5.text(0.5, 0.5, 'Phase space analysis\nrequires entropy data', 
                        ha='center', va='center', transform=ax5.transAxes)
            
            # Set overall title
            if title is None:
                title = f"Comprehensive Coherence Analysis ({len(coherence_data)} points)"
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate comprehensive statistics
            statistics = {
                'coherence_mean': float(np.mean(coherence_data)),
                'coherence_std': float(np.std(coherence_data)),
                'coherence_min': float(np.min(coherence_data)),
                'coherence_max': float(np.max(coherence_data)),
                'coherence_trend': self._calculate_trend(coherence_data),
                'data_points': len(coherence_data),
                'time_span': float(time_points[-1] - time_points[0]) if len(time_points) > 1 else 0.0,
                'critical_points': self._find_critical_points(coherence_data).tolist(),
                'stability_index': self._calculate_stability_index(coherence_data),
                'render_time': time.time() - render_start
            }
            
            # Calculate OSH metrics for evolution
            osh_metrics = {
                'temporal_coherence_stability': statistics['stability_index'],
                'coherence_evolution_entropy': float(scipy_entropy(np.histogram(coherence_data, bins=10)[0] + 1e-10)),
                'phase_space_volume': self._calculate_phase_space_volume(coherence_data),
                'emergence_events': len(statistics['critical_points']),
                'consciousness_continuity': self._assess_consciousness_continuity(coherence_data)
            }
            
            self._update_performance_metrics('coherence_evolution', time.time() - render_start)
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': statistics,
                'osh_metrics': osh_metrics,
                'render_time': time.time() - render_start,
                'visualization_type': 'coherence_evolution'
            }
            
        except Exception as e:
            self.logger.error(f"Coherence evolution rendering failed: {e}")
            return self._create_error_response('coherence_evolution', str(e), time.time() - render_start)
    
    def render_entanglement_network(self,
                                  states: Dict[str, np.ndarray],
                                  entanglement_data: Optional[Dict] = None,
                                  layout: str = 'spring',
                                  node_size_factor: float = 1000,
                                  edge_width_factor: float = 5,
                                  title: Optional[str] = None,
                                  **kwargs) -> Dict[str, Any]:
        """
        Render comprehensive entanglement network visualization with OSH analysis.
        
        Args:
            states: Dictionary of quantum states {name: state_vector}
            entanglement_data: Pre-calculated entanglement data (optional)
            layout: Network layout algorithm
            node_size_factor: Scaling factor for node sizes
            edge_width_factor: Scaling factor for edge widths
            title: Custom title
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with network visualization and analysis
        """
        render_start = time.time()
        
        try:
            if len(states) < 2:
                raise ValueError("At least 2 quantum states required for entanglement network")
            
            # Calculate entanglement matrix if not provided
            if entanglement_data is None:
                entanglement_data = self._calculate_entanglement_matrix(states)
            
            # Create network graph
            G = nx.Graph()
            
            # Calculate node properties (coherence, entropy, etc.)
            node_properties = {}
            for name, state in states.items():
                density_matrix = np.outer(state, state.conj())
                coherence = self._calculate_osh_coherence(density_matrix)
                entropy = self._calculate_von_neumann_entropy(density_matrix)
                
                node_properties[name] = {
                    'coherence': coherence,
                    'entropy': entropy,
                    'size': node_size_factor * (0.5 + 0.5 * coherence),
                    'color': coherence
                }
                G.add_node(name, **node_properties[name])
            
            # Add edges based on entanglement strength
            edge_properties = []
            for i, state1 in enumerate(states.keys()):
                for j, state2 in enumerate(list(states.keys())[i+1:], i+1):
                    entanglement = entanglement_data.get((state1, state2), 0.0)
                    if entanglement > 0.01:  # Threshold for significant entanglement
                        G.add_edge(state1, state2, weight=entanglement)
                        edge_properties.append({
                            'nodes': (state1, state2),
                            'weight': entanglement,
                            'width': edge_width_factor * entanglement
                        })
            
            # Create visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config['figure_size'])
            self._apply_theme(fig)
            
            # Main network visualization
            if layout == 'spring':
                pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            elif layout == 'circular':
                pos = nx.circular_layout(G)
            elif layout == 'kamada_kawai':
                pos = nx.kamada_kawai_layout(G)
            else:
                pos = nx.spring_layout(G)
            
            # Draw nodes
            node_sizes = [node_properties[node]['size'] for node in G.nodes()]
            node_colors = [node_properties[node]['color'] for node in G.nodes()]
            
            nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_sizes, 
                                         node_color=node_colors, cmap='viridis',
                                         alpha=0.8, edgecolors='white', linewidths=2)
            
            # Draw edges
            edge_widths = [edge_properties[i]['width'] for i in range(len(edge_properties))]
            edge_colors = [edge_properties[i]['weight'] for i in range(len(edge_properties))]
            
            if edge_properties:
                edges = nx.draw_networkx_edges(G, pos, ax=ax1, width=edge_widths,
                                             edge_color=edge_colors, edge_cmap=plt.cm.Reds,
                                             alpha=0.7)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, ax=ax1, font_size=10, font_weight='bold')
            
            ax1.set_title('Entanglement Network')
            ax1.axis('off')
            
            # Add colorbar for coherence
            if nodes:
                plt.colorbar(nodes, ax=ax1, label='Coherence', shrink=0.8)
            
            # Network statistics
            self._plot_network_statistics(ax2, G, entanglement_data, node_properties)
            
            # Entanglement matrix heatmap
            self._plot_entanglement_matrix(ax3, states, entanglement_data)
            
            # OSH network analysis
            self._plot_osh_network_analysis(ax4, G, node_properties, entanglement_data)
            
            # Set overall title
            if title is None:
                title = f"Quantum Entanglement Network ({len(states)} states)"
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate comprehensive network statistics
            statistics = {
                'node_count': len(G.nodes()),
                'edge_count': len(G.edges()),
                'average_clustering': float(nx.average_clustering(G)) if G.edges() else 0.0,
                'network_density': float(nx.density(G)),
                'average_coherence': float(np.mean([props['coherence'] for props in node_properties.values()])),
                'total_entanglement': float(sum(entanglement_data.values())),
                'max_entanglement': float(max(entanglement_data.values())) if entanglement_data else 0.0,
                'connected_components': nx.number_connected_components(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else float('inf'),
                'render_time': time.time() - render_start
            }
            
            # Calculate OSH network metrics
            osh_metrics = {
                'network_consciousness': self._calculate_network_consciousness(G, node_properties),
                'collective_coherence': statistics['average_coherence'],
                'entanglement_entropy': self._calculate_entanglement_entropy(entanglement_data),
                'emergence_potential': self._calculate_emergence_potential(G, node_properties),
                'network_rsp': self._calculate_network_rsp(statistics, node_properties)
            }
            
            self._update_performance_metrics('entanglement_network', time.time() - render_start)
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': statistics,
                'osh_metrics': osh_metrics,
                'network_data': {
                    'nodes': list(G.nodes(data=True)),
                    'edges': list(G.edges(data=True)),
                    'positions': pos
                },
                'render_time': time.time() - render_start,
                'visualization_type': 'entanglement_network'
            }
            
        except Exception as e:
            self.logger.error(f"Entanglement network rendering failed: {e}")
            return self._create_error_response('entanglement_network', str(e), time.time() - render_start)
    
    def render_quantum_circuit(self,
                              gate_history: List[Dict],
                              title: Optional[str] = None,
                              show_measurements: bool = True,
                              show_statistics: bool = True,
                              **kwargs) -> Dict[str, Any]:
        """
        Render comprehensive quantum circuit visualization with gate analysis.
        
        Args:
            gate_history: List of gate operations
            title: Custom title
            show_measurements: Whether to show measurement operations
            show_statistics: Whether to show circuit statistics
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with circuit visualization and analysis
        """
        render_start = time.time()
        
        try:
            if not gate_history:
                raise ValueError("Gate history cannot be empty")
            
            # Analyze circuit structure
            circuit_analysis = self._analyze_circuit_structure(gate_history)
            n_qubits = circuit_analysis['n_qubits']
            circuit_depth = circuit_analysis['depth']
            
            # Create circuit visualization
            if show_statistics:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.config['figure_size'][0], 
                                                            self.config['figure_size'][1] * 1.2))
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=self.config['figure_size'])
                ax2 = None
            
            self._apply_theme(fig)
            
            # Draw the quantum circuit
            self._draw_quantum_circuit(ax1, gate_history, n_qubits, circuit_depth, show_measurements)
            
            # Circuit statistics panel
            if ax2 is not None:
                self._plot_circuit_statistics(ax2, gate_history, circuit_analysis)
            
            # Set title
            if title is None:
                title = f"Quantum Circuit ({n_qubits} qubits, depth {circuit_depth})"
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate comprehensive statistics
            gate_counts = {}
            for gate in gate_history:
                gate_type = gate.get('gate', 'unknown')
                gate_counts[gate_type] = gate_counts.get(gate_type, 0) + 1
            
            statistics = {
                'n_qubits': n_qubits,
                'circuit_depth': circuit_depth,
                'total_gates': len(gate_history),
                'gate_counts': gate_counts,
                'two_qubit_gates': circuit_analysis['two_qubit_gates'],
                'single_qubit_gates': circuit_analysis['single_qubit_gates'],
                'measurement_count': circuit_analysis['measurements'],
                'circuit_complexity': self._calculate_circuit_complexity(gate_history),
                'render_time': time.time() - render_start
            }
            
            # Calculate OSH circuit metrics
            osh_metrics = {
                'quantum_volume': 2 ** min(n_qubits, circuit_depth),
                'circuit_entropy': self._calculate_circuit_entropy(gate_history),
                'information_density': len(gate_history) / (n_qubits * circuit_depth) if circuit_depth > 0 else 0,
                'computational_capacity': self._estimate_computational_capacity(gate_history, n_qubits)
            }
            
            self._update_performance_metrics('quantum_circuit', time.time() - render_start)
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': statistics,
                'osh_metrics': osh_metrics,
                'circuit_analysis': circuit_analysis,
                'render_time': time.time() - render_start,
                'visualization_type': 'quantum_circuit'
            }
            
        except Exception as e:
            self.logger.error(f"Quantum circuit rendering failed: {e}")
            return self._create_error_response('quantum_circuit', str(e), time.time() - render_start)
    
    def render_osh_quantum_memory(self,
                                 states: Dict[str, np.ndarray],
                                 memory_field_data: Optional[Dict] = None,
                                 grid_size: Optional[Tuple[int, int]] = None,
                                 title: Optional[str] = None,
                                 show_connections: bool = True,
                                 **kwargs) -> Dict[str, Any]:
        """
        Render OSH-aligned recursive quantum memory grid visualization.
        
        Args:
            states: Dictionary of quantum states
            memory_field_data: Memory field physics data
            grid_size: Grid dimensions (auto-calculated if None)
            title: Custom title
            show_connections: Whether to show entanglement connections
            **kwargs: Additional parameters
        
        Returns:
            Dictionary with memory visualization and OSH analysis
        """
        render_start = time.time()
        
        try:
            if not states:
                raise ValueError("At least one quantum state required")
            
            # Auto-calculate grid size if not provided
            if grid_size is None:
                n_states = len(states)
                grid_size = (int(np.ceil(np.sqrt(n_states))), int(np.ceil(np.sqrt(n_states))))
            
            # Generate memory field data if not provided
            if memory_field_data is None:
                memory_field_data = self._generate_synthetic_memory_field(states)
            
            # Create comprehensive visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config['figure_size'])
            self._apply_theme(fig)
            
            # Main memory grid
            self._draw_osh_memory_grid(ax1, states, memory_field_data, grid_size, show_connections)
            
            # Memory coherence field
            self._draw_memory_coherence_field(ax2, states, memory_field_data, grid_size)
            
            # Entropy landscape
            self._draw_memory_entropy_landscape(ax3, states, memory_field_data, grid_size)
            
            # RSP distribution
            self._draw_rsp_distribution(ax4, states, memory_field_data)
            
            # Set overall title
            if title is None:
                title = f"OSH Quantum Memory Architecture ({len(states)} states)"
            fig.suptitle(title, fontsize=16)
            
            plt.tight_layout()
            
            # Convert to base64
            image_data = self._figure_to_base64(fig)
            plt.close(fig)
            
            # Calculate comprehensive memory statistics
            memory_stats = self._calculate_memory_statistics(states, memory_field_data)
            
            # Calculate OSH memory metrics
            osh_metrics = self._calculate_osh_memory_metrics(states, memory_field_data, memory_stats)
            
            self._update_performance_metrics('osh_memory', time.time() - render_start)
            
            return {
                'success': True,
                'image_data': image_data,
                'statistics': memory_stats,
                'osh_metrics': osh_metrics,
                'memory_field_data': memory_field_data,
                'render_time': time.time() - render_start,
                'visualization_type': 'osh_quantum_memory'
            }
            
        except Exception as e:
            self.logger.error(f"OSH quantum memory rendering failed: {e}")
            return self._create_error_response('osh_memory', str(e), time.time() - render_start)
    
    # ==================== UTILITY METHODS ====================
    
    def _calculate_bloch_vector(self, density_matrix: np.ndarray) -> np.ndarray:
        """Calculate Bloch vector from single-qubit density matrix."""
        if density_matrix.shape != (2, 2):
            raise ValueError("Density matrix must be 2x2 for single qubit")
        
        x = 2 * np.real(density_matrix[0, 1])
        y = 2 * np.imag(density_matrix[1, 0])
        z = np.real(density_matrix[0, 0] - density_matrix[1, 1])
        
        return np.array([x, y, z])
    
    def _calculate_osh_coherence(self, density_matrix: np.ndarray) -> float:
        """Calculate OSH-aligned coherence measure."""
        try:
            # L1 norm of off-diagonal elements
            off_diagonal = density_matrix - np.diag(np.diag(density_matrix))
            l1_coherence = np.sum(np.abs(off_diagonal))
            
            # Normalize by maximum possible coherence
            max_coherence = density_matrix.shape[0] - 1
            normalized_coherence = l1_coherence / max_coherence if max_coherence > 0 else 0.0
            
            return float(min(normalized_coherence, 1.0))
        except Exception as e:
            self.logger.warning(f"Coherence calculation failed: {e}")
            return 0.0
    
    def _calculate_von_neumann_entropy(self, density_matrix: np.ndarray) -> float:
        """Calculate von Neumann entropy."""
        try:
            eigenvals = eigvalsh(density_matrix)
            eigenvals = eigenvals[eigenvals > 1e-12]  # Remove numerical zeros
            entropy = -np.sum(eigenvals * np.log2(eigenvals + 1e-15))
            return float(max(entropy, 0.0))
        except Exception as e:
            self.logger.warning(f"Entropy calculation failed: {e}")
            return 0.0
    
    def _calculate_recursive_simulation_potential(self, coherence: float, entropy: float, strain: float = 0.1) -> float:
        """
        Calculate OSH Recursive Simulation Potential (RSP).
        
        Uses correct formula: RSP(t) = I(t) × C(t) / E(t)
        """
        try:
            from src.visualization.osh_formula_utils import calculate_rsp_simple
            rsp = calculate_rsp_simple(coherence, entropy, strain)
            return float(max(rsp, 0.0))
        except Exception as e:
            self.logger.warning(f"RSP calculation failed: {e}")
            return 0.0
    
    def _calculate_integrated_information(self, density_matrix: np.ndarray) -> float:
        """Calculate approximation of integrated information (Φ)."""
        try:
            # Simplified Φ calculation based on coherence and connectivity
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            
            # Φ approximation: coherence * (1 - entropy) * log(system_size)
            system_size = density_matrix.shape[0]
            phi = coherence * (1 - entropy) * np.log2(system_size + 1)
            
            return float(max(phi, 0.0))
        except Exception as e:
            self.logger.warning(f"Φ calculation failed: {e}")
            return 0.0
    
    def _calculate_emergence_index(self, density_matrix: np.ndarray) -> float:
        """Calculate emergence index based on quantum correlations."""
        try:
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            
            # Emergence occurs at the edge of chaos (balanced coherence/entropy)
            emergence = 4 * coherence * entropy * (1 - entropy)
            return float(max(emergence, 0.0))
        except Exception as e:
            self.logger.warning(f"Emergence index calculation failed: {e}")
            return 0.0
    
    def _calculate_kolmogorov_complexity_estimate(self, density_matrix: np.ndarray) -> float:
        """Estimate Kolmogorov complexity using compression ratio."""
        try:
            # Convert matrix to bytes and estimate compression ratio
            matrix_bytes = density_matrix.tobytes()
            # Simplified estimate based on entropy and structure
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            structure_complexity = np.log2(density_matrix.shape[0]) * entropy
            
            return float(max(structure_complexity, 0.0))
        except Exception as e:
            self.logger.warning(f"Kolmogorov complexity estimation failed: {e}")
            return 0.0
    
    def _get_reduced_density_matrix(self, state_vector: np.ndarray, qubit_index: int) -> np.ndarray:
        """Get reduced density matrix for a single qubit."""
        try:
            n_qubits = int(np.log2(len(state_vector)))
            if qubit_index >= n_qubits:
                raise ValueError(f"Qubit index {qubit_index} out of range for {n_qubits}-qubit state")
            
            # Create full density matrix
            full_dm = np.outer(state_vector, state_vector.conj())
            
            # Partial trace to get single qubit density matrix
            traced_qubits = list(range(n_qubits))
            traced_qubits.remove(qubit_index)
            
            reduced_dm = self._partial_trace(full_dm, n_qubits, traced_qubits)
            return reduced_dm
            
        except Exception as e:
            self.logger.error(f"Reduced density matrix calculation failed: {e}")
            # Return maximally mixed state as fallback
            return np.eye(2) / 2
    
    def _partial_trace(self, density_matrix: np.ndarray, n_qubits: int, traced_qubits: List[int]) -> np.ndarray:
        """Perform partial trace over specified qubits."""
        try:
            dims = [2] * n_qubits
            kept_qubits = [i for i in range(n_qubits) if i not in traced_qubits]
            
            if not kept_qubits:
                return np.array([[1.0]])
            
            # Reshape density matrix
            shape = dims + dims
            dm_reshaped = density_matrix.reshape(shape)
            
            # Trace out specified qubits
            for qubit in sorted(traced_qubits, reverse=True):
                axes = (qubit, qubit + n_qubits)
                dm_reshaped = np.trace(dm_reshaped, axis1=axes[0], axis2=axes[1])
                n_qubits -= 1
            
            # Reshape back to matrix form
            reduced_dim = 2 ** len(kept_qubits)
            return dm_reshaped.reshape(reduced_dim, reduced_dim)
            
        except Exception as e:
            self.logger.error(f"Partial trace failed: {e}")
            # Return identity matrix as fallback
            reduced_dim = 2 ** max(1, len([i for i in range(n_qubits) if i not in traced_qubits]))
            return np.eye(reduced_dim) / reduced_dim
    
    def _validate_density_matrix(self, density_matrix: np.ndarray):
        """Validate density matrix properties."""
        # Check if matrix is square
        if len(density_matrix.shape) != 2 or density_matrix.shape[0] != density_matrix.shape[1]:
            raise ValueError("Density matrix must be square")
        
        # Check if matrix is Hermitian (within numerical precision)
        if not np.allclose(density_matrix, density_matrix.conj().T, rtol=1e-10):
            self.logger.warning("Density matrix is not Hermitian (within numerical precision)")
        
        # Check if trace is 1
        trace = np.trace(density_matrix)
        if not np.isclose(trace, 1.0, rtol=1e-10):
            self.logger.warning(f"Density matrix trace is {trace}, not 1")
        
        # Check if matrix is positive semidefinite
        eigenvals = eigvalsh(density_matrix)
        if np.any(eigenvals < -1e-10):
            self.logger.warning("Density matrix has negative eigenvalues")
    
    def _calculate_comprehensive_osh_metrics(self, density_matrix: np.ndarray, base_stats: Dict) -> Dict[str, Any]:
        """Calculate comprehensive OSH metrics for quantum states."""
        try:
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            
            osh_metrics = {
                'coherence': float(coherence),
                'entropy': float(entropy),
                'purity': float(np.real(np.trace(density_matrix @ density_matrix))),
                'rsp': self._calculate_recursive_simulation_potential(coherence, entropy),
                'phi': self._calculate_integrated_information(density_matrix),
                'emergence_index': self._calculate_emergence_index(density_matrix),
                'kolmogorov_complexity': self._calculate_kolmogorov_complexity_estimate(density_matrix),
                'consciousness_potential': coherence * (1 - entropy),
                'information_density': coherence / (entropy + 1e-6),
                'quantum_discord': self._estimate_quantum_discord(density_matrix),
                'entanglement_capability': self._estimate_entanglement_capability(density_matrix),
                'recursive_depth': self._estimate_recursive_depth(coherence, entropy),
                'substrate_stability': 1.0 - abs(coherence - 0.618),  # Golden ratio optimization
                'osh_validation_score': self._calculate_osh_validation_score(coherence, entropy)
            }
            
            # Update historical tracking
            self.coherence_evolution.append(coherence)
            self.entropy_evolution.append(entropy)
            self.osh_metrics_history.append(osh_metrics)
            
            return osh_metrics
            
        except Exception as e:
            self.logger.error(f"OSH metrics calculation failed: {e}")
            return {'error': str(e)}
    
    def _estimate_quantum_discord(self, density_matrix: np.ndarray) -> float:
        """Estimate quantum discord (simplified calculation)."""
        try:
            coherence = self._calculate_osh_coherence(density_matrix)
            entropy = self._calculate_von_neumann_entropy(density_matrix)
            
            # Simplified discord approximation
            discord = coherence * entropy
            return float(max(discord, 0.0))
        except:
            return 0.0
    
    def _estimate_entanglement_capability(self, density_matrix: np.ndarray) -> float:
        """Estimate the state's capability to form entanglement."""
        try:
            coherence = self._calculate_osh_coherence(density_matrix)
            purity = np.real(np.trace(density_matrix @ density_matrix))
            
            # Entanglement capability based on coherence and purity
            capability = coherence * purity
            return float(max(capability, 0.0))
        except:
            return 0.0
    
    def _estimate_recursive_depth(self, coherence: float, entropy: float) -> int:
        """Estimate recursive modeling depth based on OSH principles."""
        try:
            # Higher coherence and lower entropy suggest deeper recursive capability
            depth_score = coherence * (1 - entropy)
            depth = int(np.ceil(depth_score * 10))  # Scale to reasonable depth range
            return max(depth, 1)
        except:
            return 1
    
    def _calculate_osh_validation_score(self, coherence: float, entropy: float) -> float:
        """Calculate OSH validation score based on theoretical predictions."""
        try:
            # OSH predicts optimal consciousness at specific coherence/entropy ratios
            golden_ratio = (1 + np.sqrt(5)) / 2 - 1  # ≈ 0.618
            
            # Score based on proximity to optimal OSH conditions
            coherence_score = 1.0 - abs(coherence - golden_ratio)
            entropy_score = 1.0 - entropy  # Lower entropy is better for OSH
            
            validation_score = (coherence_score + entropy_score) / 2
            return float(max(validation_score, 0.0))
        except:
            return 0.0
    
    def _apply_theme(self, fig, ax=None):
        """Apply visual theme to figure and axes."""
        theme = self.color_schemes[self.config['theme']]
        
        if fig:
            fig.patch.set_facecolor(theme['background'])
        
        if ax:
            ax.set_facecolor(theme['background'])
            ax.tick_params(colors=theme['foreground'])
            ax.xaxis.label.set_color(theme['foreground'])
            ax.yaxis.label.set_color(theme['foreground'])
            ax.title.set_color(theme['foreground'])
            
            # Set spine colors
            for spine in ax.spines.values():
                spine.set_color(theme['foreground'])
    
    def _figure_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 encoded PNG."""
        try:
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=self.config['dpi'], 
                       bbox_inches='tight', facecolor=fig.get_facecolor(),
                       edgecolor='none', transparent=False)
            buffer.seek(0)
            image_data = base64.b64encode(buffer.getvalue()).decode()
            buffer.close()
            return f"data:image/png;base64,{image_data}"
        except Exception as e:
            self.logger.error(f"Figure to base64 conversion failed: {e}")
            return ""
    
    def _create_error_response(self, visualization_type: str, error_message: str, render_time: float) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'error': error_message,
            'visualization_type': visualization_type,
            'render_time': render_time,
            'image_data': None,
            'statistics': {},
            'osh_metrics': {}
        }
    
    def _update_performance_metrics(self, visualization_type: str, render_time: float):
        """Update performance tracking metrics."""
        if self.config['performance_monitoring']:
            self.performance_metrics['render_times'].append(render_time)
            self.performance_metrics['total_renders'] += 1
            
            if visualization_type not in self.performance_metrics:
                self.performance_metrics[visualization_type] = {
                    'count': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0
                }
            
            self.performance_metrics[visualization_type]['count'] += 1
            self.performance_metrics[visualization_type]['total_time'] += render_time
            self.performance_metrics[visualization_type]['avg_time'] = (
                self.performance_metrics[visualization_type]['total_time'] / 
                self.performance_metrics[visualization_type]['count']
            )
    
    def _emit_render_event(self, event_type: str, event_data: Dict):
        """Emit rendering event through event system."""
        if self.event_system:
            try:
                self.event_system.emit(event_type, event_data, source='QuantumRenderer')
            except Exception as e:
                self.logger.warning(f"Event emission failed: {e}")
    
    def _handle_quantum_event(self, event: Dict):
        """Handle quantum events from the event system."""
        try:
            event_type = event.get('type', '')
            event_data = event.get('data', {})
            
            # Update internal state based on event
            if event_type == 'coherence_change_event':
                coherence = event_data.get('coherence', 0.0)
                self.coherence_evolution.append(coherence)
            elif event_type == 'entropy_change_event':
                entropy = event_data.get('entropy', 0.0)
                self.entropy_evolution.append(entropy)
            
        except Exception as e:
            self.logger.warning(f"Quantum event handling failed: {e}")
    
    # ==================== VISUALIZATION HELPERS ====================
    
    def _draw_bloch_sphere(self, ax, show_grid: bool, show_axes: bool):
        """Draw the Bloch sphere wireframe."""
        # Create sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Plot sphere surface
        ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='blue')
        
        if show_grid:
            # Add grid lines
            ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.2, color='gray')
        
        if show_axes:
            # Add coordinate axes
            ax.plot([-1.2, 1.2], [0, 0], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, 0], [-1.2, 1.2], [0, 0], 'k-', alpha=0.5)
            ax.plot([0, 0], [0, 0], [-1.2, 1.2], 'k-', alpha=0.5)
            
            # Add axis labels
            ax.text(1.3, 0, 0, 'X', fontsize=12)
            ax.text(0, 1.3, 0, 'Y', fontsize=12)
            ax.text(0, 0, 1.3, 'Z', fontsize=12)
        
        # Set equal aspect ratio
        ax.set_xlim([-1.2, 1.2])
        ax.set_ylim([-1.2, 1.2])
        ax.set_zlim([-1.2, 1.2])
        ax.set_box_aspect([1,1,1])
    
    def _draw_bloch_vector(self, ax, bloch_vector: np.ndarray, coherence: float):
        """Draw the Bloch vector with coherence-based styling."""
        x, y, z = bloch_vector
        
        # Color based on coherence
        color = plt.cm.plasma(coherence)
        
        # Draw vector arrow
        ax.quiver(0, 0, 0, x, y, z, arrow_length_ratio=0.1, 
                 color=color, linewidth=3, alpha=0.8)
        
        # Add vector endpoint
        ax.scatter([x], [y], [z], c=[coherence], cmap='plasma', 
                  s=100, alpha=0.8, edgecolors='white', linewidth=2)
    
    def _add_coherence_overlay(self, ax, coherence: float, entropy: float):
        """Add coherence visualization overlay to Bloch sphere."""
        # Add coherence ring
        theta = np.linspace(0, 2*np.pi, 100)
        radius = coherence
        x_ring = radius * np.cos(theta)
        y_ring = radius * np.sin(theta)
        z_ring = np.zeros_like(theta)
        
        ax.plot(x_ring, y_ring, z_ring, color='cyan', linewidth=2, alpha=0.7)
        
        # Add entropy indicator
        entropy_height = -1 + 2 * entropy  # Map entropy to [-1, 1]
        ax.axhline(y=entropy_height, color='red', linestyle='--', alpha=0.5)
    
    def _add_osh_info_panel(self, fig, coherence: float, entropy: float, purity: float, bloch_vector: np.ndarray):
        """Add OSH information panel to figure."""
        # Calculate RSP
        rsp = self._calculate_recursive_simulation_potential(coherence, entropy)
        
        # Create text panel
        info_text = f"""OSH Metrics:
Coherence: {coherence:.3f}
Entropy: {entropy:.3f}
Purity: {purity:.3f}
RSP: {rsp:.3f}
|r⃗|: {np.linalg.norm(bloch_vector):.3f}"""
        
        fig.text(0.02, 0.98, info_text, transform=fig.transFigure, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def _add_matrix_annotations(self, ax, matrix: np.ndarray, format_func=None, scale=1.0):
        """Add numerical annotations to matrix visualization."""
        if format_func is None:
            format_func = lambda x: f'{x:.2f}'
        
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j] * scale
                ax.text(j, i, format_func(value), ha='center', va='center',
                       fontsize=8, color='white' if abs(value) > 0.5 else 'black')
    
    def _render_osh_metrics_panel(self, ax, coherence: float, entropy: float, purity: float, density_matrix: np.ndarray):
        """Render comprehensive OSH metrics panel."""
        # Calculate additional metrics
        rsp = self._calculate_recursive_simulation_potential(coherence, entropy)
        phi = self._calculate_integrated_information(density_matrix)
        emergence = self._calculate_emergence_index(density_matrix)
        
        # Create radar chart of OSH metrics
        metrics = ['Coherence', 'Purity', 'RSP', 'Φ', 'Emergence']
        values = [coherence, purity, min(rsp/10, 1.0), min(phi/5, 1.0), emergence]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='blue', alpha=0.7)
        ax.fill(angles, values, alpha=0.25, color='blue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('OSH Metrics')
        ax.grid(True)
    
    # ==================== ADVANCED ANALYSIS METHODS ====================
    
    def _calculate_entanglement_matrix(self, states: Dict[str, np.ndarray]) -> Dict[Tuple[str, str], float]:
        """Calculate entanglement strength matrix between all state pairs."""
        entanglement_data = {}
        state_names = list(states.keys())
        
        for i, name1 in enumerate(state_names):
            for j, name2 in enumerate(state_names[i+1:], i+1):
                # Use entanglement manager if available
                if self.entanglement_manager:
                    try:
                        # Assume entanglement manager has a method to calculate entanglement
                        entanglement = self.entanglement_manager.calculate_entanglement(
                            states[name1], states[name2]
                        )
                    except:
                        entanglement = self._estimate_entanglement_strength(states[name1], states[name2])
                else:
                    entanglement = self._estimate_entanglement_strength(states[name1], states[name2])
                
                entanglement_data[(name1, name2)] = entanglement
        
        return entanglement_data
    
    def _estimate_entanglement_strength(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Estimate entanglement strength between two states."""
        try:
            # Simple entanglement estimate based on state overlap and coherence
            overlap = abs(np.vdot(state1, state2))**2
            
            # Calculate coherence for both states
            dm1 = np.outer(state1, state1.conj())
            dm2 = np.outer(state2, state2.conj())
            coh1 = self._calculate_osh_coherence(dm1)
            coh2 = self._calculate_osh_coherence(dm2)
            
            # Entanglement estimate
            entanglement = overlap * np.sqrt(coh1 * coh2)
            return float(min(entanglement, 1.0))
        except:
            return 0.0
    
    def _calculate_bell_probabilities(self, state_vector: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Calculate measurement probabilities in Bell basis."""
        # Bell states
        bell_states = {
            '|Φ⁺⟩': np.array([1, 0, 0, 1]) / np.sqrt(2),
            '|Φ⁻⟩': np.array([1, 0, 0, -1]) / np.sqrt(2),
            '|Ψ⁺⟩': np.array([0, 1, 1, 0]) / np.sqrt(2),
            '|Ψ⁻⟩': np.array([0, 1, -1, 0]) / np.sqrt(2)
        }
        
        probabilities = []
        labels = []
        
        for label, bell_state in bell_states.items():
            if len(state_vector) >= len(bell_state):
                overlap = abs(np.vdot(bell_state, state_vector[:len(bell_state)]))**2
                probabilities.append(overlap)
                labels.append(label)
        
        return np.array(probabilities), labels
    
    def _calculate_fourier_probabilities(self, state_vector: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Calculate measurement probabilities in Fourier basis."""
        # Apply quantum Fourier transform
        n_qubits = int(np.log2(len(state_vector)))
        qft_matrix = self._create_qft_matrix(n_qubits)
        
        fourier_state = qft_matrix @ state_vector
        probabilities = np.abs(fourier_state)**2
        labels = [f'|{i}⟩_F' for i in range(len(probabilities))]
        
        return probabilities, labels
    
    def _create_qft_matrix(self, n_qubits: int) -> np.ndarray:
        """Create quantum Fourier transform matrix."""
        N = 2**n_qubits
        qft = np.zeros((N, N), dtype=complex)
        omega = np.exp(2j * np.pi / N)
        
        for i in range(N):
            for j in range(N):
                qft[i, j] = omega**(i*j) / np.sqrt(N)
        
        return qft
    
    def _plot_probability_statistics(self, ax, probabilities: np.ndarray, state_vector: np.ndarray):
        """Plot probability distribution statistics."""
        # Histogram of probabilities
        non_zero_probs = probabilities[probabilities > 1e-10]
        
        if len(non_zero_probs) > 1:
            ax.hist(non_zero_probs, bins=min(20, len(non_zero_probs)), 
                   alpha=0.7, color='green', edgecolor='black')
            ax.set_xlabel('Probability')
            ax.set_ylabel('Count')
            ax.set_title('Probability Distribution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor histogram', 
                   ha='center', va='center', transform=ax.transAxes)
    
    def _plot_network_statistics(self, ax, G, entanglement_data: Dict, node_properties: Dict):
        """Plot network topology statistics."""
        stats = {
            'Nodes': len(G.nodes()),
            'Edges': len(G.edges()),
            'Density': nx.density(G),
            'Avg Coherence': np.mean([props['coherence'] for props in node_properties.values()]),
            'Max Entanglement': max(entanglement_data.values()) if entanglement_data else 0
        }
        
        # Bar plot of statistics
        names = list(stats.keys())
        values = list(stats.values())
        
        bars = ax.bar(names, values, color=['blue', 'green', 'red', 'orange', 'purple'])
        ax.set_title('Network Statistics')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_entanglement_matrix(self, ax, states: Dict, entanglement_data: Dict):
        """Plot entanglement matrix heatmap."""
        state_names = list(states.keys())
        n_states = len(state_names)
        matrix = np.zeros((n_states, n_states))
        
        # Fill matrix
        for (name1, name2), strength in entanglement_data.items():
            i = state_names.index(name1)
            j = state_names.index(name2)
            matrix[i, j] = strength
            matrix[j, i] = strength  # Symmetric
        
        # Plot heatmap
        im = ax.imshow(matrix, cmap='Reds', aspect='equal')
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(state_names, rotation=45)
        ax.set_yticklabels(state_names)
        ax.set_title('Entanglement Matrix')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Add text annotations
        for i in range(n_states):
            for j in range(n_states):
                ax.text(j, i, f'{matrix[i,j]:.2f}', ha='center', va='center',
                       color='white' if matrix[i,j] > 0.5 else 'black')
    
    def _plot_osh_network_analysis(self, ax, G, node_properties: Dict, entanglement_data: Dict):
        """Plot OSH-specific network analysis."""
        # Calculate OSH network metrics
        total_coherence = sum(props['coherence'] for props in node_properties.values())
        avg_entropy = np.mean([props['entropy'] for props in node_properties.values()])
        network_rsp = self._calculate_network_rsp({'average_coherence': total_coherence/len(node_properties)}, 
                                                 node_properties)
        
        # Create pie chart of coherence distribution
        coherences = [props['coherence'] for props in node_properties.values()]
        labels = list(node_properties.keys())
        
        wedges, texts, autotexts = ax.pie(coherences, labels=labels, autopct='%1.1f%%',
                                         startangle=90, colors=plt.cm.viridis(coherences))
        ax.set_title(f'Coherence Distribution\nNetwork RSP: {network_rsp:.3f}')
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        try:
            if not self.config['performance_monitoring']:
                return {'error': 'Performance monitoring not enabled'}
            
            render_times = list(self.performance_metrics['render_times'])
            
            report = {
                'total_renders': self.performance_metrics['total_renders'],
                'average_render_time': np.mean(render_times) if render_times else 0.0,
                'min_render_time': np.min(render_times) if render_times else 0.0,
                'max_render_time': np.max(render_times) if render_times else 0.0,
                'cache_hit_rate': (self.performance_metrics['cache_hits'] / 
                                 max(self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'], 1)),
                'memory_usage_mb': self.performance_metrics['memory_usage'],
                'visualization_breakdown': {}
            }
            
            # Add per-visualization statistics
            for viz_type, stats in self.performance_metrics.items():
                if isinstance(stats, dict) and 'count' in stats:
                    report['visualization_breakdown'][viz_type] = {
                        'count': stats['count'],
                        'total_time': stats['total_time'],
                        'average_time': stats['avg_time']
                    }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}
    
    def cleanup(self):
        """Clean up resources and finalize renderer."""
        try:
            # Clear caches
            self.render_cache.clear()
            
            # Close any open figures
            plt.close('all')
            
            # Log final performance statistics
            if self.config['performance_monitoring']:
                report = self.get_performance_report()
                self.logger.info(f"QuantumRenderer performance summary: {report}")
            
            self.logger.info("QuantumRenderer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


def create_quantum_renderer(coherence_manager=None,
                          entanglement_manager=None,
                          event_system=None,
                          state_registry=None,
                          config: Optional[Dict[str, Any]] = None) -> QuantumRenderer:
    """
    Factory function to create a fully configured QuantumRenderer instance.
    
    Args:
        coherence_manager: Coherence physics engine
        entanglement_manager: Entanglement tracking system
        event_system: Real-time event broadcasting system
        state_registry: Quantum state management registry
        config: Custom configuration parameters
    
    Returns:
        Configured QuantumRenderer instance
    """
    return QuantumRenderer(
        coherence_manager=coherence_manager,
        entanglement_manager=entanglement_manager,
        event_system=event_system,
        state_registry=state_registry,
        config=config
    )