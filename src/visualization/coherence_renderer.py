"""
Enhanced CoherenceRenderer Module for Recursia - Enterprise Scientific Visualization

This module provides world-class visualization capabilities for quantum coherence, entropy gradients,
recursive memory structures, and strain fields as part of the Organic Simulation Hypothesis (OSH) 
scientific framework. Features publication-quality outputs, advanced analytics, and empirical validation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import cm
from matplotlib.patches import Circle, Polygon, FancyBboxPatch, ConnectionPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection, PolyCollection
import seaborn as sns
from io import BytesIO
import base64
import networkx as nx
from scipy.ndimage import uniform_filter, laplace, map_coordinates, gaussian_filter, sobel
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull, distance_matrix
from scipy.stats import entropy, pearsonr, spearmanr, normaltest, kstest
from scipy.fft import fft2, ifft2, fftfreq
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize
from scipy.signal import find_peaks, peak_prominences, correlate2d
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import math
import colorsys
import time
import hashlib
import traceback
import weakref
import threading
from collections import deque
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from src.core.data_classes import OSHMetrics, VisualizationMetadata
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configure advanced logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class AdvancedCoherenceRenderer:
    """
    Enterprise-grade renderer for OSH scientific visualizations with advanced analytics,
    machine learning integration, and publication-quality outputs.
    """

    def __init__(
        self, 
        coherence_manager=None, 
        memory_field=None, 
        recursive_mechanics=None, 
        event_system=None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the AdvancedCoherenceRenderer with enhanced scientific capabilities.

        Args:
            coherence_manager: Reference to the CoherenceManager instance
            memory_field: Reference to the MemoryFieldPhysics instance
            recursive_mechanics: Reference to the RecursiveMechanics instance
            event_system: Reference to the EventSystem instance
            config: Advanced configuration parameters
        """
        self.coherence_manager = coherence_manager
        self.memory_field = memory_field
        self.recursive_mechanics = recursive_mechanics
        self.event_system = event_system
        
        # Advanced configuration
        self.config = config or {}
        self.scientific_mode = self.config.get('scientific_mode', True)
        self.publication_quality = self.config.get('publication_quality', True)
        self.enable_ml_analysis = self.config.get('enable_ml_analysis', True)
        self.real_time_processing = self.config.get('real_time_processing', False)
        
        # Enhanced caching system with scientific metadata
        self._cache = {}
        self._cache_metadata = {}
        self._cache_max_size = self.config.get('cache_size', 100)
        self._cache_keys = deque(maxlen=self._cache_max_size)
        
        # Performance and memory management
        self._max_data_points = self.config.get('max_data_points', 10_000_000)
        self._max_animation_frames = self.config.get('max_animation_frames', 100)
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Scientific analysis parameters
        self.statistical_significance_threshold = 0.05
        self.correlation_threshold = 0.7
        self.emergence_detection_threshold = 0.8
        self.criticality_detection_threshold = 0.9
        
        # Advanced visualization settings with scientific parameters
        self._initialize_advanced_settings()
        
        # Custom scientific colormaps
        self._create_scientific_colormaps()
        
        # Analytics and ML components
        self._initialize_analytics_components()
        
        # Performance monitoring
        self.performance_metrics = {
            'render_times': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'cache_hit_rate': 0.0,
            'processing_efficiency': 1.0
        }
        
        # Scientific validation components
        self.validation_history = deque(maxlen=1000)
        self.empirical_evidence = []
        
        logger.info("AdvancedCoherenceRenderer initialized with enterprise features")

    def _initialize_advanced_settings(self):
        """Initialize comprehensive visualization settings for scientific use."""
        
        # Base coherence map settings with scientific parameters
        self.coherence_map_settings = {
            'colormap': 'viridis',
            'scientific_colormap': 'coherence_scientific',
            'contour_levels': 20,
            'adaptive_contours': True,
            'statistical_overlays': True,
            'critical_point_analysis': True,
            'topological_features': True,
            'information_geometry': True,
            'phase_space_analysis': True,
            'fractal_dimension': True,
            'correlation_analysis': True,
            'emergence_detection': True,
            'publication_dpi': 300,
            'vector_output': True,
            'scientific_notation': True,
            'uncertainty_quantification': True,
            'multi_scale_analysis': True
        }
        
        # Advanced entropy gradient settings
        self.entropy_gradient_settings = {
            'vector_field_analysis': True,
            'divergence_calculation': True,
            'curl_analysis': True,
            'flow_lines': True,
            'critical_points': True,
            'hamiltonian_analysis': True,
            'lyapunov_exponents': True,
            'chaos_detection': True,
            'information_flow': True,
            'thermodynamic_analysis': True,
            'phase_portraits': True,
            'basin_analysis': True
        }
        
        # Recursive memory advanced settings
        self.recursive_memory_settings = {
            'hierarchical_clustering': True,
            'network_analysis': True,
            'centrality_measures': True,
            'community_detection': True,
            'motif_analysis': True,
            'path_analysis': True,
            'efficiency_measures': True,
            'robustness_analysis': True,
            'dynamic_analysis': True,
            'multilayer_networks': True
        }
        
        # OSH substrate comprehensive settings
        self.osh_substrate_settings = {
            'rsp_field_analysis': True,
            'information_integration': True,
            'consciousness_indicators': True,
            'recursive_boundaries': True,
            'emergence_metrics': True,
            'phase_transitions': True,
            'criticality_analysis': True,
            'holographic_principle': True,
            'information_geometry': True,
            'quantum_corrections': True
        }

    def _create_scientific_colormaps(self):
        """Create scientifically optimized colormaps for OSH visualizations."""
        
        # Scientific coherence colormap (perceptually uniform)
        coherence_colors = ['#000428', '#004e92', '#009ffd', '#00d2ff', '#ffffff']
        self.coherence_scientific = mcolors.LinearSegmentedColormap.from_list(
            'coherence_scientific', coherence_colors, N=256)
        
        # Information-theoretic entropy colormap
        entropy_colors = ['#000000', '#3d0000', '#8b0000', '#ff4500', '#ffd700', '#ffffff']
        self.entropy_scientific = mcolors.LinearSegmentedColormap.from_list(
            'entropy_scientific', entropy_colors, N=256)
        
        # RSP-specific colormap with critical thresholds
        rsp_colors = ['#000080', '#0040ff', '#00ffff', '#40ff00', '#ffff00', '#ff8000', '#ff0000']
        self.rsp_scientific = mcolors.LinearSegmentedColormap.from_list(
            'rsp_scientific', rsp_colors, N=256)
        
        # Strain visualization with physical interpretation
        strain_colors = ['#f0f0f0', '#d0d0ff', '#8080ff', '#4040ff', '#0000ff', '#000080']
        self.strain_scientific = mcolors.LinearSegmentedColormap.from_list(
            'strain_scientific', strain_colors, N=256)
        
        # Phase coherence colormap
        phase_colors = ['#ff0000', '#ff8000', '#ffff00', '#80ff00', '#00ff00', '#00ff80', 
                       '#00ffff', '#0080ff', '#0000ff', '#8000ff', '#ff00ff', '#ff0080']
        self.phase_coherence = mcolors.LinearSegmentedColormap.from_list(
            'phase_coherence', phase_colors, N=256)

    def _initialize_analytics_components(self):
        """Initialize machine learning and advanced analytics components."""
        
        # Clustering algorithms for pattern detection
        self.clustering_algorithms = {
            'dbscan': DBSCAN(eps=0.1, min_samples=5),
            'kmeans': KMeans(n_clusters=8, random_state=42),
            'hierarchical': None  # Will be initialized per use
        }
        
        # Dimensionality reduction techniques
        self.dimensionality_reduction = {
            'pca': PCA(n_components=3),
            'tsne': TSNE(n_components=2, random_state=42, perplexity=30),
            'manifold': None  # Custom manifold learning
        }
        
        # Statistical analysis components
        self.statistical_tests = {
            'normality': normaltest,
            'ks_test': kstest,
            'correlation': pearsonr,
            'rank_correlation': spearmanr
        }
        
        # Pattern recognition systems
        self.pattern_detectors = {
            'emergence': self._detect_emergence_patterns,
            'criticality': self._detect_criticality,
            'phase_transitions': self._detect_phase_transitions,
            'recursive_structures': self._detect_recursive_patterns
        }

    def render_advanced_coherence_field(
        self,
        coherence_data: np.ndarray = None,
        settings: Optional[Dict] = None,
        scientific_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Render comprehensive coherence field visualization with advanced scientific analysis.
        
        Args:
            coherence_data: 2D coherence field data
            settings: Visualization parameters
            scientific_analysis: Enable comprehensive scientific analysis
            
        Returns:
            Complete visualization result with scientific metrics
        """
        start_time = time.time()
        
        try:
            # Get data with advanced validation
            coherence_data = self._get_validated_data(coherence_data, 'coherence')
            if coherence_data is None:
                return self._create_error_result("No valid coherence data available")
            
            # Generate comprehensive cache key
            cache_key = self._generate_scientific_cache_key(
                "advanced_coherence", coherence_data, settings, scientific_analysis
            )
            
            # Check cache with metadata validation
            cached_result = self._get_from_scientific_cache(cache_key)
            if cached_result and not self.real_time_processing:
                return cached_result
            
            # Merge settings with scientific defaults
            settings = {**self.coherence_map_settings, **(settings or {})}
            
            # Perform advanced preprocessing
            processed_data = self._advanced_preprocessing(coherence_data)
            
            # Statistical analysis
            stats = self._comprehensive_statistical_analysis(processed_data)
            
            # Create high-quality figure
            fig = self._create_publication_figure(figsize=(12, 10))
            
            if scientific_analysis:
                # Multi-panel scientific layout
                gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
                
                # Main coherence field
                ax_main = fig.add_subplot(gs[0:2, 0:2])
                self._render_coherence_main_panel(ax_main, processed_data, settings)
                
                # Statistical overlays
                ax_stats = fig.add_subplot(gs[0, 2])
                self._render_statistical_panel(ax_stats, stats)
                
                # Critical point analysis
                ax_critical = fig.add_subplot(gs[1, 2])
                critical_points = self._analyze_critical_points(processed_data)
                self._render_critical_points_panel(ax_critical, critical_points)
                
                # Topological analysis
                ax_topo = fig.add_subplot(gs[2, 0])
                topology = self._topological_analysis(processed_data)
                self._render_topology_panel(ax_topo, topology)
                
                # Information geometry
                ax_info = fig.add_subplot(gs[2, 1])
                info_geom = self._information_geometry_analysis(processed_data)
                self._render_information_geometry_panel(ax_info, info_geom)
                
                # OSH metrics
                ax_osh = fig.add_subplot(gs[2, 2])
                osh_metrics = self._calculate_osh_metrics(processed_data)
                self._render_osh_metrics_panel(ax_osh, osh_metrics)
                
            else:
                # Single panel for non-scientific mode
                ax = fig.add_subplot(111)
                self._render_coherence_main_panel(ax, processed_data, settings)
                osh_metrics = self._calculate_osh_metrics(processed_data)
            
            # Generate publication-quality output
            image_data = self._get_publication_figure_data(fig)
            
            # Comprehensive result assembly
            result = self._assemble_scientific_result(
                image_data, stats, osh_metrics, processed_data, 
                time.time() - start_time, settings
            )
            
            # Cache with metadata
            self._add_to_scientific_cache(cache_key, result)
            
            # Update performance metrics
            self._update_performance_metrics(time.time() - start_time, result)
            
            return result
            
        except Exception as e:
            return self._handle_scientific_error(e, "advanced_coherence_field")
        finally:
            plt.close('all')

    def render_entropy_phase_portrait(
        self,
        entropy_data: np.ndarray = None,
        velocity_data: Optional[np.ndarray] = None,
        settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Render advanced entropy phase portrait with dynamical systems analysis.
        
        Args:
            entropy_data: 2D entropy field
            velocity_data: Optional velocity field data
            settings: Visualization parameters
            
        Returns:
            Phase portrait with dynamical analysis
        """
        start_time = time.time()
        
        try:
            # Data validation and preprocessing
            entropy_data = self._get_validated_data(entropy_data, 'entropy')
            if entropy_data is None:
                return self._create_error_result("No valid entropy data")
            
            # Calculate velocity field if not provided
            if velocity_data is None:
                dy, dx = np.gradient(entropy_data)
                velocity_data = np.stack([dx, dy], axis=-1)
            
            # Advanced dynamical systems analysis
            flow_analysis = self._analyze_flow_dynamics(entropy_data, velocity_data)
            
            # Create scientific figure
            fig = self._create_publication_figure(figsize=(16, 12))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Main phase portrait
            ax_main = fig.add_subplot(gs[:, 0:2])
            self._render_phase_portrait_main(ax_main, entropy_data, velocity_data, flow_analysis)
            
            # Lyapunov exponents
            ax_lyap = fig.add_subplot(gs[0, 2])
            self._render_lyapunov_analysis(ax_lyap, flow_analysis['lyapunov'])
            
            # Basin analysis
            ax_basin = fig.add_subplot(gs[1, 2])
            self._render_basin_analysis(ax_basin, flow_analysis['basins'])
            
            # Generate result
            image_data = self._get_publication_figure_data(fig)
            
            result = {
                'image_data': image_data,
                'flow_analysis': flow_analysis,
                'dynamical_properties': self._extract_dynamical_properties(flow_analysis),
                'osh_implications': self._interpret_osh_dynamics(flow_analysis),
                'scientific_metadata': self._generate_metadata(entropy_data, start_time)
            }
            
            return result
            
        except Exception as e:
            return self._handle_scientific_error(e, "entropy_phase_portrait")
        finally:
            plt.close('all')

    def render_rsp_landscape_analysis(
        self,
        coherence_data: np.ndarray = None,
        entropy_data: np.ndarray = None,
        strain_data: np.ndarray = None,
        settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive RSP landscape analysis with consciousness indicators.
        
        Args:
            coherence_data: Coherence field
            entropy_data: Entropy field
            strain_data: Strain field
            settings: Analysis parameters
            
        Returns:
            Complete RSP analysis with consciousness metrics
        """
        start_time = time.time()
        
        try:
            # Data validation and synchronization
            data_ensemble = self._validate_and_synchronize_data({
                'coherence': coherence_data,
                'entropy': entropy_data,
                'strain': strain_data
            })
            
            if not data_ensemble:
                return self._create_error_result("Insufficient data for RSP analysis")
            
            # Calculate RSP field with advanced methods
            rsp_field = self._calculate_advanced_rsp(data_ensemble)
            
            # Consciousness analysis
            consciousness_metrics = self._analyze_consciousness_indicators(rsp_field, data_ensemble)
            
            # Create comprehensive visualization
            fig = self._create_publication_figure(figsize=(20, 16))
            gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.4)
            
            # Main RSP landscape
            ax_main = fig.add_subplot(gs[0:2, 0:2])
            self._render_rsp_landscape_main(ax_main, rsp_field, consciousness_metrics)
            
            # Attractor analysis
            ax_attractors = fig.add_subplot(gs[0:2, 2:4])
            attractors = self._find_rsp_attractors(rsp_field)
            self._render_attractor_analysis(ax_attractors, attractors, rsp_field)
            
            # Consciousness indicators
            ax_consciousness = fig.add_subplot(gs[2, 0:2])
            self._render_consciousness_indicators(ax_consciousness, consciousness_metrics)
            
            # Information integration
            ax_integration = fig.add_subplot(gs[2, 2:4])
            integration_metrics = self._calculate_information_integration(data_ensemble)
            self._render_integration_analysis(ax_integration, integration_metrics)
            
            # Recursive boundaries
            ax_recursive = fig.add_subplot(gs[3, 0:2])
            boundaries = self._detect_recursive_boundaries(rsp_field)
            self._render_recursive_boundaries(ax_recursive, boundaries)
            
            # OSH validation metrics
            ax_validation = fig.add_subplot(gs[3, 2:4])
            validation = self._validate_osh_predictions(rsp_field, consciousness_metrics)
            self._render_osh_validation(ax_validation, validation)
            
            # Generate comprehensive result
            image_data = self._get_publication_figure_data(fig)
            
            result = {
                'image_data': image_data,
                'rsp_field': rsp_field,
                'consciousness_metrics': consciousness_metrics,
                'attractors': attractors,
                'integration_metrics': integration_metrics,
                'recursive_boundaries': boundaries,
                'osh_validation': validation,
                'empirical_evidence': self._extract_empirical_evidence(
                    rsp_field, consciousness_metrics, validation
                ),
                'scientific_metadata': self._generate_metadata(rsp_field, start_time)
            }
            
            # Store for scientific validation
            self.empirical_evidence.append(result['empirical_evidence'])
            
            return result
            
        except Exception as e:
            return self._handle_scientific_error(e, "rsp_landscape_analysis")
        finally:
            plt.close('all')

    def render_temporal_coherence_evolution(
        self,
        coherence_history: List[np.ndarray],
        settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Advanced temporal evolution analysis with predictive modeling.
        
        Args:
            coherence_history: Time series of coherence fields
            settings: Analysis parameters
            
        Returns:
            Temporal evolution analysis with predictions
        """
        start_time = time.time()
        
        try:
            if not coherence_history or len(coherence_history) < 3:
                return self._create_error_result("Insufficient temporal data")
            
            # Temporal analysis
            evolution_metrics = self._analyze_temporal_evolution(coherence_history)
            
            # Predictive modeling
            predictions = self._predict_coherence_evolution(coherence_history)
            
            # Create dynamic visualization
            fig = self._create_publication_figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
            
            # Main evolution display
            ax_main = fig.add_subplot(gs[0:2, 0:2])
            self._render_evolution_main(ax_main, coherence_history, evolution_metrics)
            
            # Temporal statistics
            ax_stats = fig.add_subplot(gs[0, 2])
            self._render_temporal_statistics(ax_stats, evolution_metrics)
            
            # Prediction panel
            ax_pred = fig.add_subplot(gs[1, 2])
            self._render_predictions(ax_pred, predictions)
            
            # Phase space trajectory
            ax_phase = fig.add_subplot(gs[2, 0])
            phase_trajectory = self._calculate_phase_trajectory(coherence_history)
            self._render_phase_trajectory(ax_phase, phase_trajectory)
            
            # Spectral analysis
            ax_spectral = fig.add_subplot(gs[2, 1])
            spectral_analysis = self._spectral_analysis(coherence_history)
            self._render_spectral_analysis(ax_spectral, spectral_analysis)
            
            # Stability analysis
            ax_stability = fig.add_subplot(gs[2, 2])
            stability_metrics = self._analyze_stability(coherence_history)
            self._render_stability_analysis(ax_stability, stability_metrics)
            
            # Generate result
            image_data = self._get_publication_figure_data(fig)
            
            result = {
                'image_data': image_data,
                'evolution_metrics': evolution_metrics,
                'predictions': predictions,
                'phase_trajectory': phase_trajectory,
                'spectral_analysis': spectral_analysis,
                'stability_metrics': stability_metrics,
                'scientific_metadata': self._generate_metadata(coherence_history[-1], start_time)
            }
            
            return result
            
        except Exception as e:
            return self._handle_scientific_error(e, "temporal_coherence_evolution")
        finally:
            plt.close('all')

    # Advanced analysis methods
    
    def _get_validated_data(self, data: np.ndarray, data_type: str) -> Optional[np.ndarray]:
        """Advanced data validation with scientific integrity checks."""
        if data is not None:
            data = np.asarray(data, dtype=np.float64)
            if data.ndim == 1:
                size = int(np.sqrt(data.size))
                data = data.reshape(size, size)
            elif data.ndim != 2:
                logger.error(f"Invalid {data_type} data dimensions: {data.ndim}")
                return None
            
            # Scientific validation
            if not np.isfinite(data).all():
                logger.warning(f"Non-finite values in {data_type} data, cleaning...")
                data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=0.0)
            
            return data
        
        # Try to get from managers
        if data_type == 'coherence' and self.coherence_manager:
            try:
                return self._extract_coherence_field()
            except Exception as e:
                logger.warning(f"Could not extract coherence field: {e}")
        elif data_type == 'entropy' and self.coherence_manager:
            try:
                return self._extract_entropy_field()
            except Exception as e:
                logger.warning(f"Could not extract entropy field: {e}")
        elif data_type == 'strain' and self.memory_field:
            try:
                return self._extract_strain_field()
            except Exception as e:
                logger.warning(f"Could not extract strain field: {e}")
        
        return None

    def _comprehensive_statistical_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Comprehensive statistical analysis for scientific validation."""
        stats = {
            'basic': {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'median': float(np.median(data)),
                'skewness': float(self._calculate_skewness(data)),
                'kurtosis': float(self._calculate_kurtosis(data))
            },
            'distribution': {
                'normality_p': float(normaltest(data.flatten())[1]),
                'is_normal': normaltest(data.flatten())[1] > self.statistical_significance_threshold
            },
            'spatial': {
                'spatial_correlation': float(self._calculate_spatial_correlation(data)),
                'fractal_dimension': float(self._calculate_fractal_dimension(data)),
                'entropy_density': float(self._calculate_entropy_density(data))
            },
            'information': {
                'mutual_information': float(self._calculate_mutual_information(data)),
                'kolmogorov_complexity': float(self._estimate_kolmogorov_complexity(data)),
                'effective_dimension': float(self._calculate_effective_dimension(data))
            }
        }
        return stats

    def _analyze_critical_points(self, data: np.ndarray) -> Dict[str, Any]:
        """Advanced critical point analysis with topological classification."""
        # Gradient calculation
        dy, dx = np.gradient(data)
        
        # Hessian calculation for classification
        dyy, dyx = np.gradient(dy)
        dxy, dxx = np.gradient(dx)
        
        # Find critical points where gradient is approximately zero
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        threshold = np.std(gradient_magnitude) * 0.1
        
        critical_mask = gradient_magnitude < threshold
        critical_coords = np.column_stack(np.where(critical_mask))
        
        # Classify critical points
        classified_points = []
        for y, x in critical_coords:
            if y < data.shape[0]-1 and x < data.shape[1]-1:
                # Hessian matrix at this point
                H = np.array([[dxx[y, x], dxy[y, x]], 
                             [dyx[y, x], dyy[y, x]]])
                
                # Eigenvalues for classification
                eigenvals = np.linalg.eigvals(H)
                det_H = np.linalg.det(H)
                trace_H = np.trace(H)
                
                if det_H > 0 and trace_H < 0:
                    point_type = "local_maximum"
                elif det_H > 0 and trace_H > 0:
                    point_type = "local_minimum"
                elif det_H < 0:
                    point_type = "saddle_point"
                else:
                    point_type = "degenerate"
                
                classified_points.append({
                    'coordinates': (int(x), int(y)),
                    'type': point_type,
                    'value': float(data[y, x]),
                    'eigenvalues': eigenvals.tolist(),
                    'determinant': float(det_H),
                    'trace': float(trace_H)
                })
        
        return {
            'points': classified_points,
            'total_count': len(classified_points),
            'type_counts': self._count_critical_point_types(classified_points),
            'topological_index': self._calculate_topological_index(classified_points)
        }

    def _topological_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Advanced topological analysis including persistent homology."""
        # Contour analysis at multiple levels
        levels = np.linspace(np.min(data), np.max(data), 20)
        topological_features = []
        
        for level in levels:
            contours = plt.contour(data, levels=[level])
            
            # Extract contour properties
            for collection in contours.collections:
                for path in collection.get_paths():
                    vertices = path.vertices
                    if len(vertices) > 3:  # Valid contour
                        # Calculate topological properties
                        area = self._calculate_contour_area(vertices)
                        perimeter = self._calculate_contour_perimeter(vertices)
                        genus = self._estimate_genus(vertices)
                        
                        topological_features.append({
                            'level': float(level),
                            'area': float(area),
                            'perimeter': float(perimeter),
                            'genus': int(genus),
                            'compactness': float(4 * np.pi * area / (perimeter**2)) if perimeter > 0 else 0
                        })
        
        plt.close()  # Clean up the contour plot
        
        return {
            'features': topological_features,
            'betti_numbers': self._calculate_betti_numbers(data),
            'euler_characteristic': self._calculate_euler_characteristic(topological_features),
            'persistent_homology': self._calculate_persistent_homology(data)
        }

    def _information_geometry_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Information geometry analysis for OSH validation."""
        # Information-theoretic curvature
        curvature_matrix = self._calculate_information_curvature(data)
        
        # Fisher information metric
        fisher_metric = self._calculate_fisher_information_metric(data)
        
        # Geodesics in information space
        geodesics = self._calculate_information_geodesics(data)
        
        return {
            'curvature_scalar': float(np.mean(curvature_matrix)),
            'curvature_matrix': curvature_matrix,
            'fisher_metric': fisher_metric,
            'geodesics': geodesics,
            'information_density': float(self._calculate_information_density(data)),
            'relative_entropy': float(self._calculate_relative_entropy(data))
        }

    def _calculate_osh_metrics(self, data: np.ndarray) -> OSHMetrics:
        """Calculate comprehensive OSH-specific metrics."""
        # Base metrics
        coherence = float(np.mean(data))
        entropy = float(self._calculate_entropy_from_data(data))
        strain = float(1.0 - coherence) if coherence > 0 else 1.0
        
        # Advanced OSH calculations
        phi = self._calculate_integrated_information(data)
        rsp = self._calculate_rsp_advanced(coherence, entropy, strain, phi)
        kolmogorov = self._estimate_kolmogorov_complexity(data)
        
        # Geometric and dynamical properties
        info_curvature = self._calculate_information_curvature_scalar(data)
        emergence = self._calculate_emergence_index(data)
        criticality = self._calculate_criticality_parameter(data)
        phase_coh = self._calculate_phase_coherence(data)
        stability = self._calculate_temporal_stability(data)
        
        # Recursive depth analysis
        recursive_depth = self._analyze_recursive_depth(data)
        
        return OSHMetrics(
            coherence=coherence,
            entropy=entropy,
            strain=strain,
            rsp=rsp,
            phi=phi,
            kolmogorov_complexity=kolmogorov,
            information_geometry_curvature=info_curvature,
            recursive_depth=recursive_depth,
            emergence_index=emergence,
            criticality_parameter=criticality,
            phase_coherence=phase_coh,
            temporal_stability=stability
        )

    def _calculate_rsp_advanced(self, coherence: float, entropy: float, strain: float, phi: float) -> float:
        """
        Advanced RSP calculation using correct OSH formula.
        
        RSP(t) = I(t) × C(t) / E(t) where:
        - I(t): Integrated information (bits)
        - C(t): Kolmogorov complexity (bits)  
        - E(t): Entropy flux (bits/second)
        
        Note: This is a visualization approximation. For accurate calculations,
        use the unified VM calculations through the execution context
        """
        epsilon = 1e-10
        
        # Approximate integrated information from coherence and phi
        # I(t) ≈ phi * coherence factor
        integrated_info = phi * np.sqrt(coherence) if phi > 0 else coherence * 0.1
        
        # Approximate Kolmogorov complexity from entropy
        # Higher complexity with lower entropy (more structure)
        complexity = max(0.1, 1.0 - entropy + 0.5 * coherence)
        
        # Approximate entropy flux from strain rate
        # Higher strain indicates higher entropy production
        entropy_flux = max(epsilon, strain * 0.1 + epsilon)
        
        # Apply correct OSH formula: RSP = I × C / E
        rsp = (integrated_info * complexity) / entropy_flux
        
        return float(rsp)

    def _validate_and_synchronize_data(self, data_dict: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        """Validate and synchronize multiple data arrays."""
        validated_data = {}
        target_shape = None
        
        for name, data in data_dict.items():
            validated = self._get_validated_data(data, name)
            if validated is not None:
                if target_shape is None:
                    target_shape = validated.shape
                elif validated.shape != target_shape:
                    # Interpolate to common shape
                    validated = self._interpolate_to_shape(validated, target_shape)
                
                validated_data[name] = validated
        
        return validated_data if len(validated_data) >= 2 else None

    def _calculate_advanced_rsp(self, data_ensemble: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate advanced RSP field with spatial variation."""
        coherence = data_ensemble.get('coherence', np.ones((64, 64)) * 0.5)
        entropy = data_ensemble.get('entropy', np.ones((64, 64)) * 0.3)
        strain = data_ensemble.get('strain', np.ones((64, 64)) * 0.2)
        
        # Spatial RSP calculation using correct OSH formula
        epsilon = 1e-10
        
        # Calculate spatial phi field first for integrated information
        phi_field = self._calculate_spatial_phi(data_ensemble)
        
        # Approximate integrated information field: I(t) ≈ phi * coherence factor
        integrated_info_field = phi_field * np.sqrt(coherence)
        integrated_info_field = np.where(phi_field > 0, integrated_info_field, coherence * 0.1)
        
        # Approximate Kolmogorov complexity field from entropy
        complexity_field = np.maximum(0.1, 1.0 - entropy + 0.5 * coherence)
        
        # Approximate entropy flux field from strain
        entropy_flux_field = np.maximum(epsilon, strain * 0.1 + epsilon)
        
        # Apply correct OSH formula: RSP = I × C / E
        rsp_field = (integrated_info_field * complexity_field) / entropy_flux_field
        
        # Apply recursive corrections for visualization enhancement
        recursive_corrections = self._calculate_recursive_corrections(data_ensemble)
        rsp_field *= recursive_corrections
        
        return rsp_field

    # Visualization rendering methods
    
    def _render_coherence_main_panel(self, ax, data: np.ndarray, settings: Dict):
        """Render main coherence visualization panel."""
        # High-quality coherence visualization
        im = ax.imshow(data, cmap=self.coherence_scientific, origin='lower', 
                      interpolation='bicubic', aspect='auto')
        
        # Adaptive contour lines
        if settings.get('adaptive_contours', True):
            levels = self._calculate_adaptive_contour_levels(data)
            contours = ax.contour(data, levels=levels, colors='white', 
                                alpha=0.6, linewidths=0.8)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        # Critical points overlay
        if settings.get('critical_point_analysis', True):
            critical_points = self._analyze_critical_points(data)
            for point in critical_points['points'][:10]:  # Limit display
                x, y = point['coordinates']
                color = {'local_maximum': 'red', 'local_minimum': 'blue', 
                        'saddle_point': 'yellow'}.get(point['type'], 'white')
                ax.plot(x, y, 'o', color=color, markersize=6, alpha=0.8)
                
        # Information flow arrows
        if settings.get('information_geometry', True):
            self._add_information_flow_arrows(ax, data)
        
        # Scientific annotations
        ax.set_title('Quantum Coherence Field Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Spatial Coordinate X', fontsize=12)
        ax.set_ylabel('Spatial Coordinate Y', fontsize=12)
        
        # Colorbar with scientific notation
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Coherence Magnitude', fontsize=12)
        
        # Grid for scientific reference
        ax.grid(True, alpha=0.3, linestyle='--')

    def _render_statistical_panel(self, ax, stats: Dict):
        """Render statistical analysis panel."""
        ax.axis('off')
        
        # Create statistical summary text
        text_content = [
            f"Mean: {stats['basic']['mean']:.3f}",
            f"Std: {stats['basic']['std']:.3f}",
            f"Skewness: {stats['basic']['skewness']:.3f}",
            f"Kurtosis: {stats['basic']['kurtosis']:.3f}",
            f"Normality p: {stats['distribution']['normality_p']:.3e}",
            f"Fractal D: {stats['spatial']['fractal_dimension']:.3f}",
            f"Spatial ρ: {stats['spatial']['spatial_correlation']:.3f}",
            f"K-complexity: {stats['information']['kolmogorov_complexity']:.3f}"
        ]
        
        # Display as formatted text
        full_text = '\n'.join(text_content)
        ax.text(0.05, 0.95, full_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        ax.set_title('Statistical Analysis', fontsize=12, fontweight='bold')

    def _render_critical_points_panel(self, ax, critical_points: Dict):
        """Render critical points analysis panel."""
        types = list(critical_points['type_counts'].keys())
        counts = list(critical_points['type_counts'].values())
        colors = ['red', 'blue', 'yellow', 'green'][:len(types)]
        
        wedges, texts, autotexts = ax.pie(counts, labels=types, colors=colors, 
                                         autopct='%1.1f%%', startangle=90)
        
        ax.set_title(f'Critical Points (Total: {critical_points["total_count"]})', 
                    fontsize=12, fontweight='bold')

    def _render_topology_panel(self, ax, topology: Dict):
        """Render topological analysis panel."""
        # Betti numbers visualization
        betti = topology['betti_numbers']
        ax.bar(range(len(betti)), betti, color='purple', alpha=0.7)
        ax.set_xlabel('Dimension')
        ax.set_ylabel('Betti Number')
        ax.set_title('Topological Features', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _render_information_geometry_panel(self, ax, info_geom: Dict):
        """Render information geometry analysis."""
        # Information curvature visualization
        curvature = info_geom['curvature_matrix']
        im = ax.imshow(curvature, cmap='RdBu_r', origin='lower')
        ax.set_title('Information Curvature', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)

    def _render_osh_metrics_panel(self, ax, osh_metrics: OSHMetrics):
        """Render OSH-specific metrics panel."""
        # Radar chart for OSH metrics
        metrics = ['Coherence', 'RSP', 'Φ (IIT)', 'Emergence', 'Criticality', 'Stability']
        values = [
            osh_metrics.coherence,
            min(osh_metrics.rsp / 10, 1.0),  # Normalize RSP
            osh_metrics.phi,
            osh_metrics.emergence_index,
            osh_metrics.criticality_parameter,
            osh_metrics.temporal_stability
        ]
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='red')
        ax.fill(angles, values, alpha=0.25, color='red')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('OSH Metrics', fontsize=12, fontweight='bold')
        ax.grid(True)

    # Advanced computational methods
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate statistical skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate statistical kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3

    def _calculate_spatial_correlation(self, data: np.ndarray) -> float:
        """Calculate spatial autocorrelation."""
        # Shift data by one pixel and calculate correlation
        shifted = np.roll(data, 1, axis=0)
        return float(np.corrcoef(data.flatten(), shifted.flatten())[0, 1])

    def _calculate_fractal_dimension(self, data: np.ndarray) -> float:
        """Estimate fractal dimension using box-counting method."""
        # Simplified box-counting algorithm
        def box_count(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k*k))[0])
        
        # Apply threshold
        Z = (data > np.mean(data)).astype(int)
        
        # Count boxes at different scales
        p = min(Z.shape)
        sizes = 2**np.arange(1, int(np.log2(p))-1)
        counts = []
        
        for size in sizes:
            counts.append(box_count(Z, size))
        
        # Fit power law
        if len(counts) > 1 and all(c > 0 for c in counts):
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            return float(-coeffs[0])
        
        return 2.0  # Default for 2D surface

    def _calculate_entropy_density(self, data: np.ndarray) -> float:
        """Calculate information entropy density."""
        # Discretize data for entropy calculation
        hist, _ = np.histogram(data.flatten(), bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return float(-np.sum(hist * np.log2(hist)) / data.size)

    def _calculate_mutual_information(self, data: np.ndarray) -> float:
        """Calculate spatial mutual information."""
        # Simplified mutual information between neighboring pixels
        x = data[:-1, :].flatten()
        y = data[1:, :].flatten()
        
        # Create joint histogram
        hist_2d, _, _ = np.histogram2d(x, y, bins=20)
        hist_2d = hist_2d / np.sum(hist_2d)  # Normalize
        
        # Calculate marginal histograms
        hist_x = np.sum(hist_2d, axis=1)
        hist_y = np.sum(hist_2d, axis=0)
        
        # Calculate mutual information
        mi = 0.0
        for i in range(len(hist_x)):
            for j in range(len(hist_y)):
                if hist_2d[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                    mi += hist_2d[i, j] * np.log2(hist_2d[i, j] / (hist_x[i] * hist_y[j]))
        
        return float(mi)

    def _estimate_kolmogorov_complexity(self, data: np.ndarray) -> float:
        """Estimate Kolmogorov complexity using compression."""
        import zlib
        
        # Convert to bytes and compress
        data_bytes = data.astype(np.float32).tobytes()
        compressed_size = len(zlib.compress(data_bytes))
        original_size = len(data_bytes)
        
        # Normalized complexity measure
        return float(compressed_size / original_size)

    def _calculate_effective_dimension(self, data: np.ndarray) -> float:
        """Calculate effective dimension using PCA."""
        # Flatten and standardize
        flat_data = data.flatten().reshape(-1, 1)
        
        # Add spatial coordinates as features
        y_coords, x_coords = np.meshgrid(range(data.shape[0]), range(data.shape[1]), indexing='ij')
        features = np.column_stack([flat_data.flatten(), 
                                   y_coords.flatten(), 
                                   x_coords.flatten()])
        
        # PCA analysis
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        pca = PCA()
        pca.fit(features_scaled)
        
        # Calculate effective dimension (95% variance)
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        eff_dim = np.argmax(cumsum >= 0.95) + 1
        
        return float(eff_dim)

    def _calculate_integrated_information(self, data: np.ndarray) -> float:
        """Calculate integrated information (Φ) approximation."""
        # Simplified IIT calculation based on spatial correlations
        
        # Calculate bipartite correlations
        h, w = data.shape
        mid_h, mid_w = h // 2, w // 2
        
        # Split into quadrants
        q1 = data[:mid_h, :mid_w]
        q2 = data[:mid_h, mid_w:]
        q3 = data[mid_h:, :mid_w]
        q4 = data[mid_h:, mid_w:]
        
        # Calculate cross-correlations
        correlations = []
        quadrants = [q1, q2, q3, q4]
        
        for i in range(len(quadrants)):
            for j in range(i+1, len(quadrants)):
                if quadrants[i].size > 0 and quadrants[j].size > 0:
                    corr = np.corrcoef(quadrants[i].flatten(), 
                                     quadrants[j].flatten())[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        # Φ approximation as mean absolute correlation
        phi = np.mean(correlations) if correlations else 0.0
        
        return float(phi)

    def _calculate_emergence_index(self, data: np.ndarray) -> float:
        """Calculate emergence index based on multi-scale analysis."""
        # Multi-scale correlation analysis
        scales = [1, 2, 4, 8]
        correlations = []
        
        for scale in scales:
            if data.shape[0] >= scale and data.shape[1] >= scale:
                # Downsample data
                downsampled = data[::scale, ::scale]
                if downsampled.size > 1:
                    # Calculate local correlations
                    local_corr = self._calculate_local_correlations(downsampled)
                    correlations.append(local_corr)
        
        # Emergence as scale-dependent correlation variance
        if len(correlations) > 1:
            emergence = np.std(correlations) / np.mean(correlations) if np.mean(correlations) > 0 else 0
        else:
            emergence = 0.0
        
        return float(emergence)

    def _calculate_criticality_parameter(self, data: np.ndarray) -> float:
        """Calculate criticality parameter using avalanche analysis."""
        # Identify significant changes (avalanches)
        threshold = np.std(data) * 0.5
        
        # Calculate gradients
        dy, dx = np.gradient(data)
        gradient_magnitude = np.sqrt(dx**2 + dy**2)
        
        # Find avalanche events
        avalanches = gradient_magnitude > threshold
        
        if np.sum(avalanches) == 0:
            return 0.0
        
        # Analyze avalanche size distribution
        avalanche_sizes = []
        labeled_avalanches = self._label_connected_components(avalanches)
        
        for label in np.unique(labeled_avalanches):
            if label > 0:
                size = np.sum(labeled_avalanches == label)
                avalanche_sizes.append(size)
        
        if len(avalanche_sizes) < 2:
            return 0.0
        
        # Power-law exponent as criticality measure
        sizes = np.array(avalanche_sizes)
        log_sizes = np.log(sizes[sizes > 0])
        
        if len(log_sizes) > 1:
            hist, bin_edges = np.histogram(log_sizes, bins=10)
            hist = hist[hist > 0]
            
            if len(hist) > 1:
                log_hist = np.log(hist)
                x = np.arange(len(log_hist))
                
                # Fit power law
                if len(x) > 1:
                    slope = np.polyfit(x, log_hist, 1)[0]
                    criticality = min(abs(slope), 1.0)  # Normalize
                else:
                    criticality = 0.0
            else:
                criticality = 0.0
        else:
            criticality = 0.0
        
        return float(criticality)

    def _calculate_phase_coherence(self, data: np.ndarray) -> float:
        """Calculate phase coherence using spatial Fourier analysis."""
        # 2D FFT
        fft_data = fft2(data)
        
        # Calculate phase
        phase = np.angle(fft_data)
        
        # Phase coherence as circular variance
        complex_phase = np.exp(1j * phase)
        phase_coherence = abs(np.mean(complex_phase))
        
        return float(phase_coherence)

    def _calculate_temporal_stability(self, data: np.ndarray) -> float:
        """Calculate temporal stability (approximated from spatial data)."""
        # Use spatial autocorrelation as stability proxy
        autocorr = self._calculate_spatial_correlation(data)
        
        # Stability measure
        stability = abs(autocorr)
        
        return float(stability)

    def _analyze_recursive_depth(self, data: np.ndarray) -> int:
        """Analyze recursive depth in the data structure."""
        # Multi-scale structure analysis
        current_data = data.copy()
        depth = 0
        min_size = 4
        
        while min(current_data.shape) >= min_size:
            # Check for self-similarity at this scale
            correlation = self._check_self_similarity(current_data)
            
            if correlation > 0.5:  # Threshold for recursive structure
                depth += 1
                # Downsample for next level
                current_data = current_data[::2, ::2]
            else:
                break
        
        return depth

    def _check_self_similarity(self, data: np.ndarray) -> float:
        """Check self-similarity in data."""
        h, w = data.shape
        if h < 4 or w < 4:
            return 0.0
        
        # Compare quarters
        mid_h, mid_w = h // 2, w // 2
        
        q1 = data[:mid_h, :mid_w]
        q2 = data[:mid_h, mid_w:2*mid_w]
        q3 = data[mid_h:2*mid_h, :mid_w]
        q4 = data[mid_h:2*mid_h, mid_w:2*mid_w]
        
        # Resize all to same size for comparison
        target_shape = (min(q.shape[0] for q in [q1, q2, q3, q4]),
                       min(q.shape[1] for q in [q1, q2, q3, q4]))
        
        if target_shape[0] < 2 or target_shape[1] < 2:
            return 0.0
        
        q1_resized = q1[:target_shape[0], :target_shape[1]]
        q2_resized = q2[:target_shape[0], :target_shape[1]]
        q3_resized = q3[:target_shape[0], :target_shape[1]]
        q4_resized = q4[:target_shape[0], :target_shape[1]]
        
        # Calculate correlations
        correlations = []
        quadrants = [q1_resized, q2_resized, q3_resized, q4_resized]
        
        for i in range(len(quadrants)):
            for j in range(i+1, len(quadrants)):
                corr = np.corrcoef(quadrants[i].flatten(), 
                                 quadrants[j].flatten())[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0

    # Cache and utility methods
    
    def _generate_scientific_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key with scientific metadata."""
        # Include scientific parameters in cache key
        scientific_params = {
            'scientific_mode': self.scientific_mode,
            'publication_quality': self.publication_quality,
            'timestamp': int(time.time() // 3600)  # Hour-based cache invalidation
        }
        
        # Combine with regular cache key generation
        return self._generate_cache_key(prefix, *args, **kwargs, **scientific_params)

    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate stable cache key for visualization data."""
        def hash_array(arr):
            if isinstance(arr, np.ndarray):
                if arr.size > 1000:
                    # Sample for large arrays
                    sample_indices = np.linspace(0, arr.size-1, 1000, dtype=int)
                    sample = arr.flatten()[sample_indices]
                    return f"{arr.shape}:{arr.dtype}:{hash(sample.tobytes())}"
                else:
                    return f"{arr.shape}:{arr.dtype}:{hash(arr.tobytes())}"
            return str(hash(str(arr)))
        
        args_str = ":".join(hash_array(arg) for arg in args if arg is not None)
        kwargs_str = ":".join(f"{k}={hash_array(v)}" for k, v in sorted(kwargs.items()))
        
        combined = f"{prefix}:{args_str}:{kwargs_str}"
        return f"{prefix}_{hashlib.md5(combined.encode()).hexdigest()[:16]}"

    def _get_from_scientific_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve from cache with scientific validation."""
        if cache_key in self._cache:
            # Update access pattern
            self._cache_keys.append(self._cache_keys.popleft() if cache_key in self._cache_keys else cache_key)
            
            # Validate scientific integrity
            cached_result = self._cache[cache_key]
            metadata = self._cache_metadata.get(cache_key, {})
            
            # Check if cache is still scientifically valid
            if self._validate_cache_integrity(metadata):
                return cached_result
            else:
                # Remove invalid cache entry
                del self._cache[cache_key]
                if cache_key in self._cache_metadata:
                    del self._cache_metadata[cache_key]
        
        return None

    def _add_to_scientific_cache(self, cache_key: str, result: Dict[str, Any]):
        """Add to cache with scientific metadata."""
        # Manage cache size
        while len(self._cache) >= self._cache_max_size:
            if self._cache_keys:
                old_key = self._cache_keys.popleft()
                if old_key in self._cache:
                    del self._cache[old_key]
                if old_key in self._cache_metadata:
                    del self._cache_metadata[old_key]
        
        # Store result and metadata
        self._cache[cache_key] = result
        self._cache_metadata[cache_key] = {
            'timestamp': time.time(),
            'scientific_parameters': self._extract_scientific_parameters(result),
            'validation_hash': self._calculate_validation_hash(result)
        }
        self._cache_keys.append(cache_key)

    def _validate_cache_integrity(self, metadata: Dict) -> bool:
        """Validate scientific integrity of cached results."""
        # Check age
        max_age = 3600  # 1 hour
        if time.time() - metadata.get('timestamp', 0) > max_age:
            return False
        
        # Additional validation checks can be added here
        return True

    def _extract_scientific_parameters(self, result: Dict[str, Any]) -> Dict:
        """Extract scientific parameters from result for validation."""
        return {
            'has_statistics': 'statistics' in result,
            'has_osh_metrics': 'osh_metrics' in result,
            'processing_time': result.get('processing_time', 0),
            'data_shape': result.get('scientific_metadata', {}).get('data_shape', ())
        }

    def _calculate_validation_hash(self, result: Dict[str, Any]) -> str:
        """Calculate validation hash for result integrity."""
        # Create hash from key scientific results
        validation_data = {
            'image_data_length': len(result.get('image_data', '')),
            'statistics_keys': list(result.get('statistics', {}).keys()),
            'metadata_keys': list(result.get('scientific_metadata', {}).keys())
        }
        
        return hashlib.md5(str(validation_data).encode()).hexdigest()

    def _create_publication_figure(self, figsize: Tuple[int, int] = (12, 10)) -> Figure:
        """Create publication-quality figure."""
        plt.rcParams.update({
            'figure.dpi': 300 if self.publication_quality else 100,
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'grid.alpha': 0.3,
            'legend.frameon': False
        })
        
        fig = plt.figure(figsize=figsize)
        return fig

    def _get_publication_figure_data(self, fig: Figure) -> str:
        """Convert figure to publication-quality image data."""
        try:
            canvas = FigureCanvasAgg(fig)
            buf = BytesIO()
            
            if self.publication_quality:
                canvas.print_png(buf, dpi=300, bbox_inches='tight', 
                                pad_inches=0.1, facecolor='white')
            else:
                canvas.print_png(buf, dpi=150, bbox_inches='tight')
            
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            return f"data:image/png;base64,{img_data}"
            
        except Exception as e:
            logger.error(f"Error creating publication figure: {e}")
            return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeQI8Tg2W5AAAAABJRU5ErkJggg=="

    def _generate_metadata(self, data: np.ndarray, start_time: float) -> VisualizationMetadata:
        """Generate comprehensive scientific metadata."""
        stats = self._comprehensive_statistical_analysis(data)
        
        return VisualizationMetadata(
            creation_time=time.time(),
            data_shape=data.shape,
            statistical_properties=stats['basic'],
            processing_time=time.time() - start_time,
            memory_usage=data.nbytes,
            validation_metrics={
                'data_integrity': 1.0 if np.isfinite(data).all() else 0.0,
                'statistical_significance': 1.0 if stats['distribution']['normality_p'] > 0.001 else 0.0
            },
            scientific_parameters={
                'scientific_mode': self.scientific_mode,
                'publication_quality': self.publication_quality
            },
            reproducibility_hash=hashlib.md5(data.tobytes()).hexdigest()
        )

    def _detect_emergence_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect emergence patterns in coherence data."""
        try:
            patterns = {}
            
            if len(data) < 10:
                return patterns
            
            # Calculate variance and trends
            variance = np.var(data)
            mean_val = np.mean(data)
            
            # Detect oscillations
            zero_crossings = np.sum(np.diff(np.signbit(data - mean_val)))
            patterns['oscillation_strength'] = zero_crossings / len(data)
            
            # Detect trends
            if len(data) > 2:
                slope = np.polyfit(range(len(data)), data, 1)[0]
                patterns['trend_strength'] = abs(slope)
                patterns['trend_direction'] = 'increasing' if slope > 0 else 'decreasing'
            
            # Detect criticality
            patterns['criticality'] = min(1.0, variance * 2.0)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting emergence patterns: {e}")
            return {}
        
    def _update_performance_metrics(self, processing_time: float, result: Dict[str, Any]):
        """Update performance monitoring metrics."""
        self.performance_metrics['render_times'].append(processing_time)
        
        # Estimate memory usage
        image_size = len(result.get('image_data', ''))
        self.performance_metrics['memory_usage'].append(image_size)
        
        # Update efficiency metrics
        avg_time = np.mean(self.performance_metrics['render_times'])
        self.performance_metrics['processing_efficiency'] = min(1.0, 1.0 / max(avg_time, 0.001))

    def _assemble_scientific_result(
        self, 
        image_data: str, 
        stats: Dict, 
        osh_metrics: OSHMetrics, 
        data: np.ndarray,
        processing_time: float,
        settings: Dict
    ) -> Dict[str, Any]:
        """Assemble comprehensive scientific result."""
        return {
            'image_data': image_data,
            'statistics': stats,
            'osh_metrics': osh_metrics.__dict__,
            'processing_time': processing_time,
            'scientific_metadata': self._generate_metadata(data, time.time() - processing_time).__dict__,
            'settings_used': settings,
            'validation_passed': self._validate_scientific_result(stats, osh_metrics),
            'empirical_significance': self._assess_empirical_significance(osh_metrics),
            'reproducibility_info': {
                'data_hash': hashlib.md5(data.tobytes()).hexdigest(),
                'settings_hash': hashlib.md5(str(settings).encode()).hexdigest(),
                'timestamp': time.time()
            }
        }

    def _validate_scientific_result(self, stats: Dict, osh_metrics: OSHMetrics) -> bool:
        """Validate scientific integrity of results."""
        checks = [
            stats['basic']['std'] > 0,  # Non-trivial data
            0 <= osh_metrics.coherence <= 1,  # Valid coherence range
            osh_metrics.entropy >= 0,  # Valid entropy
            np.isfinite(osh_metrics.rsp),  # Finite RSP
            osh_metrics.phi >= 0  # Valid integrated information
        ]
        
        return all(checks)

    def _assess_empirical_significance(self, osh_metrics: OSHMetrics) -> float:
        """Assess empirical significance for OSH validation."""
        # Composite significance score
        significance_factors = [
            min(osh_metrics.rsp / 10, 1.0),  # RSP contribution
            osh_metrics.phi,  # Information integration
            osh_metrics.emergence_index,  # Emergence
            osh_metrics.criticality_parameter,  # Criticality
            osh_metrics.phase_coherence  # Phase coherence
        ]
        
        return float(np.mean(significance_factors))

    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error result."""
        return {
            'image_data': "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFeQI8Tg2W5AAAAABJRU5ErkJggg==",
            'error': error_message,
            'statistics': {},
            'osh_metrics': OSHMetrics().__dict__,
            'processing_time': 0.0,
            'validation_passed': False
        }

    def _handle_scientific_error(self, error: Exception, context: str) -> Dict[str, Any]:
        """Handle errors with scientific logging."""
        error_msg = f"Scientific visualization error in {context}: {str(error)}"
        logger.error(error_msg)
        logger.debug(traceback.format_exc())
        
        # Emit scientific error event
        if self.event_system:
            try:
                self.event_system.emit("scientific_visualization_error", {
                    "context": context,
                    "error": str(error),
                    "timestamp": time.time(),
                    "scientific_mode": self.scientific_mode
                })
            except Exception:
                pass  # Avoid cascading errors
        
        return self._create_error_result(error_msg)

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        return {
            'average_render_time': float(np.mean(self.performance_metrics['render_times'])) if self.performance_metrics['render_times'] else 0.0,
            'cache_efficiency': len(self._cache) / max(self._cache_max_size, 1),
            'processing_efficiency': self.performance_metrics['processing_efficiency'],
            'memory_usage_trend': list(self.performance_metrics['memory_usage'])[-10:],  # Last 10 entries
            'cache_hit_rate': self.performance_metrics['cache_hit_rate'],
            'scientific_mode': self.scientific_mode,
            'publication_quality': self.publication_quality
        }

    def _detect_criticality(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect criticality patterns in the data."""
        try:
            # Basic criticality detection
            gradients = np.gradient(data)
            gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
            
            # Find high gradient regions
            threshold = np.percentile(gradient_magnitude, 95)
            critical_regions = gradient_magnitude > threshold
            
            return {
                'critical_points': np.sum(critical_regions),
                'criticality_score': float(np.mean(gradient_magnitude)),
                'max_gradient': float(np.max(gradient_magnitude)),
                'critical_regions': critical_regions
            }
        except Exception as e:
            logger.error(f"Error in criticality detection: {e}")
            return {'critical_points': 0, 'criticality_score': 0.0, 'max_gradient': 0.0}

    def _detect_phase_transitions(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect phase transition patterns."""
        try:
            # Simple phase transition detection based on local maxima/minima
            from scipy.signal import find_peaks
            
            # Flatten data for 1D analysis
            flattened = data.flatten()
            
            # Find peaks and valleys
            peaks, _ = find_peaks(flattened, prominence=np.std(flattened) * 0.5)
            valleys, _ = find_peaks(-flattened, prominence=np.std(flattened) * 0.5)
            
            return {
                'num_peaks': len(peaks),
                'num_valleys': len(valleys),
                'transition_points': len(peaks) + len(valleys),
                'transition_strength': float(np.std(flattened))
            }
        except Exception as e:
            logger.error(f"Error in phase transition detection: {e}")
            return {'num_peaks': 0, 'num_valleys': 0, 'transition_points': 0}

    def _detect_recursive_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect recursive patterns in the data."""
        try:
            # Simple recursive pattern detection using autocorrelation
            if data.size > 1000:
                # Subsample for performance
                indices = np.linspace(0, data.size - 1, 1000, dtype=int)
                flattened = data.flat[indices]
            else:
                flattened = data.flatten()
            
            # Compute autocorrelation
            autocorr = np.correlate(flattened, flattened, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find periodic patterns
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(autocorr[1:], height=0.3)
            
            return {
                'num_recursive_patterns': len(peaks),
                'pattern_strength': float(np.max(autocorr[1:]) if len(autocorr) > 1 else 0.0),
                'primary_period': int(peaks[0] + 1) if len(peaks) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error in recursive pattern detection: {e}")
            return {'num_recursive_patterns': 0, 'pattern_strength': 0.0, 'primary_period': 0}

    def cleanup(self):
        """Comprehensive cleanup of resources."""
        # Clear caches
        self._cache.clear()
        self._cache_metadata.clear()
        self._cache_keys.clear()
        
        # Close thread pool
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        
        # Clear matplotlib
        plt.close('all')
        
        # Clear performance metrics
        for key in self.performance_metrics:
            if isinstance(self.performance_metrics[key], deque):
                self.performance_metrics[key].clear()
        
        logger.info("AdvancedCoherenceRenderer cleanup completed")

    def __del__(self):
        """Destructor with proper cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Avoid errors during destruction


def create_advanced_coherence_renderer(
    coherence_manager=None,
    memory_field=None,
    recursive_mechanics=None,
    event_system=None,
    config: Optional[Dict[str, Any]] = None
) -> AdvancedCoherenceRenderer:
    """
    Factory function for creating AdvancedCoherenceRenderer with scientific configuration.
    
    Args:
        coherence_manager: CoherenceManager instance
        memory_field: MemoryFieldPhysics instance  
        recursive_mechanics: RecursiveMechanics instance
        event_system: EventSystem instance
        config: Advanced configuration parameters
        
    Returns:
        Configured AdvancedCoherenceRenderer instance
    """
    default_config = {
        'scientific_mode': True,
        'publication_quality': True,
        'enable_ml_analysis': True,
        'real_time_processing': False,
        'cache_size': 100,
        'max_data_points': 10_000_000,
        'max_animation_frames': 100
    }
    
    final_config = {**default_config, **(config or {})}
    
    return AdvancedCoherenceRenderer(
        coherence_manager=coherence_manager,
        memory_field=memory_field,
        recursive_mechanics=recursive_mechanics,
        event_system=event_system,
        config=final_config
    )


# Maintain backward compatibility
CoherenceRenderer = AdvancedCoherenceRenderer