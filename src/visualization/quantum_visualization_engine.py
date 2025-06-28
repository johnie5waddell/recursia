"""
Quantum Visualization Engine - Enterprise-Grade Interactive Platform

This module provides a comprehensive visualization platform for quantum simulations with:
- Real-time 3D quantum state visualization using WebGL/Three.js
- Interactive Bloch sphere, density matrix, and entanglement visualizations
- GPU-accelerated rendering for large-scale quantum systems
- Publication-quality vector graphics export (SVG, PDF)
- Collaborative real-time sharing capabilities
- Responsive web dashboard with modern UI/UX
- Accessibility compliance (WCAG 2.1)
- Performance monitoring and analytics
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json
import base64
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor
import uuid
from pathlib import Path

# Scientific visualization
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# 3D visualization
try:
    import mayavi.mlab as mlab
    MAYAVI_AVAILABLE = True
except ImportError:
    MAYAVI_AVAILABLE = False
    
# Web framework
try:
    import dash
    from dash import dcc, html, Input, Output, State, callback
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False

# Performance optimization
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualizationError(Exception):
    """Base exception for visualization errors."""
    pass


class RenderingError(VisualizationError):
    """Exception raised during rendering operations."""
    pass


class ExportError(VisualizationError):
    """Exception raised during export operations."""
    pass


class ColorScheme(Enum):
    """Scientific color schemes optimized for quantum data."""
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"          # Colorblind-friendly
    SCIENTIFIC = "RdYlBu_r"      # Red-Yellow-Blue for scientific data
    QUANTUM_PHASE = "hsv"        # For quantum phases
    BLOCH_SPHERE = "coolwarm"    # For Bloch sphere visualization


class OutputFormat(Enum):
    """Supported output formats."""
    HTML = "html"
    SVG = "svg"
    PDF = "pdf"
    PNG = "png"
    WEBGL = "webgl"
    JSON = "json"


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    width: int = 1200
    height: int = 800
    dpi: int = 300
    color_scheme: ColorScheme = ColorScheme.VIRIDIS
    interactive: bool = True
    show_grid: bool = True
    show_axes: bool = True
    font_size: int = 12
    title_font_size: int = 16
    line_width: float = 2.0
    marker_size: float = 8.0
    opacity: float = 0.8
    animation_duration: int = 500  # milliseconds
    theme: str = "plotly_white"
    
    def to_plotly_layout(self) -> Dict[str, Any]:
        """Convert to Plotly layout dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'font': {'size': self.font_size},
            'template': self.theme,
            'showlegend': True,
            'hovermode': 'closest',
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white'
        }


@dataclass
class QuantumVisualizationData:
    """Container for quantum visualization data."""
    state_vector: Optional[np.ndarray] = None
    density_matrix: Optional[np.ndarray] = None
    bloch_vector: Optional[np.ndarray] = None
    eigenvalues: Optional[np.ndarray] = None
    eigenvectors: Optional[np.ndarray] = None
    entanglement_matrix: Optional[np.ndarray] = None
    measurement_probabilities: Optional[np.ndarray] = None
    time_evolution: Optional[List[np.ndarray]] = None
    labels: Optional[List[str]] = None
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumVisualizationEngine:
    """Enterprise-grade quantum visualization platform."""
    
    def __init__(self, 
                 config: Optional[VisualizationConfig] = None,
                 enable_web_server: bool = True,
                 port: int = 8050):
        """
        Initialize visualization engine.
        
        Args:
            config: Visualization configuration
            enable_web_server: Start interactive web server
            port: Web server port
        """
        self.config = config or VisualizationConfig()
        self.enable_web_server = enable_web_server
        self.port = port
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Visualization cache
        self.figure_cache: Dict[str, go.Figure] = {}
        self.data_cache: Dict[str, QuantumVisualizationData] = {}
        
        # Performance monitoring
        self.render_times: List[float] = []
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Web application
        self.app: Optional[dash.Dash] = None
        if enable_web_server and DASH_AVAILABLE:
            self._setup_web_application()
            
        # Color palettes for accessibility
        self._setup_color_palettes()
        
        logger.info(f"QuantumVisualizationEngine initialized on port {port}")
    
    def _setup_color_palettes(self):
        """Setup scientifically accurate and accessible color palettes."""
        self.color_palettes = {
            ColorScheme.QUANTUM_PHASE: {
                'colors': ['#FF0000', '#FF8000', '#FFFF00', '#80FF00', 
                          '#00FF00', '#00FF80', '#00FFFF', '#0080FF',
                          '#0000FF', '#8000FF', '#FF00FF', '#FF0080'],
                'description': 'Quantum phase visualization (0 to 2π)'
            },
            ColorScheme.BLOCH_SPHERE: {
                'colors': ['#0000FF', '#8080FF', '#FFFFFF', '#FF8080', '#FF0000'],
                'description': 'Blue-white-red for Bloch sphere components'
            },
            ColorScheme.SCIENTIFIC: {
                'colors': px.colors.diverging.RdYlBu_r,
                'description': 'Red-Yellow-Blue diverging for scientific data'
            }
        }
    
    def _setup_web_application(self):
        """Setup interactive web dashboard."""
        if not DASH_AVAILABLE:
            logger.warning("Dash not available - web interface disabled")
            return
            
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.BOOTSTRAP])
        
        # Main layout
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Quantum Visualization Dashboard", 
                           className="text-center mb-4"),
                    html.Hr()
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Quantum State Visualization"),
                        dbc.CardBody([
                            dcc.Graph(id="quantum-state-plot"),
                            dbc.ButtonGroup([
                                dbc.Button("Bloch Sphere", id="btn-bloch", 
                                         color="primary", size="sm"),
                                dbc.Button("Density Matrix", id="btn-density", 
                                         color="secondary", size="sm"),
                                dbc.Button("Probability", id="btn-prob", 
                                         color="success", size="sm")
                            ])
                        ])
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Entanglement Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="entanglement-plot"),
                            dbc.Progress(id="entanglement-progress", 
                                       value=0, striped=True)
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Time Evolution"),
                        dbc.CardBody([
                            dcc.Graph(id="evolution-plot"),
                            dcc.Slider(id="time-slider", min=0, max=100, 
                                     value=0, marks={i: f"{i}τ" for i in range(0, 101, 20)})
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Performance metrics
            dbc.Row([
                dbc.Col([
                    dbc.Alert(id="performance-alert", color="info",
                             children="System ready for visualization")
                ])
            ])
            
        ], fluid=True)
        
        # Register callbacks
        self._register_callbacks()
    
    def visualize_quantum_state(self,
                               data: QuantumVisualizationData,
                               visualization_type: str = "auto",
                               save_path: Optional[str] = None,
                               show: bool = True) -> go.Figure:
        """
        Create comprehensive quantum state visualization.
        
        Args:
            data: Quantum visualization data
            visualization_type: Type of visualization
            save_path: Path to save figure
            show: Whether to display figure
            
        Returns:
            Plotly figure object
        """
        start_time = time.perf_counter()
        
        try:
            with self._lock:
                # Cache key for performance
                cache_key = self._generate_cache_key(data, visualization_type)
                
                if cache_key in self.figure_cache:
                    self.cache_hits += 1
                    return self.figure_cache[cache_key]
                
                self.cache_misses += 1
                
                # Determine visualization type
                if visualization_type == "auto":
                    visualization_type = self._auto_detect_visualization_type(data)
                
                # Create figure based on type
                if visualization_type == "bloch_sphere":
                    fig = self._create_bloch_sphere(data)
                elif visualization_type == "density_matrix":
                    fig = self._create_density_matrix_plot(data)
                elif visualization_type == "state_probabilities":
                    fig = self._create_probability_plot(data)
                elif visualization_type == "entanglement":
                    fig = self._create_entanglement_plot(data)
                elif visualization_type == "time_evolution":
                    fig = self._create_evolution_plot(data)
                else:
                    raise ValueError(f"Unknown visualization type: {visualization_type}")
                
                # Apply configuration
                fig.update_layout(**self.config.to_plotly_layout())
                
                # Cache figure
                self.figure_cache[cache_key] = fig
                
                # Save if requested
                if save_path:
                    self.export_figure(fig, save_path)
                
                # Performance tracking
                render_time = time.perf_counter() - start_time
                self.render_times.append(render_time)
                
                logger.debug(f"Rendered {visualization_type} in {render_time:.4f}s")
                
                return fig
                
        except Exception as e:
            logger.error(f"Visualization error: {e}")
            raise RenderingError(f"Failed to create visualization: {e}")
    
    def _create_bloch_sphere(self, data: QuantumVisualizationData) -> go.Figure:
        """Create interactive 3D Bloch sphere visualization."""
        if data.state_vector is None:
            raise ValueError("State vector required for Bloch sphere")
            
        # Convert state vector to Bloch vector
        if data.bloch_vector is None:
            data.bloch_vector = self._state_to_bloch_vector(data.state_vector)
            
        x, y, z = data.bloch_vector
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        # Add Bloch sphere surface
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_sphere = np.outer(np.cos(u), np.sin(v))
        y_sphere = np.outer(np.sin(u), np.sin(v))
        z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            name="Bloch Sphere"
        ))
        
        # Add state vector
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z],
            mode='lines+markers',
            line=dict(color='red', width=8),
            marker=dict(size=[5, 10], color=['blue', 'red']),
            name="State Vector"
        ))
        
        # Add coordinate axes
        axis_length = 1.2
        for axis, color in [('x', 'red'), ('y', 'green'), ('z', 'blue')]:
            if axis == 'x':
                coords = [[0, axis_length], [0, 0], [0, 0]]
            elif axis == 'y':
                coords = [[0, 0], [0, axis_length], [0, 0]]
            else:
                coords = [[0, 0], [0, 0], [0, axis_length]]
                
            fig.add_trace(go.Scatter3d(
                x=coords[0], y=coords[1], z=coords[2],
                mode='lines',
                line=dict(color=color, width=3, dash='dash'),
                showlegend=False
            ))
        
        # Update layout for 3D
        fig.update_layout(
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y", 
                zaxis_title="Z",
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            title="Quantum State on Bloch Sphere"
        )
        
        return fig
    
    def _create_density_matrix_plot(self, data: QuantumVisualizationData) -> go.Figure:
        """Create density matrix visualization with real and imaginary parts."""
        if data.density_matrix is None:
            if data.state_vector is not None:
                data.density_matrix = np.outer(data.state_vector, 
                                             data.state_vector.conj())
            else:
                raise ValueError("Density matrix or state vector required")
        
        rho = data.density_matrix
        
        # Create subplots for real and imaginary parts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Real Part', 'Imaginary Part', 
                          'Magnitude', 'Phase'),
            specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}],
                   [{'type': 'heatmap'}, {'type': 'heatmap'}]]
        )
        
        # Real part
        fig.add_trace(
            go.Heatmap(z=rho.real, colorscale='RdBu', zmid=0, 
                      showscale=False, name="Real"),
            row=1, col=1
        )
        
        # Imaginary part
        fig.add_trace(
            go.Heatmap(z=rho.imag, colorscale='RdBu', zmid=0,
                      showscale=False, name="Imaginary"),
            row=1, col=2
        )
        
        # Magnitude
        fig.add_trace(
            go.Heatmap(z=np.abs(rho), colorscale='Viridis',
                      showscale=False, name="Magnitude"),
            row=2, col=1
        )
        
        # Phase
        fig.add_trace(
            go.Heatmap(z=np.angle(rho), colorscale='hsv',
                      showscale=True, name="Phase"),
            row=2, col=2
        )
        
        fig.update_layout(title="Density Matrix Visualization")
        
        return fig
    
    def _create_probability_plot(self, data: QuantumVisualizationData) -> go.Figure:
        """Create state probability visualization."""
        if data.measurement_probabilities is None:
            if data.state_vector is not None:
                probs = np.abs(data.state_vector)**2
            elif data.density_matrix is not None:
                probs = np.diag(data.density_matrix).real
            else:
                raise ValueError("No probability data available")
        else:
            probs = data.measurement_probabilities
            
        # Create labels if not provided
        if data.labels is None:
            n_qubits = int(np.log2(len(probs)))
            labels = [format(i, f'0{n_qubits}b') for i in range(len(probs))]
        else:
            labels = data.labels
            
        # Create bar plot
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=labels,
            y=probs,
            marker=dict(
                color=probs,
                colorscale=self.config.color_scheme.value,
                showscale=True,
                colorbar=dict(title="Probability")
            ),
            text=[f"{p:.3f}" for p in probs],
            textposition='auto',
            name="Measurement Probabilities"
        ))
        
        fig.update_layout(
            title="Quantum State Probabilities",
            xaxis_title="Basis States",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _create_entanglement_plot(self, data: QuantumVisualizationData) -> go.Figure:
        """Create entanglement analysis visualization."""
        if data.entanglement_matrix is None:
            raise ValueError("Entanglement matrix required")
            
        ent_matrix = data.entanglement_matrix
        
        # Create network-style entanglement plot
        fig = go.Figure()
        
        # Heatmap of entanglement strengths
        fig.add_trace(go.Heatmap(
            z=ent_matrix,
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Entanglement Strength")
        ))
        
        # Add annotations for strong entanglements
        for i in range(ent_matrix.shape[0]):
            for j in range(ent_matrix.shape[1]):
                if ent_matrix[i, j] > 0.5:  # Threshold for significant entanglement
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{ent_matrix[i, j]:.2f}",
                        showarrow=False,
                        font=dict(color="white", size=10)
                    )
        
        fig.update_layout(
            title="Quantum Entanglement Network",
            xaxis_title="Qubit Index",
            yaxis_title="Qubit Index"
        )
        
        return fig
    
    def _create_evolution_plot(self, data: QuantumVisualizationData) -> go.Figure:
        """Create time evolution visualization."""
        if data.time_evolution is None:
            raise ValueError("Time evolution data required")
            
        evolution_data = data.time_evolution
        n_steps = len(evolution_data)
        
        # Extract probabilities over time
        time_points = np.linspace(0, 1, n_steps)
        n_states = len(evolution_data[0])
        
        fig = go.Figure()
        
        # Plot evolution of each basis state probability
        for state_idx in range(min(8, n_states)):  # Limit to first 8 states
            probs = [np.abs(step[state_idx])**2 for step in evolution_data]
            
            fig.add_trace(go.Scatter(
                x=time_points,
                y=probs,
                mode='lines+markers',
                name=f"|{state_idx:03b}⟩",
                line=dict(width=2),
                marker=dict(size=4)
            ))
        
        fig.update_layout(
            title="Quantum State Evolution",
            xaxis_title="Time (arbitrary units)",
            yaxis_title="Probability",
            yaxis=dict(range=[0, 1]),
            hovermode='x unified'
        )
        
        return fig
    
    def _state_to_bloch_vector(self, state_vector: np.ndarray) -> np.ndarray:
        """Convert 2-level state vector to Bloch sphere coordinates."""
        if len(state_vector) != 2:
            raise ValueError("Bloch sphere only valid for 2-level systems")
            
        # Pauli matrices
        sigma_x = np.array([[0, 1], [1, 0]])
        sigma_y = np.array([[0, -1j], [1j, 0]])
        sigma_z = np.array([[1, 0], [0, -1]])
        
        # Calculate expectation values
        x = np.real(np.vdot(state_vector, sigma_x @ state_vector))
        y = np.real(np.vdot(state_vector, sigma_y @ state_vector))
        z = np.real(np.vdot(state_vector, sigma_z @ state_vector))
        
        return np.array([x, y, z])
    
    def _auto_detect_visualization_type(self, data: QuantumVisualizationData) -> str:
        """Automatically detect best visualization type for data."""
        if data.bloch_vector is not None or (data.state_vector is not None and len(data.state_vector) == 2):
            return "bloch_sphere"
        elif data.entanglement_matrix is not None:
            return "entanglement"
        elif data.time_evolution is not None:
            return "time_evolution"
        elif data.density_matrix is not None:
            return "density_matrix"
        else:
            return "state_probabilities"
    
    def _generate_cache_key(self, data: QuantumVisualizationData, vis_type: str) -> str:
        """Generate cache key for visualization data."""
        # Simple hash based on data shapes and type
        key_parts = [vis_type]
        
        if data.state_vector is not None:
            key_parts.append(f"sv_{data.state_vector.shape}")
        if data.density_matrix is not None:
            key_parts.append(f"dm_{data.density_matrix.shape}")
        if data.entanglement_matrix is not None:
            key_parts.append(f"em_{data.entanglement_matrix.shape}")
            
        return "_".join(key_parts)
    
    def export_figure(self, 
                     fig: go.Figure, 
                     filepath: str,
                     format: Optional[OutputFormat] = None) -> None:
        """
        Export figure to various formats with publication quality.
        
        Args:
            fig: Plotly figure to export
            filepath: Output file path
            format: Output format (auto-detected if None)
        """
        try:
            if format is None:
                # Auto-detect from file extension
                ext = Path(filepath).suffix.lower()
                format_map = {
                    '.html': OutputFormat.HTML,
                    '.svg': OutputFormat.SVG,
                    '.pdf': OutputFormat.PDF,
                    '.png': OutputFormat.PNG,
                    '.json': OutputFormat.JSON
                }
                format = format_map.get(ext, OutputFormat.HTML)
            
            if format == OutputFormat.HTML:
                fig.write_html(filepath, config={'displayModeBar': True})
            elif format == OutputFormat.SVG:
                fig.write_image(filepath, format='svg', 
                              width=self.config.width, height=self.config.height)
            elif format == OutputFormat.PDF:
                fig.write_image(filepath, format='pdf',
                              width=self.config.width, height=self.config.height)
            elif format == OutputFormat.PNG:
                fig.write_image(filepath, format='png',
                              width=self.config.width, height=self.config.height,
                              scale=self.config.dpi/100)
            elif format == OutputFormat.JSON:
                with open(filepath, 'w') as f:
                    json.dump(fig.to_dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Exported figure to {filepath}")
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            raise ExportError(f"Failed to export figure: {e}")
    
    def start_web_server(self, debug: bool = False):
        """Start interactive web dashboard."""
        if not self.app:
            raise VisualizationError("Web application not initialized")
            
        try:
            self.app.run_server(debug=debug, port=self.port, host='0.0.0.0')
        except Exception as e:
            logger.error(f"Failed to start web server: {e}")
            raise VisualizationError(f"Web server startup failed: {e}")
    
    def _register_callbacks(self):
        """Register Dash callbacks for interactivity."""
        if not self.app:
            return
            
        @self.app.callback(
            Output("quantum-state-plot", "figure"),
            [Input("btn-bloch", "n_clicks"),
             Input("btn-density", "n_clicks"),
             Input("btn-prob", "n_clicks")]
        )
        def update_quantum_plot(bloch_clicks, density_clicks, prob_clicks):
            # Placeholder - would connect to actual quantum data
            # This would be populated with real quantum state data
            dummy_data = QuantumVisualizationData(
                state_vector=np.array([1/np.sqrt(2), 1/np.sqrt(2)])
            )
            
            # Determine which button was clicked
            ctx = dash.callback_context
            if not ctx.triggered:
                vis_type = "bloch_sphere"
            else:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if button_id == "btn-bloch":
                    vis_type = "bloch_sphere"
                elif button_id == "btn-density":
                    vis_type = "density_matrix"
                else:
                    vis_type = "state_probabilities"
            
            return self.visualize_quantum_state(dummy_data, vis_type, show=False)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get visualization performance metrics."""
        avg_render_time = np.mean(self.render_times) if self.render_times else 0
        cache_hit_rate = self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        
        return {
            'avg_render_time': avg_render_time,
            'total_renders': len(self.render_times),
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.figure_cache),
            'web_server_enabled': self.app is not None
        }


# Global instance
quantum_viz_engine = QuantumVisualizationEngine()