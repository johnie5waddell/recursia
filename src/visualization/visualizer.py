"""
Cross-Platform Visualizer for Recursia Framework
================================================

Enterprise-grade visualization system with cross-platform support for Ubuntu, Windows, and macOS.
Provides both GUI and fallback display mechanisms for quantum state visualization.
"""

import logging
import os
import sys
import tempfile
import webbrowser
from typing import Dict, List, Optional, Tuple, Union, Any
import base64
import io
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import plotly.graph_objs as go
import plotly.offline as pyo

# Import enterprise quantum analyzer
try:
    from .quantum_state_analyzer import QuantumStateAnalyzer, AnalysisMode, create_quantum_analyzer
    ENTERPRISE_ANALYSIS_AVAILABLE = True
except ImportError:
    ENTERPRISE_ANALYSIS_AVAILABLE = False
    logger.warning("Enterprise quantum analysis not available")

# Configure matplotlib for cross-platform compatibility
if sys.platform == 'darwin':  # macOS
    matplotlib.use('MacOSX')
elif sys.platform == 'win32':  # Windows
    matplotlib.use('TkAgg')
else:  # Linux/Unix
    # Try to use TkAgg first, fall back to Agg if display not available
    try:
        matplotlib.use('TkAgg')
        import tkinter
    except:
        matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Enterprise-grade cross-platform visualizer for quantum states and operations.
    
    Supports multiple output modes:
    - 'web': Opens visualizations in web browser (cross-platform)
    - 'file': Saves to file system
    - 'display': Shows GUI popup (when available)
    - 'auto': Automatically detects best available mode
    """
    
    def __init__(self, output_mode='auto', save_path=None):
        """
        Initialize the visualizer with specified output mode.
        
        Args:
            output_mode: Output mode ('web', 'file', 'display', 'auto', 'gui', 'html', 'console')
            save_path: Directory to save files (for 'file' mode)
        """
        self.output_mode = output_mode
        self.save_path = save_path or tempfile.gettempdir()
        self.display_available = self._check_display()
        
        # Set appropriate backend based on display availability and platform
        if self.output_mode == 'auto':
            # Force web mode for better cross-platform compatibility
            if sys.platform == 'win32' or 'microsoft' in sys.platform.lower() or 'WSL' in os.environ.get('WSL_DISTRO_NAME', ''):
                self.output_mode = 'html'  # Use web browser for Windows/WSL
            elif self.display_available:
                self.output_mode = 'html'  # Use web browser for better reliability
            else:
                self.output_mode = 'file'  # Fallback to file mode
                
        # Set matplotlib backend for headless environments
        if not self.display_available or self.output_mode in ['file', 'html']:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        
        logger.info(f"Visualizer initialized with mode: {self.output_mode}, display available: {self.display_available}")
    
    def _check_display(self):
        """Check if a display is available for GUI output."""
        if sys.platform == 'win32':
            return True  # Windows always has display
        elif sys.platform == 'darwin':
            return True  # macOS always has display
        else:
            # Linux - check for X11 or Wayland
            display = os.environ.get('DISPLAY')
            wayland = os.environ.get('WAYLAND_DISPLAY')
            return bool(display or wayland)
    
    def plot_probability_distribution(self, state_vector: np.ndarray, title: str = "Probability Distribution") -> Any:
        """
        Plot probability distribution of a quantum state.
        
        Args:
            state_vector: Complex state vector
            title: Plot title
            
        Returns:
            Figure object or HTML string depending on mode
        """
        # Calculate probabilities
        probabilities = np.abs(state_vector) ** 2
        n_qubits = int(np.log2(len(probabilities)))
        
        # Create basis state labels
        basis_states = [format(i, f'0{n_qubits}b') for i in range(len(probabilities))]
        
        if self.output_mode in ['gui', 'file']:
            # Matplotlib plot
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(range(len(probabilities)), probabilities, color='steelblue', alpha=0.8)
            
            # Highlight significant probabilities
            max_prob_idx = np.argmax(probabilities)
            bars[max_prob_idx].set_color('darkred')
            
            ax.set_xlabel('Basis State', fontsize=12)
            ax.set_ylabel('Probability', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(basis_states)))
            ax.set_xticklabels(basis_states, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3)
            
            # Add probability values on bars
            for i, (bar, prob) in enumerate(zip(bars, probabilities)):
                if prob > 0.01:  # Only show if probability > 1%
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{prob:.3f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            return fig
            
        else:  # HTML/Plotly output
            # Create interactive Plotly plot
            data = [go.Bar(
                x=basis_states,
                y=probabilities,
                text=[f'{p:.3f}' for p in probabilities],
                textposition='auto',
                marker=dict(
                    color=probabilities,
                    colorscale='Blues',
                    showscale=True,
                    colorbar=dict(title="Probability")
                )
            )]
            
            layout = go.Layout(
                title=dict(text=title, font=dict(size=18)),
                xaxis=dict(title='Basis State', tickangle=-45),
                yaxis=dict(title='Probability', range=[0, 1.1]),
                showlegend=False,
                hovermode='closest'
            )
            
            fig = go.Figure(data=data, layout=layout)
            return fig
    
    def plot_entanglement_network(self, entanglement_data: Dict[Tuple[str, str], float], 
                                 state_names: List[str]) -> Any:
        """
        Plot entanglement network between quantum states.
        
        Args:
            entanglement_data: Dictionary mapping state pairs to entanglement strength
            state_names: List of all state names
            
        Returns:
            Figure object or HTML string
        """
        if self.output_mode in ['gui', 'file']:
            # Matplotlib network plot
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create graph
            G = nx.Graph()
            G.add_nodes_from(state_names)
            
            for (state1, state2), strength in entanglement_data.items():
                G.add_edge(state1, state2, weight=strength)
            
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=2000, alpha=0.8, ax=ax)
            
            # Draw edges with varying thickness based on entanglement strength
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[w*5 for w in weights], 
                                 alpha=0.6, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
            
            # Add edge labels for entanglement strength
            edge_labels = {(u, v): f'{G[u][v]["weight"]:.2f}' 
                          for u, v in G.edges() if G[u][v]['weight'] > 0.1}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
            
            ax.set_title('Quantum Entanglement Network', fontsize=16, fontweight='bold')
            ax.axis('off')
            plt.tight_layout()
            return fig
            
        else:  # HTML/Plotly output
            # Create interactive Plotly network
            G = nx.Graph()
            G.add_nodes_from(state_names)
            
            for (state1, state2), strength in entanglement_data.items():
                G.add_edge(state1, state2, weight=strength)
            
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Create edge traces
            edge_traces = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                weight = G[edge[0]][edge[1]]['weight']
                
                edge_trace = go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=weight*10, color=f'rgba(125,125,125,{weight})'),
                    hoverinfo='text',
                    text=f'{edge[0]} - {edge[1]}: {weight:.3f}'
                )
                edge_traces.append(edge_trace)
            
            # Create node trace
            node_trace = go.Scatter(
                x=[pos[node][0] for node in G.nodes()],
                y=[pos[node][1] for node in G.nodes()],
                mode='markers+text',
                text=list(G.nodes()),
                textposition="top center",
                marker=dict(
                    size=30,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                ),
                hoverinfo='text'
            )
            
            layout = go.Layout(
                title='Quantum Entanglement Network',
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            fig = go.Figure(data=edge_traces + [node_trace], layout=layout)
            return fig
    
    def show_plot(self, fig: Any, window_title: str = "Recursia Visualization"):
        """
        Display the plot using the appropriate method for the platform.
        
        Args:
            fig: Figure object (matplotlib or plotly)
            window_title: Title for the window/tab
        """
        try:
            if self.output_mode == 'gui' and isinstance(fig, plt.Figure):
                # Show matplotlib figure in GUI window
                fig.canvas.manager.set_window_title(window_title)
                plt.show(block=False)  # Non-blocking show
                plt.pause(0.1)  # Allow GUI to update
                
            elif self.output_mode == 'file' and isinstance(fig, plt.Figure):
                # Save matplotlib figure to file
                filename = os.path.join(self.save_path, f"{window_title.replace(' ', '_')}.png")
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
                logger.info(f"Saved visualization to: {filename}")
                
                # Also save to Windows desktop if WSL environment detected
                try:
                    import glob
                    windows_desktops = glob.glob("/mnt/c/Users/*/Desktop")
                    if windows_desktops:
                        windows_viz_dir = f"{windows_desktops[0]}/Recursia_Visualizations"
                        os.makedirs(windows_viz_dir, exist_ok=True)
                        windows_filename = os.path.join(windows_viz_dir, f"{window_title.replace(' ', '_')}.png")
                        fig.savefig(windows_filename, dpi=300, bbox_inches='tight', facecolor='black', edgecolor='none')
                        logger.info(f"Also saved to Windows: {windows_filename}")
                except Exception as e:
                    logger.debug(f"Could not save to Windows desktop: {e}")
                
                plt.close(fig)
                
            elif self.output_mode in ['html', 'auto'] or isinstance(fig, go.Figure):
                # Save as HTML and open in browser
                filename = os.path.join(self.save_path, f"{window_title.replace(' ', '_')}.html")
                
                if isinstance(fig, go.Figure):
                    # Plotly figure
                    pyo.plot(fig, filename=filename, auto_open=True)
                else:
                    # Convert matplotlib to HTML
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                    
                    # Create simple HTML content
                    html_parts = [
                        "<!DOCTYPE html>",
                        "<html>",
                        "<head>",
                        f"<title>{window_title}</title>",
                        "<style>",
                        "body { margin: 20px; background-color: lightgray; }",
                        ".container { text-align: center; }",
                        "img { max-width: 100%; height: auto; }",
                        "h1 { font-family: Arial, sans-serif; }",
                        "</style>",
                        "</head>",
                        "<body>",
                        '<div class="container">',
                        f"<h1>{window_title}</h1>",
                        f'<img src="data:image/png;base64,{img_base64}" alt="{window_title}">',
                        "</div>",
                        "</body>",
                        "</html>"
                    ]
                    html_content = "\n".join(html_parts)
                    
                    with open(filename, 'w') as f:
                        f.write(html_content)
                    
                    # Open in browser
                    webbrowser.open(f'file://{os.path.abspath(filename)}')
                
                logger.info(f"Opened visualization in browser: {filename}")
                
            elif self.output_mode == 'console':
                # Fallback console output
                logger.info(f"Console mode visualization: {window_title}")
                if hasattr(fig, 'to_string'):
                    print(fig.to_string())
                else:
                    print(f"[Visualization: {window_title}]")
                    
        except Exception as e:
            logger.error(f"Failed to display visualization: {str(e)}")
            # Fallback to console output
            print(f"\n{'='*60}")
            print(f"Visualization: {window_title}")
            print(f"{'='*60}")
            print(f"Error displaying plot: {str(e)}")
            print(f"Plot type: {type(fig).__name__}")
            print(f"{'='*60}\n")
    
    def create_enterprise_analysis(self, state_vector: np.ndarray, state_name: str = "QuantumState") -> Dict[str, Any]:
        """
        Create enterprise-grade quantum state analysis with publication-quality visualization.
        
        Args:
            state_vector: Complex state vector to analyze
            state_name: Name of the quantum state
            
        Returns:
            Dictionary containing analysis results and visualization data
        """
        if not ENTERPRISE_ANALYSIS_AVAILABLE:
            logger.warning("Enterprise analysis not available, falling back to basic analysis")
            return self._basic_state_analysis(state_vector, state_name)
        
        try:
            # Create enterprise analyzer
            analyzer = create_quantum_analyzer(AnalysisMode.PUBLICATION)
            
            # Perform comprehensive analysis
            logger.info(f"Starting enterprise analysis of state '{state_name}'")
            metrics = analyzer.analyze_state(state_vector, state_name)
            
            # Create publication-quality visualization
            fig = analyzer.create_publication_plot(state_vector, metrics, f"Enterprise Analysis: {state_name}")
            
            # Export comprehensive report
            analysis_report = analyzer.export_analysis_report(state_name, metrics, state_vector)
            
            # Save and display visualization
            html_filename = os.path.join(self.save_path, f"enterprise_analysis_{state_name.replace(' ', '_')}.html")
            pyo.plot(fig, filename=html_filename, auto_open=(self.output_mode == 'html'))
            
            logger.info(f"Enterprise analysis completed for '{state_name}' - saved to {html_filename}")
            
            return {
                'success': True,
                'analysis_report': analysis_report,
                'visualization_file': html_filename,
                'metrics': {
                    'purity': metrics.purity,
                    'entropy': metrics.von_neumann_entropy,
                    'concurrence': metrics.concurrence,
                    'coherence': metrics.coherence_l1
                }
            }
            
        except Exception as e:
            logger.error(f"Enterprise analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_analysis': self._basic_state_analysis(state_vector, state_name)
            }
    
    def _basic_state_analysis(self, state_vector: np.ndarray, state_name: str) -> Dict[str, Any]:
        """Fallback basic quantum state analysis."""
        probabilities = np.abs(state_vector) ** 2
        phases = np.angle(state_vector)
        
        # Basic purity calculation
        rho = np.outer(state_vector, np.conj(state_vector))
        purity = float(np.real(np.trace(rho @ rho)))
        
        # Basic entropy
        eigenvals = np.linalg.eigvals(rho)
        eigenvals = eigenvals[eigenvals > 1e-12]
        entropy = float(-np.sum(eigenvals * np.log2(eigenvals))) if len(eigenvals) > 0 else 0.0
        
        return {
            'state_name': state_name,
            'probabilities': probabilities.tolist(),
            'phases': phases.tolist(),
            'purity': purity,
            'entropy': entropy,
            'analysis_type': 'basic'
        }
    
    def close_all(self):
        # Close all matplotlib windows
        if self.output_mode == 'gui':
            plt.close('all')
