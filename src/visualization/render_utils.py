"""
render_utils.py - Recursia Advanced Rendering Utilities

Central orchestration and composition layer for building and rendering Recursia's
most sophisticated dashboards with OSH-aligned visualization capabilities.

Features:
- Modular high-level render functions (per-panel)
- System diagnostics and health visualizations  
- Simulation controls and performance monitors
- Emergent phenomena and phase space detection
- Recursive system hierarchy rendering
- Full layout orchestration via render_dashboard
"""

import base64
import io
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Import correct OSH formula utilities
from src.visualization.osh_formula_utils import calculate_rsp_simple
from matplotlib.colors import LinearSegmentedColormap, Normalize
try:
    import seaborn as sns
except ImportError:
    sns = None
try:
    from scipy import stats
    from scipy.spatial import ConvexHull  
    from scipy.signal import find_peaks
except ImportError:
    stats = None
    ConvexHull = None
    find_peaks = None

try:
    from sklearn.cluster import DBSCAN
except ImportError:
    DBSCAN = None

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# Configure logging
logger = logging.getLogger(__name__)

# OSH-aligned color schemes
OSH_COLORS = {
    'coherence': '#00ff9f',      # Bright green - representing order
    'entropy': '#ff4757',        # Red - representing disorder
    'strain': '#ffa726',         # Orange - representing tension
    'rsp': '#3742fa',            # Blue - representing potential
    'emergence': '#2ed573',      # Green - representing growth
    'consciousness': '#5f27cd',   # Purple - representing awareness
    'memory': '#00d2d3',         # Cyan - representing storage
    'recursion': '#ff6b6b',      # Coral - representing depth
    'background': '#0a0a0f',     # Deep space blue
    'foreground': '#e8e8e8',     # Light gray
    'accent': '#64ffda',         # Teal accent
    'warning': '#ffb74d',        # Amber warning
    'critical': '#f44336',       # Red critical
    'success': '#4caf50'         # Green success
}

def render_dashboard(
    quantum_renderer=None,
    coherence_renderer=None, 
    field_panel=None,
    observer_panel=None,
    simulation_panel=None,
    memory_field=None,
    recursive_mechanics=None,
    coherence_manager=None,
    entanglement_manager=None,
    event_system=None,
    current_metrics=None,
    metrics_history=None,
    phenomena_detector=None,
    config=None,
    layout_manager=None,
    performance_tracker=None
) -> Dict[str, Any]:
    """
    Renders the ultimate Recursia dashboard with 9-panel composite visualization.
    
    Returns comprehensive dashboard with real-time OSH metrics, quantum states,
    field dynamics, observer networks, memory analysis, and emergent phenomena.
    """
    start_time = time.time()
    
    try:
        # Initialize configuration
        if config is None:
            config = _get_default_dashboard_config()
            
        # Set up the figure with OSH theme
        fig = _setup_dashboard_figure(config)
        
        # Create layout manager
        if layout_manager is None:
            layout_manager = _create_advanced_layout_manager(config)
            
        # Initialize panel results container
        panel_results = {}
        render_errors = []
        
        # Render each panel with error isolation
        panels = [
            ('quantum', _render_quantum_panel_wrapper),
            ('field', _render_field_panel_wrapper),
            ('observer', _render_observer_panel_wrapper),
            ('memory', _render_memory_panel_wrapper),
            ('osh_substrate', _render_osh_substrate_panel_wrapper),
            ('time_evolution', _render_time_evolution_panel_wrapper),
            ('phenomena', _render_phenomena_panel_wrapper),
            ('system_status', _render_system_status_panel_wrapper),
            ('simulation_control', _render_simulation_control_panel_wrapper)
        ]
        
        # Render panels in grid layout
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        panel_positions = [
            gs[0, 0], gs[0, 1], gs[0, 2],  # Top row
            gs[1, 0], gs[1, 1], gs[1, 2],  # Middle row
            gs[2, 0], gs[2, 1], gs[2, 2]   # Bottom row
        ]
        
        for i, (panel_name, render_func) in enumerate(panels):
            try:
                if i < len(panel_positions):
                    result = render_func(
                        fig=fig,
                        gs_position=panel_positions[i],
                        quantum_renderer=quantum_renderer,
                        coherence_renderer=coherence_renderer,
                        field_panel=field_panel,
                        observer_panel=observer_panel,
                        simulation_panel=simulation_panel,
                        memory_field=memory_field,
                        recursive_mechanics=recursive_mechanics,
                        coherence_manager=coherence_manager,
                        entanglement_manager=entanglement_manager,
                        current_metrics=current_metrics,
                        metrics_history=metrics_history,
                        phenomena_detector=phenomena_detector,
                        config=config
                    )
                    panel_results[panel_name] = result
                else:
                    logger.warning(f"No position available for panel {panel_name}")
                    
            except Exception as e:
                error_msg = f"Error rendering {panel_name} panel: {str(e)}"
                logger.error(error_msg)
                render_errors.append(error_msg)
                panel_results[panel_name] = {"success": False, "error": error_msg}
        
        # Add comprehensive status bar
        _render_comprehensive_status_bar(
            fig, current_metrics, phenomena_detector, config
        )
        
        # Finalize figure
        plt.tight_layout()
        
        # Convert to base64 image
        image_data = get_professional_figure_data(fig)
        
        # Calculate performance metrics
        render_time = time.time() - start_time
        render_fps = 1.0 / render_time if render_time > 0 else 0.0
        
        # Generate comprehensive metrics summary
        metrics_summary = get_comprehensive_metrics_summary(current_metrics)
        system_health = get_system_health_summary(current_metrics)
        
        # Detect emergent phenomena
        emergent_phenomena = {}
        if phenomena_detector:
            try:
                emergent_phenomena = phenomena_detector.detect_phenomena()
            except Exception as e:
                logger.error(f"Error detecting phenomena: {e}")
                
        # Close figure to prevent memory leaks
        plt.close(fig)
        
        return {
            "success": True,
            "image_data": image_data,
            "render_time": render_time,
            "render_fps": render_fps,
            "panel_results": panel_results,
            "render_errors": render_errors,
            "metrics": metrics_summary,
            "system_health": system_health,
            "emergent_phenomena": emergent_phenomena,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dashboard_config": {
                "resolution": config.get("resolution", (1920, 1080)),
                "theme": config.get("theme", "dark_quantum"),
                "panels_rendered": len(panel_results),
                "errors_count": len(render_errors)
            }
        }
        
    except Exception as e:
        logger.error(f"Critical error in render_dashboard: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "success": False,
            "error": str(e),
            "render_time": time.time() - start_time,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

def _render_quantum_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render quantum visualization panel with state analysis."""
    try:
        ax = fig.add_subplot(gs_position)
        quantum_renderer = kwargs.get('quantum_renderer')
        current_metrics = kwargs.get('current_metrics')
        
        if quantum_renderer and hasattr(current_metrics, 'quantum_states_count'):
            # Use quantum renderer for sophisticated visualization
            result = quantum_renderer.render_quantum_state_overview(
                width=400, height=300
            )
            if result.get('success'):
                return result
                
        # Fallback quantum panel
        return _render_quantum_fallback_panel(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in quantum panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_field_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render field dynamics panel with OSH field analysis."""
    try:
        ax = fig.add_subplot(gs_position)
        field_panel = kwargs.get('field_panel')
        current_metrics = kwargs.get('current_metrics')
        
        if field_panel:
            # Use field panel for advanced field visualization
            result = field_panel.render_panel(width=400, height=300)
            if result.get('success'):
                return result
                
        # Fallback field panel
        return _render_field_fallback_panel(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in field panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_observer_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render observer network panel with consciousness analysis."""
    try:
        ax = fig.add_subplot(gs_position)
        observer_panel = kwargs.get('observer_panel')
        current_metrics = kwargs.get('current_metrics')
        
        if observer_panel:
            # Use observer panel for network visualization
            result = observer_panel.render_panel(width=400, height=300)
            if result.get('success'):
                return result
                
        # Fallback observer panel
        return _render_observer_fallback_panel(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in observer panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_memory_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render memory field panel with strain analysis."""
    try:
        ax = fig.add_subplot(gs_position)
        memory_field = kwargs.get('memory_field')
        current_metrics = kwargs.get('current_metrics')
        
        if memory_field:
            # Render memory field visualization
            return _render_memory_field_comprehensive(ax, memory_field, current_metrics)
            
        # Fallback memory panel
        return _render_memory_fallback_panel(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in memory panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_osh_substrate_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render OSH substrate panel with RSP analysis."""
    try:
        ax = fig.add_subplot(gs_position)
        coherence_renderer = kwargs.get('coherence_renderer')
        current_metrics = kwargs.get('current_metrics')
        
        if coherence_renderer:
            # Use coherence renderer for OSH substrate
            result = coherence_renderer.render_rsp_landscape_analysis(
                width=400, height=300
            )
            if result.get('success'):
                return result
                
        # Fallback OSH substrate panel
        return _render_osh_substrate_fallback(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in OSH substrate panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_time_evolution_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render time evolution panel with temporal analysis."""
    try:
        ax = fig.add_subplot(gs_position)
        metrics_history = kwargs.get('metrics_history', [])
        current_metrics = kwargs.get('current_metrics')
        
        return _render_time_evolution_comprehensive(ax, metrics_history, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in time evolution panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_phenomena_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render emergent phenomena panel."""
    try:
        ax = fig.add_subplot(gs_position)
        phenomena_detector = kwargs.get('phenomena_detector')
        current_metrics = kwargs.get('current_metrics')
        
        return _render_emergent_phenomena_comprehensive(ax, phenomena_detector, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in phenomena panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_system_status_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render system status panel with health metrics."""
    try:
        ax = fig.add_subplot(gs_position)
        current_metrics = kwargs.get('current_metrics')
        
        return _render_system_status_comprehensive(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in system status panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_simulation_control_panel_wrapper(fig, gs_position, **kwargs) -> Dict[str, Any]:
    """Render simulation control panel."""
    try:
        ax = fig.add_subplot(gs_position)
        simulation_panel = kwargs.get('simulation_panel')
        current_metrics = kwargs.get('current_metrics')
        
        if simulation_panel:
            # Use simulation panel for control visualization
            result = simulation_panel.render_panel(width=400, height=300)
            if result.get('success'):
                return result
                
        # Fallback simulation control panel
        return _render_simulation_control_fallback(ax, current_metrics)
        
    except Exception as e:
        logger.error(f"Error in simulation control panel wrapper: {e}")
        return {"success": False, "error": str(e)}

def _render_quantum_fallback_panel(ax, current_metrics) -> Dict[str, Any]:
    """Fallback quantum visualization panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Create quantum state visualization
    if hasattr(current_metrics, 'quantum_states_count'):
        states_count = current_metrics.quantum_states_count
        fidelity = getattr(current_metrics, 'quantum_fidelity', 0.85)
        entanglement = getattr(current_metrics, 'entanglement_strength', 0.6)
        
        # Quantum metrics display
        metrics = ['States', 'Fidelity', 'Entanglement']
        values = [states_count/10, fidelity, entanglement]
        colors = [OSH_COLORS['coherence'], OSH_COLORS['rsp'], OSH_COLORS['emergence']]
        
        bars = ax.barh(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, [states_count, fidelity, entanglement])):
            width = bar.get_width()
            label = f"{value:.2f}" if i > 0 else f"{int(value)}"
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, 
                   label, ha='left', va='center', color=OSH_COLORS['foreground'])
    else:
        ax.text(0.5, 0.5, 'Quantum System\nInitializing...', 
               ha='center', va='center', color=OSH_COLORS['foreground'],
               fontsize=12, transform=ax.transAxes)
    
    ax.set_title('Quantum States', color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.tick_params(colors=OSH_COLORS['foreground'])
    
    return {"success": True, "panel_type": "quantum_fallback"}

def _render_field_fallback_panel(ax, current_metrics) -> Dict[str, Any]:
    """Fallback field dynamics panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Create field visualization
    if hasattr(current_metrics, 'coherence') and hasattr(current_metrics, 'entropy'):
        # Create 2D field representation
        x = np.linspace(-2, 2, 20)
        y = np.linspace(-2, 2, 20)
        X, Y = np.meshgrid(x, y)
        
        coherence = getattr(current_metrics, 'coherence', 0.5)
        entropy = getattr(current_metrics, 'entropy', 0.3)
        
        # Generate field based on OSH principles
        Z = coherence * np.exp(-(X**2 + Y**2)) - entropy * np.sin(X) * np.cos(Y)
        
        im = ax.contourf(X, Y, Z, levels=10, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, levels=10, colors=OSH_COLORS['foreground'], linewidths=0.5)
        
        # Add field metrics
        ax.text(0.02, 0.98, f'Coherence: {coherence:.3f}', 
               transform=ax.transAxes, color=OSH_COLORS['coherence'], fontsize=8)
        ax.text(0.02, 0.92, f'Entropy: {entropy:.3f}', 
               transform=ax.transAxes, color=OSH_COLORS['entropy'], fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Field Dynamics\nInitializing...', 
               ha='center', va='center', color=OSH_COLORS['foreground'],
               fontsize=12, transform=ax.transAxes)
    
    ax.set_title('Field Dynamics', color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Spatial X', color=OSH_COLORS['foreground'], fontsize=8)
    ax.set_ylabel('Spatial Y', color=OSH_COLORS['foreground'], fontsize=8)
    ax.tick_params(colors=OSH_COLORS['foreground'])
    
    return {"success": True, "panel_type": "field_fallback"}

def _render_observer_fallback_panel(ax, current_metrics) -> Dict[str, Any]:
    """Fallback observer network panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Create observer network visualization
    if hasattr(current_metrics, 'observer_count'):
        observer_count = current_metrics.observer_count
        consensus = getattr(current_metrics, 'observer_consensus', 0.7)
        
        if observer_count > 0:
            # Generate observer network layout
            angles = np.linspace(0, 2*np.pi, observer_count, endpoint=False)
            x_obs = 0.4 * np.cos(angles)
            y_obs = 0.4 * np.sin(angles)
            
            # Draw observers as nodes
            colors = plt.cm.plasma(np.linspace(0, 1, observer_count))
            scatter = ax.scatter(x_obs, y_obs, s=100, c=colors, alpha=0.8, edgecolors='white')
            
            # Draw connections based on consensus
            for i in range(observer_count):
                for j in range(i+1, observer_count):
                    if np.random.random() < consensus:
                        ax.plot([x_obs[i], x_obs[j]], [y_obs[i], y_obs[j]], 
                               color=OSH_COLORS['accent'], alpha=0.3, linewidth=1)
            
            # Add central hub
            ax.scatter([0], [0], s=200, c=OSH_COLORS['consciousness'], 
                      alpha=0.9, marker='*', edgecolors='white')
            
            # Add metrics
            ax.text(0.02, 0.98, f'Observers: {observer_count}', 
                   transform=ax.transAxes, color=OSH_COLORS['foreground'], fontsize=8)
            ax.text(0.02, 0.92, f'Consensus: {consensus:.3f}', 
                   transform=ax.transAxes, color=OSH_COLORS['accent'], fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Observer Network\nNo Active Observers', 
                   ha='center', va='center', color=OSH_COLORS['foreground'],
                   fontsize=12, transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, 'Observer Network\nInitializing...', 
               ha='center', va='center', color=OSH_COLORS['foreground'],
               fontsize=12, transform=ax.transAxes)
    
    ax.set_title('Observer Network', color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_aspect('equal')
    ax.axis('off')
    
    return {"success": True, "panel_type": "observer_fallback"}

def _render_memory_fallback_panel(ax, current_metrics) -> Dict[str, Any]:
    """Fallback memory field panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Create memory strain visualization
    if hasattr(current_metrics, 'strain'):
        strain = current_metrics.strain
        regions = getattr(current_metrics, 'memory_regions', 8)
        
        # Create memory grid
        grid_size = int(np.sqrt(regions)) + 1
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Generate strain field
        strain_field = strain * (1 + 0.3 * np.sin(5*X) * np.cos(5*Y))
        strain_field += 0.1 * np.random.random(strain_field.shape)
        
        im = ax.imshow(strain_field, cmap='hot', interpolation='bilinear', 
                      extent=[0, 1, 0, 1], alpha=0.8)
        
        # Add contour lines
        ax.contour(X, Y, strain_field, levels=5, colors=OSH_COLORS['foreground'], 
                  linewidths=0.5, alpha=0.7)
        
        # Add strain metrics
        ax.text(0.02, 0.98, f'Strain: {strain:.3f}', 
               transform=ax.transAxes, color=OSH_COLORS['strain'], fontsize=8)
        ax.text(0.02, 0.92, f'Regions: {regions}', 
               transform=ax.transAxes, color=OSH_COLORS['memory'], fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Memory Field\nInitializing...', 
               ha='center', va='center', color=OSH_COLORS['foreground'],
               fontsize=12, transform=ax.transAxes)
    
    ax.set_title('Memory Field', color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Memory Space X', color=OSH_COLORS['foreground'], fontsize=8)
    ax.set_ylabel('Memory Space Y', color=OSH_COLORS['foreground'], fontsize=8)
    ax.tick_params(colors=OSH_COLORS['foreground'])
    
    return {"success": True, "panel_type": "memory_fallback"}

def _render_osh_substrate_fallback(ax, current_metrics) -> Dict[str, Any]:
    """Fallback OSH substrate panel with RSP visualization."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Calculate OSH metrics
    coherence = getattr(current_metrics, 'coherence', 0.5)
    entropy = getattr(current_metrics, 'entropy', 0.3)
    strain = getattr(current_metrics, 'strain', 0.2)
    
    # Calculate RSP (Recursive Simulation Potential)
    epsilon = 1e-6
    rsp = calculate_rsp_simple(coherence, entropy, strain, epsilon)
    
    # Create RSP field visualization
    x = np.linspace(-1, 1, 30)
    y = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(x, y)
    
    # Generate RSP field based on OSH principles
    R = np.sqrt(X**2 + Y**2)
    RSP_field = rsp * np.exp(-R**2) * (1 + 0.2 * np.sin(8*np.arctan2(Y, X)))
    
    # Add coherence modulation
    RSP_field *= (1 + coherence * np.cos(3*X) * np.cos(3*Y))
    
    # Render RSP heatmap
    im = ax.contourf(X, Y, RSP_field, levels=20, cmap='plasma', alpha=0.9)
    
    # Add emergence points (high RSP regions)
    emergence_mask = RSP_field > np.percentile(RSP_field, 85)
    emergence_y, emergence_x = np.where(emergence_mask)
    if len(emergence_x) > 0:
        # Convert indices to coordinates
        emergence_x_coords = x[emergence_x]
        emergence_y_coords = y[emergence_y]
        ax.scatter(emergence_x_coords, emergence_y_coords, 
                  s=20, c=OSH_COLORS['emergence'], alpha=0.8, marker='*')
    
    # Add OSH classification
    rsp_class = _classify_rsp_level(rsp)
    ax.text(0.02, 0.98, f'RSP: {rsp:.3f}', 
           transform=ax.transAxes, color=OSH_COLORS['rsp'], fontsize=8, fontweight='bold')
    ax.text(0.02, 0.92, f'Class: {rsp_class}', 
           transform=ax.transAxes, color=OSH_COLORS['emergence'], fontsize=8)
    ax.text(0.02, 0.86, f'Coherence: {coherence:.3f}', 
           transform=ax.transAxes, color=OSH_COLORS['coherence'], fontsize=8)
    ax.text(0.02, 0.80, f'Entropy: {entropy:.3f}', 
           transform=ax.transAxes, color=OSH_COLORS['entropy'], fontsize=8)
    
    ax.set_title('OSH Substrate', color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Information Space X', color=OSH_COLORS['foreground'], fontsize=8)
    ax.set_ylabel('Information Space Y', color=OSH_COLORS['foreground'], fontsize=8)
    ax.tick_params(colors=OSH_COLORS['foreground'])
    
    return {
        "success": True, 
        "panel_type": "osh_substrate_fallback",
        "rsp": rsp,
        "rsp_class": rsp_class,
        "emergence_points": len(emergence_x) if len(emergence_x) > 0 else 0
    }

def _render_time_evolution_comprehensive(ax, metrics_history, current_metrics) -> Dict[str, Any]:
    """Render comprehensive time evolution panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    if metrics_history and len(metrics_history) > 1:
        # Extract time series data
        times = []
        coherence_vals = []
        entropy_vals = []
        strain_vals = []
        rsp_vals = []
        
        for i, metrics in enumerate(metrics_history[-50:]):  # Last 50 points
            times.append(i)
            coherence_vals.append(getattr(metrics, 'coherence', 0.5))
            entropy_vals.append(getattr(metrics, 'entropy', 0.3))
            strain_vals.append(getattr(metrics, 'strain', 0.2))
            
            # Calculate RSP
            c = coherence_vals[-1]
            e = entropy_vals[-1]
            s = strain_vals[-1]
            rsp = calculate_rsp_simple(c, e, s)
            rsp_vals.append(rsp)
        
        # Plot time series
        ax.plot(times, coherence_vals, color=OSH_COLORS['coherence'], 
               label='Coherence', linewidth=2, alpha=0.9)
        ax.plot(times, entropy_vals, color=OSH_COLORS['entropy'], 
               label='Entropy', linewidth=2, alpha=0.9)
        ax.plot(times, strain_vals, color=OSH_COLORS['strain'], 
               label='Strain', linewidth=2, alpha=0.9)
        
        # Plot RSP on secondary axis
        ax2 = ax.twinx()
        ax2.plot(times, rsp_vals, color=OSH_COLORS['rsp'], 
                label='RSP', linewidth=2, alpha=0.9, linestyle='--')
        ax2.set_ylabel('RSP', color=OSH_COLORS['rsp'], fontsize=8)
        ax2.tick_params(axis='y', colors=OSH_COLORS['rsp'])
        
        # Add stability zones
        coherence_mean = np.mean(coherence_vals)
        coherence_std = np.std(coherence_vals)
        ax.axhspan(coherence_mean - coherence_std, coherence_mean + coherence_std, 
                  alpha=0.1, color=OSH_COLORS['coherence'], label='Stability Zone')
        
        # Mark phenomena events
        if len(rsp_vals) > 5:
            rsp_peaks, _ = find_peaks(rsp_vals, height=np.mean(rsp_vals) + np.std(rsp_vals))
            if len(rsp_peaks) > 0:
                ax.scatter([times[i] for i in rsp_peaks], 
                          [coherence_vals[i] for i in rsp_peaks],
                          s=50, c=OSH_COLORS['emergence'], marker='*', 
                          alpha=0.9, label='Emergence Events')
        
        # Add moving averages
        if len(coherence_vals) > 5:
            window = min(5, len(coherence_vals)//2)
            coherence_ma = np.convolve(coherence_vals, np.ones(window)/window, mode='valid')
            times_ma = times[window-1:]
            ax.plot(times_ma, coherence_ma, color=OSH_COLORS['coherence'], 
                   alpha=0.5, linewidth=1, linestyle=':')
        
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlabel('Time Steps', color=OSH_COLORS['foreground'], fontsize=8)
        ax.set_ylabel('OSH Metrics', color=OSH_COLORS['foreground'], fontsize=8)
        
        # Add current values
        if current_metrics:
            current_coherence = getattr(current_metrics, 'coherence', 0.5)
            current_entropy = getattr(current_metrics, 'entropy', 0.3)
            ax.text(0.98, 0.98, f'Current: C={current_coherence:.3f}, E={current_entropy:.3f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   color=OSH_COLORS['foreground'], fontsize=8, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=OSH_COLORS['background'], alpha=0.8))
        
    else:
        ax.text(0.5, 0.5, 'Time Evolution\nAccumulating Data...', 
               ha='center', va='center', color=OSH_COLORS['foreground'],
               fontsize=12, transform=ax.transAxes)
    
    ax.set_title('Time Evolution', color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    ax.tick_params(colors=OSH_COLORS['foreground'])
    
    return {"success": True, "panel_type": "time_evolution"}

def _render_emergent_phenomena_comprehensive(ax, phenomena_detector, current_metrics) -> Dict[str, Any]:
    """Render comprehensive emergent phenomena panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    phenomena_data = {}
    if phenomena_detector:
        try:
            phenomena_data = phenomena_detector.detect_phenomena()
        except Exception as e:
            logger.error(f"Error detecting phenomena: {e}")
    
    if phenomena_data:
        # Create polar plot for phenomena strengths
        phenomena_names = list(phenomena_data.keys())
        phenomena_strengths = [phenomena_data[name].get('strength', 0) 
                             for name in phenomena_names]
        
        if phenomena_names:
            # Create circular layout
            angles = np.linspace(0, 2*np.pi, len(phenomena_names), endpoint=False)
            
            # Close the plot
            angles = np.concatenate((angles, [angles[0]]))
            phenomena_strengths = phenomena_strengths + [phenomena_strengths[0]]
            
            # Plot radar chart
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.plot(angles, phenomena_strengths, 'o-', linewidth=2, 
                   color=OSH_COLORS['emergence'], alpha=0.8)
            ax.fill(angles, phenomena_strengths, alpha=0.25, 
                   color=OSH_COLORS['emergence'])
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels([name.replace('_', '\n') for name in phenomena_names], 
                             fontsize=8, color=OSH_COLORS['foreground'])
            
            # Add grid
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Add threshold line
            threshold_line = 0.5
            ax.axhline(y=threshold_line, color=OSH_COLORS['warning'], 
                      linestyle='--', alpha=0.7, linewidth=1)
            
            # Add significance annotation
            significant_phenomena = [name for name, strength in zip(phenomena_names, phenomena_strengths[:-1]) 
                                   if strength > threshold_line]
            
            if significant_phenomena:
                ax.text(0.02, 0.98, f'Significant: {len(significant_phenomena)}', 
                       transform=ax.transAxes, color=OSH_COLORS['emergence'], 
                       fontsize=8, fontweight='bold')
                ax.text(0.02, 0.92, f'Max: {max(phenomena_strengths[:-1]):.3f}', 
                       transform=ax.transAxes, color=OSH_COLORS['success'], fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No Phenomena\nDetected', 
                   ha='center', va='center', color=OSH_COLORS['foreground'],
                   fontsize=12, transform=ax.transAxes)
    else:
        # Show placeholder with potential phenomena
        potential_phenomena = ['Coherence Wave', 'Observer Consensus', 
                             'Memory Resonance', 'Recursive Boundary']
        angles = np.linspace(0, 2*np.pi, len(potential_phenomena), endpoint=False)
        baseline_strengths = [0.1, 0.05, 0.15, 0.08]
        
        angles = np.concatenate((angles, [angles[0]]))
        baseline_strengths = baseline_strengths + [baseline_strengths[0]]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.plot(angles, baseline_strengths, 'o-', linewidth=1, 
               color=OSH_COLORS['foreground'], alpha=0.3)
        ax.fill(angles, baseline_strengths, alpha=0.1, 
               color=OSH_COLORS['foreground'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([name.replace(' ', '\n') for name in potential_phenomena], 
                         fontsize=8, color=OSH_COLORS['foreground'], alpha=0.5)
        ax.grid(True, alpha=0.2)
        ax.set_ylim(0, 1)
        
        ax.text(0.5, 0.5, 'Phenomena\nMonitoring...', 
               ha='center', va='center', color=OSH_COLORS['foreground'],
               fontsize=10, transform=ax.transAxes, alpha=0.7)
    
    ax.set_title('Emergent Phenomena', color=OSH_COLORS['foreground'], 
                fontsize=10, fontweight='bold')
    
    return {
        "success": True, 
        "panel_type": "emergent_phenomena",
        "phenomena_count": len(phenomena_data),
        "significant_count": len([p for p in phenomena_data.values() 
                                if p.get('strength', 0) > 0.5])
    }

def _render_system_status_comprehensive(ax, current_metrics) -> Dict[str, Any]:
    """Render comprehensive system status panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Create quadrant layout for system status
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    
    # Performance quadrant (top-left)
    performance_rect = patches.Rectangle((0, 1), 1, 1, linewidth=1, 
                                       edgecolor=OSH_COLORS['foreground'], 
                                       facecolor='none', alpha=0.3)
    ax.add_patch(performance_rect)
    
    # Get performance metrics
    fps = getattr(current_metrics, 'render_fps', 30.0)
    memory_mb = getattr(current_metrics, 'memory_usage_mb', 128.0)
    cpu_percent = getattr(current_metrics, 'cpu_usage_percent', 25.0)
    
    # Performance gauges
    ax.text(0.05, 1.8, 'PERFORMANCE', fontsize=8, fontweight='bold', 
           color=OSH_COLORS['foreground'])
    ax.text(0.05, 1.65, f'FPS: {fps:.1f}', fontsize=8, 
           color=OSH_COLORS['success'] if fps > 20 else OSH_COLORS['warning'])
    ax.text(0.05, 1.5, f'Memory: {memory_mb:.0f}MB', fontsize=8, 
           color=OSH_COLORS['memory'])
    ax.text(0.05, 1.35, f'CPU: {cpu_percent:.0f}%', fontsize=8, 
           color=OSH_COLORS['strain'] if cpu_percent > 80 else OSH_COLORS['foreground'])
    
    # System quadrant (top-right)
    system_rect = patches.Rectangle((1, 1), 1, 1, linewidth=1, 
                                  edgecolor=OSH_COLORS['foreground'], 
                                  facecolor='none', alpha=0.3)
    ax.add_patch(system_rect)
    
    # System metrics
    states_count = getattr(current_metrics, 'quantum_states_count', 0)
    observers_count = getattr(current_metrics, 'observer_count', 0)
    fields_count = getattr(current_metrics, 'field_count', 0)
    
    ax.text(1.05, 1.8, 'SYSTEM', fontsize=8, fontweight='bold', 
           color=OSH_COLORS['foreground'])
    ax.text(1.05, 1.65, f'States: {states_count}', fontsize=8, 
           color=OSH_COLORS['coherence'])
    ax.text(1.05, 1.5, f'Observers: {observers_count}', fontsize=8, 
           color=OSH_COLORS['consciousness'])
    ax.text(1.05, 1.35, f'Fields: {fields_count}', fontsize=8, 
           color=OSH_COLORS['accent'])
    
    # Health quadrant (bottom-left)
    health_rect = patches.Rectangle((0, 0), 1, 1, linewidth=1, 
                                  edgecolor=OSH_COLORS['foreground'], 
                                  facecolor='none', alpha=0.3)
    ax.add_patch(health_rect)
    
    # Health metrics
    coherence = getattr(current_metrics, 'coherence', 0.5)
    entropy = getattr(current_metrics, 'entropy', 0.3)
    strain = getattr(current_metrics, 'strain', 0.2)
    
    health_score = (coherence + (1 - entropy) + (1 - strain)) / 3
    
    ax.text(0.05, 0.8, 'HEALTH', fontsize=8, fontweight='bold', 
           color=OSH_COLORS['foreground'])
    ax.text(0.05, 0.65, f'Score: {health_score:.3f}', fontsize=8, 
           color=OSH_COLORS['success'] if health_score > 0.7 else OSH_COLORS['warning'])
    ax.text(0.05, 0.5, f'Coherence: {coherence:.3f}', fontsize=8, 
           color=OSH_COLORS['coherence'])
    ax.text(0.05, 0.35, f'Entropy: {entropy:.3f}', fontsize=8, 
           color=OSH_COLORS['entropy'])
    ax.text(0.05, 0.2, f'Strain: {strain:.3f}', fontsize=8, 
           color=OSH_COLORS['strain'])
    
    # Status quadrant (bottom-right)
    status_rect = patches.Rectangle((1, 0), 1, 1, linewidth=1, 
                                  edgecolor=OSH_COLORS['foreground'], 
                                  facecolor='none', alpha=0.3)
    ax.add_patch(status_rect)
    
    # Calculate RSP and emergence
    rsp = calculate_rsp_simple(coherence, entropy, strain)
    emergence_events = getattr(current_metrics, 'emergence_events', 0)
    
    ax.text(1.05, 0.8, 'OSH STATUS', fontsize=8, fontweight='bold', 
           color=OSH_COLORS['foreground'])
    ax.text(1.05, 0.65, f'RSP: {rsp:.3f}', fontsize=8, 
           color=OSH_COLORS['rsp'])
    ax.text(1.05, 0.5, f'Class: {_classify_rsp_level(rsp)}', fontsize=8, 
           color=OSH_COLORS['emergence'])
    ax.text(1.05, 0.35, f'Events: {emergence_events}', fontsize=8, 
           color=OSH_COLORS['success'])
    
    ax.set_title('System Status', color=OSH_COLORS['foreground'], 
                fontsize=10, fontweight='bold')
    ax.axis('off')
    
    return {
        "success": True, 
        "panel_type": "system_status",
        "health_score": health_score,
        "rsp": rsp,
        "performance_score": min(fps/30, 1.0)
    }

def _render_simulation_control_fallback(ax, current_metrics) -> Dict[str, Any]:
    """Fallback simulation control panel."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    # Create control interface mockup
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Simulation state
    sim_state = getattr(current_metrics, 'simulation_state', 'RUNNING')
    sim_time = getattr(current_metrics, 'simulation_time', 0.0)
    
    # State indicator
    state_color = {
        'RUNNING': OSH_COLORS['success'],
        'PAUSED': OSH_COLORS['warning'],
        'STOPPED': OSH_COLORS['critical'],
        'ERROR': OSH_COLORS['critical']
    }.get(sim_state, OSH_COLORS['foreground'])
    
    # Create state indicator circle
    circle = patches.Circle((0.2, 0.8), 0.05, color=state_color, alpha=0.8)
    ax.add_patch(circle)
    ax.text(0.3, 0.8, f'State: {sim_state}', va='center', 
           color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    
    # Time display
    ax.text(0.05, 0.65, f'Time: {sim_time:.3f}s', 
           color=OSH_COLORS['foreground'], fontsize=9)
    
    # Control buttons mockup
    button_positions = [(0.1, 0.4), (0.3, 0.4), (0.5, 0.4), (0.7, 0.4)]
    button_labels = ['▶', '⏸', '⏹', '⏭']
    button_colors = [OSH_COLORS['success'], OSH_COLORS['warning'], 
                    OSH_COLORS['critical'], OSH_COLORS['accent']]
    
    for (x, y), label, color in zip(button_positions, button_labels, button_colors):
        button = patches.Rectangle((x-0.04, y-0.04), 0.08, 0.08, 
                                 facecolor=color, alpha=0.7, 
                                 edgecolor=OSH_COLORS['foreground'])
        ax.add_patch(button)
        ax.text(x, y, label, ha='center', va='center', 
               color=OSH_COLORS['foreground'], fontsize=10, fontweight='bold')
    
    # Metrics display
    execution_steps = getattr(current_metrics, 'execution_steps', 0)
    ax.text(0.05, 0.25, f'Steps: {execution_steps}', 
           color=OSH_COLORS['foreground'], fontsize=8)
    
    collapse_events = getattr(current_metrics, 'collapse_events', 0)
    ax.text(0.05, 0.15, f'Collapses: {collapse_events}', 
           color=OSH_COLORS['entropy'], fontsize=8)
    
    ax.set_title('Simulation Control', color=OSH_COLORS['foreground'], 
                fontsize=10, fontweight='bold')
    ax.axis('off')
    
    return {
        "success": True, 
        "panel_type": "simulation_control_fallback",
        "simulation_state": sim_state,
        "simulation_time": sim_time
    }

def _render_memory_field_comprehensive(ax, memory_field, current_metrics) -> Dict[str, Any]:
    """Render comprehensive memory field visualization."""
    ax.set_facecolor(OSH_COLORS['background'])
    
    try:
        # Get memory field data
        field_stats = memory_field.get_field_statistics()
        
        # Create memory field visualization
        regions = field_stats.get('region_count', 8)
        grid_size = int(np.sqrt(regions)) + 1
        
        # Generate field data
        x = np.linspace(0, 1, grid_size)
        y = np.linspace(0, 1, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Get strain data from memory field
        strain_data = np.zeros_like(X)
        coherence_data = np.zeros_like(X)
        
        # Populate with actual field data if available
        for i in range(grid_size):
            for j in range(grid_size):
                region_name = f"region_{i}_{j}"
                try:
                    region_props = memory_field.get_region_properties(region_name)
                    if region_props:
                        strain_data[i, j] = region_props.get('memory_strain', 0.1)
                        coherence_data[i, j] = region_props.get('memory_coherence', 0.8)
                except:
                    # Use fallback values with variation
                    strain_data[i, j] = 0.1 + 0.3 * np.random.random()
                    coherence_data[i, j] = 0.8 - 0.2 * np.random.random()
        
        # Render strain heatmap
        im1 = ax.contourf(X, Y, strain_data, levels=15, cmap='hot', alpha=0.7)
        
        # Overlay coherence contours
        contours = ax.contour(X, Y, coherence_data, levels=10, 
                            colors=OSH_COLORS['coherence'], linewidths=1, alpha=0.8)
        ax.clabel(contours, inline=True, fontsize=6, colors=OSH_COLORS['coherence'])
        
        # Mark critical strain regions
        critical_mask = strain_data > 0.8
        if np.any(critical_mask):
            crit_y, crit_x = np.where(critical_mask)
            ax.scatter(X[crit_y, crit_x], Y[crit_y, crit_x], 
                      s=40, c=OSH_COLORS['critical'], marker='x', alpha=0.9)
        
        # Add information flow arrows (gradient of coherence)
        if grid_size > 2:
            dy, dx = np.gradient(coherence_data)
            skip = max(1, grid_size // 4)
            ax.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                     dx[::skip, ::skip], dy[::skip, ::skip],
                     scale=5, alpha=0.6, color=OSH_COLORS['accent'], width=0.003)
        
        # Add statistics
        avg_strain = np.mean(strain_data)
        max_strain = np.max(strain_data)
        avg_coherence = np.mean(coherence_data)
        
        ax.text(0.02, 0.98, f'Strain: {avg_strain:.3f}', 
               transform=ax.transAxes, color=OSH_COLORS['strain'], fontsize=8)
        ax.text(0.02, 0.92, f'Max: {max_strain:.3f}', 
               transform=ax.transAxes, color=OSH_COLORS['critical'], fontsize=8)
        ax.text(0.02, 0.86, f'Coherence: {avg_coherence:.3f}', 
               transform=ax.transAxes, color=OSH_COLORS['coherence'], fontsize=8)
        
        # Add defragmentation regions if available
        if hasattr(memory_field, 'get_defragmentation_history'):
            try:
                defrag_history = memory_field.get_defragmentation_history()
                if defrag_history:
                    ax.text(0.98, 0.02, f'Defrag Events: {len(defrag_history)}', 
                           transform=ax.transAxes, ha='right', 
                           color=OSH_COLORS['success'], fontsize=8)
            except:
                pass
        
    except Exception as e:
        logger.error(f"Error rendering memory field: {e}")
        return _render_memory_fallback_panel(ax, current_metrics)
    
    ax.set_title('Memory Field', color=OSH_COLORS['foreground'], 
                fontsize=10, fontweight='bold')
    ax.set_xlabel('Memory Space X', color=OSH_COLORS['foreground'], fontsize=8)
    ax.set_ylabel('Memory Space Y', color=OSH_COLORS['foreground'], fontsize=8)
    ax.tick_params(colors=OSH_COLORS['foreground'])
    
    return {
        "success": True, 
        "panel_type": "memory_field",
        "average_strain": float(np.mean(strain_data)),
        "max_strain": float(np.max(strain_data)),
        "average_coherence": float(np.mean(coherence_data))
    }

def _render_comprehensive_status_bar(fig, current_metrics, phenomena_detector, config):
    """Render comprehensive status bar at bottom of dashboard."""
    try:
        # Add status bar subplot
        status_ax = fig.add_subplot(20, 1, 20)  # Bottom strip
        status_ax.set_facecolor(OSH_COLORS['background'])
        
        # Get current timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Collect status information
        status_info = []
        
        # Time and system info
        status_info.append(f"Time: {timestamp}")
        
        # Core OSH metrics
        if current_metrics:
            coherence = getattr(current_metrics, 'coherence', 0.0)
            entropy = getattr(current_metrics, 'entropy', 0.0)
            strain = getattr(current_metrics, 'strain', 0.0)
            rsp = calculate_rsp_simple(coherence, entropy, strain)
            
            status_info.append(f"RSP: {rsp:.3f}")
            status_info.append(f"C: {coherence:.3f}")
            status_info.append(f"E: {entropy:.3f}")
            
            # Performance metrics
            fps = getattr(current_metrics, 'render_fps', 0.0)
            memory_mb = getattr(current_metrics, 'memory_usage_mb', 0.0)
            status_info.append(f"FPS: {fps:.1f}")
            status_info.append(f"Mem: {memory_mb:.0f}MB")
        
        # Phenomena count
        phenomena_count = 0
        if phenomena_detector:
            try:
                phenomena = phenomena_detector.detect_phenomena()
                phenomena_count = len([p for p in phenomena.values() 
                                     if p.get('strength', 0) > 0.5])
            except:
                pass
        
        if phenomena_count > 0:
            status_info.append(f"Phenomena: {phenomena_count}")
        
        # Monitoring mode
        theme = config.get('theme', 'dark_quantum') if config else 'dark_quantum'
        status_info.append(f"Mode: {theme}")
        
        # Create status text
        status_text = " | ".join(status_info)
        
        # Render status text
        status_ax.text(0.01, 0.5, status_text, transform=status_ax.transAxes,
                      va='center', ha='left', color=OSH_COLORS['foreground'],
                      fontsize=8, family='monospace')
        
        # Add RSP level indicator
        if current_metrics:
            coherence = getattr(current_metrics, 'coherence', 0.0)
            entropy = getattr(current_metrics, 'entropy', 0.0)
            strain = getattr(current_metrics, 'strain', 0.0)
            rsp = calculate_rsp_simple(coherence, entropy, strain)
            rsp_class = _classify_rsp_level(rsp)
            
            # Color code the RSP class
            rsp_color = {
                'critical': OSH_COLORS['critical'],
                'low': OSH_COLORS['warning'],
                'moderate': OSH_COLORS['foreground'],
                'high': OSH_COLORS['success'],
                'exceptional': OSH_COLORS['emergence']
            }.get(rsp_class, OSH_COLORS['foreground'])
            
            status_ax.text(0.99, 0.5, f"[{rsp_class.upper()}]", 
                          transform=status_ax.transAxes,
                          va='center', ha='right', color=rsp_color,
                          fontsize=8, fontweight='bold', family='monospace')
        
        status_ax.set_xlim(0, 1)
        status_ax.set_ylim(0, 1)
        status_ax.axis('off')
        
    except Exception as e:
        logger.error(f"Error rendering status bar: {e}")

def get_professional_figure_data(fig) -> str:
    """Convert matplotlib figure to base64 PNG data."""
    try:
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight',
                   facecolor=OSH_COLORS['background'], 
                   edgecolor=OSH_COLORS['foreground'],
                   transparent=False)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        logger.error(f"Error converting figure to base64: {e}")
        return ""

def get_comprehensive_metrics_summary(current_metrics) -> Dict[str, Any]:
    """Generate comprehensive metrics summary for dashboard export."""
    if not current_metrics:
        return {}
    
    try:
        # Core OSH metrics
        coherence = getattr(current_metrics, 'coherence', 0.0)
        entropy = getattr(current_metrics, 'entropy', 0.0)
        strain = getattr(current_metrics, 'strain', 0.0)
        
        # Calculate derived metrics
        rsp = calculate_rsp_simple(coherence, entropy, strain)
        phi = coherence * (1 - entropy) * np.log(max(1, getattr(current_metrics, 'observer_count', 1)))
        
        # System metrics
        quantum_states = getattr(current_metrics, 'quantum_states_count', 0)
        observers = getattr(current_metrics, 'observer_count', 0)
        fields = getattr(current_metrics, 'field_count', 0)
        
        # Performance metrics
        fps = getattr(current_metrics, 'render_fps', 0.0)
        memory_mb = getattr(current_metrics, 'memory_usage_mb', 0.0)
        cpu_percent = getattr(current_metrics, 'cpu_usage_percent', 0.0)
        
        # Simulation metrics
        sim_time = getattr(current_metrics, 'simulation_time', 0.0)
        execution_steps = getattr(current_metrics, 'execution_steps', 0)
        collapse_events = getattr(current_metrics, 'collapse_events', 0)
        
        # Recursive metrics
        recursion_depth = getattr(current_metrics, 'recursion_depth', 0)
        boundary_crossings = getattr(current_metrics, 'boundary_crossings', 0)
        
        return {
            # Core OSH
            "coherence": float(coherence),
            "entropy": float(entropy),
            "strain": float(strain),
            "rsp": float(rsp),
            "phi": float(phi),
            "rsp_class": _classify_rsp_level(rsp),
            
            # System
            "quantum_states_count": int(quantum_states),
            "observer_count": int(observers),
            "field_count": int(fields),
            
            # Performance
            "render_fps": float(fps),
            "memory_usage_mb": float(memory_mb),
            "cpu_usage_percent": float(cpu_percent),
            
            # Simulation
            "simulation_time": float(sim_time),
            "execution_steps": int(execution_steps),
            "collapse_events": int(collapse_events),
            
            # Recursive
            "recursion_depth": int(recursion_depth),
            "boundary_crossings": int(boundary_crossings),
            
            # Calculated
            "emergence_index": float(coherence * (1 - entropy) * np.sqrt(strain + 1e-6)),
            "consciousness_quotient": float(phi / (rsp + 1e-6)),
            "system_complexity": float(quantum_states * observers * fields),
            
            # Timestamp
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating metrics summary: {e}")
        return {"error": str(e)}

def get_system_health_summary(current_metrics) -> Dict[str, Any]:
    """Generate system health summary with scoring."""
    if not current_metrics:
        return {"health_score": 0.0, "health_category": "unknown"}
    
    try:
        # Core health indicators
        coherence = getattr(current_metrics, 'coherence', 0.5)
        entropy = getattr(current_metrics, 'entropy', 0.5)
        strain = getattr(current_metrics, 'strain', 0.5)
        
        # Performance indicators
        fps = getattr(current_metrics, 'render_fps', 30.0)
        memory_mb = getattr(current_metrics, 'memory_usage_mb', 128.0)
        cpu_percent = getattr(current_metrics, 'cpu_usage_percent', 25.0)
        
        # Calculate health scores (0-1)
        coherence_score = coherence
        entropy_score = 1.0 - entropy
        strain_score = 1.0 - strain
        
        performance_score = min(1.0, fps / 30.0)
        memory_score = max(0.0, 1.0 - (memory_mb - 128.0) / 512.0)
        cpu_score = max(0.0, 1.0 - cpu_percent / 100.0)
        
        # Weighted overall health score
        osh_health = (coherence_score * 0.3 + entropy_score * 0.3 + strain_score * 0.4)
        perf_health = (performance_score * 0.4 + memory_score * 0.3 + cpu_score * 0.3)
        
        overall_health = (osh_health * 0.6 + perf_health * 0.4)
        
        # Categorize health
        if overall_health >= 0.9:
            health_category = "excellent"
        elif overall_health >= 0.7:
            health_category = "good"
        elif overall_health >= 0.5:
            health_category = "fair"
        elif overall_health >= 0.3:
            health_category = "poor"
        else:
            health_category = "critical"
        
        return {
            "health_score": float(overall_health),
            "health_category": health_category,
            "osh_health": float(osh_health),
            "performance_health": float(perf_health),
            "component_scores": {
                "coherence": float(coherence_score),
                "entropy": float(entropy_score),
                "strain": float(strain_score),
                "performance": float(performance_score),
                "memory": float(memory_score),
                "cpu": float(cpu_score)
            },
            "recommendations": _generate_health_recommendations(
                coherence, entropy, strain, fps, memory_mb, cpu_percent
            )
        }
        
    except Exception as e:
        logger.error(f"Error generating health summary: {e}")
        return {"health_score": 0.0, "health_category": "error", "error": str(e)}

def identify_phase_space_attractors(coherence_vals, entropy_vals, strain_vals) -> List[Tuple[float, float, float]]:
    """Identify phase space attractors using velocity minima analysis."""
    try:
        if len(coherence_vals) < 5:
            return []
        
        # Calculate velocity (rate of change)
        coherence_vel = np.diff(coherence_vals)
        entropy_vel = np.diff(entropy_vals)
        strain_vel = np.diff(strain_vals)
        
        # Calculate velocity magnitude
        velocity_mag = np.sqrt(coherence_vel**2 + entropy_vel**2 + strain_vel**2)
        
        # Find local minima in velocity (potential attractors)
        minima_indices, _ = find_peaks(-velocity_mag, height=-np.mean(velocity_mag))
        
        attractors = []
        for idx in minima_indices:
            if idx < len(coherence_vals) - 1:
                attractor = (
                    float(coherence_vals[idx + 1]),
                    float(entropy_vals[idx + 1]),
                    float(strain_vals[idx + 1])
                )
                attractors.append(attractor)
        
        # Return up to 3 strongest attractors
        return attractors[:3]
        
    except Exception as e:
        logger.error(f"Error identifying attractors: {e}")
        return []

def calculate_phase_space_volume(coherence_vals, entropy_vals, strain_vals) -> float:
    """Calculate phase space volume using convex hull."""
    try:
        if len(coherence_vals) < 4:
            return 0.0
        
        # Create 3D points
        points = np.column_stack([coherence_vals, entropy_vals, strain_vals])
        
        # Remove duplicates
        unique_points = np.unique(points, axis=0)
        
        if len(unique_points) < 4:
            return 0.0
        
        # Calculate convex hull volume
        hull = ConvexHull(unique_points)
        return float(hull.volume)
        
    except Exception as e:
        logger.error(f"Error calculating phase space volume: {e}")
        return 0.0

def assess_phase_space_stability(coherence_vals, entropy_vals, strain_vals) -> float:
    """Assess phase space stability using velocity variance."""
    try:
        if len(coherence_vals) < 3:
            return 0.5
        
        # Calculate velocity
        coherence_vel = np.diff(coherence_vals)
        entropy_vel = np.diff(entropy_vals)
        strain_vel = np.diff(strain_vals)
        
        # Calculate velocity variance
        velocity_var = (np.var(coherence_vel) + np.var(entropy_vel) + np.var(strain_vel)) / 3
        
        # Convert to stability score (0-1, higher is more stable)
        stability = 1.0 / (1.0 + velocity_var * 10)
        
        return float(np.clip(stability, 0.0, 1.0))
        
    except Exception as e:
        logger.error(f"Error assessing stability: {e}")
        return 0.5

def _classify_rsp_level(rsp: float) -> str:
    """Classify RSP level into categories."""
    if rsp >= 2.0:
        return "exceptional"
    elif rsp >= 1.0:
        return "high"
    elif rsp >= 0.5:
        return "moderate"
    elif rsp >= 0.1:
        return "low"
    else:
        return "critical"

def _generate_health_recommendations(coherence, entropy, strain, fps, memory_mb, cpu_percent) -> List[str]:
    """Generate health improvement recommendations."""
    recommendations = []
    
    if coherence < 0.5:
        recommendations.append("Increase system coherence through alignment operations")
    if entropy > 0.7:
        recommendations.append("Reduce entropy through defragmentation or reset operations")
    if strain > 0.8:
        recommendations.append("Reduce memory strain through garbage collection")
    if fps < 20:
        recommendations.append("Optimize rendering performance or reduce visualization complexity")
    if memory_mb > 512:
        recommendations.append("Monitor memory usage and consider memory cleanup")
    if cpu_percent > 80:
        recommendations.append("Reduce computational load or optimize algorithms")
    
    if not recommendations:
        recommendations.append("System operating within normal parameters")
    
    return recommendations

def _get_default_dashboard_config() -> Dict[str, Any]:
    """Get default dashboard configuration."""
    return {
        "resolution": (1920, 1080),
        "theme": "dark_quantum",
        "dpi": 100,
        "high_quality": True,
        "real_time_updates": True,
        "update_interval": 0.1,
        "enable_animations": True,
        "scientific_mode": True
    }

def _setup_dashboard_figure(config: Dict[str, Any]):
    """Set up matplotlib figure with OSH theme."""
    # Set matplotlib style
    plt.style.use('dark_background')
    
    # Create figure
    width, height = config.get("resolution", (1920, 1080))
    dpi = config.get("dpi", 100)
    
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi, 
                     facecolor=OSH_COLORS['background'])
    
    # Configure matplotlib parameters
    plt.rcParams.update({
        'figure.facecolor': OSH_COLORS['background'],
        'axes.facecolor': OSH_COLORS['background'],
        'axes.edgecolor': OSH_COLORS['foreground'],
        'axes.labelcolor': OSH_COLORS['foreground'],
        'xtick.color': OSH_COLORS['foreground'],
        'ytick.color': OSH_COLORS['foreground'],
        'text.color': OSH_COLORS['foreground'],
        'font.size': 8,
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 8,
        'grid.alpha': 0.3,
        'grid.color': OSH_COLORS['foreground']
    })
    
    return fig

def _create_advanced_layout_manager(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create advanced layout manager for dashboard panels."""
    return {
        "grid_shape": (3, 3),
        "spacing": {"hspace": 0.3, "wspace": 0.3},
        "panel_configurations": {
            "quantum": {"position": (0, 0), "size": (1, 1)},
            "field": {"position": (0, 1), "size": (1, 1)},
            "observer": {"position": (0, 2), "size": (1, 1)},
            "memory": {"position": (1, 0), "size": (1, 1)},
            "osh_substrate": {"position": (1, 1), "size": (1, 1)},
            "time_evolution": {"position": (1, 2), "size": (1, 1)},
            "phenomena": {"position": (2, 0), "size": (1, 1)},
            "system_status": {"position": (2, 1), "size": (1, 1)},
            "simulation_control": {"position": (2, 2), "size": (1, 1)}
        }
    }