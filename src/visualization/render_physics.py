"""
render_physics.py - Unified Visualization Orchestrator for Recursia OSH Dashboard

This module implements the PhysicsRenderer class, which serves as the central rendering hub
for all scientific visual panels in the Recursia OSH dashboard. It dynamically integrates:

* QuantumRenderer for state, circuit, and entanglement visuals
* FieldPanel for scalar/vector/tensor dynamics  
* ObserverPanel for recursive observer networks
* MemoryFieldPhysics for entropy, strain, and coherence maps
* CoherenceRenderer for RSP and OSH substrate views
* Live metric overlays and multi-panel visualization orchestration

The system provides fail-soft architecture with visual fallback messages,
multi-renderer fusion capabilities, RSP-aware OSH metrics, and time-aware
history trend tracking with live event overlays.
"""

import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Rectangle
from matplotlib.collections import LineCollection
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any, Union
import base64
from io import BytesIO
from collections import defaultdict, deque
import json
import traceback

# Configure logging
logger = logging.getLogger(__name__)


class PhysicsRenderer:
    """
    Central rendering hub for all scientific visual panels in the Recursia OSH dashboard.
    
    Coordinates all Recursia renderers into grid panels, safely handles missing data
    or disabled modules, provides fallback visualizations with descriptions, and
    returns full success/error metadata and visual states.
    """
    
    def __init__(self,
                 quantum_renderer=None,
                 field_panel=None,
                 field_dynamics=None,
                 observer_panel=None,
                 observer_dynamics=None,
                 memory_field=None,
                 coherence_renderer=None,
                 entanglement_manager=None,
                 state=None,
                 current_colors=None,
                 current_metrics=None,
                 metrics_history=None,
                 error_manager=None):
        """Initialize the PhysicsRenderer with all available subsystems."""
        
        # Core renderers and subsystems
        self.quantum_renderer = quantum_renderer
        self.field_panel = field_panel
        self.field_dynamics = field_dynamics
        self.observer_panel = observer_panel
        self.observer_dynamics = observer_dynamics
        self.memory_field = memory_field
        self.coherence_renderer = coherence_renderer
        self.entanglement_manager = entanglement_manager
        
        # State and metrics
        self.state = state
        self.current_colors = current_colors or self._get_default_colors()
        self.current_metrics = current_metrics
        self.metrics_history = metrics_history or deque(maxlen=1000)
        self.error_manager = error_manager
        
        # Performance tracking
        self.render_times = defaultdict(list)
        self.fallback_counts = defaultdict(int)
        
        # Visualization cache
        self.visualization_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # OSH computation engine
        self._initialize_osh_calculators()
        
        logger.info("PhysicsRenderer initialized with comprehensive subsystem integration")
    
    def _get_default_colors(self) -> Dict[str, str]:
        """Get default color scheme for visualizations."""
        return {
            'background': '#0a0a0a',
            'foreground': '#ffffff',
            'accent': '#00ff88',
            'warning': '#ffaa00',
            'error': '#ff4444',
            'coherence': '#00aaff',
            'entropy': '#ff6600',
            'strain': '#ff0066',
            'rsp': '#88ff00'
        }
    
    def _initialize_osh_calculators(self):
        """Initialize OSH metric calculation engines."""
        self.osh_calculators = {
            'rsp': self._calculate_recursive_simulation_potential,
            'phi': self._calculate_integrated_information,
            'emergence': self._calculate_emergence_index,
            'coherence_stability': self._calculate_coherence_stability,
            'entropy_flux': self._calculate_entropy_flux,
            'information_curvature': self._calculate_information_curvature
        }
    
    def render_quantum_visualization_panel(self, fig, gs_position) -> Dict:
        """
        Render comprehensive quantum visualization panel.
        
        Displays a 2×2 grid of:
        * Bloch Sphere
        * Probability Bar Graph  
        * Density Matrix Heatmap
        * Entanglement Network
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, visualizations, and statistics
        """
        start_time = time.time()
        
        try:
            # Create 2x2 subplot grid
            gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, gs_position, 
                                                    hspace=0.3, wspace=0.3)
            
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Get quantum states
            quantum_states = self._get_available_quantum_states()
            
            if not quantum_states:
                return self._render_quantum_placeholder(fig, gs_position)
            
            # Select primary state for single-state visualizations
            primary_state = quantum_states[0]
            
            # Render Bloch Sphere
            ax_bloch = fig.add_subplot(gs_sub[0, 0], projection='3d')
            bloch_result = self._render_bloch_sphere_component(ax_bloch, primary_state)
            result['visualizations']['bloch_sphere'] = bloch_result
            
            # Render Probability Distribution
            ax_prob = fig.add_subplot(gs_sub[0, 1])
            prob_result = self._render_probability_distribution_component(ax_prob, primary_state)
            result['visualizations']['probability'] = prob_result
            
            # Render Density Matrix
            ax_density = fig.add_subplot(gs_sub[1, 0])
            density_result = self._render_density_matrix_component(ax_density, primary_state)
            result['visualizations']['density_matrix'] = density_result
            
            # Render Entanglement Network
            ax_entangle = fig.add_subplot(gs_sub[1, 1])
            entangle_result = self._render_entanglement_network_component(ax_entangle, quantum_states)
            result['visualizations']['entanglement'] = entangle_result
            
            # Calculate comprehensive statistics
            result['statistics'] = self._calculate_quantum_panel_statistics(quantum_states)
            result['osh_metrics'] = self._calculate_quantum_osh_metrics(quantum_states)
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['quantum_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering quantum visualization panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['quantum_panel'] += 1
            return self._render_quantum_fallback(fig, gs_position, str(e))
    
    def render_field_dynamics_comprehensive_panel(self, fig, gs_position) -> Dict:
        """
        Render comprehensive field dynamics visualization panel.
        
        Renders:
        * Field Amplitudes
        * Coherence Map
        * Entropy Map  
        * Time Evolution (5 steps, wave_equation)
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, field analysis, and OSH metrics
        """
        start_time = time.time()
        
        try:
            # Create 2x2 subplot grid
            gs_sub = gridspec.GridSpecFromSubplotSpec(2, 2, gs_position,
                                                    hspace=0.3, wspace=0.3)
            
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Get available fields
            available_fields = self._get_available_fields()
            
            if not available_fields:
                return self._render_field_placeholder(fig, gs_position)
            
            # Select primary field
            primary_field = available_fields[0]
            field_data = self._get_field_data(primary_field)
            
            # Render Field Amplitudes
            ax_amplitude = fig.add_subplot(gs_sub[0, 0])
            amplitude_result = self._render_field_amplitude_component(ax_amplitude, 
                                                                   primary_field, field_data)
            result['visualizations']['amplitude'] = amplitude_result
            
            # Render Coherence Map
            ax_coherence = fig.add_subplot(gs_sub[0, 1])
            coherence_result = self._render_field_coherence_component(ax_coherence, 
                                                                    primary_field, field_data)
            result['visualizations']['coherence'] = coherence_result
            
            # Render Entropy Map
            ax_entropy = fig.add_subplot(gs_sub[1, 0])
            entropy_result = self._render_field_entropy_component(ax_entropy, 
                                                               primary_field, field_data)
            result['visualizations']['entropy'] = entropy_result
            
            # Render Time Evolution
            ax_evolution = fig.add_subplot(gs_sub[1, 1])
            evolution_result = self._render_field_evolution_component(ax_evolution, 
                                                                    primary_field, field_data)
            result['visualizations']['evolution'] = evolution_result
            
            # Calculate comprehensive statistics
            result['statistics'] = self._calculate_field_panel_statistics(available_fields)
            result['osh_metrics'] = self._calculate_field_osh_metrics(available_fields)
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['field_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering field dynamics panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['field_panel'] += 1
            return self._render_field_fallback(fig, gs_position, str(e))
    
    def render_observer_network_comprehensive_panel(self, fig, gs_position) -> Dict:
        """
        Render comprehensive observer network visualization panel.
        
        Renders observer topology with:
        * Phase-colored observer nodes
        * Strength-weighted connections
        * Central consensus score
        * Labels for observer ID and phase
        * Legend for observer phases
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, network metrics, and OSH analysis
        """
        start_time = time.time()
        
        try:
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Get observer data
            observers = self._get_observer_data()
            
            if len(observers) < 2:
                return self._render_observer_placeholder(fig, gs_position)
            
            # Create main subplot
            ax = fig.add_subplot(gs_position)
            
            # Create network graph
            G = nx.Graph()
            
            # Add observer nodes
            node_colors = []
            node_sizes = []
            phase_colors = {
                'passive': '#666666',
                'active': '#00aaff', 
                'measuring': '#ffaa00',
                'analyzing': '#ff6600',
                'entangled': '#ff00aa',
                'collapsed': '#ff4444'
            }
            
            for obs_name, obs_data in observers.items():
                G.add_node(obs_name)
                phase = obs_data.get('phase', 'passive')
                coherence = obs_data.get('coherence', 0.5)
                
                node_colors.append(phase_colors.get(phase, '#666666'))
                node_sizes.append(200 + 800 * coherence)  # Size by coherence
            
            # Add relationship edges
            relationships = self._get_observer_relationships(observers)
            for (obs1, obs2), strength in relationships.items():
                if strength > 0.1:  # Only show significant relationships
                    G.add_edge(obs1, obs2, weight=strength)
            
            # Layout network
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw network
            # Draw edges with weight-based width
            edges = G.edges()
            edge_weights = [G[u][v]['weight'] for u, v in edges]
            edge_widths = [1 + 4 * w for w in edge_weights]
            
            nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                                 alpha=0.6, edge_color='#444444')
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, 
                                 node_size=node_sizes, alpha=0.8)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, 
                                  font_color='white', font_weight='bold')
            
            # Add consensus score
            consensus_score = self._calculate_observer_consensus(observers)
            ax.text(0.02, 0.98, f'Consensus: {consensus_score:.3f}', 
                   transform=ax.transAxes, fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                   color='white', verticalalignment='top')
            
            # Add phase legend
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, markersize=10, 
                                        label=phase.capitalize())
                             for phase, color in phase_colors.items()]
            ax.legend(handles=legend_elements, loc='upper right', 
                     frameon=True, fancybox=True, shadow=True)
            
            ax.set_title('Observer Network Topology', fontsize=14, fontweight='bold',
                        color='white')
            ax.set_facecolor('black')
            ax.axis('off')
            
            # Calculate statistics
            result['statistics'] = {
                'observer_count': len(observers),
                'relationship_count': len(relationships),
                'consensus_score': consensus_score,
                'network_density': nx.density(G),
                'average_clustering': nx.average_clustering(G) if len(G) > 2 else 0,
                'phase_distribution': self._get_phase_distribution(observers)
            }
            
            # Calculate OSH metrics
            result['osh_metrics'] = self._calculate_observer_osh_metrics(observers)
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['observer_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering observer network panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['observer_panel'] += 1
            return self._render_observer_fallback(fig, gs_position, str(e))
    
    def render_memory_field_comprehensive_panel(self, fig, gs_position) -> Dict:
        """
        Render comprehensive memory field visualization panel.
        
        Displays:
        * Memory strain heatmap
        * Coherence contours
        * Critical strain overlays
        * Defragmentation indicators
        * Information flow arrows (gradient of coherence)
        * Strain/entropy/defrag stat boxes
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, memory analysis, and OSH metrics
        """
        start_time = time.time()
        
        try:
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Get memory field data
            memory_data = self._get_memory_field_data()
            
            if not memory_data:
                return self._render_memory_placeholder(fig, gs_position)
            
            # Create main subplot
            ax = fig.add_subplot(gs_position)
            
            # Create 2D memory field representation
            strain_field = self._create_memory_strain_field(memory_data)
            coherence_field = self._create_memory_coherence_field(memory_data)
            
            # Render strain heatmap
            im_strain = ax.imshow(strain_field, cmap='hot_r', alpha=0.8, 
                                extent=[0, 10, 0, 10], origin='lower')
            
            # Add coherence contours
            x = np.linspace(0, 10, coherence_field.shape[1])
            y = np.linspace(0, 10, coherence_field.shape[0])
            X, Y = np.meshgrid(x, y)
            
            contours = ax.contour(X, Y, coherence_field, levels=5, 
                                colors='cyan', alpha=0.6, linewidths=1.5)
            ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
            
            # Highlight critical strain regions
            critical_mask = strain_field > 0.8
            if np.any(critical_mask):
                ax.contour(X, Y, critical_mask.astype(float), levels=[0.5], 
                          colors='red', linewidths=3, alpha=0.8)
            
            # Add information flow arrows (coherence gradient)
            grad_y, grad_x = np.gradient(coherence_field)
            step = 2
            ax.quiver(X[::step, ::step], Y[::step, ::step], 
                     grad_x[::step, ::step], grad_y[::step, ::step],
                     scale=10, alpha=0.6, color='white', width=0.003)
            
            # Add defragmentation indicators
            defrag_regions = self._get_defragmentation_regions(memory_data)
            for region in defrag_regions:
                circle = Circle((region['x'], region['y']), region['radius'], 
                              fill=False, color='lime', linewidth=2, alpha=0.8)
                ax.add_patch(circle)
            
            # Add statistics boxes
            stats = self._calculate_memory_field_statistics(memory_data)
            
            # Strain statistics box
            strain_text = f"Strain\nAvg: {stats['avg_strain']:.3f}\nMax: {stats['max_strain']:.3f}"
            ax.text(0.02, 0.98, strain_text, transform=ax.transAxes, 
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='red', alpha=0.7),
                   color='white', verticalalignment='top')
            
            # Coherence statistics box  
            coherence_text = f"Coherence\nAvg: {stats['avg_coherence']:.3f}\nStd: {stats['coherence_std']:.3f}"
            ax.text(0.02, 0.78, coherence_text, transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7),
                   color='white', verticalalignment='top')
            
            # Defragmentation statistics box
            defrag_text = f"Defrag\nRegions: {len(defrag_regions)}\nEvents: {stats['defrag_events']}"
            ax.text(0.02, 0.58, defrag_text, transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='green', alpha=0.7),
                   color='white', verticalalignment='top')
            
            # Colorbar for strain
            cbar = plt.colorbar(im_strain, ax=ax, shrink=0.8)
            cbar.set_label('Memory Strain', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            ax.set_title('Memory Field Dynamics', fontsize=14, fontweight='bold', color='white')
            ax.set_xlabel('Memory Space X', color='white')
            ax.set_ylabel('Memory Space Y', color='white')
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            
            result['statistics'] = stats
            result['osh_metrics'] = self._calculate_memory_osh_metrics(memory_data)
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['memory_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering memory field panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['memory_panel'] += 1
            return self._render_memory_fallback(fig, gs_position, str(e))
    
    def render_osh_substrate_comprehensive_panel(self, fig, gs_position) -> Dict:
        """
        Render comprehensive OSH substrate visualization panel.
        
        Visualizes Recursive Simulation Potential (RSP) via:
        * Composite RSP heatmap
        * Coherence contour overlays
        * High RSP mask regions
        * Colorbar with units
        * RSP status classification (Exceptional → Critical)
        * Overlay with C, H, S, RSP metrics
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, RSP analysis, and OSH validation metrics
        """
        start_time = time.time()
        
        try:
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Try coherence renderer first
            if self.coherence_renderer:
                try:
                    coherence_result = self.coherence_renderer.render_rsp_landscape_analysis(
                        width=800, height=600
                    )
                    if coherence_result.get('success'):
                        # Parse base64 image and embed in subplot
                        ax = fig.add_subplot(gs_position)
                        self._embed_coherence_renderer_result(ax, coherence_result)
                        
                        result['statistics'] = coherence_result.get('statistics', {})
                        result['osh_metrics'] = coherence_result.get('osh_metrics', {})
                        
                        render_time = time.time() - start_time
                        self.render_times['osh_panel'].append(render_time)
                        result['render_time'] = render_time
                        
                        return result
                except Exception as e:
                    logger.warning(f"Coherence renderer failed, using fallback: {e}")
            
            # Fallback to manual RSP visualization
            ax = fig.add_subplot(gs_position)
            
            # Create RSP field
            rsp_field = self._create_rsp_field()
            coherence_field = self._create_coherence_field()
            entropy_field = self._create_entropy_field() 
            strain_field = self._create_strain_field()
            
            # Render RSP heatmap
            im_rsp = ax.imshow(rsp_field, cmap='coolwarm', alpha=0.8,
                             extent=[0, 10, 0, 10], origin='lower')
            
            # Add coherence contours
            x = np.linspace(0, 10, coherence_field.shape[1])
            y = np.linspace(0, 10, coherence_field.shape[0])
            X, Y = np.meshgrid(x, y)
            
            contours = ax.contour(X, Y, coherence_field, levels=6,
                                colors='black', alpha=0.5, linewidths=1)
            ax.clabel(contours, inline=True, fontsize=8, fmt='C=%.2f')
            
            # Highlight high RSP regions
            high_rsp_mask = rsp_field > np.percentile(rsp_field, 80)
            if np.any(high_rsp_mask):
                ax.contour(X, Y, high_rsp_mask.astype(float), levels=[0.5],
                          colors='yellow', linewidths=2, alpha=0.9)
            
            # Add OSH metrics overlay
            rsp_avg = np.mean(rsp_field)
            coherence_avg = np.mean(coherence_field)
            entropy_avg = np.mean(entropy_field)
            strain_avg = np.mean(strain_field)
            
            # RSP classification
            rsp_class = self._classify_rsp_level(rsp_avg)
            
            # Metrics text box
            metrics_text = (f"RSP: {rsp_avg:.3f} ({rsp_class})\n"
                          f"C: {coherence_avg:.3f}\n"
                          f"H: {entropy_avg:.3f}\n" 
                          f"S: {strain_avg:.3f}")
            
            ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                   color='white', verticalalignment='top', horizontalalignment='right')
            
            # Add emergence indicators
            emergence_points = self._detect_emergence_points(rsp_field, coherence_field)
            for point in emergence_points:
                ax.scatter(point[0], point[1], s=100, c='lime', marker='*', 
                          alpha=0.8, edgecolors='black', linewidth=2)
            
            # Colorbar
            cbar = plt.colorbar(im_rsp, ax=ax, shrink=0.8)
            cbar.set_label('Recursive Simulation Potential', color='white')
            cbar.ax.yaxis.set_tick_params(color='white')
            plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
            
            ax.set_title('OSH Substrate Analysis', fontsize=14, fontweight='bold', color='white')
            ax.set_xlabel('Information Space X', color='white')
            ax.set_ylabel('Information Space Y', color='white')
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            
            # Calculate comprehensive statistics
            result['statistics'] = {
                'rsp_mean': rsp_avg,
                'rsp_std': np.std(rsp_field),
                'rsp_classification': rsp_class,
                'coherence_mean': coherence_avg,
                'entropy_mean': entropy_avg, 
                'strain_mean': strain_avg,
                'emergence_points': len(emergence_points),
                'high_rsp_coverage': np.sum(high_rsp_mask) / high_rsp_mask.size
            }
            
            # Calculate OSH metrics
            result['osh_metrics'] = {
                'rsp': rsp_avg,
                'phi': self.osh_calculators['phi'](coherence_avg, entropy_avg),
                'emergence_index': self.osh_calculators['emergence'](coherence_field, entropy_field),
                'information_curvature': self.osh_calculators['information_curvature'](rsp_field),
                'substrate_stability': self._calculate_substrate_stability(rsp_field),
                'osh_validation_score': self._calculate_osh_validation_score(result['statistics'])
            }
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['osh_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering OSH substrate panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['osh_panel'] += 1
            return self._render_osh_fallback(fig, gs_position, str(e))
    
    def render_time_evolution_comprehensive_panel(self, fig, gs_position) -> Dict:
        """
        Render comprehensive time evolution visualization panel.
        
        Plots real-time simulation evolution from metrics_history:
        * Primary axis: Coherence, Entropy, Strain
        * Secondary axis: RSP
        * Overlays: quantum states (normalized), observers, stability/critical zones
        * Event markers: collapses, recursive boundary crossings
        * Moving averages (trend detection)
        * Annotated emergent phenomena
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, temporal analysis, and trend statistics
        """
        start_time = time.time()
        
        try:
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Check for sufficient history
            if not self.metrics_history or len(self.metrics_history) < 2:
                return self._render_time_placeholder(fig, gs_position)
            
            # Create main subplot with secondary y-axis
            ax1 = fig.add_subplot(gs_position)
            ax2 = ax1.twinx()
            
            # Extract time series data
            time_data = []
            coherence_data = []
            entropy_data = []
            strain_data = []
            rsp_data = []
            quantum_states_data = []
            observers_data = []
            
            for i, metrics in enumerate(self.metrics_history):
                time_data.append(i)
                coherence_data.append(getattr(metrics, 'coherence', 0))
                entropy_data.append(getattr(metrics, 'entropy', 0))
                strain_data.append(getattr(metrics, 'strain', 0))
                rsp_data.append(getattr(metrics, 'rsp', 0))
                quantum_states_data.append(getattr(metrics, 'quantum_states_count', 0))
                observers_data.append(getattr(metrics, 'observer_count', 0))
            
            # Convert to numpy arrays
            time_data = np.array(time_data)
            coherence_data = np.array(coherence_data)
            entropy_data = np.array(entropy_data)
            strain_data = np.array(strain_data)
            rsp_data = np.array(rsp_data)
            quantum_states_data = np.array(quantum_states_data)
            observers_data = np.array(observers_data)
            
            # Plot primary metrics
            line1 = ax1.plot(time_data, coherence_data, 'c-', linewidth=2, 
                           label='Coherence', alpha=0.8)
            line2 = ax1.plot(time_data, entropy_data, 'r-', linewidth=2, 
                           label='Entropy', alpha=0.8)
            line3 = ax1.plot(time_data, strain_data, 'm-', linewidth=2, 
                           label='Strain', alpha=0.8)
            
            # Plot RSP on secondary axis
            line4 = ax2.plot(time_data, rsp_data, 'g-', linewidth=3, 
                           label='RSP', alpha=0.9)
            
            # Add normalized quantum states and observers
            if np.max(quantum_states_data) > 0:
                normalized_states = quantum_states_data / np.max(quantum_states_data)
                ax1.plot(time_data, normalized_states, 'b--', alpha=0.6, 
                        label='States (norm)')
            
            if np.max(observers_data) > 0:
                normalized_observers = observers_data / np.max(observers_data)
                ax1.plot(time_data, normalized_observers, 'y--', alpha=0.6,
                        label='Observers (norm)')
            
            # Add moving averages for trend detection
            if len(time_data) >= 10:
                window = min(10, len(time_data) // 3)
                coherence_ma = self._moving_average(coherence_data, window)
                entropy_ma = self._moving_average(entropy_data, window)
                
                ax1.plot(time_data[window-1:], coherence_ma, 'c:', linewidth=3, alpha=0.7)
                ax1.plot(time_data[window-1:], entropy_ma, 'r:', linewidth=3, alpha=0.7)
            
            # Add stability zones
            ax1.axhspan(0.7, 1.0, alpha=0.1, color='green', label='Stable Zone')
            ax1.axhspan(0.0, 0.3, alpha=0.1, color='red', label='Critical Zone')
            
            # Mark significant events
            self._mark_significant_events(ax1, time_data, coherence_data, entropy_data)
            
            # Add emergent phenomena annotations
            phenomena_events = self._detect_phenomena_events(time_data, coherence_data, 
                                                           entropy_data, rsp_data)
            for event in phenomena_events:
                ax1.annotate(event['type'], xy=(event['time'], event['value']),
                           xytext=(event['time'], event['value'] + 0.1),
                           arrowprops=dict(arrowstyle='->', color='yellow', alpha=0.8),
                           fontsize=8, color='yellow', fontweight='bold')
            
            # Formatting
            ax1.set_xlabel('Time Steps', color='white')
            ax1.set_ylabel('Primary Metrics', color='white')
            ax2.set_ylabel('RSP', color='green')
            ax1.set_title('Temporal Evolution Analysis', fontsize=14, fontweight='bold', color='white')
            
            # Legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', 
                      frameon=True, fancybox=True, shadow=True)
            
            # Styling
            ax1.set_facecolor('black')
            ax1.tick_params(colors='white')
            ax2.tick_params(colors='green')
            ax1.grid(True, alpha=0.3)
            
            # Calculate trend statistics
            result['statistics'] = {
                'coherence_trend': self._calculate_trend(coherence_data),
                'entropy_trend': self._calculate_trend(entropy_data),
                'strain_trend': self._calculate_trend(strain_data),
                'rsp_trend': self._calculate_trend(rsp_data),
                'phenomena_events': len(phenomena_events),
                'stability_score': self._calculate_stability_score(coherence_data, entropy_data),
                'temporal_complexity': self._calculate_temporal_complexity(time_data, 
                                                                        [coherence_data, entropy_data, rsp_data])
            }
            
            # Calculate OSH temporal metrics
            result['osh_metrics'] = {
                'temporal_coherence': np.mean(coherence_data),
                'entropy_flux': self.osh_calculators['entropy_flux'](entropy_data),
                'rsp_stability': self.osh_calculators['coherence_stability'](rsp_data),
                'emergence_events': len([e for e in phenomena_events if 'emergence' in e['type']]),
                'phase_transitions': len([e for e in phenomena_events if 'transition' in e['type']]),
                'temporal_prediction': self._predict_next_values(coherence_data, entropy_data, rsp_data)
            }
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['time_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering time evolution panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['time_panel'] += 1
            return self._render_time_fallback(fig, gs_position, str(e))
    
    def render_quantum_entanglement_topology_panel(self, fig, gs_position) -> Dict:
        """
        Render quantum entanglement topology visualization panel.
        
        Generates an entanglement topology graph:
        * Nodes: Quantum states sized by coherence
        * Edges: Weighted by entanglement strength
        * Node labels and color-coded coherence
        * Edge legend for entanglement tiers
        * Returns network metrics (connectivity, entangled pairs, etc.)
        
        Args:
            fig: Matplotlib figure object
            gs_position: GridSpec position for the panel
            
        Returns:
            Dict with success status, topology metrics, and entanglement analysis
        """
        start_time = time.time()
        
        try:
            result = {
                'success': True,
                'visualizations': {},
                'statistics': {},
                'osh_metrics': {}
            }
            
            # Get quantum states and entanglement data
            quantum_states = self._get_available_quantum_states()
            
            if len(quantum_states) < 2:
                return self._render_entanglement_placeholder(fig, gs_position)
            
            # Create main subplot
            ax = fig.add_subplot(gs_position)
            
            # Create network graph
            G = nx.Graph()
            
            # Add quantum state nodes
            node_colors = []
            node_sizes = []
            coherence_values = []
            
            for state_name in quantum_states:
                state_data = self._get_quantum_state_data(state_name)
                coherence = state_data.get('coherence', 0.5)
                
                G.add_node(state_name)
                coherence_values.append(coherence)
                node_sizes.append(300 + 700 * coherence)
            
            # Normalize coherence for color mapping
            if coherence_values:
                coherence_min = min(coherence_values)
                coherence_max = max(coherence_values)
                coherence_range = coherence_max - coherence_min
                
                for coherence in coherence_values:
                    if coherence_range > 0:
                        normalized = (coherence - coherence_min) / coherence_range
                    else:
                        normalized = 0.5
                    node_colors.append(plt.cm.viridis(normalized))
                    
            # Add entanglement edges
            entanglement_data = self._get_entanglement_relationships(quantum_states)
            edge_weights = []
            edge_colors = []
            
            for (state1, state2), strength in entanglement_data.items():
                if strength > 0.05:  # Only show significant entanglement
                    G.add_edge(state1, state2, weight=strength)
                    edge_weights.append(1 + 8 * strength)
                    
                    # Color edges by strength
                    if strength > 0.8:
                        edge_colors.append('red')      # Strong entanglement
                    elif strength > 0.5:
                        edge_colors.append('orange')   # Moderate entanglement
                    else:
                        edge_colors.append('yellow')   # Weak entanglement
            
            # Layout network
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=3, iterations=100)
                
                # Draw edges
                if len(G.edges()) > 0:
                    nx.draw_networkx_edges(G, pos, ax=ax, width=edge_weights,
                                         edge_color=edge_colors, alpha=0.7)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                     node_size=node_sizes, alpha=0.9)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=10,
                                      font_color='white', font_weight='bold')
                
                # Add entanglement strength legend
                legend_elements = [
                    plt.Line2D([0], [0], color='red', lw=4, label='Strong (>0.8)'),
                    plt.Line2D([0], [0], color='orange', lw=4, label='Moderate (0.5-0.8)'),
                    plt.Line2D([0], [0], color='yellow', lw=4, label='Weak (0.05-0.5)')
                ]
                ax.legend(handles=legend_elements, loc='upper right',
                         title='Entanglement Strength', title_fontsize=10,
                         frameon=True, fancybox=True, shadow=True)
                
                # Add network statistics
                stats_text = (f"States: {len(quantum_states)}\n"
                            f"Entangled Pairs: {len(entanglement_data)}\n"
                            f"Density: {nx.density(G):.3f}\n"
                            f"Avg Coherence: {np.mean(coherence_values):.3f}")
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=10, bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
                       color='white', verticalalignment='top')
            
            ax.set_title('Quantum Entanglement Topology', fontsize=14, fontweight='bold', color='white')
            ax.set_facecolor('black')
            ax.axis('off')
            
            # Calculate network metrics
            if len(G.nodes()) > 0:
                result['statistics'] = {
                    'state_count': len(quantum_states),
                    'entangled_pairs': len(entanglement_data),
                    'network_density': nx.density(G),
                    'average_clustering': nx.average_clustering(G) if len(G) > 2 else 0,
                    'connectivity': nx.is_connected(G),
                    'average_coherence': np.mean(coherence_values),
                    'coherence_std': np.std(coherence_values),
                    'max_entanglement': max(entanglement_data.values()) if entanglement_data else 0,
                    'entanglement_distribution': self._analyze_entanglement_distribution(entanglement_data)
                }
            else:
                result['statistics'] = {'state_count': 0, 'entangled_pairs': 0}
            
            # Calculate OSH entanglement metrics
            result['osh_metrics'] = {
                'entanglement_complexity': self._calculate_entanglement_complexity(G, entanglement_data),
                'quantum_discord': self._estimate_quantum_discord(entanglement_data, coherence_values),
                'entanglement_entropy': self._calculate_entanglement_entropy(entanglement_data),
                'network_emergence': self._calculate_network_emergence(G, coherence_values),
                'topology_stability': self._calculate_topology_stability(entanglement_data),
                'quantum_coherence_flow': self._calculate_coherence_flow(G, coherence_values, entanglement_data)
            }
            
            # Performance tracking
            render_time = time.time() - start_time
            self.render_times['entanglement_panel'].append(render_time)
            result['render_time'] = render_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error rendering entanglement topology panel: {e}")
            logger.error(traceback.format_exc())
            self.fallback_counts['entanglement_panel'] += 1
            return self._render_entanglement_fallback(fig, gs_position, str(e))
    
    # OSH Metric Calculation Methods
    
    def _calculate_recursive_simulation_potential(self, coherence: float, entropy: float, 
                                                strain: float = 0.1) -> float:
        """Calculate Recursive Simulation Potential (RSP)."""
        try:
            epsilon = 1e-6
            integrated_info = coherence * (1 - entropy) * np.log(1 + coherence)
            kolmogorov_approx = -entropy * np.log(entropy + epsilon)
            entropy_flux = strain + epsilon
            
            rsp = (integrated_info * kolmogorov_approx) / entropy_flux
            return max(0, rsp)
        except Exception as e:
            logger.warning(f"Error calculating RSP: {e}")
            return 0.0
    
    def _calculate_integrated_information(self, coherence: float, entropy: float) -> float:
        """Calculate integrated information (Φ) approximation."""
        try:
            if coherence <= 0:
                return 0.0
            
            phi = coherence * (1 - entropy) * np.log(1 + coherence)
            return max(0, phi)
        except Exception as e:
            logger.warning(f"Error calculating integrated information: {e}")
            return 0.0
    
    def _calculate_emergence_index(self, coherence_field: np.ndarray, 
                                 entropy_field: np.ndarray) -> float:
        """Calculate emergence index from field data."""
        try:
            coherence_var = np.var(coherence_field)
            entropy_corr = np.corrcoef(coherence_field.flatten(), 
                                     entropy_field.flatten())[0, 1]
            
            if np.isnan(entropy_corr):
                entropy_corr = 0
            
            emergence = coherence_var * (1 - abs(entropy_corr))
            return max(0, min(1, emergence))
        except Exception as e:
            logger.warning(f"Error calculating emergence index: {e}")
            return 0.0
    
    def _calculate_coherence_stability(self, values: np.ndarray) -> float:
        """Calculate coherence stability metric."""
        try:
            if len(values) < 2:
                return 0.0
            
            mean_val = np.mean(values)
            if mean_val == 0:
                return 0.0
            
            cv = np.std(values) / mean_val
            stability = 1 / (1 + cv)
            return max(0, min(1, stability))
        except Exception as e:
            logger.warning(f"Error calculating coherence stability: {e}")
            return 0.0
    
    def _calculate_entropy_flux(self, entropy_values: np.ndarray) -> float:
        """Calculate entropy flux rate."""
        try:
            if len(entropy_values) < 2:
                return 0.0
            
            diff = np.diff(entropy_values)
            flux = np.mean(np.abs(diff))
            return max(0, flux)
        except Exception as e:
            logger.warning(f"Error calculating entropy flux: {e}")
            return 0.0
    
    def _calculate_information_curvature(self, field: np.ndarray) -> float:
        """Calculate information geometry curvature."""
        try:
            # Calculate second derivatives (curvature approximation)
            grad_y, grad_x = np.gradient(field)
            curvature_x = np.gradient(grad_x, axis=1)
            curvature_y = np.gradient(grad_y, axis=0)
            
            # Scalar curvature (trace)
            scalar_curvature = np.mean(curvature_x + curvature_y)
            return scalar_curvature
        except Exception as e:
            logger.warning(f"Error calculating information curvature: {e}")
            return 0.0
    
    # Utility Methods for Data Access
    
    def _get_available_quantum_states(self) -> List[str]:
        """Get list of available quantum states."""
        try:
            if self.state and hasattr(self.state, 'quantum_states'):
                return list(self.state.quantum_states.keys())
            elif self.quantum_renderer and hasattr(self.quantum_renderer, 'state_registry'):
                return list(self.quantum_renderer.state_registry.get_all_states().keys())
            else:
                return []
        except Exception as e:
            logger.warning(f"Error getting quantum states: {e}")
            return []
    
    def _get_available_fields(self) -> List[str]:
        """Get list of available fields."""
        try:
            if self.field_dynamics and hasattr(self.field_dynamics, 'fields'):
                return list(self.field_dynamics.fields.keys())
            elif self.field_panel and hasattr(self.field_panel, 'get_available_fields'):
                return self.field_panel.get_available_fields()
            else:
                return []
        except Exception as e:
            logger.warning(f"Error getting fields: {e}")
            return []
    
    def _get_observer_data(self) -> Dict:
        """Get observer data."""
        try:
            if self.observer_dynamics and hasattr(self.observer_dynamics, 'observers'):
                return self.observer_dynamics.observers
            elif self.state and hasattr(self.state, 'observers'):
                return self.state.observers
            else:
                return {}
        except Exception as e:
            logger.warning(f"Error getting observer data: {e}")
            return {}
    
    def _get_memory_field_data(self) -> Dict:
        """Get memory field data."""
        try:
            if self.memory_field:
                return {
                    'regions': getattr(self.memory_field, 'memory_regions', {}),
                    'strain': getattr(self.memory_field, 'memory_strain', {}),
                    'coherence': getattr(self.memory_field, 'memory_coherence', {}),
                    'connectivity': getattr(self.memory_field, 'region_connectivity', {})
                }
            else:
                return {}
        except Exception as e:
            logger.warning(f"Error getting memory field data: {e}")
            return {}
    
    # Placeholder and Fallback Rendering Methods
    
    def _render_quantum_placeholder(self, fig, gs_position) -> Dict:
        """Render placeholder for quantum visualization."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, 'Quantum Visualization\n\nNo quantum states available\nor renderer not initialized',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': True, 'status': 'placeholder'}
    
    def _render_field_placeholder(self, fig, gs_position) -> Dict:
        """Render placeholder for field visualization."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, 'Field Dynamics\n\nNo fields available\nor field system not initialized',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': True, 'status': 'placeholder'}
    
    def _render_observer_placeholder(self, fig, gs_position) -> Dict:
        """Render placeholder for observer visualization."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, 'Observer Network\n\nInsufficient observers\nor observer system not initialized',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': True, 'status': 'placeholder'}
    
    def _render_memory_placeholder(self, fig, gs_position) -> Dict:
        """Render placeholder for memory field visualization."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, 'Memory Field\n\nNo memory field data\nor memory system not initialized',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': True, 'status': 'placeholder'}
    
    def _render_time_placeholder(self, fig, gs_position) -> Dict:
        """Render placeholder for time evolution visualization."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, 'Time Evolution\n\nInsufficient history data\nfor temporal analysis',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': True, 'status': 'placeholder'}
    
    def _render_entanglement_placeholder(self, fig, gs_position) -> Dict:
        """Render placeholder for entanglement topology visualization."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, 'Entanglement Topology\n\nInsufficient quantum states\nfor topology analysis',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=12, color='white', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.7))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': True, 'status': 'placeholder'}
    
    # Component-level rendering methods
    
    def _render_bloch_sphere_component(self, ax, state_name: str) -> Dict:
        """Render Bloch sphere component for a single quantum state."""
        try:
            state_data = self._get_quantum_state_data(state_name)
            state_vector = state_data.get('state_vector', np.array([1, 0]))
            
            if len(state_vector) >= 2:
                # Calculate Bloch vector coordinates
                alpha, beta = state_vector[0], state_vector[1]
                
                # Bloch sphere coordinates
                x = 2 * np.real(np.conj(alpha) * beta)
                y = 2 * np.imag(np.conj(alpha) * beta)
                z = np.abs(alpha)**2 - np.abs(beta)**2
                
                # Draw sphere wireframe
                u = np.linspace(0, 2 * np.pi, 50)
                v = np.linspace(0, np.pi, 50)
                sphere_x = np.outer(np.cos(u), np.sin(v))
                sphere_y = np.outer(np.sin(u), np.sin(v))
                sphere_z = np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_wireframe(sphere_x, sphere_y, sphere_z, alpha=0.1, color='white')
                
                # Draw Bloch vector
                ax.quiver(0, 0, 0, x, y, z, color='red', arrow_length_ratio=0.1, linewidth=3)
                
                # Add coordinate axes
                ax.plot([0, 1.2], [0, 0], [0, 0], 'b-', alpha=0.6)
                ax.plot([0, 0], [0, 1.2], [0, 0], 'g-', alpha=0.6)
                ax.plot([0, 0], [0, 0], [0, 1.2], 'r-', alpha=0.6)
                
                # Labels
                ax.text(1.3, 0, 0, 'X', color='blue')
                ax.text(0, 1.3, 0, 'Y', color='green')
                ax.text(0, 0, 1.3, 'Z', color='red')
                
                # State labels
                ax.text(0, 0, 1.2, '|0⟩', color='white', fontsize=10)
                ax.text(0, 0, -1.2, '|1⟩', color='white', fontsize=10)
                
                ax.set_title(f'Bloch Sphere - {state_name}', color='white', fontsize=10)
                ax.set_facecolor('black')
                
                return {
                    'success': True,
                    'bloch_vector': [x, y, z],
                    'coherence': state_data.get('coherence', 0),
                    'purity': np.abs(alpha)**2 + np.abs(beta)**2
                }
            else:
                ax.text(0.5, 0.5, 'Invalid State Vector', ha='center', va='center',
                       transform=ax.transAxes, color='red')
                return {'success': False, 'error': 'Invalid state vector'}
                
        except Exception as e:
            logger.error(f"Error rendering Bloch sphere: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_probability_distribution_component(self, ax, state_name: str) -> Dict:
        """Render probability distribution component for a quantum state."""
        try:
            state_data = self._get_quantum_state_data(state_name)
            state_vector = state_data.get('state_vector', np.array([1, 0]))
            
            # Calculate probabilities
            probabilities = np.abs(state_vector)**2
            n_qubits = int(np.log2(len(probabilities)))
            
            # Create basis state labels
            basis_states = [format(i, f'0{n_qubits}b') for i in range(len(probabilities))]
            
            # Bar plot
            bars = ax.bar(range(len(probabilities)), probabilities, 
                         color=self.current_colors['accent'], alpha=0.7)
            
            # Highlight maximum probability
            max_idx = np.argmax(probabilities)
            bars[max_idx].set_color('red')
            bars[max_idx].set_alpha(0.9)
            
            ax.set_xlabel('Basis States', color='white')
            ax.set_ylabel('Probability', color='white')
            ax.set_title(f'State Probabilities - {state_name}', color='white', fontsize=10)
            ax.set_xticks(range(len(probabilities)))
            ax.set_xticklabels([f'|{state}⟩' for state in basis_states], rotation=45)
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
            return {
                'success': True,
                'probabilities': probabilities.tolist(),
                'max_probability': float(np.max(probabilities)),
                'entropy': -np.sum(probabilities * np.log2(probabilities + 1e-10))
            }
            
        except Exception as e:
            logger.error(f"Error rendering probability distribution: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_density_matrix_component(self, ax, state_name: str) -> Dict:
        """Render density matrix component for a quantum state."""
        try:
            state_data = self._get_quantum_state_data(state_name)
            density_matrix = state_data.get('density_matrix')
            
            if density_matrix is None:
                # Calculate from state vector
                state_vector = state_data.get('state_vector', np.array([1, 0]))
                density_matrix = np.outer(state_vector, np.conj(state_vector))
            
            # Plot magnitude
            im = ax.imshow(np.abs(density_matrix), cmap='viridis', interpolation='nearest')
            
            # Add contours for phase information
            phase_matrix = np.angle(density_matrix)
            contours = ax.contour(phase_matrix, levels=8, colors='white', alpha=0.5, linewidths=0.5)
            
            ax.set_title(f'Density Matrix - {state_name}', color='white', fontsize=10)
            ax.set_xlabel('Qubit Index', color='white')
            ax.set_ylabel('Qubit Index', color='white')
            ax.tick_params(colors='white')
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(colors='white')
            cbar.ax.yaxis.label.set_color('white')
            
            return {
                'success': True,
                'purity': float(np.trace(density_matrix @ density_matrix).real),
                'trace': float(np.trace(density_matrix).real),
                'rank': np.linalg.matrix_rank(density_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error rendering density matrix: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_entanglement_network_component(self, ax, quantum_states: List[str]) -> Dict:
        """Render entanglement network component."""
        try:
            if len(quantum_states) < 2:
                ax.text(0.5, 0.5, 'Need ≥2 states\nfor entanglement', ha='center', va='center',
                       transform=ax.transAxes, color='white')
                return {'success': False, 'error': 'Insufficient states'}
            
            # Create demo entanglement network
            G = nx.Graph()
            
            # Add nodes
            for state in quantum_states:
                G.add_node(state)
            
            # Add demo edges (in real implementation, get from entanglement manager)
            entanglement_strengths = {}
            for i, state1 in enumerate(quantum_states):
                for j, state2 in enumerate(quantum_states[i+1:], i+1):
                    # Demo entanglement calculation
                    strength = np.random.beta(2, 5)  # Realistic distribution
                    if strength > 0.2:
                        G.add_edge(state1, state2, weight=strength)
                        entanglement_strengths[(state1, state2)] = strength
            
            if len(G.edges()) > 0:
                # Layout
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Draw edges
                edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
                edge_colors = ['red' if w > 0.7 else 'orange' if w > 0.4 else 'yellow' 
                              for w in edge_weights]
                edge_widths = [1 + 5*w for w in edge_weights]
                
                nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, 
                                     edge_color=edge_colors, alpha=0.8)
                
                # Draw nodes
                node_colors = [self.current_colors['coherence']] * len(quantum_states)
                nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                                     node_size=500, alpha=0.9)
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=8,
                                      font_color='white', font_weight='bold')
                
                ax.set_title('Entanglement Network', color='white', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No significant\nentanglement detected', 
                       ha='center', va='center', transform=ax.transAxes, color='white')
            
            ax.set_facecolor('black')
            ax.axis('off')
            
            return {
                'success': True,
                'entangled_pairs': len(entanglement_strengths),
                'max_entanglement': max(entanglement_strengths.values()) if entanglement_strengths else 0,
                'network_density': nx.density(G)
            }
            
        except Exception as e:
            logger.error(f"Error rendering entanglement network: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_field_amplitude_component(self, ax, field_name: str, field_data: np.ndarray) -> Dict:
        """Render field amplitude component."""
        try:
            if field_data.ndim == 1:
                # 1D field
                ax.plot(field_data, color=self.current_colors['accent'], linewidth=2)
                ax.set_title(f'Field Amplitude - {field_name}', color='white', fontsize=10)
                ax.set_xlabel('Position', color='white')
                ax.set_ylabel('Amplitude', color='white')
            elif field_data.ndim == 2:
                # 2D field
                im = ax.imshow(field_data, cmap='viridis', origin='lower')
                ax.set_title(f'Field Amplitude - {field_name}', color='white', fontsize=10)
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                # Higher dimensional - show slice
                field_slice = field_data.mean(axis=tuple(range(2, field_data.ndim)))
                im = ax.imshow(field_slice, cmap='viridis', origin='lower')
                ax.set_title(f'Field Amplitude (slice) - {field_name}', color='white', fontsize=10)
                plt.colorbar(im, ax=ax, shrink=0.8)
            
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.3)
            
            return {
                'success': True,
                'amplitude_stats': {
                    'mean': float(np.mean(field_data)),
                    'std': float(np.std(field_data)),
                    'max': float(np.max(field_data)),
                    'min': float(np.min(field_data))
                }
            }
            
        except Exception as e:
            logger.error(f"Error rendering field amplitude: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_field_coherence_component(self, ax, field_name: str, field_data: np.ndarray) -> Dict:
        """Render field coherence component."""
        try:
            # Calculate coherence field (simplified)
            if field_data.dtype == complex:
                coherence_field = np.abs(field_data)
            else:
                # For real fields, use local variation as coherence proxy
                if field_data.ndim >= 2:
                    grad_y, grad_x = np.gradient(field_data)
                    coherence_field = 1 / (1 + np.sqrt(grad_x**2 + grad_y**2))
                else:
                    grad = np.gradient(field_data)
                    coherence_field = 1 / (1 + np.abs(grad))
            
            if coherence_field.ndim == 2:
                im = ax.imshow(coherence_field, cmap='plasma', origin='lower')
                contours = ax.contour(coherence_field, levels=5, colors='white', alpha=0.6)
                ax.clabel(contours, inline=True, fontsize=8)
                plt.colorbar(im, ax=ax, shrink=0.8, label='Coherence')
            else:
                ax.plot(coherence_field, color='orange', linewidth=2)
                ax.set_ylabel('Coherence', color='white')
            
            ax.set_title(f'Field Coherence - {field_name}', color='white', fontsize=10)
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            
            return {
                'success': True,
                'coherence_stats': {
                    'mean': float(np.mean(coherence_field)),
                    'std': float(np.std(coherence_field)),
                    'max': float(np.max(coherence_field))
                }
            }
            
        except Exception as e:
            logger.error(f"Error rendering field coherence: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_field_entropy_component(self, ax, field_name: str, field_data: np.ndarray) -> Dict:
        """Render field entropy component."""
        try:
            # Calculate entropy field (simplified Shannon entropy)
            if field_data.ndim >= 2:
                entropy_field = np.zeros_like(field_data)
                for i in range(1, field_data.shape[0]-1):
                    for j in range(1, field_data.shape[1]-1):
                        local_patch = field_data[i-1:i+2, j-1:j+2].flatten()
                        # Normalize to probabilities
                        local_patch = np.abs(local_patch)
                        if np.sum(local_patch) > 0:
                            local_patch = local_patch / np.sum(local_patch)
                            entropy_field[i, j] = -np.sum(local_patch * np.log2(local_patch + 1e-10))
                
                im = ax.imshow(entropy_field, cmap='hot', origin='lower')
                plt.colorbar(im, ax=ax, shrink=0.8, label='Entropy')
            else:
                # 1D entropy
                window_size = min(5, len(field_data)//4)
                entropy_series = []
                for i in range(window_size, len(field_data)-window_size):
                    window = field_data[i-window_size:i+window_size+1]
                    window = np.abs(window)
                    if np.sum(window) > 0:
                        window = window / np.sum(window)
                        entropy = -np.sum(window * np.log2(window + 1e-10))
                        entropy_series.append(entropy)
                    else:
                        entropy_series.append(0)
                
                ax.plot(range(window_size, len(field_data)-window_size), entropy_series, 
                       color='red', linewidth=2)
                ax.set_ylabel('Entropy', color='white')
                entropy_field = np.array(entropy_series)
            
            ax.set_title(f'Field Entropy - {field_name}', color='white', fontsize=10)
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            
            return {
                'success': True,
                'entropy_stats': {
                    'mean': float(np.mean(entropy_field)),
                    'std': float(np.std(entropy_field)),
                    'max': float(np.max(entropy_field))
                }
            }
            
        except Exception as e:
            logger.error(f"Error rendering field entropy: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    def _render_field_evolution_component(self, ax, field_name: str, field_data: np.ndarray) -> Dict:
        """Render field evolution component."""
        try:
            # Simulate 5-step evolution (simplified wave equation)
            evolution_steps = []
            current_field = field_data.copy()
            
            for step in range(5):
                if current_field.ndim >= 2:
                    # 2D wave equation step
                    laplacian = self._calculate_laplacian_2d(current_field)
                    current_field = current_field + 0.1 * laplacian
                else:
                    # 1D wave equation step  
                    laplacian = np.roll(current_field, -1) + np.roll(current_field, 1) - 2*current_field
                    current_field = current_field + 0.1 * laplacian
                
                evolution_steps.append(current_field.copy())
            
            # Plot evolution
            if field_data.ndim >= 2:
                # Show final state
                im = ax.imshow(evolution_steps[-1], cmap='coolwarm', origin='lower')
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Add evolution arrows
                center_y, center_x = np.array(field_data.shape) // 2
                for i, step_field in enumerate(evolution_steps[::2]):  # Every other step
                    intensity = np.abs(step_field[center_y, center_x])
                    circle = Circle((center_x, center_y), 2 + i*3, 
                                  fill=False, color='yellow', alpha=0.3 + 0.1*i)
                    ax.add_patch(circle)
            else:
                # 1D evolution plot
                for i, step_field in enumerate(evolution_steps):
                    alpha = 0.3 + 0.14*i
                    ax.plot(step_field, alpha=alpha, linewidth=1+i*0.3,
                           label=f'Step {i+1}')
                ax.legend(loc='upper right', fontsize=8)
                ax.set_ylabel('Amplitude', color='white')
            
            ax.set_title(f'Field Evolution - {field_name}', color='white', fontsize=10)
            ax.set_facecolor('black')
            ax.tick_params(colors='white')
            
            return {
                'success': True,
                'evolution_stats': {
                    'steps': len(evolution_steps),
                    'final_energy': float(np.sum(evolution_steps[-1]**2)),
                    'energy_change': float(np.sum(evolution_steps[-1]**2) - np.sum(field_data**2))
                }
            }
            
        except Exception as e:
            logger.error(f"Error rendering field evolution: {e}")
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center',
                   transform=ax.transAxes, color='red')
            return {'success': False, 'error': str(e)}
    
    # Data access and calculation utilities
    
    def _get_quantum_state_data(self, state_name: str) -> Dict:
        """Get quantum state data."""
        try:
            if self.state and hasattr(self.state, 'quantum_states'):
                if state_name in self.state.quantum_states:
                    state_obj = self.state.quantum_states[state_name]
                    return {
                        'state_vector': getattr(state_obj, 'state_vector', np.array([1, 0])),
                        'density_matrix': getattr(state_obj, 'density_matrix', None),
                        'coherence': getattr(state_obj, 'coherence', 0.5),
                        'entropy': getattr(state_obj, 'entropy', 0.0)
                    }
            
            # Fallback demo data
            return {
                'state_vector': np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                'density_matrix': None,
                'coherence': 0.8,
                'entropy': 0.2
            }
            
        except Exception as e:
            logger.warning(f"Error getting quantum state data: {e}")
            return {
                'state_vector': np.array([1, 0]),
                'density_matrix': None,
                'coherence': 0.0,
                'entropy': 1.0
            }
    
    def _get_field_data(self, field_name: str) -> np.ndarray:
        """Get field data."""
        try:
            if self.field_dynamics and hasattr(self.field_dynamics, 'get_field_values'):
                return self.field_dynamics.get_field_values(field_name)
            elif self.field_panel and hasattr(self.field_panel, 'get_field_data'):
                return self.field_panel.get_field_data(field_name)
            else:
                # Generate demo field data
                x = np.linspace(0, 10, 50)
                y = np.linspace(0, 10, 50)
                X, Y = np.meshgrid(x, y)
                return np.sin(X) * np.cos(Y) * np.exp(-0.1*(X**2 + Y**2))
                
        except Exception as e:
            logger.warning(f"Error getting field data: {e}")
            # Fallback demo data
            return np.random.randn(20, 20) * 0.5
    
    def _calculate_laplacian_2d(self, field: np.ndarray) -> np.ndarray:
        """Calculate 2D Laplacian of field."""
        try:
            laplacian = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
                        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4*field)
            return laplacian
        except Exception as e:
            logger.warning(f"Error calculating Laplacian: {e}")
            return np.zeros_like(field)
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """Calculate moving average."""
        try:
            return np.convolve(data, np.ones(window)/window, mode='valid')
        except Exception as e:
            logger.warning(f"Error calculating moving average: {e}")
            return data
    
    def _calculate_trend(self, data: np.ndarray) -> str:
        """Calculate trend direction."""
        try:
            if len(data) < 3:
                return "insufficient_data"
            
            # Linear regression slope
            x = np.arange(len(data))
            slope = np.polyfit(x, data, 1)[0]
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"Error calculating trend: {e}")
            return "unknown"
    
    # Additional fallback rendering methods
    
    def _render_quantum_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for quantum panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'Quantum Visualization Error\n\n{error_msg}\n\nUsing fallback rendering',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    def _render_field_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for field panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'Field Dynamics Error\n\n{error_msg}\n\nCheck field system initialization',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    def _render_observer_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for observer panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'Observer Network Error\n\n{error_msg}\n\nCheck observer system status',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    def _render_memory_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for memory panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'Memory Field Error\n\n{error_msg}\n\nCheck memory field initialization',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    def _render_osh_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for OSH panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'OSH Substrate Error\n\n{error_msg}\n\nFalling back to basic metrics',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    def _render_time_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for time evolution panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'Time Evolution Error\n\n{error_msg}\n\nInsufficient temporal data',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    def _render_entanglement_fallback(self, fig, gs_position, error_msg: str) -> Dict:
        """Render fallback for entanglement topology panel errors."""
        ax = fig.add_subplot(gs_position)
        ax.text(0.5, 0.5, f'Entanglement Topology Error\n\n{error_msg}\n\nCheck quantum state availability',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='red', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.3))
        ax.set_facecolor('black')
        ax.axis('off')
        return {'success': False, 'status': 'fallback', 'error': error_msg}
    
    # Statistics and analysis methods
    
    def _calculate_quantum_panel_statistics(self, quantum_states: List[str]) -> Dict:
        """Calculate comprehensive statistics for quantum panel."""
        try:
            stats = {
                'state_count': len(quantum_states),
                'total_qubits': 0,
                'average_coherence': 0.0,
                'average_entropy': 0.0,
                'entangled_states': 0,
                'purity_distribution': []
            }
            
            coherence_values = []
            entropy_values = []
            purity_values = []
            
            for state_name in quantum_states:
                state_data = self._get_quantum_state_data(state_name)
                coherence_values.append(state_data.get('coherence', 0))
                entropy_values.append(state_data.get('entropy', 0))
                
                # Calculate purity from state vector
                state_vector = state_data.get('state_vector', np.array([1, 0]))
                purity = np.sum(np.abs(state_vector)**4)
                purity_values.append(purity)
                
                # Count qubits
                stats['total_qubits'] += int(np.log2(len(state_vector)))
                
                # Check if entangled (simplified)
                if purity < 0.9:
                    stats['entangled_states'] += 1
            
            if coherence_values:
                stats['average_coherence'] = np.mean(coherence_values)
                stats['coherence_std'] = np.std(coherence_values)
                
            if entropy_values:
                stats['average_entropy'] = np.mean(entropy_values)
                stats['entropy_std'] = np.std(entropy_values)
                
            if purity_values:
                stats['average_purity'] = np.mean(purity_values)
                stats['purity_distribution'] = purity_values
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating quantum panel statistics: {e}")
            return {'state_count': 0, 'error': str(e)}
    
    def _calculate_quantum_osh_metrics(self, quantum_states: List[str]) -> Dict:
        """Calculate OSH metrics for quantum states."""
        try:
            if not quantum_states:
                return {}
            
            coherence_values = []
            entropy_values = []
            
            for state_name in quantum_states:
                state_data = self._get_quantum_state_data(state_name)
                coherence_values.append(state_data.get('coherence', 0))
                entropy_values.append(state_data.get('entropy', 0))
            
            avg_coherence = np.mean(coherence_values)
            avg_entropy = np.mean(entropy_values)
            avg_strain = 0.1  # Default strain
            
            return {
                'rsp': self.osh_calculators['rsp'](avg_coherence, avg_entropy, avg_strain),
                'phi': self.osh_calculators['phi'](avg_coherence, avg_entropy),
                'quantum_coherence': avg_coherence,
                'quantum_entropy': avg_entropy,
                'consciousness_potential': avg_coherence * (1 - avg_entropy),
                'information_integration': np.mean(coherence_values) * len(quantum_states)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating quantum OSH metrics: {e}")
            return {}
    
    def _calculate_field_panel_statistics(self, available_fields: List[str]) -> Dict:
        """Calculate comprehensive statistics for field panel."""
        try:
            stats = {
                'field_count': len(available_fields),
                'total_energy': 0.0,
                'average_amplitude': 0.0,
                'coherence_measure': 0.0,
                'entropy_measure': 0.0
            }
            
            energy_values = []
            amplitude_values = []
            
            for field_name in available_fields:
                field_data = self._get_field_data(field_name)
                
                # Energy
                energy = np.sum(field_data**2)
                energy_values.append(energy)
                
                # Average amplitude
                amplitude = np.mean(np.abs(field_data))
                amplitude_values.append(amplitude)
            
            if energy_values:
                stats['total_energy'] = np.sum(energy_values)
                stats['average_energy'] = np.mean(energy_values)
                
            if amplitude_values:
                stats['average_amplitude'] = np.mean(amplitude_values)
                stats['amplitude_std'] = np.std(amplitude_values)
            
            # Field coherence (simplified)
            if available_fields:
                primary_field = self._get_field_data(available_fields[0])
                if primary_field.ndim >= 2:
                    grad_y, grad_x = np.gradient(primary_field)
                    stats['coherence_measure'] = 1 / (1 + np.mean(np.sqrt(grad_x**2 + grad_y**2)))
            
            return stats
            
        except Exception as e:
            logger.warning(f"Error calculating field panel statistics: {e}")
            return {'field_count': 0, 'error': str(e)}
    
    def _calculate_field_osh_metrics(self, available_fields: List[str]) -> Dict:
        """Calculate OSH metrics for fields."""
        try:
            if not available_fields:
                return {}
            
            primary_field = self._get_field_data(available_fields[0])
            
            # Field-based coherence
            if primary_field.ndim >= 2:
                grad_y, grad_x = np.gradient(primary_field)
                coherence = 1 / (1 + np.mean(np.sqrt(grad_x**2 + grad_y**2)))
            else:
                grad = np.gradient(primary_field)
                coherence = 1 / (1 + np.mean(np.abs(grad)))
            
            # Field entropy (local variation)
            entropy = np.std(primary_field) / (np.mean(np.abs(primary_field)) + 1e-10)
            entropy = min(1.0, entropy)
            
            return {
                'field_coherence': coherence,
                'field_entropy': entropy,
                'field_rsp': self.osh_calculators['rsp'](coherence, entropy),
                'field_energy_density': np.mean(primary_field**2),
                'field_information_content': -np.sum(np.abs(primary_field) * 
                                                   np.log(np.abs(primary_field) + 1e-10))
            }
            
        except Exception as e:
            logger.warning(f"Error calculating field OSH metrics: {e}")
            return {}
    
    def _get_observer_relationships(self, observers: Dict) -> Dict:
        """Get observer relationships."""
        try:
            relationships = {}
            observer_names = list(observers.keys())
            
            for i, obs1 in enumerate(observer_names):
                for obs2 in observer_names[i+1:]:
                    # Demo relationship strength calculation
                    obs1_data = observers[obs1]
                    obs2_data = observers[obs2]
                    
                    # Phase similarity
                    phase1 = obs1_data.get('phase', 'passive')
                    phase2 = obs2_data.get('phase', 'passive')
                    phase_similarity = 1.0 if phase1 == phase2 else 0.3
                    
                    # Focus similarity
                    focus1 = obs1_data.get('focus', '')
                    focus2 = obs2_data.get('focus', '')
                    focus_similarity = 1.0 if focus1 == focus2 and focus1 else 0.1
                    
                    strength = (phase_similarity + focus_similarity) / 2
                    if strength > 0.1:
                        relationships[(obs1, obs2)] = strength
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Error getting observer relationships: {e}")
            return {}
    
    def _calculate_observer_consensus(self, observers: Dict) -> float:
        """Calculate observer consensus score."""
        try:
            if not observers:
                return 0.0
            
            phases = [obs.get('phase', 'passive') for obs in observers.values()]
            phase_counts = {}
            for phase in phases:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            # Consensus is the fraction of observers in the dominant phase
            max_count = max(phase_counts.values()) if phase_counts else 0
            consensus = max_count / len(observers) if observers else 0.0
            
            return consensus
            
        except Exception as e:
            logger.warning(f"Error calculating observer consensus: {e}")
            return 0.0
    
    def _get_phase_distribution(self, observers: Dict) -> Dict:
        """Get distribution of observer phases."""
        try:
            phase_dist = {}
            for obs_data in observers.values():
                phase = obs_data.get('phase', 'passive')
                phase_dist[phase] = phase_dist.get(phase, 0) + 1
            return phase_dist
        except Exception as e:
            logger.warning(f"Error getting phase distribution: {e}")
            return {}
    
    def _calculate_observer_osh_metrics(self, observers: Dict) -> Dict:
        """Calculate OSH metrics for observers."""
        try:
            if not observers:
                return {}
            
            consensus = self._calculate_observer_consensus(observers)
            active_count = sum(1 for obs in observers.values() 
                             if obs.get('phase', 'passive') != 'passive')
            
            coherence_values = [obs.get('coherence', 0.5) for obs in observers.values()]
            avg_coherence = np.mean(coherence_values)
            
            return {
                'observer_consensus': consensus,
                'active_observer_ratio': active_count / len(observers),
                'collective_coherence': avg_coherence,
                'emergence_potential': consensus * avg_coherence,
                'observer_entropy': -sum((count/len(observers)) * np.log2(count/len(observers)) 
                                       for count in self._get_phase_distribution(observers).values()),
                'consciousness_indicator': consensus * avg_coherence * np.log(len(observers) + 1)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating observer OSH metrics: {e}")
            return {}
    
    def _create_memory_strain_field(self, memory_data: Dict) -> np.ndarray:
        """Create 2D memory strain field."""
        try:
            # Create synthetic strain field
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 10, 50)
            X, Y = np.meshgrid(x, y)
            
            # Base strain pattern
            strain_field = 0.3 * np.exp(-((X-5)**2 + (Y-5)**2)/8)
            
            # Add regional variations
            strain_data = memory_data.get('strain', {})
            for region_name, strain_value in strain_data.items():
                # Add localized strain peaks
                rx, ry = np.random.uniform(2, 8, 2)
                strain_field += strain_value * np.exp(-((X-rx)**2 + (Y-ry)**2)/2)
            
            return np.clip(strain_field, 0, 1)
            
        except Exception as e:
            logger.warning(f"Error creating memory strain field: {e}")
            return np.random.uniform(0, 0.5, (50, 50))
    
    def _create_memory_coherence_field(self, memory_data: Dict) -> np.ndarray:
        """Create 2D memory coherence field."""
        try:
            # Create synthetic coherence field
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 10, 50)
            X, Y = np.meshgrid(x, y)
            
            # Base coherence pattern (inverse of strain)
            coherence_field = 0.8 - 0.3 * np.exp(-((X-5)**2 + (Y-5)**2)/8)
            
            # Add regional variations
            coherence_data = memory_data.get('coherence', {})
            for region_name, coherence_value in coherence_data.items():
                rx, ry = np.random.uniform(2, 8, 2)
                coherence_field += (coherence_value - 0.5) * np.exp(-((X-rx)**2 + (Y-ry)**2)/3)
            
            return np.clip(coherence_field, 0, 1)
            
        except Exception as e:
            logger.warning(f"Error creating memory coherence field: {e}")
            return np.random.uniform(0.3, 0.9, (50, 50))
    
    def _get_defragmentation_regions(self, memory_data: Dict) -> List[Dict]:
        """Get defragmentation regions."""
        try:
            regions = []
            
            # Extract defragmentation events or regions
            if 'defrag_events' in memory_data:
                for event in memory_data['defrag_events']:
                    regions.append({
                        'x': event.get('x', np.random.uniform(1, 9)),
                        'y': event.get('y', np.random.uniform(1, 9)),
                        'radius': event.get('radius', 0.5)
                    })
            else:
                # Generate synthetic regions
                for _ in range(np.random.randint(2, 6)):
                    regions.append({
                        'x': np.random.uniform(1, 9),
                        'y': np.random.uniform(1, 9),
                        'radius': np.random.uniform(0.3, 0.8)
                    })
            
            return regions
            
        except Exception as e:
            logger.warning(f"Error getting defragmentation regions: {e}")
            return []
    
    def _calculate_memory_field_statistics(self, memory_data: Dict) -> Dict:
        """Calculate memory field statistics."""
        try:
            strain_field = self._create_memory_strain_field(memory_data)
            coherence_field = self._create_memory_coherence_field(memory_data)
            
            return {
                'avg_strain': float(np.mean(strain_field)),
                'max_strain': float(np.max(strain_field)),
                'strain_std': float(np.std(strain_field)),
                'avg_coherence': float(np.mean(coherence_field)),
                'coherence_std': float(np.std(coherence_field)),
                'critical_strain_regions': int(np.sum(strain_field > 0.8)),
                'defrag_events': len(self._get_defragmentation_regions(memory_data)),
                'memory_efficiency': float(np.mean(coherence_field) / (np.mean(strain_field) + 0.1))
            }
            
        except Exception as e:
            logger.warning(f"Error calculating memory field statistics: {e}")
            return {}
    
    def _calculate_memory_osh_metrics(self, memory_data: Dict) -> Dict:
        """Calculate OSH metrics for memory field."""
        try:
            strain_field = self._create_memory_strain_field(memory_data)
            coherence_field = self._create_memory_coherence_field(memory_data)
            
            avg_coherence = np.mean(coherence_field)
            avg_entropy = 1 - avg_coherence  # Simplified
            avg_strain = np.mean(strain_field)
            
            return {
                'memory_rsp': self.osh_calculators['rsp'](avg_coherence, avg_entropy, avg_strain),
                'memory_coherence': avg_coherence,
                'memory_entropy': avg_entropy,
                'memory_strain': avg_strain,
                'memory_stability': avg_coherence / (avg_strain + 0.1),
                'information_density': np.sum(coherence_field * (1 - strain_field)),
                'field_complexity': np.std(coherence_field) + np.std(strain_field)
            }
            
        except Exception as e:
            logger.warning(f"Error calculating memory OSH metrics: {e}")
            return {}
    
    def _create_rsp_field(self) -> np.ndarray:
        """Create RSP field for OSH visualization."""
        try:
            coherence_field = self._create_coherence_field()
            entropy_field = self._create_entropy_field()
            strain_field = self._create_strain_field()
            
            # Calculate RSP for each point
            rsp_field = np.zeros_like(coherence_field)
            for i in range(rsp_field.shape[0]):
                for j in range(rsp_field.shape[1]):
                    c = coherence_field[i, j]
                    h = entropy_field[i, j]
                    s = strain_field[i, j]
                    rsp_field[i, j] = self.osh_calculators['rsp'](c, h, s)
            
            return rsp_field
            
        except Exception as e:
            logger.warning(f"Error creating RSP field: {e}")
            return np.random.uniform(0, 2, (50, 50))
    
    def _create_coherence_field(self) -> np.ndarray:
        """Create coherence field."""
        try:
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 10, 50)
            X, Y = np.meshgrid(x, y)
            
            # Multiple coherence centers
            field = 0.5 * np.ones_like(X)
            centers = [(3, 3), (7, 7), (2, 8), (8, 2)]
            
            for cx, cy in centers:
                field += 0.3 * np.exp(-((X-cx)**2 + (Y-cy)**2)/4)
            
            return np.clip(field, 0, 1)
            
        except Exception as e:
            logger.warning(f"Error creating coherence field: {e}")
            return np.random.uniform(0.3, 0.9, (50, 50))
    
    def _create_entropy_field(self) -> np.ndarray:
        """Create entropy field."""
        try:
            coherence_field = self._create_coherence_field()
            # Entropy is roughly inverse of coherence with some noise
            entropy_field = 1 - coherence_field + 0.1 * np.random.randn(*coherence_field.shape)
            return np.clip(entropy_field, 0, 1)
        except Exception as e:
            logger.warning(f"Error creating entropy field: {e}")
            return np.random.uniform(0.1, 0.7, (50, 50))
    
    def _create_strain_field(self) -> np.ndarray:
        """Create strain field."""
        try:
            x = np.linspace(0, 10, 50)
            y = np.linspace(0, 10, 50)
            X, Y = np.meshgrid(x, y)
            
            # Strain peaks at boundaries and centers
            field = 0.2 * np.ones_like(X)
            field += 0.4 * np.exp(-((X-5)**2 + (Y-5)**2)/12)  # Central strain
            field += 0.3 * (np.exp(-X/2) + np.exp(-(10-X)/2))  # Edge effects
            
            return np.clip(field, 0, 1)
            
        except Exception as e:
            logger.warning(f"Error creating strain field: {e}")
            return np.random.uniform(0.1, 0.6, (50, 50))
    
    def _classify_rsp_level(self, rsp_value: float) -> str:
        """Classify RSP level."""
        if rsp_value > 2.0:
            return "Exceptional"
        elif rsp_value > 1.5:
            return "High"
        elif rsp_value > 1.0:
            return "Moderate"
        elif rsp_value > 0.5:
            return "Low"
        else:
            return "Critical"
    
    def _detect_emergence_points(self, rsp_field: np.ndarray, coherence_field: np.ndarray) -> List[Tuple[float, float]]:
        """Detect emergence points in fields."""
        try:
            points = []
            
            # Find local maxima in RSP that coincide with high coherence
            from scipy.ndimage import maximum_filter
            
            # Local maxima in RSP
            rsp_maxima = (rsp_field == maximum_filter(rsp_field, size=5))
            
            # High coherence regions
            high_coherence = coherence_field > 0.7
            
            # Emergence points are overlap
            emergence_mask = rsp_maxima & high_coherence
            
            # Convert to coordinates
            y_coords, x_coords = np.where(emergence_mask)
            
            # Convert indices to field coordinates
            for y, x in zip(y_coords, x_coords):
                field_x = 10 * x / rsp_field.shape[1]
                field_y = 10 * y / rsp_field.shape[0]
                points.append((field_x, field_y))
            
            return points[:5]  # Limit to top 5
            
        except Exception as e:
            logger.warning(f"Error detecting emergence points: {e}")
            return [(5, 5)]  # Default center point
    
    def _calculate_substrate_stability(self, rsp_field: np.ndarray) -> float:
        """Calculate substrate stability."""
        try:
            # Stability is inverse of RSP variance
            rsp_var = np.var(rsp_field)
            stability = 1 / (1 + rsp_var)
            return min(1.0, stability)
        except Exception as e:
            logger.warning(f"Error calculating substrate stability: {e}")
            return 0.5
    
    def _calculate_osh_validation_score(self, statistics: Dict) -> float:
        """Calculate OSH validation score."""
        try:
            score = 0.0
            
            # RSP component (25%)
            rsp = statistics.get('rsp_mean', 0)
            if rsp > 1.0:
                score += 0.25
            elif rsp > 0.5:
                score += 0.15
            
            # Coherence component (25%)
            coherence = statistics.get('coherence_mean', 0)
            if coherence > 0.7:
                score += 0.25
            elif coherence > 0.5:
                score += 0.15
            
            # Entropy component (20%)
            entropy = statistics.get('entropy_mean', 1)
            if entropy < 0.3:
                score += 0.20
            elif entropy < 0.5:
                score += 0.10
            
            # Emergence component (15%)
            emergence_points = statistics.get('emergence_points', 0)
            if emergence_points > 3:
                score += 0.15
            elif emergence_points > 1:
                score += 0.10
            
            # Coverage component (15%)
            coverage = statistics.get('high_rsp_coverage', 0)
            if coverage > 0.3:
                score += 0.15
            elif coverage > 0.1:
                score += 0.10
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Error calculating OSH validation score: {e}")
            return 0.0
    
    # Time evolution utility methods
    
    def _mark_significant_events(self, ax, time_data: np.ndarray, coherence_data: np.ndarray, 
                               entropy_data: np.ndarray):
        """Mark significant events on time evolution plot."""
        try:
            # Detect coherence drops
            coherence_diff = np.diff(coherence_data)
            drops = np.where(coherence_diff < -0.1)[0]
            
            for drop_idx in drops:
                if drop_idx < len(time_data):
                    ax.axvline(time_data[drop_idx], color='red', alpha=0.6, 
                              linestyle='--', linewidth=1)
                    ax.text(time_data[drop_idx], coherence_data[drop_idx] + 0.05, 
                           'Drop', rotation=90, fontsize=8, color='red')
            
            # Detect entropy spikes
            entropy_diff = np.diff(entropy_data)
            spikes = np.where(entropy_diff > 0.1)[0]
            
            for spike_idx in spikes:
                if spike_idx < len(time_data):
                    ax.axvline(time_data[spike_idx], color='orange', alpha=0.6,
                              linestyle=':', linewidth=1)
                    ax.text(time_data[spike_idx], entropy_data[spike_idx] + 0.05,
                           'Spike', rotation=90, fontsize=8, color='orange')
            
        except Exception as e:
            logger.warning(f"Error marking significant events: {e}")
    
    def _detect_phenomena_events(self, time_data: np.ndarray, coherence_data: np.ndarray,
                               entropy_data: np.ndarray, rsp_data: np.ndarray) -> List[Dict]:
        """Detect phenomena events in time series."""
        try:
            events = []
            
            # Coherence oscillations
            if len(coherence_data) > 10:
                coherence_fft = np.fft.fft(coherence_data)
                dominant_freq = np.argmax(np.abs(coherence_fft[1:len(coherence_fft)//2])) + 1
                if dominant_freq > 2:
                    events.append({
                        'type': 'coherence_oscillation',
                        'time': len(time_data) // 2,
                        'value': np.mean(coherence_data),
                        'frequency': dominant_freq
                    })
            
            # RSP emergence events
            rsp_peaks = np.where(rsp_data > np.percentile(rsp_data, 90))[0]
            for peak_idx in rsp_peaks:
                if peak_idx < len(time_data):
                    events.append({
                        'type': 'rsp_emergence',
                        'time': time_data[peak_idx],
                        'value': rsp_data[peak_idx]
                    })
            
            # Phase transitions (rapid changes)
            coherence_rate = np.abs(np.diff(coherence_data))
            transitions = np.where(coherence_rate > 0.2)[0]
            for trans_idx in transitions:
                if trans_idx < len(time_data) - 1:
                    events.append({
                        'type': 'phase_transition',
                        'time': time_data[trans_idx + 1],
                        'value': coherence_data[trans_idx + 1]
                    })
            
            return events[:10]  # Limit to top 10 events
            
        except Exception as e:
            logger.warning(f"Error detecting phenomena events: {e}")
            return []
    
    def _calculate_stability_score(self, coherence_data: np.ndarray, entropy_data: np.ndarray) -> float:
        """Calculate overall stability score."""
        try:
            # Stability is low variance in key metrics
            coherence_stability = 1 / (1 + np.var(coherence_data))
            entropy_stability = 1 / (1 + np.var(entropy_data))
            
            # Combined score
            stability = (coherence_stability + entropy_stability) / 2
            return min(1.0, stability)
            
        except Exception as e:
            logger.warning(f"Error calculating stability score: {e}")
            return 0.5
    
    def _calculate_temporal_complexity(self, time_data: np.ndarray, metric_arrays: List[np.ndarray]) -> float:
        """Calculate temporal complexity."""
        try:
            complexity = 0.0
            
            for metrics in metric_arrays:
                if len(metrics) > 2:
                    # Complexity as information content
                    # Approximate with entropy of normalized data
                    normalized = (metrics - np.min(metrics)) / (np.max(metrics) - np.min(metrics) + 1e-10)
                    hist, _ = np.histogram(normalized, bins=10, density=True)
                    hist = hist + 1e-10  # Avoid log(0)
                    entropy = -np.sum(hist * np.log2(hist))
                    complexity += entropy
            
            return complexity / len(metric_arrays) if metric_arrays else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating temporal complexity: {e}")
            return 0.0
    
    def _predict_next_values(self, coherence_data: np.ndarray, entropy_data: np.ndarray, 
                           rsp_data: np.ndarray) -> Dict:
        """Predict next values using simple extrapolation."""
        try:
            predictions = {}
            
            if len(coherence_data) >= 3:
                # Linear extrapolation
                x = np.arange(len(coherence_data))
                
                # Coherence prediction
                coherence_poly = np.polyfit(x[-5:], coherence_data[-5:], 1)
                next_coherence = np.polyval(coherence_poly, len(coherence_data))
                predictions['coherence'] = float(np.clip(next_coherence, 0, 1))
                
                # Entropy prediction
                entropy_poly = np.polyfit(x[-5:], entropy_data[-5:], 1)
                next_entropy = np.polyval(entropy_poly, len(entropy_data))
                predictions['entropy'] = float(np.clip(next_entropy, 0, 1))
                
                # RSP prediction
                rsp_poly = np.polyfit(x[-5:], rsp_data[-5:], 1)
                next_rsp = np.polyval(rsp_poly, len(rsp_data))
                predictions['rsp'] = float(max(0, next_rsp))
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Error predicting next values: {e}")
            return {}
    
    # Entanglement analysis methods
    
    def _get_entanglement_relationships(self, quantum_states: List[str]) -> Dict:
        """Get entanglement relationships between quantum states."""
        try:
            relationships = {}
            
            # In real implementation, get from entanglement_manager
            if self.entanglement_manager:
                for i, state1 in enumerate(quantum_states):
                    for state2 in quantum_states[i+1:]:
                        strength = self._calculate_entanglement_strength(state1, state2)
                        if strength > 0:
                            relationships[(state1, state2)] = strength
            else:
                # Demo entanglement data
                for i, state1 in enumerate(quantum_states):
                    for state2 in quantum_states[i+1:]:
                        # Random but realistic entanglement
                        if np.random.random() < 0.3:  # 30% chance of entanglement
                            strength = np.random.beta(2, 5)  # Realistic distribution
                            relationships[(state1, state2)] = strength
            
            return relationships
            
        except Exception as e:
            logger.warning(f"Error getting entanglement relationships: {e}")
            return {}
    
    def _calculate_entanglement_strength(self, state1: str, state2: str) -> float:
        """Calculate entanglement strength between two states."""
        try:
            if self.entanglement_manager and hasattr(self.entanglement_manager, 'get_entanglement_strength'):
                return self.entanglement_manager.get_entanglement_strength(state1, state2)
            else:
                # Simplified calculation based on state data
                state1_data = self._get_quantum_state_data(state1)
                state2_data = self._get_quantum_state_data(state2)
                
                # Entanglement proxy: anticorrelation of coherence values
                coh1 = state1_data.get('coherence', 0.5)
                coh2 = state2_data.get('coherence', 0.5)
                
                # Simple entanglement measure
                return abs(coh1 - coh2) * np.random.uniform(0.3, 0.9)
                
        except Exception as e:
            logger.warning(f"Error calculating entanglement strength: {e}")
            return 0.0
    
    def _analyze_entanglement_distribution(self, entanglement_data: Dict) -> Dict:
        """Analyze entanglement strength distribution."""
        try:
            if not entanglement_data:
                return {}
            
            strengths = list(entanglement_data.values())
            
            return {
                'mean_strength': np.mean(strengths),
                'std_strength': np.std(strengths), 
                'max_strength': np.max(strengths),
                'min_strength': np.min(strengths),
                'strong_pairs': sum(1 for s in strengths if s > 0.7),
                'weak_pairs': sum(1 for s in strengths if s < 0.3)
            }
            
        except Exception as e:
            logger.warning(f"Error analyzing entanglement distribution: {e}")
            return {}
    
    def _calculate_entanglement_complexity(self, graph, entanglement_data: Dict) -> float:
        """Calculate entanglement complexity."""
        try:
            if not entanglement_data:
                return 0.0
            
            # Complexity based on network structure and strength distribution
            network_complexity = len(graph.edges()) / max(1, len(graph.nodes())**2)
            strength_complexity = np.std(list(entanglement_data.values()))
            
            return (network_complexity + strength_complexity) / 2
            
        except Exception as e:
            logger.warning(f"Error calculating entanglement complexity: {e}")
            return 0.0
    
    def _estimate_quantum_discord(self, entanglement_data: Dict, coherence_values: List[float]) -> float:
        """Estimate quantum discord."""
        try:
            if not entanglement_data or not coherence_values:
                return 0.0
            
            # Simplified discord estimate
            avg_entanglement = np.mean(list(entanglement_data.values()))
            avg_coherence = np.mean(coherence_values)
            
            discord = avg_entanglement * (1 - avg_coherence)
            return max(0.0, discord)
            
        except Exception as e:
            logger.warning(f"Error estimating quantum discord: {e}")
            return 0.0
    
    def _calculate_entanglement_entropy(self, entanglement_data: Dict) -> float:
        """Calculate entanglement entropy."""
        try:
            if not entanglement_data:
                return 0.0
            
            strengths = np.array(list(entanglement_data.values()))
            # Normalize to probabilities
            strengths = strengths / np.sum(strengths)
            
            # Shannon entropy
            entropy = -np.sum(strengths * np.log2(strengths + 1e-10))
            return entropy
            
        except Exception as e:
            logger.warning(f"Error calculating entanglement entropy: {e}")
            return 0.0
    
    def _calculate_network_emergence(self, graph, coherence_values: List[float]) -> float:
        """Calculate network emergence measure."""
        try:
            if not coherence_values or len(graph.nodes()) == 0:
                return 0.0
            
            # Emergence as coherence variance weighted by connectivity
            coherence_var = np.var(coherence_values)
            connectivity = len(graph.edges()) / max(1, len(graph.nodes()))
            
            emergence = coherence_var * connectivity
            return min(1.0, emergence)
            
        except Exception as e:
            logger.warning(f"Error calculating network emergence: {e}")
            return 0.0
    
    def _calculate_topology_stability(self, entanglement_data: Dict) -> float:
        """Calculate topology stability."""
        try:
            if not entanglement_data:
                return 0.0
            
            # Stability as inverse of strength variance
            strengths = list(entanglement_data.values())
            stability = 1 / (1 + np.var(strengths))
            
            return min(1.0, stability)
            
        except Exception as e:
            logger.warning(f"Error calculating topology stability: {e}")
            return 0.5
    
    def _calculate_coherence_flow(self, graph, coherence_values: List[float], 
                                entanglement_data: Dict) -> float:
        """Calculate coherence flow through network."""
        try:
            if not coherence_values or not entanglement_data:
                return 0.0
            
            # Flow as weighted sum of coherence differences across edges
            total_flow = 0.0
            node_coherence = {node: coh for node, coh in zip(graph.nodes(), coherence_values)}
            
            for (node1, node2), strength in entanglement_data.items():
                coh1 = node_coherence.get(node1, 0.5)
                coh2 = node_coherence.get(node2, 0.5)
                flow = strength * abs(coh1 - coh2)
                total_flow += flow
            
            return total_flow / max(1, len(entanglement_data))
            
        except Exception as e:
            logger.warning(f"Error calculating coherence flow: {e}")
            return 0.0
    
    def _embed_coherence_renderer_result(self, ax, coherence_result: Dict):
        """Embed coherence renderer result in subplot."""
        try:
            # Extract base64 image data
            image_data = coherence_result.get('image_data', '')
            if image_data.startswith('data:image/png;base64,'):
                image_data = image_data.split(',')[1]
            
            # Decode and display
            import io
            from PIL import Image
            
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            ax.imshow(np.array(image))
            ax.axis('off')
            ax.set_title('OSH Substrate Analysis (CoherenceRenderer)', 
                        color='white', fontsize=12, fontweight='bold')
            
        except Exception as e:
            logger.warning(f"Error embedding coherence renderer result: {e}")
            ax.text(0.5, 0.5, 'CoherenceRenderer\nResult Unavailable', 
                   ha='center', va='center', transform=ax.transAxes,
                   color='white', fontsize=12)
    
    def get_render_performance_report(self) -> Dict:
        """Get comprehensive performance report for all render methods."""
        try:
            report = {
                'render_times': {},
                'fallback_counts': dict(self.fallback_counts),
                'average_times': {},
                'total_renders': sum(len(times) for times in self.render_times.values()),
                'performance_summary': {},
                'bottleneck_analysis': {},
                'recommendations': []
            }
            
            # Analyze render times for each panel type
            for panel_name, times in self.render_times.items():
                if times:
                    recent_times = times[-20:] if len(times) >= 20 else times
                    
                    report['render_times'][panel_name] = {
                        'count': len(times),
                        'total_time': sum(times),
                        'min_time': min(times),
                        'max_time': max(times),
                        'mean_time': np.mean(times),
                        'median_time': np.median(times),
                        'std_time': np.std(times),
                        'recent_average': np.mean(recent_times),
                        'trend': self._calculate_performance_trend(times)
                    }
                    report['average_times'][panel_name] = np.mean(times)
            
            # Performance summary
            if report['average_times']:
                all_times = [time for times in self.render_times.values() for time in times]
                report['performance_summary'] = {
                    'overall_average': np.mean(all_times),
                    'overall_median': np.median(all_times),
                    'slowest_panel': max(report['average_times'].items(), key=lambda x: x[1])[0],
                    'fastest_panel': min(report['average_times'].items(), key=lambda x: x[1])[0],
                    'total_fallbacks': sum(self.fallback_counts.values()),
                    'reliability_score': 1 - (sum(self.fallback_counts.values()) / max(1, report['total_renders']))
                }
            
            # Bottleneck analysis
            report['bottleneck_analysis'] = self._analyze_bottlenecks()
            
            # Performance recommendations
            report['recommendations'] = self._generate_performance_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {'error': str(e), 'total_renders': 0}
    
    def _calculate_performance_trend(self, times: List[float]) -> str:
        """Calculate performance trend over time."""
        try:
            if len(times) < 10:
                return "insufficient_data"
            
            # Compare first half to second half
            mid_point = len(times) // 2
            first_half_avg = np.mean(times[:mid_point])
            second_half_avg = np.mean(times[mid_point:])
            
            ratio = second_half_avg / first_half_avg
            
            if ratio > 1.1:
                return "degrading"
            elif ratio < 0.9:
                return "improving"
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"Error calculating performance trend: {e}")
            return "unknown"
    
    def _analyze_bottlenecks(self) -> Dict:
        """Analyze performance bottlenecks."""
        try:
            bottlenecks = {
                'slow_panels': [],
                'high_fallback_panels': [],
                'resource_intensive': [],
                'optimization_targets': []
            }
            
            # Identify slow panels (>500ms average)
            for panel_name, times in self.render_times.items():
                if times:
                    avg_time = np.mean(times)
                    if avg_time > 0.5:
                        bottlenecks['slow_panels'].append({
                            'panel': panel_name,
                            'avg_time': avg_time,
                            'severity': 'high' if avg_time > 1.0 else 'medium'
                        })
            
            # Identify high fallback panels (>10% fallback rate)
            for panel_name, fallback_count in self.fallback_counts.items():
                render_count = len(self.render_times.get(panel_name, []))
                if render_count > 0:
                    fallback_rate = fallback_count / render_count
                    if fallback_rate > 0.1:
                        bottlenecks['high_fallback_panels'].append({
                            'panel': panel_name,
                            'fallback_rate': fallback_rate,
                            'fallback_count': fallback_count
                        })
            
            # Identify resource intensive panels (high variance)
            for panel_name, times in self.render_times.items():
                if len(times) > 5:
                    variance = np.var(times)
                    if variance > 0.01:  # High variance in render times
                        bottlenecks['resource_intensive'].append({
                            'panel': panel_name,
                            'variance': variance,
                            'instability': 'high' if variance > 0.05 else 'medium'
                        })
            
            # Generate optimization targets
            all_panels = set(self.render_times.keys()) | set(self.fallback_counts.keys())
            for panel_name in all_panels:
                score = self._calculate_optimization_priority(panel_name)
                if score > 0.5:
                    bottlenecks['optimization_targets'].append({
                        'panel': panel_name,
                        'priority_score': score,
                        'priority': 'high' if score > 0.8 else 'medium'
                    })
            
            return bottlenecks
            
        except Exception as e:
            logger.warning(f"Error analyzing bottlenecks: {e}")
            return {}
    
    def _calculate_optimization_priority(self, panel_name: str) -> float:
        """Calculate optimization priority score for a panel."""
        try:
            score = 0.0
            
            # Time component (40%)
            times = self.render_times.get(panel_name, [])
            if times:
                avg_time = np.mean(times)
                time_score = min(1.0, avg_time / 2.0)  # Normalize by 2 seconds
                score += 0.4 * time_score
            
            # Fallback component (30%)
            fallback_count = self.fallback_counts.get(panel_name, 0)
            render_count = len(times) if times else 1
            fallback_rate = fallback_count / render_count
            score += 0.3 * fallback_rate
            
            # Variance component (20%)
            if len(times) > 2:
                variance = np.var(times)
                variance_score = min(1.0, variance / 0.1)  # Normalize by 0.1
                score += 0.2 * variance_score
            
            # Usage component (10%)
            usage_score = min(1.0, render_count / 100)  # Normalize by 100 renders
            score += 0.1 * usage_score
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"Error calculating optimization priority: {e}")
            return 0.0
    
    def _generate_performance_recommendations(self, report: Dict) -> List[str]:
        """Generate performance optimization recommendations."""
        try:
            recommendations = []
            
            # Check overall performance
            summary = report.get('performance_summary', {})
            if summary.get('overall_average', 0) > 1.0:
                recommendations.append("Overall render performance is slow (>1s average). Consider optimizing data structures and caching.")
            
            # Check reliability
            reliability = summary.get('reliability_score', 1.0)
            if reliability < 0.9:
                recommendations.append(f"Low reliability score ({reliability:.2f}). Investigate frequent fallbacks and improve error handling.")
            
            # Check specific bottlenecks
            bottlenecks = report.get('bottleneck_analysis', {})
            
            slow_panels = bottlenecks.get('slow_panels', [])
            if slow_panels:
                slowest = max(slow_panels, key=lambda x: x['avg_time'])
                recommendations.append(f"Optimize {slowest['panel']} panel (avg: {slowest['avg_time']:.3f}s). Consider data reduction or caching.")
            
            high_fallback = bottlenecks.get('high_fallback_panels', [])
            if high_fallback:
                worst = max(high_fallback, key=lambda x: x['fallback_rate'])
                recommendations.append(f"Fix {worst['panel']} panel reliability ({worst['fallback_rate']:.1%} fallback rate).")
            
            resource_intensive = bottlenecks.get('resource_intensive', [])
            if resource_intensive:
                most_unstable = max(resource_intensive, key=lambda x: x['variance'])
                recommendations.append(f"Stabilize {most_unstable['panel']} panel performance (high variance: {most_unstable['variance']:.4f}).")
            
            # Check for missing subsystems
            if not self.quantum_renderer:
                recommendations.append("Initialize QuantumRenderer for better quantum visualizations.")
            
            if not self.coherence_renderer:
                recommendations.append("Initialize CoherenceRenderer for advanced OSH substrate analysis.")
            
            if not self.field_dynamics:
                recommendations.append("Initialize FieldDynamics for field evolution capabilities.")
            
            # Check cache effectiveness
            if hasattr(self, 'visualization_cache') and len(self.visualization_cache) == 0:
                recommendations.append("Enable visualization caching to improve repeated render performance.")
            
            # Memory optimization
            total_renders = report.get('total_renders', 0)
            if total_renders > 1000:
                recommendations.append("Consider implementing render result cleanup for long-running sessions.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            return ["Error generating recommendations - check system logs."]
    
    def get_subsystem_status(self) -> Dict:
        """Get status of all integrated subsystems."""
        try:
            status = {
                'quantum_renderer': {
                    'available': self.quantum_renderer is not None,
                    'type': type(self.quantum_renderer).__name__ if self.quantum_renderer else None,
                    'features': []
                },
                'field_panel': {
                    'available': self.field_panel is not None,
                    'type': type(self.field_panel).__name__ if self.field_panel else None,
                    'features': []
                },
                'field_dynamics': {
                    'available': self.field_dynamics is not None,
                    'type': type(self.field_dynamics).__name__ if self.field_dynamics else None,
                    'features': []
                },
                'observer_panel': {
                    'available': self.observer_panel is not None,
                    'type': type(self.observer_panel).__name__ if self.observer_panel else None,
                    'features': []
                },
                'observer_dynamics': {
                    'available': self.observer_dynamics is not None,
                    'type': type(self.observer_dynamics).__name__ if self.observer_dynamics else None,
                    'features': []
                },
                'memory_field': {
                    'available': self.memory_field is not None,
                    'type': type(self.memory_field).__name__ if self.memory_field else None,
                    'features': []
                },
                'coherence_renderer': {
                    'available': self.coherence_renderer is not None,
                    'type': type(self.coherence_renderer).__name__ if self.coherence_renderer else None,
                    'features': []
                },
                'entanglement_manager': {
                    'available': self.entanglement_manager is not None,
                    'type': type(self.entanglement_manager).__name__ if self.entanglement_manager else None,
                    'features': []
                },
                'state': {
                    'available': self.state is not None,
                    'type': type(self.state).__name__ if self.state else None,
                    'features': []
                }
            }
            
            # Check features for each subsystem
            if self.quantum_renderer:
                status['quantum_renderer']['features'] = [
                    'bloch_sphere', 'density_matrix', 'probability_distribution', 'circuit_visualization'
                ]
            
            if self.field_panel:
                status['field_panel']['features'] = [
                    'field_visualization', 'coherence_analysis', 'entropy_mapping', 'evolution_tracking'
                ]
            
            if self.field_dynamics:
                status['field_dynamics']['features'] = [
                    'field_evolution', 'pde_solving', 'boundary_conditions', 'coupling_analysis'
                ]
            
            if self.observer_panel:
                status['observer_panel']['features'] = [
                    'observer_network', 'phase_analysis', 'consensus_tracking', 'relationship_mapping'
                ]
            
            if self.observer_dynamics:
                status['observer_dynamics']['features'] = [
                    'observer_simulation', 'phase_transitions', 'collapse_modeling', 'consensus_calculation'
                ]
            
            if self.memory_field:
                status['memory_field']['features'] = [
                    'memory_regions', 'strain_analysis', 'coherence_fields', 'defragmentation'
                ]
            
            if self.coherence_renderer:
                status['coherence_renderer']['features'] = [
                    'rsp_visualization', 'osh_metrics', 'substrate_analysis', 'emergence_detection'
                ]
            
            if self.entanglement_manager:
                status['entanglement_manager']['features'] = [
                    'entanglement_protocols', 'strength_calculation', 'network_analysis', 'topology_mapping'
                ]
            
            if self.state:
                status['state']['features'] = [
                    'quantum_states', 'observers', 'variables', 'execution_context'
                ]
            
            return status
            
        except Exception as e:
            logger.error(f"Error getting subsystem status: {e}")
            return {'error': str(e)}
    
    def get_osh_validation_report(self) -> Dict:
        """Generate comprehensive OSH validation report."""
        try:
            report = {
                'validation_score': 0.0,
                'evidence_summary': {},
                'theoretical_alignment': {},
                'experimental_predictions': {},
                'confidence_metrics': {},
                'recommendations': []
            }
            
            # Collect OSH evidence from all subsystems
            evidence = {
                'recursive_simulation_potential': 0.0,
                'consciousness_substrate': 0.0,
                'information_geometry': 0.0,
                'entropy_minimization': 0.0,
                'observer_collapse_model': 0.0
            }
            
            # Quantum subsystem evidence
            quantum_states = self._get_available_quantum_states()
            if quantum_states:
                quantum_osh = self._calculate_quantum_osh_metrics(quantum_states)
                evidence['recursive_simulation_potential'] += quantum_osh.get('rsp', 0) * 0.3
                evidence['consciousness_substrate'] += quantum_osh.get('consciousness_potential', 0) * 0.3
            
            # Field subsystem evidence
            available_fields = self._get_available_fields()
            if available_fields:
                field_osh = self._calculate_field_osh_metrics(available_fields)
                evidence['information_geometry'] += field_osh.get('field_coherence', 0) * 0.3
                evidence['entropy_minimization'] += (1 - field_osh.get('field_entropy', 1)) * 0.3
            
            # Observer subsystem evidence
            observers = self._get_observer_data()
            if observers:
                observer_osh = self._calculate_observer_osh_metrics(observers)
                evidence['observer_collapse_model'] += observer_osh.get('observer_consensus', 0) * 0.3
                evidence['consciousness_substrate'] += observer_osh.get('consciousness_indicator', 0) * 0.3
            
            # Memory subsystem evidence
            memory_data = self._get_memory_field_data()
            if memory_data:
                memory_osh = self._calculate_memory_osh_metrics(memory_data)
                evidence['recursive_simulation_potential'] += memory_osh.get('memory_rsp', 0) * 0.2
                evidence['entropy_minimization'] += memory_osh.get('memory_stability', 0) * 0.2
            
            # Calculate overall validation score
            evidence_weights = {
                'recursive_simulation_potential': 0.25,
                'consciousness_substrate': 0.25,
                'information_geometry': 0.20,
                'entropy_minimization': 0.15,
                'observer_collapse_model': 0.15
            }
            
            validation_score = sum(evidence[key] * weight for key, weight in evidence_weights.items())
            report['validation_score'] = min(1.0, validation_score)
            
            # Evidence summary
            report['evidence_summary'] = {
                key: {
                    'strength': value,
                    'classification': self._classify_evidence_strength(value),
                    'weight': evidence_weights[key]
                }
                for key, value in evidence.items()
            }
            
            # Theoretical alignment
            report['theoretical_alignment'] = {
                'osh_core_principles': {
                    'recursive_self_modeling': evidence['recursive_simulation_potential'] > 0.5,
                    'consciousness_substrate': evidence['consciousness_substrate'] > 0.5,
                    'information_curvature': evidence['information_geometry'] > 0.5,
                    'entropy_regulation': evidence['entropy_minimization'] > 0.5,
                    'observer_participation': evidence['observer_collapse_model'] > 0.5
                },
                'alignment_score': sum(1 for v in evidence.values() if v > 0.5) / len(evidence),
                'supporting_evidence_count': sum(1 for v in evidence.values() if v > 0.3),
                'contradictory_evidence_count': sum(1 for v in evidence.values() if v < 0.1)
            }
            
            # Experimental predictions
            report['experimental_predictions'] = {
                'coherence_wave_detection': evidence['information_geometry'] > 0.6,
                'observer_consensus_emergence': evidence['observer_collapse_model'] > 0.6,
                'recursive_boundary_effects': evidence['recursive_simulation_potential'] > 0.7,
                'entropy_flow_anomalies': evidence['entropy_minimization'] > 0.6,
                'consciousness_signatures': evidence['consciousness_substrate'] > 0.7
            }
            
            # Confidence metrics
            report['confidence_metrics'] = {
                'data_completeness': self._calculate_data_completeness(),
                'measurement_reliability': self._calculate_measurement_reliability(),
                'theoretical_consistency': report['theoretical_alignment']['alignment_score'],
                'prediction_strength': sum(report['experimental_predictions'].values()) / len(report['experimental_predictions']),
                'overall_confidence': (validation_score + report['theoretical_alignment']['alignment_score']) / 2
            }
            
            # Recommendations
            report['recommendations'] = self._generate_osh_recommendations(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating OSH validation report: {e}")
            return {'error': str(e), 'validation_score': 0.0}
    
    def _classify_evidence_strength(self, value: float) -> str:
        """Classify evidence strength."""
        if value > 0.8:
            return "strong"
        elif value > 0.6:
            return "moderate"
        elif value > 0.3:
            return "weak"
        else:
            return "insufficient"
    
    def _calculate_data_completeness(self) -> float:
        """Calculate data completeness score."""
        try:
            completeness = 0.0
            total_subsystems = 5
            
            if self._get_available_quantum_states():
                completeness += 0.2
            
            if self._get_available_fields():
                completeness += 0.2
            
            if self._get_observer_data():
                completeness += 0.2
            
            if self._get_memory_field_data():
                completeness += 0.2
            
            if self.metrics_history and len(self.metrics_history) > 10:
                completeness += 0.2
            
            return completeness
            
        except Exception as e:
            logger.warning(f"Error calculating data completeness: {e}")
            return 0.0
    
    def _calculate_measurement_reliability(self) -> float:
        """Calculate measurement reliability score."""
        try:
            total_renders = sum(len(times) for times in self.render_times.values())
            total_fallbacks = sum(self.fallback_counts.values())
            
            if total_renders == 0:
                return 0.0
            
            reliability = 1 - (total_fallbacks / total_renders)
            return max(0.0, reliability)
            
        except Exception as e:
            logger.warning(f"Error calculating measurement reliability: {e}")
            return 0.5
    
    def _generate_osh_recommendations(self, report: Dict) -> List[str]:
        """Generate OSH-specific recommendations."""
        try:
            recommendations = []
            
            validation_score = report.get('validation_score', 0)
            
            if validation_score < 0.3:
                recommendations.append("Low OSH validation score. Improve data collection and subsystem integration.")
            elif validation_score < 0.6:
                recommendations.append("Moderate OSH validation. Focus on strengthening weak evidence areas.")
            else:
                recommendations.append("Strong OSH validation. Continue monitoring and refine predictions.")
            
            # Specific evidence recommendations
            evidence = report.get('evidence_summary', {})
            
            for key, data in evidence.items():
                if data.get('classification') == 'insufficient':
                    if key == 'recursive_simulation_potential':
                        recommendations.append("Enhance RSP measurement by improving coherence and entropy tracking.")
                    elif key == 'consciousness_substrate':
                        recommendations.append("Strengthen consciousness evidence by expanding observer network analysis.")
                    elif key == 'information_geometry':
                        recommendations.append("Improve information geometry measurement through field curvature analysis.")
                    elif key == 'entropy_minimization':
                        recommendations.append("Enhance entropy tracking and defragmentation monitoring.")
                    elif key == 'observer_collapse_model':
                        recommendations.append("Expand observer consensus and collapse probability analysis.")
            
            # Prediction recommendations
            predictions = report.get('experimental_predictions', {})
            strong_predictions = [k for k, v in predictions.items() if v]
            
            if strong_predictions:
                recommendations.append(f"Strong predictions for: {', '.join(strong_predictions)}. Design targeted experiments.")
            
            confidence = report.get('confidence_metrics', {})
            if confidence.get('data_completeness', 0) < 0.7:
                recommendations.append("Improve data completeness by initializing missing subsystems.")
            
            if confidence.get('measurement_reliability', 0) < 0.8:
                recommendations.append("Improve measurement reliability by reducing fallback rates.")
            
            return recommendations
            
        except Exception as e:
            logger.warning(f"Error generating OSH recommendations: {e}")
            return ["Error generating recommendations - check system logs."]
    
    def reset_performance_tracking(self):
        """Reset all performance tracking data."""
        try:
            self.render_times.clear()
            self.fallback_counts.clear()
            logger.info("Performance tracking data reset")
        except Exception as e:
            logger.error(f"Error resetting performance tracking: {e}")
    
    def cleanup(self):
        """Clean up resources and caches."""
        try:
            # Clear caches
            if hasattr(self, 'visualization_cache'):
                self.visualization_cache.clear()
            
            # Clear performance tracking
            self.render_times.clear()
            self.fallback_counts.clear()
            
            # Clear any matplotlib figures to prevent memory leaks
            plt.close('all')
            
            logger.info("PhysicsRenderer cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during PhysicsRenderer cleanup: {e}")


def create_physics_renderer(**kwargs) -> PhysicsRenderer:
    """
    Factory function to create a PhysicsRenderer with comprehensive configuration.
    
    Args:
        **kwargs: Configuration parameters for the PhysicsRenderer
        
    Returns:
        Configured PhysicsRenderer instance
    """
    try:
        # Extract configuration
        config = kwargs.get('config', {})
        
        # Create renderer with all available subsystems
        renderer = PhysicsRenderer(
            quantum_renderer=kwargs.get('quantum_renderer'),
            field_panel=kwargs.get('field_panel'),
            field_dynamics=kwargs.get('field_dynamics'),
            observer_panel=kwargs.get('observer_panel'),
            observer_dynamics=kwargs.get('observer_dynamics'),
            memory_field=kwargs.get('memory_field'),
            coherence_renderer=kwargs.get('coherence_renderer'),
            entanglement_manager=kwargs.get('entanglement_manager'),
            state=kwargs.get('state'),
            current_colors=kwargs.get('current_colors'),
            current_metrics=kwargs.get('current_metrics'),
            metrics_history=kwargs.get('metrics_history'),
            error_manager=kwargs.get('error_manager')
        )
        
        logger.info("PhysicsRenderer created successfully with factory function")
        return renderer
        
    except Exception as e:
        logger.error(f"Error creating PhysicsRenderer: {e}")
        logger.error(traceback.format_exc())
        # Return minimal renderer for graceful degradation
        return PhysicsRenderer()


# Utility functions for PhysicsRenderer integration

def validate_physics_renderer_config(config: Dict) -> Dict:
    """
    Validate PhysicsRenderer configuration and return validation report.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Validation report with errors, warnings, and recommendations
    """
    try:
        report = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'completeness_score': 0.0
        }
        
        # Check required subsystems
        required_subsystems = [
            'quantum_renderer', 'field_panel', 'observer_panel', 
            'memory_field', 'coherence_renderer'
        ]
        
        missing_subsystems = [sub for sub in required_subsystems if not config.get(sub)]
        
        if missing_subsystems:
            report['warnings'].append(f"Missing subsystems: {', '.join(missing_subsystems)}")
            report['recommendations'].append("Initialize missing subsystems for full functionality")
        
        # Calculate completeness
        available_subsystems = sum(1 for sub in required_subsystems if config.get(sub))
        report['completeness_score'] = available_subsystems / len(required_subsystems)
        
        # Check configuration parameters
        if not config.get('current_colors'):
            report['warnings'].append("No color scheme specified - using defaults")
        
        if not config.get('current_metrics'):
            report['warnings'].append("No current metrics provided - some features may be limited")
        
        if not config.get('metrics_history'):
            report['warnings'].append("No metrics history - temporal analysis will be limited")
            
        # Performance recommendations
        if report['completeness_score'] < 0.6:
            report['recommendations'].append("Initialize more subsystems for comprehensive visualization")
        
        if report['completeness_score'] == 1.0:
            report['recommendations'].append("Full subsystem integration - optimal configuration")
        
        return report
        
    except Exception as e:
        logger.error(f"Error validating PhysicsRenderer config: {e}")
        return {
            'valid': False,
            'errors': [str(e)],
            'warnings': [],
            'recommendations': [],
            'completeness_score': 0.0
        }


def get_physics_renderer_capabilities(renderer: PhysicsRenderer) -> Dict:
    """
    Get comprehensive capabilities report for a PhysicsRenderer instance.
    
    Args:
        renderer: PhysicsRenderer instance
        
    Returns:
        Capabilities report with available features and limitations
    """
    try:
        capabilities = {
            'available_panels': [],
            'visualization_modes': [],
            'osh_features': [],
            'analysis_capabilities': [],
            'performance_features': [],
            'limitations': []
        }
        
        # Check available panels
        if renderer.quantum_renderer:
            capabilities['available_panels'].append('quantum_visualization')
            capabilities['visualization_modes'].extend([
                'bloch_sphere', 'density_matrix', 'probability_distribution', 'entanglement_network'
            ])
        
        if renderer.field_panel or renderer.field_dynamics:
            capabilities['available_panels'].append('field_dynamics')
            capabilities['visualization_modes'].extend([
                'field_amplitude', 'field_coherence', 'field_entropy', 'field_evolution'
            ])
        
        if renderer.observer_panel or renderer.observer_dynamics:
            capabilities['available_panels'].append('observer_network')
            capabilities['visualization_modes'].extend([
                'observer_topology', 'phase_distribution', 'consensus_analysis'
            ])
        
        if renderer.memory_field:
            capabilities['available_panels'].append('memory_field')
            capabilities['visualization_modes'].extend([
                'memory_strain', 'memory_coherence', 'defragmentation_analysis'
            ])
        
        if renderer.coherence_renderer:
            capabilities['available_panels'].append('osh_substrate')
            capabilities['visualization_modes'].extend([
                'rsp_landscape', 'emergence_detection', 'substrate_analysis'
            ])
        
        # OSH features
        capabilities['osh_features'] = [
            'recursive_simulation_potential',
            'integrated_information',
            'emergence_detection',
            'coherence_analysis',
            'entropy_tracking',
            'observer_consensus',
            'information_curvature'
        ]
        
        # Analysis capabilities
        capabilities['analysis_capabilities'] = [
            'performance_profiling',
            'bottleneck_analysis',
            'trend_detection',
            'statistical_analysis',
            'validation_reporting'
        ]
        
        # Performance features
        capabilities['performance_features'] = [
            'fallback_rendering',
            'error_recovery',
            'performance_tracking',
            'cache_management',
            'resource_cleanup'
        ]
        
        # Identify limitations
        if not renderer.quantum_renderer:
            capabilities['limitations'].append('No quantum visualization - install quantum renderer')
        
        if not renderer.coherence_renderer:
            capabilities['limitations'].append('Limited OSH analysis - install coherence renderer')
        
        if not renderer.field_dynamics:
            capabilities['limitations'].append('No field evolution - install field dynamics')
        
        if not renderer.entanglement_manager:
            capabilities['limitations'].append('Limited entanglement analysis - install entanglement manager')
        
        return capabilities
        
    except Exception as e:
        logger.error(f"Error getting PhysicsRenderer capabilities: {e}")
        return {'error': str(e)}


# Export the main class and factory function
__all__ = [
    'PhysicsRenderer',
    'create_physics_renderer',
    'validate_physics_renderer_config',
    'get_physics_renderer_capabilities'
]