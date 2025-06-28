"""
general_utils.py - Recursia Enterprise Visualization Utilities

Comprehensive utility module providing advanced OSH-aligned system analysis,
performance monitoring, data processing, and scientific visualization support
for the Recursia quantum simulation framework.
"""

import io
import base64
import json
import gzip
import time
import logging
import traceback
import asyncio
import threading
import hashlib
import pickle
import numpy as np
import pandas as pd
from collections import deque, defaultdict, OrderedDict
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy import stats, signal, optimize
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import warnings

from src.core.data_classes import OSHMetrics, SystemHealthProfile

# Suppress matplotlib threading warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Configure logging
logger = logging.getLogger(__name__)

class AdvancedMetricsProcessor:
    """Advanced metrics processing and analysis engine."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.metrics_history = deque(maxlen=history_size)
        self.health_history = deque(maxlen=history_size)
        self.performance_cache = {}
        self.analysis_cache = {}
        self._lock = threading.RLock()
        
    def process_comprehensive_metrics(self, current_metrics: Any) -> Dict[str, Any]:
        """Extract and process comprehensive metrics with advanced analytics."""
        with self._lock:
            try:
                # Extract base metrics
                base_metrics = self._extract_base_metrics(current_metrics)
                
                # Calculate OSH-specific metrics
                osh_metrics = self._calculate_osh_metrics(current_metrics)
                
                # Perform temporal analysis
                temporal_analysis = self._analyze_temporal_patterns()
                
                # Calculate system correlations
                correlations = self._calculate_system_correlations()
                
                # Detect anomalies
                anomalies = self._detect_metric_anomalies(base_metrics)
                
                # Generate predictions
                predictions = self._generate_metric_predictions()
                
                comprehensive_summary = {
                    "timestamp": time.time(),
                    "base_metrics": base_metrics,
                    "osh_metrics": osh_metrics.to_dict(),
                    "temporal_analysis": temporal_analysis,
                    "correlations": correlations,
                    "anomalies": anomalies,
                    "predictions": predictions,
                    "statistical_summary": self._generate_statistical_summary(base_metrics),
                    "complexity_measures": self._calculate_complexity_measures(current_metrics),
                    "entropy_analysis": self._analyze_entropy_dynamics(current_metrics),
                    "coherence_analysis": self._analyze_coherence_dynamics(current_metrics),
                    "recursive_analysis": self._analyze_recursive_patterns(current_metrics),
                    "field_analysis": self._analyze_field_dynamics(current_metrics),
                    "observer_analysis": self._analyze_observer_dynamics(current_metrics),
                    "memory_analysis": self._analyze_memory_patterns(current_metrics),
                    "quantum_analysis": self._analyze_quantum_metrics(current_metrics),
                    "performance_analysis": self._analyze_performance_metrics(current_metrics)
                }
                
                # Store in history
                self.metrics_history.append(comprehensive_summary)
                
                return comprehensive_summary
                
            except Exception as e:
                logger.error(f"Error processing comprehensive metrics: {e}")
                return self._generate_fallback_metrics()
    
    def _extract_base_metrics(self, current_metrics: Any) -> Dict[str, Any]:
        """Extract base metrics with robust error handling."""
        base_metrics = {
            "timestamp": getattr(current_metrics, 'timestamp', time.time()),
            "osh_metrics": {
                "coherence": self._safe_get_metric(current_metrics, 'coherence', 0.0),
                "entropy": self._safe_get_metric(current_metrics, 'entropy', 0.0),
                "strain": self._safe_get_metric(current_metrics, 'strain', 0.0),
                "rsp": self._safe_get_metric(current_metrics, 'rsp', 0.0)
            },
            "quantum_metrics": {
                "states_count": self._safe_get_metric(current_metrics, 'quantum_states_count', 0),
                "entanglement_strength": self._safe_get_metric(current_metrics, 'entanglement_strength', 0.0),
                "total_qubits": self._safe_get_metric(current_metrics, 'total_qubits', 0),
                "measurement_count": self._safe_get_metric(current_metrics, 'measurement_count', 0),
                "gate_operations": self._safe_get_metric(current_metrics, 'gate_operations_count', 0),
                "quantum_fidelity": self._safe_get_metric(current_metrics, 'quantum_fidelity', 0.0),
                "collapse_events": self._safe_get_metric(current_metrics, 'collapse_events', 0),
                "teleportation_events": self._safe_get_metric(current_metrics, 'teleportation_events', 0)
            },
            "observer_metrics": {
                "observer_count": self._safe_get_metric(current_metrics, 'observer_count', 0),
                "active_observers": self._safe_get_metric(current_metrics, 'active_observers', 0),
                "consensus": self._safe_get_metric(current_metrics, 'observer_consensus', 0.0),
                "observation_events": self._safe_get_metric(current_metrics, 'observation_events', 0),
                "phase_transitions": self._safe_get_metric(current_metrics, 'phase_transitions', 0)
            },
            "field_metrics": {
                "field_count": self._safe_get_metric(current_metrics, 'field_count', 0),
                "total_energy": self._safe_get_metric(current_metrics, 'total_field_energy', 0.0),
                "evolution_steps": self._safe_get_metric(current_metrics, 'field_evolution_steps', 0),
                "pde_solver_calls": self._safe_get_metric(current_metrics, 'pde_solver_calls', 0)
            },
            "memory_metrics": {
                "regions": self._safe_get_metric(current_metrics, 'memory_regions', 0),
                "strain_avg": self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0),
                "strain_max": self._safe_get_metric(current_metrics, 'memory_strain_max', 0.0),
                "critical_regions": self._safe_get_metric(current_metrics, 'critical_strain_regions', 0),
                "defragmentation_events": self._safe_get_metric(current_metrics, 'defragmentation_events', 0)
            },
            "recursive_metrics": {
                "depth": self._safe_get_metric(current_metrics, 'recursion_depth', 0),
                "systems": self._safe_get_metric(current_metrics, 'recursive_systems', 0),
                "boundary_crossings": self._safe_get_metric(current_metrics, 'boundary_crossings', 0),
                "recursive_strain": self._safe_get_metric(current_metrics, 'recursive_strain', 0.0)
            },
            "simulation_metrics": {
                "time": self._safe_get_metric(current_metrics, 'simulation_time', 0.0),
                "steps": self._safe_get_metric(current_metrics, 'execution_steps', 0),
                "collapse_events": self._safe_get_metric(current_metrics, 'collapse_events', 0),
                "teleportation_events": self._safe_get_metric(current_metrics, 'teleportation_events', 0)
            },
            "performance_metrics": {
                "render_fps": self._safe_get_metric(current_metrics, 'render_fps', 0.0),
                "memory_usage_mb": self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0),
                "cpu_usage_percent": self._safe_get_metric(current_metrics, 'cpu_usage_percent', 0.0)
            },
            "emergent_phenomena": {
                "detected": self._safe_get_metric(current_metrics, 'emergent_phenomena', []),
                "strength": self._safe_get_metric(current_metrics, 'phenomena_strength', 0.0)
            }
        }
        
        return base_metrics
    
    def _calculate_osh_metrics(self, current_metrics: Any) -> OSHMetrics:
        """Calculate comprehensive OSH metrics."""
        try:
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            observer_count = self._safe_get_metric(current_metrics, 'observer_count', 1)
            
            # Calculate Integrated Information (Φ)
            phi = coherence * (1 - entropy) * np.log(max(observer_count, 1))
            
            # Calculate Kolmogorov Complexity approximation
            kolmogorov_complexity = self._approximate_kolmogorov_complexity(current_metrics)
            
            # Calculate RSP
            entropy_flux = max(0.001, entropy + strain * 0.1)
            rsp = (phi * kolmogorov_complexity) / entropy_flux
            
            # Calculate Information Geometry Curvature
            info_curvature = self._calculate_information_curvature(coherence, entropy, strain)
            
            # Calculate Emergence Index
            emergence_index = self._calculate_emergence_index(current_metrics)
            
            # Calculate Criticality Parameter
            criticality = self._calculate_criticality_parameter(current_metrics)
            
            # Calculate Phase Coherence
            phase_coherence = self._calculate_phase_coherence(current_metrics)
            
            # Calculate Temporal Stability
            temporal_stability = self._calculate_temporal_stability()
            
            return OSHMetrics(
                coherence=coherence,
                entropy=entropy,
                strain=strain,
                rsp=rsp,
                phi=phi,
                kolmogorov_complexity=kolmogorov_complexity,
                information_geometry_curvature=info_curvature,
                recursive_depth=self._safe_get_metric(current_metrics, 'recursion_depth', 0),
                emergence_index=emergence_index,
                criticality_parameter=criticality,
                phase_coherence=phase_coherence,
                temporal_stability=temporal_stability,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error calculating OSH metrics: {e}")
            return OSHMetrics()
    
    def _safe_get_metric(self, metrics_obj: Any, attr_name: str, default: Any) -> Any:
        """Safely extract metric with fallback."""
        try:
            if hasattr(metrics_obj, attr_name):
                value = getattr(metrics_obj, attr_name)
                return value if value is not None else default
            return default
        except (AttributeError, TypeError):
            return default
    
    def _approximate_kolmogorov_complexity(self, current_metrics: Any) -> float:
        """Approximate Kolmogorov complexity using entropy and information measures."""
        try:
            # Gather system state indicators
            quantum_states = self._safe_get_metric(current_metrics, 'quantum_states_count', 0)
            observers = self._safe_get_metric(current_metrics, 'observer_count', 0)
            fields = self._safe_get_metric(current_metrics, 'field_count', 0)
            memory_regions = self._safe_get_metric(current_metrics, 'memory_regions', 0)
            
            # Create histogram of system state
            state_vector = [quantum_states, observers, fields, memory_regions]
            if sum(state_vector) == 0:
                return 0.0
            
            # Calculate entropy of state distribution
            state_probs = np.array(state_vector) / max(sum(state_vector), 1)
            state_probs = state_probs[state_probs > 0]  # Remove zeros
            
            if len(state_probs) == 0:
                return 0.0
                
            entropy = -np.sum(state_probs * np.log2(state_probs + 1e-10))
            
            # Scale by system complexity
            complexity_factor = np.log2(max(sum(state_vector), 1) + 1)
            
            return entropy * complexity_factor
            
        except Exception as e:
            logger.error(f"Error approximating Kolmogorov complexity: {e}")
            return 0.0
    
    def _calculate_information_curvature(self, coherence: float, entropy: float, strain: float) -> float:
        """Calculate information geometry curvature."""
        try:
            # Simulate 2D information field
            x = np.linspace(0, 1, 10)
            y = np.linspace(0, 1, 10)
            X, Y = np.meshgrid(x, y)
            
            # Information density field based on metrics
            I = coherence * np.exp(-entropy * (X**2 + Y**2)) * (1 - strain)
            
            # Calculate second derivatives (discrete approximation)
            dI_dx2 = np.gradient(np.gradient(I, axis=1), axis=1)
            dI_dy2 = np.gradient(np.gradient(I, axis=0), axis=0)
            dI_dxdy = np.gradient(np.gradient(I, axis=1), axis=0)
            
            # Scalar curvature approximation
            curvature = np.mean(dI_dx2 + dI_dy2)
            
            # Information geometry tensor components
            tensor_anisotropy = np.mean(np.abs(dI_dx2 - dI_dy2))
            
            return float(curvature + 0.5 * tensor_anisotropy)
            
        except Exception as e:
            logger.error(f"Error calculating information curvature: {e}")
            return 0.0
    
    def _calculate_emergence_index(self, current_metrics: Any) -> float:
        """Calculate emergence index based on system complexity."""
        try:
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            
            # Emergence as coherence variance relative to entropy correlation
            if len(self.metrics_history) < 2:
                return 0.0
            
            recent_coherence = [m['osh_metrics']['coherence'] for m in list(self.metrics_history)[-10:]]
            recent_entropy = [m['osh_metrics']['entropy'] for m in list(self.metrics_history)[-10:]]
            
            if len(recent_coherence) < 2:
                return 0.0
            
            coherence_var = np.var(recent_coherence)
            entropy_corr = np.corrcoef(recent_coherence, recent_entropy)[0, 1] if len(recent_coherence) > 1 else 0
            
            emergence = coherence_var * entropy / max(strain + 1e-6, 1e-6) * (1 - abs(entropy_corr))
            
            return float(emergence)
            
        except Exception as e:
            logger.error(f"Error calculating emergence index: {e}")
            return 0.0
    
    def _calculate_criticality_parameter(self, current_metrics: Any) -> float:
        """Calculate criticality parameter indicating phase transition proximity."""
        try:
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            
            # Criticality based on proximity to phase boundaries
            coherence_criticality = 1.0 - abs(coherence - 0.5) * 2  # Peak at 0.5
            entropy_criticality = entropy if entropy < 0.5 else (1.0 - entropy)  # Peak at entropy edges
            strain_criticality = strain if strain > 0.7 else 0  # High strain indicates criticality
            
            # Combined with avalanche scaling approximation
            if len(self.metrics_history) > 5:
                recent_changes = []
                for i in range(1, min(6, len(self.metrics_history))):
                    prev_metrics = list(self.metrics_history)[-i-1]['osh_metrics']
                    curr_metrics = list(self.metrics_history)[-i]['osh_metrics']
                    change = abs(curr_metrics['coherence'] - prev_metrics['coherence'])
                    recent_changes.append(change)
                
                if recent_changes and max(recent_changes) > 0:
                    # Power law scaling approximation
                    changes_sorted = sorted(recent_changes, reverse=True)
                    scaling_exponent = -np.log(changes_sorted[-1] / max(changes_sorted[0], 1e-6)) / np.log(len(changes_sorted))
                    avalanche_criticality = min(abs(scaling_exponent - 1.0), 1.0)  # Critical exponent ≈ 1
                else:
                    avalanche_criticality = 0.0
            else:
                avalanche_criticality = 0.0
            
            criticality = (coherence_criticality + entropy_criticality + strain_criticality + avalanche_criticality) / 4.0
            
            return float(criticality)
            
        except Exception as e:
            logger.error(f"Error calculating criticality parameter: {e}")
            return 0.0
    
    def _calculate_phase_coherence(self, current_metrics: Any) -> float:
        """Calculate phase coherence using FFT analysis."""
        try:
            if len(self.metrics_history) < 8:
                return 0.0
            
            # Extract coherence time series
            coherence_series = [m['osh_metrics']['coherence'] for m in list(self.metrics_history)[-16:]]
            
            if len(coherence_series) < 8:
                return 0.0
            
            # Apply FFT
            fft_result = np.fft.fft(coherence_series)
            phases = np.angle(fft_result)
            
            # Calculate phase coherence (phase locking)
            phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
            
            return float(phase_coherence)
            
        except Exception as e:
            logger.error(f"Error calculating phase coherence: {e}")
            return 0.0
    
    def _calculate_temporal_stability(self) -> float:
        """Calculate temporal stability of the system."""
        try:
            if len(self.metrics_history) < 5:
                return 1.0
            
            # Extract key metrics over time
            recent_metrics = list(self.metrics_history)[-10:]
            
            coherence_values = [m['osh_metrics']['coherence'] for m in recent_metrics]
            entropy_values = [m['osh_metrics']['entropy'] for m in recent_metrics]
            strain_values = [m['osh_metrics']['strain'] for m in recent_metrics]
            
            # Calculate coefficient of variation for each metric
            coherence_cv = np.std(coherence_values) / (np.mean(coherence_values) + 1e-6)
            entropy_cv = np.std(entropy_values) / (np.mean(entropy_values) + 1e-6)
            strain_cv = np.std(strain_values) / (np.mean(strain_values) + 1e-6)
            
            # Stability is inverse of variability
            stability = 1.0 / (1.0 + coherence_cv + entropy_cv + strain_cv)
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Error calculating temporal stability: {e}")
            return 1.0
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in metrics."""
        if len(self.metrics_history) < 3:
            return {"insufficient_data": True}
        
        try:
            recent_metrics = list(self.metrics_history)[-20:]
            
            # Extract time series
            timestamps = [m['timestamp'] for m in recent_metrics]
            coherence_series = [m['osh_metrics']['coherence'] for m in recent_metrics]
            entropy_series = [m['osh_metrics']['entropy'] for m in recent_metrics]
            
            analysis = {
                "trend_analysis": self._analyze_trends(timestamps, coherence_series, entropy_series),
                "periodicity": self._detect_periodicity(coherence_series),
                "stability_assessment": self._assess_stability(coherence_series, entropy_series),
                "change_points": self._detect_change_points(coherence_series),
                "forecast": self._generate_short_term_forecast(coherence_series, entropy_series)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing temporal patterns: {e}")
            return {"error": str(e)}
    
    def _analyze_trends(self, timestamps: List[float], coherence: List[float], entropy: List[float]) -> Dict[str, Any]:
        """Analyze trends in key metrics."""
        try:
            if len(timestamps) < 3:
                return {"insufficient_data": True}
            
            # Normalize timestamps
            time_deltas = np.array(timestamps) - timestamps[0]
            
            # Linear regression for trends
            coherence_slope, coherence_intercept, coherence_r, _, _ = stats.linregress(time_deltas, coherence)
            entropy_slope, entropy_intercept, entropy_r, _, _ = stats.linregress(time_deltas, entropy)
            
            return {
                "coherence_trend": {
                    "slope": float(coherence_slope),
                    "direction": "increasing" if coherence_slope > 0.01 else "decreasing" if coherence_slope < -0.01 else "stable",
                    "strength": float(abs(coherence_r)),
                    "confidence": float(coherence_r**2)
                },
                "entropy_trend": {
                    "slope": float(entropy_slope),
                    "direction": "increasing" if entropy_slope > 0.01 else "decreasing" if entropy_slope < -0.01 else "stable",
                    "strength": float(abs(entropy_r)),
                    "confidence": float(entropy_r**2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {"error": str(e)}
    
    def _detect_periodicity(self, series: List[float]) -> Dict[str, Any]:
        """Detect periodic patterns in time series."""
        try:
            if len(series) < 8:
                return {"insufficient_data": True}
            
            # Autocorrelation analysis
            series_array = np.array(series)
            autocorr = np.correlate(series_array, series_array, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(autocorr[1:], height=0.3, distance=2)
            
            periodicity_detected = len(peaks) > 0
            
            return {
                "periodic": periodicity_detected,
                "dominant_period": int(peaks[0] + 1) if periodicity_detected else None,
                "strength": float(max(autocorr[peaks + 1])) if periodicity_detected else 0.0,
                "multiple_periods": len(peaks) > 1
            }
            
        except Exception as e:
            logger.error(f"Error detecting periodicity: {e}")
            return {"error": str(e)}
    
    def _assess_stability(self, coherence: List[float], entropy: List[float]) -> Dict[str, Any]:
        """Assess system stability."""
        try:
            if len(coherence) < 3:
                return {"insufficient_data": True}
            
            # Calculate various stability metrics
            coherence_cv = np.std(coherence) / (np.mean(coherence) + 1e-6)
            entropy_cv = np.std(entropy) / (np.mean(entropy) + 1e-6)
            
            # Lyapunov-like exponent approximation
            diffs = np.diff(coherence)
            if len(diffs) > 1 and np.std(diffs) > 0:
                lyapunov_approx = np.mean(np.log(np.abs(diffs) + 1e-6))
            else:
                lyapunov_approx = 0.0
            
            # Overall stability score
            stability_score = 1.0 / (1.0 + coherence_cv + entropy_cv + abs(lyapunov_approx))
            
            # Classify stability
            if stability_score > 0.8:
                stability_class = "highly_stable"
            elif stability_score > 0.6:
                stability_class = "stable"
            elif stability_score > 0.4:
                stability_class = "moderately_stable"
            elif stability_score > 0.2:
                stability_class = "unstable"
            else:
                stability_class = "chaotic"
            
            return {
                "stability_score": float(stability_score),
                "stability_class": stability_class,
                "coherence_variability": float(coherence_cv),
                "entropy_variability": float(entropy_cv),
                "lyapunov_estimate": float(lyapunov_approx)
            }
            
        except Exception as e:
            logger.error(f"Error assessing stability: {e}")
            return {"error": str(e)}
    
    def _detect_change_points(self, series: List[float]) -> List[int]:
        """Detect change points in time series."""
        try:
            if len(series) < 5:
                return []
            
            series_array = np.array(series)
            
            # Simple change point detection using moving variance
            window_size = min(3, len(series) // 3)
            change_points = []
            
            for i in range(window_size, len(series) - window_size):
                left_var = np.var(series_array[i-window_size:i])
                right_var = np.var(series_array[i:i+window_size])
                
                # Significant change in variance indicates potential change point
                if abs(left_var - right_var) > 0.1 * (left_var + right_var + 1e-6):
                    change_points.append(i)
            
            return change_points
            
        except Exception as e:
            logger.error(f"Error detecting change points: {e}")
            return []
    
    def _generate_short_term_forecast(self, coherence: List[float], entropy: List[float]) -> Dict[str, Any]:
        """Generate short-term forecast."""
        try:
            if len(coherence) < 5:
                return {"insufficient_data": True}
            
            # Simple linear extrapolation
            time_points = np.arange(len(coherence))
            
            # Fit polynomial trends
            coherence_poly = np.polyfit(time_points, coherence, deg=min(2, len(coherence)-1))
            entropy_poly = np.polyfit(time_points, entropy, deg=min(2, len(entropy)-1))
            
            # Forecast next 3 points
            future_points = np.arange(len(coherence), len(coherence) + 3)
            coherence_forecast = np.polyval(coherence_poly, future_points)
            entropy_forecast = np.polyval(entropy_poly, future_points)
            
            # Ensure reasonable bounds
            coherence_forecast = np.clip(coherence_forecast, 0.0, 1.0)
            entropy_forecast = np.clip(entropy_forecast, 0.0, 2.0)
            
            return {
                "forecast_horizon": 3,
                "coherence_forecast": coherence_forecast.tolist(),
                "entropy_forecast": entropy_forecast.tolist(),
                "confidence": "low"  # Simple method has low confidence
            }
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            return {"error": str(e)}
    
    def _calculate_system_correlations(self) -> Dict[str, Any]:
        """Calculate correlations between system components."""
        if len(self.metrics_history) < 5:
            return {"insufficient_data": True}
        
        try:
            recent_metrics = list(self.metrics_history)[-20:]
            
            # Extract metrics for correlation analysis
            metrics_matrix = []
            for m in recent_metrics:
                osh = m['osh_metrics']
                quantum = m['base_metrics']['quantum_metrics']
                observer = m['base_metrics']['observer_metrics']
                
                row = [
                    osh['coherence'], osh['entropy'], osh['strain'], osh['rsp'],
                    quantum['entanglement_strength'], quantum['quantum_fidelity'],
                    observer['consensus'], float(observer['active_observers'])
                ]
                metrics_matrix.append(row)
            
            if len(metrics_matrix) < 3:
                return {"insufficient_data": True}
            
            metrics_array = np.array(metrics_matrix)
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(metrics_array.T)
            
            metric_names = [
                'coherence', 'entropy', 'strain', 'rsp',
                'entanglement', 'fidelity', 'consensus', 'active_observers'
            ]
            
            # Extract significant correlations
            significant_correlations = {}
            for i in range(len(metric_names)):
                for j in range(i+1, len(metric_names)):
                    corr_val = correlation_matrix[i, j]
                    if abs(corr_val) > 0.5:  # Significant correlation threshold
                        key = f"{metric_names[i]}_vs_{metric_names[j]}"
                        significant_correlations[key] = {
                            "correlation": float(corr_val),
                            "strength": "strong" if abs(corr_val) > 0.7 else "moderate",
                            "direction": "positive" if corr_val > 0 else "negative"
                        }
            
            return {
                "correlation_matrix": correlation_matrix.tolist(),
                "metric_names": metric_names,
                "significant_correlations": significant_correlations,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {"error": str(e)}
    
    def _detect_metric_anomalies(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in current metrics."""
        if len(self.metrics_history) < 10:
            return {"insufficient_data": True}
        
        try:
            # Extract historical values for comparison
            historical_coherence = [m['osh_metrics']['coherence'] for m in list(self.metrics_history)[-20:]]
            historical_entropy = [m['osh_metrics']['entropy'] for m in list(self.metrics_history)[-20:]]
            historical_strain = [m['osh_metrics']['strain'] for m in list(self.metrics_history)[-20:]]
            
            current_coherence = base_metrics['osh_metrics']['coherence']
            current_entropy = base_metrics['osh_metrics']['entropy']
            current_strain = base_metrics['osh_metrics']['strain']
            
            anomalies = {}
            
            # Z-score based anomaly detection
            for metric_name, historical, current in [
                ('coherence', historical_coherence, current_coherence),
                ('entropy', historical_entropy, current_entropy),
                ('strain', historical_strain, current_strain)
            ]:
                if len(historical) > 2:
                    mean_val = np.mean(historical)
                    std_val = np.std(historical)
                    
                    if std_val > 0:
                        z_score = abs(current - mean_val) / std_val
                        
                        if z_score > 2.5:  # Significant anomaly
                            anomalies[metric_name] = {
                                "anomaly_detected": True,
                                "z_score": float(z_score),
                                "severity": "high" if z_score > 3.0 else "moderate",
                                "direction": "high" if current > mean_val else "low",
                                "current_value": float(current),
                                "historical_mean": float(mean_val),
                                "historical_std": float(std_val)
                            }
            
            return {
                "anomalies_detected": len(anomalies) > 0,
                "anomaly_count": len(anomalies),
                "anomaly_details": anomalies,
                "detection_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {"error": str(e)}
    
    def _generate_metric_predictions(self) -> Dict[str, Any]:
        """Generate predictions for key metrics."""
        if len(self.metrics_history) < 5:
            return {"insufficient_data": True}
        
        try:
            # Use the forecast method from temporal analysis
            recent_metrics = list(self.metrics_history)[-10:]
            coherence_series = [m['osh_metrics']['coherence'] for m in recent_metrics]
            entropy_series = [m['osh_metrics']['entropy'] for m in recent_metrics]
            
            forecast = self._generate_short_term_forecast(coherence_series, entropy_series)
            
            # Add prediction intervals and confidence
            predictions = {
                "prediction_horizon": 3,
                "predictions": forecast,
                "methodology": "polynomial_extrapolation",
                "confidence_level": "low",
                "prediction_timestamp": time.time()
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"error": str(e)}
    
    def _generate_statistical_summary(self, base_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        try:
            summary = {
                "descriptive_statistics": {},
                "distribution_analysis": {},
                "trend_indicators": {}
            }
            
            if len(self.metrics_history) >= 3:
                # Extract time series for analysis
                coherence_history = [m['osh_metrics']['coherence'] for m in list(self.metrics_history)[-20:]]
                entropy_history = [m['osh_metrics']['entropy'] for m in list(self.metrics_history)[-20:]]
                
                for name, series in [('coherence', coherence_history), ('entropy', entropy_history)]:
                    if len(series) > 2:
                        summary["descriptive_statistics"][name] = {
                            "mean": float(np.mean(series)),
                            "std": float(np.std(series)),
                            "min": float(np.min(series)),
                            "max": float(np.max(series)),
                            "median": float(np.median(series)),
                            "skewness": float(stats.skew(series)) if len(series) > 2 else 0.0,
                            "kurtosis": float(stats.kurtosis(series)) if len(series) > 2 else 0.0
                        }
                        
                        # Distribution analysis
                        _, p_value = stats.normaltest(series) if len(series) > 7 else (0, 1)
                        summary["distribution_analysis"][name] = {
                            "normality_p_value": float(p_value),
                            "is_normal": p_value > 0.05,
                            "sample_size": len(series)
                        }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating statistical summary: {e}")
            return {"error": str(e)}
    
    def _calculate_complexity_measures(self, current_metrics: Any) -> Dict[str, Any]:
        """Calculate various complexity measures."""
        try:
            complexity_measures = {
                "kolmogorov_approximation": self._approximate_kolmogorov_complexity(current_metrics),
                "logical_depth": self._calculate_logical_depth(current_metrics),
                "effective_complexity": self._calculate_effective_complexity(current_metrics),
                "thermodynamic_depth": self._calculate_thermodynamic_depth(current_metrics)
            }
            
            return complexity_measures
            
        except Exception as e:
            logger.error(f"Error calculating complexity measures: {e}")
            return {"error": str(e)}
    
    def _calculate_logical_depth(self, current_metrics: Any) -> float:
        """Calculate logical depth approximation."""
        try:
            # Logical depth approximated by processing steps and recursive depth
            execution_steps = self._safe_get_metric(current_metrics, 'execution_steps', 0)
            recursion_depth = self._safe_get_metric(current_metrics, 'recursion_depth', 0)
            gate_operations = self._safe_get_metric(current_metrics, 'gate_operations_count', 0)
            
            # Combine measures with logarithmic scaling
            logical_depth = np.log(max(execution_steps, 1)) + recursion_depth + np.log(max(gate_operations, 1))
            
            return float(logical_depth)
            
        except Exception as e:
            logger.error(f"Error calculating logical depth: {e}")
            return 0.0
    
    def _calculate_effective_complexity(self, current_metrics: Any) -> float:
        """Calculate effective complexity (regularities vs randomness)."""
        try:
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            
            # Effective complexity peaks at intermediate values
            # Regular (low entropy) and random (high entropy) have low effective complexity
            # Structured randomness has high effective complexity
            effective_complexity = coherence * entropy * (1 - entropy)
            
            return float(effective_complexity)
            
        except Exception as e:
            logger.error(f"Error calculating effective complexity: {e}")
            return 0.0
    
    def _calculate_thermodynamic_depth(self, current_metrics: Any) -> float:
        """Calculate thermodynamic depth approximation."""
        try:
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            field_energy = self._safe_get_metric(current_metrics, 'total_field_energy', 0.0)
            
            # Thermodynamic depth related to entropy production and energy dissipation
            thermodynamic_depth = entropy * np.log(max(field_energy, 1) + 1)
            
            return float(thermodynamic_depth)
            
        except Exception as e:
            logger.error(f"Error calculating thermodynamic depth: {e}")
            return 0.0
    
    def _analyze_entropy_dynamics(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze entropy dynamics and flow."""
        try:
            current_entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            
            entropy_analysis = {
                "current_entropy": float(current_entropy),
                "entropy_classification": self._classify_entropy_level(current_entropy),
                "entropy_flow_rate": self._calculate_entropy_flow_rate(),
                "entropy_production": self._calculate_entropy_production(current_metrics),
                "entropy_gradients": self._calculate_entropy_gradients(current_metrics)
            }
            
            return entropy_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing entropy dynamics: {e}")
            return {"error": str(e)}
    
    def _classify_entropy_level(self, entropy: float) -> str:
        """Classify entropy level."""
        if entropy < 0.2:
            return "very_low"
        elif entropy < 0.4:
            return "low"
        elif entropy < 0.6:
            return "moderate"
        elif entropy < 0.8:
            return "high"
        else:
            return "very_high"
    
    def _calculate_entropy_flow_rate(self) -> float:
        """Calculate entropy flow rate."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0
            
            recent_entropies = [m['osh_metrics']['entropy'] for m in list(self.metrics_history)[-5:]]
            
            if len(recent_entropies) < 2:
                return 0.0
            
            # Calculate rate of change
            flow_rate = (recent_entropies[-1] - recent_entropies[0]) / max(len(recent_entropies) - 1, 1)
            
            return float(flow_rate)
            
        except Exception as e:
            logger.error(f"Error calculating entropy flow rate: {e}")
            return 0.0
    
    def _calculate_entropy_production(self, current_metrics: Any) -> float:
        """Calculate entropy production rate."""
        try:
            # Entropy production from irreversible processes
            collapse_events = self._safe_get_metric(current_metrics, 'collapse_events', 0)
            measurement_count = self._safe_get_metric(current_metrics, 'measurement_count', 0)
            defrag_events = self._safe_get_metric(current_metrics, 'defragmentation_events', 0)
            
            # Irreversible processes produce entropy
            entropy_production = np.log(max(collapse_events + measurement_count + defrag_events, 1))
            
            return float(entropy_production)
            
        except Exception as e:
            logger.error(f"Error calculating entropy production: {e}")
            return 0.0
    
    def _calculate_entropy_gradients(self, current_metrics: Any) -> Dict[str, float]:
        """Calculate entropy gradients across different subsystems."""
        try:
            # Simulate entropy distribution across subsystems
            quantum_entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0) * 0.4
            memory_entropy = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0) * 0.3
            observer_entropy = (1.0 - self._safe_get_metric(current_metrics, 'observer_consensus', 0.0)) * 0.2
            field_entropy = min(self._safe_get_metric(current_metrics, 'total_field_energy', 0.0) / 10.0, 1.0) * 0.1
            
            gradients = {
                "quantum_entropy": float(quantum_entropy),
                "memory_entropy": float(memory_entropy),
                "observer_entropy": float(observer_entropy),
                "field_entropy": float(field_entropy),
                "total_gradient": float(quantum_entropy + memory_entropy + observer_entropy + field_entropy)
            }
            
            return gradients
            
        except Exception as e:
            logger.error(f"Error calculating entropy gradients: {e}")
            return {"error": str(e)}
    
    def _analyze_coherence_dynamics(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze coherence dynamics."""
        try:
            current_coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            
            coherence_analysis = {
                "current_coherence": float(current_coherence),
                "coherence_classification": self._classify_coherence_level(current_coherence),
                "coherence_stability": self._calculate_coherence_stability(),
                "decoherence_rate": self._calculate_decoherence_rate(),
                "coherence_recovery": self._calculate_coherence_recovery_potential(current_metrics)
            }
            
            return coherence_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing coherence dynamics: {e}")
            return {"error": str(e)}
    
    def _classify_coherence_level(self, coherence: float) -> str:
        """Classify coherence level."""
        if coherence > 0.9:
            return "excellent"
        elif coherence > 0.7:
            return "good"
        elif coherence > 0.5:
            return "moderate"
        elif coherence > 0.3:
            return "poor"
        else:
            return "critical"
    
    def _calculate_coherence_stability(self) -> float:
        """Calculate coherence stability."""
        try:
            if len(self.metrics_history) < 3:
                return 1.0
            
            recent_coherence = [m['osh_metrics']['coherence'] for m in list(self.metrics_history)[-10:]]
            
            if len(recent_coherence) < 2:
                return 1.0
            
            # Stability as inverse of coefficient of variation
            mean_coherence = np.mean(recent_coherence)
            std_coherence = np.std(recent_coherence)
            
            stability = 1.0 / (1.0 + std_coherence / max(mean_coherence, 1e-6))
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Error calculating coherence stability: {e}")
            return 1.0
    
    def _calculate_decoherence_rate(self) -> float:
        """Calculate decoherence rate."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0
            
            recent_coherence = [m['osh_metrics']['coherence'] for m in list(self.metrics_history)[-5:]]
            
            if len(recent_coherence) < 2:
                return 0.0
            
            # Calculate rate of coherence loss
            coherence_change = recent_coherence[0] - recent_coherence[-1]
            time_span = max(len(recent_coherence) - 1, 1)
            
            decoherence_rate = max(coherence_change / time_span, 0.0)  # Only positive rates
            
            return float(decoherence_rate)
            
        except Exception as e:
            logger.error(f"Error calculating decoherence rate: {e}")
            return 0.0
    
    def _calculate_coherence_recovery_potential(self, current_metrics: Any) -> float:
        """Calculate potential for coherence recovery."""
        try:
            current_coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            current_strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            observer_consensus = self._safe_get_metric(current_metrics, 'observer_consensus', 0.0)
            
            # Recovery potential based on available resources and system state
            recovery_potential = (1.0 - current_coherence) * (1.0 - current_strain) * observer_consensus
            
            return float(recovery_potential)
            
        except Exception as e:
            logger.error(f"Error calculating coherence recovery potential: {e}")
            return 0.0
    
    def _analyze_recursive_patterns(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze recursive patterns and hierarchy."""
        try:
            recursion_depth = self._safe_get_metric(current_metrics, 'recursion_depth', 0)
            recursive_systems = self._safe_get_metric(current_metrics, 'recursive_systems', 0)
            boundary_crossings = self._safe_get_metric(current_metrics, 'boundary_crossings', 0)
            
            recursive_analysis = {
                "current_depth": int(recursion_depth),
                "system_count": int(recursive_systems),
                "boundary_activity": int(boundary_crossings),
                "recursive_complexity": self._calculate_recursive_complexity(current_metrics),
                "hierarchy_stability": self._calculate_hierarchy_stability(current_metrics),
                "recursive_efficiency": self._calculate_recursive_efficiency(current_metrics)
            }
            
            return recursive_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing recursive patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_recursive_complexity(self, current_metrics: Any) -> float:
        """Calculate recursive complexity measure."""
        try:
            depth = self._safe_get_metric(current_metrics, 'recursion_depth', 0)
            systems = self._safe_get_metric(current_metrics, 'recursive_systems', 0)
            
            # Complexity as depth * log(systems)
            complexity = depth * np.log(max(systems, 1) + 1)
            
            return float(complexity)
            
        except Exception as e:
            logger.error(f"Error calculating recursive complexity: {e}")
            return 0.0
    
    def _calculate_hierarchy_stability(self, current_metrics: Any) -> float:
        """Calculate hierarchy stability."""
        try:
            if len(self.metrics_history) < 3:
                return 1.0
            
            recent_depths = [self._safe_get_metric(m, 'recursive_depth', 0) 
                           for m in list(self.metrics_history)[-10:]]
            
            if len(recent_depths) < 2:
                return 1.0
            
            # Stability as inverse of depth variance
            depth_variance = np.var(recent_depths)
            stability = 1.0 / (1.0 + depth_variance)
            
            return float(stability)
            
        except Exception in Exception as e:
            logger.error(f"Error calculating hierarchy stability: {e}")
            return 1.0
    
    def _calculate_recursive_efficiency(self, current_metrics: Any) -> float:
        """Calculate recursive efficiency."""
        try:
            depth = self._safe_get_metric(current_metrics, 'recursion_depth', 0)
            systems = self._safe_get_metric(current_metrics, 'recursive_systems', 0)
            boundary_crossings = self._safe_get_metric(current_metrics, 'boundary_crossings', 0)
            
            if depth == 0:
                return 1.0
            
            # Efficiency as useful work (systems) per depth unit, penalized by boundary crossings
            efficiency = systems / max(depth, 1) / max(1 + boundary_crossings * 0.1, 1)
            
            return float(min(efficiency, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating recursive efficiency: {e}")
            return 0.0
    
    def _analyze_field_dynamics(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze field dynamics."""
        try:
            field_count = self._safe_get_metric(current_metrics, 'field_count', 0)
            total_energy = self._safe_get_metric(current_metrics, 'total_field_energy', 0.0)
            evolution_steps = self._safe_get_metric(current_metrics, 'field_evolution_steps', 0)
            
            field_analysis = {
                "field_count": int(field_count),
                "total_energy": float(total_energy),
                "evolution_activity": int(evolution_steps),
                "energy_density": float(total_energy / max(field_count, 1)),
                "field_coupling_strength": self._calculate_field_coupling_strength(current_metrics),
                "field_stability": self._calculate_field_stability(current_metrics)
            }
            
            return field_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing field dynamics: {e}")
            return {"error": str(e)}
    
    def _calculate_field_coupling_strength(self, current_metrics: Any) -> float:
        """Calculate field coupling strength."""
        try:
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            field_energy = self._safe_get_metric(current_metrics, 'total_field_energy', 0.0)
            
            # Coupling strength as correlation between coherence and field energy
            coupling_strength = coherence * np.tanh(field_energy / 10.0)  # Saturating function
            
            return float(coupling_strength)
            
        except Exception as e:
            logger.error(f"Error calculating field coupling strength: {e}")
            return 0.0
    
    def _calculate_field_stability(self, current_metrics: Any) -> float:
        """Calculate field stability."""
        try:
            if len(self.metrics_history) < 3:
                return 1.0
            
            recent_energies = [self._safe_get_metric(m['base_metrics']['field_metrics'], 'total_energy', 0.0)
                             for m in list(self.metrics_history)[-10:]]
            
            if len(recent_energies) < 2:
                return 1.0
            
            # Stability as inverse of energy variance
            energy_variance = np.var(recent_energies)
            stability = 1.0 / (1.0 + energy_variance)
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Error calculating field stability: {e}")
            return 1.0
    
    def _analyze_observer_dynamics(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze observer dynamics."""
        try:
            observer_count = self._safe_get_metric(current_metrics, 'observer_count', 0)
            active_observers = self._safe_get_metric(current_metrics, 'active_observers', 0)
            consensus = self._safe_get_metric(current_metrics, 'observer_consensus', 0.0)
            
            observer_analysis = {
                "total_observers": int(observer_count),
                "active_observers": int(active_observers),
                "activity_ratio": float(active_observers / max(observer_count, 1)),
                "consensus_level": float(consensus),
                "consensus_stability": self._calculate_consensus_stability(),
                "observer_efficiency": self._calculate_observer_efficiency(current_metrics)
            }
            
            return observer_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing observer dynamics: {e}")
            return {"error": str(e)}
    
    def _calculate_consensus_stability(self) -> float:
        """Calculate consensus stability."""
        try:
            if len(self.metrics_history) < 3:
                return 1.0
            
            recent_consensus = [self._safe_get_metric(m['base_metrics']['observer_metrics'], 'consensus', 0.0)
                              for m in list(self.metrics_history)[-10:]]
            
            if len(recent_consensus) < 2:
                return 1.0
            
            # Stability as inverse of consensus variance
            consensus_variance = np.var(recent_consensus)
            stability = 1.0 / (1.0 + consensus_variance)
            
            return float(stability)
            
        except Exception as e:
            logger.error(f"Error calculating consensus stability: {e}")
            return 1.0
    
    def _calculate_observer_efficiency(self, current_metrics: Any) -> float:
        """Calculate observer efficiency."""
        try:
            active_observers = self._safe_get_metric(current_metrics, 'active_observers', 0)
            observation_events = self._safe_get_metric(current_metrics, 'observation_events', 0)
            consensus = self._safe_get_metric(current_metrics, 'observer_consensus', 0.0)
            
            if active_observers == 0:
                return 0.0
            
            # Efficiency as consensus-weighted observations per active observer
            efficiency = (observation_events / active_observers) * consensus
            
            return float(min(efficiency, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating observer efficiency: {e}")
            return 0.0
    
    def _analyze_memory_patterns(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze memory patterns."""
        try:
            memory_regions = self._safe_get_metric(current_metrics, 'memory_regions', 0)
            strain_avg = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0)
            strain_max = self._safe_get_metric(current_metrics, 'memory_strain_max', 0.0)
            critical_regions = self._safe_get_metric(current_metrics, 'critical_strain_regions', 0)
            
            memory_analysis = {
                "total_regions": int(memory_regions),
                "average_strain": float(strain_avg),
                "maximum_strain": float(strain_max),
                "critical_regions": int(critical_regions),
                "strain_distribution": self._calculate_strain_distribution(current_metrics),
                "memory_fragmentation": self._calculate_memory_fragmentation(current_metrics),
                "memory_efficiency": self._calculate_memory_efficiency(current_metrics)
            }
            
            return memory_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing memory patterns: {e}")
            return {"error": str(e)}
    
    def _calculate_strain_distribution(self, current_metrics: Any) -> Dict[str, float]:
        """Calculate strain distribution statistics."""
        try:
            strain_avg = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0)
            strain_max = self._safe_get_metric(current_metrics, 'memory_strain_max', 0.0)
            critical_regions = self._safe_get_metric(current_metrics, 'critical_strain_regions', 0)
            total_regions = self._safe_get_metric(current_metrics, 'memory_regions', 1)
            
            distribution = {
                "mean_strain": float(strain_avg),
                "peak_strain": float(strain_max),
                "strain_variance": float((strain_max - strain_avg) ** 2),  # Approximation
                "critical_fraction": float(critical_regions / max(total_regions, 1))
            }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error calculating strain distribution: {e}")
            return {"error": str(e)}
    
    def _calculate_memory_fragmentation(self, current_metrics: Any) -> float:
        """Calculate memory fragmentation level."""
        try:
            strain_avg = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0)
            strain_max = self._safe_get_metric(current_metrics, 'memory_strain_max', 0.0)
            
            # Fragmentation as strain heterogeneity
            fragmentation = strain_max - strain_avg
            
            return float(fragmentation)
            
        except Exception as e:
            logger.error(f"Error calculating memory fragmentation: {e}")
            return 0.0
    
    def _calculate_memory_efficiency(self, current_metrics: Any) -> float:
        """Calculate memory efficiency."""
        try:
            strain_avg = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0)
            memory_usage = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            defrag_events = self._safe_get_metric(current_metrics, 'defragmentation_events', 0)
            
            # Efficiency as inverse of strain, adjusted for memory usage and defrag activity
            base_efficiency = 1.0 - strain_avg
            usage_penalty = min(memory_usage / 1000.0, 1.0) * 0.1  # Small penalty for high usage
            defrag_bonus = min(defrag_events / 10.0, 0.1)  # Small bonus for defragmentation activity
            
            efficiency = base_efficiency - usage_penalty + defrag_bonus
            
            return float(max(0.0, min(1.0, efficiency)))
            
        except Exception as e:
            logger.error(f"Error calculating memory efficiency: {e}")
            return 0.0
    
    def _analyze_quantum_metrics(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze quantum metrics."""
        try:
            quantum_states = self._safe_get_metric(current_metrics, 'quantum_states_count', 0)
            total_qubits = self._safe_get_metric(current_metrics, 'total_qubits', 0)
            entanglement = self._safe_get_metric(current_metrics, 'entanglement_strength', 0.0)
            fidelity = self._safe_get_metric(current_metrics, 'quantum_fidelity', 0.0)
            
            quantum_analysis = {
                "state_count": int(quantum_states),
                "total_qubits": int(total_qubits),
                "average_qubits_per_state": float(total_qubits / max(quantum_states, 1)),
                "entanglement_strength": float(entanglement),
                "quantum_fidelity": float(fidelity),
                "quantum_complexity": self._calculate_quantum_complexity(current_metrics),
                "quantum_coherence_quality": self._calculate_quantum_coherence_quality(current_metrics)
            }
            
            return quantum_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing quantum metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_quantum_complexity(self, current_metrics: Any) -> float:
        """Calculate quantum complexity measure."""
        try:
            qubits = self._safe_get_metric(current_metrics, 'total_qubits', 0)
            entanglement = self._safe_get_metric(current_metrics, 'entanglement_strength', 0.0)
            gate_ops = self._safe_get_metric(current_metrics, 'gate_operations_count', 0)
            
            # Quantum complexity as log of Hilbert space dimension, weighted by entanglement and operations
            hilbert_dimension = 2 ** qubits if qubits < 20 else float('inf')  # Avoid overflow
            
            if hilbert_dimension == float('inf'):
                complexity = qubits * np.log(2) * entanglement * np.log(max(gate_ops, 1) + 1)
            else:
                complexity = np.log(hilbert_dimension) * entanglement * np.log(max(gate_ops, 1) + 1)
            
            return float(complexity)
            
        except Exception as e:
            logger.error(f"Error calculating quantum complexity: {e}")
            return 0.0
    
    def _calculate_quantum_coherence_quality(self, current_metrics: Any) -> float:
        """Calculate quantum coherence quality."""
        try:
            fidelity = self._safe_get_metric(current_metrics, 'quantum_fidelity', 0.0)
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            entanglement = self._safe_get_metric(current_metrics, 'entanglement_strength', 0.0)
            
            # Quality as combination of fidelity, coherence, and entanglement
            quality = (fidelity + coherence + entanglement) / 3.0
            
            return float(quality)
            
        except Exception as e:
            logger.error(f"Error calculating quantum coherence quality: {e}")
            return 0.0
    
    def _analyze_performance_metrics(self, current_metrics: Any) -> Dict[str, Any]:
        """Analyze performance metrics."""
        try:
            fps = self._safe_get_metric(current_metrics, 'render_fps', 0.0)
            memory_usage = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            cpu_usage = self._safe_get_metric(current_metrics, 'cpu_usage_percent', 0.0)
            
            performance_analysis = {
                "render_fps": float(fps),
                "memory_usage_mb": float(memory_usage),
                "cpu_usage_percent": float(cpu_usage),
                "performance_score": self._calculate_performance_score(fps, memory_usage, cpu_usage),
                "resource_efficiency": self._calculate_resource_efficiency(current_metrics),
                "bottleneck_analysis": self._identify_performance_bottlenecks(fps, memory_usage, cpu_usage)
            }
            
            return performance_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance metrics: {e}")
            return {"error": str(e)}
    
    def _calculate_performance_score(self, fps: float, memory_usage: float, cpu_usage: float) -> float:
        """Calculate overall performance score."""
        try:
            # Normalize metrics (higher is better for fps, lower is better for usage)
            fps_score = min(fps / 60.0, 1.0)  # Target 60 FPS
            memory_score = max(1.0 - memory_usage / 2000.0, 0.0)  # Target under 2GB
            cpu_score = max(1.0 - cpu_usage / 100.0, 0.0)  # Target under 100%
            
            # Weighted average
            performance_score = 0.4 * fps_score + 0.3 * memory_score + 0.3 * cpu_score
            
            return float(performance_score)
            
        except Exception as e:
            logger.error(f"Error calculating performance score: {e}")
            return 0.0
    
    def _calculate_resource_efficiency(self, current_metrics: Any) -> float:
        """Calculate resource efficiency."""
        try:
            quantum_states = self._safe_get_metric(current_metrics, 'quantum_states_count', 0)
            observers = self._safe_get_metric(current_metrics, 'observer_count', 0)
            fields = self._safe_get_metric(current_metrics, 'field_count', 0)
            memory_usage = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            
            # Efficiency as useful work per resource unit
            useful_work = quantum_states + observers + fields
            resource_usage = max(memory_usage / 100.0, 1.0)  # Normalize to reasonable scale
            
            efficiency = useful_work / resource_usage
            
            return float(min(efficiency, 1.0))
            
        except Exception as e:
            logger.error(f"Error calculating resource efficiency: {e}")
            return 0.0
    
    def _identify_performance_bottlenecks(self, fps: float, memory_usage: float, cpu_usage: float) -> List[str]:
        """Identify performance bottlenecks."""
        try:
            bottlenecks = []
            
            if fps < 30:
                bottlenecks.append("low_fps")
            if memory_usage > 1500:
                bottlenecks.append("high_memory_usage")
            if cpu_usage > 80:
                bottlenecks.append("high_cpu_usage")
            
            # Combined bottlenecks
            if fps < 30 and cpu_usage > 80:
                bottlenecks.append("cpu_render_bottleneck")
            if memory_usage > 1500 and fps < 30:
                bottlenecks.append("memory_render_bottleneck")
            
            return bottlenecks
            
        except Exception as e:
            logger.error(f"Error identifying performance bottlenecks: {e}")
            return ["analysis_error"]
    
    def _generate_fallback_metrics(self) -> Dict[str, Any]:
        """Generate fallback metrics in case of errors."""
        return {
            "timestamp": time.time(),
            "base_metrics": {
                "osh_metrics": {"coherence": 0.0, "entropy": 0.0, "strain": 0.0, "rsp": 0.0},
                "quantum_metrics": {},
                "observer_metrics": {},
                "field_metrics": {},
                "memory_metrics": {},
                "recursive_metrics": {},
                "simulation_metrics": {},
                "performance_metrics": {},
                "emergent_phenomena": {}
            },
            "osh_metrics": OSHMetrics().to_dict(),
            "error": "Failed to process metrics - using fallback values"
        }


class AdvancedHealthAnalyzer:
    """Advanced system health analysis with predictive capabilities."""
    
    def __init__(self, history_size: int = 100):
        self.history_size = history_size
        self.health_history = deque(maxlen=history_size)
        self._lock = threading.RLock()
        
    def analyze_system_health(self, current_metrics: Any) -> SystemHealthProfile:
        """Perform comprehensive system health analysis."""
        with self._lock:
            try:
                # Extract health components
                component_health = self._calculate_component_health(current_metrics)
                
                # Calculate overall health
                overall_health = self._calculate_overall_health(component_health)
                
                # Determine health status
                health_status = self._classify_health_status(overall_health)
                
                # Analyze performance metrics
                performance_metrics = self._analyze_performance_health(current_metrics)
                
                # Calculate resource utilization
                resource_utilization = self._calculate_resource_utilization(current_metrics)
                
                # Assess stability indicators
                stability_indicators = self._calculate_stability_indicators(current_metrics)
                
                # Generate alerts
                alerts = self._generate_health_alerts(current_metrics, component_health)
                
                # Generate recommendations
                recommendations = self._generate_health_recommendations(current_metrics, component_health)
                
                # Identify critical issues
                critical_issues = self._identify_critical_issues(current_metrics, component_health)
                
                # Analyze health trend
                health_trend = self._analyze_health_trend()
                
                # Generate predictive alerts
                predictive_alerts = self._generate_predictive_alerts(current_metrics)
                
                health_profile = SystemHealthProfile(
                    overall_health=overall_health,
                    health_status=health_status,
                    component_health=component_health,
                    performance_metrics=performance_metrics,
                    resource_utilization=resource_utilization,
                    stability_indicators=stability_indicators,
                    alerts=alerts,
                    recommendations=recommendations,
                    critical_issues=critical_issues,
                    timestamp=time.time(),
                    health_trend=health_trend,
                    predictive_alerts=predictive_alerts
                )
                
                # Store in history
                self.health_history.append(health_profile)
                
                return health_profile
                
            except Exception as e:
                logger.error(f"Error analyzing system health: {e}")
                return self._generate_fallback_health_profile()
    
    def _calculate_component_health(self, current_metrics: Any) -> Dict[str, float]:
        """Calculate health score for each system component."""
        try:
            # OSH Core Health
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            
            osh_health = (coherence + (1.0 - min(entropy, 1.0)) + (1.0 - strain)) / 3.0
            
            # Quantum System Health
            quantum_fidelity = self._safe_get_metric(current_metrics, 'quantum_fidelity', 0.0)
            entanglement_strength = self._safe_get_metric(current_metrics, 'entanglement_strength', 0.0)
            
            quantum_health = (quantum_fidelity + entanglement_strength) / 2.0
            
            # Observer System Health
            observer_consensus = self._safe_get_metric(current_metrics, 'observer_consensus', 0.0)
            observer_count = self._safe_get_metric(current_metrics, 'observer_count', 0)
            active_observers = self._safe_get_metric(current_metrics, 'active_observers', 0)
            
            observer_activity_ratio = active_observers / max(observer_count, 1)
            observer_health = (observer_consensus + observer_activity_ratio) / 2.0
            
            # Memory System Health
            memory_strain_avg = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0)
            critical_regions = self._safe_get_metric(current_metrics, 'critical_strain_regions', 0)
            memory_regions = self._safe_get_metric(current_metrics, 'memory_regions', 1)
            
            critical_ratio = critical_regions / max(memory_regions, 1)
            memory_health = (1.0 - memory_strain_avg) * (1.0 - critical_ratio)
            
            # Performance Health
            render_fps = self._safe_get_metric(current_metrics, 'render_fps', 0.0)
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            cpu_usage_percent = self._safe_get_metric(current_metrics, 'cpu_usage_percent', 0.0)
            
            fps_health = min(render_fps / 60.0, 1.0)  # Target 60 FPS
            memory_usage_health = max(1.0 - memory_usage_mb / 2000.0, 0.0)  # Target under 2GB
            cpu_health = max(1.0 - cpu_usage_percent / 100.0, 0.0)
            
            performance_health = (fps_health + memory_usage_health + cpu_health) / 3.0
            
            # Field System Health
            total_field_energy = self._safe_get_metric(current_metrics, 'total_field_energy', 0.0)
            field_count = self._safe_get_metric(current_metrics, 'field_count', 0)
            
            field_health = min(total_field_energy / max(field_count * 10.0, 1.0), 1.0) if field_count > 0 else 1.0
            
            # Recursive System Health
            recursion_depth = self._safe_get_metric(current_metrics, 'recursion_depth', 0)
            boundary_crossings = self._safe_get_metric(current_metrics, 'boundary_crossings', 0)
            
            recursive_health = max(1.0 - recursion_depth / 10.0, 0.0) * max(1.0 - boundary_crossings / 20.0, 0.0)
            
            return {
                "osh_core": float(osh_health),
                "quantum_system": float(quantum_health),
                "observer_system": float(observer_health),
                "memory_system": float(memory_health),
                "performance": float(performance_health),
                "field_system": float(field_health),
                "recursive_system": float(recursive_health)
            }
            
        except Exception as e:
            logger.error(f"Error calculating component health: {e}")
            return {
                "osh_core": 0.0,
                "quantum_system": 0.0,
                "observer_system": 0.0,
                "memory_system": 0.0,
                "performance": 0.0,
                "field_system": 0.0,
                "recursive_system": 0.0
            }
    
    def _safe_get_metric(self, metrics_obj: Any, attr_name: str, default: Any) -> Any:
        """Safely extract metric with fallback."""
        try:
            if hasattr(metrics_obj, attr_name):
                value = getattr(metrics_obj, attr_name)
                return value if value is not None else default
            return default
        except (AttributeError, TypeError):
            return default
    
    def _calculate_overall_health(self, component_health: Dict[str, float]) -> float:
        """Calculate overall system health score."""
        try:
            # Weighted average of component health scores
            weights = {
                "osh_core": 0.25,
                "quantum_system": 0.20,
                "observer_system": 0.15,
                "memory_system": 0.15,
                "performance": 0.15,
                "field_system": 0.05,
                "recursive_system": 0.05
            }
            
            overall_health = sum(component_health.get(component, 0.0) * weight 
                               for component, weight in weights.items())
            
            return float(max(0.0, min(1.0, overall_health)))
            
        except Exception as e:
            logger.error(f"Error calculating overall health: {e}")
            return 0.0
    
    def _classify_health_status(self, overall_health: float) -> str:
        """Classify overall health status."""
        if overall_health >= 0.9:
            return "excellent"
        elif overall_health >= 0.7:
            return "good"
        elif overall_health >= 0.5:
            return "fair"
        elif overall_health >= 0.3:
            return "poor"
        elif overall_health >= 0.1:
            return "critical"
        else:
            return "system_failure"
    
    def _analyze_performance_health(self, current_metrics: Any) -> Dict[str, float]:
        """Analyze performance-specific health metrics."""
        try:
            render_fps = self._safe_get_metric(current_metrics, 'render_fps', 0.0)
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            cpu_usage_percent = self._safe_get_metric(current_metrics, 'cpu_usage_percent', 0.0)
            
            # Calculate performance efficiency metrics
            throughput_efficiency = min(render_fps / 60.0, 1.0)
            memory_efficiency = max(1.0 - memory_usage_mb / 2000.0, 0.0)
            cpu_efficiency = max(1.0 - cpu_usage_percent / 100.0, 0.0)
            
            # Calculate response time approximation
            response_time_health = 1.0 / (1.0 + max(1.0 / max(render_fps, 1.0) - 1.0/60.0, 0.0))
            
            return {
                "throughput_efficiency": float(throughput_efficiency),
                "memory_efficiency": float(memory_efficiency),
                "cpu_efficiency": float(cpu_efficiency),
                "response_time_health": float(response_time_health),
                "overall_performance": float((throughput_efficiency + memory_efficiency + cpu_efficiency) / 3.0)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing performance health: {e}")
            return {
                "throughput_efficiency": 0.0,
                "memory_efficiency": 0.0,
                "cpu_efficiency": 0.0,
                "response_time_health": 0.0,
                "overall_performance": 0.0
            }
    
    def _calculate_resource_utilization(self, current_metrics: Any) -> Dict[str, float]:
        """Calculate resource utilization metrics."""
        try:
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            cpu_usage_percent = self._safe_get_metric(current_metrics, 'cpu_usage_percent', 0.0)
            
            # Quantum resource utilization
            quantum_states = self._safe_get_metric(current_metrics, 'quantum_states_count', 0)
            total_qubits = self._safe_get_metric(current_metrics, 'total_qubits', 0)
            
            # Observer resource utilization
            observer_count = self._safe_get_metric(current_metrics, 'observer_count', 0)
            active_observers = self._safe_get_metric(current_metrics, 'active_observers', 0)
            
            # Field resource utilization
            field_count = self._safe_get_metric(current_metrics, 'field_count', 0)
            total_field_energy = self._safe_get_metric(current_metrics, 'total_field_energy', 0.0)
            
            return {
                "memory_utilization": float(memory_usage_mb / 2000.0),  # Fraction of 2GB target
                "cpu_utilization": float(cpu_usage_percent / 100.0),
                "quantum_state_utilization": float(min(quantum_states / 10.0, 1.0)),  # Target 10 states
                "qubit_utilization": float(min(total_qubits / 50.0, 1.0)),  # Target 50 qubits
                "observer_utilization": float(min(observer_count / 5.0, 1.0)),  # Target 5 observers
                "observer_activity_utilization": float(active_observers / max(observer_count, 1)),
                "field_utilization": float(min(field_count / 3.0, 1.0)),  # Target 3 fields
                "field_energy_utilization": float(min(total_field_energy / 100.0, 1.0))  # Target 100 units
            }
            
        except Exception as e:
            logger.error(f"Error calculating resource utilization: {e}")
            return {
                "memory_utilization": 0.0,
                "cpu_utilization": 0.0,
                "quantum_state_utilization": 0.0,
                "qubit_utilization": 0.0,
                "observer_utilization": 0.0,
                "observer_activity_utilization": 0.0,
                "field_utilization": 0.0,
                "field_energy_utilization": 0.0
            }
    
    def _calculate_stability_indicators(self, current_metrics: Any) -> Dict[str, float]:
        """Calculate stability indicators."""
        try:
            if len(self.health_history) < 3:
                return {
                    "coherence_stability": 1.0,
                    "entropy_stability": 1.0,
                    "performance_stability": 1.0,
                    "overall_stability": 1.0
                }
            
            # Extract recent health data
            recent_health = list(self.health_history)[-10:]
            
            # Calculate stability metrics based on variance
            coherence_values = [self._safe_get_metric(h, 'component_health', {}).get('osh_core', 0.0) for h in recent_health]
            performance_values = [self._safe_get_metric(h, 'component_health', {}).get('performance', 0.0) for h in recent_health]
            
            coherence_stability = 1.0 / (1.0 + np.var(coherence_values)) if coherence_values else 1.0
            performance_stability = 1.0 / (1.0 + np.var(performance_values)) if performance_values else 1.0
            
            # Entropy stability (inverse of entropy growth rate)
            entropy_stability = 1.0  # Placeholder - would need entropy history
            
            overall_stability = (coherence_stability + entropy_stability + performance_stability) / 3.0
            
            return {
                "coherence_stability": float(coherence_stability),
                "entropy_stability": float(entropy_stability),
                "performance_stability": float(performance_stability),
                "overall_stability": float(overall_stability)
            }
            
        except Exception as e:
            logger.error(f"Error calculating stability indicators: {e}")
            return {
                "coherence_stability": 1.0,
                "entropy_stability": 1.0,
                "performance_stability": 1.0,
                "overall_stability": 1.0
            }
    
    def _generate_health_alerts(self, current_metrics: Any, component_health: Dict[str, float]) -> List[str]:
        """Generate health alerts based on current state."""
        try:
            alerts = []
            
            # OSH Core alerts
            if component_health.get('osh_core', 0.0) < 0.3:
                alerts.append("Critical OSH core degradation detected - system coherence at risk")
            
            # Quantum system alerts
            if component_health.get('quantum_system', 0.0) < 0.4:
                alerts.append("Quantum system performance degraded - check fidelity and entanglement")
            
            # Observer system alerts
            if component_health.get('observer_system', 0.0) < 0.5:
                alerts.append("Observer consensus low - potential measurement instability")
            
            # Memory system alerts
            if component_health.get('memory_system', 0.0) < 0.3:
                alerts.append("Critical memory strain detected - defragmentation required")
            
            # Performance alerts
            render_fps = self._safe_get_metric(current_metrics, 'render_fps', 0.0)
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            cpu_usage_percent = self._safe_get_metric(current_metrics, 'cpu_usage_percent', 0.0)
            
            if render_fps < 10:
                alerts.append("Severe rendering performance degradation - optimization needed")
            if memory_usage_mb > 1800:
                alerts.append("High memory usage approaching limits - garbage collection advised")
            if cpu_usage_percent > 90:
                alerts.append("CPU usage critical - system may become unresponsive")
            
            # Specific metric alerts
            strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            entropy = self._safe_get_metric(current_metrics, 'entropy', 0.0)
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            
            if strain > 0.8:
                alerts.append("System strain exceeds safe threshold - stability at risk")
            if entropy > 0.9:
                alerts.append("High entropy levels - information degradation occurring")
            if coherence < 0.2:
                alerts.append("Coherence critically low - system integrity compromised")
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating health alerts: {e}")
            return ["Error generating health alerts - system monitoring compromised"]
    
    def _generate_health_recommendations(self, current_metrics: Any, component_health: Dict[str, float]) -> List[str]:
        """Generate health improvement recommendations."""
        try:
            recommendations = []
            
            # OSH Core recommendations
            if component_health.get('osh_core', 0.0) < 0.6:
                recommendations.append("Increase coherence through state alignment operations")
                recommendations.append("Reduce entropy through information organization")
                recommendations.append("Address memory strain through defragmentation")
            
            # Quantum system recommendations
            if component_health.get('quantum_system', 0.0) < 0.7:
                recommendations.append("Optimize quantum circuit depth and gate sequences")
                recommendations.append("Enhance entanglement protocols for better fidelity")
            
            # Observer system recommendations
            observer_consensus = self._safe_get_metric(current_metrics, 'observer_consensus', 0.0)
            if observer_consensus < 0.5:
                recommendations.append("Improve observer consensus through phase alignment")
                recommendations.append("Increase observer activity for better measurement stability")
            
            # Memory system recommendations
            memory_strain_avg = self._safe_get_metric(current_metrics, 'memory_strain_avg', 0.0)
            critical_regions = self._safe_get_metric(current_metrics, 'critical_strain_regions', 0)
            
            if memory_strain_avg > 0.6:
                recommendations.append("Schedule regular memory defragmentation cycles")
            if critical_regions > 3:
                recommendations.append("Emergency defragmentation required for critical regions")
                recommendations.append("Consider memory pool expansion or optimization")
            
            # Performance recommendations
            render_fps = self._safe_get_metric(current_metrics, 'render_fps', 0.0)
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            
            if render_fps < 30:
                recommendations.append("Optimize rendering pipeline for better performance")
                recommendations.append("Consider reducing visualization complexity")
            if memory_usage_mb > 1500:
                recommendations.append("Implement more aggressive garbage collection")
                recommendations.append("Optimize data structures for memory efficiency")
            
            # Field system recommendations
            field_count = self._safe_get_metric(current_metrics, 'field_count', 0)
            if field_count > 10:
                recommendations.append("Consider field consolidation for better performance")
                recommendations.append("Optimize field coupling parameters")
            
            # Recursive system recommendations
            recursion_depth = self._safe_get_metric(current_metrics, 'recursion_depth', 0)
            if recursion_depth > 5:
                recommendations.append("Monitor recursive depth to prevent stack overflow")
                recommendations.append("Implement recursive boundary optimization")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating health recommendations: {e}")
            return ["Unable to generate recommendations - system analysis required"]
    
    def _identify_critical_issues(self, current_metrics: Any, component_health: Dict[str, float]) -> List[str]:
        """Identify critical issues requiring immediate attention."""
        try:
            critical_issues = []
            
            # Critical thresholds
            if component_health.get('osh_core', 0.0) < 0.2:
                critical_issues.append("OSH core system failure imminent")
            
            if component_health.get('performance', 0.0) < 0.1:
                critical_issues.append("System performance critically degraded")
            
            if component_health.get('memory_system', 0.0) < 0.1:
                critical_issues.append("Memory system approaching failure")
            
            # Specific critical conditions
            strain = self._safe_get_metric(current_metrics, 'strain', 0.0)
            coherence = self._safe_get_metric(current_metrics, 'coherence', 0.0)
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            
            if strain > 0.95:
                critical_issues.append("System strain at maximum - immediate intervention required")
            
            if coherence < 0.1:
                critical_issues.append("Coherence collapse detected - system integrity lost")
            
            if memory_usage_mb > 1900:
                critical_issues.append("Memory exhaustion imminent - system crash risk")
            
            # Combined critical conditions
            if strain > 0.8 and coherence < 0.3:
                critical_issues.append("Combined strain-coherence crisis - system stability critical")
            
            return critical_issues
            
        except Exception as e:
            logger.error(f"Error identifying critical issues: {e}")
            return ["Critical issue analysis failed - manual inspection required"]
    
    def _analyze_health_trend(self) -> str:
        """Analyze health trend over time."""
        try:
            if len(self.health_history) < 3:
                return "insufficient_data"
            
            recent_health_scores = [h.overall_health for h in list(self.health_history)[-5:]]
            
            if len(recent_health_scores) < 2:
                return "stable"
            
            # Calculate trend using linear regression
            time_points = np.arange(len(recent_health_scores))
            slope, _, r_value, _, _ = stats.linregress(time_points, recent_health_scores)
            
            # Classify trend based on slope and correlation
            if abs(r_value) < 0.5:  # Weak correlation
                return "stable"
            elif slope > 0.05:
                return "improving"
            elif slope < -0.05:
                return "degrading"
            else:
                return "stable"
                
        except Exception as e:
            logger.error(f"Error analyzing health trend: {e}")
            return "unknown"
    
    def _generate_predictive_alerts(self, current_metrics: Any) -> List[str]:
        """Generate predictive alerts based on trend analysis."""
        try:
            if len(self.health_history) < 5:
                return []
            
            predictive_alerts = []
            
            # Analyze trends in key metrics
            recent_health = list(self.health_history)[-10:]
            
            # Extract time series for prediction
            osh_health_series = [h.component_health.get('osh_core', 0.0) for h in recent_health]
            performance_series = [h.component_health.get('performance', 0.0) for h in recent_health]
            memory_series = [h.component_health.get('memory_system', 0.0) for h in recent_health]
            
            # Predict next values using simple linear extrapolation
            for name, series in [
                ('OSH core', osh_health_series),
                ('performance', performance_series),
                ('memory', memory_series)
            ]:
                if len(series) >= 3:
                    time_points = np.arange(len(series))
                    slope, intercept, r_value, _, _ = stats.linregress(time_points, series)
                    
                    # Predict next 3 points
                    if abs(r_value) > 0.6:  # Strong correlation
                        predicted_next = slope * len(series) + intercept
                        predicted_future = slope * (len(series) + 2) + intercept
                        
                        if predicted_next < 0.3 and series[-1] > 0.4:
                            predictive_alerts.append(f"{name} system degradation predicted - proactive measures recommended")
                        
                        if predicted_future < 0.2 and predicted_next > 0.2:
                            predictive_alerts.append(f"{name} system failure risk in near future - immediate attention needed")
            
            # Memory usage prediction
            memory_usage_mb = self._safe_get_metric(current_metrics, 'memory_usage_mb', 0.0)
            if memory_usage_mb > 1200:
                # Simple growth rate estimation
                if len(self.health_history) >= 3:
                    prev_memory = getattr(list(self.health_history)[-3], 'memory_usage_mb', memory_usage_mb)
                    growth_rate = (memory_usage_mb - prev_memory) / 3.0
                    
                    if growth_rate > 50:  # Growing by >50MB per step
                        predicted_usage = memory_usage_mb + growth_rate * 5
                        if predicted_usage > 1800:
                            predictive_alerts.append("Memory usage trending toward critical levels - optimization needed")
            
            return predictive_alerts
            
        except Exception as e:
            logger.error(f"Error generating predictive alerts: {e}")
            return []
    
    def _generate_fallback_health_profile(self) -> SystemHealthProfile:
        """Generate fallback health profile in case of errors."""
        return SystemHealthProfile(
            overall_health=0.0,
            health_status="unknown",
            component_health={
                "osh_core": 0.0,
                "quantum_system": 0.0,
                "observer_system": 0.0,
                "memory_system": 0.0,
                "performance": 0.0,
                "field_system": 0.0,
                "recursive_system": 0.0
            },
            performance_metrics={
                "throughput_efficiency": 0.0,
                "memory_efficiency": 0.0,
                "cpu_efficiency": 0.0,
                "response_time_health": 0.0,
                "overall_performance": 0.0
            },
            resource_utilization={},
            stability_indicators={
                "coherence_stability": 0.0,
                "entropy_stability": 0.0,
                "performance_stability": 0.0,
                "overall_stability": 0.0
            },
            alerts=["System health analysis failed"],
            recommendations=["Manual system inspection required"],
            critical_issues=["Health monitoring system compromised"],
            timestamp=time.time(),
            health_trend="unknown",
            predictive_alerts=[]
        )


class ProfessionalFigureProcessor:
    """Professional figure processing and optimization for scientific visualization."""
    
    def __init__(self, high_dpi: bool = True, optimization_level: int = 2):
        self.high_dpi = high_dpi
        self.optimization_level = optimization_level
        self.color_profiles = self._initialize_color_profiles()
        self._lock = threading.RLock()
        
    def _initialize_color_profiles(self) -> Dict[str, Dict[str, str]]:
        """Initialize professional color profiles."""
        return {
            'dark_quantum': {
                'background': '#0a0a0a',
                'foreground': '#ffffff',
                'accent': '#00ffff',
                'secondary': '#ff6b35',
                'grid': '#333333'
            },
            'light_scientific': {
                'background': '#ffffff',
                'foreground': '#000000',
                'accent': '#1f77b4',
                'secondary': '#ff7f0e',
                'grid': '#cccccc'
            },
            'publication': {
                'background': '#ffffff',
                'foreground': '#000000',
                'accent': '#2c3e50',
                'secondary': '#e74c3c',
                'grid': '#ecf0f1'
            }
        }
    
    def process_figure_to_base64(self, fig: plt.Figure, 
                                color_profile: str = 'dark_quantum',
                                format: str = 'png',
                                optimize: bool = True) -> str:
        """Convert matplotlib figure to optimized base64 string."""
        with self._lock:
            try:
                # Apply color profile
                colors = self.color_profiles.get(color_profile, self.color_profiles['dark_quantum'])
                
                # Configure figure for professional output
                self._configure_figure_for_export(fig, colors)
                
                # Create buffer
                buf = io.BytesIO()
                
                # Determine DPI and format settings
                dpi = 300 if self.high_dpi and self.optimization_level >= 2 else 150
                
                format_params = {
                    'format': format,
                    'dpi': dpi,
                    'bbox_inches': 'tight',
                    'facecolor': colors['background'],
                    'edgecolor': 'none',
                    'transparent': False,
                    'pad_inches': 0.1
                }
                
                if format == 'png':
                    format_params.update({
                        'optimize': optimize and self.optimization_level >= 1,
                        'pil_kwargs': {'optimize': True, 'quality': 95} if optimize else {}
                    })
                elif format == 'svg':
                    format_params.update({
                        'metadata': {
                            'Creator': 'Recursia Visualization System',
                            'Date': time.strftime('%Y-%m-%d'),
                            'Description': 'OSH Scientific Visualization'
                        }
                    })
                
                # Save figure to buffer
                fig.savefig(buf, **format_params)
                buf.seek(0)
                
                # Apply additional optimization if requested
                if optimize and self.optimization_level >= 3:
                    buf = self._optimize_image_data(buf, format)
                
                # Encode to base64
                img_data = base64.b64encode(buf.read()).decode('utf-8')
                return f"data:image/{format};base64,{img_data}"
                
            except Exception as e:
                logger.error(f"Error processing figure to base64: {e}")
                return self._generate_fallback_image(format)
            finally:
                # Clean up buffer
                if 'buf' in locals():
                    buf.close()
    
    def _configure_figure_for_export(self, fig: plt.Figure, colors: Dict[str, str]):
        """Configure figure for professional export."""
        try:
            # Set figure background
            fig.patch.set_facecolor(colors['background'])
            
            # Configure all axes
            for ax in fig.get_axes():
                # Set axis colors
                ax.set_facecolor(colors['background'])
                ax.tick_params(colors=colors['foreground'], which='both')
                ax.xaxis.label.set_color(colors['foreground'])
                ax.yaxis.label.set_color(colors['foreground'])
                ax.title.set_color(colors['foreground'])
                
                # Configure spines
                for spine in ax.spines.values():
                    spine.set_color(colors['foreground'])
                    spine.set_linewidth(0.8)
                
                # Configure grid
                if ax.grid:
                    ax.grid(True, alpha=0.3, color=colors['grid'], linewidth=0.5)
                
                # Configure legend if present
                legend = ax.get_legend()
                if legend:
                    legend.get_frame().set_facecolor(colors['background'])
                    legend.get_frame().set_edgecolor(colors['foreground'])
                    legend.get_frame().set_alpha(0.9)
                    for text in legend.get_texts():
                        text.set_color(colors['foreground'])
            
            # Set figure title color if present
            if fig._suptitle:
                fig._suptitle.set_color(colors['foreground'])
            
        except Exception as e:
            logger.error(f"Error configuring figure for export: {e}")
    
    def _optimize_image_data(self, buf: io.BytesIO, format: str) -> io.BytesIO:
        """Apply advanced image optimization."""
        try:
            if format == 'png':
                # For PNG, we could apply additional compression
                # This is a placeholder for more sophisticated optimization
                compressed_buf = io.BytesIO()
                
                # Read original data
                buf.seek(0)
                original_data = buf.read()
                
                # Apply gzip compression for additional size reduction
                # (Note: This is more of a demonstration - real PNG optimization 
                # would require PIL/Pillow optimizations)
                compressed_data = gzip.compress(original_data, compresslevel=9)
                
                # For this implementation, we'll return the original
                # as gzip compression on PNG is not standard
                compressed_buf.write(original_data)
                compressed_buf.seek(0)
                
                return compressed_buf
            
            return buf
            
        except Exception as e:
            logger.error(f"Error optimizing image data: {e}")
            return buf
    
    def _generate_fallback_image(self, format: str) -> str:
        """Generate fallback image data."""
        # 1x1 transparent pixel as fallback
        fallback_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
        return f"data:image/{format};base64,{fallback_data}"


# Utility functions for external access
def get_system_health_summary(current_metrics: Any) -> Dict[str, Any]:
    """Get comprehensive system health summary."""
    try:
        health_analyzer = AdvancedHealthAnalyzer()
        health_profile = health_analyzer.analyze_system_health()
        
        return {
            "overall_health": health_profile.overall_health,
            "health_status": health_profile.health_status,
            "component_health": health_profile.component_health,
            "alerts": health_profile.alerts,
            "recommendations": health_profile.recommendations,
            "critical_issues": health_profile.critical_issues,
            "health_trend": health_profile.health_trend,
            "predictive_alerts": health_profile.predictive_alerts,
            "timestamp": health_profile.timestamp,
            "stability_indicators": health_profile.stability_indicators
        }
        
    except Exception as e:
        logger.error(f"Error getting system health summary: {e}")
        return {
            "overall_health": 0.0,
            "health_status": "unknown",
            "component_health": {},
            "alerts": ["Health analysis failed"],
            "recommendations": ["Manual inspection required"],
            "critical_issues": ["Health monitoring unavailable"],
            "health_trend": "unknown",
            "predictive_alerts": [],
            "timestamp": time.time(),
            "stability_indicators": {}
        }


def get_comprehensive_metrics_summary(current_metrics: Any) -> Dict[str, Any]:
    """Get comprehensive summary of all current metrics."""
    try:
        processor = AdvancedMetricsProcessor()
        return processor.process_comprehensive_metrics(current_metrics)
        
    except Exception as e:
        logger.error(f"Error getting comprehensive metrics summary: {e}")
        return {
            "timestamp": time.time(),
            "base_metrics": {},
            "osh_metrics": OSHMetrics().to_dict(),
            "error": "Failed to process comprehensive metrics"
        }


def get_professional_figure_data(fig: plt.Figure, 
                                color_profile: str = 'dark_quantum',
                                high_dpi: bool = True,
                                format: str = 'png') -> str:
    """Convert matplotlib figure to professional base64 PNG with optimization."""
    try:
        processor = ProfessionalFigureProcessor(high_dpi=high_dpi, optimization_level=2)
        return processor.process_figure_to_base64(fig, color_profile, format, optimize=True)
        
    except Exception as e:
        logger.error(f"Error converting figure to professional image data: {e}")
        # Return fallback 1x1 transparent pixel
        return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="


def calculate_osh_metrics(current_metrics: Any) -> OSHMetrics:
    """Calculate comprehensive OSH metrics."""
    try:
        processor = AdvancedMetricsProcessor()
        return processor._calculate_osh_metrics(current_metrics)
        
    except Exception as e:
        logger.error(f"Error calculating OSH metrics: {e}")
        return OSHMetrics()


def analyze_system_trends(metrics_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze system trends from metrics history."""
    try:
        if len(metrics_history) < 3:
            return {"insufficient_data": True}
        
        # Extract time series
        timestamps = [m.get('timestamp', 0) for m in metrics_history]
        coherence_series = [m.get('osh_metrics', {}).get('coherence', 0) for m in metrics_history]
        entropy_series = [m.get('osh_metrics', {}).get('entropy', 0) for m in metrics_history]
        rsp_series = [m.get('osh_metrics', {}).get('rsp', 0) for m in metrics_history]
        
        # Analyze trends
        trends = {}
        
        for name, series in [
            ('coherence', coherence_series),
            ('entropy', entropy_series),
            ('rsp', rsp_series)
        ]:
            if len(series) >= 3 and len(timestamps) == len(series):
                time_deltas = np.array(timestamps) - timestamps[0]
                slope, intercept, r_value, p_value, std_err = stats.linregress(time_deltas, series)
                
                trends[name] = {
                    "slope": float(slope),
                    "direction": "increasing" if slope > 0.01 else "decreasing" if slope < -0.01 else "stable",
                    "strength": float(abs(r_value)),
                    "confidence": float(r_value**2),
                    "p_value": float(p_value),
                    "significance": "significant" if p_value < 0.05 else "not_significant"
                }
        
        # Overall trend assessment
        coherence_trend = trends.get('coherence', {}).get('direction', 'stable')
        entropy_trend = trends.get('entropy', {}).get('direction', 'stable')
        
        if coherence_trend == 'increasing' and entropy_trend == 'decreasing':
            overall_trend = 'improving'
        elif coherence_trend == 'decreasing' and entropy_trend == 'increasing':
            overall_trend = 'degrading'
        else:
            overall_trend = 'stable'
        
        return {
            "individual_trends": trends,
            "overall_trend": overall_trend,
            "analysis_timestamp": time.time(),
            "data_points_analyzed": len(metrics_history)
        }
        
    except Exception as e:
        logger.error(f"Error analyzing system trends: {e}")
        return {"error": str(e)}


def detect_anomalies(current_metrics: Any, historical_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect anomalies in current metrics compared to historical data."""
    try:
        if len(historical_metrics) < 10:
            return {"insufficient_data": True}
        
        # Extract current values
        current_coherence = getattr(current_metrics, 'coherence', 0.0)
        current_entropy = getattr(current_metrics, 'entropy', 0.0)
        current_strain = getattr(current_metrics, 'strain', 0.0)
        
        # Extract historical values
        historical_coherence = [m.get('osh_metrics', {}).get('coherence', 0) for m in historical_metrics]
        historical_entropy = [m.get('osh_metrics', {}).get('entropy', 0) for m in historical_metrics]
        historical_strain = [m.get('osh_metrics', {}).get('strain', 0) for m in historical_metrics]
        
        anomalies = {}
        
        # Z-score based anomaly detection
        for metric_name, historical, current in [
            ('coherence', historical_coherence, current_coherence),
            ('entropy', historical_entropy, current_entropy),
            ('strain', historical_strain, current_strain)
        ]:
            if len(historical) > 2:
                mean_val = np.mean(historical)
                std_val = np.std(historical)
                
                if std_val > 0:
                    z_score = abs(current - mean_val) / std_val
                    
                    if z_score > 2.0:  # Anomaly threshold
                        anomalies[metric_name] = {
                            "anomaly_detected": True,
                            "z_score": float(z_score),
                            "severity": "high" if z_score > 3.0 else "moderate",
                            "direction": "high" if current > mean_val else "low",
                            "current_value": float(current),
                            "historical_mean": float(mean_val),
                            "historical_std": float(std_val)
                        }
        
        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomaly_count": len(anomalies),
            "anomaly_details": anomalies,
            "detection_timestamp": time.time(),
            "analysis_method": "z_score",
            "threshold": 2.0
        }
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return {"error": str(e)}


def generate_scientific_summary(current_metrics: Any, 
                              metrics_summary: Dict[str, Any]) -> str:
    """Generate a scientific summary of the current system state."""
    try:
        # Extract key metrics
        osh_metrics = metrics_summary.get('osh_metrics', {})
        coherence = osh_metrics.get('coherence', 0.0)
        entropy = osh_metrics.get('entropy', 0.0)
        strain = osh_metrics.get('strain', 0.0)
        rsp = osh_metrics.get('rsp', 0.0)
        
        base_metrics = metrics_summary.get('base_metrics', {})
        quantum_metrics = base_metrics.get('quantum_metrics', {})
        observer_metrics = base_metrics.get('observer_metrics', {})
        
        # Generate summary
        summary_parts = []
        
        # System state assessment
        if coherence > 0.8:
            coherence_desc = "excellent"
        elif coherence > 0.6:
            coherence_desc = "good"
        elif coherence > 0.4:
            coherence_desc = "moderate"
        else:
            coherence_desc = "poor"
        
        summary_parts.append(f"System coherence is {coherence_desc} at {coherence:.3f}")
        
        # Entropy assessment
        if entropy < 0.3:
            entropy_desc = "low, indicating high organization"
        elif entropy < 0.6:
            entropy_desc = "moderate"
        else:
            entropy_desc = "high, indicating system disorder"
        
        summary_parts.append(f"entropy is {entropy_desc} ({entropy:.3f})")
        
        # RSP assessment
        if rsp > 10.0:
            rsp_desc = "exceptional"
        elif rsp > 5.0:
            rsp_desc = "high"
        elif rsp > 1.0:
            rsp_desc = "moderate"
        else:
            rsp_desc = "low"
        
        summary_parts.append(f"Recursive Simulation Potential is {rsp_desc} ({rsp:.2f})")
        
        # Quantum system status
        quantum_states = quantum_metrics.get('states_count', 0)
        entanglement = quantum_metrics.get('entanglement_strength', 0.0)
        
        if quantum_states > 0:
            summary_parts.append(f"The quantum subsystem contains {quantum_states} states with entanglement strength {entanglement:.3f}")
        
        # Observer system status
        observer_count = observer_metrics.get('observer_count', 0)
        consensus = observer_metrics.get('consensus', 0.0)
        
        if observer_count > 0:
            summary_parts.append(f"{observer_count} observers are active with consensus level {consensus:.3f}")
        
        # Memory system status
        memory_metrics = base_metrics.get('memory_metrics', {})
        memory_strain = memory_metrics.get('strain_avg', 0.0)
        
        if memory_strain > 0.7:
            summary_parts.append("Memory strain is elevated, indicating potential fragmentation")
        elif memory_strain < 0.3:
            summary_parts.append("Memory system is operating efficiently")
        
        # Performance indicators
        performance_metrics = base_metrics.get('performance_metrics', {})
        fps = performance_metrics.get('render_fps', 0.0)
        
        if fps > 30:
            summary_parts.append("rendering performance is optimal")
        elif fps > 15:
            summary_parts.append("rendering performance is adequate")
        else:
            summary_parts.append("rendering performance requires optimization")
        
        # Combine into coherent summary
        summary = ". ".join(summary_parts).replace(". .", ".") + "."
        summary = summary[0].upper() + summary[1:]  # Capitalize first letter
        
        return summary
        
    except Exception as e:
        logger.error(f"Error generating scientific summary: {e}")
        return "Unable to generate system summary due to analysis error."


# Global instances for easy access
_global_metrics_processor = None
_global_health_analyzer = None
_global_figure_processor = None


def get_global_metrics_processor() -> AdvancedMetricsProcessor:
    """Get global metrics processor instance."""
    global _global_metrics_processor
    if _global_metrics_processor is None:
        _global_metrics_processor = AdvancedMetricsProcessor()
    return _global_metrics_processor


def get_global_health_analyzer() -> AdvancedHealthAnalyzer:
    """Get global health analyzer instance."""
    global _global_health_analyzer
    if _global_health_analyzer is None:
        _global_health_analyzer = AdvancedHealthAnalyzer()
    return _global_health_analyzer


def get_global_figure_processor() -> ProfessionalFigureProcessor:
    """Get global figure processor instance."""
    global _global_figure_processor
    if _global_figure_processor is None:
        _global_figure_processor = ProfessionalFigureProcessor()
    return _global_figure_processor


# Export key classes and functions
__all__ = [
    'OSHMetrics',
    'SystemHealthProfile',
    'AdvancedMetricsProcessor',
    'AdvancedHealthAnalyzer',
    'ProfessionalFigureProcessor',
    'get_system_health_summary',
    'get_comprehensive_metrics_summary',
    'get_professional_figure_data',
    'calculate_osh_metrics',
    'analyze_system_trends',
    'detect_anomalies',
    'generate_scientific_summary',
    'get_global_metrics_processor',
    'get_global_health_analyzer',
    'get_global_figure_processor'
]