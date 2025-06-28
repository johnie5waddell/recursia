import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import os
from collections import defaultdict

class SimulationReportBuilder:
    """
    Comprehensive report generator for physics simulation data.
    
    Produces detailed summaries, trends, and statistical analysis from 
    simulation runs with specialized OSH-aligned metrics. Tracks coherence,
    entropy, memory strain, and emergent phenomena to evaluate the stability
    and dynamics of the simulation.
    
    Attributes:
        metrics (Dict[str, Dict[str, float]]): Custom metrics tracked over time
        time_series (Dict[str, List[Tuple[float, float]]]): Time-indexed metric values
        event_counts (Dict[str, int]): Counter for different event types
        stability_history (List[Tuple[float, float]]): Time-stability pairs
        analysis_cache (Dict[str, Any]): Cached analysis results
        osh_metrics (Dict[str, float]): Simulation-wide OSH metrics
        logger (logging.Logger): Logger instance
    """
    
    # Constants for OSH alignment
    OSH_COHERENCE_THRESHOLD = 0.7
    OSH_ENTROPY_THRESHOLD = 0.3
    OSH_STABILITY_THRESHOLD = 0.6
    
    # Event categories for grouping
    EVENT_CATEGORIES = {
        "quantum": ["collapse_events", "entanglement_events", "teleportation_events"],
        "memory": ["strain_threshold_events", "defragmentation_events"],
        "observer": ["observation_events", "consensus_events"],
        "recursive": ["boundary_events", "resonance_events"]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize report builder with optional configuration.
        
        Args:
            config: Optional configuration dictionary containing:
                max_history_size: Maximum data points to retain in time series
                osh_coherence_threshold: Custom coherence threshold
                osh_entropy_threshold: Custom entropy threshold
                log_level: Logging level for the report builder
        """
        self.metrics = defaultdict(dict)
        self.time_series = defaultdict(list)
        self.event_counts = defaultdict(int)
        self.stability_history = []
        self.analysis_cache = {}
        self.osh_metrics = {
            "rsp": 0.0,  # Recursive Simulation Potential
            "coherence_stability": 0.0,
            "entropy_flux": 0.0,
            "recursion_depth": 0.0,
            "system_strain": 0.0
        }
        
        # Configure from input or defaults
        self.config = {
            "max_history_size": 1000,
            "osh_coherence_threshold": self.OSH_COHERENCE_THRESHOLD,
            "osh_entropy_threshold": self.OSH_ENTROPY_THRESHOLD,
            "log_level": logging.INFO
        }
        
        if config:
            self.config.update(config)
        
        # Initialize logger
        self.logger = logging.getLogger("SimulationReportBuilder")
        self.logger.setLevel(self.config["log_level"])
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.start_time = datetime.now()
        # self.logger.info("SimulationReportBuilder initialized")
    
    def set_physics_context(self, physics_engine, event_system):
        """Set physics engine and event system context for enhanced reporting."""
        self.physics_engine = physics_engine
        self.event_system = event_system
        self.logger.info("Physics context set for enhanced reporting")

    def collect_physics_data(self, time: float) -> bool:
        """Collect data from physics engine if available."""
        if not hasattr(self, 'physics_engine') or not self.physics_engine:
            return False
        
        try:
            # Get current metrics from physics engine
            current_metrics = self.physics_engine.current_metrics
            system_state = self.physics_engine.get_system_state()
            
            # Convert to the format expected by collect_step_data
            step_results = {
                "changes": {
                    "coherence": {
                        "coherence_level": current_metrics.coherence,
                        "coherence_changes": {
                            name: self.physics_engine.coherence_manager.get_state_coherence(name) or 0.0
                            for name in system_state.get('quantum_states', {}).keys()
                        }
                    },
                    "memory": {
                        "entropy_level": current_metrics.entropy,
                        "region_updates": {
                            f"region_{i}": {
                                "strain": current_metrics.strain,
                                "coherence": current_metrics.coherence,
                                "entropy": current_metrics.entropy
                            }
                            for i in range(1)  # At least one region
                        }
                    },
                    "observers": {
                        "observation_events": [
                            {"observer": name, "type": "observation"}
                            for name in system_state.get('observers', {}).keys()
                        ]
                    },
                    "recursive": {
                        "max_depth": current_metrics.recursive_depth,
                        "boundary_events": []
                    }
                }
            }
            
            return self.collect_step_data(time, step_results)
            
        except Exception as e:
            self.logger.error(f"Error collecting physics data: {e}")
            return False
        
    def collect_step_data(self, time: float, step_results: Dict[str, Any]) -> bool:
        """Collect data from a simulation step and update internal metrics.
        
        Extracts coherence changes, memory strain, observer interactions, and
        emergent phenomena from simulation step results. Calculates stability
        metrics and maintains time series for all tracked quantities.
        
        Args:
            time: Simulation time point
            step_results: Results dictionary from simulation step
            
        Returns:
            bool: True if data was successfully processed, False otherwise
            
        Raises:
            ValueError: If time decreases (non-monotonic)
            TypeError: If step_results is not a dictionary
        """
        try:
            # Validate inputs
            if not isinstance(step_results, dict):
                raise TypeError("step_results must be a dictionary")
                
            # Check time monotonicity
            if self.stability_history and time < self.stability_history[-1][0]:
                raise ValueError(f"Time must be monotonically increasing. Got {time} after {self.stability_history[-1][0]}")
            
            # Extract changes dictionary, defaulting to empty if not present
            changes = step_results.get("changes", {})
            
            # Process coherence changes
            self._process_coherence_data(time, changes)
            
            # Process memory/strain data
            self._process_memory_data(time, changes)
            
            # Process observer events
            self._process_observer_data(time, changes)
            
            # Extract other tracked events
            self._process_event_data(time, changes)
            
            # Calculate stability score
            self._calculate_stability_score(time, changes)
            
            # Update OSH metrics
            self._update_osh_metrics(changes)
            
            # Prune historical data if needed
            self._prune_time_series_if_needed()
            
            # Invalidate analysis cache since data changed
            self.analysis_cache.clear()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error collecting step data: {str(e)}", exc_info=True)
            return False
    
    def _process_coherence_data(self, time: float, changes: Dict[str, Any]) -> None:
        """Process coherence-related data from simulation step.
        
        Args:
            time: Simulation time
            changes: Changes dictionary from step results
        """
        if "coherence" in changes:
            coherence_data = changes["coherence"]
            
            # Process coherence changes
            coherence_changes = coherence_data.get("coherence_changes", {})
            if coherence_changes:
                # Store individual coherence values for each entity
                for entity, value in coherence_changes.items():
                    entity_key = f"coherence_{entity}"
                    if entity_key not in self.time_series:
                        self.time_series[entity_key] = []
                    self.time_series[entity_key].append((time, value))
                
                # Calculate average coherence change
                coherence_values = list(coherence_changes.values())
                if coherence_values:
                    avg_change = sum(coherence_values) / len(coherence_values)
                    self.time_series["coherence_change"].append((time, avg_change))
                    
                    # Store absolute coherence level if provided
                    if "coherence_level" in coherence_data:
                        self.time_series["coherence_level"].append(
                            (time, coherence_data["coherence_level"])
                        )
            
            # Process resonance events
            resonance_events = coherence_data.get("resonance_events", [])
            if resonance_events:
                self.event_counts["resonance_events"] += len(resonance_events)
                
                # Store resonance strength if available
                if isinstance(resonance_events, list) and resonance_events and isinstance(resonance_events[0], dict):
                    # Calculate average resonance strength
                    strength_values = [event.get("strength", 0.0) for event in resonance_events 
                                      if isinstance(event, dict)]
                    if strength_values:
                        avg_strength = sum(strength_values) / len(strength_values)
                        self.time_series["resonance_strength"].append((time, avg_strength))
    
    def _process_memory_data(self, time: float, changes: Dict[str, Any]) -> None:
        """Process memory and strain-related data from simulation step.
        
        Args:
            time: Simulation time
            changes: Changes dictionary from step results
        """
        if "memory" in changes:
            memory_data = changes["memory"]
            
            # Process region updates
            region_updates = memory_data.get("region_updates", {})
            if region_updates:
                # Extract strain values from updates
                strain_values = [update.get("strain", 0.0) 
                                for update in region_updates.values()
                                if isinstance(update, dict)]
                
                # Process overall strain statistics
                if strain_values:
                    avg_strain = sum(strain_values) / len(strain_values)
                    max_strain = max(strain_values)
                    
                    # Add to time series
                    self.time_series["avg_strain"].append((time, avg_strain))
                    self.time_series["max_strain"].append((time, max_strain))
                    
                    # Track high-strain regions
                    high_strain_count = sum(1 for s in strain_values if s > self.config["osh_entropy_threshold"])
                    self.time_series["high_strain_regions"].append((time, high_strain_count))
                
                # Store per-region metrics
                for region, update in region_updates.items():
                    if isinstance(update, dict):
                        # Track region-specific strain
                        strain_key = f"strain_{region}"
                        if "strain" in update:
                            if strain_key not in self.time_series:
                                self.time_series[strain_key] = []
                            self.time_series[strain_key].append((time, update["strain"]))
                        
                        # Track other region metrics
                        for metric in ["coherence", "entropy"]:
                            if metric in update:
                                metric_key = f"{metric}_{region}"
                                if metric_key not in self.time_series:
                                    self.time_series[metric_key] = []
                                self.time_series[metric_key].append((time, update[metric]))
            
            # Process strain thresholds
            strain_thresholds = memory_data.get("strain_thresholds", [])
            if strain_thresholds:
                self.event_counts["strain_threshold_events"] += len(strain_thresholds)
                
                # Track critical threshold regions
                critical_regions = [event.get("region") for event in strain_thresholds 
                                   if isinstance(event, dict)]
                if critical_regions:
                    self.metrics["critical_regions"] = critical_regions
            
            # Process defragmentation events
            defrag_events = memory_data.get("defragmentation_events", [])
            if defrag_events:
                self.event_counts["defragmentation_events"] += len(defrag_events)
    
    def _process_observer_data(self, time: float, changes: Dict[str, Any]) -> None:
        """Process observer-related data from simulation step.
        
        Args:
            time: Simulation time
            changes: Changes dictionary from step results
        """
        if "observers" in changes:
            observer_data = changes["observers"]
            
            # Process collapse events
            collapse_events = observer_data.get("collapse_events", [])
            if collapse_events:
                self.event_counts["collapse_events"] += len(collapse_events)
                
                # Track collapse strength if available
                if isinstance(collapse_events, list) and collapse_events and isinstance(collapse_events[0], dict):
                    # Extract probability values
                    prob_values = [event.get("probability", 0.0) for event in collapse_events 
                                  if isinstance(event, dict)]
                    if prob_values:
                        avg_prob = sum(prob_values) / len(prob_values)
                        self.time_series["collapse_probability"].append((time, avg_prob))
            
            # Process observation events
            observation_events = observer_data.get("observation_events", [])
            if observation_events:
                self.event_counts["observation_events"] += len(observation_events)
                
                # Extract observer counts
                if isinstance(observation_events, list):
                    # Get unique observer count
                    unique_observers = set()
                    for event in observation_events:
                        if isinstance(event, dict) and "observer" in event:
                            unique_observers.add(event["observer"])
                    
                    if unique_observers:
                        self.time_series["active_observers"].append((time, len(unique_observers)))
            
            # Process consensus events
            consensus_events = observer_data.get("consensus_events", [])
            if consensus_events:
                self.event_counts["consensus_events"] += len(consensus_events)
                
                # Track consensus strength if available
                if isinstance(consensus_events, list) and consensus_events and isinstance(consensus_events[0], dict):
                    strength_values = [event.get("strength", 0.0) for event in consensus_events 
                                      if isinstance(event, dict)]
                    if strength_values:
                        avg_strength = sum(strength_values) / len(strength_values)
                        self.time_series["consensus_strength"].append((time, avg_strength))
    
    def _process_event_data(self, time: float, changes: Dict[str, Any]) -> None:
        """Process general event data from simulation step.
        
        Args:
            time: Simulation time
            changes: Changes dictionary from step results
        """
        # Process recursive boundary events
        if "recursive" in changes:
            recursive_data = changes["recursive"]
            
            # Process boundary events
            boundary_events = recursive_data.get("boundary_events", [])
            if boundary_events:
                self.event_counts["boundary_events"] += len(boundary_events)
                
                # Track boundary crossing direction
                if isinstance(boundary_events, list) and boundary_events:
                    upward = sum(1 for event in boundary_events 
                                if isinstance(event, dict) and 
                                event.get("direction") == "upward")
                    downward = sum(1 for event in boundary_events 
                                  if isinstance(event, dict) and 
                                  event.get("direction") == "downward")
                    
                    self.time_series["upward_boundary_crossings"].append((time, upward))
                    self.time_series["downward_boundary_crossings"].append((time, downward))
        
        # Process field events
        if "field" in changes:
            field_data = changes["field"]
            
            # Process coherence waves
            wave_events = field_data.get("coherence_wave_events", [])
            if wave_events:
                self.event_counts["coherence_wave_events"] += len(wave_events)
                
                # Track wave strength
                if isinstance(wave_events, list) and wave_events and isinstance(wave_events[0], dict):
                    strength_values = [event.get("strength", 0.0) for event in wave_events 
                                      if isinstance(event, dict)]
                    if strength_values:
                        max_strength = max(strength_values)
                        self.time_series["coherence_wave_strength"].append((time, max_strength))
        
        # Track general events across all categories
        for section, events in changes.items():
            if isinstance(events, dict):
                for event_type, event_data in events.items():
                    if event_type.endswith("_events") and isinstance(event_data, (list, tuple)):
                        event_count = len(event_data)
                        if event_count > 0:
                            self.event_counts[event_type] += event_count
    
    def _calculate_stability_score(self, time: float, changes: Dict[str, Any]) -> None:
        """Calculate overall stability score from simulation data.
        
        Combines coherence trends, strain levels, and other factors to compute
        a normalized stability score between 0.0 (critical) and 1.0 (highly stable).
        
        Args:
            time: Simulation time
            changes: Changes dictionary from step results
        """
        stability_factors = []
        
        # Add coherence-based stability factor
        if "coherence" in changes:
            coherence_data = changes["coherence"]
            coherence_changes = coherence_data.get("coherence_changes", {})
            
            if coherence_changes:
                # Positive coherence change indicates increasing stability
                coherence_values = list(coherence_changes.values())
                avg_change = sum(coherence_values) / len(coherence_values)
                
                # Map avg_change to stability score in [0,1] range
                # Starting at 0.5 (neutral), adjust up to 0.5 points in either direction
                coherence_stability = 0.5 + min(0.5, max(-0.5, avg_change * 5))
                stability_factors.append(coherence_stability)
                
                # Consider absolute coherence level if available
                if "coherence_level" in coherence_data:
                    level = coherence_data["coherence_level"]
                    # High coherence (>threshold) indicates more stability
                    if level > self.config["osh_coherence_threshold"]:
                        stability_factors.append(0.7)  # Good coherence
                    else:
                        stability_factors.append(0.3)  # Poor coherence
        
        # Add strain-based stability factor
        if "memory" in changes:
            memory_data = changes["memory"]
            
            # Factor 1: Strain thresholds indicate instability
            strain_thresholds = memory_data.get("strain_thresholds", [])
            if strain_thresholds:
                # More thresholds = less stability, with diminishing effect
                threshold_count = len(strain_thresholds)
                strain_stability = max(0.0, 1.0 - 0.2 * min(5, threshold_count))
                stability_factors.append(strain_stability)
            
            # Factor 2: Average strain level
            region_updates = memory_data.get("region_updates", {})
            if region_updates:
                strain_values = [update.get("strain", 0.0) 
                                for update in region_updates.values()
                                if isinstance(update, dict)]
                if strain_values:
                    avg_strain = sum(strain_values) / len(strain_values)
                    # Lower strain = more stability
                    strain_level_stability = max(0.0, 1.0 - avg_strain)
                    stability_factors.append(strain_level_stability)
        
        # Add observer-based stability factor
        if "observers" in changes:
            observer_data = changes["observers"]
            
            # Factor: Consensus events indicate stability
            consensus_events = observer_data.get("consensus_events", [])
            if consensus_events:
                consensus_stability = min(1.0, 0.5 + 0.1 * len(consensus_events))
                stability_factors.append(consensus_stability)
            
            # Factor: Too many collapse events may indicate instability
            collapse_events = observer_data.get("collapse_events", [])
            if collapse_events:
                collapse_count = len(collapse_events)
                if collapse_count > 10:  # Arbitrary threshold
                    collapse_stability = max(0.0, 1.0 - 0.05 * (collapse_count - 10))
                    stability_factors.append(collapse_stability)
        
        # Add recursive boundary stability factor
        if "recursive" in changes:
            recursive_data = changes["recursive"]
            
            # Boundary events indicate instability
            boundary_events = recursive_data.get("boundary_events", [])
            if boundary_events:
                boundary_stability = max(0.0, 1.0 - 0.1 * len(boundary_events))
                stability_factors.append(boundary_stability)
        
        # Calculate overall stability if factors available
        if stability_factors:
            # Apply weighted average if there are multiple factors
            if len(stability_factors) > 1:
                # Give higher weight to strain and coherence factors
                weights = []
                for i, _ in enumerate(stability_factors):
                    if i < 2:  # First two factors (coherence, strain)
                        weights.append(2.0)
                    else:
                        weights.append(1.0)
                
                # Calculate weighted average
                stability = sum(f * w for f, w in zip(stability_factors, weights)) / sum(weights)
            else:
                stability = stability_factors[0]
            
            # Add to history
            self.stability_history.append((time, stability))
    
    def _update_osh_metrics(self, changes: Dict[str, Any]) -> None:
        """Update OSH-specific metrics from simulation data.
        
        Args:
            changes: Changes dictionary from step results
        """
        # Calculate Recursive Simulation Potential (RSP)
        coherence = 0.0
        entropy = 1.0  # Default high entropy
        strain = 0.5   # Default medium strain
        
        # Extract coherence if available
        if "coherence" in changes:
            coherence_data = changes["coherence"]
            if "coherence_level" in coherence_data:
                coherence = coherence_data["coherence_level"]
        
        # Extract entropy if available
        if "memory" in changes:
            memory_data = changes["memory"]
            if "entropy_level" in memory_data:
                entropy = memory_data["entropy_level"]
        
        # Extract strain if available
        if "memory" in changes:
            memory_data = changes["memory"]
            region_updates = memory_data.get("region_updates", {})
            if region_updates:
                strain_values = [update.get("strain", 0.0) 
                                for update in region_updates.values()
                                if isinstance(update, dict)]
                if strain_values:
                    strain = sum(strain_values) / len(strain_values)
        
        # Calculate stability metrics
        self.osh_metrics["coherence_stability"] = coherence
        self.osh_metrics["system_strain"] = strain
        
        # Calculate RSP
        # Formula: RSP = (Coherence * (1 - Entropy)) / (Strain + epsilon)
        epsilon = 0.001  # Prevent division by zero
        rsp = (coherence * (1.0 - entropy)) / (strain + epsilon)
        self.osh_metrics["rsp"] = rsp
        
        # Extract recursive depth if available
        if "recursive" in changes:
            recursive_data = changes["recursive"]
            if "max_depth" in recursive_data:
                self.osh_metrics["recursion_depth"] = recursive_data["max_depth"]
        
        # Calculate entropy flux rate
        if "memory" in changes:
            memory_data = changes["memory"]
            if "entropy_delta" in memory_data:
                self.osh_metrics["entropy_flux"] = abs(memory_data["entropy_delta"])
    
    def _prune_time_series_if_needed(self) -> None:
        """Prune time series data if it exceeds maximum history size."""
        max_history = self.config["max_history_size"]
        
        for series_name, data_points in self.time_series.items():
            if len(data_points) > max_history:
                # Keep only the most recent points
                self.time_series[series_name] = data_points[-max_history:]
        
        # Prune stability history
        if len(self.stability_history) > max_history:
            self.stability_history = self.stability_history[-max_history:]
    
    def generate_summary(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a comprehensive simulation summary.
        
        Analyzes time series data, event counts, and stability metrics to produce
        a detailed report of the simulation run. Includes trends, patterns, and
        statistical analysis of key metrics.
        
        Args:
            results: Optional simulation results to include
            
        Returns:
            Dict[str, Any]: Detailed simulation summary and analysis
        """
        try:
            # Check if we have cached analysis
            if self.analysis_cache:
                # If results are the same as cached, return cached analysis
                cached_results_key = "results_hash"
                if cached_results_key in self.analysis_cache:
                    results_hash = hash(str(results)) if results else None
                    if results_hash == self.analysis_cache[cached_results_key]:
                        return self.analysis_cache["summary"].copy()
            
            # Generate new summary
            summary = {
                "time_series_analysis": self._analyze_time_series(),
                "event_summary": self._summarize_events(),
                "stability_analysis": self._analyze_stability(),
                "osh_metrics": self._get_osh_metrics(),
                "histogram_data": self._generate_histograms(),
                "metadata": {
                    "generation_time": datetime.now().isoformat(),
                    "simulation_duration": self._get_simulation_duration(),
                    "data_points": {k: len(v) for k, v in self.time_series.items()},
                    "total_events": sum(self.event_counts.values())
                }
            }
            
            # Include results if provided
            if results:
                summary["results"] = results
                # Cache results hash for future comparison
                self.analysis_cache["results_hash"] = hash(str(results))
            
            # Cache the summary
            self.analysis_cache["summary"] = summary.copy()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}", exc_info=True)
            return {
                "error": str(e),
                "metadata": {
                    "generation_time": datetime.now().isoformat(),
                    "error_type": type(e).__name__
                }
            }
    
    def _analyze_time_series(self) -> Dict[str, Any]:
        """Analyze time series data for trends and patterns.
        
        Performs statistical analysis on time series data, including trend detection,
        oscillation identification, and forecasting. Uses curve fitting for trends
        and Fourier analysis for oscillatory patterns when sufficient data is available.
        
        Returns:
            Dict[str, Any]: Time series analysis results
        """
        analysis = {}
        
        # Analyze each time series
        for series_name, data_points in self.time_series.items():
            if not data_points or len(data_points) < 2:
                continue
                
            # Extract time and values
            times = np.array([t for t, _ in data_points])
            values = np.array([v for _, v in data_points])
            
            try:
                # Calculate basic statistics
                series_analysis = {
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std_dev": float(np.std(values)),
                    "start_value": float(values[0]),
                    "end_value": float(values[-1]),
                    "duration": float(times[-1] - times[0]) if len(times) > 1 else 0.0
                }
                
                # Calculate percentiles
                series_analysis["percentiles"] = {
                    "25": float(np.percentile(values, 25)),
                    "50": float(np.percentile(values, 50)),
                    "75": float(np.percentile(values, 75)),
                    "90": float(np.percentile(values, 90))
                }
                
                # Calculate trend direction
                if len(values) > 5:
                    # Use simple linear regression
                    x = np.arange(len(values))
                    
                    # Calculate slope using least squares
                    coeffs = np.polyfit(x, values, 1)
                    slope = float(coeffs[0])
                    intercept = float(coeffs[1])
                    
                    # Calculate R-squared for fit quality
                    y_pred = slope * x + intercept
                    ss_total = np.sum((values - np.mean(values))**2)
                    ss_residual = np.sum((values - y_pred)**2)
                    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                    
                    # Determine trend direction and strength
                    if abs(slope) < 0.001:
                        trend = "stable"
                        strength = "flat"
                    else:
                        trend = "increasing" if slope > 0 else "decreasing"
                        # Classify trend strength based on R-squared
                        if r_squared > 0.7:
                            strength = "strong"
                        elif r_squared > 0.4:
                            strength = "moderate"
                        else:
                            strength = "weak"
                            
                    series_analysis["trend"] = trend
                    series_analysis["trend_strength"] = strength
                    series_analysis["slope"] = slope
                    series_analysis["r_squared"] = float(r_squared)
                    
                    # Check for oscillatory behavior
                    if len(values) > 10:
                        # Detrend the data
                        detrended = values - (slope * x + intercept)
                        
                        # Count sign changes
                        sign_changes = sum(1 for i in range(len(detrended)-1) 
                                        if detrended[i] * detrended[i+1] < 0)
                        
                        # For oscillatory behavior, expect many sign changes
                        if sign_changes > len(detrended) / 3:
                            series_analysis["pattern"] = "oscillatory"
                            # Estimate period as twice the average distance between sign changes
                            if sign_changes > 0:
                                estimated_period = 2 * (times[-1] - times[0]) / sign_changes
                                series_analysis["estimated_period"] = float(estimated_period)
                                series_analysis["oscillation_frequency"] = float(sign_changes / (times[-1] - times[0]))
                            
                            # For sufficient data, try spectral analysis
                            if len(values) > 20:
                                try:
                                    # Apply Fourier transform to find dominant frequencies
                                    # We need uniform time spacing for standard FFT
                                    if np.allclose(np.diff(times), np.diff(times)[0], rtol=0.1):
                                        # Uniform spacing case
                                        from scipy import signal
                                        freqs, power = signal.periodogram(detrended)
                                        # Find dominant frequency
                                        if len(freqs) > 1:
                                            idx = np.argmax(power[1:]) + 1  # Skip DC component
                                            dominant_freq = freqs[idx]
                                            if dominant_freq > 0:
                                                series_analysis["dominant_period"] = float(1.0 / dominant_freq)
                                except Exception as e:
                                    self.logger.debug(f"Spectral analysis failed: {str(e)}")
                        else:
                            series_analysis["pattern"] = "trend"
                
                # Add to analysis dictionary
                analysis[series_name] = series_analysis
            
            except Exception as e:
                self.logger.warning(f"Error analyzing time series {series_name}: {str(e)}")
                # Add basic info despite error
                analysis[series_name] = {
                    "error": str(e),
                    "data_points": len(data_points)
                }
            
        return analysis
    
    def _summarize_events(self) -> Dict[str, Any]:
        """Summarize event counts and patterns.
        
        Categorizes and analyzes event occurrences, including frequency analysis,
        temporal clustering, and category distribution.
        
        Returns:
            Dict[str, Any]: Event summary
        """
        total_events = sum(self.event_counts.values())
        
        # Basic event summary
        summary = {
            "total_events": total_events,
            "event_counts": dict(self.event_counts),
            "event_frequencies": {
                event: count / max(1, total_events)
                for event, count in self.event_counts.items()
            }
        }
        
        # Categorize events
        categories = {
            category: {
                "count": sum(self.event_counts.get(event, 0) for event in events),
                "events": {event: self.event_counts.get(event, 0) for event in events}
            }
            for category, events in self.EVENT_CATEGORIES.items()
        }
        
        # Calculate category percentages
        for category, data in categories.items():
            if total_events > 0:
                data["percentage"] = data["count"] / total_events
            else:
                data["percentage"] = 0.0
        
        summary["categories"] = categories
        
        # Identify dominant event types (>10% of total)
        if total_events > 0:
            dominant_events = [
                event for event, count in self.event_counts.items()
                if count / total_events > 0.1
            ]
            summary["dominant_events"] = dominant_events
        
        # Add critical thresholds exceeded
        if "critical_regions" in self.metrics:
            summary["critical_regions"] = self.metrics["critical_regions"]
        
        return summary
    
    def _analyze_stability(self) -> Dict[str, Any]:
        """Analyze stability trends over time.
        
        Evaluates system stability based on the calculated stability scores,
        classifying the system state and identifying trends. Includes statistical
        analysis and critical point identification.
        
        Returns:
            Dict[str, Any]: Stability analysis
        """
        if not self.stability_history or len(self.stability_history) < 2:
            return {"status": "insufficient_data"}
            
        # Extract stability values
        times = np.array([t for t, _ in self.stability_history])
        stability_values = np.array([s for _, s in self.stability_history])
        
        # Calculate basic statistics
        mean_stability = float(np.mean(stability_values))
        median_stability = float(np.median(stability_values))
        std_dev = float(np.std(stability_values))
        min_stability = float(np.min(stability_values))
        max_stability = float(np.max(stability_values))
        
        # Determine stability class
        # Determine stability class
        if mean_stability > 0.8:
            stability_class = "highly_stable"
        elif mean_stability > 0.6:
            stability_class = "stable"
        elif mean_stability > 0.4:
            stability_class = "meta_stable"
        elif mean_stability > 0.2:
            stability_class = "unstable"
        else:
            stability_class = "critical"
            
        # Analyze trend
        # Use more sophisticated windows for trend analysis
        window_size = min(5, len(stability_values) // 5)
        if window_size > 0:
            start_window = stability_values[:window_size]
            end_window = stability_values[-window_size:]
            
            start_avg = float(np.mean(start_window))
            end_avg = float(np.mean(end_window))
            
            # Calculate trend slope
            x = np.arange(len(stability_values))
            slope, intercept = np.polyfit(x, stability_values, 1)
            
            # Determine trend direction and significance
            if abs(end_avg - start_avg) < 0.05:
                trend = "stable"
                significance = "minor"
            else:
                if end_avg > start_avg:
                    trend = "improving"
                else:
                    trend = "degrading"
                
                # Determine significance
                delta = abs(end_avg - start_avg)
                if delta > 0.2:
                    significance = "major"
                elif delta > 0.1:
                    significance = "moderate"
                else:
                    significance = "minor"
        else:
            trend = "unknown"
            significance = "unknown"
            slope = 0.0
        
        # Identify stability inflection points
        inflection_points = []
        if len(stability_values) > 10:
            # Calculate first derivative
            deriv = np.diff(stability_values)
            
            # Find sign changes in derivative (inflection points)
            sign_changes = []
            for i in range(len(deriv) - 1):
                if deriv[i] * deriv[i+1] < 0:
                    sign_changes.append(i + 1)  # +1 because diff reduces length by 1
            
            # Only keep significant inflections (filter noise)
            threshold = 0.05
            for idx in sign_changes:
                if idx > 1 and idx < len(stability_values) - 1:
                    if abs(stability_values[idx] - stability_values[idx-1]) > threshold:
                        inflection_points.append({
                            "time": float(times[idx]),
                            "stability": float(stability_values[idx]),
                            "direction": "improving" if deriv[idx] > 0 else "degrading"
                        })
        
        # Check for critical stability episodes
        critical_episodes = []
        if len(stability_values) > 2:
            critical_threshold = 0.2
            in_critical = False
            episode_start = None
            
            for i, (t, stab) in enumerate(zip(times, stability_values)):
                if stab < critical_threshold and not in_critical:
                    in_critical = True
                    episode_start = t
                elif stab >= critical_threshold and in_critical:
                    in_critical = False
                    critical_episodes.append({
                        "start_time": float(episode_start),
                        "end_time": float(t),
                        "duration": float(t - episode_start),
                        "min_stability": float(min(stability_values[i-j] for j in range(1, i+1) 
                                                  if i-j >= 0 and times[i-j] >= episode_start))
                    })
            
            # Handle case where episode is ongoing
            if in_critical:
                critical_episodes.append({
                    "start_time": float(episode_start),
                    "end_time": float(times[-1]),
                    "duration": float(times[-1] - episode_start),
                    "min_stability": float(min(stability_values[i] for i in range(len(times)) 
                                            if times[i] >= episode_start)),
                    "ongoing": True
                })
        
        return {
            "mean_stability": mean_stability,
            "median_stability": median_stability,
            "min_stability": min_stability,
            "max_stability": max_stability,
            "stability_variance": std_dev,
            "stability_class": stability_class,
            "stability_trend": trend,
            "trend_significance": significance,
            "trend_slope": float(slope),
            "initial_stability": float(stability_values[0]),
            "final_stability": float(stability_values[-1]),
            "inflection_points": inflection_points,
            "critical_episodes": critical_episodes
        }
    
    def _get_osh_metrics(self) -> Dict[str, Any]:
        """Get OSH-specific metrics with detailed analysis.
        
        Returns:
            Dict[str, Any]: OSH metrics with analysis
        """
        # Start with current metric values
        metrics = {k: float(v) for k, v in self.osh_metrics.items()}
        
        # Add analysis for RSP
        rsp = self.osh_metrics["rsp"]
        if rsp > 2.0:
            rsp_class = "exceptional"
            description = "Strong recursive simulation capacity"
        elif rsp > 1.0:
            rsp_class = "high"
            description = "Good recursive simulation capacity"
        elif rsp > 0.5:
            rsp_class = "moderate"
            description = "Adequate recursive simulation capacity"
        elif rsp > 0.2:
            rsp_class = "low"
            description = "Limited recursive simulation capacity"
        else:
            rsp_class = "critical"
            description = "Minimal recursive simulation capacity"
        
        metrics["rsp_class"] = rsp_class
        metrics["rsp_description"] = description
        
        # Calculate RSP stability
        if "rsp" in self.time_series and len(self.time_series["rsp"]) > 1:
            rsp_values = [v for _, v in self.time_series["rsp"]]
            metrics["rsp_stability"] = 1.0 - float(np.std(rsp_values) / max(0.0001, np.mean(rsp_values)))
        
        return metrics
    
    def _generate_histograms(self) -> Dict[str, Any]:
        """Generate histogram data for key metrics.
        
        Returns:
            Dict[str, Any]: Histogram data for visualization
        """
        histograms = {}
        
        # Generate histograms for stability, coherence, and strain
        for series_name in ["coherence_change", "avg_strain"]:
            if series_name in self.time_series and len(self.time_series[series_name]) > 10:
                values = [v for _, v in self.time_series[series_name]]
                
                # Compute bins appropriately for the data range
                data_min = min(values)
                data_max = max(values)
                if data_min == data_max:
                    continue
                
                # Compute appropriate number of bins
                n_bins = min(20, max(5, len(values) // 10))
                bins = np.linspace(data_min, data_max, n_bins)
                
                # Compute histogram
                hist, bin_edges = np.histogram(values, bins=bins)
                
                # Convert to serializable format
                histograms[series_name] = {
                    "counts": [int(h) for h in hist],
                    "bin_edges": [float(e) for e in bin_edges],
                    "bin_centers": [float((bin_edges[i] + bin_edges[i+1])/2) for i in range(len(bin_edges)-1)]
                }
        
        # Also generate histogram for stability if available
        if self.stability_history and len(self.stability_history) > 10:
            values = [s for _, s in self.stability_history]
            
            # Compute histogram with fixed bins for stability (0.0 to 1.0)
            bins = np.linspace(0.0, 1.0, 11)  # 10 bins from 0 to 1
            hist, bin_edges = np.histogram(values, bins=bins)
            
            # Convert to serializable format
            histograms["stability"] = {
                "counts": [int(h) for h in hist],
                "bin_edges": [float(e) for e in bin_edges],
                "bin_centers": [float((bin_edges[i] + bin_edges[i+1])/2) for i in range(len(bin_edges)-1)]
            }
        
        return histograms
    
    def _get_simulation_duration(self) -> float:
        """Get the simulation duration in seconds.
        
        Returns:
            float: Simulation duration in seconds
        """
        if not self.stability_history:
            return 0.0
        
        return self.stability_history[-1][0] - self.stability_history[0][0] if len(self.stability_history) > 1 else 0.0
    
    def export_report(self, filename: str = None, format: str = "json") -> str:
        """Export the simulation report to a file.
        
        Args:
            filename: Optional filename, defaults to timestamped name
            format: Export format, one of "json", "csv", or "txt"
            
        Returns:
            str: Path to the exported file
            
        Raises:
            ValueError: If the format is not supported
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simulation_report_{timestamp}.{format}"
        
        # Generate the report summary
        summary = self.generate_summary()
        
        try:
            if format.lower() == "json":
                # Export as JSON
                with open(filename, 'w') as f:
                    json.dump(summary, f, indent=2)
                return filename
            
            elif format.lower() == "csv":
                # Export time series data as CSV
                import csv
                
                # Create a directory for multiple CSV files
                base_name = os.path.splitext(filename)[0]
                os.makedirs(base_name, exist_ok=True)
                
                # Export each time series to a separate CSV file
                for series_name, data_points in self.time_series.items():
                    csv_path = os.path.join(base_name, f"{series_name}.csv")
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(["time", "value"])
                        for time, value in data_points:
                            writer.writerow([time, value])
                
                # Export event counts
                events_path = os.path.join(base_name, "events.csv")
                with open(events_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["event_type", "count"])
                    for event, count in self.event_counts.items():
                        writer.writerow([event, count])
                
                # Export metadata
                metadata_path = os.path.join(base_name, "metadata.csv")
                with open(metadata_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["property", "value"])
                    writer.writerow(["start_time", self.start_time.isoformat()])
                    writer.writerow(["export_time", datetime.now().isoformat()])
                    writer.writerow(["total_events", sum(self.event_counts.values())])
                    
                    # Add OSH metrics
                    for key, value in self.osh_metrics.items():
                        writer.writerow([key, value])
                
                return base_name
            
            elif format.lower() == "txt":
                # Export as formatted text
                with open(filename, 'w') as f:
                    f.write("SIMULATION REPORT\n")
                    f.write("=" * 80 + "\n\n")
                    
                    # Write metadata
                    f.write("METADATA\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Generated: {datetime.now().isoformat()}\n")
                    f.write(f"Simulation Duration: {self._get_simulation_duration()} time units\n")
                    f.write(f"Total Events: {sum(self.event_counts.values())}\n\n")
                    
                    # Write stability analysis
                    f.write("STABILITY ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    stability = self._analyze_stability()
                    if "status" in stability and stability["status"] == "insufficient_data":
                        f.write("Insufficient data for stability analysis\n\n")
                    else:
                        f.write(f"Mean Stability: {stability.get('mean_stability', 'N/A'):.4f}\n")
                        f.write(f"Stability Class: {stability.get('stability_class', 'N/A')}\n")
                        f.write(f"Stability Trend: {stability.get('stability_trend', 'N/A')} ({stability.get('trend_significance', 'N/A')})\n")
                        f.write(f"Initial Stability: {stability.get('initial_stability', 'N/A'):.4f}\n")
                        f.write(f"Final Stability: {stability.get('final_stability', 'N/A'):.4f}\n\n")
                    
                    # Write event summary
                    f.write("EVENT SUMMARY\n")
                    f.write("-" * 40 + "\n")
                    for event, count in sorted(self.event_counts.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"{event}: {count}\n")
                    f.write("\n")
                    
                    # Write OSH metrics
                    f.write("OSH METRICS\n")
                    f.write("-" * 40 + "\n")
                    osh_metrics = self._get_osh_metrics()
                    f.write(f"RSP (Recursive Simulation Potential): {osh_metrics.get('rsp', 'N/A'):.4f}\n")
                    f.write(f"RSP Class: {osh_metrics.get('rsp_class', 'N/A')}\n")
                    f.write(f"Coherence Stability: {osh_metrics.get('coherence_stability', 'N/A'):.4f}\n")
                    f.write(f"System Strain: {osh_metrics.get('system_strain', 'N/A'):.4f}\n")
                    f.write(f"Entropy Flux: {osh_metrics.get('entropy_flux', 'N/A'):.4f}\n")
                    
                return filename
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            self.logger.error(f"Error exporting report: {str(e)}", exc_info=True)
            raise
    
    def reset(self) -> None:
        """Reset all collected data."""
        self.metrics = defaultdict(dict)
        self.time_series = defaultdict(list)
        self.event_counts = defaultdict(int)
        self.stability_history = []
        self.analysis_cache.clear()
        self.osh_metrics = {
            "rsp": 0.0,
            "coherence_stability": 0.0,
            "entropy_flux": 0.0,
            "recursion_depth": 0.0,
            "system_strain": 0.0
        }
        self.start_time = datetime.now()
        self.logger.info("SimulationReportBuilder reset")