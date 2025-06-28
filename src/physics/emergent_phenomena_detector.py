from typing import Any, Dict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
from pyparsing import deque


class EmergentPhenomenaDetector:
    """
    Advanced detector for emergent patterns and phenomena in OSH simulations.
    
    Uses statistical methods, clustering, and time series analysis to identify
    emergent patterns across coherence, entropy, strain, and observer data.
    """
    
    def __init__(self, history_window=100):
        """Initialize detector with history tracking.
        
        Args:
            history_window: Number of time steps to retain for analysis
        """
        self.history_window = history_window
        self.coherence_history = deque(maxlen=history_window)
        self.entropy_history = deque(maxlen=history_window)
        self.strain_history = deque(maxlen=history_window)
        self.observer_history = deque(maxlen=history_window)
        self.field_history = {}
        
        # Detection thresholds
        self.coherence_wave_threshold = 0.3
        self.critical_strain_threshold = 0.8
        self.recursion_boundary_threshold = 0.7
        self.resonance_threshold = 0.85
        
    def record_state(self, time, coherence_values=None, entropy_values=None, 
                    strain_values=None, observer_data=None, field_data=None,
                    # Legacy parameter names for backward compatibility
                    coherence=None, entropy=None, strain=None, observer=None, field=None):
        """Record current state for time series analysis.
        
        Args:
            time: Current simulation time
            coherence_values: Dict of state coherence values
            entropy_values: Dict of state entropy values
            strain_values: Dict of memory region strain values
            observer_data: Dict of observer metrics
            field_data: Dict of field metrics
            
        Legacy Args (deprecated):
            coherence: Same as coherence_values
            entropy: Same as entropy_values  
            strain: Same as strain_values
            observer: Same as observer_data
            field: Same as field_data
        """
        # Handle legacy parameter names
        if coherence is not None and coherence_values is None:
            coherence_values = coherence
        if entropy is not None and entropy_values is None:
            entropy_values = entropy
        if strain is not None and strain_values is None:
            strain_values = strain
        if observer is not None and observer_data is None:
            observer_data = observer
        if field is not None and field_data is None:
            field_data = field
            
        # Ensure we have valid data
        coherence_values = coherence_values or {}
        entropy_values = entropy_values or {}
        strain_values = strain_values or {}
        observer_data = observer_data or {}
        field_data = field_data or {}
        # Record basic state values
        self.coherence_history.append((time, coherence_values))
        self.entropy_history.append((time, entropy_values))
        self.strain_history.append((time, strain_values))
        self.observer_history.append((time, observer_data))
        
        # Record field data
        for field_name, values in field_data.items():
            if field_name not in self.field_history:
                self.field_history[field_name] = deque(maxlen=self.history_window)
            self.field_history[field_name].append((time, values))
    
    def detect_phenomena(self):
        """Detect emergent phenomena in the recorded state history.
        
        Returns:
            Dict[str, Any]: Detected phenomena with details
        """
        phenomena = {}
        
        # Skip if insufficient history
        if len(self.coherence_history) < 5:
            return phenomena
            
        # Detect coherence waves
        coherence_wave = self._detect_coherence_wave()
        if coherence_wave:
            phenomena["coherence_wave"] = coherence_wave
            
        # Detect critical memory strain
        critical_strain = self._detect_critical_strain()
        if critical_strain:
            phenomena["critical_strain"] = critical_strain
            
        # Detect observer consensus
        consensus = self._detect_observer_consensus()
        if consensus:
            phenomena["observer_consensus"] = consensus
            
        # Detect resonance patterns
        resonance = self._detect_resonance()
        if resonance:
            phenomena["resonance"] = resonance
            
        # Detect boundary instability in recursion layers
        boundary_instability = self._detect_boundary_instability()
        if boundary_instability:
            phenomena["boundary_instability"] = boundary_instability
            
        return phenomena
    
    def _detect_coherence_wave(self):
        """Detect coherence wave patterns using statistical methods.
        
        Returns:
            Dict or None: Wave properties if detected
        """
        if len(self.coherence_history) < 5:
            return None
            
        # Extract coherence values across time
        coherence_values = []
        times = []
        
        for time, values in self.coherence_history:
            avg_coherence = np.mean(list(values.values())) if values else 0
            coherence_values.append(avg_coherence)
            times.append(time)
            
        coherence_values = np.array(coherence_values)
        
        # Calculate first and second derivatives
        first_derivative = np.diff(coherence_values)
        if len(first_derivative) > 1:
            second_derivative = np.diff(first_derivative)
            
            # Look for oscillatory patterns (sign changes in derivatives)
            sign_changes = np.sum(np.diff(np.signbit(first_derivative)))
            
            # Check for significant standard deviation (wave amplitude)
            std_dev = np.std(coherence_values)
            
            if std_dev > self.coherence_wave_threshold and sign_changes >= 2:
                # Approximate frequency using FFT if enough data points
                if len(coherence_values) >= 8:
                    # Compute dominant frequency via FFT
                    fft_values = np.abs(np.fft.rfft(coherence_values - np.mean(coherence_values)))
                    freqs = np.fft.rfftfreq(len(coherence_values), d=np.mean(np.diff(times)))
                    
                    if len(freqs) > 1:
                        dominant_idx = np.argmax(fft_values[1:]) + 1  # Skip DC component
                        dominant_freq = freqs[dominant_idx]
                        
                        return {
                            "strength": std_dev,
                            "frequency": float(dominant_freq),
                            "amplitude": float(np.max(coherence_values) - np.min(coherence_values)),
                            "sign_changes": int(sign_changes),
                            "time": times[-1]
                        }
                else:
                    # Simple detection for shorter sequences
                    return {
                        "strength": std_dev,
                        "time": times[-1]
                    }
        
        return None
    
    def _detect_critical_strain(self):
        """Detect critical memory strain conditions.
        
        Returns:
            Dict or None: Strain properties if critical condition detected
        """
        if not self.strain_history:
            return None
            
        # Get latest strain values
        _, strain_values = self.strain_history[-1]
        
        if not strain_values:
            return None
            
        # Calculate average and max strain
        avg_strain = np.mean(list(strain_values.values()))
        max_strain = np.max(list(strain_values.values()))
        
        # Find critical regions (above threshold)
        critical_regions = [region for region, strain in strain_values.items() 
                           if strain > self.critical_strain_threshold]
        
        if avg_strain > 0.7 or max_strain > 0.9 or len(critical_regions) >= 3:
            return {
                "average_strain": float(avg_strain),
                "max_strain": float(max_strain),
                "critical_regions": critical_regions,
                "time": self.strain_history[-1][0]
            }
            
        return None
    
    def _detect_observer_consensus(self):
        """Detect observer consensus events.
        
        Returns:
            Dict or None: Consensus properties if detected
        """
        if not self.observer_history:
            return None
            
        # Get latest observer data
        _, observer_data = self.observer_history[-1]
        
        # Skip if insufficient observers
        if not observer_data or "observers" not in observer_data:
            return None
            
        observers = observer_data.get("observers", {})
        
        # Find observation targets with multiple observers
        target_observers = {}
        for obs_name, obs_data in observers.items():
            if "observations" in obs_data:
                for observation in obs_data["observations"]:
                    target = observation.get("state")
                    strength = observation.get("strength", 0)
                    
                    if target and strength > 0.5:  # Only strong observations
                        if target not in target_observers:
                            target_observers[target] = []
                        target_observers[target].append((obs_name, strength))
        
        # Find consensus (3+ observers on same target)
        consensus_targets = {target: observers for target, observers in target_observers.items() 
                            if len(observers) >= 3}
        
        if consensus_targets:
            # Find strongest consensus
            strongest_target = max(consensus_targets.keys(), 
                                  key=lambda t: sum(s for _, s in consensus_targets[t]))
            
            return {
                "target": strongest_target,
                "observer_count": len(consensus_targets[strongest_target]),
                "cumulative_strength": sum(s for _, s in consensus_targets[strongest_target]),
                "observers": [o for o, _ in consensus_targets[strongest_target]],
                "time": self.observer_history[-1][0]
            }
            
        return None
    
    def _detect_resonance(self):
        """Detect resonance patterns between states or fields.
        
        Returns:
            Dict or None: Resonance properties if detected
        """
        if len(self.coherence_history) < 3:
            return None
            
        # Get latest coherence values
        _, coherence_values = self.coherence_history[-1]
        
        if not coherence_values or len(coherence_values) < 2:
            return None
            
        # Find highly coherent states
        high_coherence_states = [state for state, coherence in coherence_values.items() 
                                if coherence > self.resonance_threshold]
        
        if len(high_coherence_states) < 2:
            return None
            
        # Check for pairs with similar coherence (within 0.05)
        resonance_pairs = []
        
        for i, state1 in enumerate(high_coherence_states):
            for state2 in high_coherence_states[i+1:]:
                if abs(coherence_values[state1] - coherence_values[state2]) < 0.05:
                    resonance_pairs.append((state1, state2))
        
        if resonance_pairs:
            # Get most coherent pair
            strongest_pair = max(resonance_pairs, 
                               key=lambda p: (coherence_values[p[0]] + coherence_values[p[1]])/2)
            
            avg_coherence = (coherence_values[strongest_pair[0]] + 
                            coherence_values[strongest_pair[1]]) / 2
            
            return {
                "state1": strongest_pair[0],
                "state2": strongest_pair[1],
                "coherence": float(avg_coherence),
                "pair_count": len(resonance_pairs),
                "time": self.coherence_history[-1][0]
            }
            
        return None
    
    def _detect_boundary_instability(self):
        """Detect recursive boundary instability.
        
        Returns:
            Dict or None: Boundary instability properties if detected
        """
        # Check if recursion boundary field exists in history
        if "recursion_boundary_field" not in self.field_history or not self.field_history["recursion_boundary_field"]:
            return None
            
        # Get latest field data
        _, field_data = self.field_history["recursion_boundary_field"][-1]
        
        if not field_data:
            return None
            
        # Check for high field values (instability markers)
        field_max = field_data.get("max", 0)
        field_mean = field_data.get("mean", 0)
        
        if field_max > self.recursion_boundary_threshold:
            # Analyze variance if we have enough history
            if len(self.field_history["recursion_boundary_field"]) > 3:
                # Get time series of max values
                max_values = [data.get("max", 0) for _, data in self.field_history["recursion_boundary_field"]]
                
                # Check for increasing trend
                is_increasing = all(max_values[i] <= max_values[i+1] 
                                   for i in range(len(max_values)-3, len(max_values)-1))
                
                return {
                    "max_value": float(field_max),
                    "mean_value": float(field_mean),
                    "increasing": is_increasing,
                    "time": self.field_history["recursion_boundary_field"][-1][0]
                }
            else:
                return {
                    "max_value": float(field_max),
                    "mean_value": float(field_mean),
                    "time": self.field_history["recursion_boundary_field"][-1][0]
                }
                
        return None