import logging
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
from typing import Any, Dict, List, Optional, Tuple, Union, Set, Callable
import datetime

from src.physics.coherence import CoherenceManager
from src.core.event_system import EventSystem

logger = logging.getLogger(__name__)


class ObserverDynamics:
    """
    Models the interaction between observers and quantum states in the Recursia framework.
    
    This class implements the quantum observer effect, where observation influences
    quantum states. It handles:
    - Observer registration and property management
    - Quantum state collapse probabilities based on observation
    - Partial and full wavefunction collapse mechanisms
    - Observer-observer interactions and entanglement
    - Recursive observation hierarchies (observers observing observers)
    - Measurement basis selection and transformation
    
    This is a key component of the Organic Simulation Hypothesis framework, capturing
    how consciousness and observation may affect the underlying quantum substrate.
    """
    
    def __init__(self, coherence_manager: Optional[CoherenceManager] = None, 
                 event_system: Optional[EventSystem] = None):
        """
        Initialize the observer dynamics system.
        
        Args:
            coherence_manager: Optional CoherenceManager instance for quantum coherence calculations.
                If None, a new instance will be created.
            event_system: Optional EventSystem instance for emitting observation events.
                If None, events will not be emitted.
        """
        self.coherence_manager = coherence_manager or CoherenceManager()
        self.event_system = event_system
        
        # Observer and state registries
        self.observers: Dict[str, Dict[str, Any]] = {}  # name -> observer properties
        self.observed_states: Dict[str, List[str]] = {}  # observer_name -> [state_names]
        self.observation_strengths: Dict[Tuple[str, str], float] = {}  # (observer, state) -> strength
        self.observation_history: Dict[str, List[Dict[str, Any]]] = {}  # observer_name -> [observation records]
        self.observer_relationships: Dict[Tuple[str, str], Dict[str, Any]] = {}  # (observer1, observer2) -> relationship data
        
        # Observer dynamics parameters
        self.collapse_threshold = 0.85  # Observer threshold for causing wave function collapse (OSH.md empirical value)
        self.observation_decay = 0.2  # Rate at which observations decay over time
        self.recursive_depth_factor = 0.8  # How much recursive observation affects deeper levels
        self.observer_consensus_factor = 1.5  # Amplification when multiple observers agree
        self.entanglement_threshold = 0.8  # Threshold for observer-state entanglement
        self.coherence_impact_factor = 0.3  # How much observation reduces coherence
        self.measurement_persistence = 0.5  # How long measurement effects persist
        
        # Observer phases
        self.observer_phases: Dict[str, str] = {}  # observer_name -> phase
        self.phase_transitions = {
            "passive": ["active", "learning", "measuring"],
            "active": ["passive", "measuring", "analyzing", "entangled"],
            "measuring": ["active", "analyzing", "collapsed"],
            "analyzing": ["active", "passive", "learning"],
            "learning": ["active", "passive", "measuring"],
            "entangled": ["active", "collapsed", "measuring"],
            "collapsed": ["passive", "reset"]
        }
        
        # Measurement basis options
        self.basis_types = {
            "standard_basis": np.eye(2),  # Computational basis |0⟩, |1⟩
            "hadamard_basis": np.array([[1, 1], [1, -1]]) / np.sqrt(2),  # |+⟩, |-⟩
            "circular_basis": np.array([[1, 1j], [1, -1j]]) / np.sqrt(2),  # |R⟩, |L⟩
            "bell_basis": None,  # Computed as needed for multi-qubit systems
        }
    
    def register_observer(self, name: str, properties: Dict[str, Any]) -> bool:
        """
        Register an observer in the simulation.
        
        Args:
            name: Observer name
            properties: Observer properties dictionary with optional keys:
                - observer_collapse_threshold: Threshold for causing collapse (0-1)
                - observer_self_awareness: Level of self-awareness (0-1)
                - observer_entanglement_sensitivity: Sensitivity to entanglement (0-1)
                - observer_focus: Current state being focused on
                - observer_measurement_bias: Bias towards certain measurement outcomes (-1 to 1)
                - observer_recursion_depth: Maximum depth for recursive observation
                
        Returns:
            bool: True if registration was successful, False if observer already exists
        """
        if name in self.observers:
            logger.warning(f"Observer {name} already exists")
            return False
            
        # Store observer with default properties if not provided
        self.observers[name] = properties.copy()
        
        # Set default properties if not provided
        defaults = {
            'observer_collapse_threshold': self.collapse_threshold,
            'observer_self_awareness': 0.5,
            'observer_entanglement_sensitivity': 0.5,
            'observer_recursion_depth': 1,
            'observer_measurement_bias': 0.0,
            'observer_time_perspective': 1.0,
            'observer_coherence': 1.0,
            'observer_attention_span': 1.0
        }
        
        for key, value in defaults.items():
            if key not in self.observers[name]:
                self.observers[name][key] = value
        
        # Initialize observation collections
        self.observed_states[name] = []
        self.observation_history[name] = []
        self.observer_phases[name] = "passive"
        
        logger.info(f"Registered observer: {name}")
        return True
    
    def update_observer(self, name: str, properties: Dict[str, Any]) -> bool:
        """
        Update properties of an existing observer.
        
        Args:
            name: Observer name
            properties: New properties to update
            
        Returns:
            bool: True if update was successful, False if observer doesn't exist
        """
        if name not in self.observers:
            logger.warning(f"Observer {name} not registered")
            return False
        
        # Update properties
        self.observers[name].update(properties)
        
        # If focus changed, update observed_states
        if 'observer_focus' in properties:
            new_focus = properties['observer_focus']
            if new_focus and new_focus not in self.observed_states[name]:
                self.observed_states[name].append(new_focus)
        
        # Phase transition if properties significantly changed
        current_phase = self.observer_phases.get(name, "passive")
        
        # Self-awareness increase can change phase
        if 'observer_self_awareness' in properties:
            old_awareness = self.observers[name].get('observer_self_awareness', 0.5)
            new_awareness = properties['observer_self_awareness']
            
            if new_awareness > old_awareness + 0.3 and current_phase == "passive":
                self.set_observer_phase(name, "active")
            elif new_awareness > old_awareness + 0.5:
                self.set_observer_phase(name, "analyzing")
        
        logger.debug(f"Updated observer {name} properties: {properties}")
        return True
    
    def delete_observer(self, name: str) -> bool:
        """
        Delete an observer from the simulation.
        
        Args:
            name: Observer name
            
        Returns:
            bool: True if deletion was successful, False if observer doesn't exist
        """
        if name not in self.observers:
            logger.warning(f"Observer {name} not registered")
            return False
        
        # Remove observer from all collections
        del self.observers[name]
        del self.observed_states[name]
        if name in self.observation_history:
            del self.observation_history[name]
        if name in self.observer_phases:
            del self.observer_phases[name]
        
        # Clean up observation strengths
        keys_to_remove = [k for k in self.observation_strengths.keys() if k[0] == name]
        for key in keys_to_remove:
            del self.observation_strengths[key]
        
        # Clean up relationships
        self._clean_observer_relationships(name)
        
        logger.info(f"Deleted observer: {name}")
        return True
    
    def _clean_observer_relationships(self, name: str) -> None:
        """
        Remove all relationships involving an observer.
        
        Args:
            name: Observer name being deleted
        """
        keys_to_remove = [k for k in self.observer_relationships.keys() if name in k]
        for key in keys_to_remove:
            del self.observer_relationships[key]
    
    def get_property(self, observer_name: str, property_name: str, default_value: Any = None) -> Any:
        """
        Get a property value for an observer, with an optional default.
        
        Args:
            observer_name: Observer name
            property_name: Property to retrieve
            default_value: Value to return if property doesn't exist
            
        Returns:
            Any: Property value or default if not found
        """
        if observer_name not in self.observers:
            return default_value
            
        return self.observers[observer_name].get(property_name, default_value)
    
    def set_property(self, observer_name: str, property_name: str, value: Any) -> bool:
        """
        Set a property value for an observer.
        
        Args:
            observer_name: Observer name
            property_name: Property to set
            value: New property value
            
        Returns:
            bool: True if property was set successfully, False if observer doesn't exist
        """
        if observer_name not in self.observers:
            return False
            
        self.observers[observer_name][property_name] = value
        return True
    
    def set_observer_phase(self, observer_name: str, phase: str) -> bool:
        """
        Set the phase of an observer.
        
        Observer phases represent different states of activity and interaction:
        - passive: Default inactive state
        - active: Actively engaging with the environment
        - measuring: Performing measurement/observation
        - analyzing: Processing observed information
        - learning: Incorporating observations into internal model
        - entangled: Entangled with a quantum state or another observer
        - collapsed: Post-measurement collapse state
        
        Args:
            observer_name: Observer name
            phase: New phase
            
        Returns:
            bool: True if phase was set successfully
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer {observer_name} not registered")
            return False
        
        # Check if phase transition is valid
        current_phase = self.observer_phases.get(observer_name, "passive")
        
        if phase == current_phase:
            return True  # Already in this phase
        
        if phase not in self.phase_transitions.get(current_phase, []):
            logger.warning(f"Invalid phase transition for observer {observer_name}: {current_phase} -> {phase}")
            return False
        
        # Record phase change in history
        self.observation_history.setdefault(observer_name, []).append({
            "type": "phase_change",
            "from": current_phase,
            "to": phase,
            "timestamp": np.datetime64('now')
        })
        
        # Update phase
        self.observer_phases[observer_name] = phase
        
        # Emit event if event system is available
        if self.event_system:
            self.event_system.emit(
                'observer_phase_change_event',
                {
                    'observer_name': observer_name,
                    'previous_phase': current_phase,
                    'new_phase': phase
                },
                source="observer_dynamics"
            )
        
        logger.debug(f"Observer {observer_name} phase changed: {current_phase} -> {phase}")
        return True
    
    def get_observer_phase(self, observer_name: str) -> Optional[str]:
        """
        Get the current phase of an observer.
        
        Args:
            observer_name: Observer name
            
        Returns:
            str: Current phase, or None if observer doesn't exist
        """
        if observer_name not in self.observers:
            return None
        
        return self.observer_phases.get(observer_name, "passive")
    
    def register_observation(self, observer_name: str, state_name: str, 
                           strength: Union[float, Dict[str, Any]] = 1.0) -> bool:
        """
        Register an observation relationship between an observer and a state.
        
        Args:
            observer_name: Observer name
            state_name: State being observed
            strength: Observation strength (0 to 1) or dictionary with observation metadata
            
        Returns:
            bool: True if registration was successful
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer {observer_name} not registered")
            return False
        
        # Extract strength and metadata
        observation_metadata = {}
        if isinstance(strength, dict):
            observation_metadata = strength.copy()
            actual_strength = observation_metadata.pop("strength", 1.0)
        else:
            actual_strength = float(strength)
        
        # Add state to observed states list if not already there
        if state_name not in self.observed_states.get(observer_name, []):
            self.observed_states.setdefault(observer_name, []).append(state_name)
        
        # Store observation strength
        self.observation_strengths[(observer_name, state_name)] = max(0.0, min(1.0, actual_strength))
        
        # Record observation in history
        history_entry = {
            "type": "observation",
            "state": state_name,
            "strength": actual_strength,
            "timestamp": np.datetime64('now')
        }
        history_entry.update(observation_metadata)
        
        self.observation_history.setdefault(observer_name, []).append(history_entry)
        
        # Update observer focus
        self.observers[observer_name]['observer_focus'] = state_name
        
        # Set observer to measuring phase if currently passive
        current_phase = self.observer_phases.get(observer_name, "passive")
        if current_phase == "passive":
            self.set_observer_phase(observer_name, "measuring")
        
        # Emit event if event system is available
        if self.event_system:
            event_data = {
                'strength': actual_strength,
                'observer_phase': self.observer_phases.get(observer_name, "passive")
            }
            event_data.update(observation_metadata)
            
            self.event_system.emit_observation_event(
                observer_name, state_name, event_data
            )
        
        logger.debug(f"Observer {observer_name} registered observation of {state_name} with strength {actual_strength}")
        return True
    
    def get_observation_strength(self, observer_name: str, state_name: str) -> float:
        """
        Get the current observation strength between an observer and a state.
        
        Args:
            observer_name: Observer name
            state_name: State name
            
        Returns:
            float: Observation strength (0 to 1)
        """
        if observer_name not in self.observers:
            return 0.0
            
        return self.observation_strengths.get((observer_name, state_name), 0.0)
    
    def get_observation_history(self, observer_name: str, 
                               limit: Optional[int] = None,
                               filter_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the observation history for an observer.
        
        Args:
            observer_name: Observer name
            limit: Optional maximum number of records to return (most recent first)
            filter_type: Optional filter to only return records of a specific type
            
        Returns:
            List[Dict[str, Any]]: Observation history records
        """
        if observer_name not in self.observers:
            return []
            
        history = self.observation_history.get(observer_name, [])
        
        # Apply type filter if provided
        if filter_type:
            history = [record for record in history if record.get('type') == filter_type]
        
        # Apply limit if provided
        if limit is not None and limit > 0:
            history = history[-limit:]
            
        return history
    
    def calculate_collapse_probability(self, state_density_matrix: np.ndarray, 
                                      observer_properties: Dict[str, Any]) -> float:
        """
        Calculate the probability of wave function collapse due to observation.
        
        Args:
            state_density_matrix: Density matrix of the observed state
            observer_properties: Properties of the observer
            
        Returns:
            float: Collapse probability (0 to 1)
        """
        # Extract relevant observer properties
        collapse_bias = observer_properties.get('observer_collapse_threshold', self.collapse_threshold)
        observer_awareness = observer_properties.get('observer_self_awareness', 0.5)
        quantum_sensitivity = observer_properties.get('observer_entanglement_sensitivity', 0.5)
        
        # Calculate state coherence
        coherence = self.coherence_manager.calculate_coherence(state_density_matrix)
        
        # Calculate collapse probability
        # Higher coherence, observer awareness, and sensitivity increase collapse probability
        collapse_probability = collapse_bias * (coherence * quantum_sensitivity + observer_awareness * 0.3)
        
        # Ensure within bounds
        return max(0.0, min(1.0, collapse_probability))
    
    def calculate_measurement_basis(self, observer_properties: Dict[str, Any], 
                                   num_qubits: int = 1) -> np.ndarray:
        """
        Calculate the measurement basis for an observer.
        
        Args:
            observer_properties: Observer properties dictionary
            num_qubits: Number of qubits in the state being measured
            
        Returns:
            np.ndarray: Measurement basis vectors as columns of a matrix
        """
        # Get preferred basis if specified
        basis_name = observer_properties.get('preferred_basis', 'standard_basis')
        
        # Get basis from predefined options
        if basis_name in self.basis_types and self.basis_types[basis_name] is not None:
            single_qubit_basis = self.basis_types[basis_name]
        else:
            # Default to standard basis
            single_qubit_basis = np.eye(2)
        
        # For multi-qubit systems, tensor product the single-qubit basis
        if num_qubits == 1:
            return single_qubit_basis
        elif basis_name == 'bell_basis' and num_qubits == 2:
            # Special case for Bell basis (maximally entangled 2-qubit basis)
            return np.array([
                [1, 0, 0, 1],  # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
                [1, 0, 0, -1],  # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
                [0, 1, 1, 0],  # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
                [0, 1, -1, 0]   # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            ]) / np.sqrt(2)
        else:
            # For other multi-qubit cases, use tensor product of single-qubit basis
            basis = single_qubit_basis
            for _ in range(num_qubits - 1):
                basis = np.kron(basis, single_qubit_basis)
            return basis
    
    def calculate_observer_consensus(self, observers: List[str], target_state: Optional[str] = None) -> float:
        """
        Calculate the consensus factor among multiple observers.
        Higher consensus amplifies observation effects.
        
        Args:
            observers: List of observer names
            target_state: Optional target state to check for focus consensus
            
        Returns:
            float: Consensus factor (1.0 means no consensus effect, >1.0 means amplified effects)
        """
        if len(observers) <= 1:
            return 1.0
        
        # Start with base consensus
        consensus_factor = 1.0
        
        # Check focus consensus
        if target_state:
            # Count observers focused on this target
            focus_count = 0
            total_focus_strength = 0.0
            
            for obs_name in observers:
                if obs_name in self.observers:
                    focus = self.get_property(obs_name, 'observer_focus')
                    if focus == target_state:
                        focus_count += 1
                        focus_strength = self.get_property(obs_name, 'focus_strength', 0.5)
                        total_focus_strength += focus_strength
            
            # Calculate focus consensus factor
            if focus_count > 1:
                focus_ratio = focus_count / len(observers)
                focus_consensus = 1.0 + 0.5 * focus_ratio * min(1.0, total_focus_strength / focus_count)
                consensus_factor *= focus_consensus
        
        # Check threshold consensus (observers with similar collapse thresholds)
        threshold_values = []
        for obs_name in observers:
            if obs_name in self.observers:
                threshold = self.get_property(obs_name, 'observer_collapse_threshold', self.collapse_threshold)
                threshold_values.append(threshold)
        
        if len(threshold_values) > 1:
            # Calculate threshold similarity - lower variance means more consensus
            threshold_variance = np.var(threshold_values) if len(threshold_values) > 1 else 0
            threshold_consensus = 1.0 + 0.5 * np.exp(-5 * threshold_variance)  # Exponential decay with variance
            consensus_factor *= threshold_consensus
        
        # Check type consensus (observers of the same type have stronger consensus)
        type_counts = {}
        for obs_name in observers:
            if obs_name in self.observers:
                obs_type = self.get_property(obs_name, 'type', 'standard_observer')
                type_counts[obs_type] = type_counts.get(obs_type, 0) + 1
        
        # Calculate type homogeneity
        if len(type_counts) > 0:
            max_type_count = max(type_counts.values())
            type_ratio = max_type_count / len(observers)
            type_consensus = 1.0 + 0.3 * (type_ratio - 1/len(type_counts))
            consensus_factor *= type_consensus
        
        # Apply additional factors from observer properties
        total_consensus_bonus = 0.0
        for obs_name in observers:
            consensus_bonus = self.get_property(obs_name, 'consensus_factor', 0.0)
            total_consensus_bonus += consensus_bonus
        
        if total_consensus_bonus > 0:
            bonus_factor = 1.0 + 0.1 * total_consensus_bonus
            consensus_factor *= bonus_factor
        
        # Cap the consensus factor
        return min(2.5, max(1.0, consensus_factor))

    def apply_observation_effect(self, density_matrix: np.ndarray, 
                            observer_properties: Dict[str, Any],
                            observation_strength: float = 1.0,
                            basis_vectors: Optional[np.ndarray] = None,
                            basis_name: Optional[str] = None) -> np.ndarray:
        """
        Apply the effect of observation to a quantum state with improved basis handling.
        
        Args:
            density_matrix: State density matrix
            observer_properties: Observer properties
            observation_strength: Strength of observation (0 to 1)
            basis_vectors: Optional specific measurement basis vectors
            basis_name: Name of basis ('Z_basis', 'X_basis', 'Y_basis', 'Bell_basis')
            
        Returns:
            np.ndarray: Updated density matrix after observation
        """
        # Transform to requested basis if specified
        if basis_name and basis_name != 'Z_basis':
            transformed_matrix, transform = self._transform_to_basis(density_matrix, basis_name)
        else:
            transformed_matrix = density_matrix
            transform = None
        
        # Use provided basis vectors or generate from basis name
        if basis_vectors is None:
            if basis_name == 'Bell_basis':
                basis_vectors = self._get_bell_basis()
            else:
                dimension = density_matrix.shape[0]
                basis_vectors = np.eye(dimension)
        
        # Calculate collapse probability
        collapse_prob = self.calculate_collapse_probability(transformed_matrix, observer_properties)
        collapse_prob *= observation_strength
        
        # Determine if collapse occurs based on probability
        collapse_occurs = np.random.random() < collapse_prob
        
        if collapse_occurs:
            # Calculate outcome probabilities using OSH formula if memory coherence available
            if observer_properties.get('use_osh_collapse', False):
                probabilities = self._calculate_osh_collapse_probabilities(
                    transformed_matrix, basis_vectors, observer_properties
                )
            else:
                # Standard quantum measurement probabilities
                probabilities = np.zeros(len(basis_vectors))
                
                for i, basis_state in enumerate(basis_vectors):
                    if len(basis_state.shape) == 1:  # State vector
                        projector = np.outer(basis_state, basis_state.conj())
                    else:  # Already a projector
                        projector = basis_state
                        
                    probabilities[i] = np.real(np.trace(projector @ transformed_matrix))
                
                # Normalize probabilities
                total_prob = np.sum(probabilities)
                if total_prob > 0:
                    probabilities /= total_prob
                else:
                    # If all probabilities are zero (numerical errors), use uniform distribution
                    probabilities = np.ones(len(basis_vectors)) / len(basis_vectors)
            
            # Choose a basis state based on probabilities
            outcome = np.random.choice(len(basis_vectors), p=probabilities)
            
            # Collapsed state
            basis_state = basis_vectors[outcome]
            if len(basis_state.shape) == 1:  # State vector
                collapsed_matrix = np.outer(basis_state, basis_state.conj())
            else:  # Already a projector
                collapsed_matrix = basis_state
                
            # If we transformed to a different basis, transform back
            if transform is not None:
                collapsed_matrix = transform @ collapsed_matrix @ transform.conj().T
            
            # Ensure trace = 1 (numerical stability)
            trace = np.trace(collapsed_matrix)
            if trace > 0:
                collapsed_matrix /= trace
                
            # Track collapse in coherence manager if available
            if self.coherence_manager:
                self.coherence_manager.register_collapse_event(collapsed_matrix)
                
            return collapsed_matrix
        else:
            # Partial collapse - move towards a more classical state
            # This models the weak observation effect
            
            # Calculate decoherence factor based on observer properties
            observer_sensitivity = observer_properties.get('observer_entanglement_sensitivity', 0.5)
            observer_awareness = observer_properties.get('observer_self_awareness', 0.5)
            
            # Calculate decoherence strength - more aware observers cause more decoherence
            decoherence_factor = collapse_prob * (0.3 + 0.7 * observer_awareness)
            
            # Apply partial decoherence in the measurement basis
            diag = np.diag(np.diag(transformed_matrix))
            off_diag = transformed_matrix - diag
            
            # Reduce off-diagonal elements proportionally to decoherence factor
            # This preserves some coherence but moves toward classical states
            partially_collapsed = diag + off_diag * (1 - decoherence_factor)
            
            # If we transformed to a different basis, transform back
            if transform is not None:
                partially_collapsed = transform @ partially_collapsed @ transform.conj().T
            
            # Ensure it's a valid density matrix
            partially_collapsed = 0.5 * (partially_collapsed + partially_collapsed.conj().T)
            trace = np.trace(partially_collapsed)
            if trace > 0:
                partially_collapsed /= trace
            
            # Track partial decoherence in coherence manager if available
            if self.coherence_manager:
                coherence_delta = -decoherence_factor * self.coherence_manager.calculate_coherence(density_matrix)
                self.coherence_manager.register_coherence_change(partially_collapsed, coherence_delta)
            
            return partially_collapsed

    def register_coherence_manager(self, coherence_manager: CoherenceManager) -> None:
        """
        Set the coherence manager for quantum coherence tracking.
        
        Args:
            coherence_manager: CoherenceManager instance to use
        """
        self.coherence_manager = coherence_manager

    def register_collapse_event(self, observer_name: str, state_name: str, outcome: str, 
                            basis: str = 'Z_basis') -> None:
        """
        Register a collapse event in the system.
        
        Args:
            observer_name: Name of the observer causing collapse
            state_name: Name of the collapsed state
            outcome: Measurement outcome
            basis: Measurement basis
        """
        # Record observation with collapse
        self.register_observation(observer_name, state_name, {
            'type': 'collapse',
            'outcome': outcome,
            'basis': basis,
            'collapsed': True,
            'strength': 1.0
        })
        
        # Set observer to "collapsed" phase if not already
        if self.get_observer_phase(observer_name) != "collapsed":
            self.set_observer_phase(observer_name, "collapsed")
        
        # Register with coherence manager if available
        if self.coherence_manager:
            self.coherence_manager.register_observer_collapse(observer_name, state_name, outcome)
        
        logger.info(f"Observer {observer_name} collapsed state {state_name} to {outcome} in {basis}")
            
    def apply_multiple_observers(self, density_matrix: np.ndarray, 
                            observer_names: List[str],
                            state_name: str,
                            basis_name: Optional[str] = None) -> np.ndarray:
        """
        Apply the effect of multiple observers on a quantum state with enhanced consensus modeling.
        
        Args:
            density_matrix: State density matrix
            observer_names: List of observer names
            state_name: Name of the observed state
            basis_name: Optional measurement basis
            
        Returns:
            np.ndarray: Updated density matrix after multiple observations
        """
        if not observer_names:
            return density_matrix
        
        # Start with the original state
        updated_matrix = density_matrix.copy()
        
        # Calculate consensus factor among observers
        consensus_factor = self.calculate_observer_consensus(observer_names, state_name)
        
        # Transform to the measurement basis if specified
        if basis_name and basis_name != 'Z_basis':
            transformed_matrix, transform = self._transform_to_basis(updated_matrix, basis_name)
            basis_vectors = None
            if basis_name == 'Bell_basis':
                basis_vectors = self._get_bell_basis()
        else:
            transformed_matrix = updated_matrix
            transform = None
            basis_vectors = None
        
        # Apply each observer's effect, amplified by consensus
        for obs_name in observer_names:
            if obs_name in self.observers:
                props = self.observers[obs_name]
                strength = self.get_observation_strength(obs_name, state_name)
                
                # Apply consensus amplification
                adjusted_strength = min(1.0, strength * consensus_factor)
                
                # Apply the observer's effect
                updated_matrix = self.apply_observation_effect(
                    updated_matrix, props, adjusted_strength, 
                    basis_vectors=basis_vectors, basis_name=basis_name)
        
        # Log consensus effect
        logger.debug(f"Applied multiple observers with consensus factor {consensus_factor:.2f}")
        
        # Register with coherence manager if available
        if self.coherence_manager and len(observer_names) > 1:
            coherence_before = self.coherence_manager.calculate_coherence(density_matrix)
            coherence_after = self.coherence_manager.calculate_coherence(updated_matrix)
            self.coherence_manager.register_consensus_effect(state_name, observer_names, 
                                                        coherence_before, coherence_after,
                                                        consensus_factor)
        
        return updated_matrix

    def _get_bell_basis(self) -> List[np.ndarray]:
        """
        Generate the Bell basis states (maximally entangled two-qubit states).
        
        Returns:
            List[np.ndarray]: The four Bell states as normalized vectors
        """
        # Bell basis states (Phi+, Phi-, Psi+, Psi-)
        bell_states = [
            # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex),
            
            # |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
            np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(2)], dtype=complex),
            
            # |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
            np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex),
            
            # |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
            np.array([0, 1/np.sqrt(2), -1/np.sqrt(2), 0], dtype=complex)
        ]
        
        return bell_states

    def _transform_to_basis(self, density_matrix: np.ndarray, basis_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform a density matrix to the specified measurement basis.
        
        Args:
            density_matrix: Original density matrix
            basis_name: Target basis name ('Z_basis', 'X_basis', 'Y_basis', 'Bell_basis')
            
        Returns:
            tuple: (transformed_matrix, transform_operator)
        """
        dim = density_matrix.shape[0]
        
        if basis_name == 'Z_basis' or basis_name is None:
            # Standard computational basis (default)
            return density_matrix, np.eye(dim)
        
        elif basis_name == 'X_basis':
            # X basis transformation (Hadamard for each qubit)
            num_qubits = int(np.log2(dim))
            h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
            transform = h_gate
            
            # Tensor product for multiple qubits
            for _ in range(1, num_qubits):
                transform = np.kron(transform, h_gate)
            
            # Apply transformation
            transformed = transform @ density_matrix @ transform.conj().T
            return transformed, transform
        
        elif basis_name == 'Y_basis':
            # Y basis transformation
            num_qubits = int(np.log2(dim))
            y_transform = np.array([[1, -1j], [1j, 1]]) / np.sqrt(2)
            transform = y_transform
            
            # Tensor product for multiple qubits
            for _ in range(1, num_qubits):
                transform = np.kron(transform, y_transform)
            
            # Apply transformation
            transformed = transform @ density_matrix @ transform.conj().T
            return transformed, transform
        
        elif basis_name == 'Bell_basis':
            # Check if this is a two-qubit system
            if dim != 4:
                raise ValueError(f"Bell basis requires exactly 2 qubits, but the density matrix has dimension {dim}")
            
            # Construct transformation from Bell basis to computational basis
            bell_states = self._get_bell_basis()
            transform = np.column_stack(bell_states)
            
            # Apply transformation
            transformed = transform.conj().T @ density_matrix @ transform
            return transformed, transform
        
        else:
            # Unrecognized basis
            logger.warning(f"Unrecognized basis: {basis_name}, using Z basis as default")
            return density_matrix, np.eye(dim)
    
    def log_recursive_observation(self, observer_name: str, state_name: str, 
                                level: int, strength: float) -> None:
        """
        Log a recursive observation event in the observation history.
        
        Args:
            observer_name: Observer name
            state_name: State name
            level: Recursion level
            strength: Observation strength
        """
        if observer_name not in self.observers:
            return
            
        self.observation_history.setdefault(observer_name, []).append({
            "type": "recursive_observation",
            "state": state_name,
            "level": level,
            "strength": strength,
            "timestamp": np.datetime64('now')
        })
        
        # Emit event if event system is available
        if self.event_system:
            self.event_system.emit(
                'recursive_observation_event',
                {
                    'observer_name': observer_name,
                    'state_name': state_name,
                    'recursion_level': level,
                    'strength': strength
                },
                source="observer_dynamics"
            )
    
    def _calculate_osh_collapse_probabilities(self, 
                                            density_matrix: np.ndarray,
                                            basis_vectors: List[np.ndarray],
                                            observer_properties: Dict[str, Any]) -> np.ndarray:
        """
        Calculate collapse probabilities using OSH formula: P(ψ→φᵢ) = Iᵢ/Σⱼ Iⱼ
        where Iᵢ is the integrated memory coherence for outcome φᵢ.
        
        Args:
            density_matrix: Current quantum state
            basis_vectors: Possible measurement outcomes
            observer_properties: Observer properties including memory coherence
            
        Returns:
            np.ndarray: Probability distribution over outcomes
        """
        memory_coherences = []
        
        # Calculate integrated memory coherence for each possible outcome
        for i, basis_state in enumerate(basis_vectors):
            if len(basis_state.shape) == 1:  # State vector
                projector = np.outer(basis_state, basis_state.conj())
            else:  # Already a projector
                projector = basis_state
            
            # Calculate overlap with current state
            overlap = np.real(np.trace(projector @ density_matrix))
            
            # Calculate memory coherence for this outcome
            # Factors: overlap, observer memory coherence, consciousness level
            observer_memory = observer_properties.get('observer_coherence', 1.0)
            consciousness_level = observer_properties.get('observer_self_awareness', 0.5)
            memory_strain = observer_properties.get('memory_strain', 0.0)
            
            # Integrated memory coherence Iᵢ
            I_i = overlap * observer_memory * consciousness_level * (1.0 - memory_strain)
            memory_coherences.append(max(0.0, I_i))
        
        # Convert to numpy array
        coherences = np.array(memory_coherences)
        
        # Apply OSH formula: P(ψ→φᵢ) = Iᵢ/Σⱼ Iⱼ
        total_coherence = np.sum(coherences)
        
        if total_coherence < 1e-10:  # Numerical safety
            # Equal probabilities when no coherence
            return np.ones(len(basis_vectors)) / len(basis_vectors)
        
        probabilities = coherences / total_coherence
        
        # Ensure normalization (numerical safety)
        probabilities = probabilities / np.sum(probabilities)
        
        return probabilities
    
    def register_event_hook(self, event_type: str, 
                          callback: Callable[[Dict[str, Any]], None]) -> int:
        """
        Register a hook for observer-related events.
        
        Args:
            event_type: Event type to listen for
            callback: Function to call when event occurs
            
        Returns:
            int: Hook ID for later removal, or -1 if event system not available
        """
        if not self.event_system:
            logger.warning("Event system not available for hook registration")
            return -1
        
        # Register the hook
        hook_id = self.event_system.add_listener(event_type, callback)
        
        logger.debug(f"Registered event hook for {event_type}, ID: {hook_id}")
        return hook_id
    
    def remove_event_hook(self, hook_id: int) -> bool:
        """
        Remove an event hook.
        
        Args:
            hook_id: Hook ID returned by register_event_hook
            
        Returns:
            bool: True if hook was removed successfully
        """
        if not self.event_system:
            logger.warning("Event system not available for hook removal")
            return False
        
        # Remove the hook
        success = self.event_system.remove_listener(hook_id)
        
        if success:
            logger.debug(f"Removed event hook: {hook_id}")
        else:
            logger.warning(f"Failed to remove event hook: {hook_id}")
        
        return success
    
    def recursive_observation(self, primary_observer: str, 
                            observed_state: str,
                            density_matrix: np.ndarray,
                            recursion_depth: Optional[int] = None) -> np.ndarray:
        """
        Model recursive observation where an observer observes themselves observing.
        This is a key aspect of the Organic Simulation Hypothesis.
        
        Args:
            primary_observer: Name of the primary observer
            observed_state: Name of the observed state
            density_matrix: Density matrix of the observed state
            recursion_depth: How many levels of recursive observation to model.
                If None, use observer's defined recursion depth.
            
        Returns:
            np.ndarray: Updated density matrix after recursive observation
        """
        try:
            if primary_observer not in self.observers:
                logger.warning(f"Observer {primary_observer} not registered")
                return density_matrix
            
            # Get observer properties
            observer_props = self.observers[primary_observer]
            
            # Determine recursion depth
            if recursion_depth is None:
                recursion_depth = self.get_property(primary_observer, 'observer_recursion_depth', 1)
            
            if recursion_depth <= 0:
                return density_matrix
            
            # First level: direct observation
            observation_strength = self.get_observation_strength(primary_observer, observed_state)
            
            # Skip if no observation relationship
            if observation_strength <= 0:
                return density_matrix
            
            # Apply first-level observation
            updated_matrix = self.apply_observation_effect(density_matrix, observer_props, observation_strength)
            
            # Extract observer self-awareness
            self_awareness = self.get_property(primary_observer, 'observer_self_awareness', 0.5)
            
            # Skip recursive observations if self-awareness is too low
            if self_awareness < 0.2:
                return updated_matrix
            
            # Recursive observations with decreasing impact
            for level in range(1, recursion_depth + 1):
                # Each level gets less impactful
                level_factor = self.recursive_depth_factor ** level
                
                # Strength of recursive observation depends on self-awareness
                recursive_strength = observation_strength * self_awareness * level_factor
                
                # Apply the recursive observation effect
                updated_matrix = self.apply_observation_effect(
                    updated_matrix, observer_props, recursive_strength)
                
                # Log recursive observation
                self.log_recursive_observation(
                    primary_observer, observed_state, level, recursive_strength)
            
            return updated_matrix
        except Exception as e:
            logger.error(f"Error in recursive observation: {e}", exc_info=True)
            # In case of error, return the original matrix to avoid breaking simulation
            return density_matrix
    
    def register_measurement_basis(self, name: str, basis_matrix: np.ndarray) -> bool:
        """
        Register a custom measurement basis.
        
        Args:
            name: Basis name
            basis_matrix: Basis vectors as columns of a matrix
            
        Returns:
            bool: True if registration was successful
        """
        try:
            # Validate matrix
            if not isinstance(basis_matrix, np.ndarray):
                logger.error(f"Basis must be a numpy array, got {type(basis_matrix)}")
                return False
            
            if basis_matrix.ndim != 2:
                logger.error(f"Basis must be a 2D matrix, got {basis_matrix.ndim}D")
                return False
            
            # For a valid basis, columns should be orthonormal
            if basis_matrix.shape[0] != basis_matrix.shape[1]:
                # Not necessarily an error for non-complete bases
                logger.warning(f"Basis matrix is not square: {basis_matrix.shape}")
            
            # Check orthonormality
            if basis_matrix.shape[0] == basis_matrix.shape[1]:
                # Square matrix should be unitary for a proper basis
                product = basis_matrix.conj().T @ basis_matrix
                identity = np.eye(basis_matrix.shape[1])
                if not np.allclose(product, identity, atol=1e-6):
                    logger.warning(f"Basis vectors are not orthonormal")
            
            # Register the basis
            self.basis_types[name] = basis_matrix
            
            logger.info(f"Registered measurement basis: {name}")
            return True
        except Exception as e:
            logger.error(f"Error registering measurement basis: {e}", exc_info=True)
            return False
    
    def measure_with_observer(self, observer_name: str, state_name: str, 
                            density_matrix: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Perform a measurement of a quantum state using a specific observer.
        
        Args:
            observer_name: Observer name
            state_name: State name
            density_matrix: Density matrix of the state
            
        Returns:
            Tuple[np.ndarray, int]: Collapsed density matrix and measurement outcome
        """
        try:
            if observer_name not in self.observers:
                logger.warning(f"Observer {observer_name} not registered")
                return density_matrix, -1
            
            # Get observer properties
            observer_props = self.observers[observer_name]
            
            # Register observation if not already observing
            if self.get_observation_strength(observer_name, state_name) == 0:
                self.register_observation(observer_name, state_name)
            
            # Change observer to measuring phase
            self.set_observer_phase(observer_name, "measuring")
            
            # Get or calculate basis
            dimension = density_matrix.shape[0]
            num_qubits = int(np.log2(dimension) + 0.5)  # Round to nearest int
            basis_vectors = self.calculate_measurement_basis(observer_props, num_qubits)
            
            # Calculate measurement probabilities
            probabilities = np.zeros(len(basis_vectors))
            
            for i, basis_state in enumerate(basis_vectors):
                basis_state = basis_state.reshape(-1, 1)  # Ensure column vector
                projector = basis_state @ basis_state.conj().T
                probabilities[i] = np.real(np.trace(projector @ density_matrix))
            
            # Apply measurement bias
            measurement_bias = self.get_property(observer_name, 'observer_measurement_bias', 0.0)
            if abs(measurement_bias) > 0.01:
                bias_factors = np.linspace(1 + measurement_bias, 1 - measurement_bias, len(probabilities))
                probabilities *= bias_factors
            
            # Normalize probabilities
            total_prob = np.sum(probabilities)
            if total_prob > 0:
                probabilities /= total_prob
            else:
                # If all probabilities are zero (numerical issues), use uniform distribution
                probabilities = np.ones(len(probabilities)) / len(probabilities)
            
            # Choose outcome based on probabilities
            outcome = np.random.choice(len(basis_vectors), p=probabilities)
            
            # Create collapsed state
            basis_state = basis_vectors[outcome].reshape(-1, 1)
            collapsed_matrix = basis_state @ basis_state.conj().T
            
            # Record measurement in history
            self.observation_history.setdefault(observer_name, []).append({
                "type": "measurement",
                "state": state_name,
                "outcome": int(outcome),
                "basis": self.get_property(observer_name, 'preferred_basis', 'standard_basis'),
                "timestamp": np.datetime64('now')
            })
            
            # After measurement, change to analyzing phase
            self.set_observer_phase(observer_name, "analyzing")
            
            # Emit measurement event if event system is available
            if self.event_system:
                self.event_system.emit_quantum_event(
                    'measurement_event',
                    state_name,
                    {
                        'observer_name': observer_name,
                        'outcome': int(outcome),
                        'basis': self.get_property(observer_name, 'preferred_basis', 'standard_basis')
                    }
                )
            
            return collapsed_matrix, int(outcome)
        except Exception as e:
            logger.error(f"Error in measure_with_observer: {e}", exc_info=True)
            # Return original state and invalid outcome
            return density_matrix, -1
    
    def calculate_observer_expectation_value(self, observer_name: str, state_name: str,
                                          density_matrix: np.ndarray, 
                                          observable: np.ndarray) -> float:
        """
        Calculate the expectation value of an observable through a specific observer.
        
        Args:
            observer_name: Observer name
            state_name: State name
            density_matrix: Density matrix of the state
            observable: Hermitian operator representing the observable
            
        Returns:
            float: Expectation value
        """
        try:
            if observer_name not in self.observers:
                logger.warning(f"Observer {observer_name} not registered")
                return 0.0
            
            # Get observer properties
            observer_props = self.observers[observer_name]
            
            # Apply observer's perspective to density matrix
            observation_strength = self.get_observation_strength(observer_name, state_name)
            
            # If already observing, apply observation effect
            if observation_strength > 0:
                modified_density_matrix = self.apply_observation_effect(
                    density_matrix, observer_props, observation_strength)
            else:
                # If not already observing, register a weak observation
                self.register_observation(observer_name, state_name, 0.3)
                modified_density_matrix = self.apply_observation_effect(
                    density_matrix, observer_props, 0.3)
            
            # Calculate expectation value: Tr(ρO)
            expectation = np.real(np.trace(modified_density_matrix @ observable))
            
            # Record in history
            self.observation_history.setdefault(observer_name, []).append({
                "type": "expectation_value",
                "state": state_name,
                "value": float(expectation),
                "timestamp": np.datetime64('now')
            })
            
            return float(expectation)
        except Exception as e:
            logger.error(f"Error calculating observer expectation value: {e}", exc_info=True)
            return 0.0
    
    def simulate_observer_interaction(self, observer1: str, observer2: str,
                                    interaction_strength: float = 0.5) -> bool:
        """
        Simulate interaction between two observers, potentially creating entanglement.
        
        Args:
            observer1: First observer name
            observer2: Second observer name
            interaction_strength: Strength of interaction (0 to 1)
            
        Returns:
            bool: True if interaction was successful
        """
        try:
            if observer1 not in self.observers or observer2 not in self.observers:
                logger.warning(f"Cannot simulate interaction: observer not found")
                return False
            
            if observer1 == observer2:
                logger.warning(f"Cannot simulate interaction: observer cannot interact with itself")
                return False
            
            # Get observer properties
            props1 = self.observers[observer1]
            props2 = self.observers[observer2]
            
            # Get current phases
            phase1 = self.observer_phases.get(observer1, "passive")
            phase2 = self.observer_phases.get(observer2, "passive")
            
            # Calculate interaction factors
            self_awareness1 = self.get_property(observer1, 'observer_self_awareness', 0.5)
            self_awareness2 = self.get_property(observer2, 'observer_self_awareness', 0.5)
            
            entanglement_sensitivity1 = self.get_property(observer1, 'observer_entanglement_sensitivity', 0.5)
            entanglement_sensitivity2 = self.get_property(observer2, 'observer_entanglement_sensitivity', 0.5)
            
            # Higher self-awareness and entanglement sensitivity increase interaction effect
            effective_strength = interaction_strength * (
                (self_awareness1 + self_awareness2) / 2 + 
                (entanglement_sensitivity1 + entanglement_sensitivity2) / 2
            ) / 2
            
            # Cap effective strength
            effective_strength = max(0.1, min(1.0, effective_strength))
            
            # Determine relationship type based on observer properties
            if entanglement_sensitivity1 > 0.7 and entanglement_sensitivity2 > 0.7:
                relationship_type = "entanglement"
            elif self_awareness1 > 0.7 and self_awareness2 > 0.7:
                relationship_type = "consciousness_sharing"
            elif phase1 == "measuring" and phase2 == "measuring":
                relationship_type = "measurement_correlation"
            else:
                relationship_type = "awareness"
            
            # Establish or update relationship
            self.establish_relationship(observer1, observer2, relationship_type, effective_strength)
            
            # Property influence - observers influence each other's properties
            self._apply_observer_property_influence(observer1, observer2, effective_strength)
            
            # Phase transitions based on interaction
            if relationship_type == "entanglement" and effective_strength > self.entanglement_threshold:
                self.set_observer_phase(observer1, "entangled")
                self.set_observer_phase(observer2, "entangled")
            elif phase1 == "passive" and phase2 == "active" and effective_strength > 0.5:
                # Active observer can activate passive observer
                self.set_observer_phase(observer1, "active")
            elif phase1 == "active" and phase2 == "passive" and effective_strength > 0.5:
                # Active observer can activate passive observer
                self.set_observer_phase(observer2, "active")
            
            # Emit interaction event if event system is available
            if self.event_system:
                self.event_system.emit(
                    'observer_interaction_event',
                    {
                        'observer1': observer1,
                        'observer2': observer2,
                        'relationship_type': relationship_type,
                        'strength': effective_strength
                    },
                    source="observer_dynamics"
                )
            
            logger.debug(f"Simulated {relationship_type} interaction between {observer1} and {observer2} with strength {effective_strength}")
            return True
        except Exception as e:
            logger.error(f"Error simulating observer interaction: {e}", exc_info=True)
            return False
    
    def _apply_observer_property_influence(self, observer1: str, observer2: str, 
                                         strength: float) -> None:
        """
        Apply mutual influence between two observers' properties.
        
        Args:
            observer1: First observer name
            observer2: Second observer name
            strength: Interaction strength
        """
        # Properties that can be influenced
        influence_properties = [
            'observer_collapse_threshold',
            'observer_measurement_bias',
            'observer_entanglement_sensitivity',
            'preferred_basis'
        ]
        
        # Get properties for both observers
        props1 = self.observers[observer1]
        props2 = self.observers[observer2]
        
        # Calculate influence factor
        influence_factor = strength * 0.1
        
        # Apply influence to numeric properties
        updates1 = {}
        updates2 = {}
        
        for prop in influence_properties:
            # Skip non-numeric properties except basis preference
            if prop == 'preferred_basis':
                # Special handling for basis preference
                # With some probability, adopt the other observer's preferred basis
                if prop in props1 and prop in props2 and props1[prop] != props2[prop]:
                    if np.random.random() < strength * 0.3:
                        # Observer 1 adopts observer 2's basis
                        updates1[prop] = props2[prop]
                    elif np.random.random() < strength * 0.3:
                        # Observer 2 adopts observer 1's basis
                        updates2[prop] = props1[prop]
            elif prop in props1 and prop in props2:
                # Only apply to numeric properties
                if isinstance(props1[prop], (int, float)) and isinstance(props2[prop], (int, float)):
                    val1 = props1[prop]
                    val2 = props2[prop]
                    
                    # Calculate influence (move slightly toward other observer's value)
                    delta1 = (val2 - val1) * influence_factor
                    delta2 = (val1 - val2) * influence_factor
                    
                    # Apply influence
                    new_val1 = val1 + delta1
                    new_val2 = val2 + delta2
                    
                    # Create updates
                    updates1[prop] = new_val1
                    updates2[prop] = new_val2
        
        # Apply updates if any
        if updates1:
            self.update_observer(observer1, updates1)
        if updates2:
            self.update_observer(observer2, updates2)
    
    def establish_relationship(self, observer1: str, observer2: str,
                              relationship_type: str = "awareness",
                              strength: float = 0.5,
                              properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Establish a relationship between two observers.
        
        Args:
            observer1: First observer name
            observer2: Second observer name
            relationship_type: Type of relationship
            strength: Relationship strength (0 to 1)
            properties: Additional relationship properties
            
        Returns:
            bool: True if relationship was established successfully
        """
        if observer1 not in self.observers or observer2 not in self.observers:
            logger.warning(f"Cannot establish relationship: observer not found")
            return False
        
        if observer1 == observer2:
            logger.warning(f"Cannot establish relationship with self: {observer1}")
            return False
        
        # Sort observers for consistent key
        key = tuple(sorted([observer1, observer2]))
        
        # Create or update relationship
        self.observer_relationships[key] = {
            'type': relationship_type,
            'strength': max(0.0, min(1.0, strength)),
            'established_at': np.datetime64('now'),
            'properties': properties or {}
        }
        
        # Record in both observers' histories
        relationship_record = {
            "type": "relationship_established",
            "relationship_type": relationship_type,
            "other_observer": observer2,
            "strength": strength,
            "timestamp": np.datetime64('now')
        }
        
        self.observation_history.setdefault(observer1, []).append(relationship_record)
        
        relationship_record2 = relationship_record.copy()
        relationship_record2["other_observer"] = observer1
        self.observation_history.setdefault(observer2, []).append(relationship_record2)
        
        # If relationship is entanglement and strong enough, change phases
        if relationship_type == "entanglement" and strength > self.entanglement_threshold:
            self.set_observer_phase(observer1, "entangled")
            self.set_observer_phase(observer2, "entangled")
        
        # Emit event if event system is available
        if self.event_system:
            self.event_system.emit(
                'observer_relationship_event',
                {
                    'observer1': observer1,
                    'observer2': observer2,
                    'relationship_type': relationship_type,
                    'strength': strength
                },
                source="observer_dynamics"
            )
        
        logger.debug(f"Established {relationship_type} relationship between {observer1} and {observer2} with strength {strength}")
        return True
    
    def get_relationship(self, observer1: str, observer2: str) -> Optional[Dict[str, Any]]:
        """
        Get the relationship between two observers.
        
        Args:
            observer1: First observer name
            observer2: Second observer name
            
        Returns:
            dict: Relationship data, or None if no relationship exists
        """
        # Sort observers for consistent key
        key = tuple(sorted([observer1, observer2]))
        
        return self.observer_relationships.get(key)
    
    def update_observations_over_time(self, time_step: float = 1.0) -> None:
        """
        Update all observation relationships over time.
        Observation strength decays naturally if not refreshed.
        
        Args:
            time_step: Time step for the update
        """
        try:
            # Calculate decay factor for this time step
            decay_factor = np.exp(-self.observation_decay * time_step)
            
            # Update all observation strengths
            for key in list(self.observation_strengths.keys()):
                observer_name, state_name = key
                
                # Skip if observer no longer exists
                if observer_name not in self.observers:
                    del self.observation_strengths[key]
                    continue
                
                # Get current strength and observer properties
                current_strength = self.observation_strengths[key]
                
                # Apply different decay rates based on observer properties
                attention_span = self.get_property(observer_name, 'observer_attention_span', 1.0)
                observer_phase = self.observer_phases.get(observer_name, "passive")
                
                # Observers in active phases maintain observations longer
                phase_factor = 1.0
                if observer_phase in ["measuring", "analyzing", "entangled"]:
                    phase_factor = 0.7  # Less decay
                elif observer_phase in ["passive", "collapsed"]:
                    phase_factor = 1.3  # More decay
                
                # Calculate effective decay rate
                effective_decay = decay_factor ** (attention_span / phase_factor)
                
                # Apply decay
                decayed_strength = current_strength * effective_decay
                
                if decayed_strength < 0.01:
                    # Remove observation relationship if strength is negligible
                    del self.observation_strengths[key]
                    if state_name in self.observed_states.get(observer_name, []):
                        self.observed_states[observer_name].remove(state_name)
                    
                    # Record in history
                    self.observation_history.setdefault(observer_name, []).append({
                        "type": "observation_lost",
                        "state": state_name,
                        "timestamp": np.datetime64('now')
                    })
                else:
                    # Update with decayed strength
                    self.observation_strengths[key] = decayed_strength
            
            # Update observer phases
            self._update_observer_phases(time_step)
        except Exception as e:
            logger.error(f"Error updating observations over time: {e}", exc_info=True)
    
    def _update_observer_phases(self, time_step: float) -> None:
        """
        Update observer phases based on current state and elapsed time.
        
        Args:
            time_step: Time step size
        """
        for name, phase in list(self.observer_phases.items()):
            # Skip if observer no longer exists
            if name not in self.observers:
                continue
                
            # Calculate phase transition probabilities based on current phase
            if phase == "active":
                # Active observers gradually become passive
                passive_prob = 0.1 * time_step
                if np.random.random() < passive_prob:
                    self.set_observer_phase(name, "passive")
            
            elif phase == "measuring":
                # Measuring observers become analyzing
                analyzing_prob = 0.2 * time_step
                if np.random.random() < analyzing_prob:
                    self.set_observer_phase(name, "analyzing")
            
            elif phase == "analyzing":
                # Analyzing observers can become active or learning
                if np.random.random() < 0.5:
                    self.set_observer_phase(name, "active")
                else:
                    self.set_observer_phase(name, "learning")
            
            elif phase == "entangled":
                # Entangled observers gradually return to active
                active_prob = 0.05 * time_step
                if np.random.random() < active_prob:
                    self.set_observer_phase(name, "active")
    
    def get_observer_stats(self, observer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about an observer or the entire system.
        
        Args:
            observer_name: Optional observer name, or None for system-wide stats
            
        Returns:
            Dict[str, Any]: Observer or system statistics
        """
        try:
            if observer_name is not None:
                # Stats for a specific observer
                if observer_name not in self.observers:
                    return {}
                
                observer_data = self.observers[observer_name]
                history = self.observation_history.get(observer_name, [])
                
                # Count observation types
                observations = 0
                measurements = 0
                recursive_observations = 0
                phase_changes = 0
                
                for entry in history:
                    entry_type = entry.get('type', '')
                    if entry_type == 'observation':
                        observations += 1
                    elif entry_type == 'measurement':
                        measurements += 1
                    elif entry_type == 'recursive_observation':
                        recursive_observations += 1
                    elif entry_type == 'phase_change':
                        phase_changes += 1
                
                # Get related states
                observed_states = self.observed_states.get(observer_name, [])
                
                # Get relationships
                relationships = []
                for key, rel_data in self.observer_relationships.items():
                    if observer_name in key:
                        other = key[0] if key[1] == observer_name else key[1]
                        relationships.append((other, rel_data['type'], rel_data['strength']))
                
                # Calculate average observation strength
                average_strength = 0
                observation_count = 0
                for key, strength in self.observation_strengths.items():
                    obs_name, _ = key
                    if obs_name == observer_name:
                        average_strength += strength
                        observation_count += 1
                
                if observation_count > 0:
                    average_strength /= observation_count
                
                return {
                    'name': observer_name,
                    'properties': observer_data,
                    'phase': self.observer_phases.get(observer_name, "passive"),
                    'observation_count': observations,
                    'measurement_count': measurements,
                    'recursive_observation_count': recursive_observations,
                    'phase_change_count': phase_changes,
                    'observed_states': observed_states,
                    'relationships': relationships,
                    'average_observation_strength': average_strength,
                    'history_size': len(history),
                    'last_activity': history[-1]['timestamp'] if history else None
                }
            else:
                # System-wide stats
                observer_count = len(self.observers)
                
                # Count phases
                phase_counts = {}
                for phase in self.observer_phases.values():
                    phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
                # Count observation relationships
                total_observations = len(self.observation_strengths)
                
                # Count relationship types
                relationship_counts = {}
                for rel_data in self.observer_relationships.values():
                    rel_type = rel_data['type']
                    relationship_counts[rel_type] = relationship_counts.get(rel_type, 0) + 1
                
                # Calculate total observations
                total_states_observed = sum(len(states) for states in self.observed_states.values())
                
                # Find most active observer
                most_active = None
                most_activity = 0
                for obs_name, history in self.observation_history.items():
                    if len(history) > most_activity:
                        most_activity = len(history)
                        most_active = obs_name
                
                # Calculate average properties
                avg_self_awareness = 0
                avg_collapse_threshold = 0
                avg_entanglement_sensitivity = 0
                
                for obs_props in self.observers.values():
                    avg_self_awareness += obs_props.get('observer_self_awareness', 0.5)
                    avg_collapse_threshold += obs_props.get('observer_collapse_threshold', self.collapse_threshold)
                    avg_entanglement_sensitivity += obs_props.get('observer_entanglement_sensitivity', 0.5)
                
                if observer_count > 0:
                    avg_self_awareness /= observer_count
                    avg_collapse_threshold /= observer_count
                    avg_entanglement_sensitivity /= observer_count
                
                return {
                    'observer_count': observer_count,
                    'phases': phase_counts,
                    'observation_count': total_observations,
                    'states_observed': total_states_observed,
                    'relationship_counts': relationship_counts,
                    'most_active_observer': most_active,
                    'most_active_observer_events': most_activity,
                    'avg_self_awareness': avg_self_awareness,
                    'avg_collapse_threshold': avg_collapse_threshold,
                    'avg_entanglement_sensitivity': avg_entanglement_sensitivity,
                    'entanglement_threshold': self.entanglement_threshold,
                    'collapse_threshold': self.collapse_threshold,
                    'observation_decay': self.observation_decay,
                    'recursive_depth_factor': self.recursive_depth_factor,
                    'observer_consensus_factor': self.observer_consensus_factor,
                    'coherence_impact_factor': self.coherence_impact_factor
                }
        except Exception as e:
            logger.error(f"Error getting observer stats: {e}", exc_info=True)
            return {'error': str(e)} if observer_name else {'error': str(e), 'observer_count': len(self.observers)}