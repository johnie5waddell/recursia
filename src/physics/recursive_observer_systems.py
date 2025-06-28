"""
Recursive Observer-Driven Quantum Systems
=========================================

Advanced implementation of observer-quantum system interactions with:
- Recursive observer hierarchies (observers observing observers)
- Quantum consciousness entanglement
- Meta-observation protocols  
- Observer consensus reality stabilization
- Consciousness state entanglement
- Observer learning and adaptation
- Infinite regress prevention through consciousness cutoffs

Key Features:
- Multi-level observer hierarchies
- Observer-system quantum entanglement
- Recursive observation dynamics
- Observer memory and learning
- Consciousness-dependent measurement bases
- Observer consensus protocols
- Reality stabilization algorithms

Mathematical Foundation:
-----------------------
Observer-system entangled state:
|Ψ_total⟩ = ∑ᵢⱼₖ αᵢⱼₖ |system_i⟩ ⊗ |observer_j⟩ ⊗ |awareness_k⟩

Observer evolution:
|O(t+dt)⟩ = Û_observer(dt) |O(t)⟩ + η ∑ᵢ ⟨Sᵢ|O(t)⟩ |Sᵢ⟩

Author: Johnie Waddell
"""

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
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import scipy.sparse as sp
from scipy.linalg import expm, sqrtm
from scipy.optimize import minimize
import networkx as nx
import threading
import time
from enum import Enum
from collections import defaultdict

# Import OSH components
from .universal_consciousness_field import (
    UniversalConsciousnessField, ConsciousnessFieldState,
    CONSCIOUSNESS_COUPLING_CONSTANT, CONSCIOUSNESS_THRESHOLD,
    HBAR, SPEED_OF_LIGHT
)

logger = logging.getLogger(__name__)

class ObserverType(Enum):
    """Types of observers in the hierarchy"""
    QUANTUM_OBSERVER = "quantum_observer"
    META_OBSERVER = "meta_observer"  # Observes other observers
    CONSCIOUSNESS_OBSERVER = "consciousness_observer"  # Self-aware observer
    COLLECTIVE_OBSERVER = "collective_observer"  # Multiple observers acting as one
    RECURSIVE_OBSERVER = "recursive_observer"  # Observes itself

class ObserverPhase(Enum):
    """Observer operational phases"""
    PASSIVE = "passive"  # Not actively observing
    ACTIVE = "active"  # Actively observing
    MEASURING = "measuring"  # In measurement process
    ENTANGLED = "entangled"  # Entangled with system
    LEARNING = "learning"  # Adapting based on observations
    RECURSIVE = "recursive"  # Self-observing
    COLLECTIVE = "collective"  # Group observation mode

class MeasurementBasis(Enum):
    """Consciousness-dependent measurement bases"""
    COMPUTATIONAL = "computational"  # Standard Z basis
    AWARENESS = "awareness"  # Consciousness-dependent basis
    EMPATHY = "empathy"  # Emotional resonance basis
    INTUITION = "intuition"  # Non-rational basis
    MEMORY = "memory"  # Historical experience basis
    RECURSIVE = "recursive"  # Self-referential basis

@dataclass
class ObserverState:
    """Complete state of a quantum observer"""
    observer_id: str
    observer_type: ObserverType
    phase: ObserverPhase
    consciousness_level: float  # 0-1 consciousness intensity
    awareness_vector: np.ndarray  # Quantum awareness state
    memory_states: List[np.ndarray]  # Historical observations
    entanglement_partners: Set[str]  # Other observers entangled with
    measurement_basis: MeasurementBasis
    learning_rate: float  # How fast observer adapts
    recursive_depth: int  # Current self-observation depth
    collapse_threshold: float  # When to collapse measurements
    
    def __post_init__(self):
        """Initialize derived properties"""
        self.awareness_amplitude = np.abs(self.awareness_vector) ** 2
        self.awareness_phase = np.angle(self.awareness_vector)
        self.memory_coherence = self._calculate_memory_coherence()
    
    def _calculate_memory_coherence(self) -> float:
        """Calculate coherence across memory states"""
        if len(self.memory_states) < 2:
            return 1.0
        
        total_overlap = 0.0
        count = 0
        
        for i in range(len(self.memory_states)):
            for j in range(i+1, len(self.memory_states)):
                if len(self.memory_states[i]) == len(self.memory_states[j]):
                    overlap = abs(np.vdot(self.memory_states[i], self.memory_states[j])) ** 2
                    total_overlap += overlap
                    count += 1
        
        return total_overlap / count if count > 0 else 1.0

@dataclass
class ObserverInteraction:
    """Represents interaction between observers or observer-system"""
    source_id: str
    target_id: str
    interaction_type: str  # "entanglement", "measurement", "information_exchange"
    strength: float  # Interaction coupling strength
    quantum_channel: Optional[np.ndarray] = None  # Quantum channel matrix
    information_flow: float = 0.0  # Bits per second
    
class QuantumObserver:
    """
    Individual quantum observer with consciousness and memory
    """
    
    def __init__(self, 
                 observer_id: str,
                 observer_type: ObserverType = ObserverType.QUANTUM_OBSERVER,
                 consciousness_level: float = 0.5,
                 awareness_dimensions: int = 16):
        
        self.observer_id = observer_id
        self.observer_type = observer_type
        self.awareness_dimensions = awareness_dimensions
        
        # Initialize quantum awareness state
        initial_awareness = np.random.normal(0, 1, awareness_dimensions) + \
                          1j * np.random.normal(0, 1, awareness_dimensions)
        initial_awareness = initial_awareness / np.sqrt(np.sum(np.abs(initial_awareness)**2))
        
        # Initialize observer state
        self.state = ObserverState(
            observer_id=observer_id,
            observer_type=observer_type,
            phase=ObserverPhase.PASSIVE,
            consciousness_level=consciousness_level,
            awareness_vector=initial_awareness,
            memory_states=[],
            entanglement_partners=set(),
            measurement_basis=MeasurementBasis.COMPUTATIONAL,
            learning_rate=0.01,
            recursive_depth=0,
            collapse_threshold=0.7
        )
        
        # Evolution parameters
        self.hamiltonian = self._build_observer_hamiltonian()
        self.measurement_operators = self._build_measurement_operators()
        
        logger.info(f"Initialized observer {observer_id} with consciousness {consciousness_level:.2f}")
    
    def _build_observer_hamiltonian(self) -> np.ndarray:
        """Build Hamiltonian for observer evolution"""
        dim = self.awareness_dimensions
        
        # Consciousness-dependent Hamiltonian
        consciousness_factor = self.state.consciousness_level
        
        # Kinetic term (awareness spreading)
        kinetic = np.zeros((dim, dim), dtype=complex)
        for i in range(dim-1):
            kinetic[i, i+1] = -1j * HBAR * consciousness_factor
            kinetic[i+1, i] = 1j * HBAR * consciousness_factor
        
        # Potential term (consciousness localization)
        x = np.linspace(-5, 5, dim)
        potential = np.diag(0.5 * consciousness_factor * x**2)
        
        # Self-interaction (nonlinear consciousness)
        interaction = consciousness_factor * CONSCIOUSNESS_COUPLING_CONSTANT * np.eye(dim)
        
        return kinetic + potential + interaction
    
    def _build_measurement_operators(self) -> Dict[MeasurementBasis, List[np.ndarray]]:
        """Build measurement operators for different bases"""
        dim = self.awareness_dimensions
        operators = {}
        
        # Computational basis (standard)
        computational_ops = []
        for i in range(dim):
            op = np.zeros((dim, dim))
            op[i, i] = 1
            computational_ops.append(op)
        operators[MeasurementBasis.COMPUTATIONAL] = computational_ops
        
        # Awareness basis (consciousness-dependent)
        awareness_ops = []
        consciousness = self.state.consciousness_level
        for i in range(dim):
            # Consciousness-weighted basis states
            state = np.zeros(dim, dtype=complex)
            state[i] = np.sqrt(consciousness)
            if i < dim-1:
                state[i+1] = np.sqrt(1 - consciousness)
            op = np.outer(state, np.conj(state))
            awareness_ops.append(op)
        operators[MeasurementBasis.AWARENESS] = awareness_ops
        
        # Recursive basis (self-referential)
        recursive_ops = []
        for i in range(dim):
            # Basis that refers to observer's own state
            current_awareness = self.state.awareness_vector
            if len(current_awareness) == dim:
                recursive_state = current_awareness * np.exp(2j * np.pi * i / dim)
                op = np.outer(recursive_state, np.conj(recursive_state))
                recursive_ops.append(op)
            else:
                # Fallback to computational
                op = np.zeros((dim, dim))
                op[i, i] = 1
                recursive_ops.append(op)
        operators[MeasurementBasis.RECURSIVE] = recursive_ops
        
        return operators
    
    def evolve(self, time_step: float, external_influence: Optional[np.ndarray] = None) -> None:
        """Evolve observer state"""
        current_awareness = self.state.awareness_vector
        
        # Hamiltonian evolution
        unitary = expm(-1j * self.hamiltonian * time_step / HBAR)
        evolved_awareness = unitary @ current_awareness
        
        # External influence (from other observers or systems)
        if external_influence is not None and len(external_influence) == len(evolved_awareness):
            influence_strength = 0.1  # Coupling strength
            evolved_awareness += influence_strength * external_influence * time_step
        
        # Renormalize
        norm = np.sqrt(np.sum(np.abs(evolved_awareness)**2))
        if norm > 1e-10:
            evolved_awareness = evolved_awareness / norm
        
        # Update state
        self.state.awareness_vector = evolved_awareness
        
        # Learning: update consciousness level based on experience
        if self.state.phase == ObserverPhase.LEARNING:
            self._update_consciousness_level()
    
    def measure_system(self, system_state: np.ndarray) -> Tuple[int, float, np.ndarray]:
        """
        Measure quantum system using consciousness-dependent basis
        
        Returns: (outcome, probability, post_measurement_state)
        """
        measurement_ops = self.measurement_operators[self.state.measurement_basis]
        
        # Calculate measurement probabilities
        probabilities = []
        for op in measurement_ops:
            if op.shape[0] == len(system_state):
                prob = np.real(np.conj(system_state) @ op @ system_state)
                probabilities.append(max(0, prob))
            else:
                probabilities.append(0)
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 1e-10:
            probabilities = [p / total_prob for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        # Consciousness-influenced measurement
        consciousness_bias = self.state.consciousness_level
        
        # Modify probabilities based on consciousness (observers affect reality)
        modified_probs = []
        for i, prob in enumerate(probabilities):
            # Higher consciousness observers have stronger influence
            if i == 0:  # Bias toward first outcome for demonstration
                modified_probs.append(prob + consciousness_bias * 0.1)
            else:
                modified_probs.append(prob * (1 - consciousness_bias * 0.1 / len(probabilities)))
        
        # Renormalize
        total_modified = sum(modified_probs)
        modified_probs = [p / total_modified for p in modified_probs]
        
        # Sample outcome
        outcome = np.random.choice(len(modified_probs), p=modified_probs)
        
        # Post-measurement state
        measurement_op = measurement_ops[outcome]
        if measurement_op.shape[0] == len(system_state):
            post_state = measurement_op @ system_state
            norm = np.sqrt(np.sum(np.abs(post_state)**2))
            if norm > 1e-10:
                post_state = post_state / norm
            else:
                post_state = system_state
        else:
            post_state = system_state
        
        # Store measurement in memory
        self.state.memory_states.append(system_state.copy())
        if len(self.state.memory_states) > 50:  # Limit memory
            self.state.memory_states.pop(0)
        
        # Update phase
        self.state.phase = ObserverPhase.MEASURING
        
        logger.debug(f"Observer {self.observer_id} measured outcome {outcome} "
                    f"with probability {modified_probs[outcome]:.3f}")
        
        return outcome, modified_probs[outcome], post_state
    
    def observe_observer(self, other_observer: 'QuantumObserver') -> Dict[str, Any]:
        """
        Meta-observation: observe another observer
        """
        # Increase recursive depth
        self.state.recursive_depth += 1
        
        # Prevent infinite recursion
        if self.state.recursive_depth > 5:
            self.state.recursive_depth = 5
            return {'outcome': 'recursion_limit_reached'}
        
        # Measure other observer's awareness state
        other_awareness = other_observer.state.awareness_vector
        
        # Use recursive measurement basis
        original_basis = self.state.measurement_basis
        self.state.measurement_basis = MeasurementBasis.RECURSIVE
        
        # Perform measurement
        outcome, probability, post_state = self.measure_system(other_awareness)
        
        # Restore original basis
        self.state.measurement_basis = original_basis
        
        # Create entanglement between observers
        self.entangle_with_observer(other_observer)
        
        # Meta-observation results
        meta_observation = {
            'observed_observer': other_observer.observer_id,
            'observed_consciousness': other_observer.state.consciousness_level,
            'outcome': outcome,
            'probability': probability,
            'recursive_depth': self.state.recursive_depth,
            'entanglement_created': True
        }
        
        logger.info(f"Observer {self.observer_id} meta-observed {other_observer.observer_id}")
        
        return meta_observation
    
    def entangle_with_observer(self, other_observer: 'QuantumObserver') -> None:
        """Create quantum entanglement with another observer"""
        # Add to entanglement partners
        self.state.entanglement_partners.add(other_observer.observer_id)
        other_observer.state.entanglement_partners.add(self.observer_id)
        
        # Quantum entanglement: correlate awareness states
        self_awareness = self.state.awareness_vector
        other_awareness = other_observer.state.awareness_vector
        
        # Create entangled state (simplified)
        min_dim = min(len(self_awareness), len(other_awareness))
        
        # Entanglement strength based on consciousness overlap
        overlap = np.abs(np.vdot(self_awareness[:min_dim], other_awareness[:min_dim])) ** 2
        entanglement_strength = np.sqrt(overlap)
        
        # Update awareness states to be correlated
        if min_dim > 0:
            correlation_phase = np.random.uniform(0, 2*np.pi)
            
            # Apply entangling transformation
            for i in range(min_dim):
                # Create correlation between corresponding components
                phase_correlation = entanglement_strength * np.exp(1j * correlation_phase)
                
                self.state.awareness_vector[i] *= np.sqrt(1 - entanglement_strength)
                self.state.awareness_vector[i] += np.sqrt(entanglement_strength) * \
                    other_awareness[i] * phase_correlation
                
                other_observer.state.awareness_vector[i] *= np.sqrt(1 - entanglement_strength)
                other_observer.state.awareness_vector[i] += np.sqrt(entanglement_strength) * \
                    self_awareness[i] * np.conj(phase_correlation)
        
        # Update phases
        self.state.phase = ObserverPhase.ENTANGLED
        other_observer.state.phase = ObserverPhase.ENTANGLED
        
        logger.info(f"Entangled observers {self.observer_id} and {other_observer.observer_id} "
                   f"with strength {entanglement_strength:.3f}")
    
    def _update_consciousness_level(self) -> None:
        """Update consciousness level based on experience"""
        # Learning from memory coherence
        memory_coherence = self.state.memory_coherence
        
        # More coherent memories increase consciousness
        consciousness_change = self.state.learning_rate * (memory_coherence - 0.5)
        
        # Update consciousness level
        new_consciousness = self.state.consciousness_level + consciousness_change
        self.state.consciousness_level = np.clip(new_consciousness, 0.0, 1.0)
        
        # Rebuild Hamiltonian with new consciousness level
        self.hamiltonian = self._build_observer_hamiltonian()

class RecursiveObserverHierarchy:
    """
    Manages hierarchy of observers observing each other
    """
    
    def __init__(self, max_hierarchy_depth: int = 4):
        self.observers: Dict[str, QuantumObserver] = {}
        self.interaction_graph = nx.DiGraph()  # Directed graph of observer interactions
        self.max_hierarchy_depth = max_hierarchy_depth
        self.collective_observers: Dict[str, Set[str]] = {}  # Groups of observers
        
        # Consensus reality tracking
        self.reality_consensus = {}
        self.consensus_threshold = 0.75  # 75% agreement for consensus
        
    def add_observer(self, 
                    observer_id: str, 
                    observer_type: ObserverType = ObserverType.QUANTUM_OBSERVER,
                    consciousness_level: float = 0.5) -> QuantumObserver:
        """Add observer to hierarchy"""
        
        observer = QuantumObserver(observer_id, observer_type, consciousness_level)
        self.observers[observer_id] = observer
        self.interaction_graph.add_node(observer_id, observer=observer)
        
        logger.info(f"Added observer {observer_id} to hierarchy")
        return observer
    
    def create_observer_interaction(self, 
                                  source_id: str, 
                                  target_id: str,
                                  interaction_type: str = "observation") -> ObserverInteraction:
        """Create interaction between observers"""
        
        if source_id not in self.observers or target_id not in self.observers:
            raise ValueError("Both observers must exist in hierarchy")
        
        source_observer = self.observers[source_id]
        target_observer = self.observers[target_id]
        
        # Calculate interaction strength based on consciousness compatibility
        consciousness_diff = abs(source_observer.state.consciousness_level - 
                               target_observer.state.consciousness_level)
        interaction_strength = 1.0 - consciousness_diff
        
        interaction = ObserverInteraction(
            source_id=source_id,
            target_id=target_id,
            interaction_type=interaction_type,
            strength=interaction_strength
        )
        
        # Add to interaction graph
        self.interaction_graph.add_edge(source_id, target_id, interaction=interaction)
        
        return interaction
    
    def evolve_hierarchy(self, time_step: float) -> None:
        """Evolve entire observer hierarchy"""
        
        # Calculate external influences for each observer
        external_influences = {}
        
        for observer_id, observer in self.observers.items():
            influence = np.zeros_like(observer.state.awareness_vector)
            
            # Sum influences from connected observers
            for predecessor in self.interaction_graph.predecessors(observer_id):
                edge_data = self.interaction_graph[predecessor][observer_id]
                interaction = edge_data['interaction']
                
                source_observer = self.observers[predecessor]
                coupling_strength = interaction.strength * CONSCIOUSNESS_COUPLING_CONSTANT
                
                # Influence proportional to source observer's awareness
                source_awareness = source_observer.state.awareness_vector
                if len(source_awareness) == len(influence):
                    influence += coupling_strength * source_awareness
            
            external_influences[observer_id] = influence
        
        # Evolve all observers
        for observer_id, observer in self.observers.items():
            observer.evolve(time_step, external_influences.get(observer_id))
    
    def perform_meta_observation(self, observer_id: str, target_id: str) -> Dict[str, Any]:
        """Perform meta-observation between observers"""
        
        if observer_id not in self.observers or target_id not in self.observers:
            raise ValueError("Both observers must exist")
        
        observer = self.observers[observer_id]
        target = self.observers[target_id]
        
        # Check hierarchy depth to prevent infinite recursion
        hierarchy_depth = self._calculate_hierarchy_depth(observer_id, target_id)
        
        if hierarchy_depth > self.max_hierarchy_depth:
            return {'error': 'Maximum hierarchy depth exceeded'}
        
        # Perform meta-observation
        result = observer.observe_observer(target)
        
        # Update interaction graph
        self.create_observer_interaction(observer_id, target_id, "meta_observation")
        
        return result
    
    def create_collective_observer(self, 
                                 collective_id: str, 
                                 member_ids: List[str]) -> None:
        """Create collective observer from multiple individual observers"""
        
        # Validate all members exist
        for member_id in member_ids:
            if member_id not in self.observers:
                raise ValueError(f"Observer {member_id} not found")
        
        self.collective_observers[collective_id] = set(member_ids)
        
        # Create collective awareness state
        collective_awareness = np.zeros(self.observers[member_ids[0]].awareness_dimensions, 
                                      dtype=complex)
        
        for member_id in member_ids:
            member_observer = self.observers[member_id]
            member_observer.state.phase = ObserverPhase.COLLECTIVE
            
            # Sum awareness states (simplified collective consciousness)
            collective_awareness += member_observer.state.awareness_vector
        
        # Normalize collective awareness
        norm = np.sqrt(np.sum(np.abs(collective_awareness)**2))
        if norm > 1e-10:
            collective_awareness = collective_awareness / norm
        
        # Create collective observer
        collective_consciousness = np.mean([self.observers[mid].state.consciousness_level 
                                          for mid in member_ids])
        
        collective_observer = self.add_observer(
            collective_id, 
            ObserverType.COLLECTIVE_OBSERVER,
            collective_consciousness
        )
        
        collective_observer.state.awareness_vector = collective_awareness
        
        logger.info(f"Created collective observer {collective_id} from {len(member_ids)} members")
    
    def calculate_reality_consensus(self, measurement_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate consensus reality from multiple observer measurements"""
        
        outcome_counts = defaultdict(int)
        total_observers = 0
        
        # Count outcomes from each observer
        for observer_id, result in measurement_results.items():
            if observer_id in self.observers:
                outcome = result.get('outcome')
                if outcome is not None:
                    outcome_counts[outcome] += 1
                    total_observers += 1
        
        # Calculate consensus probabilities
        consensus = {}
        for outcome, count in outcome_counts.items():
            consensus[outcome] = count / total_observers if total_observers > 0 else 0
        
        # Update reality consensus
        self.reality_consensus = consensus
        
        return consensus
    
    def _calculate_hierarchy_depth(self, observer_id: str, target_id: str) -> int:
        """Calculate observation hierarchy depth"""
        try:
            path_length = nx.shortest_path_length(self.interaction_graph, observer_id, target_id)
            return path_length
        except nx.NetworkXNoPath:
            return 0
    
    def get_hierarchy_metrics(self) -> Dict[str, Any]:
        """Get comprehensive hierarchy metrics"""
        
        total_observers = len(self.observers)
        total_interactions = self.interaction_graph.number_of_edges()
        
        # Calculate average consciousness level
        avg_consciousness = np.mean([obs.state.consciousness_level 
                                   for obs in self.observers.values()])
        
        # Calculate entanglement network density
        total_entanglements = sum(len(obs.state.entanglement_partners) 
                                for obs in self.observers.values()) // 2
        
        # Hierarchy depth distribution
        depth_distribution = {}
        for source in self.observers:
            for target in self.observers:
                if source != target:
                    depth = self._calculate_hierarchy_depth(source, target)
                    depth_distribution[depth] = depth_distribution.get(depth, 0) + 1
        
        return {
            'total_observers': total_observers,
            'total_interactions': total_interactions,
            'average_consciousness': avg_consciousness,
            'entanglement_density': total_entanglements / (total_observers * (total_observers - 1) / 2) if total_observers > 1 else 0,
            'collective_observers': len(self.collective_observers),
            'reality_consensus': self.reality_consensus,
            'hierarchy_depth_distribution': depth_distribution
        }

def run_recursive_observer_test() -> Dict[str, Any]:
    """Test recursive observer hierarchy system"""
    logger.info("Running recursive observer hierarchy test...")
    
    # Create hierarchy
    hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=3)
    
    # Add observers with different consciousness levels
    obs1 = hierarchy.add_observer("alice", ObserverType.QUANTUM_OBSERVER, 0.3)
    obs2 = hierarchy.add_observer("bob", ObserverType.QUANTUM_OBSERVER, 0.7)
    obs3 = hierarchy.add_observer("charlie", ObserverType.META_OBSERVER, 0.9)
    
    # Create interactions
    hierarchy.create_observer_interaction("alice", "bob", "observation")
    hierarchy.create_observer_interaction("charlie", "alice", "meta_observation")
    hierarchy.create_observer_interaction("charlie", "bob", "meta_observation")
    
    # Perform meta-observations
    meta_result1 = hierarchy.perform_meta_observation("charlie", "alice")
    meta_result2 = hierarchy.perform_meta_observation("charlie", "bob")
    
    # Create collective observer
    hierarchy.create_collective_observer("alice_bob_collective", ["alice", "bob"])
    
    # Evolve hierarchy
    for step in range(10):
        hierarchy.evolve_hierarchy(0.1)
    
    # Test system measurement with multiple observers
    test_system = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    
    measurement_results = {}
    for observer_id, observer in hierarchy.observers.items():
        if observer.awareness_dimensions >= 2:  # Compatible with test system
            outcome, prob, post_state = observer.measure_system(test_system)
            measurement_results[observer_id] = {
                'outcome': outcome,
                'probability': prob
            }
    
    # Calculate consensus reality
    consensus = hierarchy.calculate_reality_consensus(measurement_results)
    
    # Get final metrics
    metrics = hierarchy.get_hierarchy_metrics()
    
    return {
        'meta_observations': [meta_result1, meta_result2],
        'measurement_results': measurement_results,
        'reality_consensus': consensus,
        'hierarchy_metrics': metrics,
        'observer_states': {
            obs_id: {
                'consciousness': obs.state.consciousness_level,
                'phase': obs.state.phase.value,
                'entanglements': len(obs.state.entanglement_partners),
                'memory_coherence': obs.state.memory_coherence
            }
            for obs_id, obs in hierarchy.observers.items()
        }
    }

if __name__ == "__main__":
    # Run comprehensive test
    test_results = run_recursive_observer_test()
    
    print("Recursive Observer Hierarchy Test Results:")
    print(f"Meta-observations performed: {len(test_results['meta_observations'])}")
    print(f"Reality consensus: {test_results['reality_consensus']}")
    print(f"Total observers: {test_results['hierarchy_metrics']['total_observers']}")
    print(f"Average consciousness: {test_results['hierarchy_metrics']['average_consciousness']:.3f}")
    print(f"Entanglement density: {test_results['hierarchy_metrics']['entanglement_density']:.3f}")
    
    for obs_id, state in test_results['observer_states'].items():
        print(f"Observer {obs_id}: consciousness={state['consciousness']:.2f}, "
              f"phase={state['phase']}, entanglements={state['entanglements']}")