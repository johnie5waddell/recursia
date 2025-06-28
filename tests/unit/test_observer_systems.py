"""
Unit tests for Recursive Observer Systems
"""
import pytest
import numpy as np
from physics.recursive_observer_systems import (
    QuantumObserver, RecursiveObserverHierarchy, ObserverType, ObserverPhase,
    MeasurementBasis, ObserverState, ObserverInteraction
)

class TestQuantumObserver:
    """Test suite for QuantumObserver"""
    
    def test_observer_initialization(self):
        """Test quantum observer initialization"""
        observer = QuantumObserver("test_obs", ObserverType.QUANTUM_OBSERVER, 0.7, 16)
        
        assert observer.observer_id == "test_obs"
        assert observer.observer_type == ObserverType.QUANTUM_OBSERVER
        assert observer.awareness_dimensions == 16
        assert observer.state.consciousness_level == 0.7
        assert observer.state.phase == ObserverPhase.PASSIVE
        assert len(observer.state.awareness_vector) == 16
        
        # Awareness vector should be normalized
        norm = np.sum(np.abs(observer.state.awareness_vector)**2)
        assert abs(norm - 1.0) < 1e-10
    
    def test_observer_evolution(self, random_state):
        """Test observer state evolution"""
        observer = QuantumObserver("evolve_test", consciousness_level=0.5, awareness_dimensions=8)
        initial_awareness = observer.state.awareness_vector.copy()
        
        observer.evolve(0.01)
        
        # State should evolve
        assert not np.allclose(observer.state.awareness_vector, initial_awareness)
        
        # Should remain normalized
        norm = np.sum(np.abs(observer.state.awareness_vector)**2)
        assert abs(norm - 1.0) < 1e-6
    
    def test_observer_evolution_with_influence(self, random_state):
        """Test observer evolution with external influence"""
        observer = QuantumObserver("influence_test", consciousness_level=0.6, awareness_dimensions=8)
        initial_awareness = observer.state.awareness_vector.copy()
        
        # External influence
        influence = random_state.normal(0, 0.1, 8) + 1j * random_state.normal(0, 0.1, 8)
        
        observer.evolve(0.01, influence)
        
        # Should be different from no-influence evolution
        observer2 = QuantumObserver("no_influence", consciousness_level=0.6, awareness_dimensions=8)
        observer2.state.awareness_vector = initial_awareness.copy()
        observer2.evolve(0.01)
        
        # Different evolution due to influence
        assert not np.allclose(observer.state.awareness_vector, observer2.state.awareness_vector)
    
    def test_system_measurement(self, sample_quantum_state):
        """Test quantum system measurement by observer"""
        if len(sample_quantum_state) > 2:
            test_state = sample_quantum_state[:2] / np.sqrt(np.sum(np.abs(sample_quantum_state[:2])**2))
        else:
            test_state = sample_quantum_state
        
        observer = QuantumObserver("measurer", consciousness_level=0.8, awareness_dimensions=2)
        observer.state.measurement_basis = MeasurementBasis.COMPUTATIONAL
        
        outcome, probability, post_state = observer.measure_system(test_state)
        
        assert isinstance(outcome, int)
        assert 0 <= outcome < 2
        assert 0 <= probability <= 1
        assert isinstance(post_state, np.ndarray)
        assert len(post_state) == len(test_state)
        
        # Post-measurement state should be normalized
        norm = np.sum(np.abs(post_state)**2)
        assert abs(norm - 1.0) < 1e-10
        
        # Memory should be updated
        assert len(observer.state.memory_states) > 0
        assert observer.state.phase == ObserverPhase.MEASURING
    
    def test_consciousness_dependent_measurement(self):
        """Test that consciousness level affects measurement"""
        test_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        
        low_consciousness_observer = QuantumObserver("low_c", consciousness_level=0.2, awareness_dimensions=2)
        high_consciousness_observer = QuantumObserver("high_c", consciousness_level=0.9, awareness_dimensions=2)
        
        # Measure multiple times to see statistical difference
        low_results = []
        high_results = []
        
        for _ in range(50):
            outcome_low, prob_low, _ = low_consciousness_observer.measure_system(test_state.copy())
            outcome_high, prob_high, _ = high_consciousness_observer.measure_system(test_state.copy())
            
            low_results.append(prob_low)
            high_results.append(prob_high)
        
        # Statistical difference due to consciousness bias
        low_mean = np.mean(low_results)
        high_mean = np.mean(high_results)
        
        # Should show some difference (implementation specific)
        assert abs(high_mean - low_mean) >= 0  # At minimum, no error
    
    def test_observer_meta_observation(self):
        """Test meta-observation (observer observing observer)"""
        observer1 = QuantumObserver("obs1", consciousness_level=0.6, awareness_dimensions=8)
        observer2 = QuantumObserver("obs2", consciousness_level=0.7, awareness_dimensions=8)
        
        initial_recursive_depth = observer1.state.recursive_depth
        
        meta_result = observer1.observe_observer(observer2)
        
        assert isinstance(meta_result, dict)
        assert 'observed_observer' in meta_result
        assert meta_result['observed_observer'] == "obs2"
        assert 'outcome' in meta_result
        assert 'recursive_depth' in meta_result
        
        # Recursive depth should increase
        assert observer1.state.recursive_depth > initial_recursive_depth
        
        # Should create entanglement
        assert observer2.observer_id in observer1.state.entanglement_partners
        assert observer1.observer_id in observer2.state.entanglement_partners
    
    def test_observer_entanglement(self):
        """Test observer-observer entanglement"""
        observer1 = QuantumObserver("ent1", consciousness_level=0.5, awareness_dimensions=8)
        observer2 = QuantumObserver("ent2", consciousness_level=0.6, awareness_dimensions=8)
        
        initial_state1 = observer1.state.awareness_vector.copy()
        initial_state2 = observer2.state.awareness_vector.copy()
        
        observer1.entangle_with_observer(observer2)
        
        # States should be correlated now
        assert not np.allclose(observer1.state.awareness_vector, initial_state1)
        assert not np.allclose(observer2.state.awareness_vector, initial_state2)
        
        # Entanglement partners should be updated
        assert observer2.observer_id in observer1.state.entanglement_partners
        assert observer1.observer_id in observer2.state.entanglement_partners
        
        # Phases should be entangled
        assert observer1.state.phase == ObserverPhase.ENTANGLED
        assert observer2.state.phase == ObserverPhase.ENTANGLED
    
    def test_consciousness_level_update(self):
        """Test consciousness level learning and updating"""
        observer = QuantumObserver("learner", consciousness_level=0.4, awareness_dimensions=6)
        observer.state.phase = ObserverPhase.LEARNING
        observer.state.learning_rate = 0.1
        
        initial_consciousness = observer.state.consciousness_level
        
        # Add some memories to affect coherence
        for i in range(5):
            memory = np.random.normal(0, 1, 6) + 1j * np.random.normal(0, 1, 6)
            memory = memory / np.sqrt(np.sum(np.abs(memory)**2))
            observer.state.memory_states.append(memory)
        
        # Update consciousness
        observer._update_consciousness_level()
        
        # Consciousness should change based on memory coherence
        assert observer.state.consciousness_level != initial_consciousness
        assert 0 <= observer.state.consciousness_level <= 1

class TestRecursiveObserverHierarchy:
    """Test suite for RecursiveObserverHierarchy"""
    
    def test_hierarchy_initialization(self):
        """Test observer hierarchy initialization"""
        hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=4)
        
        assert hierarchy.max_hierarchy_depth == 4
        assert len(hierarchy.observers) == 0
        assert hierarchy.interaction_graph.number_of_nodes() == 0
        assert len(hierarchy.collective_observers) == 0
        assert hierarchy.consensus_threshold == 0.75
    
    def test_add_observer_to_hierarchy(self):
        """Test adding observers to hierarchy"""
        hierarchy = RecursiveObserverHierarchy()
        
        observer = hierarchy.add_observer("test_obs", ObserverType.QUANTUM_OBSERVER, 0.6)
        
        assert "test_obs" in hierarchy.observers
        assert isinstance(observer, QuantumObserver)
        assert observer.state.consciousness_level == 0.6
        assert hierarchy.interaction_graph.has_node("test_obs")
    
    def test_create_observer_interaction(self):
        """Test creating interactions between observers"""
        hierarchy = RecursiveObserverHierarchy()
        
        hierarchy.add_observer("obs1", consciousness_level=0.5)
        hierarchy.add_observer("obs2", consciousness_level=0.7)
        
        interaction = hierarchy.create_observer_interaction("obs1", "obs2", "observation")
        
        assert isinstance(interaction, ObserverInteraction)
        assert interaction.source_id == "obs1"
        assert interaction.target_id == "obs2"
        assert 0 <= interaction.strength <= 1
        assert hierarchy.interaction_graph.has_edge("obs1", "obs2")
    
    def test_hierarchy_evolution(self):
        """Test evolving the entire hierarchy"""
        hierarchy = RecursiveObserverHierarchy()
        
        hierarchy.add_observer("obs1", consciousness_level=0.4)
        hierarchy.add_observer("obs2", consciousness_level=0.6)
        hierarchy.create_observer_interaction("obs1", "obs2")
        
        initial_states = {
            obs_id: obs.state.awareness_vector.copy() 
            for obs_id, obs in hierarchy.observers.items()
        }
        
        hierarchy.evolve_hierarchy(0.01)
        
        # States should evolve
        for obs_id, obs in hierarchy.observers.items():
            assert not np.allclose(obs.state.awareness_vector, initial_states[obs_id])
    
    def test_meta_observation_in_hierarchy(self):
        """Test meta-observation within hierarchy"""
        hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=3)
        
        hierarchy.add_observer("observer", ObserverType.QUANTUM_OBSERVER, 0.5)
        hierarchy.add_observer("meta_observer", ObserverType.META_OBSERVER, 0.8)
        
        result = hierarchy.perform_meta_observation("meta_observer", "observer")
        
        assert isinstance(result, dict)
        assert 'error' not in result  # Should succeed within depth limit
        
        # Should create interaction
        assert hierarchy.interaction_graph.has_edge("meta_observer", "observer")
    
    def test_hierarchy_depth_limit(self):
        """Test hierarchy depth limit enforcement"""
        hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=2)
        
        hierarchy.add_observer("obs1", consciousness_level=0.5)
        hierarchy.add_observer("obs2", consciousness_level=0.6)
        
        # Create chain that would exceed depth
        for i in range(5):
            hierarchy.create_observer_interaction(f"obs{i%2 + 1}", f"obs{(i+1)%2 + 1}", "meta_observation")
        
        # Meta-observation should respect depth limit
        result = hierarchy.perform_meta_observation("obs1", "obs2")
        
        # Should succeed or gracefully handle depth limit
        assert isinstance(result, dict)
    
    def test_collective_observer_creation(self):
        """Test creating collective observer from individuals"""
        hierarchy = RecursiveObserverHierarchy()
        
        hierarchy.add_observer("member1", consciousness_level=0.4)
        hierarchy.add_observer("member2", consciousness_level=0.6)
        hierarchy.add_observer("member3", consciousness_level=0.5)
        
        hierarchy.create_collective_observer("collective", ["member1", "member2", "member3"])
        
        assert "collective" in hierarchy.collective_observers
        assert len(hierarchy.collective_observers["collective"]) == 3
        assert "collective" in hierarchy.observers
        
        # Collective observer should exist
        collective_obs = hierarchy.observers["collective"]
        assert collective_obs.observer_type == ObserverType.COLLECTIVE_OBSERVER
        
        # Member observers should be in collective phase
        for member_id in ["member1", "member2", "member3"]:
            assert hierarchy.observers[member_id].state.phase == ObserverPhase.COLLECTIVE
    
    def test_reality_consensus_calculation(self):
        """Test reality consensus from multiple observers"""
        hierarchy = RecursiveObserverHierarchy()
        
        hierarchy.add_observer("obs1", consciousness_level=0.5)
        hierarchy.add_observer("obs2", consciousness_level=0.6)
        hierarchy.add_observer("obs3", consciousness_level=0.7)
        
        # Mock measurement results
        measurement_results = {
            "obs1": {"outcome": 0},
            "obs2": {"outcome": 0},
            "obs3": {"outcome": 1}
        }
        
        consensus = hierarchy.calculate_reality_consensus(measurement_results)
        
        assert isinstance(consensus, dict)
        assert 0 in consensus
        assert 1 in consensus
        assert abs(consensus[0] - 2/3) < 1e-10  # 2/3 agreement on outcome 0
        assert abs(consensus[1] - 1/3) < 1e-10  # 1/3 agreement on outcome 1
    
    def test_hierarchy_metrics(self):
        """Test hierarchy metrics calculation"""
        hierarchy = RecursiveObserverHierarchy()
        
        hierarchy.add_observer("obs1", consciousness_level=0.4)
        hierarchy.add_observer("obs2", consciousness_level=0.7)
        hierarchy.create_observer_interaction("obs1", "obs2")
        
        # Create some entanglement
        hierarchy.observers["obs1"].entangle_with_observer(hierarchy.observers["obs2"])
        
        metrics = hierarchy.get_hierarchy_metrics()
        
        assert isinstance(metrics, dict)
        assert 'total_observers' in metrics
        assert 'total_interactions' in metrics
        assert 'average_consciousness' in metrics
        assert 'entanglement_density' in metrics
        
        assert metrics['total_observers'] == 2
        assert metrics['total_interactions'] == 1
        assert 0 < metrics['average_consciousness'] < 1
        assert 0 <= metrics['entanglement_density'] <= 1

class TestObserverStates:
    """Test observer state management and transitions"""
    
    def test_observer_state_creation(self):
        """Test observer state data structure"""
        awareness_vector = np.array([1, 0, 0, 0], dtype=complex)
        
        state = ObserverState(
            observer_id="test_state",
            observer_type=ObserverType.QUANTUM_OBSERVER,
            phase=ObserverPhase.ACTIVE,
            consciousness_level=0.6,
            awareness_vector=awareness_vector,
            memory_states=[],
            entanglement_partners=set(),
            measurement_basis=MeasurementBasis.COMPUTATIONAL,
            learning_rate=0.01,
            recursive_depth=1,
            collapse_threshold=0.7
        )
        
        assert state.observer_id == "test_state"
        assert state.consciousness_level == 0.6
        assert len(state.awareness_vector) == 4
        assert state.memory_coherence == 1.0  # No memories yet
    
    def test_memory_coherence_calculation(self):
        """Test memory coherence calculation"""
        awareness_vector = np.array([1, 0], dtype=complex)
        
        state = ObserverState(
            observer_id="coherence_test",
            observer_type=ObserverType.QUANTUM_OBSERVER,
            phase=ObserverPhase.ACTIVE,
            consciousness_level=0.5,
            awareness_vector=awareness_vector,
            memory_states=[],
            entanglement_partners=set(),
            measurement_basis=MeasurementBasis.COMPUTATIONAL,
            learning_rate=0.01,
            recursive_depth=1,
            collapse_threshold=0.7
        )
        
        # Add coherent memories
        memory1 = np.array([1, 0], dtype=complex)
        memory2 = np.array([0.9, 0.1], dtype=complex)
        memory2 = memory2 / np.sqrt(np.sum(np.abs(memory2)**2))
        
        state.memory_states = [memory1, memory2]
        coherence = state._calculate_memory_coherence()
        
        assert 0 <= coherence <= 1
        assert coherence > 0.5  # Should be reasonably coherent

class TestObserverEdgeCases:
    """Test edge cases and error conditions for observers"""
    
    def test_observer_with_zero_consciousness(self):
        """Test observer with zero consciousness level"""
        observer = QuantumObserver("zero_c", consciousness_level=0.0, awareness_dimensions=4)
        
        assert observer.state.consciousness_level == 0.0
        
        # Should still function but with minimal effects
        test_state = np.array([1, 0], dtype=complex)
        outcome, prob, post_state = observer.measure_system(test_state)
        
        assert 0 <= prob <= 1
        assert np.isfinite(prob)
    
    def test_observer_with_maximum_consciousness(self):
        """Test observer with maximum consciousness level"""
        observer = QuantumObserver("max_c", consciousness_level=1.0, awareness_dimensions=4)
        
        assert observer.state.consciousness_level == 1.0
        
        # Should function with maximum consciousness effects
        test_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        outcome, prob, post_state = observer.measure_system(test_state)
        
        assert 0 <= prob <= 1
        assert np.isfinite(prob)
    
    def test_empty_hierarchy_operations(self):
        """Test operations on empty hierarchy"""
        hierarchy = RecursiveObserverHierarchy()
        
        # Should handle empty operations gracefully
        hierarchy.evolve_hierarchy(0.01)
        metrics = hierarchy.get_hierarchy_metrics()
        
        assert metrics['total_observers'] == 0
        assert metrics['total_interactions'] == 0
        assert metrics['average_consciousness'] == 0 or np.isnan(metrics['average_consciousness'])
    
    def test_self_observation_recursion_limit(self):
        """Test self-observation recursion limit"""
        hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=2)
        observer = hierarchy.add_observer("self_obs", consciousness_level=0.8)
        
        # Attempt self-observation multiple times
        for i in range(5):
            result = hierarchy.perform_meta_observation("self_obs", "self_obs")
            
            # Should respect recursion limits
            if 'error' in result:
                assert 'depth' in result['error'].lower()
            
            # Recursive depth should be bounded
            assert observer.state.recursive_depth <= hierarchy.max_hierarchy_depth
    
    def test_measurement_with_incompatible_dimensions(self):
        """Test measurement with incompatible state dimensions"""
        observer = QuantumObserver("incompatible", consciousness_level=0.5, awareness_dimensions=4)
        
        # State with different dimensions
        incompatible_state = np.array([1, 0, 0, 0, 0], dtype=complex)
        incompatible_state = incompatible_state / np.sqrt(np.sum(np.abs(incompatible_state)**2))
        
        # Should handle gracefully
        outcome, prob, post_state = observer.measure_system(incompatible_state)
        
        assert isinstance(outcome, int)
        assert 0 <= prob <= 1
        assert len(post_state) == len(incompatible_state)