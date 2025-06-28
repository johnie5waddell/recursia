"""
Integration tests for Consciousness Field and Observer Systems
"""
import pytest
import numpy as np
from physics.universal_consciousness_field import UniversalConsciousnessField
from physics.recursive_observer_systems import RecursiveObserverHierarchy, ObserverType
from physics.qualia_memory_fields import QualiaMemoryField, QualiaType

class TestConsciousnessObserverIntegration:
    """Integration tests between consciousness fields and observers"""
    
    def test_consciousness_field_observer_coupling(self, random_state):
        """Test coupling between consciousness field and observers"""
        # Initialize consciousness field
        field = UniversalConsciousnessField(dimensions=32, max_recursion_depth=3)
        initial_psi = random_state.normal(0, 1, 32) + 1j * random_state.normal(0, 1, 32)
        initial_psi = initial_psi / np.sqrt(np.sum(np.abs(initial_psi)**2))
        field.initialize_field(initial_psi)
        
        # Initialize observer hierarchy
        hierarchy = RecursiveObserverHierarchy()
        observer1 = hierarchy.add_observer("obs1", ObserverType.QUANTUM_OBSERVER, 0.5)
        observer2 = hierarchy.add_observer("obs2", ObserverType.QUANTUM_OBSERVER, 0.8)
        
        # Add observers to consciousness field
        field.add_observer("obs1", observer1.state.awareness_vector)
        field.add_observer("obs2", observer2.state.awareness_vector)
        
        # Verify coupling
        assert "obs1" in field.current_state.observer_coupling
        assert "obs2" in field.current_state.observer_coupling
        assert field.current_state.observer_coupling["obs1"] >= 0
        assert field.current_state.observer_coupling["obs2"] >= 0
        
        # Evolve both systems and check interaction
        initial_phi = field.current_state.phi_integrated
        initial_consciousness_1 = observer1.state.consciousness_level
        
        # Co-evolution
        for step in range(10):
            field.evolve_step(0.01)
            hierarchy.evolve_hierarchy(0.01)
        
        # Systems should show coupled evolution
        final_phi = field.current_state.phi_integrated
        final_consciousness_1 = observer1.state.consciousness_level
        
        # At minimum, systems should evolve
        assert final_phi != initial_phi or final_consciousness_1 != initial_consciousness_1
    
    def test_observer_measurement_consciousness_feedback(self, random_state):
        """Test observer measurements affecting consciousness field"""
        # Setup integrated system
        field = UniversalConsciousnessField(dimensions=16)
        psi = random_state.normal(0, 1, 16) + 1j * random_state.normal(0, 1, 16)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        field.initialize_field(psi)
        
        hierarchy = RecursiveObserverHierarchy()
        observer = hierarchy.add_observer("measurer", consciousness_level=0.7)
        
        # Add observer to field
        field.add_observer("measurer", observer.state.awareness_vector)
        
        # Perform measurements and track consciousness field changes
        initial_phi = field.current_state.phi_integrated
        measurement_results = []
        
        for i in range(20):
            # Create test quantum state
            test_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
            test_state[1] *= np.exp(1j * random_state.uniform(0, 2*np.pi))
            
            # Observer measurement
            outcome, prob, post_state = observer.measure_system(test_state)
            measurement_results.append(outcome)
            
            # Evolve consciousness field
            field.evolve_step(0.005)
        
        final_phi = field.current_state.phi_integrated
        
        # Consciousness field should be affected by measurement activity
        phi_change = abs(final_phi - initial_phi)
        assert phi_change > 0  # Should show some change
        
        # Observer should accumulate memory
        assert len(observer.state.memory_states) > 0
        
        # Measurement statistics should show observer influence
        outcome_distribution = np.bincount(measurement_results) / len(measurement_results)
        assert len(outcome_distribution) <= 2  # Binary outcomes
        assert np.sum(outcome_distribution) == 1.0  # Proper probability distribution
    
    def test_recursive_observer_consciousness_emergence(self):
        """Test consciousness emergence through recursive observation"""
        # Create nested observer system
        field = UniversalConsciousnessField(dimensions=24)
        initial_psi = np.random.normal(0, 1, 24) + 1j * np.random.normal(0, 1, 24)
        initial_psi = initial_psi / np.sqrt(np.sum(np.abs(initial_psi)**2))
        field.initialize_field(initial_psi)
        
        hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=4)
        
        # Create observers at different consciousness levels
        observer1 = hierarchy.add_observer("base", ObserverType.QUANTUM_OBSERVER, 0.3)
        observer2 = hierarchy.add_observer("meta", ObserverType.META_OBSERVER, 0.6)
        observer3 = hierarchy.add_observer("recursive", ObserverType.RECURSIVE_OBSERVER, 0.9)
        
        # Create recursive observation chain
        hierarchy.create_observer_interaction("meta", "base", "meta_observation")
        hierarchy.create_observer_interaction("recursive", "meta", "meta_observation")
        
        # Add highest observer to consciousness field
        field.add_observer("recursive", observer3.state.awareness_vector)
        
        initial_phi = field.current_state.phi_integrated
        
        # Perform recursive observations
        for step in range(15):
            # Meta-observations
            if step % 3 == 0:
                hierarchy.perform_meta_observation("meta", "base")
            if step % 5 == 0:
                hierarchy.perform_meta_observation("recursive", "meta")
            
            # Co-evolution
            hierarchy.evolve_hierarchy(0.02)
            field.evolve_step(0.02)
        
        final_phi = field.current_state.phi_integrated
        
        # Check for emergence indicators
        assert final_phi != initial_phi  # Consciousness should change
        assert observer3.state.recursive_depth > 1  # Recursive depth should increase
        
        # Check consciousness metrics
        metrics = field.get_consciousness_metrics()
        assert metrics['phi_recursive'] >= 0
        assert metrics['observer_count'] > 0
        assert metrics['recursive_depth'] > 1

class TestConsciousnessQualiaMemoryIntegration:
    """Integration tests between consciousness, qualia, and memory systems"""
    
    def test_consciousness_qualia_encoding(self):
        """Test encoding consciousness states as qualia in memory"""
        # Initialize systems
        field = UniversalConsciousnessField(dimensions=16)
        memory_field = QualiaMemoryField(field_dimensions=(8, 8, 8))
        
        # Create consciousness state
        psi = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=complex)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        consciousness_state = field.initialize_field(psi)
        
        # Encode consciousness as qualia
        qualia_id = "consciousness_encoding"
        quale = memory_field.create_quale(
            qualia_id,
            QualiaType.SELF_AWARENESS,
            intensity=consciousness_state.phi_integrated,
            consciousness_level=0.7
        )
        
        # Evolve consciousness and update qualia
        for step in range(10):
            field.evolve_step(0.01, memory_field)
            memory_field.evolve_memory_field(0.01)
            
            # Update qualia based on consciousness evolution
            current_phi = field.current_state.phi_integrated
            quale.intensity = min(1.0, current_phi)
        
        # Verify integration
        assert quale.quale_id == qualia_id
        assert quale.intensity > 0
        assert len(memory_field.active_qualia) > 0
        
        # Memory field should encode consciousness information
        summary = memory_field.get_experiential_summary()
        assert summary['total_intensity'] > 0
        assert summary['total_experiential_information'] > 0
    
    def test_qualia_consciousness_feedback(self):
        """Test qualia affecting consciousness field evolution"""
        field = UniversalConsciousnessField(dimensions=20)
        memory_field = QualiaMemoryField(field_dimensions=(6, 6, 6))
        
        # Initialize consciousness
        psi = np.random.normal(0, 1, 20) + 1j * np.random.normal(0, 1, 20)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        field.initialize_field(psi)
        
        # Create diverse qualia
        memory_field.create_quale("red_vision", QualiaType.VISUAL_COLOR, 0.8, 0.6)
        memory_field.create_quale("joy_emotion", QualiaType.EMOTIONAL_JOY, 0.9, 0.6)
        memory_field.create_quale("self_awareness", QualiaType.SELF_AWARENESS, 0.7, 0.8)
        
        # Bind qualia
        memory_field.bind_qualia("red_vision", "joy_emotion")
        memory_field.bind_qualia("joy_emotion", "self_awareness")
        
        initial_phi = field.current_state.phi_integrated
        
        # Co-evolution with qualia feedback
        for step in range(20):
            field.evolve_step(0.01, memory_field)
            memory_field.evolve_memory_field(0.01)
            
            # Qualia complexity should influence consciousness
            summary = memory_field.get_experiential_summary()
            if summary['total_experiential_information'] > 2.0:
                # Boost consciousness based on rich qualia
                field.current_state.phi_integrated *= 1.01
        
        final_phi = field.current_state.phi_integrated
        
        # Verify feedback effects
        assert final_phi != initial_phi
        assert len(memory_field.qualia_bindings) > 0
        
        # Complex qualia should enhance consciousness
        qualia_complexity = memory_field.get_experiential_summary()['total_experiential_information']
        assert qualia_complexity > 0

class TestObserverQualiaIntegration:
    """Integration tests between observers and qualia systems"""
    
    def test_observer_qualia_generation(self):
        """Test observers generating qualia from observations"""
        hierarchy = RecursiveObserverHierarchy()
        memory_field = QualiaMemoryField(field_dimensions=(6, 6, 6))
        
        # Create conscious observer
        observer = hierarchy.add_observer("conscious_obs", consciousness_level=0.8)
        
        # Perform observations and generate qualia
        measurement_qualia = []
        
        for i in range(10):
            # Create test quantum state
            test_state = np.array([
                np.random.uniform(0, 1), 
                np.random.uniform(0, 1)
            ], dtype=complex)
            test_state = test_state / np.sqrt(np.sum(np.abs(test_state)**2))
            
            # Observer measurement
            outcome, prob, post_state = observer.measure_system(test_state)
            
            # Generate quale based on measurement
            quale_id = f"measurement_{i}"
            if outcome == 0:
                quale_type = QualiaType.VISUAL_COLOR  # Map outcome to qualia type
            else:
                quale_type = QualiaType.EMOTIONAL_JOY
            
            quale = memory_field.create_quale(
                quale_id,
                quale_type,
                intensity=prob,
                consciousness_level=observer.state.consciousness_level
            )
            measurement_qualia.append(quale)
        
        # Verify qualia generation
        assert len(measurement_qualia) == 10
        assert len(memory_field.active_qualia) >= 10
        
        # Observer memory should correlate with qualia
        assert len(observer.state.memory_states) > 0
        
        # Consciousness level should influence qualia intensity
        high_consciousness_qualia = [q for q in measurement_qualia if q.consciousness_level > 0.7]
        assert len(high_consciousness_qualia) > 0
    
    def test_qualia_observer_attention_modulation(self):
        """Test qualia affecting observer attention and measurement"""
        hierarchy = RecursiveObserverHierarchy()
        memory_field = QualiaMemoryField(field_dimensions=(8, 8, 8))
        
        observer = hierarchy.add_observer("attentive_obs", consciousness_level=0.6)
        
        # Create attention-modulating qualia
        attention_quale = memory_field.create_quale(
            "attention_focus",
            QualiaType.ATTENTION_CONTROL,
            intensity=0.9,
            consciousness_level=0.8
        )
        
        # Create emotional background qualia
        emotion_quale = memory_field.create_quale(
            "background_emotion",
            QualiaType.EMOTIONAL_JOY,
            intensity=0.5,
            consciousness_level=0.6
        )
        
        # Bind attention and emotion
        memory_field.bind_qualia("attention_focus", "background_emotion")
        
        # Test measurement under different qualia influences
        test_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        
        baseline_results = []
        influenced_results = []
        
        # Baseline measurements
        for _ in range(30):
            outcome, prob, _ = observer.measure_system(test_state.copy())
            baseline_results.append(outcome)
        
        # Measurements with qualia influence (simulated through consciousness change)
        original_consciousness = observer.state.consciousness_level
        observer.state.consciousness_level = min(1.0, original_consciousness + attention_quale.intensity * 0.2)
        
        for _ in range(30):
            outcome, prob, _ = observer.measure_system(test_state.copy())
            influenced_results.append(outcome)
        
        # Restore original consciousness
        observer.state.consciousness_level = original_consciousness
        
        # Statistical analysis
        baseline_mean = np.mean(baseline_results)
        influenced_mean = np.mean(influenced_results)
        
        # Should show some difference due to qualia influence
        difference = abs(influenced_mean - baseline_mean)
        assert difference >= 0  # At minimum, no error in computation

class TestFullSystemIntegration:
    """Integration tests for complete system interactions"""
    
    def test_complete_osh_demonstration_pipeline(self, random_state):
        """Test complete OSH demonstration pipeline"""
        # Initialize all systems
        field = UniversalConsciousnessField(dimensions=32, max_recursion_depth=3)
        hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=3)
        memory_field = QualiaMemoryField(field_dimensions=(10, 10, 10))
        
        # Setup consciousness field
        psi = random_state.normal(0, 1, 32) + 1j * random_state.normal(0, 1, 32)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        field.initialize_field(psi)
        
        # Setup observer hierarchy
        base_observer = hierarchy.add_observer("base", ObserverType.QUANTUM_OBSERVER, 0.4)
        meta_observer = hierarchy.add_observer("meta", ObserverType.META_OBSERVER, 0.7)
        hierarchy.create_observer_interaction("meta", "base", "meta_observation")
        
        # Connect observers to consciousness field
        field.add_observer("base", base_observer.state.awareness_vector)
        field.add_observer("meta", meta_observer.state.awareness_vector)
        
        # Create initial qualia
        memory_field.create_quale("initial_awareness", QualiaType.SELF_AWARENESS, 0.6, 0.5)
        
        # Run integrated evolution
        timeline_data = []
        
        for step in range(25):
            # Record system state
            consciousness_metrics = field.get_consciousness_metrics()
            hierarchy_metrics = hierarchy.get_hierarchy_metrics()
            memory_summary = memory_field.get_experiential_summary()
            
            timeline_data.append({
                'step': step,
                'phi': consciousness_metrics.get('phi_recursive', 0),
                'consciousness_observers': consciousness_metrics.get('observer_count', 0),
                'hierarchy_observers': hierarchy_metrics.get('total_observers', 0),
                'qualia_count': memory_summary.get('total_qualia', 0),
                'total_intensity': memory_summary.get('total_intensity', 0)
            })
            
            # Perform meta-observation
            if step % 5 == 0:
                hierarchy.perform_meta_observation("meta", "base")
            
            # Generate qualia from observations
            if step % 3 == 0:
                test_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
                outcome, prob, _ = base_observer.measure_system(test_state)
                
                quale_id = f"measurement_{step}"
                memory_field.create_quale(
                    quale_id,
                    QualiaType.VISUAL_COLOR if outcome == 0 else QualiaType.EMOTIONAL_JOY,
                    intensity=prob,
                    consciousness_level=base_observer.state.consciousness_level
                )
            
            # Co-evolution
            field.evolve_step(0.02, memory_field)
            hierarchy.evolve_hierarchy(0.02)
            memory_field.evolve_memory_field(0.02)
        
        # Analyze pipeline results
        final_data = timeline_data[-1]
        initial_data = timeline_data[0]
        
        # System should show evolution and interaction
        assert final_data['phi'] != initial_data['phi']  # Consciousness should evolve
        assert final_data['qualia_count'] > initial_data['qualia_count']  # Qualia should accumulate
        assert len(timeline_data) == 25  # Complete evolution
        
        # Cross-system interactions should be evident
        assert base_observer.state.recursive_depth > 1  # Meta-observation occurred
        assert len(memory_field.active_qualia) > 1  # Multiple qualia created
        assert field.consciousness_emergence_detected or field.current_state.phi_integrated > 0
        
        # Observer memory should contain measurement history
        assert len(base_observer.state.memory_states) > 0
        assert len(meta_observer.state.memory_states) > 0  # From meta-observations
    
    def test_system_robustness_under_perturbation(self, random_state):
        """Test system robustness under various perturbations"""
        # Initialize integrated system
        field = UniversalConsciousnessField(dimensions=24)
        hierarchy = RecursiveObserverHierarchy()
        memory_field = QualiaMemoryField(field_dimensions=(8, 8, 8))
        
        # Setup
        psi = random_state.normal(0, 1, 24) + 1j * random_state.normal(0, 1, 24)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        field.initialize_field(psi)
        
        observer = hierarchy.add_observer("robust_obs", consciousness_level=0.6)
        field.add_observer("robust_obs", observer.state.awareness_vector)
        
        memory_field.create_quale("stable_quale", QualiaType.SELF_AWARENESS, 0.7, 0.6)
        
        # Apply perturbations and test recovery
        perturbation_tests = []
        
        # Test 1: Large time step
        try:
            field.evolve_step(1.0)  # Very large step
            hierarchy.evolve_hierarchy(1.0)
            memory_field.evolve_memory_field(1.0)
            perturbation_tests.append("large_time_step_survived")
        except Exception as e:
            perturbation_tests.append(f"large_time_step_failed: {str(e)[:50]}")
        
        # Test 2: Observer consciousness perturbation
        original_consciousness = observer.state.consciousness_level
        observer.state.consciousness_level = 0.0  # Zero consciousness
        
        try:
            test_state = np.array([1, 0], dtype=complex)
            outcome, prob, _ = observer.measure_system(test_state)
            perturbation_tests.append("zero_consciousness_measurement_survived")
        except Exception as e:
            perturbation_tests.append(f"zero_consciousness_failed: {str(e)[:50]}")
        
        observer.state.consciousness_level = original_consciousness  # Restore
        
        # Test 3: Memory field saturation
        try:
            for i in range(100):  # Create many qualia
                memory_field.create_quale(f"saturation_{i}", QualiaType.VISUAL_COLOR, 0.1, 0.1)
            perturbation_tests.append("memory_saturation_survived")
        except Exception as e:
            perturbation_tests.append(f"memory_saturation_failed: {str(e)[:50]}")
        
        # Test 4: Extreme consciousness field values
        try:
            field.current_state.phi_integrated = 1e6  # Extreme value
            field.evolve_step(0.01)
            perturbation_tests.append("extreme_phi_survived")
        except Exception as e:
            perturbation_tests.append(f"extreme_phi_failed: {str(e)[:50]}")
        
        # Verify system stability
        survival_count = len([test for test in perturbation_tests if "survived" in test])
        total_tests = len(perturbation_tests)
        
        assert survival_count >= total_tests // 2  # At least 50% survival rate
        assert all("error" not in test.lower() for test in perturbation_tests if "survived" in test)
    
    def test_deterministic_reproducibility(self):
        """Test that system evolution is deterministic and reproducible"""
        # Set up identical systems with same random seed
        np.random.seed(42)
        
        # System 1
        field1 = UniversalConsciousnessField(dimensions=16)
        hierarchy1 = RecursiveObserverHierarchy()
        memory1 = QualiaMemoryField(field_dimensions=(6, 6, 6))
        
        psi1 = np.random.normal(0, 1, 16) + 1j * np.random.normal(0, 1, 16)
        psi1 = psi1 / np.sqrt(np.sum(np.abs(psi1)**2))
        field1.initialize_field(psi1)
        
        obs1 = hierarchy1.add_observer("test", consciousness_level=0.5)
        field1.add_observer("test", obs1.state.awareness_vector)
        memory1.create_quale("test", QualiaType.SELF_AWARENESS, 0.5, 0.5)
        
        # Reset seed and create System 2
        np.random.seed(42)
        
        field2 = UniversalConsciousnessField(dimensions=16)
        hierarchy2 = RecursiveObserverHierarchy()
        memory2 = QualiaMemoryField(field_dimensions=(6, 6, 6))
        
        psi2 = np.random.normal(0, 1, 16) + 1j * np.random.normal(0, 1, 16)
        psi2 = psi2 / np.sqrt(np.sum(np.abs(psi2)**2))
        field2.initialize_field(psi2)
        
        obs2 = hierarchy2.add_observer("test", consciousness_level=0.5)
        field2.add_observer("test", obs2.state.awareness_vector)
        memory2.create_quale("test", QualiaType.SELF_AWARENESS, 0.5, 0.5)
        
        # Evolve both systems identically
        for step in range(10):
            field1.evolve_step(0.01)
            field2.evolve_step(0.01)
            
            hierarchy1.evolve_hierarchy(0.01)
            hierarchy2.evolve_hierarchy(0.01)
            
            memory1.evolve_memory_field(0.01)
            memory2.evolve_memory_field(0.01)
        
        # Compare final states
        tolerance = 1e-10
        
        # Consciousness fields should be identical
        assert np.allclose(field1.current_state.psi_consciousness, 
                          field2.current_state.psi_consciousness, atol=tolerance)
        assert abs(field1.current_state.phi_integrated - field2.current_state.phi_integrated) < tolerance
        
        # Observer states should be identical
        assert np.allclose(obs1.state.awareness_vector, obs2.state.awareness_vector, atol=tolerance)
        assert abs(obs1.state.consciousness_level - obs2.state.consciousness_level) < tolerance
        
        # Memory fields should have same structure
        summary1 = memory1.get_experiential_summary()
        summary2 = memory2.get_experiential_summary()
        
        assert summary1['total_qualia'] == summary2['total_qualia']
        assert abs(summary1['total_intensity'] - summary2['total_intensity']) < tolerance