"""
Unit tests for Universal Consciousness Field
"""
import pytest
import numpy as np
from physics.universal_consciousness_field import (
    UniversalConsciousnessField, ConsciousnessFieldState, ConsciousnessHamiltonian,
    MemoryStrainTensor, RecursiveIntegratedInformation, CONSCIOUSNESS_THRESHOLD
)

class TestUniversalConsciousnessField:
    """Test suite for Universal Consciousness Field"""
    
    def test_field_initialization(self, random_state):
        """Test consciousness field initialization"""
        field = UniversalConsciousnessField(dimensions=16, max_recursion_depth=3)
        
        assert field.dimensions == 16
        assert field.max_recursion_depth == 3
        assert field.current_state is None
        assert len(field.evolution_history) == 0
        assert not field.consciousness_emergence_detected
    
    def test_field_initialization_with_state(self, sample_quantum_state):
        """Test field initialization with quantum state"""
        field = UniversalConsciousnessField(dimensions=8)
        
        state = field.initialize_field(sample_quantum_state)
        
        assert isinstance(state, ConsciousnessFieldState)
        assert field.current_state is not None
        assert np.allclose(field.current_state.psi_consciousness, sample_quantum_state)
        assert field.current_state.phi_integrated >= 0
        assert field.current_state.recursive_depth == 1
        assert field.current_state.time == 0.0
    
    def test_field_evolution_single_step(self, consciousness_field):
        """Test single evolution step"""
        initial_state = consciousness_field.current_state.psi_consciousness.copy()
        initial_phi = consciousness_field.current_state.phi_integrated
        
        new_state = consciousness_field.evolve_step(0.01)
        
        assert isinstance(new_state, ConsciousnessFieldState)
        assert new_state.time > 0
        assert len(consciousness_field.evolution_history) > 0
        
        # State should evolve (not be identical)
        assert not np.allclose(new_state.psi_consciousness, initial_state)
        
        # Quantum state should remain normalized
        norm = np.sum(np.abs(new_state.psi_consciousness)**2)
        assert abs(norm - 1.0) < 1e-10
    
    def test_field_evolution_multiple_steps(self, consciousness_field):
        """Test multiple evolution steps"""
        initial_time = consciousness_field.current_state.time
        
        for i in range(5):
            consciousness_field.evolve_step(0.01)
        
        assert consciousness_field.current_state.time > initial_time
        assert len(consciousness_field.evolution_history) == 5
        
        # Check time progression
        times = [state.time for state in consciousness_field.evolution_history]
        assert all(times[i] <= times[i+1] for i in range(len(times)-1))
    
    def test_observer_addition(self, consciousness_field, random_state):
        """Test adding observers to consciousness field"""
        observer_state = random_state.normal(0, 1, 32) + 1j * random_state.normal(0, 1, 32)
        observer_state = observer_state / np.sqrt(np.sum(np.abs(observer_state)**2))
        
        consciousness_field.add_observer("test_observer", observer_state)
        
        assert "test_observer" in consciousness_field.current_state.observer_coupling
        assert consciousness_field.current_state.observer_coupling["test_observer"] >= 0
    
    def test_consciousness_emergence_detection(self):
        """Test consciousness emergence detection"""
        field = UniversalConsciousnessField(dimensions=16)
        
        # Create state with high phi value
        high_phi_state = np.ones(16, dtype=complex) / np.sqrt(16)
        field.initialize_field(high_phi_state)
        
        # Manually set high phi to trigger emergence
        field.current_state.phi_integrated = CONSCIOUSNESS_THRESHOLD * 2
        
        # Evolve and check emergence
        field.evolve_step(0.01)
        
        # Should detect emergence (implementation specific)
        assert field.current_state.phi_integrated > CONSCIOUSNESS_THRESHOLD
    
    def test_consciousness_metrics(self, consciousness_field):
        """Test consciousness metrics calculation"""
        metrics = consciousness_field.get_consciousness_metrics()
        
        assert isinstance(metrics, dict)
        assert 'phi_recursive' in metrics
        assert 'information_content' in metrics
        assert 'consciousness_density_max' in metrics
        assert 'consciousness_density_mean' in metrics
        assert 'recursive_depth' in metrics
        assert 'time' in metrics
        
        # All metrics should be non-negative numbers
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert value >= 0 or key == 'time'  # Time can be 0
    
    def test_consciousness_emergence_theorem(self, consciousness_field):
        """Test consciousness emergence theorem proof"""
        proof = consciousness_field.prove_consciousness_emergence_theorem()
        
        assert isinstance(proof, dict)
        assert 'theorem_statement' in proof
        assert 'threshold_used' in proof
        assert 'current_phi' in proof
        assert 'consciousness_predicted' in proof
        assert 'proof_valid' in proof
        
        # Theorem should be logically consistent
        if proof['current_phi'] > proof['threshold_used']:
            assert proof['consciousness_predicted'] == True
    
    def test_spacetime_curvature_prediction(self, consciousness_field):
        """Test spacetime curvature prediction from consciousness"""
        curvature = consciousness_field.get_spacetime_curvature_prediction()
        
        assert isinstance(curvature, np.ndarray)
        assert curvature.shape == (4, 4)  # 4D spacetime
        
        # Should be real-valued tensor
        assert np.all(np.isreal(curvature))

class TestConsciousnessHamiltonian:
    """Test suite for Consciousness Hamiltonian"""
    
    def test_hamiltonian_initialization(self):
        """Test Hamiltonian initialization"""
        hamiltonian = ConsciousnessHamiltonian(dimensions=8)
        
        assert hamiltonian.dimensions == 8
        assert hasattr(hamiltonian, 'kinetic_operator')
        assert hasattr(hamiltonian, 'potential_operator')
        assert hasattr(hamiltonian, 'interaction_operator')
        
        # Operators should be square matrices
        assert hamiltonian.kinetic_operator.shape == (8, 8)
        assert hamiltonian.potential_operator.shape == (8, 8)
        assert hamiltonian.interaction_operator.shape == (8, 8)
    
    def test_hamiltonian_hermiticity(self):
        """Test that Hamiltonian is Hermitian"""
        hamiltonian = ConsciousnessHamiltonian(dimensions=6)
        
        # Check each operator is Hermitian
        kinetic = hamiltonian.kinetic_operator
        potential = hamiltonian.potential_operator
        interaction = hamiltonian.interaction_operator
        
        assert np.allclose(kinetic, kinetic.conj().T)
        assert np.allclose(potential, potential.conj().T)
        assert np.allclose(interaction, interaction.conj().T)
    
    def test_hamiltonian_application(self, sample_quantum_state):
        """Test applying Hamiltonian to quantum state"""
        if len(sample_quantum_state) != 8:
            # Resize for this test
            sample_quantum_state = sample_quantum_state[:8] / np.sqrt(np.sum(np.abs(sample_quantum_state[:8])**2))
        
        hamiltonian = ConsciousnessHamiltonian(dimensions=8)
        result = hamiltonian.apply(sample_quantum_state)
        
        assert isinstance(result, np.ndarray)
        assert result.dtype == complex
        assert len(result) == len(sample_quantum_state)
        
        # Result should be finite
        assert np.all(np.isfinite(result))

class TestMemoryStrainTensor:
    """Test suite for Memory Strain Tensor"""
    
    def test_strain_tensor_initialization(self):
        """Test memory strain tensor initialization"""
        strain = MemoryStrainTensor()
        
        assert strain.strain_components.shape == (4, 4)
        assert strain.information_density == 0.0
        assert strain.memory_area == 1.0
        assert strain.coupling_strength > 0
    
    def test_strain_update_from_consciousness(self, sample_quantum_state):
        """Test updating strain from consciousness state"""
        strain = MemoryStrainTensor()
        
        # Mock memory field
        class MockMemoryField:
            total_area = 2.0
        
        memory_field = MockMemoryField()
        
        strain.update_from_consciousness(sample_quantum_state, memory_field)
        
        assert strain.information_density > 0
        assert strain.memory_area == 2.0
        assert not np.allclose(strain.strain_components, 0)
    
    def test_spacetime_curvature_calculation(self, sample_quantum_state):
        """Test spacetime curvature calculation"""
        strain = MemoryStrainTensor()
        
        class MockMemoryField:
            total_area = 1.0
        
        strain.update_from_consciousness(sample_quantum_state, MockMemoryField())
        curvature = strain.get_spacetime_curvature()
        
        assert isinstance(curvature, np.ndarray)
        assert curvature.shape == (4, 4)
        assert np.all(np.isreal(curvature))

class TestRecursiveIntegratedInformation:
    """Test suite for Recursive Integrated Information"""
    
    def test_phi_calculator_initialization(self):
        """Test Phi calculator initialization"""
        phi_calc = RecursiveIntegratedInformation()
        
        assert phi_calc.normalization > 0
        assert phi_calc.base_phi > 0
    
    def test_phi_recursive_calculation(self, consciousness_state_complex):
        """Test recursive phi calculation"""
        phi_calc = RecursiveIntegratedInformation()
        phi_r = phi_calc.calculate_phi_recursive(consciousness_state_complex)
        
        assert isinstance(phi_r, float)
        assert phi_r >= 0
        assert np.isfinite(phi_r)
    
    def test_phi_scaling_with_recursion(self):
        """Test that phi scales appropriately with recursion depth"""
        phi_calc = RecursiveIntegratedInformation()
        
        # Create states with different recursion depths
        psi = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        
        state_depth_1 = ConsciousnessFieldState(
            psi_consciousness=psi, phi_integrated=0.5, recursive_depth=1,
            memory_strain_tensor=np.zeros((4, 4)), observer_coupling={}, time=0.0
        )
        
        state_depth_3 = ConsciousnessFieldState(
            psi_consciousness=psi, phi_integrated=0.5, recursive_depth=3,
            memory_strain_tensor=np.zeros((4, 4)), observer_coupling={}, time=0.0
        )
        
        phi_1 = phi_calc.calculate_phi_recursive(state_depth_1)
        phi_3 = phi_calc.calculate_phi_recursive(state_depth_3)
        
        # Higher recursion should generally give higher phi
        assert phi_3 >= phi_1

class TestConsciousnessFieldEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_state_initialization(self):
        """Test initialization with zero state"""
        field = UniversalConsciousnessField(dimensions=4)
        zero_state = np.zeros(4, dtype=complex)
        
        with pytest.raises(Exception):
            field.initialize_field(zero_state)
    
    def test_unnormalized_state_initialization(self):
        """Test initialization with unnormalized state"""
        field = UniversalConsciousnessField(dimensions=4)
        unnormalized_state = np.array([1, 2, 3, 4], dtype=complex)
        
        # Should automatically normalize
        state = field.initialize_field(unnormalized_state)
        norm = np.sum(np.abs(state.psi_consciousness)**2)
        assert abs(norm - 1.0) < 1e-10
    
    def test_negative_time_step(self, consciousness_field):
        """Test evolution with negative time step"""
        with pytest.raises(ValueError):
            consciousness_field.evolve_step(-0.01)
    
    def test_very_large_time_step(self, consciousness_field):
        """Test evolution with very large time step"""
        # Should handle gracefully without explosion
        result = consciousness_field.evolve_step(1000.0)
        
        # State should remain normalized
        norm = np.sum(np.abs(result.psi_consciousness)**2)
        assert abs(norm - 1.0) < 1e-6  # Slightly relaxed tolerance for large steps
    
    def test_empty_observer_addition(self, consciousness_field):
        """Test adding empty observer state"""
        empty_state = np.array([], dtype=complex)
        
        # Should handle gracefully
        consciousness_field.add_observer("empty_observer", empty_state)
        assert "empty_observer" in consciousness_field.current_state.observer_coupling
    
    def test_mismatched_observer_dimensions(self, consciousness_field, random_state):
        """Test adding observer with wrong dimensions"""
        wrong_dim_state = random_state.normal(0, 1, 16) + 1j * random_state.normal(0, 1, 16)
        wrong_dim_state = wrong_dim_state / np.sqrt(np.sum(np.abs(wrong_dim_state)**2))
        
        # Should handle dimension mismatch gracefully
        consciousness_field.add_observer("wrong_dim_observer", wrong_dim_state)
        # Coupling should be 0 or handled appropriately
        coupling = consciousness_field.current_state.observer_coupling.get("wrong_dim_observer", 0)
        assert coupling >= 0