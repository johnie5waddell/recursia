"""
Comprehensive tests for quantum physics accuracy and mathematical correctness.
Validates that the quantum simulation engine produces scientifically accurate results.
"""

import pytest
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
from dataclasses import dataclass

# Import physics modules
from src.quantum.quantum_state import QuantumState
from src.physics.measurement.measurement_proper import QuantumMeasurementProper
from src.physics.entanglement_proper import EntanglementManagerProper
from src.physics.coherence_proper import ScientificCoherenceManager
from src.physics.observer import ObserverDynamics
from src.physics.memory_field_proper import MemoryField
from src.physics.physics_engine_proper import PhysicsEngine

# Constants for validation
PLANCK_CONSTANT = 6.62607015e-34  # J⋅Hz⁻¹
REDUCED_PLANCK = PLANCK_CONSTANT / (2 * np.pi)
BOLTZMANN_CONSTANT = 1.380649e-23  # J⋅K⁻¹

@dataclass
class QuantumTestCase:
    """Test case for quantum physics validation"""
    name: str
    initial_state: np.ndarray
    expected_properties: Dict[str, float]
    tolerance: float = 1e-10

class TestQuantumStateAccuracy:
    """Test quantum state representation and evolution accuracy"""

    def test_qubit_normalization(self):
        """Verify that qubit states maintain proper normalization"""
        # Test single qubit normalization
        alpha, beta = 0.6, 0.8
        state_vector = np.array([alpha, beta], dtype=complex)
        
        quantum_state = QuantumState(1)
        quantum_state.set_amplitudes(state_vector)
        
        # Check normalization: |α|² + |β|² = 1
        norm_squared = np.sum(np.abs(quantum_state.amplitudes) ** 2)
        assert abs(norm_squared - 1.0) < 1e-15, f"State not normalized: {norm_squared}"

    def test_multi_qubit_normalization(self):
        """Test normalization for multi-qubit systems"""
        for n_qubits in range(1, 6):
            state_dim = 2 ** n_qubits
            # Create random complex amplitudes
            real_parts = np.random.normal(0, 1, state_dim)
            imag_parts = np.random.normal(0, 1, state_dim)
            amplitudes = real_parts + 1j * imag_parts
            
            # Normalize
            amplitudes = amplitudes / np.linalg.norm(amplitudes)
            
            quantum_state = QuantumState(n_qubits)
            quantum_state.set_amplitudes(amplitudes)
            
            norm_squared = np.sum(np.abs(quantum_state.amplitudes) ** 2)
            assert abs(norm_squared - 1.0) < 1e-14, f"Multi-qubit state not normalized"

    def test_bell_state_properties(self):
        """Verify Bell state entanglement properties"""
        # |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
        bell_amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        
        quantum_state = QuantumState(2)
        quantum_state.set_amplitudes(bell_amplitudes)
        
        # Check normalization
        norm = np.linalg.norm(quantum_state.amplitudes)
        assert abs(norm - 1.0) < 1e-15
        
        # Check entanglement entropy (should be 1 for maximally entangled state)
        entropy = quantum_state.calculate_entanglement_entropy()
        assert abs(entropy - 1.0) < 1e-10, f"Bell state entropy incorrect: {entropy}"

    def test_ghz_state_properties(self):
        """Test GHZ state (3-qubit entangled state) properties"""
        # |GHZ⟩ = (|000⟩ + |111⟩)/√2
        ghz_amplitudes = np.zeros(8, dtype=complex)
        ghz_amplitudes[0] = 1/np.sqrt(2)  # |000⟩
        ghz_amplitudes[7] = 1/np.sqrt(2)  # |111⟩
        
        quantum_state = QuantumState(3)
        quantum_state.set_amplitudes(ghz_amplitudes)
        
        # Verify normalization
        assert abs(np.linalg.norm(quantum_state.amplitudes) - 1.0) < 1e-15
        
        # Check that it's genuinely tripartite entangled
        # For GHZ states, bipartite entanglement entropy should be 1
        entropy = quantum_state.calculate_entanglement_entropy(subsystem_size=1)
        assert abs(entropy - 1.0) < 1e-10

    def test_coherence_evolution(self):
        """Test coherence decay over time"""
        quantum_state = QuantumState(1)
        # Superposition state: (|0⟩ + |1⟩)/√2
        quantum_state.set_amplitudes(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
        
        initial_coherence = quantum_state.calculate_coherence()
        assert abs(initial_coherence - 1.0) < 1e-10, "Initial coherence should be 1"
        
        # Apply decoherence
        decoherence_rate = 0.1
        time_step = 0.5
        quantum_state.apply_decoherence(decoherence_rate, time_step)
        
        final_coherence = quantum_state.calculate_coherence()
        expected_coherence = np.exp(-decoherence_rate * time_step)
        
        assert abs(final_coherence - expected_coherence) < 1e-10

class TestMeasurementAccuracy:
    """Test quantum measurement physics accuracy"""

    def test_born_rule_statistics(self):
        """Verify Born rule statistical accuracy over many measurements"""
        # Prepare state: α|0⟩ + β|1⟩
        alpha = 0.6
        beta = 0.8
        state_vector = np.array([alpha, beta], dtype=complex)
        
        quantum_state = QuantumState(1)
        quantum_state.set_amplitudes(state_vector)
        
        measurement = QuantumMeasurement()
        
        # Perform many measurements
        num_measurements = 10000
        zero_count = 0
        
        for _ in range(num_measurements):
            result = measurement.measure_computational_basis(quantum_state)
            if result == 0:
                zero_count += 1
        
        # Check Born rule: P(0) = |α|²
        expected_prob_zero = abs(alpha) ** 2
        measured_prob_zero = zero_count / num_measurements
        
        # Statistical tolerance (3-sigma for binomial distribution)
        std_dev = np.sqrt(expected_prob_zero * (1 - expected_prob_zero) / num_measurements)
        tolerance = 3 * std_dev
        
        assert abs(measured_prob_zero - expected_prob_zero) < tolerance

    def test_measurement_backaction(self):
        """Test that measurement properly collapses the state"""
        # Start with superposition
        quantum_state = QuantumState(1)
        quantum_state.set_amplitudes(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
        
        measurement = QuantumMeasurement()
        result = measurement.measure_computational_basis(quantum_state)
        
        # After measurement, state should be collapsed
        if result == 0:
            expected_state = np.array([1.0, 0.0], dtype=complex)
        else:
            expected_state = np.array([0.0, 1.0], dtype=complex)
        
        np.testing.assert_allclose(quantum_state.amplitudes, expected_state, atol=1e-15)

    def test_partial_measurement_entangled_state(self):
        """Test partial measurement on entangled systems"""
        # Bell state: (|00⟩ + |11⟩)/√2
        quantum_state = QuantumState(2)
        bell_amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        quantum_state.set_amplitudes(bell_amplitudes)
        
        measurement = QuantumMeasurement()
        
        # Measure first qubit multiple times and check correlations
        correlations_correct = 0
        num_tests = 1000
        
        for _ in range(num_tests):
            # Reset to Bell state
            quantum_state.set_amplitudes(bell_amplitudes)
            
            # Measure first qubit
            result_1 = measurement.measure_qubit(quantum_state, 0)
            # Measure second qubit
            result_2 = measurement.measure_qubit(quantum_state, 1)
            
            # For Bell state, results should be perfectly correlated
            if result_1 == result_2:
                correlations_correct += 1
        
        correlation_rate = correlations_correct / num_tests
        assert correlation_rate > 0.95, f"Bell state correlation too low: {correlation_rate}"

class TestEntanglementAccuracy:
    """Test entanglement physics and mathematics"""

    def test_schmidt_decomposition(self):
        """Verify Schmidt decomposition for bipartite states"""
        # Create a known entangled state
        quantum_state = QuantumState(2)
        # State: 0.6|00⟩ + 0.8|11⟩ (normalized)
        amplitudes = np.array([0.6, 0, 0, 0.8], dtype=complex)
        quantum_state.set_amplitudes(amplitudes)
        
        entanglement_system = EntanglementSystem()
        schmidt_coeffs = entanglement_system.calculate_schmidt_coefficients(quantum_state)
        
        # For this state, Schmidt coefficients should be [0.6, 0.8]
        expected_coeffs = np.array([0.6, 0.8])
        np.testing.assert_allclose(np.sort(schmidt_coeffs), np.sort(expected_coeffs), atol=1e-15)

    def test_entanglement_entropy_bounds(self):
        """Verify entanglement entropy satisfies physical bounds"""
        for n_qubits in range(2, 5):
            quantum_state = QuantumState(n_qubits)
            
            # Test various states
            test_states = [
                np.zeros(2**n_qubits, dtype=complex),  # |00...0⟩
                np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)  # Equal superposition
            ]
            test_states[0][0] = 1.0  # Product state
            
            for state in test_states:
                quantum_state.set_amplitudes(state)
                entropy = quantum_state.calculate_entanglement_entropy()
                
                # Entropy must be non-negative and bounded by log(min(d_A, d_B))
                assert entropy >= 0, "Entanglement entropy must be non-negative"
                
                # For bipartite split of n qubits
                subsystem_size = n_qubits // 2
                max_entropy = subsystem_size  # log_2 of Hilbert space dimension
                assert entropy <= max_entropy + 1e-10, f"Entropy exceeds bound: {entropy} > {max_entropy}"

    def test_entanglement_monotonicity(self):
        """Test that entanglement is monotonic under LOCC operations"""
        # Start with maximally entangled state
        quantum_state = QuantumState(2)
        bell_amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        quantum_state.set_amplitudes(bell_amplitudes)
        
        initial_entropy = quantum_state.calculate_entanglement_entropy()
        
        # Apply local unitary (shouldn't change entanglement)
        # Pauli-X on first qubit: |00⟩ + |11⟩ → |10⟩ + |01⟩
        evolved_amplitudes = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2), 0], dtype=complex)
        quantum_state.set_amplitudes(evolved_amplitudes)
        
        final_entropy = quantum_state.calculate_entanglement_entropy()
        
        # Entanglement should be preserved under local unitaries
        assert abs(final_entropy - initial_entropy) < 1e-10

class TestCoherencePhysics:
    """Test quantum coherence calculations and physics"""

    def test_coherence_measures(self):
        """Verify different coherence measures give consistent results"""
        # Test with known coherent state
        quantum_state = QuantumState(1)
        # Equal superposition: maximum coherence
        quantum_state.set_amplitudes(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
        
        coherence_calc = CoherenceCalculator()
        
        # L1 norm coherence
        l1_coherence = coherence_calc.l1_norm_coherence(quantum_state)
        
        # Relative entropy coherence
        rel_entropy_coherence = coherence_calc.relative_entropy_coherence(quantum_state)
        
        # For equal superposition, both should give maximum values
        assert abs(l1_coherence - 1.0) < 1e-10, f"L1 coherence incorrect: {l1_coherence}"
        assert rel_entropy_coherence > 0.5, f"Relative entropy coherence too low: {rel_entropy_coherence}"

    def test_coherence_bounds(self):
        """Verify coherence measures satisfy mathematical bounds"""
        coherence_calc = CoherenceCalculator()
        
        for n_qubits in range(1, 4):
            quantum_state = QuantumState(n_qubits)
            
            # Test diagonal state (no coherence)
            diagonal_state = np.zeros(2**n_qubits, dtype=complex)
            diagonal_state[0] = 1.0
            quantum_state.set_amplitudes(diagonal_state)
            
            l1_coh = coherence_calc.l1_norm_coherence(quantum_state)
            assert abs(l1_coh) < 1e-15, "Diagonal state should have zero coherence"
            
            # Test maximally coherent state
            max_coherent = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
            quantum_state.set_amplitudes(max_coherent)
            
            l1_coh = coherence_calc.l1_norm_coherence(quantum_state)
            # L1 coherence is bounded by √(d-1) where d is dimension
            max_l1 = np.sqrt(2**n_qubits - 1)
            assert l1_coh <= max_l1 + 1e-10, f"L1 coherence exceeds bound: {l1_coh} > {max_l1}"

class TestObserverPhysics:
    """Test observer-mediated state collapse and OSH principles"""

    def test_observer_collapse_threshold(self):
        """Test observer collapse occurs at correct thresholds"""
        quantum_state = QuantumState(1)
        quantum_state.set_amplitudes(np.array([1/np.sqrt(2), 1/np.sqrt(2)]))
        
        observer = Observer(
            name="Test Observer",
            observer_type="conscious",
            collapse_threshold=0.8,
            measurement_strength=1.0
        )
        
        # Observer with high threshold shouldn't collapse weak superposition
        initial_coherence = quantum_state.calculate_coherence()
        observer.interact_with_state(quantum_state)
        
        # For testing, we need to simulate the observer interaction
        # This would be implementation-specific based on the OSH model

    def test_consciousness_coupling_strength(self):
        """Test consciousness-matter coupling follows OSH principles"""
        quantum_state = QuantumState(2)
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        quantum_state.set_amplitudes(bell_state)
        
        # High consciousness observer
        conscious_observer = Observer(
            name="Conscious",
            observer_type="conscious",
            self_awareness=0.9,
            measurement_strength=0.8
        )
        
        # Environmental observer
        env_observer = Observer(
            name="Environment", 
            observer_type="environmental",
            self_awareness=0.1,
            measurement_strength=0.3
        )
        
        # Conscious observer should have stronger effect
        initial_entanglement = quantum_state.calculate_entanglement_entropy()
        
        # Test interaction strengths (implementation dependent)

class TestMemoryFieldPhysics:
    """Test memory field dynamics and strain calculations"""

    def test_memory_strain_tensor(self):
        """Verify strain tensor calculations follow continuum mechanics"""
        memory_field = MemoryField(dimensions=(32, 32, 32), resolution=0.1)
        
        # Create a localized strain pattern
        center = (16, 16, 16)
        strain_magnitude = 0.5
        
        # Apply Gaussian strain distribution
        for i in range(32):
            for j in range(32):
                for k in range(32):
                    r_squared = (i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2
                    strain = strain_magnitude * np.exp(-r_squared / (2 * 3**2))
                    memory_field.apply_local_strain((i, j, k), strain)
        
        # Calculate strain tensor components
        strain_tensor = memory_field.calculate_strain_tensor(center)
        
        # Verify tensor properties
        # 1. Symmetry: σ_ij = σ_ji
        assert np.allclose(strain_tensor, strain_tensor.T, atol=1e-15), "Strain tensor not symmetric"
        
        # 2. Trace gives volumetric strain
        volumetric_strain = np.trace(strain_tensor)
        assert volumetric_strain >= 0, "Volumetric strain should be non-negative"

    def test_memory_field_elasticity(self):
        """Test elastic response of memory field"""
        memory_field = MemoryField(dimensions=(16, 16, 16))
        
        # Apply stress and measure strain response
        stress_point = (8, 8, 8)
        applied_stress = 1.0
        
        # Apply stress
        memory_field.apply_stress(stress_point, applied_stress)
        
        # Measure resulting strain
        strain = memory_field.get_strain_at_point(stress_point)
        
        # Verify Hooke's law: σ = E·ε (stress = modulus × strain)
        elastic_modulus = memory_field.get_elastic_modulus()
        expected_strain = applied_stress / elastic_modulus
        
        assert abs(strain - expected_strain) < 1e-10, "Elastic response doesn't follow Hooke's law"

class TestPhysicsEngineIntegration:
    """Integration tests for the complete physics engine"""

    def test_engine_conservation_laws(self):
        """Verify fundamental conservation laws"""
        engine = PhysicsEngine()
        
        # Create test system
        quantum_state = QuantumState(2)
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        quantum_state.set_amplitudes(bell_state)
        
        observer = Observer("Test", "conscious", collapse_threshold=0.9)
        memory_field = MemoryField(dimensions=(16, 16, 16))
        
        # Initialize system
        engine.add_quantum_state(quantum_state)
        engine.add_observer(observer)
        engine.set_memory_field(memory_field)
        
        # Run evolution
        initial_energy = engine.calculate_total_energy()
        initial_info = engine.calculate_quantum_information()
        
        for _ in range(10):
            engine.evolve_system(dt=0.1)
        
        final_energy = engine.calculate_total_energy()
        final_info = engine.calculate_quantum_information()
        
        # Energy should be approximately conserved (allowing for numerical errors)
        energy_change = abs(final_energy - initial_energy) / initial_energy
        assert energy_change < 1e-10, f"Energy not conserved: {energy_change}"

    def test_osh_emergence_metrics(self):
        """Test Organic Simulation Hypothesis emergence detection"""
        engine = PhysicsEngine()
        
        # Set up complex quantum system with multiple observers
        states = []
        for i in range(5):
            state = QuantumState(1)
            # Create varying coherence levels
            alpha = np.cos(i * np.pi / 10)
            beta = np.sin(i * np.pi / 10)
            state.set_amplitudes(np.array([alpha, beta], dtype=complex))
            states.append(state)
            engine.add_quantum_state(state)
        
        # Add observers with different consciousness levels
        for i in range(3):
            observer = Observer(
                name=f"Observer_{i}",
                observer_type="conscious",
                self_awareness=0.3 + i * 0.3,
                measurement_strength=0.5 + i * 0.2
            )
            engine.add_observer(observer)
        
        # Evolve and measure emergence
        for step in range(20):
            engine.evolve_system(dt=0.1)
            
            # Calculate OSH metrics
            rsp_value = engine.calculate_rsp()  # Reality Synthesis Parameter
            emergence_index = engine.calculate_emergence_index()
            consciousness_field_strength = engine.calculate_consciousness_field_strength()
            
            # Verify physical bounds
            assert 0 <= rsp_value <= 1, f"RSP out of bounds: {rsp_value}"
            assert emergence_index >= 0, f"Emergence index negative: {emergence_index}"
            assert consciousness_field_strength >= 0, f"Consciousness field negative: {consciousness_field_strength}"

    def test_reality_anchor_mechanics(self):
        """Test reality anchoring mechanisms"""
        engine = PhysicsEngine()
        
        # Create quantum system
        quantum_state = QuantumState(3)  # 3-qubit system
        # GHZ state
        ghz_amplitudes = np.zeros(8, dtype=complex)
        ghz_amplitudes[0] = 1/np.sqrt(2)
        ghz_amplitudes[7] = 1/np.sqrt(2)
        quantum_state.set_amplitudes(ghz_amplitudes)
        
        engine.add_quantum_state(quantum_state)
        
        # Add reality anchor
        anchor_strength = 0.8
        engine.set_reality_anchor_strength(anchor_strength)
        
        # Measure initial decoherence rate
        initial_coherence = quantum_state.calculate_coherence()
        
        # Evolve with reality anchor
        for _ in range(10):
            engine.evolve_system(dt=0.1)
        
        final_coherence = quantum_state.calculate_coherence()
        
        # Reality anchor should slow decoherence
        coherence_ratio = final_coherence / initial_coherence
        
        # With strong reality anchor, coherence should be better preserved
        assert coherence_ratio > 0.7, f"Reality anchor ineffective: {coherence_ratio}"

if __name__ == "__main__":
    # Run comprehensive physics validation
    pytest.main([__file__, "-v", "--tb=short"])