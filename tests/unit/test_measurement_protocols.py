"""
Unit tests for measurement protocols.
Tests quantum measurement, statistical analysis, and measurement backaction.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.physics.measurement.measurement_protocols import (
    MeasurementProtocol,
    ProjectiveMeasurement,
    WeakMeasurement,
    POVMeasurement,
    ContinuousMeasurement,
    MeasurementBackaction,
    MeasurementStatistics
)
from src.physics.measurement.measurement_utils import (
    construct_measurement_operators,
    calculate_measurement_probabilities,
    post_measurement_state
)
from src.quantum.quantum_state import QuantumState
from src.physics.observer import QuantumObserver


class TestProjectiveMeasurement:
    """Test projective (von Neumann) measurements."""
    
    def test_computational_basis_measurement(self):
        """Test measurement in computational basis."""
        proj = ProjectiveMeasurement(basis="computational")
        
        # Test on |+> state
        plus_state = QuantumState(n_qubits=1)
        plus_state.initialize_plus()
        
        outcome, prob, post_state = proj.measure(plus_state)
        
        # Should get 0 or 1 with equal probability
        assert outcome in [0, 1]
        assert abs(prob - 0.5) < 0.1  # Close to 0.5
        
        # Post-measurement state should be |0> or |1>
        if outcome == 0:
            assert abs(post_state.amplitudes[0] - 1.0) < 1e-10
            assert abs(post_state.amplitudes[1]) < 1e-10
        else:
            assert abs(post_state.amplitudes[0]) < 1e-10
            assert abs(post_state.amplitudes[1] - 1.0) < 1e-10
            
    def test_pauli_basis_measurements(self):
        """Test measurements in Pauli X, Y, Z bases."""
        # X basis measurement on |0>
        x_meas = ProjectiveMeasurement(basis="X")
        zero_state = QuantumState(n_qubits=1)
        zero_state.initialize_zero()
        
        outcomes = []
        for _ in range(100):
            outcome, _, _ = x_meas.measure(zero_state.copy())
            outcomes.append(outcome)
            
        # Should get roughly equal +1 and -1
        plus_count = sum(1 for o in outcomes if o == 1)
        assert 40 < plus_count < 60
        
        # Y basis measurement on |+>
        y_meas = ProjectiveMeasurement(basis="Y")
        plus_state = QuantumState(n_qubits=1)
        plus_state.initialize_plus()
        
        outcome, prob, post = y_meas.measure(plus_state)
        assert outcome in [-1, 1]
        
        # Z basis measurement on |1>
        z_meas = ProjectiveMeasurement(basis="Z")
        one_state = QuantumState(n_qubits=1)
        one_state.initialize_one()
        
        outcome, prob, post = z_meas.measure(one_state)
        assert outcome == -1  # |1> has eigenvalue -1 for Z
        assert abs(prob - 1.0) < 1e-10
        
    def test_custom_basis_measurement(self):
        """Test measurement in custom basis."""
        # Define custom basis (45-degree rotation)
        theta = np.pi / 4
        basis_vectors = [
            np.array([np.cos(theta/2), np.sin(theta/2)]),
            np.array([np.sin(theta/2), -np.cos(theta/2)])
        ]
        
        custom_meas = ProjectiveMeasurement(
            basis="custom",
            basis_vectors=basis_vectors
        )
        
        # Measure state aligned with first basis vector
        state = QuantumState(n_qubits=1)
        state.amplitudes = basis_vectors[0].copy()
        
        outcome, prob, post = custom_meas.measure(state)
        assert outcome == 0  # Should project to first basis state
        assert abs(prob - 1.0) < 1e-10
        
    def test_multi_qubit_measurement(self):
        """Test measurement of multi-qubit systems."""
        proj = ProjectiveMeasurement(basis="computational")
        
        # Create Bell state |Φ+>
        bell = QuantumState(n_qubits=2)
        bell.amplitudes[0] = 1/np.sqrt(2)
        bell.amplitudes[3] = 1/np.sqrt(2)
        
        outcome, prob, post = proj.measure(bell, qubits=[0])
        
        # Measuring first qubit should give 0 or 1
        assert outcome in [0, 1]
        assert abs(prob - 0.5) < 0.1
        
        # Should collapse to |00> or |11>
        if outcome == 0:
            assert abs(post.amplitudes[0] - 1.0) < 1e-10
        else:
            assert abs(post.amplitudes[3] - 1.0) < 1e-10
            
    def test_measurement_statistics(self):
        """Test statistical properties of measurements."""
        proj = ProjectiveMeasurement(basis="computational")
        
        # Prepare biased state
        state = QuantumState(n_qubits=1)
        state.amplitudes[0] = np.sqrt(0.8)
        state.amplitudes[1] = np.sqrt(0.2)
        
        # Collect statistics
        outcomes = []
        for _ in range(1000):
            outcome, _, _ = proj.measure(state.copy())
            outcomes.append(outcome)
            
        # Check bias
        zero_count = outcomes.count(0)
        zero_prob = zero_count / 1000
        
        assert abs(zero_prob - 0.8) < 0.05  # Within statistical error


class TestWeakMeasurement:
    """Test weak measurement protocols."""
    
    def test_weak_value_calculation(self):
        """Test weak value computation."""
        weak = WeakMeasurement(coupling_strength=0.1)
        
        # Pre-selected state |ψi> = |+>
        pre_state = QuantumState(n_qubits=1)
        pre_state.initialize_plus()
        
        # Post-selected state |ψf> = |0>
        post_state = QuantumState(n_qubits=1)
        post_state.initialize_zero()
        
        # Measure σx
        weak_value = weak.calculate_weak_value(
            pre_state, post_state,
            observable="X"
        )
        
        # Weak value of σx between |+> and |0> is 1
        assert abs(weak_value - 1.0) < 1e-10
        
    def test_anomalous_weak_values(self):
        """Test anomalous (outside eigenvalue range) weak values."""
        weak = WeakMeasurement(coupling_strength=0.01)
        
        # Create states that give anomalous weak value
        pre_state = QuantumState(n_qubits=1)
        pre_state.amplitudes = np.array([1, 1]) / np.sqrt(2)
        
        post_state = QuantumState(n_qubits=1)
        post_state.amplitudes = np.array([1, 0.1]) / np.sqrt(1.01)
        
        # Calculate weak value
        weak_value = weak.calculate_weak_value(
            pre_state, post_state,
            observable="Z"
        )
        
        # Should be outside [-1, 1] range
        assert abs(weak_value) > 1.0
        
    def test_weak_measurement_disturbance(self):
        """Test minimal disturbance property of weak measurements."""
        weak = WeakMeasurement(coupling_strength=0.01)
        strong = ProjectiveMeasurement(basis="Z")
        
        # Initial state
        state = QuantumState(n_qubits=1)
        state.initialize_plus()
        
        # Weak measurement
        weak_state = state.copy()
        weak_outcome, weak_post = weak.measure(weak_state, observable="Z")
        
        # Strong measurement
        strong_state = state.copy()
        strong_outcome, _, strong_post = strong.measure(strong_state)
        
        # Weak measurement should disturb less
        weak_fidelity = np.abs(np.vdot(state.amplitudes, weak_post.amplitudes))**2
        strong_fidelity = np.abs(np.vdot(state.amplitudes, strong_post.amplitudes))**2
        
        assert weak_fidelity > strong_fidelity
        assert weak_fidelity > 0.98  # Very little disturbance
        
    def test_weak_measurement_amplification(self):
        """Test weak measurement amplification technique."""
        weak = WeakMeasurement(
            coupling_strength=0.1,
            amplification_factor=100
        )
        
        # Small parameter to measure
        epsilon = 0.01
        state = QuantumState(n_qubits=1)
        state.amplitudes = np.array([np.cos(epsilon), np.sin(epsilon)])
        
        # Amplified measurement
        result = weak.amplified_measurement(
            state,
            observable="Y",
            post_selection_angle=np.pi/2 - epsilon
        )
        
        # Should amplify small rotation
        assert abs(result) > epsilon * 10


class TestPOVMeasurement:
    """Test Positive Operator-Valued Measure (POVM) measurements."""
    
    def test_povm_completeness(self):
        """Test POVM completeness relation."""
        # Define a simple POVM
        povm_elements = [
            0.5 * np.array([[1, 0], [0, 0]]),  # E1
            0.5 * np.array([[0, 0], [0, 1]]),  # E2
            0.5 * np.array([[1, 0], [0, 1]])   # E3
        ]
        
        povm = POVMeasurement(povm_elements)
        
        # Check completeness: sum(Ei) = I
        sum_povm = sum(povm_elements)
        identity = np.eye(2)
        assert np.allclose(sum_povm, identity)
        
    def test_symmetric_informationally_complete_povm(self):
        """Test SIC-POVM (symmetric informationally complete)."""
        povm = POVMeasurement.create_sic_povm(dimension=2)
        
        # Should have d^2 = 4 elements for d=2
        assert len(povm.elements) == 4
        
        # Check properties
        d = 2
        for i, Ei in enumerate(povm.elements):
            # Trace should be 1/d
            assert abs(np.trace(Ei) - 1/d) < 1e-10
            
            # Check symmetry
            for j, Ej in enumerate(povm.elements):
                if i != j:
                    overlap = np.trace(Ei @ Ej)
                    expected = 1/(d**2 * (d+1))
                    assert abs(overlap - expected) < 1e-10
                    
    def test_povm_measurement_outcomes(self):
        """Test POVM measurement statistics."""
        # Three-outcome POVM
        sqrt2 = np.sqrt(2)
        povm_elements = [
            np.array([[1, 0], [0, 0]]) / sqrt2,  # Mostly |0>
            np.array([[0, 0], [0, 1]]) / sqrt2,  # Mostly |1>
            np.array([[0.5, 0.5], [0.5, 0.5]]) / sqrt2  # Superposition
        ]
        
        povm = POVMeasurement(povm_elements)
        
        # Test on |+> state
        plus = QuantumState(n_qubits=1)
        plus.initialize_plus()
        
        outcomes = []
        for _ in range(1000):
            outcome, _ = povm.measure(plus.copy())
            outcomes.append(outcome)
            
        # Should get all three outcomes
        assert set(outcomes) == {0, 1, 2}
        
        # Third outcome should be most likely for |+>
        count2 = outcomes.count(2)
        assert count2 > 400  # Most probable
        
    def test_quantum_state_tomography_povm(self):
        """Test POVM for quantum state tomography."""
        # Create tomographically complete POVM
        povm = POVMeasurement.create_tomography_povm(n_qubits=1)
        
        # Unknown state to reconstruct
        theta = np.pi / 3
        unknown = QuantumState(n_qubits=1)
        unknown.amplitudes = np.array([np.cos(theta/2), np.sin(theta/2)])
        
        # Collect measurement statistics
        measurements = []
        for _ in range(5000):
            outcome, _ = povm.measure(unknown.copy())
            measurements.append(outcome)
            
        # Reconstruct state
        reconstructed = povm.reconstruct_state(measurements)
        
        # Check fidelity
        fidelity = np.abs(np.vdot(unknown.amplitudes, reconstructed))**2
        assert fidelity > 0.95


class TestContinuousMeasurement:
    """Test continuous measurement protocols."""
    
    def test_continuous_position_measurement(self):
        """Test continuous monitoring of position."""
        cont = ContinuousMeasurement(
            measurement_strength=0.1,
            time_step=0.01
        )
        
        # Initial superposition state
        state = QuantumState(n_qubits=1)
        state.initialize_plus()
        
        # Monitor for some time
        trajectory = []
        for _ in range(100):
            measurement, state = cont.measure_continuously(
                state,
                observable="Z"
            )
            trajectory.append(measurement)
            
        # Should see gradual collapse
        trajectory = np.array(trajectory)
        
        # Early measurements should fluctuate
        early_var = np.var(trajectory[:20])
        # Late measurements should stabilize
        late_var = np.var(trajectory[-20:])
        
        assert early_var > late_var
        
    def test_quantum_trajectory(self):
        """Test quantum trajectory unraveling."""
        cont = ContinuousMeasurement(
            measurement_strength=0.5,
            time_step=0.01
        )
        
        # Run multiple trajectories
        n_trajectories = 100
        final_states = []
        
        for _ in range(n_trajectories):
            state = QuantumState(n_qubits=1)
            state.initialize_plus()
            
            # Evolve with continuous measurement
            for _ in range(50):
                _, state = cont.measure_continuously(state, "Z")
                
            final_states.append(state.amplitudes.copy())
            
        # Check ensemble properties
        final_states = np.array(final_states)
        
        # Should collapse to either |0> or |1>
        collapsed_to_zero = np.sum(np.abs(final_states[:, 0])**2 > 0.9)
        collapsed_to_one = np.sum(np.abs(final_states[:, 1])**2 > 0.9)
        
        assert collapsed_to_zero + collapsed_to_one > 90
        
        # Should be roughly 50/50
        assert 30 < collapsed_to_zero < 70
        
    def test_measurement_backaction_dynamics(self):
        """Test measurement backaction on system dynamics."""
        cont = ContinuousMeasurement(
            measurement_strength=0.2,
            time_step=0.01,
            include_backaction=True
        )
        
        # System with Hamiltonian evolution
        H = np.array([[0, 1], [1, 0]])  # σx Hamiltonian
        
        state = QuantumState(n_qubits=1)
        state.initialize_zero()
        
        # Evolve with measurement
        energies = []
        for _ in range(200):
            # Hamiltonian evolution
            U = cont.hamiltonian_evolution(H, cont.time_step)
            state.apply_unitary(U)
            
            # Measurement
            _, state = cont.measure_continuously(state, "Z")
            
            # Record energy
            energy = np.real(state.expectation_value(H))
            energies.append(energy)
            
        # Measurement should affect energy evolution
        energies = np.array(energies)
        
        # Without measurement, energy would oscillate perfectly
        # With measurement, oscillations should decay
        early_amplitude = np.max(np.abs(energies[:50]))
        late_amplitude = np.max(np.abs(energies[-50:]))
        
        assert late_amplitude < early_amplitude * 0.8


class TestMeasurementBackaction:
    """Test measurement backaction effects."""
    
    def test_state_reduction(self):
        """Test wavefunction collapse from measurement."""
        backaction = MeasurementBackaction()
        
        # Superposition state
        state = QuantumState(n_qubits=1)
        state.initialize_plus()
        
        # Apply measurement backaction
        measurement_op = np.array([[1, 0], [0, 0]])  # |0><0|
        
        collapsed_state = backaction.apply_backaction(
            state,
            measurement_op,
            outcome_prob=0.5
        )
        
        # Should be |0> state
        assert abs(collapsed_state.amplitudes[0] - 1.0) < 1e-10
        assert abs(collapsed_state.amplitudes[1]) < 1e-10
        
    def test_partial_collapse(self):
        """Test partial collapse from weak measurement."""
        backaction = MeasurementBackaction()
        
        # Initial state
        state = QuantumState(n_qubits=1)
        state.initialize_plus()
        
        # Weak measurement operator
        strength = 0.1
        M0 = np.sqrt(1 - strength) * np.eye(2) + np.sqrt(strength) * np.diag([1, 0])
        
        partial_state = backaction.apply_backaction(state, M0)
        
        # Should be partially collapsed toward |0>
        prob_0 = np.abs(partial_state.amplitudes[0])**2
        assert 0.5 < prob_0 < 0.8  # Between equal and fully collapsed
        
    def test_measurement_induced_decoherence(self):
        """Test decoherence from repeated measurements."""
        backaction = MeasurementBackaction()
        
        # Start with cat state
        state = QuantumState(n_qubits=2)
        state.amplitudes[0] = 1/np.sqrt(2)  # |00>
        state.amplitudes[3] = 1/np.sqrt(2)  # |11>
        
        # Repeated weak measurements on first qubit
        for _ in range(10):
            # Weak measurement
            strength = 0.05
            M = np.kron(
                np.sqrt(1 - strength) * np.eye(2) + np.sqrt(strength) * np.diag([1, 0]),
                np.eye(2)
            )
            
            state = backaction.apply_backaction(state, M)
            
        # Check coherence loss
        coherence = abs(state.amplitudes[0] * np.conj(state.amplitudes[3]))
        assert coherence < 0.4  # Significant decoherence
        
    def test_quantum_zeno_effect(self):
        """Test quantum Zeno effect from frequent measurements."""
        backaction = MeasurementBackaction()
        
        # Hamiltonian for rotation
        H = np.array([[0, -1j], [1j, 0]])  # σy
        
        # Test with different measurement frequencies
        frequencies = [0, 10, 100]  # measurements per unit time
        final_states = []
        
        for freq in frequencies:
            state = QuantumState(n_qubits=1)
            state.initialize_zero()
            
            if freq == 0:
                # Just evolve
                U = backaction.hamiltonian_evolution(H, 1.0)
                state.apply_unitary(U)
            else:
                # Evolve with measurements
                dt = 1.0 / freq
                for _ in range(freq):
                    # Small evolution
                    U = backaction.hamiltonian_evolution(H, dt)
                    state.apply_unitary(U)
                    
                    # Measurement in computational basis
                    if np.random.rand() < 0.5:
                        M = np.array([[1, 0], [0, 0]])
                    else:
                        M = np.array([[0, 0], [0, 1]])
                    state = backaction.apply_backaction(state, M)
                    
            final_states.append(np.abs(state.amplitudes[0])**2)
            
        # More frequent measurements should keep state closer to |0>
        assert final_states[2] > final_states[1] > final_states[0]
        assert final_states[2] > 0.9  # Strong Zeno effect


class TestMeasurementStatistics:
    """Test measurement statistics and analysis."""
    
    def test_measurement_uncertainty(self):
        """Test measurement uncertainty relations."""
        stats = MeasurementStatistics()
        
        # Prepare state
        state = QuantumState(n_qubits=1)
        state.initialize_plus()
        
        # Measure uncertainty in X and Z
        delta_x = stats.calculate_uncertainty(state, "X")
        delta_z = stats.calculate_uncertainty(state, "Z")
        
        # For |+> state: ΔX = 0, ΔZ = 1
        assert delta_x < 0.01
        assert abs(delta_z - 1.0) < 0.01
        
        # Check uncertainty relation
        uncertainty_product = delta_x * delta_z
        
    def test_measurement_correlation_functions(self):
        """Test two-point correlation functions."""
        stats = MeasurementStatistics()
        
        # Bell state
        bell = QuantumState(n_qubits=2)
        bell.amplitudes[0] = 1/np.sqrt(2)
        bell.amplitudes[3] = 1/np.sqrt(2)
        
        # Calculate correlations
        corr_xx = stats.correlation_function(bell, "X", "X", [0, 1])
        corr_zz = stats.correlation_function(bell, "Z", "Z", [0, 1])
        corr_xz = stats.correlation_function(bell, "X", "Z", [0, 1])
        
        # For |Φ+>: <XX> = 1, <ZZ> = 1, <XZ> = 0
        assert abs(corr_xx - 1.0) < 0.01
        assert abs(corr_zz - 1.0) < 0.01
        assert abs(corr_xz) < 0.01
        
    def test_measurement_entropy(self):
        """Test measurement entropy calculations."""
        stats = MeasurementStatistics()
        
        # Pure state has zero entropy
        pure = QuantumState(n_qubits=1)
        pure.initialize_zero()
        entropy_pure = stats.measurement_entropy(pure)
        assert entropy_pure < 0.01
        
        # Maximally mixed state
        mixed = QuantumState(n_qubits=1)
        mixed.density_matrix = 0.5 * np.eye(2)
        entropy_mixed = stats.measurement_entropy(mixed)
        assert abs(entropy_mixed - np.log(2)) < 0.01
        
    def test_fisher_information(self):
        """Test quantum Fisher information."""
        stats = MeasurementStatistics()
        
        # Parameterized state |ψ(θ)> = cos(θ/2)|0> + sin(θ/2)|1>
        def parameterized_state(theta):
            state = QuantumState(n_qubits=1)
            state.amplitudes = np.array([np.cos(theta/2), np.sin(theta/2)])
            return state
            
        # Calculate Fisher information at θ = π/4
        theta0 = np.pi / 4
        fisher_info = stats.quantum_fisher_information(
            parameterized_state,
            theta0,
            delta=0.001
        )
        
        # For this state, F = 1
        assert abs(fisher_info - 1.0) < 0.1


# Edge cases and error handling
class TestMeasurementEdgeCases:
    """Test edge cases in measurement protocols."""
    
    def test_invalid_measurement_basis(self):
        """Test handling of invalid measurement bases."""
        with pytest.raises(ValueError):
            ProjectiveMeasurement(basis="invalid_basis")
            
    def test_non_normalized_povm(self):
        """Test POVM with non-normalized elements."""
        # Elements don't sum to identity
        bad_povm = [
            np.array([[1, 0], [0, 0]]),
            np.array([[0, 0], [0, 0.5]])
        ]
        
        with pytest.raises(ValueError):
            POVMeasurement(bad_povm)
            
    def test_measurement_on_invalid_state(self):
        """Test measurement on non-physical states."""
        proj = ProjectiveMeasurement()
        
        # Non-normalized state
        bad_state = QuantumState(n_qubits=1)
        bad_state.amplitudes = np.array([1, 1])
        
        with pytest.raises(ValueError):
            proj.measure(bad_state)
            
    def test_continuous_measurement_stability(self):
        """Test numerical stability of continuous measurement."""
        cont = ContinuousMeasurement(
            measurement_strength=10.0,  # Very strong
            time_step=0.001
        )
        
        state = QuantumState(n_qubits=1)
        state.initialize_plus()
        
        # Should handle strong measurement gracefully
        for _ in range(100):
            _, state = cont.measure_continuously(state, "Z")
            
        # State should remain normalized
        norm = np.linalg.norm(state.amplitudes)
        assert abs(norm - 1.0) < 0.01