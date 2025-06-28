"""
Unit tests for coherence module.
Tests coherence calculations, decoherence dynamics, and quantum memory effects.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.physics.coherence_proper import (
    CoherenceField,
    QuantumCoherence,
    MemoryCoherence,
    ObserverCoherence,
    DecoherenceChannel,
    CoherenceMetrics
)
from src.core.types import RecursiaFloat, RecursiaComplex
from src.quantum.quantum_state import QuantumState
from src.physics.observer import QuantumObserver


class TestQuantumCoherence:
    """Test QuantumCoherence calculations and properties."""
    
    def test_coherence_from_density_matrix(self):
        """Test coherence calculation from density matrix."""
        # Pure state should have coherence 1.0
        pure_state = np.array([[1, 0], [0, 0]], dtype=complex)
        qc = QuantumCoherence(2)
        coherence = qc.calculate_coherence(pure_state)
        assert abs(coherence - 1.0) < 1e-10
        
        # Maximally mixed state should have coherence 0.0
        mixed_state = np.array([[0.5, 0], [0, 0.5]], dtype=complex)
        coherence = qc.calculate_coherence(mixed_state)
        assert abs(coherence) < 1e-10
        
    def test_superposition_coherence(self):
        """Test coherence of superposition states."""
        qc = QuantumCoherence(2)
        
        # |+> state (equal superposition)
        plus_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        coherence = qc.calculate_coherence(plus_state)
        assert 0.9 < coherence <= 1.0  # Should be highly coherent
        
    def test_entangled_state_coherence(self):
        """Test coherence of entangled states."""
        qc = QuantumCoherence(4)  # 2-qubit system
        
        # Bell state |Φ+> = (|00> + |11>)/√2
        bell_state = np.zeros((4, 4), dtype=complex)
        bell_state[0, 0] = 0.5
        bell_state[0, 3] = 0.5
        bell_state[3, 0] = 0.5
        bell_state[3, 3] = 0.5
        
        coherence = qc.calculate_coherence(bell_state)
        assert coherence > 0.8  # Entangled states should be coherent
        
    def test_coherence_bounds(self):
        """Test that coherence values are properly bounded."""
        qc = QuantumCoherence(3)
        
        # Generate random density matrices
        for _ in range(10):
            # Create random Hermitian matrix
            H = np.random.randn(8, 8) + 1j * np.random.randn(8, 8)
            H = H + H.conj().T
            
            # Make it a valid density matrix via eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(H)
            eigvals = np.abs(eigvals)
            eigvals = eigvals / np.sum(eigvals)
            rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
            
            coherence = qc.calculate_coherence(rho)
            assert 0.0 <= coherence <= 1.0


class TestMemoryCoherence:
    """Test memory-induced coherence effects."""
    
    def test_memory_strain_effect(self):
        """Test how memory strain affects coherence."""
        mc = MemoryCoherence(strain_threshold=0.8)
        
        # Low strain should maintain coherence
        coherence = mc.apply_memory_effects(
            initial_coherence=0.9,
            memory_strain=0.2,
            time_step=0.1
        )
        assert coherence > 0.85  # Small degradation
        
        # High strain should reduce coherence
        coherence = mc.apply_memory_effects(
            initial_coherence=0.9,
            memory_strain=0.95,
            time_step=0.1
        )
        assert coherence < 0.7  # Significant degradation
        
    def test_memory_recovery(self):
        """Test coherence recovery after memory defragmentation."""
        mc = MemoryCoherence(strain_threshold=0.8, recovery_rate=0.5)
        
        # Simulate defragmentation
        coherence = mc.apply_memory_recovery(
            current_coherence=0.3,
            recovery_factor=0.8,
            time_step=0.1
        )
        assert coherence > 0.3  # Should increase
        assert coherence < 0.9  # But not instantly to maximum
        
    def test_recursive_memory_effects(self):
        """Test coherence in recursive memory contexts."""
        mc = MemoryCoherence(strain_threshold=0.8)
        
        # Deeper recursion should affect coherence
        coherence_depth_1 = mc.apply_recursive_effects(
            base_coherence=0.9,
            recursion_depth=1
        )
        coherence_depth_5 = mc.apply_recursive_effects(
            base_coherence=0.9,
            recursion_depth=5
        )
        
        assert coherence_depth_5 < coherence_depth_1
        assert coherence_depth_5 > 0  # Should still be positive


class TestObserverCoherence:
    """Test observer-induced coherence effects."""
    
    def test_observation_decoherence(self):
        """Test decoherence from observation."""
        oc = ObserverCoherence(observation_strength=0.5)
        
        # Strong observation should cause decoherence
        coherence = oc.apply_observation(
            initial_coherence=0.95,
            observer_focus=1.0,
            measurement_strength=0.9
        )
        assert coherence < 0.5  # Strong decoherence
        
        # Weak observation should preserve more coherence
        coherence = oc.apply_observation(
            initial_coherence=0.95,
            observer_focus=0.1,
            measurement_strength=0.1
        )
        assert coherence > 0.8  # Minimal decoherence
        
    def test_multi_observer_effects(self):
        """Test coherence with multiple observers."""
        oc = ObserverCoherence(observation_strength=0.5)
        
        observers = [
            {"focus": 0.3, "strength": 0.2},
            {"focus": 0.5, "strength": 0.3},
            {"focus": 0.2, "strength": 0.1}
        ]
        
        coherence = oc.apply_multi_observer(
            initial_coherence=0.9,
            observers=observers
        )
        
        # Multiple weak observers should cause moderate decoherence
        assert 0.5 < coherence < 0.8
        
    def test_observer_entanglement_protection(self):
        """Test that entangled observers can protect coherence."""
        oc = ObserverCoherence(
            observation_strength=0.5,
            entanglement_protection=True
        )
        
        # Entangled observers should preserve more coherence
        coherence = oc.apply_entangled_observation(
            initial_coherence=0.9,
            observer_entanglement=0.8,
            measurement_strength=0.5
        )
        assert coherence > 0.7  # Better than non-entangled case


class TestDecoherenceChannel:
    """Test various decoherence channels."""
    
    def test_amplitude_damping(self):
        """Test amplitude damping channel."""
        dc = DecoherenceChannel(channel_type="amplitude_damping")
        
        # Apply to excited state
        excited = np.array([[0, 0], [0, 1]], dtype=complex)
        damped = dc.apply_channel(excited, damping_parameter=0.3)
        
        # Should have partial decay to ground state
        assert damped[0, 0] > 0  # Some population in ground
        assert damped[1, 1] < 1  # Reduced excited population
        
    def test_phase_damping(self):
        """Test phase damping channel."""
        dc = DecoherenceChannel(channel_type="phase_damping")
        
        # Apply to superposition state
        plus_state = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        damped = dc.apply_channel(plus_state, damping_parameter=0.5)
        
        # Off-diagonal elements should be reduced
        assert abs(damped[0, 1]) < abs(plus_state[0, 1])
        assert abs(damped[1, 0]) < abs(plus_state[1, 0])
        
    def test_depolarizing_channel(self):
        """Test depolarizing channel."""
        dc = DecoherenceChannel(channel_type="depolarizing")
        
        # Apply to pure state
        pure = np.array([[1, 0], [0, 0]], dtype=complex)
        depolarized = dc.apply_channel(pure, error_probability=0.5)
        
        # Should move toward maximally mixed
        assert depolarized[0, 0] < 1
        assert depolarized[1, 1] > 0
        
    def test_custom_kraus_operators(self):
        """Test custom Kraus operator decoherence."""
        # Define custom Kraus operators for bit flip
        K0 = np.sqrt(0.9) * np.eye(2)
        K1 = np.sqrt(0.1) * np.array([[0, 1], [1, 0]])
        
        dc = DecoherenceChannel(
            channel_type="custom",
            kraus_operators=[K0, K1]
        )
        
        # Apply to |0> state
        zero = np.array([[1, 0], [0, 0]], dtype=complex)
        flipped = dc.apply_channel(zero)
        
        # Should have small probability of bit flip
        assert flipped[0, 0] > 0.8  # Mostly |0>
        assert flipped[1, 1] < 0.2  # Small |1> component


class TestCoherenceField:
    """Test the full CoherenceField system."""
    
    @pytest.fixture
    def coherence_field(self):
        """Create a test coherence field."""
        return CoherenceField(
            dimension=10,
            coupling_strength=0.1,
            temperature=0.01
        )
        
    def test_field_initialization(self, coherence_field):
        """Test field initialization."""
        assert coherence_field.dimension == 10
        assert coherence_field.field_values.shape == (10, 10)
        assert np.all(coherence_field.field_values >= 0)
        assert np.all(coherence_field.field_values <= 1)
        
    def test_field_evolution(self, coherence_field):
        """Test coherence field time evolution."""
        initial_field = coherence_field.field_values.copy()
        
        # Evolve for several steps
        for _ in range(10):
            coherence_field.evolve(time_step=0.1)
            
        # Field should change but remain valid
        assert not np.array_equal(initial_field, coherence_field.field_values)
        assert np.all(coherence_field.field_values >= 0)
        assert np.all(coherence_field.field_values <= 1)
        
    def test_field_measurement_backaction(self, coherence_field):
        """Test measurement effects on coherence field."""
        # Perform measurement at center
        measurement_position = (5, 5)
        measurement_strength = 0.8
        
        pre_measurement = coherence_field.field_values.copy()
        coherence_field.apply_measurement(
            position=measurement_position,
            strength=measurement_strength
        )
        
        # Should cause local decoherence
        assert coherence_field.field_values[measurement_position] < \
               pre_measurement[measurement_position]
               
    def test_field_entanglement_distribution(self, coherence_field):
        """Test how entanglement affects coherence distribution."""
        # Create entangled region
        region1 = (2, 2)
        region2 = (7, 7)
        
        coherence_field.create_entanglement(
            region1, region2,
            entanglement_strength=0.9
        )
        
        # Check correlation between regions
        correlation = coherence_field.calculate_correlation(region1, region2)
        assert correlation > 0.7  # Should be strongly correlated


class TestCoherenceMetrics:
    """Test coherence metrics and analysis."""
    
    def test_l1_coherence_measure(self):
        """Test l1-norm coherence measure."""
        metrics = CoherenceMetrics()
        
        # Diagonal matrix has zero coherence
        diagonal = np.diag([0.5, 0.3, 0.2])
        l1_coherence = metrics.l1_coherence(diagonal)
        assert abs(l1_coherence) < 1e-10
        
        # Full matrix has non-zero coherence
        full = np.ones((3, 3)) / 3
        l1_coherence = metrics.l1_coherence(full)
        assert l1_coherence > 0
        
    def test_relative_entropy_coherence(self):
        """Test relative entropy coherence measure."""
        metrics = CoherenceMetrics()
        
        # Pure state
        pure = np.array([[1, 0], [0, 0]], dtype=complex)
        entropy_coherence = metrics.relative_entropy_coherence(pure)
        assert entropy_coherence == 0  # Pure diagonal state
        
        # Superposition state
        plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
        entropy_coherence = metrics.relative_entropy_coherence(plus)
        assert entropy_coherence > 0
        
    def test_coherence_variance(self):
        """Test coherence variance calculations."""
        metrics = CoherenceMetrics()
        
        # Uniform coherence should have low variance
        coherence_samples = [0.8, 0.81, 0.79, 0.8, 0.82]
        variance = metrics.coherence_variance(coherence_samples)
        assert variance < 0.01
        
        # Variable coherence should have high variance
        variable_samples = [0.1, 0.9, 0.2, 0.8, 0.15]
        variance = metrics.coherence_variance(variable_samples)
        assert variance > 0.1


# Edge case tests
class TestCoherenceEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_dimensional_system(self):
        """Test handling of zero-dimensional systems."""
        with pytest.raises(ValueError):
            QuantumCoherence(0)
            
    def test_invalid_density_matrix(self):
        """Test handling of invalid density matrices."""
        qc = QuantumCoherence(2)
        
        # Non-Hermitian matrix
        non_hermitian = np.array([[1, 1j], [0, 0]], dtype=complex)
        with pytest.raises(ValueError):
            qc.calculate_coherence(non_hermitian)
            
        # Non-normalized matrix
        non_normalized = np.array([[2, 0], [0, 0]], dtype=complex)
        with pytest.raises(ValueError):
            qc.calculate_coherence(non_normalized)
            
    def test_extreme_parameters(self):
        """Test behavior with extreme parameter values."""
        # Very high temperature
        high_temp_field = CoherenceField(
            dimension=5,
            temperature=1000.0
        )
        # Should still initialize successfully
        assert high_temp_field is not None
        
        # Zero coupling
        zero_coupling = CoherenceField(
            dimension=5,
            coupling_strength=0.0
        )
        initial = zero_coupling.field_values.copy()
        zero_coupling.evolve(1.0)
        # Should not evolve with zero coupling
        assert np.allclose(initial, zero_coupling.field_values)