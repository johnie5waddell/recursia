"""
Unit tests for entanglement module.
Tests entanglement creation, measurement, dynamics, and multi-party effects.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.physics.entanglement_proper import (
    EntanglementManager,
    EntanglementPair,
    MultipartyEntanglement,
    EntanglementWitness,
    EntanglementDynamics,
    EntanglementMetrics
)
from src.quantum.quantum_state import QuantumState
from src.core.types import RecursiaFloat, RecursiaComplex


class TestEntanglementPair:
    """Test two-party entanglement functionality."""
    
    def test_bell_state_creation(self):
        """Test creation of Bell states."""
        # Create Bell state |Φ+>
        pair = EntanglementPair()
        phi_plus = pair.create_bell_state("phi_plus")
        
        # Verify it's maximally entangled
        entanglement = pair.calculate_entanglement(phi_plus)
        assert abs(entanglement - 1.0) < 1e-10  # Max entanglement
        
        # Test all Bell states
        bell_states = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
        for state_name in bell_states:
            state = pair.create_bell_state(state_name)
            ent = pair.calculate_entanglement(state)
            assert abs(ent - 1.0) < 1e-10
            
    def test_partial_entanglement(self):
        """Test partially entangled states."""
        pair = EntanglementPair()
        
        # Create state: α|00> + β|11> with α > β
        alpha = 0.8
        beta = np.sqrt(1 - alpha**2)
        
        state = pair.create_entangled_state(alpha, beta)
        entanglement = pair.calculate_entanglement(state)
        
        # Should be less than maximal
        assert 0 < entanglement < 1
        
    def test_separable_state(self):
        """Test that separable states have zero entanglement."""
        pair = EntanglementPair()
        
        # Product state |00>
        separable = np.zeros((4, 4), dtype=complex)
        separable[0, 0] = 1.0
        
        entanglement = pair.calculate_entanglement(separable)
        assert abs(entanglement) < 1e-10
        
    def test_entanglement_swapping(self):
        """Test entanglement swapping protocol."""
        pair = EntanglementPair()
        
        # Create two Bell pairs
        pair1 = pair.create_bell_state("phi_plus")
        pair2 = pair.create_bell_state("phi_plus")
        
        # Perform swapping
        swapped = pair.entanglement_swapping(pair1, pair2)
        
        # Result should be entangled
        entanglement = pair.calculate_entanglement(swapped)
        assert entanglement > 0.5
        
    def test_entanglement_purification(self):
        """Test entanglement purification protocol."""
        pair = EntanglementPair()
        
        # Create noisy entangled states
        noise_level = 0.2
        noisy_state1 = pair.add_noise(
            pair.create_bell_state("phi_plus"),
            noise_level
        )
        noisy_state2 = pair.add_noise(
            pair.create_bell_state("phi_plus"),
            noise_level
        )
        
        # Purify
        purified = pair.purify_entanglement(noisy_state1, noisy_state2)
        
        # Should have higher entanglement than inputs
        ent_purified = pair.calculate_entanglement(purified)
        ent_noisy = pair.calculate_entanglement(noisy_state1)
        assert ent_purified > ent_noisy


class TestMultipartyEntanglement:
    """Test multi-party entanglement scenarios."""
    
    def test_ghz_state_creation(self):
        """Test GHZ state creation for various party numbers."""
        mpe = MultipartyEntanglement()
        
        for n_parties in [3, 4, 5]:
            ghz = mpe.create_ghz_state(n_parties)
            
            # Check it's genuinely multipartite entangled
            gme = mpe.genuine_multipartite_entanglement(ghz)
            assert gme > 0.9  # Should be highly entangled
            
    def test_w_state_creation(self):
        """Test W state creation and properties."""
        mpe = MultipartyEntanglement()
        
        # Create 3-party W state
        w_state = mpe.create_w_state(3)
        
        # W states should be robust against particle loss
        reduced_state = mpe.trace_out_party(w_state, party_index=0)
        entanglement = mpe.calculate_residual_entanglement(reduced_state)
        assert entanglement > 0  # Still entangled after loss
        
    def test_cluster_state(self):
        """Test cluster state creation for measurement-based QC."""
        mpe = MultipartyEntanglement()
        
        # Create 2D cluster state
        cluster = mpe.create_cluster_state(rows=2, cols=3)
        
        # Verify graph state properties
        assert mpe.is_graph_state(cluster)
        assert mpe.calculate_connectivity(cluster) > 0
        
    def test_entanglement_percolation(self):
        """Test entanglement percolation through network."""
        mpe = MultipartyEntanglement()
        
        # Create entanglement network
        network = mpe.create_entanglement_network(
            n_nodes=6,
            connectivity=0.5
        )
        
        # Test percolation
        can_percolate = mpe.test_percolation(
            network,
            source=0,
            target=5
        )
        
        # With 50% connectivity, should sometimes percolate
        assert isinstance(can_percolate, bool)
        
    def test_monogamy_constraints(self):
        """Test entanglement monogamy relations."""
        mpe = MultipartyEntanglement()
        
        # Create 3-party state
        state = mpe.create_random_state(3)
        
        # Calculate pairwise entanglements
        e_12 = mpe.bipartite_entanglement(state, [0], [1])
        e_13 = mpe.bipartite_entanglement(state, [0], [2])
        e_23 = mpe.bipartite_entanglement(state, [1], [2])
        
        # Check monogamy inequality
        assert mpe.check_monogamy(e_12, e_13, e_23)


class TestEntanglementWitness:
    """Test entanglement witness operators."""
    
    def test_witness_construction(self):
        """Test construction of entanglement witnesses."""
        witness = EntanglementWitness()
        
        # Create witness for |Φ+> state
        target_state = np.zeros((4, 4), dtype=complex)
        target_state[0, 0] = 0.5
        target_state[0, 3] = 0.5
        target_state[3, 0] = 0.5
        target_state[3, 3] = 0.5
        
        W = witness.construct_witness(target_state)
        
        # Witness should detect target entanglement
        expectation = np.trace(W @ target_state)
        assert expectation < 0  # Negative for entangled state
        
    def test_witness_optimization(self):
        """Test optimal witness construction."""
        witness = EntanglementWitness()
        
        # Generate random entangled states
        entangled_states = []
        for _ in range(5):
            pair = EntanglementPair()
            state = pair.create_bell_state("phi_plus")
            # Add small noise
            noisy = pair.add_noise(state, 0.1)
            entangled_states.append(noisy)
            
        # Optimize witness
        optimal_W = witness.optimize_witness(entangled_states)
        
        # Should detect all training states
        for state in entangled_states:
            exp_val = np.trace(optimal_W @ state)
            assert exp_val < 0
            
    def test_witness_robustness(self):
        """Test witness robustness to noise."""
        witness = EntanglementWitness()
        pair = EntanglementPair()
        
        # Create witness for Bell state
        bell = pair.create_bell_state("phi_plus")
        W = witness.construct_witness(bell)
        
        # Test detection under increasing noise
        noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4]
        detected = []
        
        for noise in noise_levels:
            noisy_state = pair.add_noise(bell, noise)
            exp_val = np.trace(W @ noisy_state)
            detected.append(exp_val < 0)
            
        # Should lose detection at high noise
        assert detected[0] == True  # No noise
        assert detected[-1] == False  # High noise


class TestEntanglementDynamics:
    """Test entanglement evolution and dynamics."""
    
    def test_entanglement_sudden_death(self):
        """Test entanglement sudden death phenomenon."""
        dynamics = EntanglementDynamics()
        pair = EntanglementPair()
        
        # Start with entangled state
        initial = pair.create_bell_state("phi_plus")
        
        # Evolve under amplitude damping
        times = np.linspace(0, 5, 50)
        entanglements = []
        
        state = initial
        for t in times:
            state = dynamics.amplitude_damping_evolution(
                state,
                damping_rate=0.5,
                time=t
            )
            ent = pair.calculate_entanglement(state)
            entanglements.append(ent)
            
        # Should show sudden death (not gradual)
        ent_array = np.array(entanglements)
        # Find where entanglement vanishes
        zero_idx = np.where(ent_array < 1e-6)[0]
        if len(zero_idx) > 0:
            # Check it stays zero (sudden death)
            assert np.all(ent_array[zero_idx[0]:] < 1e-6)
            
    def test_entanglement_revival(self):
        """Test entanglement revival under certain conditions."""
        dynamics = EntanglementDynamics()
        
        # Create initial state with potential for revival
        initial = dynamics.create_revival_state()
        
        # Evolve with special Hamiltonian
        times = np.linspace(0, 10, 100)
        entanglements = []
        
        for t in times:
            state = dynamics.revival_evolution(initial, t)
            ent = dynamics.calculate_entanglement(state)
            entanglements.append(ent)
            
        ent_array = np.array(entanglements)
        
        # Should show at least one revival
        # Find local minima and maxima
        minima = (ent_array[1:-1] < ent_array[:-2]) & \
                 (ent_array[1:-1] < ent_array[2:])
        maxima = (ent_array[1:-1] > ent_array[:-2]) & \
                 (ent_array[1:-1] > ent_array[2:])
                 
        # Should have alternating pattern
        assert np.sum(minima) > 0 and np.sum(maxima) > 0
        
    def test_entanglement_distillation(self):
        """Test entanglement distillation protocols."""
        dynamics = EntanglementDynamics()
        pair = EntanglementPair()
        
        # Create many noisy pairs
        n_pairs = 10
        noise = 0.3
        noisy_pairs = []
        
        for _ in range(n_pairs):
            bell = pair.create_bell_state("phi_plus")
            noisy = pair.add_noise(bell, noise)
            noisy_pairs.append(noisy)
            
        # Distill
        distilled = dynamics.distill_entanglement(
            noisy_pairs,
            target_fidelity=0.9
        )
        
        # Should have fewer but better pairs
        assert len(distilled) < n_pairs
        
        # Check quality
        for state in distilled:
            fidelity = dynamics.bell_state_fidelity(state)
            assert fidelity > 0.85


class TestEntanglementMetrics:
    """Test various entanglement measures and metrics."""
    
    def test_concurrence_calculation(self):
        """Test concurrence calculation for two qubits."""
        metrics = EntanglementMetrics()
        
        # Test known cases
        # Separable state
        separable = np.diag([1, 0, 0, 0])
        conc = metrics.concurrence(separable)
        assert abs(conc) < 1e-10
        
        # Maximally entangled
        bell = np.zeros((4, 4), dtype=complex)
        bell[0, 0] = 0.5
        bell[0, 3] = 0.5
        bell[3, 0] = 0.5
        bell[3, 3] = 0.5
        conc = metrics.concurrence(bell)
        assert abs(conc - 1.0) < 1e-10
        
    def test_negativity_calculation(self):
        """Test negativity as entanglement measure."""
        metrics = EntanglementMetrics()
        
        # Create test states
        pair = EntanglementPair()
        
        # Entangled state
        entangled = pair.create_bell_state("psi_minus")
        neg = metrics.negativity(entangled, [0, 1])
        assert neg > 0.4  # Should be positive
        
        # Separable state
        separable = np.diag([0.5, 0.5, 0, 0])
        neg = metrics.negativity(separable, [0, 1])
        assert abs(neg) < 1e-10
        
    def test_entanglement_entropy(self):
        """Test entanglement entropy calculation."""
        metrics = EntanglementMetrics()
        
        # Pure product state
        product = np.zeros(4, dtype=complex)
        product[0] = 1.0
        entropy = metrics.entanglement_entropy(product, [0])
        assert abs(entropy) < 1e-10
        
        # Maximally entangled state
        bell_vec = np.zeros(4, dtype=complex)
        bell_vec[0] = 1/np.sqrt(2)
        bell_vec[3] = 1/np.sqrt(2)
        entropy = metrics.entanglement_entropy(bell_vec, [0])
        assert abs(entropy - np.log(2)) < 1e-10  # Max entropy
        
    def test_relative_entropy_entanglement(self):
        """Test relative entropy of entanglement."""
        metrics = EntanglementMetrics()
        
        # Test state
        test_state = np.diag([0.4, 0.3, 0.2, 0.1])
        
        # Calculate
        rel_entropy = metrics.relative_entropy_entanglement(test_state)
        
        # Should be non-negative
        assert rel_entropy >= 0
        
    def test_entanglement_cost_and_distillation(self):
        """Test entanglement cost and distillable entanglement."""
        metrics = EntanglementMetrics()
        pair = EntanglementPair()
        
        # Create mixed entangled state
        pure = pair.create_bell_state("phi_plus")
        mixed = 0.8 * pure + 0.2 * np.eye(4) / 4
        
        # Calculate bounds
        e_cost = metrics.entanglement_cost(mixed)
        e_distill = metrics.distillable_entanglement(mixed)
        
        # Cost >= Distillable (general theorem)
        assert e_cost >= e_distill - 1e-10
        

# Edge cases and error handling
class TestEntanglementEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_state_dimensions(self):
        """Test handling of invalid state dimensions."""
        pair = EntanglementPair()
        
        # Non-power-of-2 dimension
        invalid_state = np.zeros((3, 3), dtype=complex)
        with pytest.raises(ValueError):
            pair.calculate_entanglement(invalid_state)
            
    def test_non_physical_states(self):
        """Test handling of non-physical states."""
        metrics = EntanglementMetrics()
        
        # Non-positive matrix
        non_positive = np.array([[1, 0], [0, -1]], dtype=complex)
        with pytest.raises(ValueError):
            metrics.concurrence(non_positive)
            
        # Non-normalized
        non_normalized = 2 * np.eye(4)
        with pytest.raises(ValueError):
            metrics.negativity(non_normalized, [0, 1])
            
    def test_extreme_multiparty_systems(self):
        """Test behavior with many parties."""
        mpe = MultipartyEntanglement()
        
        # Large GHZ state
        large_ghz = mpe.create_ghz_state(10)
        
        # Should still be valid
        assert large_ghz.shape[0] == 2**10
        
        # But computational measures might fail gracefully
        try:
            gme = mpe.genuine_multipartite_entanglement(large_ghz)
            assert 0 <= gme <= 1
        except MemoryError:
            # Acceptable for very large systems
            pass
            
    def test_numerical_precision(self):
        """Test numerical precision in calculations."""
        metrics = EntanglementMetrics()
        
        # Create state very close to separable
        almost_separable = np.diag([0.5, 0.5, 0, 0])
        almost_separable[0, 3] = 1e-15  # Tiny coherence
        almost_separable[3, 0] = 1e-15
        
        # Should handle gracefully
        conc = metrics.concurrence(almost_separable)
        assert conc >= 0  # No negative values from numerical errors
        assert conc < 1e-10  # Still essentially separable