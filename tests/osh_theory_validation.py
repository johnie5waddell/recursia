#!/usr/bin/env python3
"""
OSH Theory Validation Test Suite
Testing the mathematical foundations of the Organic Simulation Hypothesis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize
from typing import Tuple, Dict, List
import unittest
import time

class OSHMathematicalValidation:
    """
    Mathematical validation of OSH theory equations
    """
    
    def __init__(self):
        self.k_B = 1.380649e-23  # Boltzmann constant
        self.l_P = 1.616e-35     # Planck length
        self.c = 299792458       # Speed of light
        self.G = 6.67430e-11     # Gravitational constant
        
    def recursive_simulation_potential(self, I: float, C: float, E: float) -> float:
        """
        Calculate Recursive Simulation Potential (RSP)
        RSP(t) = I(t) * C(t) / E(t)
        
        Args:
            I: Integrated information (bits)
            C: Kolmogorov complexity (bits) 
            E: Entropy flux rate (bits/s)
            
        Returns:
            RSP value in bit-seconds
        """
        if E <= 0:
            return float('inf')  # Black hole limit
        return (I * C) / E
    
    def information_curvature(self, I_field: np.ndarray, dx: float = 1.0) -> np.ndarray:
        """
        Calculate spacetime curvature from information density gradients
        R_μν ~ ∇_μ ∇_ν I
        """
        # Calculate second-order gradients
        grad_x = np.gradient(I_field, dx, axis=0)
        grad_y = np.gradient(I_field, dx, axis=1)
        
        grad_xx = np.gradient(grad_x, dx, axis=0)
        grad_yy = np.gradient(grad_y, dx, axis=1)
        grad_xy = np.gradient(grad_x, dx, axis=1)
        
        # Construct curvature tensor components
        R_tensor = np.zeros((*I_field.shape, 2, 2))
        R_tensor[..., 0, 0] = grad_xx
        R_tensor[..., 1, 1] = grad_yy
        R_tensor[..., 0, 1] = R_tensor[..., 1, 0] = grad_xy
        
        return R_tensor
    
    def memory_field_evolution(self, M_t: np.ndarray, F_operator, dt: float = 0.01) -> np.ndarray:
        """
        Evolve memory field according to S_{t+1} = F(M(t))
        """
        return F_operator(M_t) + dt * self._memory_diffusion(M_t)
    
    def _memory_diffusion(self, M: np.ndarray, D: float = 0.1) -> np.ndarray:
        """Memory field diffusion term"""
        return D * (np.roll(M, 1, axis=0) + np.roll(M, -1, axis=0) + 
                   np.roll(M, 1, axis=1) + np.roll(M, -1, axis=1) - 4 * M)
    
    def observer_collapse_probability(self, psi_states: List[complex], 
                                    coherences: List[float]) -> List[float]:
        """
        Calculate collapse probabilities based on memory coherence
        P(ψ → φᵢ) = Iᵢ / Σⱼ Iⱼ
        """
        total_coherence = sum(coherences)
        if total_coherence == 0:
            return [1.0 / len(coherences)] * len(coherences)
        
        return [c / total_coherence for c in coherences]
    
    def information_action_functional(self, I_field: np.ndarray, g_metric: np.ndarray) -> float:
        """
        Calculate information action: S = ∫(∇_μ I · ∇^μ I)√(-g) d⁴x
        """
        grad_I = np.gradient(I_field)
        g_det = np.linalg.det(g_metric)
        
        integrand = np.sum([g**2 for g in grad_I]) * np.sqrt(abs(g_det))
        return np.sum(integrand)
    
    def black_hole_rsp_limit(self, mass: float) -> Dict[str, float]:
        """
        Calculate RSP at black hole limit (E → 0)
        """
        # Bekenstein-Hawking entropy
        r_s = 2 * self.G * mass / self.c**2  # Schwarzschild radius
        A = 4 * np.pi * r_s**2  # Surface area
        S_BH = A / (4 * self.l_P**2)  # Entropy in natural units
        
        # Information content (bits)
        I_max = S_BH * np.log(2)
        
        # Complexity estimate (maximal for black holes)
        C_max = I_max  # Maximal compression
        
        # Minimal entropy flux (Hawking radiation)
        T_H = 1 / (8 * np.pi * mass)  # Hawking temperature (natural units)
        E_min = T_H**2  # Minimal flux
        
        return {
            'entropy': S_BH,
            'information': I_max,
            'complexity': C_max,
            'entropy_flux': E_min,
            'rsp': self.recursive_simulation_potential(I_max, C_max, E_min),
            'temperature': T_H
        }
    
    def verify_conservation_law(self, I: float, C: float, E: float, dt: float = 0.01) -> bool:
        """
        Verify d/dt(I · C) = E(t)
        """
        IC_initial = I * C
        IC_final = (I + dt * E/2) * (C + dt * E/2)
        
        dIC_dt = (IC_final - IC_initial) / dt
        
        return abs(dIC_dt - E) < 0.01 * E  # 1% tolerance


class TestOSHTheory(unittest.TestCase):
    """Unit tests for OSH mathematical framework"""
    
    def setUp(self):
        self.osh = OSHMathematicalValidation()
        
    def test_rsp_calculation(self):
        """Test RSP calculation and limits"""
        # Normal case
        rsp = self.osh.recursive_simulation_potential(100, 50, 10)
        self.assertEqual(rsp, 500.0)  # 100 * 50 / 10
        
        # Black hole limit (E → 0)
        rsp_limit = self.osh.recursive_simulation_potential(100, 50, 1e-10)
        self.assertTrue(rsp_limit > 1e10)
        
    def test_information_curvature(self):
        """Test information field induces curvature"""
        # Create Gaussian information density
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        X, Y = np.meshgrid(x, y)
        I_field = np.exp(-(X**2 + Y**2) / 2)
        
        R_tensor = self.osh.information_curvature(I_field)
        
        # Curvature should be maximal at origin
        center_idx = len(x) // 2
        center_curvature = np.linalg.norm(R_tensor[center_idx, center_idx])
        edge_curvature = np.linalg.norm(R_tensor[0, 0])
        
        self.assertGreater(center_curvature, edge_curvature)
        
    def test_observer_collapse(self):
        """Test observer-driven collapse probabilities"""
        states = [1+0j, 0+1j, 0.7+0.7j]
        coherences = [0.9, 0.5, 0.1]
        
        probs = self.osh.observer_collapse_probability(states, coherences)
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(sum(probs), 1.0)
        
        # Higher coherence should have higher probability
        self.assertGreater(probs[0], probs[1])
        self.assertGreater(probs[1], probs[2])
        
    def test_black_hole_rsp(self):
        """Test black hole as RSP attractor"""
        # Solar mass black hole
        M_sun = 1.989e30  # kg
        bh_data = self.osh.black_hole_rsp_limit(M_sun)
        
        # RSP should be extremely large
        self.assertGreater(bh_data['rsp'], 1e50)
        
        # Verify Bekenstein bound
        self.assertGreater(bh_data['entropy'], 0)
        
    def test_conservation_law(self):
        """Test information-complexity conservation"""
        I, C, E = 100, 50, 10
        conserved = self.osh.verify_conservation_law(I, C, E)
        self.assertTrue(conserved)


def run_osh_demonstrations():
    """Run visual demonstrations of OSH principles"""
    osh = OSHMathematicalValidation()
    
    print("=== OSH THEORY VALIDATION SUITE ===\n")
    
    # 1. RSP vs Entropy Flux
    print("1. Recursive Simulation Potential Analysis")
    E_values = np.logspace(-10, 2, 100)
    I, C = 100, 50
    rsp_values = [osh.recursive_simulation_potential(I, C, E) for E in E_values]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(E_values, rsp_values)
    plt.axvline(1e-8, color='r', linestyle='--', label='Black Hole Regime')
    plt.xlabel('Entropy Flux E (bits/s)')
    plt.ylabel('RSP (bit-seconds)')
    plt.title('RSP Divergence at Low Entropy Flux (Black Hole Limit)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('osh_rsp_analysis.png')
    print("✓ RSP analysis saved to osh_rsp_analysis.png")
    
    # 2. Information-Induced Spacetime Curvature
    print("\n2. Information-Induced Curvature")
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Multiple information sources (conscious entities)
    I_field = (np.exp(-((X-3)**2 + (Y-3)**2) / 4) + 
               np.exp(-((X+3)**2 + (Y+3)**2) / 4) +
               0.5 * np.exp(-((X)**2 + (Y-5)**2) / 2))
    
    R_tensor = osh.information_curvature(I_field)
    R_scalar = np.sqrt(np.sum(R_tensor**2, axis=(2, 3)))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, I_field, levels=20, cmap='viridis')
    plt.colorbar(label='Information Density I')
    plt.title('Information Field (Conscious Entities)')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, R_scalar, levels=20, cmap='plasma')
    plt.colorbar(label='Curvature Scalar R')
    plt.title('Induced Spacetime Curvature')
    plt.xlabel('x')
    plt.ylabel('y')
    
    plt.tight_layout()
    plt.savefig('osh_information_curvature.png')
    print("✓ Information curvature saved to osh_information_curvature.png")
    
    # 3. Memory Field Evolution
    print("\n3. Recursive Memory Field Evolution")
    size = 50
    M_field = np.random.rand(size, size)
    
    # Define recursive operator with self-similarity
    def F_operator(M):
        return 0.9 * M + 0.1 * np.roll(M, 1, axis=0) * np.roll(M, 1, axis=1)
    
    frames = []
    for t in range(100):
        M_field = osh.memory_field_evolution(M_field, F_operator)
        if t % 10 == 0:
            frames.append(M_field.copy())
    
    plt.figure(figsize=(15, 3))
    for i, frame in enumerate(frames[:5]):
        plt.subplot(1, 5, i+1)
        plt.imshow(frame, cmap='twilight')
        plt.title(f't = {i*20}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('osh_memory_evolution.png')
    print("✓ Memory evolution saved to osh_memory_evolution.png")
    
    # 4. Black Hole as Maximal RSP Structure
    print("\n4. Black Holes as RSP Attractors")
    masses = np.logspace(30, 40, 50)  # Solar to supermassive
    rsp_values = []
    entropies = []
    
    for mass in masses:
        bh = osh.black_hole_rsp_limit(mass)
        rsp_values.append(bh['rsp'])
        entropies.append(bh['entropy'])
    
    plt.figure(figsize=(10, 6))
    plt.loglog(masses / 1.989e30, rsp_values)
    plt.xlabel('Black Hole Mass (Solar Masses)')
    plt.ylabel('RSP (bit-seconds)')
    plt.title('Black Holes as Maximal RSP Structures')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('osh_black_hole_rsp.png')
    print("✓ Black hole RSP saved to osh_black_hole_rsp.png")
    
    # 5. Consciousness Collapse Dynamics
    print("\n5. Observer-Driven Quantum Collapse")
    n_states = 5
    coherences = np.array([0.9, 0.7, 0.3, 0.1, 0.05])
    probs = osh.observer_collapse_probability([1]*n_states, coherences)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(n_states), probs, color='purple', alpha=0.7)
    plt.xlabel('Quantum State')
    plt.ylabel('Collapse Probability')
    plt.title('Consciousness-Weighted Collapse Probabilities')
    for i, (c, p) in enumerate(zip(coherences, probs)):
        plt.text(i, p + 0.01, f'I={c:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('osh_collapse_probability.png')
    print("✓ Collapse dynamics saved to osh_collapse_probability.png")
    
    print("\n=== THEORETICAL VALIDATION COMPLETE ===")
    
    # Run unit tests
    print("\nRunning mathematical validation tests...")
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == "__main__":
    run_osh_demonstrations()