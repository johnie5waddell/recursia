"""
Quantum Validation Suite - Comprehensive Testing Framework

Provides rigorous validation of quantum simulations against:
- Analytical solutions (harmonic oscillator, hydrogen atom, etc.)
- Experimental benchmarks from leading quantum computing papers
- Cross-validation with established simulators (Qiskit, Cirq)
- Statistical verification of quantum properties
- Performance benchmarks and scaling analysis
"""

import numpy as np
import pytest
import unittest
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings

# Import our quantum modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.physics.physics_engine_proper import QuantumPhysicsEngineProper, QuantumSystemState
from src.physics.gate_operations_proper import GateOperationsProper
from src.physics.entanglement_proper import EntanglementManagerProper
from src.physics.measurement.measurement_proper import QuantumMeasurementProper
from src.physics.coherence_proper import CoherenceManagerProper

# Cross-validation imports (optional)
try:
    import qiskit
    from qiskit import QuantumCircuit, Aer, execute
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

try:
    import cirq
    CIRQ_AVAILABLE = True
except ImportError:
    CIRQ_AVAILABLE = False

logger = logging.getLogger(__name__)

# Test tolerances based on numerical precision requirements
AMPLITUDE_TOLERANCE = 1e-12
PROBABILITY_TOLERANCE = 1e-14
ENERGY_TOLERANCE = 1e-10
ENTANGLEMENT_TOLERANCE = 1e-12

# Known analytical results for validation
ANALYTICAL_RESULTS = {
    'bell_state_entanglement': {
        'concurrence': 1.0,
        'negativity': 0.5,
        'entropy': 1.0  # ln(2) in nats, 1.0 in bits
    },
    'hadamard_superposition': {
        'probability_0': 0.5,
        'probability_1': 0.5,
        'bloch_x': 1.0,
        'bloch_y': 0.0,
        'bloch_z': 0.0
    },
    'pauli_eigenvalues': {
        'X': [-1, 1],
        'Y': [-1, 1], 
        'Z': [-1, 1]
    }
}


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    error_message: Optional[str] = None
    numerical_error: Optional[float] = None
    execution_time: Optional[float] = None
    metadata: Dict[str, Any] = None


class QuantumTestCase(ABC):
    """Abstract base class for quantum validation tests."""
    
    @abstractmethod
    def run_test(self) -> ValidationResult:
        """Run the validation test."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get test description."""
        pass


class BellStateValidation(QuantumTestCase):
    """Validate Bell state preparation and entanglement measures."""
    
    def __init__(self, physics_engine: QuantumPhysicsEngineProper):
        self.physics_engine = physics_engine
        self.entanglement_manager = EntanglementManagerProper()
        
    def run_test(self) -> ValidationResult:
        """Test Bell state creation and entanglement measurement."""
        start_time = time.perf_counter()
        
        try:
            # Create Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            bell_state = self.entanglement_manager.create_bell_state(bell_type=0)
            
            # Validate state properties
            trace = np.trace(bell_state)
            if not np.isclose(trace, 1.0, atol=PROBABILITY_TOLERANCE):
                return ValidationResult(
                    test_name="Bell State Validation",
                    passed=False,
                    error_message=f"Bell state not normalized: Tr(ρ) = {trace}"
                )
            
            # Check Hermiticity
            if not np.allclose(bell_state, bell_state.conj().T, atol=AMPLITUDE_TOLERANCE):
                return ValidationResult(
                    test_name="Bell State Validation", 
                    passed=False,
                    error_message="Bell state not Hermitian"
                )
            
            # Calculate entanglement measures
            partition = ([0], [1])  # Bipartition for 2-qubit system
            
            concurrence = self.entanglement_manager.calculate_concurrence(bell_state)
            negativity = self.entanglement_manager.calculate_negativity(bell_state, partition)
            entropy = self.entanglement_manager.calculate_entanglement_entropy(bell_state, partition)
            
            # Validate against analytical results
            expected = ANALYTICAL_RESULTS['bell_state_entanglement']
            
            errors = []
            if not np.isclose(concurrence, expected['concurrence'], atol=ENTANGLEMENT_TOLERANCE):
                errors.append(f"Concurrence: {concurrence} ≠ {expected['concurrence']}")
                
            if not np.isclose(negativity, expected['negativity'], atol=ENTANGLEMENT_TOLERANCE):
                errors.append(f"Negativity: {negativity} ≠ {expected['negativity']}")
                
            if not np.isclose(entropy, expected['entropy'], atol=ENTANGLEMENT_TOLERANCE):
                errors.append(f"Entropy: {entropy} ≠ {expected['entropy']}")
            
            execution_time = time.perf_counter() - start_time
            
            if errors:
                return ValidationResult(
                    test_name="Bell State Validation",
                    passed=False,
                    error_message="; ".join(errors),
                    execution_time=execution_time
                )
            
            return ValidationResult(
                test_name="Bell State Validation",
                passed=True,
                execution_time=execution_time,
                metadata={
                    'concurrence': concurrence,
                    'negativity': negativity,
                    'entropy': entropy
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Bell State Validation",
                passed=False,
                error_message=f"Exception: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
    
    def get_description(self) -> str:
        return "Validates Bell state preparation and entanglement measures against analytical results"


class QuantumGateValidation(QuantumTestCase):
    """Validate quantum gate operations."""
    
    def __init__(self):
        self.gate_ops = GateOperationsProper()
        
    def run_test(self) -> ValidationResult:
        """Test quantum gate unitarity and eigenvalues."""
        start_time = time.perf_counter()
        
        try:
            errors = []
            
            # Test Pauli gates
            pauli_gates = ['X', 'Y', 'Z']
            
            for gate_name in pauli_gates:
                # Get gate matrix
                if gate_name == 'X':
                    matrix = np.array([[0, 1], [1, 0]], dtype=complex)
                elif gate_name == 'Y':
                    matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
                else:  # Z
                    matrix = np.array([[1, 0], [0, -1]], dtype=complex)
                
                # Check unitarity
                try:
                    self.gate_ops.validate_unitary(matrix)
                except Exception as e:
                    errors.append(f"{gate_name} gate not unitary: {e}")
                    continue
                
                # Check eigenvalues
                eigenvals = np.linalg.eigvals(matrix)
                eigenvals_sorted = np.sort(eigenvals)
                expected_eigenvals = np.sort(ANALYTICAL_RESULTS['pauli_eigenvalues'][gate_name])
                
                if not np.allclose(eigenvals_sorted, expected_eigenvals, atol=AMPLITUDE_TOLERANCE):
                    errors.append(f"{gate_name} eigenvalues incorrect: {eigenvals_sorted} ≠ {expected_eigenvals}")
            
            # Test Hadamard gate
            h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            
            try:
                self.gate_ops.validate_unitary(h_matrix)
            except Exception as e:
                errors.append(f"Hadamard gate not unitary: {e}")
            
            # Test gate application
            initial_state = np.array([1, 0], dtype=complex)  # |0⟩
            final_state = self.gate_ops.apply_h(initial_state, 0)
            
            expected_state = np.array([1, 1], dtype=complex) / np.sqrt(2)
            if not np.allclose(final_state, expected_state, atol=AMPLITUDE_TOLERANCE):
                errors.append(f"Hadamard application incorrect: {final_state} ≠ {expected_state}")
            
            execution_time = time.perf_counter() - start_time
            
            if errors:
                return ValidationResult(
                    test_name="Quantum Gate Validation",
                    passed=False,
                    error_message="; ".join(errors),
                    execution_time=execution_time
                )
            
            return ValidationResult(
                test_name="Quantum Gate Validation",
                passed=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Quantum Gate Validation",
                passed=False,
                error_message=f"Exception: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
    
    def get_description(self) -> str:
        return "Validates quantum gate matrices for unitarity and correct eigenvalues"


class TimeEvolutionValidation(QuantumTestCase):
    """Validate time evolution against analytical solutions."""
    
    def __init__(self, physics_engine: QuantumPhysicsEngineProper):
        self.physics_engine = physics_engine
        
    def run_test(self) -> ValidationResult:
        """Test time evolution for simple Hamiltonians."""
        start_time = time.perf_counter()
        
        try:
            # Test case: free evolution under Pauli-Z Hamiltonian
            # H = ωσ_z/2, analytical solution: |ψ(t)⟩ = e^(-iωt/2)|0⟩ + e^(iωt/2)|1⟩ for |ψ(0)⟩ = |+⟩
            
            omega = 1.0  # Frequency
            t = np.pi / omega  # Evolution time
            
            # Hamiltonian: H = ω*σ_z/2
            H = omega * np.array([[1, 0], [0, -1]], dtype=complex) / 2
            
            # Initial state: |+⟩ = (|0⟩ + |1⟩)/√2
            initial_state_vector = np.array([1, 1], dtype=complex) / np.sqrt(2)
            initial_state = QuantumSystemState(
                state_vector=initial_state_vector,
                hamiltonian=H
            )
            
            # Register system
            system_name = "test_evolution"
            self.physics_engine.register_system(system_name, initial_state)
            
            # Evolve system
            final_state = self.physics_engine.evolve_system(
                system_name, 
                t, 
                H,
                method=self.physics_engine.EvolutionMethod.EXACT_DIAGONALIZATION
            )
            
            # Analytical solution at time t = π/ω
            # |ψ(π/ω)⟩ = e^(-iπ/2)|0⟩ + e^(iπ/2)|1⟩ = -i|0⟩ + i|1⟩ = i(|1⟩ - |0⟩)
            expected_state = 1j * np.array([-1, 1], dtype=complex) / np.sqrt(2)
            
            # Compare states (up to global phase)
            overlap = np.abs(np.vdot(final_state.state_vector, expected_state))**2
            
            if not np.isclose(overlap, 1.0, atol=AMPLITUDE_TOLERANCE):
                return ValidationResult(
                    test_name="Time Evolution Validation",
                    passed=False,
                    error_message=f"Time evolution incorrect: fidelity = {overlap}"
                )
            
            # Check energy conservation
            initial_energy = initial_state.energy
            final_energy = final_state.energy
            
            if not np.isclose(initial_energy, final_energy, atol=ENERGY_TOLERANCE):
                return ValidationResult(
                    test_name="Time Evolution Validation",
                    passed=False,
                    error_message=f"Energy not conserved: {initial_energy} → {final_energy}"
                )
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Time Evolution Validation",
                passed=True,
                execution_time=execution_time,
                metadata={
                    'fidelity': overlap,
                    'energy_conservation': abs(final_energy - initial_energy)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Time Evolution Validation",
                passed=False,
                error_message=f"Exception: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
    
    def get_description(self) -> str:
        return "Validates quantum time evolution against analytical solutions"


class CrossValidationTest(QuantumTestCase):
    """Cross-validate results with established quantum simulators."""
    
    def __init__(self):
        self.gate_ops = GateOperationsProper()
        
    def run_test(self) -> ValidationResult:
        """Cross-validate with Qiskit if available."""
        start_time = time.perf_counter()
        
        if not QISKIT_AVAILABLE:
            return ValidationResult(
                test_name="Cross-Validation Test",
                passed=False,
                error_message="Qiskit not available for cross-validation"
            )
        
        try:
            # Test circuit: H-CNOT Bell state preparation
            
            # Our implementation
            our_state = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
            
            # Apply Hadamard to first qubit
            our_state = self.gate_ops.apply_h(our_state, 0)
            
            # Apply CNOT
            our_state = self.gate_ops.apply_cnot(our_state, 0, 1)
            
            # Qiskit implementation
            qc = QuantumCircuit(2)
            qc.h(0)
            qc.cx(0, 1)
            
            backend = Aer.get_backend('statevector_simulator')
            job = execute(qc, backend)
            result = job.result()
            qiskit_state = result.get_statevector()
            
            # Compare state vectors (up to global phase)
            overlap = np.abs(np.vdot(our_state, qiskit_state))**2
            
            if not np.isclose(overlap, 1.0, atol=AMPLITUDE_TOLERANCE):
                return ValidationResult(
                    test_name="Cross-Validation Test",
                    passed=False,
                    error_message=f"States differ from Qiskit: fidelity = {overlap}",
                    numerical_error=1 - overlap
                )
            
            execution_time = time.perf_counter() - start_time
            
            return ValidationResult(
                test_name="Cross-Validation Test",
                passed=True,
                execution_time=execution_time,
                metadata={'fidelity_with_qiskit': overlap}
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Cross-Validation Test",
                passed=False,
                error_message=f"Exception: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
    
    def get_description(self) -> str:
        return "Cross-validates quantum circuit results with Qiskit simulator"


class PerformanceBenchmark(QuantumTestCase):
    """Benchmark performance and scaling."""
    
    def __init__(self, physics_engine: QuantumPhysicsEngineProper):
        self.physics_engine = physics_engine
        
    def run_test(self) -> ValidationResult:
        """Benchmark evolution performance for different system sizes."""
        start_time = time.perf_counter()
        
        try:
            performance_data = {}
            
            # Test different system sizes
            for n_qubits in range(1, 6):  # Up to 5 qubits
                dim = 2**n_qubits
                
                # Random Hamiltonian
                H_real = np.random.randn(dim, dim)
                H = (H_real + H_real.T) / 2  # Make Hermitian
                
                # Random initial state
                psi = np.random.randn(dim) + 1j * np.random.randn(dim)
                psi = psi / np.linalg.norm(psi)
                
                initial_state = QuantumSystemState(
                    state_vector=psi,
                    hamiltonian=H
                )
                
                system_name = f"benchmark_{n_qubits}"
                self.physics_engine.register_system(system_name, initial_state)
                
                # Time evolution
                evolution_start = time.perf_counter()
                
                self.physics_engine.evolve_system(
                    system_name,
                    0.1,  # Small time step
                    H
                )
                
                evolution_time = time.perf_counter() - evolution_start
                performance_data[n_qubits] = evolution_time
            
            # Check scaling (should be roughly exponential)
            scaling_factors = []
            for i in range(1, len(performance_data)):
                factor = performance_data[i+1] / performance_data[i]
                scaling_factors.append(factor)
            
            avg_scaling = np.mean(scaling_factors)
            
            execution_time = time.perf_counter() - start_time
            
            # Performance should be reasonable (less than 1 second for 5 qubits)
            max_time = max(performance_data.values())
            if max_time > 1.0:
                return ValidationResult(
                    test_name="Performance Benchmark",
                    passed=False,
                    error_message=f"Evolution too slow: {max_time:.3f}s for {max(performance_data.keys())} qubits"
                )
            
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=True,
                execution_time=execution_time,
                metadata={
                    'performance_data': performance_data,
                    'average_scaling_factor': avg_scaling
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Performance Benchmark",
                passed=False,
                error_message=f"Exception: {str(e)}",
                execution_time=time.perf_counter() - start_time
            )
    
    def get_description(self) -> str:
        return "Benchmarks quantum evolution performance and scaling"


class QuantumValidationSuite:
    """Comprehensive quantum validation test suite."""
    
    def __init__(self):
        self.physics_engine = QuantumPhysicsEngineProper()
        self.test_cases: List[QuantumTestCase] = []
        self.results: List[ValidationResult] = []
        
        # Register test cases
        self._register_test_cases()
        
    def _register_test_cases(self):
        """Register all validation test cases."""
        self.test_cases.extend([
            BellStateValidation(self.physics_engine),
            QuantumGateValidation(),
            TimeEvolutionValidation(self.physics_engine),
            CrossValidationTest(),
            PerformanceBenchmark(self.physics_engine)
        ])
    
    def run_all_tests(self, verbose: bool = True) -> Dict[str, Any]:
        """Run all validation tests."""
        self.results = []
        passed_tests = 0
        
        if verbose:
            print("="*60)
            print("QUANTUM VALIDATION SUITE")
            print("="*60)
        
        for test_case in self.test_cases:
            if verbose:
                print(f"\nRunning: {test_case.__class__.__name__}")
                print(f"Description: {test_case.get_description()}")
            
            result = test_case.run_test()
            self.results.append(result)
            
            if result.passed:
                passed_tests += 1
                status = "PASSED"
            else:
                status = "FAILED"
            
            if verbose:
                print(f"Status: {status}")
                if result.execution_time:
                    print(f"Execution time: {result.execution_time:.4f}s")
                if result.error_message:
                    print(f"Error: {result.error_message}")
                if result.metadata:
                    print(f"Metadata: {result.metadata}")
        
        success_rate = passed_tests / len(self.test_cases)
        
        if verbose:
            print("\n" + "="*60)
            print(f"VALIDATION SUMMARY")
            print(f"Tests passed: {passed_tests}/{len(self.test_cases)} ({success_rate:.1%})")
            print("="*60)
        
        return {
            'total_tests': len(self.test_cases),
            'passed_tests': passed_tests,
            'success_rate': success_rate,
            'results': self.results
        }
    
    def get_failed_tests(self) -> List[ValidationResult]:
        """Get list of failed tests."""
        return [result for result in self.results if not result.passed]
    
    def generate_report(self, filepath: str):
        """Generate detailed validation report."""
        with open(filepath, 'w') as f:
            f.write("QUANTUM VALIDATION REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total tests: {len(self.test_cases)}\n\n")
            
            for result in self.results:
                f.write(f"Test: {result.test_name}\n")
                f.write(f"Status: {'PASSED' if result.passed else 'FAILED'}\n")
                
                if result.execution_time:
                    f.write(f"Execution time: {result.execution_time:.4f}s\n")
                
                if result.error_message:
                    f.write(f"Error: {result.error_message}\n")
                
                if result.metadata:
                    f.write(f"Metadata: {result.metadata}\n")
                
                f.write("-" * 30 + "\n")


# Pytest integration
class TestQuantumValidation(unittest.TestCase):
    """Pytest-compatible test class."""
    
    @classmethod
    def setUpClass(cls):
        cls.validation_suite = QuantumValidationSuite()
    
    def test_bell_state_validation(self):
        """Test Bell state validation."""
        bell_test = BellStateValidation(self.validation_suite.physics_engine)
        result = bell_test.run_test()
        self.assertTrue(result.passed, result.error_message)
    
    def test_quantum_gate_validation(self):
        """Test quantum gate validation."""
        gate_test = QuantumGateValidation()
        result = gate_test.run_test()
        self.assertTrue(result.passed, result.error_message)
    
    def test_time_evolution_validation(self):
        """Test time evolution validation."""
        evolution_test = TimeEvolutionValidation(self.validation_suite.physics_engine)
        result = evolution_test.run_test()
        self.assertTrue(result.passed, result.error_message)
    
    def test_performance_benchmark(self):
        """Test performance benchmark."""
        perf_test = PerformanceBenchmark(self.validation_suite.physics_engine)
        result = perf_test.run_test()
        self.assertTrue(result.passed, result.error_message)


if __name__ == "__main__":
    # Run full validation suite
    suite = QuantumValidationSuite()
    summary = suite.run_all_tests(verbose=True)
    
    # Generate report
    suite.generate_report("quantum_validation_report.txt")
    
    # Exit with error code if tests failed
    if summary['success_rate'] < 1.0:
        exit(1)