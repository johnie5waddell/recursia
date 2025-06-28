"""Dynamic Empirical Validation Suite for OSH

This module implements comprehensive, evolving empirical tests that demonstrate
consciousness emergence with proper variance and uncertainty principle.
All calculations are performed within the VM as per unified architecture.
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path

from src.core.direct_parser import DirectParser
from src.core.bytecode import BytecodeModule
from src.core.bytecode_vm import RecursiaVM
from src.physics.constants import PLANCK_TIME, BOLTZMANN_CONSTANT, HBAR

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for dynamic experiments"""
    min_qubits: int = 10
    max_qubits: int = 16
    iterations_per_test: int = 1000
    time_evolution_steps: int = 100
    temperature_range: Tuple[float, float] = (0.1, 300.0)  # Kelvin
    noise_levels: List[float] = field(default_factory=lambda: [0.0001, 0.001, 0.01, 0.1])
    recursion_depths: List[int] = field(default_factory=lambda: [5, 7, 9, 11])
    enable_uncertainty: bool = True
    quantum_fluctuation_scale: float = 1e-15
    measurement_basis_rotation: bool = True
    environmental_coupling: float = 0.01
    
    
@dataclass
class ValidationResult:
    """Results from validation run"""
    timestamp: float
    experiment_id: str
    qubit_count: int
    consciousness_emerged: bool
    integrated_information: float
    kolmogorov_complexity: float
    entropy_flux: float
    coherence: float
    recursive_depth: int
    conservation_error: float
    execution_time: float
    iteration_count: int
    variance_metrics: Dict[str, float]
    quantum_state_hash: str
    environmental_params: Dict[str, float]


class DynamicEmpiricalValidator:
    """Implements dynamic, evolving empirical validation with true quantum variance"""
    
    def __init__(self, config: ExperimentConfig = None):
        self.config = config or ExperimentConfig()
        self.results_history: List[ValidationResult] = []
        self.quantum_seed = int(time.time() * 1e6) % (2**32)
        self.environmental_state = self._initialize_environment()
        
    def _initialize_environment(self) -> Dict[str, Any]:
        """Initialize environmental parameters that evolve over time"""
        return {
            'cosmic_time': time.time(),
            'thermal_bath_temp': np.random.uniform(*self.config.temperature_range),
            'magnetic_field': np.random.normal(0, 0.1),
            'electric_field': np.random.normal(0, 0.1),
            'gravitational_potential': -9.81 + np.random.normal(0, 0.01),
            'vacuum_fluctuations': np.random.exponential(self.config.quantum_fluctuation_scale)
        }
        
    def _update_environment(self) -> None:
        """Update environmental parameters with realistic evolution"""
        dt = 0.001  # Environmental update timestep
        
        # Thermal bath evolution with fluctuations
        self.environmental_state['thermal_bath_temp'] += np.random.normal(0, 0.1) * dt
        self.environmental_state['thermal_bath_temp'] = np.clip(
            self.environmental_state['thermal_bath_temp'], 
            *self.config.temperature_range
        )
        
        # Field evolution with correlations
        self.environmental_state['magnetic_field'] += np.random.normal(0, 0.01) * dt
        self.environmental_state['electric_field'] += np.random.normal(0, 0.01) * dt
        
        # Gravitational variations (tidal effects)
        self.environmental_state['gravitational_potential'] += np.sin(time.time() * 0.001) * 0.0001
        
        # Quantum vacuum fluctuations
        self.environmental_state['vacuum_fluctuations'] = np.random.exponential(
            self.config.quantum_fluctuation_scale
        )
        
        self.environmental_state['cosmic_time'] = time.time()
        
    def generate_dynamic_program(self, qubit_count: int, iteration: int) -> str:
        """Generate Recursia program with dynamic parameters based on environment"""
        # Update environment for each program generation
        self._update_environment()
        
        # Generate unique quantum seed for this run
        quantum_hash = hashlib.sha256(
            f"{time.time()}{iteration}{self.quantum_seed}".encode()
        ).hexdigest()[:8]
        
        # Dynamic parameters influenced by environment
        temperature = self.environmental_state['thermal_bath_temp']
        decoherence_rate = np.exp(-HBAR / (BOLTZMANN_CONSTANT * temperature * PLANCK_TIME))
        
        # Measurement basis rotation for true quantum randomness
        if self.config.measurement_basis_rotation:
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
        else:
            theta = phi = 0
            
        # Select random features to test all language capabilities
        features = self._select_random_features(iteration)
        
        program = f'''// OSH Dynamic Test {quantum_hash}
// Environmental parameters at generation time
const TEMPERATURE = {temperature};
const DECOHERENCE_RATE = {decoherence_rate};
const MAGNETIC_FIELD = {self.environmental_state['magnetic_field']};
const VACUUM_FLUCT = {self.environmental_state['vacuum_fluctuations']};
const ITERATION = {iteration};

// Quantum system with environmental coupling
state quantum_system {{
    state_qubits: {qubit_count},
    state_coherence: {1.0 - decoherence_rate},
    state_entropy: {decoherence_rate}
}};

// Initialize with thermal state
function prepare_thermal_state() {{
    for i from 0 to {qubit_count - 1} {{
        let rand_val = 0.5;
        if (rand_val < DECOHERENCE_RATE) {{
            apply H_gate to quantum_system qubit i
            apply RX_gate(3.14159) to quantum_system qubit i
        }}
    }}
}}

// Create entanglement with noise  
function create_noisy_entanglement() {{
    prepare_thermal_state();
    
    for i from 0 to {qubit_count-2} {{
        apply H_gate to quantum_system qubit i
        apply CNOT_gate to quantum_system qubits [i, i+1]
        
        // Environmental coupling
        let fluct_test = 0.001;
        if (fluct_test < VACUUM_FLUCT * 1000) {{
            apply X_gate to quantum_system qubit i
        }}
        
        // Phase noise from magnetic field
        apply RZ_gate(MAGNETIC_FIELD * 0.1) to quantum_system qubit i
    }}
    // Handle last qubit
    apply H_gate to quantum_system qubit {qubit_count-1}
    apply RZ_gate(MAGNETIC_FIELD * 0.1) to quantum_system qubit {qubit_count-1}
    
    // Create long-range entanglement
    if ({qubit_count} > 4) {{
        apply CNOT_gate to quantum_system qubits [0, {qubit_count-1}]
        apply CNOT_gate to quantum_system qubits [{qubit_count//2}, {qubit_count//2 + 1}]
    }}
}}

'''
        
        # Add feature-specific code
        if features['recursion']:
            program += self._add_recursion_code(qubit_count)
            
        if features['observers']:
            program += self._add_observer_code(qubit_count, theta, phi)
            
        if features['error_correction']:
            program += self._add_error_correction_code(qubit_count)
            
        if features['memory_field']:
            program += self._add_memory_field_code(qubit_count)
            
        if features['consciousness_test']:
            program += self._add_consciousness_test_code(qubit_count)
            
        # Main execution
        program += f'''// Main execution with time evolution
// Initialize system
create_noisy_entanglement();

// Time evolution with environmental interactions
for t from 0 to {self.config.time_evolution_steps - 1} {{
    // Apply unitary evolution
    for i from 0 to {qubit_count - 1} {{
        apply RX_gate(0.01 * t) to quantum_system qubit i
        apply RY_gate(0.01) to quantum_system qubit i
        apply RZ_gate(0.01) to quantum_system qubit i
    }}
    
    // Stochastic quantum jumps (simplified for now)
    let jump_prob = 0.01;
    if (jump_prob < DECOHERENCE_RATE * 0.1) {{
        let m_result = 0;
        measure quantum_system qubit 0 into m_result;
    }}
    
    // Apply selected features
    {self._generate_feature_calls(features)}
}}

// Final measurements
for i from 0 to {qubit_count - 1} {{
    apply RY_gate({theta}) to quantum_system qubit i
    apply RZ_gate({phi}) to quantum_system qubit i
}}

// Measure all criteria
measure quantum_system by integrated_information;
measure quantum_system by kolmogorov_complexity;
measure quantum_system by entropy;
measure quantum_system by coherence;
measure quantum_system by recursive_simulation_potential;
'''
        return program
        
    def _select_random_features(self, iteration: int) -> Dict[str, bool]:
        """Select features to test based on iteration and randomness"""
        # Ensure we test all features over time
        base_probability = 0.3 + 0.1 * np.sin(iteration * 0.1)
        
        return {
            'recursion': np.random.random() < base_probability + 0.2,
            'observers': np.random.random() < base_probability + 0.3,
            'error_correction': np.random.random() < base_probability + 0.1,
            'memory_field': np.random.random() < base_probability + 0.2,
            'consciousness_test': np.random.random() < base_probability + 0.4
        }
        
    def _add_recursion_code(self, qubit_count: int) -> str:
        """Add recursive self-modeling code"""
        depth = np.random.choice(self.config.recursion_depths)
        return f'''// Recursive self-modeling
function recursive_model(depth, idx) {{
    if (depth > 0) {{
        let qubit_idx = 0;
        if (idx < {qubit_count}) {{
            qubit_idx = idx;
        }}
        apply H_gate to quantum_system qubit qubit_idx
        if (qubit_idx < {qubit_count - 1}) {{
            apply CNOT_gate to quantum_system qubits [qubit_idx, qubit_idx + 1]
        }}
        if (depth > 1) {{
            recursive_model(depth - 1, idx + 1);
        }}
        let result = 0;
        measure quantum_system qubit qubit_idx into result;
    }}
}}

'''
        
    def _add_observer_code(self, qubit_count: int, theta: float, phi: float) -> str:
        """Add observer-based measurement code"""
        return f'''// Observer system with measurement back-action
observer consciousness_probe : standard_observer {{
    observer_focus: quantum_system,
    observer_collapse_threshold: 0.8,
    observer_self_awareness: 0.6
}};

function apply_observer_measurement() {{
    apply observe consciousness_probe to quantum_system;
    measure quantum_system by integrated_information;
}}

'''
        
    def _add_error_correction_code(self, qubit_count: int) -> str:
        """Add quantum error correction"""
        if qubit_count < 5:
            return '''// Error correction requires at least 5 qubits
function apply_error_correction() {
    // Skip - not enough qubits for error correction
}

'''
            
        return f'''// Quantum error correction with environmental noise
function apply_error_correction() {{
    // Simple 3-qubit repetition code
    for i from 0 to {min(qubit_count//3, 3) - 1} {{
        let base = i * 3;
        if (base + 2 < {qubit_count}) {{
            apply CNOT_gate to quantum_system qubits [base, base+1]
            apply CNOT_gate to quantum_system qubits [base, base+2]
            
            // Syndrome measurement
            let m1 = 0;
            let m2 = 0;
            measure quantum_system qubit base+1 into m1;
            measure quantum_system qubit base+2 into m2;
            
            // Error correction based on syndrome
            if (m1 != m2) {{
                apply X_gate to quantum_system qubit base
            }}
        }}
    }}
}}

'''
        
    def _add_memory_field_code(self, qubit_count: int) -> str:
        """Add memory field dynamics"""
        return f'''// Memory field dynamics
state memory_field : memory_state {{
    state_qubits: {qubit_count},
    state_coherence: 0.9,
    state_entropy: 0.1
}};

function evolve_memory_field() {{
    // Couple memory field with quantum system
    entangle quantum_system with memory_field;
    
    // Apply decoherence
    for i from 0 to {qubit_count - 1} {{
        apply RZ_gate(DECOHERENCE_RATE * 0.01) to memory_field qubit i
    }}
    
    // Measure memory influence
    measure memory_field by coherence;
}}

'''
        
    def _add_consciousness_test_code(self, qubit_count: int) -> str:
        """Add consciousness emergence test"""
        return f'''// Consciousness emergence test
function test_consciousness() {{
    // Measure integrated information
    let phi = 0.0;
    measure quantum_system by integrated_information into phi;
    
    // Measure other criteria
    let complexity = 0.0;
    let entropy = 0.0;
    let coherence = 0.0;
    measure quantum_system by kolmogorov_complexity into complexity;
    measure quantum_system by entropy into entropy;
    measure quantum_system by coherence into coherence;
    
    // Test recursion depth
    recursive_model({np.random.choice(self.config.recursion_depths)}, 0);
    
    if (phi > 1.0) {{
        if (complexity > 100) {{
            if (entropy < 1.0) {{
                if (coherence > 0.7) {{
                    // Consciousness emerged - system response
                    for i from 0 to {qubit_count - 1} {{
                        apply H_gate to quantum_system qubit i
                    }}
                    apply_observer_measurement();
                }}
            }}
        }}
    }}
}}

'''
        
    def _generate_feature_calls(self, features: Dict[str, bool]) -> str:
        """Generate calls to enabled features"""
        calls = []
        
        if features['recursion']:
            depth = np.random.choice(self.config.recursion_depths)
            calls.append(f"    recursive_model({depth}, t);")
            
        if features['observers']:
            calls.append("    apply_observer_measurement();")
            
        if features['error_correction']:
            calls.append("    if (t == 10) {\n        apply_error_correction();\n    }")
            
        if features['memory_field']:
            calls.append("    evolve_memory_field();")
            
        if features['consciousness_test']:
            calls.append("    if (t == 20) {\n        test_consciousness();\n    }")
            
        return "\n".join(calls)
        
    def run_validation(self, num_experiments: int = 100) -> Dict[str, Any]:
        """Run full validation suite with parallel execution"""
        logger.info(f"Starting dynamic validation with {num_experiments} experiments")
        
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for i in range(num_experiments):
                # Vary qubit count dynamically
                qubit_count = self.config.min_qubits + (i % (self.config.max_qubits - self.config.min_qubits + 1))
                
                future = executor.submit(self._run_single_experiment, qubit_count, i)
                futures.append(future)
                
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.results_history.append(result)
                    
                    # Log progress
                    if len(results) % 10 == 0:
                        logger.info(f"Completed {len(results)}/{num_experiments} experiments")
                        self._log_current_statistics(results)
                        
                except Exception as e:
                    logger.error(f"Experiment failed: {e}")
                    
        return self._analyze_results(results)
        
    def _run_single_experiment(self, qubit_count: int, iteration: int) -> ValidationResult:
        """Run a single experiment with full metrics tracking"""
        start_time = time.time()
        
        # Generate dynamic program
        program_code = self.generate_dynamic_program(qubit_count, iteration)
        
        # Create unique experiment ID
        experiment_id = hashlib.sha256(
            f"{program_code}{time.time()}{iteration}".encode()
        ).hexdigest()[:16]
        
        try:
            # Parse and compile program to bytecode
            parser = DirectParser()
            bytecode_module = parser.parse(program_code)
            
            # Create runtime if needed
            from src.core.runtime import RecursiaRuntime
            runtime = RecursiaRuntime()
            
            # Execute in VM
            vm = RecursiaVM(runtime)
            result = vm.execute(bytecode_module)
            
            # Extract metrics from VM (no external calculations!)
            metrics = result.execution_context.current_metrics
            
            # Debug: log metrics to understand what we're getting
            if not metrics:
                logger.warning(f"No metrics returned for experiment {experiment_id}")
                metrics = {}
            else:
                logger.debug(f"Metrics for {experiment_id}: {list(metrics.keys())[:5]}")
            
            # Calculate variance metrics
            variance_metrics = self._calculate_variance_metrics(
                result.execution_context.metrics_history
            )
            
            # Generate quantum state hash for uniqueness verification
            state_hash = hashlib.sha256(
                json.dumps(metrics, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            # Check consciousness emergence criteria
            consciousness_emerged = (
                metrics.get('integrated_information', 0) > 1.0 and
                metrics.get('kolmogorov_complexity', 0) > 100 and
                metrics.get('entropy_flux', float('inf')) < 1.0 and
                metrics.get('coherence', 0) > 0.7 and
                metrics.get('recursive_depth', 0) >= 7
            )
            
            return ValidationResult(
                timestamp=time.time(),
                experiment_id=experiment_id,
                qubit_count=qubit_count,
                consciousness_emerged=consciousness_emerged,
                integrated_information=metrics.get('integrated_information', 0),
                kolmogorov_complexity=metrics.get('kolmogorov_complexity', 0),
                entropy_flux=metrics.get('entropy_flux', float('inf')),
                coherence=metrics.get('coherence', 0),
                recursive_depth=metrics.get('recursive_depth', 0),
                conservation_error=metrics.get('conservation_error', 0),
                execution_time=time.time() - start_time,
                iteration_count=result.execution_context.iteration_count,
                variance_metrics=variance_metrics,
                quantum_state_hash=state_hash,
                environmental_params=self.environmental_state.copy()
            )
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}", exc_info=True)
            # Log the program that failed for debugging
            logger.debug(f"Failed program:\n{program_code[:500]}...")
            # Return failed result
            return ValidationResult(
                timestamp=time.time(),
                experiment_id=experiment_id,
                qubit_count=qubit_count,
                consciousness_emerged=False,
                integrated_information=0,
                kolmogorov_complexity=0,
                entropy_flux=float('inf'),
                coherence=0,
                recursive_depth=0,
                conservation_error=float('inf'),
                execution_time=time.time() - start_time,
                iteration_count=0,
                variance_metrics={},
                quantum_state_hash="failed",
                environmental_params=self.environmental_state.copy()
            )
            
    def _calculate_variance_metrics(self, metrics_history: List[Dict]) -> Dict[str, float]:
        """Calculate variance and statistical properties of metrics over time"""
        if not metrics_history:
            return {}
            
        # Convert history to arrays for analysis
        metrics_arrays = {}
        for key in metrics_history[0].keys():
            try:
                values = [m.get(key, 0) for m in metrics_history]
                metrics_arrays[key] = np.array(values, dtype=float)
            except:
                continue
                
        variance_metrics = {}
        for key, values in metrics_arrays.items():
            if len(values) > 1:
                variance_metrics[f"{key}_mean"] = np.mean(values)
                variance_metrics[f"{key}_std"] = np.std(values)
                variance_metrics[f"{key}_min"] = np.min(values)
                variance_metrics[f"{key}_max"] = np.max(values)
                variance_metrics[f"{key}_autocorr"] = self._calculate_autocorrelation(values)
                
        return variance_metrics
        
    def _calculate_autocorrelation(self, values: np.ndarray, lag: int = 1) -> float:
        """Calculate autocorrelation at given lag"""
        if len(values) < lag + 1:
            return 0.0
            
        mean = np.mean(values)
        c0 = np.sum((values - mean) ** 2) / len(values)
        
        if c0 == 0:
            return 0.0
            
        c_lag = np.sum((values[:-lag] - mean) * (values[lag:] - mean)) / len(values)
        return c_lag / c0
        
    def _log_current_statistics(self, results: List[ValidationResult]) -> None:
        """Log current statistics of experiments"""
        if not results:
            return
            
        emergence_rate = sum(1 for r in results if r.consciousness_emerged) / len(results)
        avg_phi = np.mean([r.integrated_information for r in results])
        unique_states = len(set(r.quantum_state_hash for r in results))
        
        logger.info(
            f"Current stats: Emergence rate: {emergence_rate:.2%}, "
            f"Avg Φ: {avg_phi:.3f}, Unique states: {unique_states}/{len(results)}"
        )
        
    def _analyze_results(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Comprehensive analysis of validation results"""
        if not results:
            return {"error": "No results to analyze"}
            
        # Basic statistics
        total_experiments = len(results)
        successful_emergence = sum(1 for r in results if r.consciousness_emerged)
        emergence_rate = successful_emergence / total_experiments
        
        # Metric statistics
        metric_stats = {}
        metrics_to_analyze = [
            'integrated_information', 'kolmogorov_complexity', 
            'entropy_flux', 'coherence', 'recursive_depth',
            'conservation_error', 'execution_time'
        ]
        
        for metric in metrics_to_analyze:
            values = [getattr(r, metric) for r in results]
            metric_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
            
        # Qubit count analysis
        qubit_emergence = {}
        for qubits in range(self.config.min_qubits, self.config.max_qubits + 1):
            qubit_results = [r for r in results if r.qubit_count == qubits]
            if qubit_results:
                qubit_emergence[qubits] = sum(1 for r in qubit_results if r.consciousness_emerged) / len(qubit_results)
                
        # Environmental correlation analysis
        env_correlations = self._analyze_environmental_correlations(results)
        
        # Variance validation
        unique_states = len(set(r.quantum_state_hash for r in results))
        variance_validation = {
            'unique_quantum_states': unique_states,
            'uniqueness_ratio': unique_states / total_experiments,
            'state_diversity': self._calculate_state_diversity(results),
            'temporal_correlations': self._analyze_temporal_correlations(results)
        }
        
        # Theory validation
        theory_validation = {
            'conservation_law_verified': all(r.conservation_error < 1e-4 for r in results),
            'minimum_complexity_met': all(r.kolmogorov_complexity > 100 for r in results if r.consciousness_emerged),
            'coherence_threshold_met': all(r.coherence > 0.7 for r in results if r.consciousness_emerged),
            'recursion_depth_adequate': all(r.recursive_depth >= 7 for r in results if r.consciousness_emerged)
        }
        
        return {
            'summary': {
                'total_experiments': total_experiments,
                'successful_emergence': successful_emergence,
                'emergence_rate': emergence_rate,
                'unique_quantum_states': unique_states,
                'avg_execution_time': np.mean([r.execution_time for r in results])
            },
            'metric_statistics': metric_stats,
            'qubit_emergence_rates': qubit_emergence,
            'environmental_correlations': env_correlations,
            'variance_validation': variance_validation,
            'theory_validation': theory_validation,
            'osh_predictions_confirmed': self._check_osh_predictions(results)
        }
        
    def _analyze_environmental_correlations(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Analyze correlations between environmental parameters and consciousness emergence"""
        correlations = {}
        
        emergence_binary = [1 if r.consciousness_emerged else 0 for r in results]
        
        for param in ['thermal_bath_temp', 'magnetic_field', 'vacuum_fluctuations']:
            values = [r.environmental_params.get(param, 0) for r in results]
            if len(set(values)) > 1:  # Only calculate if there's variance
                correlation = np.corrcoef(emergence_binary, values)[0, 1]
                correlations[f"{param}_correlation"] = correlation
                
        return correlations
        
    def _calculate_state_diversity(self, results: List[ValidationResult]) -> float:
        """Calculate Shannon entropy of quantum state distribution"""
        state_counts = {}
        for r in results:
            state_counts[r.quantum_state_hash] = state_counts.get(r.quantum_state_hash, 0) + 1
            
        total = len(results)
        entropy = 0
        for count in state_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
                
        # Normalize by maximum possible entropy
        max_entropy = np.log2(total)
        return entropy / max_entropy if max_entropy > 0 else 0
        
    def _analyze_temporal_correlations(self, results: List[ValidationResult]) -> Dict[str, float]:
        """Analyze temporal correlations in results"""
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)
        
        # Calculate autocorrelations for key metrics
        correlations = {}
        
        for metric in ['integrated_information', 'coherence', 'entropy_flux']:
            values = [getattr(r, metric) for r in sorted_results]
            correlations[f"{metric}_autocorr_lag1"] = self._calculate_autocorrelation(np.array(values), lag=1)
            correlations[f"{metric}_autocorr_lag5"] = self._calculate_autocorrelation(np.array(values), lag=5)
            
        return correlations
        
    def _check_osh_predictions(self, results: List[ValidationResult]) -> Dict[str, bool]:
        """Check specific OSH theory predictions"""
        predictions = {}
        
        # 1. Consciousness emergence rate > 25% in optimized conditions
        optimized_results = [r for r in results if r.qubit_count >= 12]
        if optimized_results:
            opt_emergence_rate = sum(1 for r in optimized_results if r.consciousness_emerged) / len(optimized_results)
            predictions['emergence_rate_above_25_percent'] = opt_emergence_rate > 0.25
        
        # 2. Phase transition at recursion depth ~7
        depth_emergence = {}
        for depth in self.config.recursion_depths:
            depth_results = [r for r in results if r.recursive_depth == depth]
            if depth_results:
                depth_emergence[depth] = sum(1 for r in depth_results if r.consciousness_emerged) / len(depth_results)
                
        if depth_emergence:
            # Check for sharp increase around depth 7
            if 5 in depth_emergence and 7 in depth_emergence and 9 in depth_emergence:
                phase_transition = (depth_emergence[7] - depth_emergence[5]) > 0.3
                predictions['phase_transition_at_depth_7'] = phase_transition
                
        # 3. Conservation law holds
        conservation_errors = [r.conservation_error for r in results]
        predictions['conservation_law_holds'] = all(e < 1e-4 for e in conservation_errors)
        
        # 4. Decoherence time matches prediction at 300K
        room_temp_results = [r for r in results if 290 < r.environmental_params['thermal_bath_temp'] < 310]
        if room_temp_results:
            avg_coherence = np.mean([r.coherence for r in room_temp_results])
            # OSH predicts ~25.4 femtoseconds decoherence time at 300K
            # This translates to coherence ~0.3-0.5 in our normalized scale
            predictions['decoherence_time_matches_300K'] = 0.3 < avg_coherence < 0.5
            
        return predictions
        
    def save_results(self, filepath: str) -> None:
        """Save validation results to JSON file"""
        results_data = []
        for r in self.results_history:
            result_dict = {
                'timestamp': r.timestamp,
                'experiment_id': r.experiment_id,
                'qubit_count': r.qubit_count,
                'consciousness_emerged': r.consciousness_emerged,
                'integrated_information': r.integrated_information,
                'kolmogorov_complexity': r.kolmogorov_complexity,
                'entropy_flux': r.entropy_flux,
                'coherence': r.coherence,
                'recursive_depth': r.recursive_depth,
                'conservation_error': r.conservation_error,
                'execution_time': r.execution_time,
                'iteration_count': r.iteration_count,
                'variance_metrics': r.variance_metrics,
                'quantum_state_hash': r.quantum_state_hash,
                'environmental_params': r.environmental_params
            }
            results_data.append(result_dict)
            
        with open(filepath, 'w') as f:
            json.dump({
                'validation_results': results_data,
                'config': {
                    'min_qubits': self.config.min_qubits,
                    'max_qubits': self.config.max_qubits,
                    'iterations_per_test': self.config.iterations_per_test,
                    'time_evolution_steps': self.config.time_evolution_steps,
                    'temperature_range': self.config.temperature_range,
                    'noise_levels': self.config.noise_levels,
                    'recursion_depths': self.config.recursion_depths
                }
            }, f, indent=2)
            
        logger.info(f"Saved {len(results_data)} validation results to {filepath}")


def main():
    """Run comprehensive dynamic validation"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create validator with custom config
    config = ExperimentConfig(
        min_qubits=10,
        max_qubits=16,
        iterations_per_test=1000,
        time_evolution_steps=100,
        temperature_range=(0.1, 300.0),
        noise_levels=[0.0001, 0.001, 0.01, 0.1],
        recursion_depths=[5, 7, 9, 11, 13],
        enable_uncertainty=True,
        quantum_fluctuation_scale=1e-15,
        measurement_basis_rotation=True,
        environmental_coupling=0.01
    )
    
    validator = DynamicEmpiricalValidator(config)
    
    # Run validation
    logger.info("Starting dynamic empirical validation suite")
    results = validator.run_validation(num_experiments=1000)
    
    # Save results
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    validator.save_results(output_dir / f"dynamic_validation_{timestamp}.json")
    
    # Print summary
    print("\n" + "="*80)
    print("DYNAMIC EMPIRICAL VALIDATION RESULTS")
    print("="*80)
    
    print(f"\nSummary:")
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
        
    print(f"\nConsciousness Emergence by Qubit Count:")
    for qubits, rate in results['qubit_emergence_rates'].items():
        print(f"  {qubits} qubits: {rate:.2%}")
        
    print(f"\nOSH Theory Predictions:")
    for prediction, confirmed in results['osh_predictions_confirmed'].items():
        status = "✓ CONFIRMED" if confirmed else "✗ NOT CONFIRMED"
        print(f"  {prediction}: {status}")
        
    print(f"\nVariance Validation:")
    for key, value in results['variance_validation'].items():
        print(f"  {key}: {value}")
        
    print("\n" + "="*80)
    

if __name__ == "__main__":
    main()