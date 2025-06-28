"""
Performance benchmarks for Recursia system components.
Tests scalability, memory usage, and execution speed.
"""

import pytest
import numpy as np
import time
import psutil
import gc
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

# Import system components
from src.physics.physics_engine_proper import PhysicsEngine
from src.physics.quantum_state import QuantumState
from src.physics.observer import Observer
from src.physics.memory_field_proper import MemoryField
from src.core.compiler import RecursiaCompiler
from src.core.interpreter import RecursiaInterpreter
from src.visualization.quantum_visualization_engine import QuantumVisualizationEngine

@dataclass
class BenchmarkResult:
    """Benchmark result data structure"""
    name: str
    execution_time: float
    memory_usage: float
    peak_memory: float
    operations_per_second: float
    success: bool
    error_message: str = ""

class PerformanceProfiler:
    """Performance profiling utilities"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_memory = 0
        self.peak_memory = 0
        
    def start_profiling(self):
        """Start performance profiling"""
        gc.collect()  # Clean up before measurement
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        
    def update_peak_memory(self):
        """Update peak memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
            
    def get_current_memory(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

class TestQuantumStatePerformance:
    """Performance tests for quantum state operations"""
    
    def test_quantum_state_scaling(self):
        """Test quantum state performance scaling with qubit count"""
        profiler = PerformanceProfiler()
        results = []
        
        qubit_counts = [1, 2, 4, 6, 8, 10, 12]  # Up to 4096-dimensional states
        
        for n_qubits in qubit_counts:
            profiler.start_profiling()
            start_time = time.time()
            
            try:
                # Create quantum state
                state = QuantumState(n_qubits)
                
                # Initialize with random amplitudes
                state_dim = 2 ** n_qubits
                amplitudes = np.random.complex128(size=state_dim)
                amplitudes = amplitudes / np.linalg.norm(amplitudes)
                state.set_amplitudes(amplitudes)
                
                # Perform operations
                for _ in range(100):
                    coherence = state.calculate_coherence()
                    entropy = state.calculate_entanglement_entropy()
                    profiler.update_peak_memory()
                
                execution_time = time.time() - start_time
                
                result = BenchmarkResult(
                    name=f"quantum_state_{n_qubits}_qubits",
                    execution_time=execution_time,
                    memory_usage=profiler.get_current_memory() - profiler.start_memory,
                    peak_memory=profiler.peak_memory - profiler.start_memory,
                    operations_per_second=200 / execution_time,  # 100 coherence + 100 entropy
                    success=True
                )
                
                results.append(result)
                
                # Memory should scale exponentially with qubit count
                expected_memory_mb = (2 ** n_qubits * 16) / (1024 * 1024)  # 16 bytes per complex number
                assert result.memory_usage < expected_memory_mb * 5, f"Memory usage too high for {n_qubits} qubits"
                
                # Execution time should be reasonable
                assert execution_time < 10.0, f"Execution too slow for {n_qubits} qubits: {execution_time:.2f}s"
                
            except Exception as e:
                result = BenchmarkResult(
                    name=f"quantum_state_{n_qubits}_qubits",
                    execution_time=0,
                    memory_usage=0,
                    peak_memory=0,
                    operations_per_second=0,
                    success=False,
                    error_message=str(e)
                )
                results.append(result)
        
        # Verify scaling behavior
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 5, "Too many benchmark failures"
        
        # Log results for analysis
        for result in results:
            logging.info(f"Benchmark: {result.name}, Time: {result.execution_time:.3f}s, "
                        f"Memory: {result.memory_usage:.1f}MB, OPS: {result.operations_per_second:.1f}")

    def test_massive_superposition_performance(self):
        """Test performance with large superposition states"""
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        
        # Create 10-qubit state (1024 dimensions)
        state = QuantumState(10)
        
        # Create equal superposition (computationally intensive)
        start_time = time.time()
        
        state_dim = 2 ** 10
        amplitudes = np.ones(state_dim, dtype=complex) / np.sqrt(state_dim)
        state.set_amplitudes(amplitudes)
        
        # Perform intensive calculations
        operations_count = 0
        for _ in range(50):
            coherence = state.calculate_coherence()
            operations_count += 1
            
            entropy = state.calculate_entanglement_entropy()
            operations_count += 1
            
            # Simulate decoherence
            state.apply_decoherence(0.01, 0.1)
            operations_count += 1
            
            profiler.update_peak_memory()
        
        execution_time = time.time() - start_time
        
        # Performance requirements
        assert execution_time < 30.0, f"Massive superposition test too slow: {execution_time:.2f}s"
        assert profiler.peak_memory < 200.0, f"Memory usage too high: {profiler.peak_memory:.1f}MB"
        
        ops_per_second = operations_count / execution_time
        assert ops_per_second > 1.0, f"Operation rate too low: {ops_per_second:.2f} ops/s"

class TestEntanglementPerformance:
    """Performance tests for entanglement operations"""
    
    def test_entanglement_network_scaling(self):
        """Test entanglement network performance with many states"""
        profiler = PerformanceProfiler()
        physics_engine = PhysicsEngine()
        
        state_counts = [5, 10, 20, 30, 50]
        
        for n_states in state_counts:
            profiler.start_profiling()
            start_time = time.time()
            
            # Create quantum states
            states = []
            for i in range(n_states):
                state = QuantumState(1)  # Single qubits for scalability
                state.name = f"state_{i}"
                # Random superposition
                alpha = np.random.random()
                beta = np.sqrt(1 - alpha**2)
                state.set_amplitudes(np.array([alpha, beta], dtype=complex))
                states.append(state)
                physics_engine.add_quantum_state(state)
            
            # Create entanglement network (all-to-all)
            entanglement_count = 0
            for i in range(n_states):
                for j in range(i+1, min(i+6, n_states)):  # Limit connections to prevent explosion
                    physics_engine.create_entanglement(states[i], states[j], strength=0.8)
                    entanglement_count += 1
                    profiler.update_peak_memory()
            
            # Calculate network properties
            for _ in range(10):
                total_entanglement = physics_engine.calculate_total_entanglement()
                network_coherence = physics_engine.calculate_network_coherence()
                profiler.update_peak_memory()
            
            execution_time = time.time() - start_time
            
            # Performance assertions
            assert execution_time < 20.0, f"Entanglement network test too slow for {n_states} states"
            assert profiler.peak_memory < 500.0, f"Memory usage too high: {profiler.peak_memory:.1f}MB"
            
            # Clean up for next iteration
            physics_engine.clear_all_states()
            gc.collect()

    def test_entanglement_evolution_performance(self):
        """Test performance of entanglement evolution over time"""
        profiler = PerformanceProfiler()
        physics_engine = PhysicsEngine()
        
        # Create entangled system
        state1 = QuantumState(2)
        state2 = QuantumState(2)
        
        # Bell state
        bell_amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        state1.set_amplitudes(bell_amplitudes[:2])
        state2.set_amplitudes(bell_amplitudes[2:])
        
        physics_engine.add_quantum_state(state1)
        physics_engine.add_quantum_state(state2)
        physics_engine.create_entanglement(state1, state2, strength=1.0)
        
        profiler.start_profiling()
        start_time = time.time()
        
        # Evolve system for many time steps
        time_steps = 1000
        dt = 0.01
        
        for step in range(time_steps):
            physics_engine.evolve_system(dt)
            
            if step % 100 == 0:  # Periodic measurements
                entanglement_strength = physics_engine.get_entanglement_strength(state1, state2)
                profiler.update_peak_memory()
        
        execution_time = time.time() - start_time
        
        # Performance requirements
        assert execution_time < 15.0, f"Entanglement evolution too slow: {execution_time:.2f}s"
        
        steps_per_second = time_steps / execution_time
        assert steps_per_second > 50.0, f"Evolution rate too low: {steps_per_second:.1f} steps/s"

class TestObserverPerformance:
    """Performance tests for observer systems"""
    
    def test_multi_observer_scaling(self):
        """Test performance with many observers"""
        profiler = PerformanceProfiler()
        physics_engine = PhysicsEngine()
        
        observer_counts = [10, 25, 50, 100, 200]
        
        for n_observers in observer_counts:
            profiler.start_profiling()
            start_time = time.time()
            
            # Create quantum states
            states = []
            for i in range(10):  # Fixed number of states
                state = QuantumState(1)
                state.set_amplitudes(np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex))
                states.append(state)
                physics_engine.add_quantum_state(state)
            
            # Create many observers
            observers = []
            for i in range(n_observers):
                observer = Observer(
                    name=f"observer_{i}",
                    observer_type="conscious" if i % 2 == 0 else "environmental",
                    self_awareness=np.random.random(),
                    measurement_strength=np.random.random(),
                    collapse_threshold=np.random.random()
                )
                observers.append(observer)
                physics_engine.add_observer(observer)
                profiler.update_peak_memory()
            
            # Simulate observer interactions
            for _ in range(100):
                for observer in observers[:min(len(observers), 20)]:  # Limit active observers
                    for state in states[:5]:  # Limit interactions
                        interaction_strength = physics_engine.calculate_observer_interaction(observer, state)
                profiler.update_peak_memory()
            
            execution_time = time.time() - start_time
            
            # Performance requirements
            assert execution_time < 30.0, f"Multi-observer test too slow for {n_observers} observers"
            
            # Memory should scale linearly with observer count
            expected_memory = n_observers * 0.01  # ~10KB per observer
            assert profiler.peak_memory < expected_memory + 50.0, "Memory usage scaling poorly"
            
            physics_engine.clear_all_observers()
            physics_engine.clear_all_states()
            gc.collect()

    def test_consciousness_field_calculation(self):
        """Test performance of consciousness field calculations"""
        profiler = PerformanceProfiler()
        physics_engine = PhysicsEngine()
        
        # Create memory field
        memory_field = MemoryField(dimensions=(64, 64, 64), resolution=0.1)
        physics_engine.set_memory_field(memory_field)
        
        # Create observers with varying consciousness
        observers = []
        for i in range(20):
            observer = Observer(
                name=f"conscious_entity_{i}",
                observer_type="conscious",
                self_awareness=i / 20.0,
                measurement_strength=0.8,
                position=(i - 10, 0, 0)
            )
            observers.append(observer)
            physics_engine.add_observer(observer)
        
        profiler.start_profiling()
        start_time = time.time()
        
        # Calculate consciousness field evolution
        for time_step in range(100):
            # Update consciousness field
            for observer in observers:
                field_influence = physics_engine.calculate_consciousness_field_influence(observer)
                memory_field.apply_consciousness_influence(observer.position, field_influence)
            
            # Calculate emergence metrics
            emergence_index = physics_engine.calculate_emergence_index()
            rsp_value = physics_engine.calculate_rsp()
            
            profiler.update_peak_memory()
        
        execution_time = time.time() - start_time
        
        # Performance requirements
        assert execution_time < 25.0, f"Consciousness field calculation too slow: {execution_time:.2f}s"
        assert profiler.peak_memory < 150.0, f"Memory usage too high: {profiler.peak_memory:.1f}MB"

class TestCompilationPerformance:
    """Performance tests for compilation pipeline"""
    
    def test_large_program_compilation(self):
        """Test compilation performance for large programs"""
        # Generate large Recursia program
        large_program = self._generate_large_program(1000)  # 1000 quantum states
        
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        start_time = time.time()
        
        compiler = RecursiaCompiler()
        
        # Compilation pipeline
        tokens = compiler.lexer.tokenize(large_program)
        ast = compiler.parser.parse(tokens)
        semantic_result = compiler.semantic_analyzer.analyze(ast)
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        
        compilation_time = time.time() - start_time
        
        # Performance requirements
        assert compilation_time < 60.0, f"Large program compilation too slow: {compilation_time:.2f}s"
        assert compiled_result.success, "Large program compilation failed"
        assert profiler.peak_memory < 300.0, f"Compilation memory usage too high: {profiler.peak_memory:.1f}MB"
        
        # Compilation rate
        tokens_per_second = len(tokens) / compilation_time
        assert tokens_per_second > 100.0, f"Token processing rate too low: {tokens_per_second:.1f}/s"
    
    def _generate_large_program(self, n_states: int) -> str:
        """Generate large Recursia program for testing"""
        program_parts = [
            "// Large-scale quantum simulation",
            "memory_field large_memory {",
            "    dimensions: [128, 128, 128]",
            "    resolution: 0.05",
            "}",
            ""
        ]
        
        # Add many quantum states
        for i in range(n_states):
            program_parts.extend([
                f"quantum_state state_{i} {{",
                f"    qubits: {1 + (i % 3)}",
                f"    coherence: {0.5 + (i % 100) / 200.0}",
                f"    position: [{i % 10}, {(i // 10) % 10}, {(i // 100) % 10}]",
                "}",
                ""
            ])
        
        # Add observers
        for i in range(min(n_states // 10, 50)):
            program_parts.extend([
                f"observer obs_{i} {{",
                f"    type: {'conscious' if i % 2 == 0 else 'environmental'}",
                f"    awareness: {i / 100.0}",
                f"    position: [{i}, 0, 0]",
                "}",
                ""
            ])
        
        # Add some operations
        program_parts.extend([
            "// Simulation loop",
            "for step in range(10) {",
            "    // Evolution step",
            "    evolve_quantum_system(step * 0.1)",
            "}",
            "",
            "// Final measurements",
            "visualize(large_memory, \"memory_field_3d\")"
        ])
        
        return "\n".join(program_parts)

    def test_concurrent_compilation(self):
        """Test concurrent compilation performance"""
        programs = [
            self._generate_test_program(100, f"program_{i}")
            for i in range(5)
        ]
        
        profiler = PerformanceProfiler()
        profiler.start_profiling()
        start_time = time.time()
        
        # Sequential compilation
        sequential_results = []
        for program in programs:
            compiler = RecursiaCompiler()
            result = compiler.compile_from_source(program, target='quantum_simulator')
            sequential_results.append(result)
        
        sequential_time = time.time() - start_time
        
        # Concurrent compilation
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for program in programs:
                compiler = RecursiaCompiler()
                future = executor.submit(compiler.compile_from_source, program, 'quantum_simulator')
                futures.append(future)
            
            concurrent_results = [future.result() for future in futures]
        
        concurrent_time = time.time() - start_time
        
        # Verify all compilations succeeded
        assert all(r.success for r in sequential_results), "Sequential compilation failures"
        assert all(r.success for r in concurrent_results), "Concurrent compilation failures"
        
        # Concurrent should be faster (assuming multi-core system)
        speedup = sequential_time / concurrent_time
        logging.info(f"Compilation speedup: {speedup:.2f}x")
        
        # Should get some speedup on multi-core systems
        assert speedup > 1.2, f"Insufficient concurrent speedup: {speedup:.2f}x"
    
    def _generate_test_program(self, complexity: int, name: str) -> str:
        """Generate test program with specified complexity"""
        return f"""
        // Test program: {name}
        quantum_state test_state {{
            qubits: {1 + complexity % 5}
            coherence: 0.95
        }}
        
        observer test_observer {{
            type: conscious
            awareness: 0.8
        }}
        
        for i in range({complexity}) {{
            measure(test_state, test_observer)
        }}
        """

class TestVisualizationPerformance:
    """Performance tests for visualization system"""
    
    def test_large_scene_rendering(self):
        """Test rendering performance for large quantum scenes"""
        profiler = PerformanceProfiler()
        viz_engine = QuantumVisualizationEngine()
        
        # Create large quantum system
        physics_engine = PhysicsEngine()
        
        # Many quantum states
        for i in range(200):
            state = QuantumState(1)
            state.name = f"state_{i}"
            state.position = (i % 20, (i // 20) % 10, i // 200)
            alpha = np.random.random()
            beta = np.sqrt(1 - alpha**2)
            state.set_amplitudes(np.array([alpha, beta], dtype=complex))
            physics_engine.add_quantum_state(state)
        
        # Many observers
        for i in range(50):
            observer = Observer(
                name=f"observer_{i}",
                observer_type="conscious",
                self_awareness=np.random.random(),
                position=(i % 10, i // 10, 0)
            )
            physics_engine.add_observer(observer)
        
        profiler.start_profiling()
        start_time = time.time()
        
        # Generate visualizations
        viz_data = viz_engine.generate_complete_scene_visualization(physics_engine)
        
        rendering_time = time.time() - start_time
        
        # Performance requirements
        assert rendering_time < 30.0, f"Large scene rendering too slow: {rendering_time:.2f}s"
        assert profiler.peak_memory < 400.0, f"Rendering memory usage too high: {profiler.peak_memory:.1f}MB"
        
        # Verify visualization data
        assert viz_data is not None, "No visualization data generated"
        assert 'quantum_states' in viz_data, "Missing quantum states in visualization"
        assert 'observers' in viz_data, "Missing observers in visualization"
        assert len(viz_data['quantum_states']) == 200, "Incorrect quantum state count in visualization"
        assert len(viz_data['observers']) == 50, "Incorrect observer count in visualization"

    def test_real_time_visualization_updates(self):
        """Test real-time visualization update performance"""
        profiler = PerformanceProfiler()
        viz_engine = QuantumVisualizationEngine()
        physics_engine = PhysicsEngine()
        
        # Create dynamic system
        state = QuantumState(2)
        bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        state.set_amplitudes(bell_state)
        physics_engine.add_quantum_state(state)
        
        observer = Observer("dynamic_observer", "conscious", self_awareness=0.8)
        physics_engine.add_observer(observer)
        
        profiler.start_profiling()
        start_time = time.time()
        
        # Simulate real-time updates (60 FPS)
        frame_times = []
        for frame in range(300):  # 5 seconds at 60 FPS
            frame_start = time.time()
            
            # Update physics
            physics_engine.evolve_system(1/60)  # 60 FPS
            
            # Generate visualization update
            viz_update = viz_engine.generate_frame_update(physics_engine)
            
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            
            profiler.update_peak_memory()
            
            # Simulate frame timing
            target_frame_time = 1/60  # 60 FPS
            if frame_time < target_frame_time:
                time.sleep(target_frame_time - frame_time)
        
        total_time = time.time() - start_time
        
        # Performance analysis
        avg_frame_time = np.mean(frame_times)
        max_frame_time = np.max(frame_times)
        fps = 1 / avg_frame_time
        
        # Performance requirements
        assert avg_frame_time < 1/30, f"Average frame time too high: {avg_frame_time*1000:.1f}ms"
        assert max_frame_time < 1/15, f"Max frame time too high: {max_frame_time*1000:.1f}ms"
        assert fps > 30.0, f"FPS too low: {fps:.1f}"
        
        logging.info(f"Real-time visualization: {fps:.1f} FPS, {avg_frame_time*1000:.1f}ms avg frame time")

class TestMemoryManagement:
    """Test memory management and garbage collection"""
    
    def test_memory_leak_detection(self):
        """Test for memory leaks in long-running simulations"""
        profiler = PerformanceProfiler()
        initial_memory = profiler.get_current_memory()
        
        # Run repeated simulation cycles
        for cycle in range(10):
            physics_engine = PhysicsEngine()
            
            # Create and destroy many objects
            for i in range(100):
                state = QuantumState(2)
                observer = Observer(f"obs_{i}", "conscious")
                
                physics_engine.add_quantum_state(state)
                physics_engine.add_observer(observer)
                
                # Some operations
                state.calculate_coherence()
                observer.interact_with_state(state)
            
            # Clean up
            del physics_engine
            gc.collect()
        
        final_memory = profiler.get_current_memory()
        memory_growth = final_memory - initial_memory
        
        # Memory growth should be minimal
        assert memory_growth < 50.0, f"Excessive memory growth: {memory_growth:.1f}MB"

    def test_large_state_garbage_collection(self):
        """Test garbage collection of large quantum states"""
        profiler = PerformanceProfiler()
        initial_memory = profiler.get_current_memory()
        
        # Create very large states
        large_states = []
        for i in range(5):
            state = QuantumState(15)  # 32K dimensional state
            large_amplitudes = np.random.complex128(size=2**15)
            large_amplitudes = large_amplitudes / np.linalg.norm(large_amplitudes)
            state.set_amplitudes(large_amplitudes)
            large_states.append(state)
        
        peak_memory = profiler.get_current_memory()
        memory_used = peak_memory - initial_memory
        
        # Should use significant memory
        assert memory_used > 20.0, f"Large states didn't use expected memory: {memory_used:.1f}MB"
        
        # Clean up
        del large_states
        gc.collect()
        
        final_memory = profiler.get_current_memory()
        memory_recovered = peak_memory - final_memory
        recovery_rate = memory_recovered / memory_used
        
        # Should recover most of the memory
        assert recovery_rate > 0.8, f"Poor memory recovery: {recovery_rate:.1%}"

if __name__ == "__main__":
    # Configure logging for benchmark results
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Run comprehensive performance benchmarks
    pytest.main([__file__, "-v", "--tb=short", "-s"])