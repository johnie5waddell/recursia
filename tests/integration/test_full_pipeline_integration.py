"""
Integration tests for complete Recursia pipelines.
Tests full workflows from parsing to execution to measurement.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.core.lexer import Lexer
from src.core.parser import Parser
from src.core.compiler import Compiler
from src.core.interpreter import Interpreter
from src.core.runtime import Runtime
from src.physics.physics_engine_proper import PhysicsEngine
from src.visualization.visualizer import Visualizer
from src.quantum.quantum_state import QuantumState


class TestFullCompilationPipeline:
    """Test complete compilation from source to execution."""
    
    @pytest.fixture
    def compiler_stack(self):
        """Create full compiler stack."""
        return {
            'lexer': Lexer(),
            'parser': Parser(),
            'compiler': Compiler(),
            'runtime': Runtime()
        }
        
    def test_hello_quantum_world(self, compiler_stack):
        """Test simple quantum hello world program."""
        code = """
        quantum q = |0⟩;
        gate H(q);
        bit result = measure(q);
        print("Measured:", result);
        """
        
        # Compile
        tokens = compiler_stack['lexer'].tokenize(code)
        ast = compiler_stack['parser'].parse(tokens)
        compiled = compiler_stack['compiler'].compile(ast)
        
        # Execute
        result = compiler_stack['runtime'].execute(compiled)
        
        # Should have executed successfully
        assert result.exit_code == 0
        assert result.measurements is not None
        assert len(result.measurements) == 1
        assert result.measurements[0] in [0, 1]
        
    def test_entanglement_creation_pipeline(self, compiler_stack):
        """Test creating and measuring entanglement."""
        code = """
        quantum_register qreg[2];
        
        // Create Bell state
        gate H(qreg[0]);
        gate CNOT(qreg[0], qreg[1]);
        
        // Measure entanglement
        entanglement ent = calculate_entanglement(qreg);
        assert(ent > 0.9, "Should be maximally entangled");
        
        // Measure qubits
        bit[2] results = measure(qreg);
        assert(results[0] == results[1], "Correlated outcomes");
        """
        
        tokens = compiler_stack['lexer'].tokenize(code)
        ast = compiler_stack['parser'].parse(tokens)
        compiled = compiler_stack['compiler'].compile(ast)
        result = compiler_stack['runtime'].execute(compiled)
        
        assert result.exit_code == 0
        assert result.assertions_passed == 2
        
    def test_observer_simulation_pipeline(self, compiler_stack):
        """Test observer-based quantum simulation."""
        code = """
        quantum_field field(dimension=10);
        observer obs {
            focus: 0.8,
            threshold: 0.6,
            phase: 0
        };
        
        simulate(steps=100, dt=0.01) {
            initialize {
                field.coherence = 0.95;
                field.add_excitation(position=5, amplitude=1.0);
            }
            
            evolution {
                field.evolve(dt);
                if (obs.detect(field)) {
                    field.collapse(obs.position);
                }
            }
            
            measurement {
                return field.total_coherence();
            }
        }
        """
        
        tokens = compiler_stack['lexer'].tokenize(code)
        ast = compiler_stack['parser'].parse(tokens)
        compiled = compiler_stack['compiler'].compile(ast)
        result = compiler_stack['runtime'].execute(compiled)
        
        assert result.exit_code == 0
        assert result.simulation_data is not None
        assert len(result.simulation_data['coherence_history']) == 100


class TestPhysicsIntegration:
    """Test integration of physics modules."""
    
    @pytest.fixture
    def physics_system(self):
        """Create integrated physics system."""
        engine = PhysicsEngine()
        engine.initialize()
        return engine
        
    def test_coherence_entanglement_coupling(self, physics_system):
        """Test coupling between coherence and entanglement."""
        # Create entangled state
        state1 = physics_system.create_quantum_state(n_qubits=2)
        state1.initialize_bell_state("phi_plus")
        
        # Check initial coherence and entanglement
        initial_coherence = physics_system.calculate_coherence(state1)
        initial_entanglement = physics_system.calculate_entanglement(state1)
        
        assert initial_coherence > 0.9
        assert initial_entanglement > 0.9
        
        # Apply decoherence
        for _ in range(10):
            physics_system.apply_decoherence(
                state1,
                channel="amplitude_damping",
                strength=0.1
            )
            
        # Both should decrease together
        final_coherence = physics_system.calculate_coherence(state1)
        final_entanglement = physics_system.calculate_entanglement(state1)
        
        assert final_coherence < initial_coherence
        assert final_entanglement < initial_entanglement
        
        # Should be correlated
        coherence_loss = initial_coherence - final_coherence
        entanglement_loss = initial_entanglement - final_entanglement
        
        assert abs(coherence_loss - entanglement_loss) < 0.2
        
    def test_memory_field_quantum_coupling(self, physics_system):
        """Test memory field effects on quantum states."""
        # Create memory field
        memory = physics_system.create_memory_field(dimension=(10, 10, 10))
        
        # Create quantum state at high strain location
        position = (5, 5, 5)
        memory.allocate_memory(position, size=2.0, coherence=0.3)
        
        # Place quantum state there
        state = physics_system.create_quantum_state(n_qubits=1)
        state.position = position
        
        # Evolve with memory coupling
        for _ in range(20):
            physics_system.evolve_with_memory_coupling(
                state, memory,
                coupling_strength=0.1,
                time_step=0.1
            )
            
        # Low memory coherence should degrade quantum coherence
        quantum_coherence = physics_system.calculate_coherence(state)
        assert quantum_coherence < 0.5
        
    def test_observer_measurement_backaction(self, physics_system):
        """Test full observer-measurement-backaction cycle."""
        # Create quantum system
        state = physics_system.create_quantum_state(n_qubits=3)
        state.initialize_ghz()
        
        # Create observer
        observer = physics_system.create_observer(
            focus=0.7,
            measurement_strength=0.5
        )
        
        # Observation cycle
        pre_entanglement = physics_system.calculate_multipartite_entanglement(state)
        
        # Observer measures
        outcome = observer.measure(state, basis="Z", qubit=1)
        
        # Check backaction
        post_entanglement = physics_system.calculate_multipartite_entanglement(state)
        
        # Should reduce entanglement
        assert post_entanglement < pre_entanglement
        
        # But not destroy it completely (weak measurement)
        assert post_entanglement > 0.3
        
    def test_field_evolution_with_sources(self, physics_system):
        """Test field evolution with quantum sources."""
        # Create field
        field = physics_system.create_field(
            type="coherence",
            dimension=(20, 20)
        )
        
        # Add quantum sources
        sources = []
        for i in range(3):
            pos = (5 + i*5, 10)
            source = physics_system.create_quantum_source(
                position=pos,
                frequency=0.1 * (i + 1),
                amplitude=1.0
            )
            sources.append(source)
            
        # Evolve and check interference
        patterns = []
        for t in range(50):
            physics_system.evolve_field_with_sources(
                field, sources,
                time=t * 0.1
            )
            patterns.append(field.values.copy())
            
        # Should see interference patterns
        patterns = np.array(patterns)
        
        # Check for oscillations at source positions
        for i, source in enumerate(sources):
            pos = source.position
            signal = patterns[:, pos[0], pos[1]]
            
            # FFT to check frequency
            fft = np.fft.fft(signal)
            freqs = np.fft.fftfreq(len(signal))
            
            # Should have peak at source frequency
            peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            peak_freq = abs(freqs[peak_idx])
            
            expected_freq = 0.1 * (i + 1) / (2 * np.pi * 0.1)  # Normalized
            assert abs(peak_freq - expected_freq) < 0.5


class TestVisualizationIntegration:
    """Test visualization system integration."""
    
    @pytest.fixture
    def viz_system(self):
        """Create visualization system."""
        return Visualizer(headless=True)  # Headless for testing
        
    def test_quantum_state_visualization(self, viz_system):
        """Test quantum state visualization pipeline."""
        # Create states
        states = []
        
        # Pure state
        pure = QuantumState(n_qubits=2)
        pure.initialize_zero()
        states.append(("Pure |00⟩", pure))
        
        # Entangled state
        bell = QuantumState(n_qubits=2)
        bell.amplitudes[0] = 1/np.sqrt(2)
        bell.amplitudes[3] = 1/np.sqrt(2)
        states.append(("Bell |Φ+⟩", bell))
        
        # Mixed state
        mixed = QuantumState(n_qubits=2)
        mixed.density_matrix = np.eye(4) / 4
        states.append(("Maximally mixed", mixed))
        
        # Generate visualizations
        for name, state in states:
            vis_data = viz_system.visualize_quantum_state(state)
            
            assert vis_data is not None
            assert 'bloch_sphere' in vis_data
            assert 'density_matrix' in vis_data
            assert 'probabilities' in vis_data
            
    def test_field_evolution_animation(self, viz_system):
        """Test field evolution animation generation."""
        # Create evolving field
        physics = PhysicsEngine()
        field = physics.create_field("memory", dimension=(50, 50))
        
        # Add some features
        field.add_gaussian_blob(center=(25, 25), width=5, amplitude=1.0)
        field.add_gaussian_blob(center=(10, 35), width=3, amplitude=0.5)
        
        # Record evolution
        frames = []
        for _ in range(30):
            physics.evolve_field(field, time_step=0.1)
            frame_data = viz_system.capture_field_frame(field)
            frames.append(frame_data)
            
        # Create animation
        animation = viz_system.create_animation(
            frames,
            fps=10,
            title="Memory Field Evolution"
        )
        
        assert animation is not None
        assert animation.frame_count == 30
        assert animation.duration == 3.0  # seconds
        
    def test_measurement_statistics_plots(self, viz_system):
        """Test measurement statistics visualization."""
        # Simulate measurements
        measurements = []
        
        # Biased coin
        for _ in range(1000):
            if np.random.rand() < 0.7:
                measurements.append(0)
            else:
                measurements.append(1)
                
        # Generate statistics plots
        plots = viz_system.create_measurement_plots(
            measurements,
            title="Quantum Measurement Statistics"
        )
        
        assert 'histogram' in plots
        assert 'time_series' in plots
        assert 'autocorrelation' in plots
        
        # Check histogram data
        hist_data = plots['histogram']
        assert abs(hist_data['probabilities'][0] - 0.7) < 0.05
        assert abs(hist_data['probabilities'][1] - 0.3) < 0.05


class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""
    
    def test_quantum_teleportation_protocol(self):
        """Test full quantum teleportation implementation."""
        code = """
        // Quantum teleportation protocol
        program teleportation {
            // Alice and Bob share entangled pair
            quantum_register alice[2];
            quantum_register bob[1];
            
            // Create entanglement
            gate H(alice[0]);
            gate CNOT(alice[0], bob[0]);
            
            // Alice's unknown state to teleport
            quantum message = 0.6|0⟩ + 0.8|1⟩;
            
            // Alice's Bell measurement
            gate CNOT(message, alice[0]);
            gate H(message);
            
            bit c1 = measure(message);
            bit c2 = measure(alice[0]);
            
            // Bob's corrections
            if (c2 == 1) { gate X(bob[0]); }
            if (c1 == 1) { gate Z(bob[0]); }
            
            // Verify fidelity
            fidelity f = calculate_fidelity(bob[0], 0.6|0⟩ + 0.8|1⟩);
            assert(f > 0.99, "High fidelity teleportation");
            
            return bob[0];
        }
        """
        
        # Full compilation and execution
        lexer = Lexer()
        parser = Parser()
        compiler = Compiler()
        runtime = Runtime()
        
        tokens = lexer.tokenize(code)
        ast = parser.parse(tokens)
        compiled = compiler.compile(ast)
        result = runtime.execute(compiled)
        
        assert result.exit_code == 0
        assert result.assertions_passed == 1
        assert result.return_value is not None
        
    def test_memory_strain_simulation(self):
        """Test memory strain and defragmentation simulation."""
        code = """
        program memory_dynamics {
            memory_field mem(dimension=(20, 20, 20));
            
            // Allocation phase
            for (i = 0; i < 50; i++) {
                position p = random_position();
                mem.allocate(p, size=random(0.5, 2.0));
                
                if (mem.strain > 0.8) {
                    print("High strain detected:", mem.strain);
                    break;
                }
            }
            
            // Measure fragmentation
            float frag_before = mem.fragmentation_score();
            
            // Defragmentation
            mem.defragment(preserve_coherence=true);
            
            float frag_after = mem.fragmentation_score();
            assert(frag_after < frag_before, "Defragmentation worked");
            
            // Test quantum state in defragged memory
            quantum q = |+⟩;
            q.position = mem.find_coherent_region(min_size=1.0);
            
            simulate(steps=50, dt=0.1) {
                evolution {
                    q.evolve_in_field(mem);
                }
            }
            
            float final_coherence = calculate_coherence(q);
            assert(final_coherence > 0.7, "Maintained coherence");
        }
        """
        
        # Execute
        system = CompilerSystem()
        result = system.compile_and_run(code)
        
        assert result.success
        assert result.assertions_passed == 2
        
    def test_consciousness_emergence_simulation(self):
        """Test consciousness field emergence simulation."""
        code = """
        program consciousness_emergence {
            // Initialize consciousness field
            consciousness_field psi(
                dimension=(30, 30, 30),
                coupling=0.1,
                threshold=0.6
            );
            
            // Add seed patterns
            psi.add_seed(
                position=(15, 15, 15),
                pattern="oscillator",
                frequency=0.1
            );
            
            // Create observer hierarchy
            observer_hierarchy obs_tree(depth=3);
            
            // Evolution with measurements
            float[] integrated_info = [];
            
            simulate(steps=200, dt=0.05) {
                evolution {
                    psi.evolve(dt);
                    
                    // Observer interactions
                    for (obs in obs_tree) {
                        if (obs.resonates_with(psi)) {
                            psi.enhance_at(obs.focus_point);
                        }
                    }
                    
                    // Calculate integrated information
                    phi = psi.integrated_information();
                    integrated_info.append(phi);
                }
            }
            
            // Check for emergence
            float max_phi = max(integrated_info);
            float final_phi = integrated_info[-1];
            
            assert(max_phi > 2.0, "Achieved high integration");
            assert(final_phi > 1.5, "Sustained consciousness");
            
            // Visualize
            visualize {
                plot(integrated_info, title="Φ Evolution");
                render_3d(psi, colormap="consciousness");
            }
        }
        """
        
        # Full execution with visualization
        system = FullRecursiaSystem()
        result = system.execute_with_visualization(code)
        
        assert result.success
        assert result.assertions_passed == 2
        assert result.visualizations is not None
        assert 'phi_evolution' in result.visualizations
        assert 'field_3d' in result.visualizations


class TestErrorHandlingIntegration:
    """Test error handling across the system."""
    
    def test_compilation_error_recovery(self):
        """Test graceful handling of compilation errors."""
        bad_code = """
        quantum q = |0⟩;
        gate INVALID_GATE(q);  // Should fail
        measure(q);
        """
        
        system = CompilerSystem()
        result = system.compile_and_run(bad_code)
        
        assert not result.success
        assert result.error_type == "CompilationError"
        assert "INVALID_GATE" in result.error_message
        assert result.error_line == 3
        
    def test_runtime_error_handling(self):
        """Test runtime error detection and reporting."""
        code = """
        quantum_register qreg[2];
        gate H(qreg[5]);  // Index out of bounds
        """
        
        system = CompilerSystem()
        result = system.compile_and_run(code)
        
        assert not result.success
        assert result.error_type == "RuntimeError"
        assert "index out of bounds" in result.error_message.lower()
        
    def test_physics_constraint_violations(self):
        """Test handling of physics constraint violations."""
        code = """
        quantum q = |0⟩;
        
        // Try to create invalid superposition
        q.amplitudes = [0.5, 0.5];  // Not normalized
        
        measure(q);  // Should fail
        """
        
        system = CompilerSystem()
        result = system.compile_and_run(code)
        
        assert not result.success
        assert result.error_type == "PhysicsError"
        assert "normalization" in result.error_message.lower()


class TestPerformanceIntegration:
    """Test system performance under load."""
    
    def test_large_scale_simulation(self):
        """Test performance with large quantum systems."""
        code = """
        program large_scale {
            quantum_register qreg[16];  // 2^16 dimensional
            
            // Create random circuit
            for (i = 0; i < 100; i++) {
                gate H(qreg[i % 16]);
                if (i > 0) {
                    gate CNOT(qreg[i % 16], qreg[(i-1) % 16]);
                }
            }
            
            // Measure performance
            timer t;
            t.start();
            
            bit[16] results = measure(qreg);
            
            float elapsed = t.elapsed();
            assert(elapsed < 5.0, "Completed in reasonable time");
            
            print("Execution time:", elapsed, "seconds");
        }
        """
        
        system = CompilerSystem()
        result = system.compile_and_run(code)
        
        assert result.success
        assert result.performance_metrics['execution_time'] < 5.0
        assert result.performance_metrics['memory_usage'] < 1e9  # < 1GB
        
    def test_parallel_simulation_execution(self):
        """Test parallel execution of independent simulations."""
        code = """
        @parallel(n_threads=4)
        program parallel_sim {
            results = [];
            
            parallel_for (trial = 0; trial < 1000; trial++) {
                quantum q = |0⟩;
                gate H(q);
                bit result = measure(q);
                results.append(result);
            }
            
            // Check statistics
            float average = mean(results);
            assert(0.45 < average < 0.55, "Fair coin statistics");
        }
        """
        
        system = CompilerSystem(enable_parallel=True)
        result = system.compile_and_run(code)
        
        assert result.success
        assert result.performance_metrics['speedup'] > 2.0  # Parallel speedup