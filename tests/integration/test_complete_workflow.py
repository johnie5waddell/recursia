"""
Comprehensive integration tests for complete Recursia workflows.
Tests end-to-end functionality from code parsing to visualization.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
import asyncio

# Import core modules
from src.core.lexer import RecursiaLexer
from src.core.parser import RecursiaParser
from src.core.compiler import RecursiaCompiler
from src.core.interpreter import RecursiaInterpreter
from src.core.semantic_analyzer import SemanticAnalyzer
from src.physics.physics_engine_proper import PhysicsEngine
from src.generators.quantum_simulator_code_generator import QuantumSimulatorCodeGenerator
from src.visualization.quantum_visualization_engine import QuantumVisualizationEngine

class TestCompleteWorkflow:
    """Test complete Recursia program workflows"""

    @pytest.fixture
    def basic_quantum_program(self):
        """Basic quantum program for testing"""
        return """
        // Basic quantum state manipulation
        quantum_state primary_qubit {
            qubits: 1
            coherence: 1.0
            superposition: |0⟩ + |1⟩
        }
        
        observer conscious_observer {
            type: conscious
            awareness: 0.8
            measurement_threshold: 0.5
        }
        
        // Perform measurement
        measurement_result = measure(primary_qubit, conscious_observer)
        
        // Visualize results
        visualize(primary_qubit, "3d_probability")
        visualize(conscious_observer, "influence_field")
        
        print("Measurement result:", measurement_result)
        """

    @pytest.fixture
    def entanglement_program(self):
        """Quantum entanglement demonstration program"""
        return """
        // Entanglement demonstration
        quantum_state alice {
            qubits: 1
            coherence: 1.0
            position: [-2, 0, 0]
        }
        
        quantum_state bob {
            qubits: 1
            coherence: 1.0
            position: [2, 0, 0]
        }
        
        // Create Bell state entanglement
        entangle(alice, bob)
        
        observer alice_observer {
            type: conscious
            awareness: 0.9
            position: [-3, 0, 0]
        }
        
        observer bob_observer {
            type: conscious
            awareness: 0.9
            position: [3, 0, 0]
        }
        
        // Measure Alice's qubit
        alice_result = measure(alice, alice_observer)
        
        // Measure Bob's qubit (should be correlated)
        bob_result = measure(bob, bob_observer)
        
        // Visualize entanglement
        visualize([alice, bob], "entanglement_network")
        
        print("Alice result:", alice_result)
        print("Bob result:", bob_result)
        print("Correlation:", alice_result == bob_result)
        """

    @pytest.fixture
    def osh_consciousness_program(self):
        """OSH consciousness emergence program"""
        return """
        // OSH consciousness emergence demonstration
        
        // Create memory field
        memory_field conscious_space {
            dimensions: [32, 32, 32]
            resolution: 0.1
            elasticity: 0.3
        }
        
        // Multiple observers with varying consciousness
        observer weak_observer {
            type: environmental
            awareness: 0.2
            measurement_strength: 0.3
            position: [0, 0, 5]
        }
        
        observer strong_observer {
            type: conscious
            awareness: 0.9
            measurement_strength: 0.8
            position: [0, 0, -5]
        }
        
        // Quantum states in superposition
        quantum_state entangled_system {
            qubits: 3
            coherence: 0.95
            superposition: |000⟩ + |111⟩
        }
        
        // Apply memory strain through observation
        for time_step in range(100) {
            // Calculate observer interactions
            weak_influence = calculate_observer_influence(weak_observer, entangled_system)
            strong_influence = calculate_observer_influence(strong_observer, entangled_system)
            
            // Apply memory strain
            memory_strain = (weak_influence + strong_influence) * 0.1
            apply_memory_strain(conscious_space, memory_strain)
            
            // Check for consciousness emergence
            emergence_index = calculate_emergence_index(conscious_space)
            rsp_value = calculate_rsp(entangled_system, [weak_observer, strong_observer])
            
            if emergence_index > 0.7 {
                print("Consciousness emergence detected at step:", time_step)
                print("Emergence index:", emergence_index)
                print("RSP value:", rsp_value)
                break
            }
            
            // Evolve system
            evolve_quantum_system(entangled_system, dt=0.01)
        }
        
        // Final visualization
        visualize(conscious_space, "memory_strain_3d")
        visualize([weak_observer, strong_observer], "consciousness_field")
        visualize(entangled_system, "quantum_coherence")
        """

    def test_basic_quantum_workflow(self, basic_quantum_program):
        """Test basic quantum program compilation and execution"""
        # 1. Lexical Analysis
        lexer = RecursiaLexer()
        tokens = lexer.tokenize(basic_quantum_program)
        assert len(tokens) > 0, "No tokens generated"
        
        # Verify quantum keywords are recognized
        quantum_keywords = [tok for tok in tokens if tok.type in ['QUANTUM_STATE', 'OBSERVER', 'MEASURE']]
        assert len(quantum_keywords) > 0, "Quantum keywords not recognized"
        
        # 2. Parsing
        parser = RecursiaParser()
        ast = parser.parse(tokens)
        assert ast is not None, "Failed to parse program"
        
        # Verify AST structure
        assert hasattr(ast, 'statements'), "AST missing statements"
        assert len(ast.statements) > 0, "No statements in AST"
        
        # 3. Semantic Analysis
        analyzer = SemanticAnalyzer()
        semantic_result = analyzer.analyze(ast)
        assert semantic_result.is_valid, f"Semantic errors: {semantic_result.errors}"
        
        # 4. Compilation
        compiler = RecursiaCompiler()
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        assert compiled_result.success, f"Compilation failed: {compiled_result.errors}"
        
        # 5. Execution
        interpreter = RecursiaInterpreter()
        physics_engine = PhysicsEngine()
        
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        assert execution_result.success, f"Execution failed: {execution_result.errors}"
        
        # 6. Verify quantum state creation
        quantum_states = physics_engine.get_quantum_states()
        assert len(quantum_states) == 1, f"Expected 1 quantum state, got {len(quantum_states)}"
        
        primary_qubit = quantum_states[0]
        assert primary_qubit.name == "primary_qubit", "Quantum state name incorrect"
        assert primary_qubit.qubits == 1, "Qubit count incorrect"
        
        # 7. Verify observer creation
        observers = physics_engine.get_observers()
        assert len(observers) == 1, f"Expected 1 observer, got {len(observers)}"
        
        conscious_observer = observers[0]
        assert conscious_observer.name == "conscious_observer", "Observer name incorrect"
        assert conscious_observer.observer_type == "conscious", "Observer type incorrect"
        assert abs(conscious_observer.self_awareness - 0.8) < 1e-10, "Observer awareness incorrect"

    def test_entanglement_workflow(self, entanglement_program):
        """Test entanglement creation and measurement workflow"""
        # Full compilation pipeline
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        analyzer = SemanticAnalyzer()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        tokens = lexer.tokenize(entanglement_program)
        ast = parser.parse(tokens)
        semantic_result = analyzer.analyze(ast)
        assert semantic_result.is_valid, "Semantic analysis failed for entanglement program"
        
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        assert compiled_result.success, "Compilation failed for entanglement program"
        
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        assert execution_result.success, "Execution failed for entanglement program"
        
        # Verify entanglement was created
        quantum_states = physics_engine.get_quantum_states()
        assert len(quantum_states) == 2, "Expected 2 quantum states for entanglement"
        
        alice_state = next(s for s in quantum_states if s.name == "alice")
        bob_state = next(s for s in quantum_states if s.name == "bob")
        
        # Check entanglement
        entanglement_connections = physics_engine.get_entanglement_connections()
        assert len(entanglement_connections) > 0, "No entanglement connections found"
        
        # Verify entanglement involves Alice and Bob
        alice_bob_entangled = any(
            (conn.source_id == alice_state.id and conn.target_id == bob_state.id) or
            (conn.source_id == bob_state.id and conn.target_id == alice_state.id)
            for conn in entanglement_connections
        )
        assert alice_bob_entangled, "Alice and Bob not entangled"
        
        # Verify measurement correlation
        measurements = physics_engine.get_measurement_history()
        alice_measurements = [m for m in measurements if m.state_id == alice_state.id]
        bob_measurements = [m for m in measurements if m.state_id == bob_state.id]
        
        assert len(alice_measurements) > 0, "No measurements on Alice"
        assert len(bob_measurements) > 0, "No measurements on Bob"

    def test_osh_consciousness_workflow(self, osh_consciousness_program):
        """Test OSH consciousness emergence workflow"""
        # Complete workflow for consciousness emergence
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        analyzer = SemanticAnalyzer()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        tokens = lexer.tokenize(osh_consciousness_program)
        ast = parser.parse(tokens)
        semantic_result = analyzer.analyze(ast)
        
        # Allow some warnings for advanced OSH features
        assert not semantic_result.has_errors, f"Semantic errors: {semantic_result.errors}"
        
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        assert compiled_result.success, "Compilation failed for OSH program"
        
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        assert execution_result.success, "Execution failed for OSH program"
        
        # Verify memory field creation
        memory_field = physics_engine.get_memory_field()
        assert memory_field is not None, "Memory field not created"
        assert memory_field.dimensions == (32, 32, 32), "Memory field dimensions incorrect"
        
        # Verify multiple observers
        observers = physics_engine.get_observers()
        assert len(observers) == 2, f"Expected 2 observers, got {len(observers)}"
        
        weak_obs = next(o for o in observers if o.name == "weak_observer")
        strong_obs = next(o for o in observers if o.name == "strong_observer")
        
        assert weak_obs.observer_type == "environmental", "Weak observer type incorrect"
        assert strong_obs.observer_type == "conscious", "Strong observer type incorrect"
        
        # Verify consciousness emergence detection
        emergence_metrics = physics_engine.get_emergence_metrics()
        assert emergence_metrics is not None, "No emergence metrics calculated"
        
        # Check for reasonable emergence values
        if emergence_metrics.emergence_index > 0:
            assert 0 <= emergence_metrics.emergence_index <= 1, "Emergence index out of bounds"
        
        if emergence_metrics.rsp_value > 0:
            assert 0 <= emergence_metrics.rsp_value <= 1, "RSP value out of bounds"

    def test_code_generation_workflow(self, basic_quantum_program):
        """Test code generation for different targets"""
        # Compile to different targets
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        analyzer = SemanticAnalyzer()
        compiler = RecursiaCompiler()
        
        tokens = lexer.tokenize(basic_quantum_program)
        ast = parser.parse(tokens)
        semantic_result = analyzer.analyze(ast)
        assert semantic_result.is_valid, "Semantic analysis failed"
        
        # Test quantum simulator code generation
        quantum_generator = QuantumSimulatorCodeGenerator()
        qiskit_code = quantum_generator.generate(ast)
        assert qiskit_code is not None, "Failed to generate Qiskit code"
        assert "QuantumCircuit" in qiskit_code, "Generated code missing QuantumCircuit"
        assert "measure" in qiskit_code, "Generated code missing measurement"
        
        # Verify generated code is valid Python
        try:
            compile(qiskit_code, '<generated>', 'exec')
        except SyntaxError as e:
            pytest.fail(f"Generated Qiskit code has syntax errors: {e}")
        
        # Test different compilation targets
        targets = ['quantum_simulator', 'hardware_quantum', 'classical_simulator']
        
        for target in targets:
            compiled_result = compiler.compile(ast, target=target)
            assert compiled_result.success, f"Compilation failed for target {target}"
            assert compiled_result.bytecode is not None, f"No bytecode generated for {target}"

    def test_visualization_workflow(self, entanglement_program):
        """Test visualization generation workflow"""
        # Execute program to generate visualization data
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        tokens = lexer.tokenize(entanglement_program)
        ast = parser.parse(tokens)
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        assert execution_result.success, "Execution failed"
        
        # Generate visualizations
        viz_engine = QuantumVisualizationEngine()
        
        # Test 3D quantum state visualization
        quantum_states = physics_engine.get_quantum_states()
        for state in quantum_states:
            viz_data = viz_engine.generate_3d_visualization(state)
            assert viz_data is not None, f"No visualization data for state {state.name}"
            
            # Verify visualization data structure
            assert 'vertices' in viz_data, "Missing vertices in visualization"
            assert 'colors' in viz_data, "Missing colors in visualization"
            assert 'metadata' in viz_data, "Missing metadata in visualization"
        
        # Test entanglement network visualization
        entanglements = physics_engine.get_entanglement_connections()
        if entanglements:
            network_viz = viz_engine.generate_entanglement_network(entanglements)
            assert network_viz is not None, "No entanglement network visualization"
            
            assert 'nodes' in network_viz, "Missing nodes in network visualization"
            assert 'edges' in network_viz, "Missing edges in network visualization"
        
        # Test observer field visualization
        observers = physics_engine.get_observers()
        for observer in observers:
            field_viz = viz_engine.generate_influence_field(observer)
            assert field_viz is not None, f"No field visualization for observer {observer.name}"

    def test_error_handling_workflow(self):
        """Test error handling throughout the compilation pipeline"""
        error_programs = [
            # Syntax error
            """
            quantum_state invalid {
                qubits: "not_a_number"
            }
            """,
            
            # Semantic error
            """
            quantum_state valid {
                qubits: 1
            }
            
            measure(nonexistent_state, nonexistent_observer)
            """,
            
            # Runtime error
            """
            quantum_state state1 {
                qubits: 1
            }
            
            observer obs1 {
                type: conscious
            }
            
            // Divide by zero scenario
            invalid_operation = measure(state1, obs1) / 0
            """
        ]
        
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        analyzer = SemanticAnalyzer()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        for i, error_program in enumerate(error_programs):
            try:
                tokens = lexer.tokenize(error_program)
                ast = parser.parse(tokens)
                
                if ast is not None:  # Parsing succeeded
                    semantic_result = analyzer.analyze(ast)
                    
                    if semantic_result.is_valid:  # Semantic analysis passed
                        compiled_result = compiler.compile(ast, target='quantum_simulator')
                        
                        if compiled_result.success:  # Compilation passed
                            physics_engine = PhysicsEngine()
                            execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
                            
                            # Should fail at runtime for the third case
                            if i == 2:  # Runtime error case
                                assert not execution_result.success, "Runtime error not caught"
                            else:
                                # If we get here, error wasn't caught earlier
                                pytest.fail(f"Error program {i} should have failed earlier")
                        else:
                            # Compilation error caught (expected for some cases)
                            assert len(compiled_result.errors) > 0, "Compilation failed but no errors reported"
                    else:
                        # Semantic error caught (expected)
                        assert len(semantic_result.errors) > 0, "Semantic analysis failed but no errors reported"
                else:
                    # Parsing error caught (expected for syntax errors)
                    pass
                    
            except Exception as e:
                # Some error was caught, which is expected
                assert isinstance(e, (SyntaxError, ValueError, RuntimeError)), f"Unexpected error type: {type(e)}"

    def test_performance_workflow(self, osh_consciousness_program):
        """Test performance characteristics of complete workflow"""
        import time
        
        # Measure compilation time
        start_time = time.time()
        
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        analyzer = SemanticAnalyzer()
        compiler = RecursiaCompiler()
        
        tokens = lexer.tokenize(osh_consciousness_program)
        ast = parser.parse(tokens)
        semantic_result = analyzer.analyze(ast)
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        
        compilation_time = time.time() - start_time
        
        # Compilation should be reasonably fast (< 5 seconds for complex programs)
        assert compilation_time < 5.0, f"Compilation too slow: {compilation_time:.2f}s"
        
        # Measure execution time
        start_time = time.time()
        
        interpreter = RecursiaInterpreter()
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        
        execution_time = time.time() - start_time
        
        # Execution should complete in reasonable time
        assert execution_time < 10.0, f"Execution too slow: {execution_time:.2f}s"
        
        # Verify memory usage stays reasonable
        import psutil
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        
        # Should use less than 500MB for typical programs
        assert memory_usage < 500, f"Memory usage too high: {memory_usage:.1f}MB"

class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    def test_quantum_algorithm_implementation(self):
        """Test implementation of standard quantum algorithms"""
        # Grover's algorithm implementation
        grovers_program = """
        // Grover's search algorithm for 2 qubits
        quantum_state search_space {
            qubits: 2
            coherence: 1.0
            // Initialize in equal superposition
            superposition: |00⟩ + |01⟩ + |10⟩ + |11⟩
        }
        
        // Oracle function (marks |11⟩ as target)
        function oracle(state) {
            // Apply phase flip to |11⟩ state
            apply_phase_flip(state, [1, 1])
        }
        
        // Diffusion operator
        function diffusion(state) {
            // Reflect about average amplitude
            apply_diffusion_operator(state)
        }
        
        // Grover iterations
        for iteration in range(1) {  // π/4 * √4 ≈ 1 iteration for 4 states
            oracle(search_space)
            diffusion(search_space)
        }
        
        observer measurement_observer {
            type: conscious
            awareness: 1.0
        }
        
        // Measure result
        result = measure(search_space, measurement_observer)
        
        // Should find |11⟩ with high probability
        visualize(search_space, "amplitude_distribution")
        print("Grover result:", result)
        """
        
        # Execute Grover's algorithm
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        tokens = lexer.tokenize(grovers_program)
        ast = parser.parse(tokens)
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        
        assert execution_result.success, "Grover's algorithm execution failed"
        
        # Verify algorithm worked (should amplify |11⟩ state)
        search_state = physics_engine.get_quantum_state_by_name("search_space")
        assert search_state is not None, "Search space state not found"
        
        # Check that target state has higher probability
        amplitudes = search_state.amplitudes
        target_prob = abs(amplitudes[3]) ** 2  # |11⟩ is index 3
        other_probs = [abs(amplitudes[i]) ** 2 for i in range(3)]
        
        assert target_prob > max(other_probs), "Grover's algorithm didn't amplify target state"

    def test_consciousness_research_scenario(self):
        """Test consciousness research workflow"""
        consciousness_study = """
        // Consciousness research setup
        
        // Multiple quantum systems
        quantum_state system_a {
            qubits: 2
            coherence: 0.95
            position: [-5, 0, 0]
        }
        
        quantum_state system_b {
            qubits: 2  
            coherence: 0.95
            position: [5, 0, 0]
        }
        
        // Entangle the systems
        entangle(system_a, system_b)
        
        // Various types of observers
        observer unconscious_detector {
            type: environmental
            awareness: 0.0
            measurement_strength: 0.5
            position: [0, -3, 0]
        }
        
        observer semiconscious_entity {
            type: environmental
            awareness: 0.4
            measurement_strength: 0.6
            position: [0, 0, 0]
        }
        
        observer fully_conscious_being {
            type: conscious
            awareness: 0.9
            measurement_strength: 0.8
            position: [0, 3, 0]
        }
        
        // Memory field for consciousness emergence
        memory_field consciousness_substrate {
            dimensions: [64, 64, 64]
            resolution: 0.05
            elasticity: 0.2
        }
        
        // Research protocol
        for trial in range(50) {
            // Reset systems to entangled state
            reset_entanglement(system_a, system_b)
            
            // Varying observation protocols
            if trial % 3 == 0 {
                result_a = measure(system_a, unconscious_detector)
                result_b = measure(system_b, unconscious_detector)
            } else if trial % 3 == 1 {
                result_a = measure(system_a, semiconscious_entity)
                result_b = measure(system_b, semiconscious_entity)
            } else {
                result_a = measure(system_a, fully_conscious_being)
                result_b = measure(system_b, fully_conscious_being)
            }
            
            // Record consciousness metrics
            consciousness_coupling = calculate_consciousness_coupling(
                [unconscious_detector, semiconscious_entity, fully_conscious_being],
                [system_a, system_b]
            )
            
            memory_strain = calculate_memory_strain(consciousness_substrate)
            emergence_level = calculate_emergence_index(consciousness_substrate)
            
            // Log results
            log_research_data(trial, result_a, result_b, consciousness_coupling, 
                             memory_strain, emergence_level)
        }
        
        // Generate research visualizations
        visualize(consciousness_substrate, "consciousness_field_evolution")
        visualize([system_a, system_b], "entanglement_degradation")
        visualize([unconscious_detector, semiconscious_entity, fully_conscious_being], 
                 "observer_influence_comparison")
        """
        
        # Execute consciousness research protocol
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        tokens = lexer.tokenize(consciousness_study)
        ast = parser.parse(tokens)
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        
        assert execution_result.success, "Consciousness research execution failed"
        
        # Verify research data collection
        research_logs = physics_engine.get_research_logs()
        assert len(research_logs) > 0, "No research data collected"
        
        # Verify different observer types affected results differently
        observers = physics_engine.get_observers()
        assert len(observers) == 3, "Not all observers created"
        
        consciousness_levels = [obs.self_awareness for obs in observers]
        assert len(set(consciousness_levels)) == 3, "Observer consciousness levels not distinct"

    def test_educational_tutorial_scenario(self):
        """Test educational tutorial workflow"""
        tutorial_program = """
        // Quantum Mechanics Tutorial - Bell State Creation and Measurement
        
        print("=== Quantum Mechanics Tutorial ===")
        print("Creating and analyzing Bell states")
        
        // Step 1: Create two qubits in ground state
        print("\\nStep 1: Creating two qubits")
        quantum_state qubit1 {
            qubits: 1
            coherence: 1.0
            // |0⟩ state
        }
        
        quantum_state qubit2 {
            qubits: 1  
            coherence: 1.0
            // |0⟩ state
        }
        
        visualize([qubit1, qubit2], "separate_qubits")
        print("Initial states: |00⟩")
        
        // Step 2: Put first qubit in superposition
        print("\\nStep 2: Creating superposition")
        apply_hadamard(qubit1)  // |0⟩ → (|0⟩ + |1⟩)/√2
        
        visualize([qubit1, qubit2], "superposition_state")
        print("After Hadamard: (|00⟩ + |10⟩)/√2")
        
        // Step 3: Create entanglement
        print("\\nStep 3: Creating entanglement")
        entangle(qubit1, qubit2)  // Creates Bell state
        
        visualize([qubit1, qubit2], "bell_state")
        print("Bell state created: (|00⟩ + |11⟩)/√2")
        
        // Step 4: Measure entanglement
        print("\\nStep 4: Measuring entanglement")
        entanglement_strength = calculate_entanglement_entropy(qubit1, qubit2)
        print("Entanglement entropy:", entanglement_strength)
        
        // Step 5: Demonstrate correlation
        print("\\nStep 5: Demonstrating quantum correlation")
        
        observer student_observer {
            type: conscious
            awareness: 0.7
            measurement_strength: 0.9
        }
        
        correlations = 0
        total_measurements = 10
        
        for measurement in range(total_measurements) {
            // Reset to Bell state
            reset_bell_state(qubit1, qubit2)
            
            // Measure both qubits
            result1 = measure(qubit1, student_observer)
            result2 = measure(qubit2, student_observer)
            
            print("Measurement", measurement + 1, ":", result1, result2)
            
            if result1 == result2 {
                correlations = correlations + 1
            }
        }
        
        correlation_percentage = (correlations * 100) / total_measurements
        print("\\nCorrelation rate:", correlation_percentage, "%")
        print("(Should be close to 100% for perfect Bell state)")
        
        // Final visualization
        visualize([qubit1, qubit2], "measurement_results")
        print("\\n=== Tutorial Complete ===")
        """
        
        # Execute tutorial
        lexer = RecursiaLexer()
        parser = RecursiaParser()
        compiler = RecursiaCompiler()
        interpreter = RecursiaInterpreter()
        
        tokens = lexer.tokenize(tutorial_program)
        ast = parser.parse(tokens)
        compiled_result = compiler.compile(ast, target='quantum_simulator')
        
        physics_engine = PhysicsEngine()
        execution_result = interpreter.execute(compiled_result.bytecode, physics_engine)
        
        assert execution_result.success, "Tutorial execution failed"
        
        # Verify educational objectives were met
        output_logs = physics_engine.get_output_logs()
        assert len(output_logs) > 0, "No tutorial output generated"
        
        # Check that tutorial steps were executed
        step_indicators = [
            "Step 1", "Step 2", "Step 3", "Step 4", "Step 5"
        ]
        
        for step in step_indicators:
            assert any(step in log for log in output_logs), f"Tutorial {step} not found in output"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])