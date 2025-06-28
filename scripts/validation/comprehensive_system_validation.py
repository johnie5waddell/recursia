#!/usr/bin/env python3
"""
Comprehensive validation script for Recursia v3 system
Tests core quantum programs, OSH calculations, API endpoints, and full pipeline
"""

import sys
import os
import time
import json
import requests
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configuration
API_BASE_URL = "http://localhost:5000"
TIMEOUT = 30  # seconds for each test

class ValidationResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.details = []
    
    def add_result(self, category: str, test_name: str, passed: bool, message: str = ""):
        if passed:
            self.passed += 1
            status = "✓ PASS"
        else:
            self.failed += 1
            status = "✗ FAIL"
        
        self.details.append({
            "category": category,
            "test": test_name,
            "passed": passed,
            "message": message
        })
        print(f"  {status} {test_name}: {message}")
    
    def add_skip(self, category: str, test_name: str, reason: str):
        self.skipped += 1
        self.details.append({
            "category": category,
            "test": test_name,
            "skipped": True,
            "message": reason
        })
        print(f"  ⚠️  SKIP {test_name}: {reason}")

results = ValidationResults()

# 1. Test Core Quantum Programs
print("\n🔬 1. TESTING CORE QUANTUM PROGRAMS")
print("=" * 60)

def test_quantum_program(program_path: str, expected_results: Dict[str, Any] = None) -> bool:
    """Test execution of a quantum program"""
    try:
        from src.core.compiler import Compiler
        from src.core.interpreter import Interpreter
        from src.core.runtime import Runtime
        
        # Read program
        with open(program_path, 'r') as f:
            code = f.read()
        
        # Compile and execute
        compiler = Compiler()
        ast = compiler.compile(code)
        
        runtime = Runtime()
        interpreter = Interpreter(runtime)
        
        # Execute with timeout
        start_time = time.time()
        result = interpreter.interpret(ast)
        execution_time = time.time() - start_time
        
        if execution_time > TIMEOUT:
            return False, f"Execution timeout ({execution_time:.2f}s > {TIMEOUT}s)"
        
        # Check results if expected
        if expected_results:
            for key, expected in expected_results.items():
                if key == "final_state" and hasattr(result, "state"):
                    # Check quantum state properties
                    state = result.state
                    if "probability_sum" in expected:
                        prob_sum = np.sum(np.abs(state.amplitudes)**2)
                        if not np.isclose(prob_sum, expected["probability_sum"], atol=1e-6):
                            return False, f"Probability sum mismatch: {prob_sum} != {expected['probability_sum']}"
        
        return True, f"Executed in {execution_time:.2f}s"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# Test basic quantum programs
quantum_programs = [
    {
        "path": "quantum_programs/basic/hello_quantum.recursia",
        "name": "Hello Quantum",
        "expected": {"probability_sum": 1.0}
    },
    {
        "path": "quantum_programs/basic/simple_entanglement.recursia",
        "name": "Simple Entanglement",
        "expected": {"probability_sum": 1.0}
    },
    {
        "path": "quantum_programs/intermediate/quantum_teleportation.recursia",
        "name": "Quantum Teleportation",
        "expected": None
    }
]

for program in quantum_programs:
    if os.path.exists(program["path"]):
        passed, message = test_quantum_program(program["path"], program.get("expected"))
        results.add_result("Quantum Programs", program["name"], passed, message)
    else:
        results.add_skip("Quantum Programs", program["name"], "File not found")

# 2. Test OSH Calculations and Consciousness Emergence
print("\n🧠 2. TESTING OSH CALCULATIONS AND CONSCIOUSNESS")
print("=" * 60)

def test_osh_calculations():
    """Test OSH calculation engine"""
    try:
        from src.engines.OSHCalculationEngine import OSHCalculationEngine
        from src.quantum.quantum_state import QuantumState
        
        engine = OSHCalculationEngine()
        
        # Test with a simple quantum state
        amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        qstate = QuantumState(amplitudes)
        
        # Calculate OSH metrics
        metrics = engine.calculate_all_metrics(qstate)
        
        # Validate results
        checks = [
            ("phi_IIT" in metrics, "IIT calculation present"),
            (0 <= metrics.get("phi_IIT", -1) <= 1000, "IIT value in valid range"),
            ("entropy_coherence_ratio" in metrics, "Entropy coherence present"),
            ("rsp_value" in metrics, "RSP calculation present"),
            ("gravitational_binding" in metrics, "Gravitational binding present")
        ]
        
        for check, desc in checks:
            if not check:
                return False, f"Failed: {desc}"
        
        return True, f"All OSH metrics calculated successfully"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

def test_consciousness_emergence():
    """Test consciousness emergence detection"""
    try:
        from src.physics.consciousness_measurement_validation import ConsciousnessMeasurementValidator
        
        validator = ConsciousnessMeasurementValidator()
        
        # Test with a known conscious-like state
        test_data = {
            "phi_IIT": 2.5,
            "entropy_coherence_ratio": 0.75,
            "observer_count": 3,
            "measurement_frequency": 10.0
        }
        
        is_conscious = validator.validate_consciousness_emergence(test_data)
        
        return True, f"Consciousness detection working: {is_conscious}"
        
    except Exception as e:
        return False, f"Error: {str(e)}"

# Run OSH tests
passed, message = test_osh_calculations()
results.add_result("OSH Calculations", "OSH Metrics Engine", passed, message)

passed, message = test_consciousness_emergence()
results.add_result("OSH Calculations", "Consciousness Emergence", passed, message)

# 3. Test API Server Endpoints
print("\n🌐 3. TESTING API SERVER ENDPOINTS")
print("=" * 60)

def check_api_health():
    """Check if API server is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_api_endpoint(endpoint: str, method: str = "GET", data: Dict = None) -> Tuple[bool, str]:
    """Test a specific API endpoint"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            return False, f"Unsupported method: {method}"
        
        if response.status_code == 200:
            # Try to parse JSON response
            try:
                result = response.json()
                return True, f"Status 200, response type: {type(result).__name__}"
            except:
                return True, f"Status 200, non-JSON response"
        else:
            return False, f"Status {response.status_code}: {response.text[:100]}"
            
    except requests.exceptions.Timeout:
        return False, "Request timeout"
    except requests.exceptions.ConnectionError:
        return False, "Connection error - API server may not be running"
    except Exception as e:
        return False, f"Error: {str(e)}"

# Check if API is running
if check_api_health():
    # Test various endpoints
    api_tests = [
        ("/api/execute", "POST", {"code": "qstate |0⟩;"}),
        ("/api/programs", "GET", None),
        ("/api/osh/calculations", "POST", {"state_data": {"amplitudes": [0.707, 0.707]}}),
        ("/api/gravitational-wave/echo-analysis", "POST", {"signal_data": [0.1, 0.2, 0.3]})
    ]
    
    for endpoint, method, data in api_tests:
        passed, message = test_api_endpoint(endpoint, method, data)
        results.add_result("API Endpoints", f"{method} {endpoint}", passed, message)
else:
    results.add_skip("API Endpoints", "All endpoints", "API server not running")

# 4. Test Frontend-Backend Integration
print("\n🔗 4. TESTING FRONTEND-BACKEND INTEGRATION")
print("=" * 60)

def test_websocket_connection():
    """Test WebSocket connectivity"""
    try:
        import websocket
        
        ws = websocket.WebSocket()
        ws.connect("ws://localhost:5000/ws", timeout=5)
        
        # Send a test message
        test_msg = json.dumps({"type": "ping"})
        ws.send(test_msg)
        
        # Wait for response
        ws.settimeout(5)
        response = ws.recv()
        ws.close()
        
        return True, "WebSocket connection successful"
        
    except Exception as e:
        return False, f"WebSocket error: {str(e)}"

def test_frontend_build():
    """Check if frontend build exists"""
    frontend_dist = project_root / "frontend" / "dist"
    if frontend_dist.exists():
        index_file = frontend_dist / "index.html"
        if index_file.exists():
            return True, "Frontend build exists"
        else:
            return False, "Frontend build incomplete (no index.html)"
    else:
        return False, "Frontend not built (no dist directory)"

# Run integration tests
passed, message = test_websocket_connection()
results.add_result("Integration", "WebSocket Connection", passed, message)

passed, message = test_frontend_build()
results.add_result("Integration", "Frontend Build", passed, message)

# 5. Test Full Pipeline Execution
print("\n🚀 5. TESTING FULL PIPELINE EXECUTION")
print("=" * 60)

def test_full_pipeline():
    """Test complete execution pipeline from code to visualization"""
    try:
        from src.core.compiler import Compiler
        from src.core.interpreter import Interpreter
        from src.core.runtime import Runtime
        from src.visualization.quantum_visualization_engine import QuantumVisualizationEngine
        
        # Simple test program
        test_code = """
        qstate |psi⟩ = |0⟩;
        apply H to |psi⟩[0];
        measure |psi⟩[0] -> result;
        """
        
        # Compile
        compiler = Compiler()
        ast = compiler.compile(test_code)
        
        # Execute
        runtime = Runtime()
        interpreter = Interpreter(runtime)
        result = interpreter.interpret(ast)
        
        # Visualize
        viz_engine = QuantumVisualizationEngine()
        viz_data = viz_engine.prepare_visualization_data(result)
        
        # Check all stages completed
        checks = [
            (ast is not None, "Compilation successful"),
            (result is not None, "Execution successful"),
            (viz_data is not None, "Visualization data generated"),
            (hasattr(result, 'measurements'), "Measurements captured"),
            (len(result.measurements) > 0, "Measurements contain data")
        ]
        
        for check, desc in checks:
            if not check:
                return False, f"Pipeline failed at: {desc}"
        
        return True, "Full pipeline executed successfully"
        
    except Exception as e:
        return False, f"Pipeline error: {str(e)}"

passed, message = test_full_pipeline()
results.add_result("Full Pipeline", "End-to-End Execution", passed, message)

# Test some specific quantum programs through the pipeline
pipeline_programs = [
    ("Bell State Creation", """
        qstate |bell⟩ = |00⟩;
        apply H to |bell⟩[0];
        apply CNOT to |bell⟩[0], |bell⟩[1];
        measure |bell⟩ -> result;
    """),
    ("GHZ State", """
        qstate |ghz⟩ = |000⟩;
        apply H to |ghz⟩[0];
        apply CNOT to |ghz⟩[0], |ghz⟩[1];
        apply CNOT to |ghz⟩[1], |ghz⟩[2];
    """),
    ("Quantum Phase Kickback", """
        qstate |psi⟩ = |0⟩|1⟩;
        apply H to |psi⟩[0];
        apply CNOT to |psi⟩[0], |psi⟩[1];
        apply H to |psi⟩[0];
    """)
]

for name, code in pipeline_programs:
    try:
        compiler = Compiler()
        ast = compiler.compile(code)
        runtime = Runtime()
        interpreter = Interpreter(runtime)
        result = interpreter.interpret(ast)
        
        passed = result is not None
        message = "Executed successfully" if passed else "Execution failed"
        results.add_result("Quantum Programs Pipeline", name, passed, message)
    except Exception as e:
        results.add_result("Quantum Programs Pipeline", name, False, str(e))

# Final Summary
print("\n" + "=" * 60)
print("📊 VALIDATION SUMMARY")
print("=" * 60)

total_tests = results.passed + results.failed + results.skipped

# Group results by category
categories = {}
for detail in results.details:
    cat = detail["category"]
    if cat not in categories:
        categories[cat] = {"passed": 0, "failed": 0, "skipped": 0}
    
    if detail.get("skipped"):
        categories[cat]["skipped"] += 1
    elif detail["passed"]:
        categories[cat]["passed"] += 1
    else:
        categories[cat]["failed"] += 1

# Print category summaries
for cat, stats in categories.items():
    total_cat = stats["passed"] + stats["failed"] + stats["skipped"]
    print(f"\n{cat}:")
    print(f"  ✓ Passed: {stats['passed']}/{total_cat}")
    print(f"  ✗ Failed: {stats['failed']}/{total_cat}")
    if stats["skipped"] > 0:
        print(f"  ⚠️  Skipped: {stats['skipped']}/{total_cat}")

# Overall summary
print(f"\nOVERALL RESULTS:")
print(f"  Total Tests: {total_tests}")
print(f"  ✓ Passed: {results.passed} ({results.passed/total_tests*100:.1f}%)")
print(f"  ✗ Failed: {results.failed} ({results.failed/total_tests*100:.1f}%)")
print(f"  ⚠️  Skipped: {results.skipped} ({results.skipped/total_tests*100:.1f}%)")

# System readiness assessment
print("\n🎯 SYSTEM READINESS ASSESSMENT:")
pass_rate = results.passed / (results.passed + results.failed) if (results.passed + results.failed) > 0 else 0

if pass_rate >= 0.95:
    print("✅ PRODUCTION READY: System is functioning excellently")
    exit_code = 0
elif pass_rate >= 0.80:
    print("⚡ MOSTLY READY: System is functional with minor issues")
    exit_code = 1
elif pass_rate >= 0.60:
    print("⚠️  NEEDS WORK: System has significant issues that need addressing")
    exit_code = 2
else:
    print("🚨 NOT READY: System has critical failures")
    exit_code = 3

# Write detailed report
report_path = project_root / "validation_report.json"
with open(report_path, 'w') as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total_tests": total_tests,
            "passed": results.passed,
            "failed": results.failed,
            "skipped": results.skipped,
            "pass_rate": pass_rate
        },
        "categories": categories,
        "details": results.details
    }, f, indent=2)

print(f"\n📄 Detailed report saved to: {report_path}")

sys.exit(exit_code)