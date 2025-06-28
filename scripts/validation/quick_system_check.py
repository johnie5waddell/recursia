#!/usr/bin/env python3
"""
Quick system validation for Recursia v3
Focused on testing core functionality without complex dependencies
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("🔍 Recursia v3 Quick System Check")
print("=" * 60)

# Track results
results = {
    "passed": 0,
    "failed": 0,
    "tests": []
}

def test_result(name: str, passed: bool, details: str = ""):
    """Record test result"""
    global results
    if passed:
        results["passed"] += 1
        print(f"✓ {name}: {details}")
    else:
        results["failed"] += 1
        print(f"✗ {name}: {details}")
    
    results["tests"].append({
        "name": name,
        "passed": passed,
        "details": details
    })

# 1. Check Python imports
print("\n1. Testing Core Python Imports...")
print("-" * 40)

try:
    from src.core.direct_parser import DirectParser
    test_result("DirectParser import", True, "Successfully imported")
except Exception as e:
    test_result("DirectParser import", False, str(e))

try:
    from src.core.bytecode_vm import RecursiaVM
    test_result("RecursiaVM import", True, "Successfully imported")
except Exception as e:
    test_result("RecursiaVM import", False, str(e))

try:
    from src.quantum.quantum_state import QuantumState
    test_result("QuantumState import", True, "Successfully imported")
except Exception as e:
    test_result("QuantumState import", False, str(e))

try:
    from src.engines.OSHCalculationEngine import OSHCalculationEngine
    test_result("OSHCalculationEngine import", True, "Successfully imported")
except Exception as e:
    test_result("OSHCalculationEngine import", False, str(e))

# 2. Test Basic Quantum Execution
print("\n2. Testing Basic Quantum Execution...")
print("-" * 40)

try:
    from src.core.direct_parser import DirectParser
    from src.core.bytecode_vm import RecursiaVM
    
    # Simple quantum program
    code = """
    qstate |psi> = |0>;
    apply H to |psi>[0];
    measure |psi>[0] -> result;
    """
    
    parser = DirectParser()
    bytecode = parser.parse(code)
    
    if bytecode:
        vm = RecursiaVM()
        start_time = time.time()
        result = vm.execute(bytecode)
        exec_time = time.time() - start_time
        
        test_result("Basic quantum execution", True, f"Executed in {exec_time:.3f}s")
    else:
        test_result("Basic quantum execution", False, "Failed to parse code")
        
except Exception as e:
    test_result("Basic quantum execution", False, str(e))

# 3. Test File System Programs
print("\n3. Testing Quantum Program Files...")
print("-" * 40)

test_programs = [
    "quantum_programs/basic/hello_quantum.recursia",
    "quantum_programs/basic/simple_entanglement.recursia",
    "quantum_programs/basic/quantum_coin_flip.recursia"
]

for program_path in test_programs:
    full_path = project_root / program_path
    if full_path.exists():
        test_result(f"File exists: {program_path}", True, "Found")
        
        # Try to parse it
        try:
            with open(full_path, 'r') as f:
                code = f.read()
            
            parser = DirectParser()
            bytecode = parser.parse(code)
            
            if bytecode:
                test_result(f"Parse: {program_path}", True, "Valid syntax")
            else:
                test_result(f"Parse: {program_path}", False, "Parse failed")
                
        except Exception as e:
            test_result(f"Parse: {program_path}", False, str(e))
    else:
        test_result(f"File exists: {program_path}", False, "Not found")

# 4. Test API Server
print("\n4. Testing API Server...")
print("-" * 40)

try:
    import requests
    response = requests.get("http://localhost:5000/health", timeout=2)
    if response.status_code == 200:
        test_result("API server health check", True, "Server is running")
        
        # Test execution endpoint
        test_data = {"code": "qstate |0>; measure |0> -> r;"}
        response = requests.post("http://localhost:5000/api/execute", 
                               json=test_data, timeout=5)
        if response.status_code == 200:
            test_result("API execute endpoint", True, "Execution successful")
        else:
            test_result("API execute endpoint", False, f"Status {response.status_code}")
    else:
        test_result("API server health check", False, f"Status {response.status_code}")
except requests.exceptions.ConnectionError:
    test_result("API server health check", False, "Connection refused - server not running")
except Exception as e:
    test_result("API server health check", False, str(e))

# 5. Test OSH Calculations
print("\n5. Testing OSH Calculations...")
print("-" * 40)

try:
    from src.engines.OSHCalculationEngine import OSHCalculationEngine
    from src.quantum.quantum_state import QuantumState
    import numpy as np
    
    # Create a test quantum state
    amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
    qstate = QuantumState(amplitudes)
    
    engine = OSHCalculationEngine()
    metrics = engine.calculate_all_metrics(qstate)
    
    if metrics and isinstance(metrics, dict):
        test_result("OSH calculations", True, f"Generated {len(metrics)} metrics")
        
        # Check key metrics
        key_metrics = ["phi_IIT", "entropy_coherence_ratio", "rsp_value"]
        for metric in key_metrics:
            if metric in metrics:
                test_result(f"OSH metric: {metric}", True, f"Value: {metrics[metric]:.4f}")
            else:
                test_result(f"OSH metric: {metric}", False, "Missing")
    else:
        test_result("OSH calculations", False, "No metrics returned")
        
except Exception as e:
    test_result("OSH calculations", False, str(e))

# 6. Quick Performance Test
print("\n6. Testing Performance...")
print("-" * 40)

try:
    from src.core.direct_parser import DirectParser
    from src.core.bytecode_vm import RecursiaVM
    
    # Test with increasing complexity
    test_cases = [
        ("2-qubit", "qstate |q> = |00>; apply H to |q>[0]; apply CNOT to |q>[0], |q>[1];"),
        ("3-qubit", "qstate |q> = |000>; apply H to |q>[0]; apply CNOT to |q>[0], |q>[1]; apply CNOT to |q>[1], |q>[2];"),
        ("4-qubit", "qstate |q> = |0000>; apply H to |q>[0]; apply H to |q>[1]; apply CNOT to |q>[0], |q>[2]; apply CNOT to |q>[1], |q>[3];")
    ]
    
    for name, code in test_cases:
        parser = DirectParser()
        bytecode = parser.parse(code)
        
        if bytecode:
            vm = RecursiaVM()
            start_time = time.time()
            result = vm.execute(bytecode)
            exec_time = time.time() - start_time
            
            if exec_time < 1.0:  # Should execute in under 1 second
                test_result(f"Performance: {name}", True, f"{exec_time:.3f}s")
            else:
                test_result(f"Performance: {name}", False, f"Too slow: {exec_time:.3f}s")
        else:
            test_result(f"Performance: {name}", False, "Parse failed")
            
except Exception as e:
    test_result("Performance tests", False, str(e))

# Summary
print("\n" + "=" * 60)
print("📊 VALIDATION SUMMARY")
print("=" * 60)

total = results["passed"] + results["failed"]
pass_rate = results["passed"] / total if total > 0 else 0

print(f"\nTotal Tests: {total}")
print(f"✓ Passed: {results['passed']} ({pass_rate*100:.1f}%)")
print(f"✗ Failed: {results['failed']} ({(1-pass_rate)*100:.1f}%)")

# System readiness
print("\n🎯 SYSTEM STATUS:")
if pass_rate >= 0.9:
    print("✅ EXCELLENT: System is working very well")
elif pass_rate >= 0.7:
    print("⚡ GOOD: System is mostly functional")
elif pass_rate >= 0.5:
    print("⚠️  FAIR: System needs some fixes")
else:
    print("🚨 POOR: System has major issues")

# Key issues
if results["failed"] > 0:
    print("\n🔧 KEY ISSUES:")
    failed_tests = [t for t in results["tests"] if not t["passed"]]
    for test in failed_tests[:5]:  # Show first 5 failures
        print(f"  - {test['name']}: {test['details']}")

# Save report
report_path = project_root / "quick_validation_report.json"
with open(report_path, 'w') as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": total,
            "passed": results["passed"],
            "failed": results["failed"],
            "pass_rate": pass_rate
        },
        "tests": results["tests"]
    }, f, indent=2)

print(f"\n📄 Report saved to: {report_path}")

sys.exit(0 if pass_rate >= 0.5 else 1)