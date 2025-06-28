#!/usr/bin/env python3
"""
Focused validation of Recursia v3 system
Tests core functionality with correct configurations
"""

import sys
import os
import time
import json
import subprocess
import requests
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("🚀 Recursia v3 Focused System Validation")
print("=" * 60)

# Correct API configuration
API_URL = "http://localhost:8080"

# Results tracking
results = []
passed = 0
failed = 0

def test(name: str, func, *args, **kwargs):
    """Run a test and track results"""
    global passed, failed
    print(f"\n▶ Testing: {name}")
    try:
        result = func(*args, **kwargs)
        if result:
            passed += 1
            print(f"  ✓ PASS: {name}")
            results.append({"test": name, "status": "PASS", "details": "Success"})
            return True
        else:
            failed += 1
            print(f"  ✗ FAIL: {name}")
            results.append({"test": name, "status": "FAIL", "details": "Test returned False"})
            return False
    except Exception as e:
        failed += 1
        print(f"  ✗ FAIL: {name} - {str(e)}")
        results.append({"test": name, "status": "FAIL", "details": str(e)})
        return False

# 1. Test Core Quantum Programs
print("\n📚 1. CORE QUANTUM PROGRAM TESTS")
print("-" * 60)

def test_quantum_program_file(filepath):
    """Test execution of a quantum program file"""
    try:
        full_path = project_root / filepath
        if not full_path.exists():
            print(f"    File not found: {filepath}")
            return False
            
        with open(full_path, 'r') as f:
            code = f.read()
        
        # Try to execute via API
        response = requests.post(f"{API_URL}/api/execute", 
                               json={"code": code}, 
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                print(f"    Execution successful")
                return True
            else:
                print(f"    Execution failed: {data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"    API error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"    Error: {str(e)}")
        return False

# Test basic quantum programs
test("Hello Quantum", test_quantum_program_file, "quantum_programs/basic/hello_quantum.recursia")
test("Simple Entanglement", test_quantum_program_file, "quantum_programs/basic/simple_entanglement.recursia")
test("Quantum Coin Flip", test_quantum_program_file, "quantum_programs/basic/quantum_coin_flip.recursia")

# 2. Test OSH Calculations
print("\n🧮 2. OSH CALCULATIONS AND CONSCIOUSNESS")
print("-" * 60)

def test_osh_calculations():
    """Test OSH calculation endpoints"""
    try:
        # Create a test quantum state
        test_state = {
            "amplitudes": [0.707, 0.707],  # Simple superposition
            "n_qubits": 1
        }
        
        response = requests.post(f"{API_URL}/api/osh/calculations",
                               json={"state_data": test_state},
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            metrics = data.get("metrics", {})
            
            # Check for expected metrics
            expected = ["phi_IIT", "entropy_coherence_ratio", "rsp_value", 
                       "gravitational_binding", "consciousness_emergence"]
            
            missing = [m for m in expected if m not in metrics]
            if missing:
                print(f"    Missing metrics: {missing}")
                return False
            
            print(f"    All OSH metrics calculated successfully")
            print(f"    IIT Φ: {metrics.get('phi_IIT', 0):.4f}")
            print(f"    Consciousness: {metrics.get('consciousness_emergence', False)}")
            return True
        else:
            print(f"    API error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"    Error: {str(e)}")
        return False

test("OSH Metrics Calculation", test_osh_calculations)

# 3. Test API Server Health
print("\n🌐 3. API SERVER TESTS")
print("-" * 60)

def test_api_health():
    """Test API server health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def test_api_programs():
    """Test programs listing endpoint"""
    try:
        response = requests.get(f"{API_URL}/api/programs", timeout=5)
        if response.status_code == 200:
            data = response.json()
            program_count = len(data.get("programs", []))
            print(f"    Found {program_count} programs")
            return program_count > 0
        return False
    except:
        return False

test("API Health Check", test_api_health)
test("API Programs Endpoint", test_api_programs)

# 4. Test Frontend-Backend Integration
print("\n🔗 4. INTEGRATION TESTS")
print("-" * 60)

def test_websocket():
    """Test WebSocket connectivity"""
    try:
        import websocket
        ws = websocket.WebSocket()
        ws.connect(f"ws://localhost:8080/ws", timeout=5)
        
        # Send test message
        ws.send(json.dumps({"type": "ping"}))
        ws.settimeout(5)
        response = ws.recv()
        ws.close()
        
        return True
    except Exception as e:
        print(f"    WebSocket error: {str(e)}")
        return False

def test_frontend_files():
    """Check if critical frontend files exist"""
    critical_files = [
        "frontend/package.json",
        "frontend/src/App.tsx",
        "frontend/src/components/QuantumOSHStudio.tsx",
        "frontend/index.html"
    ]
    
    missing = []
    for file in critical_files:
        if not (project_root / file).exists():
            missing.append(file)
    
    if missing:
        print(f"    Missing files: {missing}")
        return False
    
    print(f"    All critical frontend files present")
    return True

test("WebSocket Connection", test_websocket)
test("Frontend File Structure", test_frontend_files)

# 5. Test Full Pipeline
print("\n⚡ 5. FULL PIPELINE TESTS")
print("-" * 60)

def test_simple_execution():
    """Test simple quantum execution through full pipeline"""
    try:
        code = """
        qstate |psi> = |0>;
        apply H to |psi>[0];
        measure |psi>[0] -> result;
        """
        
        response = requests.post(f"{API_URL}/api/execute",
                               json={"code": code},
                               timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                # Check for expected fields
                expected_fields = ["final_state", "measurements", "execution_time"]
                missing = [f for f in expected_fields if f not in data]
                
                if missing:
                    print(f"    Missing response fields: {missing}")
                    return False
                
                print(f"    Execution time: {data.get('execution_time', 0):.3f}s")
                print(f"    Measurements: {len(data.get('measurements', []))} results")
                return True
            else:
                print(f"    Execution failed: {data.get('error')}")
                return False
        else:
            print(f"    API error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"    Error: {str(e)}")
        return False

def test_complex_circuit():
    """Test more complex quantum circuit"""
    try:
        code = """
        qstate |bell> = |00>;
        apply H to |bell>[0];
        apply CNOT to |bell>[0], |bell>[1];
        measure |bell> -> results;
        """
        
        response = requests.post(f"{API_URL}/api/execute",
                               json={"code": code, "iterations": 100},
                               timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                measurements = data.get("measurements", [])
                if len(measurements) >= 100:
                    print(f"    Successfully ran 100 iterations")
                    return True
                else:
                    print(f"    Only got {len(measurements)} measurements")
                    return False
            else:
                print(f"    Execution failed: {data.get('error')}")
                return False
        else:
            print(f"    API error: Status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"    Error: {str(e)}")
        return False

test("Simple Quantum Execution", test_simple_execution)
test("Complex Circuit (Bell State)", test_complex_circuit)

# Summary Report
print("\n" + "=" * 60)
print("📊 VALIDATION SUMMARY")
print("=" * 60)

total = passed + failed
pass_rate = passed / total if total > 0 else 0

print(f"\nTotal Tests: {total}")
print(f"✓ Passed: {passed} ({pass_rate*100:.1f}%)")
print(f"✗ Failed: {failed} ({(1-pass_rate)*100:.1f}%)")

# Categorize results
categories = {}
for result in results:
    test_name = result["test"]
    if "Quantum" in test_name or "program" in test_name.lower():
        cat = "Quantum Programs"
    elif "OSH" in test_name or "Consciousness" in test_name:
        cat = "OSH/Consciousness"
    elif "API" in test_name:
        cat = "API Server"
    elif "WebSocket" in test_name or "Frontend" in test_name:
        cat = "Integration"
    else:
        cat = "Pipeline"
    
    if cat not in categories:
        categories[cat] = {"passed": 0, "failed": 0}
    
    if result["status"] == "PASS":
        categories[cat]["passed"] += 1
    else:
        categories[cat]["failed"] += 1

print("\nBy Category:")
for cat, stats in categories.items():
    cat_total = stats["passed"] + stats["failed"]
    cat_rate = stats["passed"] / cat_total if cat_total > 0 else 0
    print(f"  {cat}: {stats['passed']}/{cat_total} ({cat_rate*100:.1f}%)")

# System Readiness
print("\n🎯 SYSTEM READINESS:")
if pass_rate >= 0.95:
    print("✅ PRODUCTION READY - System is fully operational")
    conclusion = "READY"
elif pass_rate >= 0.80:
    print("⚡ MOSTLY READY - Minor issues present")
    conclusion = "MOSTLY_READY"
elif pass_rate >= 0.60:
    print("⚠️  NEEDS WORK - Several components need attention")
    conclusion = "NEEDS_WORK"
else:
    print("🚨 NOT READY - Critical failures detected")
    conclusion = "NOT_READY"

# Issues summary
if failed > 0:
    print("\n🔧 FAILED TESTS:")
    for result in results:
        if result["status"] == "FAIL":
            print(f"  - {result['test']}: {result['details']}")

# Save detailed report
report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "summary": {
        "total_tests": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "conclusion": conclusion
    },
    "categories": categories,
    "detailed_results": results
}

report_path = project_root / "focused_validation_report.json"
with open(report_path, 'w') as f:
    json.dump(report, f, indent=2)

print(f"\n📄 Detailed report saved to: {report_path}")

# Exit with appropriate code
sys.exit(0 if pass_rate >= 0.60 else 1)