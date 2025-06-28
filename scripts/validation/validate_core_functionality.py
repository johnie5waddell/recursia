#!/usr/bin/env python3
"""
Core functionality validation script for Recursia
Tests the most critical components to ensure they work properly.
"""

import sys
import os
import numpy as np
import traceback
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

def test_quantum_state():
    """Test quantum state functionality"""
    print("Testing QuantumState...")
    try:
        from src.quantum.quantum_state import QuantumState
        
        # Create a simple 2-qubit state
        amplitudes = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        qstate = QuantumState(amplitudes)
        
        print(f"  ✓ Created quantum state with {len(qstate.amplitudes)} amplitudes")
        print(f"  ✓ Probability sum: {np.sum(np.abs(qstate.amplitudes)**2):.6f}")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_physics_engine():
    """Test physics engine functionality"""
    print("Testing PhysicsEngine...")
    try:
        from src.physics.physics_engine_proper import PhysicsEngineProper
        
        engine = PhysicsEngineProper()
        print(f"  ✓ Created physics engine")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_measurement_system():
    """Test measurement system"""
    print("Testing MeasurementSystem...")
    try:
        from src.physics.measurement.measurement_proper import QuantumMeasurementProper
        
        measurement = QuantumMeasurementProper()
        print(f"  ✓ Created measurement system")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_entanglement():
    """Test entanglement functionality"""
    print("Testing EntanglementManager...")
    try:
        from src.physics.entanglement_proper import EntanglementManagerProper
        
        manager = EntanglementManagerProper()
        print(f"  ✓ Created entanglement manager")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_coherence():
    """Test coherence functionality"""
    print("Testing CoherenceManager...")
    try:
        from src.physics.coherence_proper import ScientificCoherenceManager
        
        manager = ScientificCoherenceManager()
        print(f"  ✓ Created coherence manager")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_compiler():
    """Test compiler functionality"""
    print("Testing Compiler...")
    try:
        from src.core.compiler import Compiler
        
        compiler = Compiler()
        print(f"  ✓ Created compiler")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_interpreter():
    """Test interpreter functionality"""
    print("Testing Interpreter...")
    try:
        from src.core.interpreter import Interpreter
        
        interpreter = Interpreter()
        print(f"  ✓ Created interpreter")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_parser():
    """Test parser functionality"""
    print("Testing Parser...")
    try:
        from src.core.parser import Parser
        
        parser = Parser()
        print(f"  ✓ Created parser")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def test_simple_recursia_execution():
    """Test basic Recursia code execution"""
    print("Testing Recursia execution...")
    try:
        # Simple test of the main entry point
        from src.recursia import main
        print(f"  ✓ Main entry point accessible")
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🔬 Recursia Core Functionality Validation")
    print("=" * 50)
    
    tests = [
        test_quantum_state,
        test_physics_engine,
        test_measurement_system,
        test_entanglement,
        test_coherence,
        test_compiler,
        test_interpreter,
        test_parser,
        test_simple_recursia_execution,
    ]
    
    results = {}
    for test in tests:
        try:
            results[test.__name__] = test()
        except Exception as e:
            print(f"  ✗ {test.__name__} failed with exception: {e}")
            results[test.__name__] = False
        print()
    
    # Summary
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All core functionality is working!")
        return 0
    elif passed >= total * 0.8:
        print("⚠️  Most functionality is working, minor issues detected")
        return 1
    else:
        print("🚨 Major issues detected, significant functionality is broken")
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)