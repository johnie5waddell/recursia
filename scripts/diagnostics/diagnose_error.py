#!/usr/bin/env python3
"""
Diagnostic script to find the exact source of the record_state error
"""

import sys
import os
import inspect
import importlib.util

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def diagnose_record_state():
    """Diagnose the record_state method signature issue."""
    
    print("=== Diagnosing record_state Error ===\n")
    
    # Import the EmergentPhenomenaDetector
    try:
        from src.physics.emergent_phenomena_detector import EmergentPhenomenaDetector
        print("✓ Successfully imported EmergentPhenomenaDetector")
        
        # Check the record_state method signature
        method = getattr(EmergentPhenomenaDetector, 'record_state', None)
        if method:
            sig = inspect.signature(method)
            print(f"\nrecord_state signature: {sig}")
            print("\nParameters:")
            for name, param in sig.parameters.items():
                print(f"  - {name}: {param}")
        else:
            print("✗ record_state method not found!")
            
    except Exception as e:
        print(f"✗ Error importing EmergentPhenomenaDetector: {e}")
        
    # Check if there are multiple versions of the file
    print("\n\nChecking for duplicate files...")
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file == 'emergent_phenomena_detector.py':
                filepath = os.path.join(root, file)
                print(f"  Found: {filepath}")
                
    # Check the actual file content
    print("\n\nChecking file content...")
    filepath = 'src/physics/emergent_phenomena_detector.py'
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if 'def record_state' in line:
                    print(f"\nFound record_state definition at line {i+1}:")
                    # Print the method signature
                    j = i
                    while j < len(lines) and not lines[j].strip().endswith(':'):
                        print(f"  {lines[j].rstrip()}")
                        j += 1
                    if j < len(lines):
                        print(f"  {lines[j].rstrip()}")
                    break
                    
    # Check where record_state is being called
    print("\n\nSearching for record_state calls...")
    search_dirs = ['src/physics', 'src/core', 'src/visualization']
    
    for search_dir in search_dirs:
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if 'record_state' in content and filepath != 'src/physics/emergent_phenomena_detector.py':
                                lines = content.split('\n')
                                for i, line in enumerate(lines):
                                    if 'record_state' in line:
                                        print(f"\n{filepath}:{i+1}")
                                        # Print context
                                        start = max(0, i-2)
                                        end = min(len(lines), i+10)
                                        for j in range(start, end):
                                            prefix = ">>> " if j == i else "    "
                                            print(f"{prefix}{j+1}: {lines[j]}")
                    except:
                        pass
                        
    # Check running processes
    print("\n\nChecking for running Python processes...")
    import psutil
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['name'] == 'python' or proc.info['name'] == 'python3':
                cmdline = proc.info['cmdline']
                if cmdline and any('recursia' in str(arg).lower() for arg in cmdline):
                    print(f"  PID {proc.info['pid']}: {' '.join(cmdline)}")
        except:
            pass
            
    print("\n\n=== Diagnosis Complete ===")
    print("\nRecommendation: Kill any running Recursia processes and restart them.")
    print("The error is likely due to a running process using old code.")


if __name__ == "__main__":
    diagnose_record_state()