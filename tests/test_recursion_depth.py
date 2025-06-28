#!/usr/bin/env python3
"""Test recursion depth calculation in Recursia."""

import requests
import json
import time

# Test recursive depth calculation
recursive_code = """
# Test Recursia program with recursive depth
state qA {
    qubits: 2,
    coherence: 0.95
}

recursive depth 5 {
    # This should increase recursion depth to 5
    print("In recursive block at depth 5")
    measure(qA, 0)
}

# Add nested recursive blocks
recursive depth 3 {
    print("Outer recursive depth 3")
    recursive depth 2 {
        print("Inner recursive depth 2")
        # Total depth should be 3 + 2 = 5
    }
}
"""

print("Testing recursion depth calculation...")
print("Code:")
print(recursive_code)
print("\n" + "="*60 + "\n")

# Send execution request
response = requests.post(
    "http://localhost:8000/api/execute",
    json={"code": recursive_code}
)

if response.status_code == 200:
    result = response.json()
    
    # Extract metrics
    metrics = result.get('metrics', {})
    recursion_depth = metrics.get('depth', 0)
    
    print(f"Execution successful!")
    print(f"Recursion depth: {recursion_depth}")
    print(f"RSP: {metrics.get('rsp', 0):.2f}")
    print(f"Coherence: {metrics.get('coherence', 0):.3f}")
    print(f"Entropy: {metrics.get('entropy', 0):.3f}")
    
    # Check detailed metrics
    detailed_metrics = result.get('metrics_detailed', {})
    if detailed_metrics:
        print(f"\nDetailed metrics:")
        print(f"  Recursion depth: {detailed_metrics.get('recursion_depth', 0)}")
        print(f"  Observer count: {detailed_metrics.get('observer_count', 0)}")
        print(f"  State count: {detailed_metrics.get('state_count', 0)}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)