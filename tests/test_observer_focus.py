#!/usr/bin/env python3
"""Test script to debug observer focus issue"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.observer_registry import ObserverRegistry
from src.physics.osh_metrics_calculator import OSHMetricsCalculator

# Create observer registry and add some test observers
registry = ObserverRegistry()

# Register an observer with properties like in the API server
registry.register_observer("alice", "quantum_observer", {
    "observer_type": "quantum_observer",
    "observer_phase": "active",
    "observer_collapse_threshold": 0.7,
    "observer_focus": 0.8
})

registry.register_observer("bob", "quantum_observer", {
    "observer_type": "quantum_observer", 
    "observer_phase": "passive",
    "observer_collapse_threshold": 0.6,
    "observer_focus": 0.6
})

# Get all observers like the API does
observers = registry.get_all_observers()

print("Observer data returned by get_all_observers():")
for obs in observers:
    print(f"  {obs['name']}: focus={obs.get('focus')} (type: {type(obs.get('focus'))}), phase={obs.get('phase')}")

# Test the metrics calculator with this data
calculator = OSHMetricsCalculator()
runtime_data = {
    'observers': observers,
    'states': {},
    'memory_fragments': []
}

try:
    metrics = calculator.calculate_complete_metrics(runtime_data)
    print(f"\nMetrics calculated successfully!")
    print(f"Observer influence: {metrics.observer_influence}")
except Exception as e:
    print(f"\nError calculating metrics: {e}")
    import traceback
    traceback.print_exc()