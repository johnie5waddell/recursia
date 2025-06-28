#!/usr/bin/env python3
"""Run the simplified dynamic validation suite"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation.dynamic_empirical_validation_v2 import (
    SimplifiedDynamicValidator, 
    ExperimentConfig
)


def main():
    parser = argparse.ArgumentParser(
        description="Run simplified dynamic validation for OSH"
    )
    
    parser.add_argument(
        '--experiments', '-n', 
        type=int, 
        default=100,
        help='Number of experiments to run (default: 100)'
    )
    
    parser.add_argument(
        '--min-qubits', 
        type=int, 
        default=10,
        help='Minimum number of qubits (default: 10)'
    )
    
    parser.add_argument(
        '--max-qubits', 
        type=int, 
        default=16,
        help='Maximum number of qubits (default: 16)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick test with 20 experiments'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = ExperimentConfig(
        min_qubits=args.min_qubits,
        max_qubits=args.max_qubits,
        time_evolution_steps=20 if args.quick else 50
    )
    
    num_experiments = 20 if args.quick else args.experiments
    
    # Create and run validator
    validator = SimplifiedDynamicValidator(config)
    
    logging.info(f"Starting simplified validation with {num_experiments} experiments")
    results = validator.run_validation(num_experiments=num_experiments)
    
    # Save results
    output_dir = Path("validation_results")
    output_dir.mkdir(exist_ok=True)
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"simplified_validation_{timestamp}.json"
    validator.save_results(str(output_file))
    
    # Print summary
    print("\n" + "="*80)
    print("SIMPLIFIED DYNAMIC VALIDATION RESULTS")
    print("="*80)
    
    print(f"\nSummary:")
    for key, value in results['summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\nConsciousness Emergence by Qubit Count:")
    for qubits, rate in sorted(results.get('qubit_emergence_rates', {}).items()):
        print(f"  {qubits} qubits: {rate:.2%}")
    
    print(f"\nMetric Statistics:")
    for metric, stats in results.get('metric_statistics', {}).items():
        if metric != 'conservation_error':
            print(f"  {metric}: μ={stats['mean']:.3f}, σ={stats['std']:.3f}")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    

if __name__ == "__main__":
    main()