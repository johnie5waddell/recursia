#!/usr/bin/env python3
"""Run the dynamic empirical validation suite with configurable parameters"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.validation.dynamic_empirical_validation import (
    DynamicEmpiricalValidator, 
    ExperimentConfig
)


def main():
    parser = argparse.ArgumentParser(
        description="Run dynamic empirical validation for OSH consciousness emergence"
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
        '--time-steps', 
        type=int, 
        default=100,
        help='Time evolution steps per experiment (default: 100)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='validation_results',
        help='Output directory for results (default: validation_results)'
    )
    
    parser.add_argument(
        '--log-level', 
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--quick', 
        action='store_true',
        help='Run quick validation with reduced parameters'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    if args.quick:
        # Quick test configuration
        config = ExperimentConfig(
            min_qubits=args.min_qubits,
            max_qubits=min(args.max_qubits, 12),
            iterations_per_test=100,
            time_evolution_steps=20,
            temperature_range=(100.0, 300.0),
            noise_levels=[0.001, 0.01],
            recursion_depths=[5, 7, 9],
            enable_uncertainty=True,
            quantum_fluctuation_scale=1e-15,
            measurement_basis_rotation=True,
            environmental_coupling=0.01
        )
        num_experiments = min(args.experiments, 20)
    else:
        # Full configuration
        config = ExperimentConfig(
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            iterations_per_test=1000,
            time_evolution_steps=args.time_steps,
            temperature_range=(0.1, 300.0),
            noise_levels=[0.0001, 0.001, 0.01, 0.1],
            recursion_depths=[5, 7, 9, 11, 13],
            enable_uncertainty=True,
            quantum_fluctuation_scale=1e-15,
            measurement_basis_rotation=True,
            environmental_coupling=0.01
        )
        num_experiments = args.experiments
    
    # Create validator
    validator = DynamicEmpiricalValidator(config)
    
    # Run validation
    logging.info(f"Starting dynamic validation with {num_experiments} experiments")
    if args.quick:
        logging.info("Running in QUICK mode with reduced parameters")
        
    results = validator.run_validation(num_experiments=num_experiments)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"dynamic_validation_{timestamp}.json"
    validator.save_results(str(output_file))
    
    # Print summary
    print("\n" + "="*80)
    print("DYNAMIC EMPIRICAL VALIDATION RESULTS")
    print("="*80)
    
    if 'summary' in results:
        print(f"\nSummary:")
        for key, value in results['summary'].items():
            print(f"  {key}: {value}")
    
    if 'qubit_emergence_rates' in results:
        print(f"\nConsciousness Emergence by Qubit Count:")
        for qubits, rate in sorted(results['qubit_emergence_rates'].items()):
            print(f"  {qubits} qubits: {rate:.2%}")
    
    if 'osh_predictions_confirmed' in results:
        print(f"\nOSH Theory Predictions:")
        for prediction, confirmed in results['osh_predictions_confirmed'].items():
            status = "✓ CONFIRMED" if confirmed else "✗ NOT CONFIRMED"
            print(f"  {prediction}: {status}")
    
    if 'variance_validation' in results:
        print(f"\nVariance Validation:")
        for key, value in results['variance_validation'].items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print(f"\nResults saved to: {output_file}")
    print("="*80)
    
    # Return appropriate exit code
    if results.get('summary', {}).get('emergence_rate', 0) > 0:
        return 0  # Success
    else:
        return 1  # No consciousness emergence
    

if __name__ == "__main__":
    sys.exit(main())