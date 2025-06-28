"""
Recursia Test Suite
==================

Comprehensive test suite for the Recursia quantum consciousness programming language
and OSH (Organic Simulation Hypothesis) framework.

Test Categories:
- Unit Tests: Individual module functionality
- Integration Tests: Cross-module interactions
- Regression Tests: Preventing unintended changes
- Performance Tests: Computational efficiency
- Scientific Validation Tests: Theory verification

Requirements:
- pytest >= 6.0
- numpy >= 1.20
- scipy >= 1.7
- All Recursia core modules

Usage:
    pytest tests/                    # Run all tests
    pytest tests/unit/              # Run unit tests only
    pytest tests/integration/       # Run integration tests
    pytest -v tests/                # Verbose output
    pytest --cov=src tests/         # Coverage report

Author: Johnie Waddell
"""

# Test configuration
RANDOM_SEED = 42
DEFAULT_TOLERANCE = 1e-10
INTEGRATION_TIMEOUT = 30  # seconds
PERFORMANCE_ITERATIONS = 100

# Test data paths
TEST_DATA_DIR = "tests/data"
REFERENCE_RESULTS_DIR = "tests/reference"
TEMP_OUTPUT_DIR = "tests/temp"

# Scientific validation thresholds
CONSCIOUSNESS_THRESHOLD_TEST = 1e-12
OBSERVER_EFFECT_THRESHOLD = 0.01
ENTANGLEMENT_FIDELITY_THRESHOLD = 0.95
MEMORY_COHERENCE_THRESHOLD = 0.8

__version__ = "1.0.0"