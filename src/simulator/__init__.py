"""
Recursia Simulator Package

This package provides the quantum simulator backend and analytical tools for Recursia.
It implements full software-based simulation of quantum states, observers, entanglement,
teleportation, and OSH-driven coherence/entropy dynamics, with comprehensive reporting
capabilities for simulation analysis and visualization.
"""

# Quantum simulator backend
from .quantum_simulator_backend import (
    QuantumSimulatorBackend,
    QuantumState
)

# Simulation analytics and reporting
# Lazy import to prevent initialization hang
def get_simulation_report_builder():
    """Get SimulationReportBuilder class (lazy import)."""
    from .simulation_report_builder import SimulationReportBuilder
    return SimulationReportBuilder

__all__ = [
    # Simulator backend
    'QuantumSimulatorBackend',
    'QuantumState',
    
    # Reporting tools
    'get_simulation_report_builder'
]