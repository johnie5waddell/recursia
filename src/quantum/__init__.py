"""
Quantum module for Recursia programming language.

This module provides quantum computation capabilities for Recursia including:
- Quantum state representation and manipulation
- Quantum register management
- Quantum gate operations
- Measurement and entanglement operations
- Quantum Fourier Transform (QFT) implementations
- Hardware abstraction layer for real quantum devices
- Validation of quantum operations in Recursia programs

The quantum module integrates with multiple quantum hardware providers including:
- IBM Quantum Experience (via Qiskit)
- Rigetti (via PyQuil)
- Google Quantum AI (via Cirq)
- IonQ (experimental support)

Usage:
    from src.quantum import QuantumRegister, QuantumState
    
    # Create a quantum register with 3 qubits
    register = QuantumRegister(3, "main_register")
    
    # Apply quantum gates
    register.apply_gate("H_gate", 0)  # Apply Hadamard to first qubit
    register.apply_gate("CNOT_gate", 1, 0)  # Apply CNOT with control=0, target=1
    
    # Measure qubits
    result = register.measure([0, 1])
"""

# Import core quantum classes
from typing import Dict
from .quantum_register import QuantumRegister
from .quantum_state import QuantumState
from .quantum_hardware_backend import QuantumHardwareBackend
# QuantumValidator disabled - AST-based validation not needed for bytecode
# from .quantum_validator import QuantumValidator

# Version information
__version__ = "0.1.0"

# Define what symbols are exported when using 'from quantum import *'
__all__ = [
    'QuantumRegister',
    'QuantumState',
    'QuantumHardwareBackend',
    # 'QuantumValidator',
]

# Package-level constants
SUPPORTED_GATES = {
    "H_gate", "X_gate", "Y_gate", "Z_gate", "S_gate", "T_gate",
    "CNOT_gate", "CZ_gate", "SWAP_gate", "Toffoli_gate", "Hadamard_gate",
    "PauliX_gate", "PauliY_gate", "PauliZ_gate", "PhaseShift_gate",
    "QFT_gate", "InverseQFT_gate", "RX_gate", "RY_gate", "RZ_gate"
}

SUPPORTED_MEASUREMENT_BASES = {
    "standard_basis", "Z_basis", "X_basis", "Y_basis", "Bell_basis",
    "computational_basis", "hadamard_basis", "pauli_basis"
}

SUPPORTED_HARDWARE_PROVIDERS = {
    "ibm": "IBM Quantum Experience (via Qiskit)",
    "rigetti": "Rigetti Forest (via PyQuil)",
    "google": "Google Quantum AI (via Cirq)",
    "ionq": "IonQ (experimental)"
}

# Helper functions
def get_provider_status():
    """
    Check the availability of quantum hardware providers based on installed packages.
    
    Returns:
        dict: Dictionary with provider names as keys and availability status as values
    """
    status = {}
    
    # Check for IBM Qiskit
    try:
        import qiskit # type: ignore
        status["ibm"] = True
    except ImportError:
        status["ibm"] = False
    
    # Check for Rigetti PyQuil
    try:
        import pyquil # type: ignore
        status["rigetti"] = True
    except ImportError:
        status["rigetti"] = False
    
    # Check for Google Cirq
    try:
        import cirq # type: ignore
        status["google"] = True
    except ImportError:
        status["google"] = False
    
    # Check for IonQ
    try:
        import ionq # type: ignore
        status["ionq"] = True
    except ImportError:
        status["ionq"] = False
    
    return status

def create_hardware_backend(provider="auto", device="auto", credentials=None, options=None):
    """
    Convenience function to create a quantum hardware backend.
    
    Args:
        provider (str): Quantum hardware provider ("ibm", "rigetti", "ionq", "auto")
        device (str): Specific quantum device to use
        credentials (dict): Authentication credentials for the provider
        options (dict): Additional options for the backend
        
    Returns:
        QuantumHardwareBackend: Configured hardware backend instance
    """
    backend = QuantumHardwareBackend(provider, device, credentials, options)
    backend.connect()
    return backend