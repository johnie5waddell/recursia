#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum State Implementation for Recursia

This module provides a comprehensive quantum state implementation for the Recursia framework.
It implements quantum states with support for gate operations, measurement, entanglement,
teleportation, and density matrix computation, while tracking coherence and entropy.

The QuantumState class is a foundational component of the quantum simulation aspects of
the Organic Simulation Hypothesis (OSH) paradigm, supporting both isolated and entangled
quantum states with a rich set of properties and operations.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        # Add ndarray type for compatibility
        ndarray = list
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from enum import Enum
import math
import cmath
import logging

from src.physics.constants import (
    NumericalParameters, CoherenceParameters,
    ConsciousnessConstants, DecoherenceRates
)

# Setup logging
logger = logging.getLogger(__name__)

class BasisState(Enum):
    """Standard qubit basis states for initialization."""
    ZERO = "|0>"
    ONE = "|1>"
    PLUS = "|+>"
    MINUS = "|->"
    PLUS_I = "|i>"
    MINUS_I = "|-i>"
    
class StateType(Enum):
    """Types of quantum states supported by the system."""
    QUANTUM = "quantum_type"
    SUPERPOSITION = "superposition_type"
    ENTANGLED = "entangled_type"
    MIXED = "mixed_type"
    CLASSICAL = "classical_type"
    COMPOSITE = "composite_type"
    MEASUREMENT = "measurement_type"
    FIELD = "field_type"
    
class QuantumState:
    """
    A comprehensive quantum state implementation supporting full quantum operations.
    
    Attributes:
        name (str): Unique identifier for the quantum state
        num_qubits (int): Number of qubits in the state
        dimension (int): Dimension of the state space (2^num_qubits)
        state_vector (np.ndarray): Complex amplitudes of the quantum state
        density_matrix (np.ndarray): Density matrix representation (computed on demand)
        is_entangled (bool): Flag indicating whether this state is entangled
        entangled_with (Set[str]): Set of state names this state is entangled with
        coherence (float): Coherence metric [0.0-1.0] 
        entropy (float): Von Neumann entropy [0.0-1.0]
        state_type (StateType): Type classification of the quantum state
        creation_time (float): Timestamp when the state was created
        measurement_history (List[Dict]): Record of measurement outcomes
        metadata (Dict): Additional properties and metadata
    """
    
    def __init__(self, name: str, num_qubits: int, initial_state: Optional[str] = None, 
                 state_type: Union[str, StateType] = StateType.QUANTUM):
        """
        Initialize a new quantum state with specified number of qubits.
        
        Args:
            name: Unique identifier for this quantum state
            num_qubits: Number of qubits in the state
            initial_state: Optional string specifying initial state (e.g., "|0>", "|+>")
            state_type: Type of quantum state to create
        
        Raises:
            ValueError: If num_qubits <= 0 or state_type is invalid
        """
        if num_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {num_qubits}")
            
        self.name = name
        self.id = str(uuid.uuid4())
        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        
        # Set state type
        if isinstance(state_type, str):
            try:
                self.state_type = StateType(state_type)
            except ValueError:
                # Default to QUANTUM if not found
                logger.warning(f"Unknown state type: {state_type}, defaulting to quantum_type")
                self.state_type = StateType.QUANTUM
        else:
            self.state_type = state_type
            
        # Initialize to |0...0⟩ state
        self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
        self.state_vector[0] = 1.0
        
        # Apply initial state if specified
        if initial_state:
            self._apply_initial_state(initial_state)
            
        # Track entanglement
        self.is_entangled = False
        self.entangled_with = set()
        
        # Properties derived from state
        self.coherence = ConsciousnessConstants.DEFAULT_COHERENCE  # OSH default: 0.95
        self.entropy = ConsciousnessConstants.DEFAULT_ENTROPY      # OSH default: 0.05
        
        # Track gate operations for entropy production
        self.gate_count = 0
        self._operation_count = 0
        self.measurement_count = 0
        self.decoherence_rate = 0.01  # Default decoherence rate
        
        # Runtime metadata
        import time
        self.creation_time = time.time()  # Use Unix timestamp for compatibility
        self.last_update_time = self.creation_time
        self.measurement_history = []
        self.gate_history = []
        self.metadata = {
            "state_type": self.state_type.value,
            "created_at": str(self.creation_time),
            "is_pure": True,
            "dimension": self.dimension
        }
        
        # History size limits to prevent memory leaks
        self.MAX_HISTORY_SIZE = 1000  # Configurable limit
        
        # Internal cache - invalidated on state changes
        self._density_matrix_cache = None
        self._bloch_vector_cache = None
        
        # Additional quantum properties
        self.phase = 0.0
        self.spin = None  # For specific spin systems
        
        # Update internal metrics and properties
        self._update_properties()
        
    def _apply_initial_state(self, initial_state: str) -> bool:
        """
        Initialize the state vector according to the specified initial state.
        
        Args:
            initial_state: String representation of desired state like "|0>", "|1>", "|+>", etc.
            
        Returns:
            bool: True if initialization was successful
            
        Note:
            For multi-qubit states, use formats like "|01>", "|00>", "|11>", etc.
        """
        # Normalize the string format
        initial_state = initial_state.strip().replace(" ", "")
        
        # Handle special single-qubit states
        if initial_state in ["|0>", "0", "|0⟩"]:
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[0] = 1.0
                return True
            else:
                # Multi-qubit |0...0> state
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[0] = 1.0
                return True
                
        elif initial_state in ["|1>", "1", "|1⟩"]:
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[1] = 1.0
                return True
            else:
                # For multi-qubit, interpret as |1000...>
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[2**(self.num_qubits-1)] = 1.0
                return True
        
        elif initial_state in ["|+>", "+", "|+⟩"]:
            # Prepare |+> = (|0> + |1>)/sqrt(2) for each qubit
            self.state_vector = np.ones(self.dimension, dtype=np.complex128) / np.sqrt(self.dimension)
            return True
            
        elif initial_state in ["|->" , "-", "|-⟩"]:
            # Prepare |-> = (|0> - |1>)/sqrt(2) for each qubit
            self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
            for i in range(self.dimension):
                # Count number of 1s in binary representation
                num_ones = bin(i).count('1')
                self.state_vector[i] = (-1)**num_ones / np.sqrt(self.dimension)
            return True
            
        elif initial_state in ["|i>", "i", "|i⟩"]:
            # Prepare |i> = (|0> + i|1>)/sqrt(2) for each qubit
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[1] = 1j / np.sqrt(2)
                return True
            else:
                # Not easily generalizable to multi-qubit
                logger.warning("Multi-qubit |i> state initialization not implemented; using |0...0>")
                return False
                
        elif initial_state in ["|-i>", "-i", "|-i⟩"]:
            # Prepare |-i> = (|0> - i|1>)/sqrt(2) for each qubit
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[1] = -1j / np.sqrt(2)
                return True
            else:
                # Not easily generalizable to multi-qubit
                logger.warning("Multi-qubit |-i> state initialization not implemented; using |0...0>")
                return False
        
        # Handle binary string input for multi-qubit states like "|010>" or "010"
        elif initial_state.startswith("|") and initial_state.endswith(">"):
            # Strip the |> symbols
            bit_str = initial_state[1:-1]
            if all(bit in '01' for bit in bit_str):
                if len(bit_str) == self.num_qubits:
                    # Convert binary string to integer index
                    index = int(bit_str, 2)
                    self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                    self.state_vector[index] = 1.0
                    return True
                else:
                    logger.warning(f"Bit string length {len(bit_str)} doesn't match num_qubits {self.num_qubits}")
            else:
                logger.warning(f"Invalid bit string: {bit_str}")
        
        # Support bare binary strings like "010" without |> symbols
        elif all(bit in '01' for bit in initial_state):
            if len(initial_state) == self.num_qubits:
                # Convert binary string to integer index
                index = int(initial_state, 2)
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[index] = 1.0
                return True
            else:
                logger.warning(f"Bit string length {len(initial_state)} doesn't match num_qubits {self.num_qubits}")
        
        # Handle Bell states for 2-qubit systems
        elif initial_state.lower() in ["bell", "|bell>", "bell_state"]:
            if self.num_qubits == 2:
                # Create Bell state (|00> + |11>)/sqrt(2)
                self.state_vector = np.zeros(4, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[3] = 1.0 / np.sqrt(2)
                return True
            else:
                logger.warning(f"Bell state requires 2 qubits, but state has {self.num_qubits}")
                return False
                
        # Handle GHZ states for 3+ qubit systems
        elif initial_state.lower() in ["ghz", "|ghz>", "ghz_state"]:
            if self.num_qubits >= 3:
                # Create GHZ state (|00...0> + |11...1>)/sqrt(2)
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[-1] = 1.0 / np.sqrt(2)
                return True
            else:
                logger.warning(f"GHZ state requires at least 3 qubits, but state has {self.num_qubits}")
                return False
                
        # Handle W states for 3+ qubit systems
        elif initial_state.lower() in ["w", "|w>", "w_state"]:
            if self.num_qubits >= 3:
                # Create W state (|100...0> + |010...0> + ... + |000...1>)/sqrt(n)
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                norm = 1.0 / np.sqrt(self.num_qubits)
                for i in range(self.num_qubits):
                    # Set amplitude for states with exactly one '1'
                    idx = 2**i
                    self.state_vector[idx] = norm
                return True
            else:
                logger.warning(f"W state requires at least 3 qubits, but state has {self.num_qubits}")
                return False
        
        # Unknown state format
        logger.warning(f"Unrecognized initial state format: {initial_state}")
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state vector.
        
        Returns:
            np.ndarray: Complex amplitudes of the quantum state
        """
        return self.state_vector.copy()
    
    def set_state_vector(self, state_vector: np.ndarray) -> bool:
        """
        Set the state vector directly.
        
        Args:
            state_vector: Complex amplitudes to set as the new state
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            The state vector will be normalized automatically.
        """
        if state_vector.shape != (self.dimension,):
            logger.error(f"State vector shape {state_vector.shape} doesn't match expected dimension {self.dimension}")
            return False
            
        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm < NumericalParameters.EIGENVALUE_CUTOFF:
            logger.error("State vector has near-zero norm, cannot normalize")
            return False
            
        self.state_vector = state_vector / norm
        self._invalidate_caches()
        self._update_properties()
        self.last_update_time = np.datetime64('now')
        
        return True
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Get the density matrix representation.
        
        Returns:
            np.ndarray: Density matrix ρ = |ψ⟩⟨ψ|
        """
        if self._density_matrix_cache is None:
            # For pure states: ρ = |ψ⟩⟨ψ|
            # Reshape to column vector and compute outer product
            psi = self.state_vector.reshape(-1, 1)
            self._density_matrix_cache = np.dot(psi, psi.conj().T)
        
        return self._density_matrix_cache.copy()
    
    def set_density_matrix(self, density_matrix: np.ndarray) -> bool:
        """
        Set state directly using a density matrix.
        
        Args:
            density_matrix: Square complex matrix of size dimension × dimension
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            If the density matrix represents a mixed state, the state_vector
            will be set to the dominant eigenvector.
        """
        expected_shape = (self.dimension, self.dimension)
        if density_matrix.shape != expected_shape:
            logger.error(f"Density matrix shape {density_matrix.shape} doesn't match expected {expected_shape}")
            return False
            
        # Verify properties of a valid density matrix
        if not self._is_valid_density_matrix(density_matrix):
            logger.error("Invalid density matrix")
            return False
            
        # Store the density matrix
        self._density_matrix_cache = density_matrix.copy()
        
        # Extract the principal eigenvector for the state vector
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
        # Sort by eigenvalue in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Set state vector to the principal eigenvector
        self.state_vector = eigenvectors[:, 0]
        
        # Update metadata
        self.metadata["is_pure"] = np.isclose(eigenvalues[0], 1.0)
        
        self._update_properties()
        self.last_update_time = np.datetime64('now')
        
        return True
    
    def _is_valid_density_matrix(self, matrix: np.ndarray, tolerance: float = NumericalParameters.EIGENVALUE_CUTOFF) -> bool:
        """
        Check if a matrix is a valid density matrix.
        
        Args:
            matrix: Matrix to validate
            tolerance: Numerical error tolerance
            
        Returns:
            bool: True if it's a valid density matrix
        """
        # Check if Hermitian
        if not np.allclose(matrix, matrix.conj().T, atol=tolerance):
            logger.warning("Density matrix is not Hermitian")
            return False
            
        # Check if trace is 1
        trace = np.trace(matrix).real
        if not np.isclose(trace, 1.0, atol=tolerance):
            logger.warning(f"Density matrix trace is {trace}, not 1.0")
            return False
            
        # Check if positive semidefinite
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues < -tolerance):
            logger.warning("Density matrix is not positive semidefinite")
            return False
            
        return True
    
    def apply_gate(self, gate: str, target_qubits: Union[List[int], int], 
                  control_qubits: Optional[Union[List[int], int]] = None,
                  params: Optional[Union[List[float], float]] = None) -> bool:
        """
        Apply a quantum gate to the state. This is a high-level interface that relies on
        external gate operations to be implemented.
        
        Args:
            gate: Name of the gate to apply
            target_qubits: Indices of target qubits
            control_qubits: Optional indices of control qubits
            params: Optional parameters for parameterized gates
            
        Returns:
            bool: True if gate was successfully applied
            
        Note:
            This is a stub that should be connected to GateOperations in a full implementation.
            Here we just record that the gate was requested and update properties.
        """
        # Validate qubit indices
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
            
        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
            
        for q in target_qubits or []:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Target qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        for q in control_qubits or []:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Control qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
        
        # Record the gate application in history
        gate_record = {
            "gate": gate,
            "target_qubits": target_qubits,
            "control_qubits": control_qubits,
            "params": params,
            "time": np.datetime64('now')
        }
        self.gate_history.append(gate_record)
        
        # In a full implementation, this would call the appropriate gate operation
        # For now, just update properties as if the gate were applied
        self._invalidate_caches()
        self._update_properties()
        self.last_update_time = np.datetime64('now')
        
        # Track gate operations for entropy production
        self.gate_count += 1
        self._operation_count += 1
        
        return True
    
    def measure(self, qubits: Optional[Union[List[int], int]] = None, 
                basis: Optional[str] = None) -> Dict[str, Any]:
        """
        Measure the quantum state.
        
        Args:
            qubits: Specific qubits to measure, or None for all qubits
            basis: Measurement basis, default is "Z_basis" (computational)
            
        Returns:
            dict: Measurement result with outcome, probabilities, etc.
            
        Note:
            This is a stub that should be connected to MeasurementOperations in a full implementation.
        """
        # Default to measuring all qubits in computational basis
        if qubits is None:
            qubits = list(range(self.num_qubits))
        elif isinstance(qubits, int):
            qubits = [qubits]
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return {"error": f"Invalid qubit index {q}"}
                
        # Compute outcome probabilities based on state vector
        probabilities = np.abs(self.state_vector)**2
        
        # Select an outcome based on probabilities
        outcome_idx = np.random.choice(self.dimension, p=probabilities)
        
        # Convert to binary string representation
        outcome = format(outcome_idx, f'0{self.num_qubits}b')
        
        # Filter to just the requested qubits
        measured_outcome = ''.join(outcome[self.num_qubits - 1 - q] for q in sorted(qubits))
        
        # Create result dictionary
        result = {
            "outcome": measured_outcome,
            "probabilities": {format(i, f'0{len(qubits)}b'): float(p) 
                              for i, p in enumerate(probabilities) if p > NumericalParameters.EIGENVALUE_CUTOFF},
            "value": int(measured_outcome, 2),
            "qubits": qubits,
            "basis": basis or "Z_basis",
            "time": np.datetime64('now')
        }
        
        # Record in measurement history
        self.measurement_history.append(result)
        self.measurement_count += 1
        
        # If this were a full implementation, we would collapse the state
        # based on the measurement outcome
        
        return result
    
    def entangle_with(self, other_state: 'QuantumState', 
                       qubits1: Optional[List[int]] = None,
                       qubits2: Optional[List[int]] = None,
                       method: Optional[str] = None) -> bool:
        """
        Entangle this state with another quantum state.
        
        Args:
            other_state: The quantum state to entangle with
            qubits1: Qubits from this state to entangle
            qubits2: Qubits from other state to entangle
            method: Entanglement method/protocol
            
        Returns:
            bool: True if entanglement was successful
            
        Note:
            This is a stub that would be connected to EntanglementManager.
            Here we just record the entanglement relationship.
        """
        if other_state.name == self.name:
            logger.error("Cannot entangle a state with itself")
            return False
            
        # Mark both states as entangled
        self.is_entangled = True
        other_state.is_entangled = True
        
        # Add to each other's entangled_with sets
        self.entangled_with.add(other_state.name)
        other_state.entangled_with.add(self.name)
        
        # Record in metadata
        if "entangled_with" not in self.metadata:
            self.metadata["entangled_with"] = []
        if other_state.name not in self.metadata["entangled_with"]:
            self.metadata["entangled_with"].append(other_state.name)
            
        # Update properties
        self._update_properties()
        other_state._update_properties()
        
        return True
    
    def teleport_to(self, destination_state: 'QuantumState',
                    source_qubit: int = 0,
                    destination_qubit: int = 0) -> bool:
        """
        Teleport a qubit state to another quantum state.
        
        Args:
            destination_state: Target state to teleport to
            source_qubit: Qubit index in this state to teleport
            destination_qubit: Qubit index in destination to teleport to
            
        Returns:
            bool: True if teleportation was successful
            
        Note:
            This is a stub for the teleportation protocol.
        """
        # Validate qubit indices
        if source_qubit < 0 or source_qubit >= self.num_qubits:
            logger.error(f"Source qubit index {source_qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        if destination_qubit < 0 or destination_qubit >= destination_state.num_qubits:
            logger.error(f"Destination qubit index {destination_qubit} out of range [0, {destination_state.num_qubits-1}]")
            return False
            
        # Check if the states are entangled (required for teleportation)
        if destination_state.name not in self.entangled_with:
            logger.error(f"States must be entangled for teleportation")
            return False
        
        # In a full implementation, this would perform the actual teleportation protocol
        # which requires entangled qubits, Bell measurement, and classical communication
        
        # For now, just update some properties to simulate the effect
        self._update_properties()
        destination_state._update_properties()
        
        # Log the teleportation event
        logger.info(f"Teleported qubit {source_qubit} from {self.name} to qubit {destination_qubit} in {destination_state.name}")
        
        return True
    
    def reset(self, qubits: Optional[List[int]] = None) -> bool:
        """
        Reset specified qubits or the entire state to |0...0>.
        
        Args:
            qubits: List of qubit indices to reset, or None for all qubits
            
        Returns:
            bool: True if reset was successful
        """
        if qubits is None:
            # Reset entire state
            self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
            self.state_vector[0] = 1.0
            self._invalidate_caches()
            self._update_properties()
            return True
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        # For a partial reset, we could apply projection operators
        # Here's a simplified implementation for single qubit reset
        if len(qubits) == 1:
            qubit = qubits[0]
            qubit_states = np.zeros(self.dimension, dtype=np.complex128)
            
            # Find all basis states where this qubit is |0>
            for i in range(self.dimension):
                # Check if the qubit is in state |0>
                # (i.e., the qubit'th bit from the right is 0)
                if (i >> qubit) & 1 == 0:
                    bin_idx = format(i, f'0{self.num_qubits}b')
                    # Calculate probability of this state
                    qubit_states[i] = self.state_vector[i]
            
            # Normalize
            norm = np.linalg.norm(qubit_states)
            if norm > NumericalParameters.EIGENVALUE_CUTOFF:
                self.state_vector = qubit_states / norm
                self._invalidate_caches()
                self._update_properties()
                return True
            else:
                # If norm is too small, reset to |0...0>
                return self.reset()
                
        # For multiple qubits, we'd need a more complex implementation
        # that projects onto the subspace where all specified qubits are |0>
        
        logger.warning("Reset of multiple specific qubits not fully implemented; resetting all qubits")
        return self.reset()
    
    def get_bloch_vector(self, qubit: int = 0) -> Tuple[float, float, float]:
        """
        Calculate the Bloch sphere coordinates for a single qubit.
        
        Args:
            qubit: Index of the qubit to analyze
            
        Returns:
            Tuple[float, float, float]: (x, y, z) coordinates on the Bloch sphere
            
        Note:
            Only applies to single qubits or specific qubits in a multi-qubit system.
        """
        if qubit < 0 or qubit >= self.num_qubits:
            logger.error(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
            return (0.0, 0.0, 0.0)
            
        if self.num_qubits == 1:
            # For single qubit, use the state vector directly
            rho = self.get_density_matrix()
        else:
            # For multi-qubit, get reduced density matrix for the target qubit
            rho = self._get_reduced_density_matrix(qubit)
            
        # Bloch vector components from density matrix
        x = 2.0 * rho[0, 1].real
        y = 2.0 * rho[0, 1].imag
        z = rho[0, 0] - rho[1, 1]
        
        return (x, y, z)
    
    def _get_reduced_density_matrix(self, qubit: int) -> np.ndarray:
        """
        Calculate the reduced density matrix for a specific qubit.
        
        Args:
            qubit: Index of the qubit to get reduced density matrix for
            
        Returns:
            np.ndarray: 2x2 reduced density matrix
        """
        # Start with the full density matrix
        rho = self.get_density_matrix()
        
        # Trace out all qubits except the target
        for q in range(self.num_qubits):
            if q != qubit:
                # Partial trace over qubit q
                dim_keep = 2  # Dimension of the qubit we're keeping
                dim_trace = 2  # Dimension of the qubit we're tracing out
                
                # This is a simplified approach for a single-qubit reduction
                # In a full implementation, this would be a proper partial trace
                # over the tensor product space
                
                # For simplicity, we'll calculate key expectation values directly
                rho_reduced = np.zeros((2, 2), dtype=np.complex128)
                
                # Calculate expectation values for Pauli matrices
                # <Z> = Pr(|0>) - Pr(|1>)
                z_expectation = 0.0
                # <X> = Re(<0|ρ|1>)
                x_expectation = 0.0
                # <Y> = Im(<0|ρ|1>)
                y_expectation = 0.0
                
                # Sum over all basis states
                for i in range(self.dimension):
                    # Check if the target qubit is 0 or 1 in this basis state
                    is_qubit_0 = ((i >> qubit) & 1) == 0
                    prob = np.abs(self.state_vector[i])**2
                    
                    if is_qubit_0:
                        z_expectation += prob
                    else:
                        z_expectation -= prob
                        
                    # For x_expectation and y_expectation, we need superposition calculations
                    # which would require full density matrix treatment
                
                # Convert expectation values to reduced density matrix
                rho_reduced[0, 0] = (1.0 + z_expectation) / 2.0
                rho_reduced[1, 1] = (1.0 - z_expectation) / 2.0
                
                # For a proper implementation, x and y components would be calculated here
                
                return rho_reduced
    
    def __del__(self):
        """Clean up resources when state is garbage collected."""
        try:
            self.measurement_history.clear()
            self.gate_history.clear()
            self.entangled_with.clear()
            self._density_matrix_cache = None
            if hasattr(self, '_bloch_vector_cache') and self._bloch_vector_cache:
                self._bloch_vector_cache.clear()
            self.state_vector = None
        except:
            pass  # Ignore errors during cleanup#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum State Implementation for Recursia

This module provides a comprehensive quantum state implementation for the Recursia framework.
It implements quantum states with support for gate operations, measurement, entanglement,
teleportation, and density matrix computation, while tracking coherence and entropy.

The QuantumState class is a foundational component of the quantum simulation aspects of
the Organic Simulation Hypothesis (OSH) paradigm, supporting both isolated and entangled
quantum states with a rich set of properties and operations.
"""

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        # Add ndarray type for compatibility
        ndarray = list
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
import uuid
from typing import Dict, List, Optional, Tuple, Union, Any, Set
from enum import Enum
import math
import cmath
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class BasisState(Enum):
    """Standard qubit basis states for initialization."""
    ZERO = "|0>"
    ONE = "|1>"
    PLUS = "|+>"
    MINUS = "|->"
    PLUS_I = "|i>"
    MINUS_I = "|-i>"
    
class StateType(Enum):
    """Types of quantum states supported by the system."""
    QUANTUM = "quantum_type"
    SUPERPOSITION = "superposition_type"
    ENTANGLED = "entangled_type"
    MIXED = "mixed_type"
    CLASSICAL = "classical_type"
    COMPOSITE = "composite_type"
    MEASUREMENT = "measurement_type"
    FIELD = "field_type"
    
class QuantumState:
    """
    A comprehensive quantum state implementation supporting full quantum operations.
    
    Attributes:
        name (str): Unique identifier for the quantum state
        num_qubits (int): Number of qubits in the state
        dimension (int): Dimension of the state space (2^num_qubits)
        state_vector (np.ndarray): Complex amplitudes of the quantum state
        density_matrix (np.ndarray): Density matrix representation (computed on demand)
        is_entangled (bool): Flag indicating whether this state is entangled
        entangled_with (Set[str]): Set of state names this state is entangled with
        coherence (float): Coherence metric [0.0-1.0] 
        entropy (float): Von Neumann entropy [0.0-1.0]
        state_type (StateType): Type classification of the quantum state
        creation_time (float): Timestamp when the state was created
        measurement_history (List[Dict]): Record of measurement outcomes
        metadata (Dict): Additional properties and metadata
    """
    
    def __init__(self, name: str, num_qubits: int, initial_state: Optional[str] = None, 
                 state_type: Union[str, StateType] = StateType.QUANTUM):
        """
        Initialize a new quantum state with specified number of qubits.
        
        Args:
            name: Unique identifier for this quantum state
            num_qubits: Number of qubits in the state
            initial_state: Optional string specifying initial state (e.g., "|0>", "|+>")
            state_type: Type of quantum state to create
        
        Raises:
            ValueError: If num_qubits <= 0 or state_type is invalid
        """
        if num_qubits <= 0:
            raise ValueError(f"Number of qubits must be positive, got {num_qubits}")
            
        self.name = name
        self.id = str(uuid.uuid4())
        self.num_qubits = num_qubits
        self.dimension = 2**num_qubits
        
        # Set state type
        if isinstance(state_type, str):
            try:
                self.state_type = StateType(state_type)
            except ValueError:
                # Default to QUANTUM if not found
                logger.warning(f"Unknown state type: {state_type}, defaulting to quantum_type")
                self.state_type = StateType.QUANTUM
        else:
            self.state_type = state_type
            
        # Initialize to |0...0⟩ state
        self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
        self.state_vector[0] = 1.0
        
        # Apply initial state if specified
        if initial_state:
            self._apply_initial_state(initial_state)
            
        # Track entanglement
        self.is_entangled = False
        self.entangled_with = set()
        
        # Properties derived from state
        self.coherence = ConsciousnessConstants.DEFAULT_COHERENCE  # OSH default: 0.95
        self.entropy = ConsciousnessConstants.DEFAULT_ENTROPY      # OSH default: 0.05
        
        # Track gate operations for entropy production
        self.gate_count = 0
        self._operation_count = 0
        self.measurement_count = 0
        self.decoherence_rate = 0.01  # Default decoherence rate
        
        # Runtime metadata
        import time
        self.creation_time = time.time()  # Use Unix timestamp for compatibility
        self.last_update_time = self.creation_time
        self.measurement_history = []
        self.gate_history = []
        self.metadata = {
            "state_type": self.state_type.value,
            "created_at": str(self.creation_time),
            "is_pure": True,
            "dimension": self.dimension
        }
        
        # Internal cache - invalidated on state changes
        self._density_matrix_cache = None
        self._bloch_vector_cache = {}  # Dict keyed by qubit index
        self._reduced_density_cache = {}  # Dict keyed by qubit index
        
        # Additional quantum properties
        self.phase = 0.0
        self.spin = None  # For specific spin systems
        self.fidelity_history = []  # Track state fidelity over time
        
        # Update internal metrics and properties
        self._update_properties()
        
    def _apply_initial_state(self, initial_state: str) -> bool:
        """
        Initialize the state vector according to the specified initial state.
        
        Args:
            initial_state: String representation of desired state like "|0>", "|1>", "|+>", etc.
            
        Returns:
            bool: True if initialization was successful
            
        Note:
            For multi-qubit states, use formats like "|01>", "|00>", "|11>", etc.
        """
        # Normalize the string format
        initial_state = initial_state.strip().replace(" ", "")
        
        # Handle special single-qubit states
        if initial_state in ["|0>", "0", "|0⟩"]:
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[0] = 1.0
                return True
            else:
                # Multi-qubit |0...0> state
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[0] = 1.0
                return True
                
        elif initial_state in ["|1>", "1", "|1⟩"]:
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[1] = 1.0
                return True
            else:
                # For multi-qubit, interpret as |1000...>
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[2**(self.num_qubits-1)] = 1.0
                return True
        
        elif initial_state in ["|+>", "+", "|+⟩"]:
            # Prepare |+> = (|0> + |1>)/sqrt(2) for each qubit
            self.state_vector = np.ones(self.dimension, dtype=np.complex128) / np.sqrt(self.dimension)
            return True
            
        elif initial_state in ["|->" , "-", "|-⟩"]:
            # Prepare |-> = (|0> - |1>)/sqrt(2) for each qubit
            self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
            for i in range(self.dimension):
                # Count number of 1s in binary representation
                num_ones = bin(i).count('1')
                self.state_vector[i] = (-1)**num_ones / np.sqrt(self.dimension)
            return True
            
        elif initial_state in ["|i>", "i", "|i⟩"]:
            # Prepare |i> = (|0> + i|1>)/sqrt(2) for each qubit
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[1] = 1j / np.sqrt(2)
                return True
            else:
                # For multi-qubit systems, tensor product of all |i> states
                single_i_state = np.array([1.0 / np.sqrt(2), 1j / np.sqrt(2)], dtype=np.complex128)
                state = single_i_state
                
                for _ in range(self.num_qubits - 1):
                    state = np.kron(state, single_i_state)
                
                self.state_vector = state
                return True
                
        elif initial_state in ["|-i>", "-i", "|-i⟩"]:
            # Prepare |-i> = (|0> - i|1>)/sqrt(2) for each qubit
            if self.num_qubits == 1:
                self.state_vector = np.zeros(2, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[1] = -1j / np.sqrt(2)
                return True
            else:
                # For multi-qubit systems, tensor product of all |-i> states
                single_neg_i_state = np.array([1.0 / np.sqrt(2), -1j / np.sqrt(2)], dtype=np.complex128)
                state = single_neg_i_state
                
                for _ in range(self.num_qubits - 1):
                    state = np.kron(state, single_neg_i_state)
                
                self.state_vector = state
                return True
        
        # Handle binary string input for multi-qubit states like "|010>" or "010"
        elif initial_state.startswith("|") and initial_state.endswith(">"):
            # Strip the |> symbols
            bit_str = initial_state[1:-1]
            if all(bit in '01' for bit in bit_str):
                if len(bit_str) == self.num_qubits:
                    # Convert binary string to integer index
                    index = int(bit_str, 2)
                    self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                    self.state_vector[index] = 1.0
                    return True
                else:
                    logger.warning(f"Bit string length {len(bit_str)} doesn't match num_qubits {self.num_qubits}")
            else:
                logger.warning(f"Invalid bit string: {bit_str}")
        
        # Support bare binary strings like "010" without |> symbols
        elif all(bit in '01' for bit in initial_state):
            if len(initial_state) == self.num_qubits:
                # Convert binary string to integer index
                index = int(initial_state, 2)
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[index] = 1.0
                return True
            else:
                logger.warning(f"Bit string length {len(initial_state)} doesn't match num_qubits {self.num_qubits}")
        
        # Handle Bell states for 2-qubit systems
        elif initial_state.lower() in ["bell", "|bell>", "bell_state"]:
            if self.num_qubits == 2:
                # Create Bell state (|00> + |11>)/sqrt(2)
                self.state_vector = np.zeros(4, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[3] = 1.0 / np.sqrt(2)
                return True
            else:
                logger.warning(f"Bell state requires 2 qubits, but state has {self.num_qubits}")
                return False
                
        # Handle GHZ states for 3+ qubit systems
        elif initial_state.lower() in ["ghz", "|ghz>", "ghz_state"]:
            if self.num_qubits >= 3:
                # Create GHZ state (|00...0> + |11...1>)/sqrt(2)
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                self.state_vector[0] = 1.0 / np.sqrt(2)
                self.state_vector[-1] = 1.0 / np.sqrt(2)
                return True
            else:
                logger.warning(f"GHZ state requires at least 3 qubits, but state has {self.num_qubits}")
                return False
                
        # Handle W states for 3+ qubit systems
        elif initial_state.lower() in ["w", "|w>", "w_state"]:
            if self.num_qubits >= 3:
                # Create W state (|100...0> + |010...0> + ... + |000...1>)/sqrt(n)
                self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
                norm = 1.0 / np.sqrt(self.num_qubits)
                for i in range(self.num_qubits):
                    # Set amplitude for states with exactly one '1'
                    idx = 2**i
                    self.state_vector[idx] = norm
                return True
            else:
                logger.warning(f"W state requires at least 3 qubits, but state has {self.num_qubits}")
                return False
        
        # Unknown state format
        logger.warning(f"Unrecognized initial state format: {initial_state}")
        return False
    
    def get_state_vector(self) -> np.ndarray:
        """
        Get the current state vector.
        
        Returns:
            np.ndarray: Complex amplitudes of the quantum state
        """
        return self.state_vector.copy()
    
    def set_state_vector(self, state_vector: np.ndarray) -> bool:
        """
        Set the state vector directly.
        
        Args:
            state_vector: Complex amplitudes to set as the new state
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            The state vector will be normalized automatically.
        """
        if state_vector.shape != (self.dimension,):
            logger.error(f"State vector shape {state_vector.shape} doesn't match expected dimension {self.dimension}")
            return False
            
        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm < NumericalParameters.EIGENVALUE_CUTOFF:
            logger.error("State vector has near-zero norm, cannot normalize")
            return False
            
        self.state_vector = state_vector / norm
        self._invalidate_caches()
        self._update_properties()
        self.last_update_time = np.datetime64('now')
        
        return True
    
    def get_density_matrix(self) -> np.ndarray:
        """
        Get the density matrix representation.
        
        Returns:
            np.ndarray: Density matrix ρ = |ψ⟩⟨ψ|
        """
        if self._density_matrix_cache is None:
            # For pure states: ρ = |ψ⟩⟨ψ|
            # Reshape to column vector and compute outer product
            psi = self.state_vector.reshape(-1, 1)
            self._density_matrix_cache = np.dot(psi, psi.conj().T)
        
        return self._density_matrix_cache.copy()
    
    def set_density_matrix(self, density_matrix: np.ndarray) -> bool:
        """
        Set state directly using a density matrix.
        
        Args:
            density_matrix: Square complex matrix of size dimension × dimension
            
        Returns:
            bool: True if successful, False otherwise
            
        Note:
            If the density matrix represents a mixed state, the state_vector
            will be set to the dominant eigenvector.
        """
        expected_shape = (self.dimension, self.dimension)
        if density_matrix.shape != expected_shape:
            logger.error(f"Density matrix shape {density_matrix.shape} doesn't match expected {expected_shape}")
            return False
            
        # Verify properties of a valid density matrix
        if not self._is_valid_density_matrix(density_matrix):
            logger.error("Invalid density matrix")
            return False
            
        # Store the density matrix
        self._density_matrix_cache = density_matrix.copy()
        
        # Extract the principal eigenvector for the state vector
        eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
        # Sort by eigenvalue in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Set state vector to the principal eigenvector
        self.state_vector = eigenvectors[:, 0]
        
        # Update metadata
        self.metadata["is_pure"] = np.isclose(eigenvalues[0], 1.0)
        
        # If not pure, update state type to MIXED
        if not self.metadata["is_pure"] and self.state_type != StateType.MIXED:
            self.state_type = StateType.MIXED
            self.metadata["state_type"] = self.state_type.value
        
        self._update_properties()
        self.last_update_time = np.datetime64('now')
        
        return True
    
    def _is_valid_density_matrix(self, matrix: np.ndarray, tolerance: float = NumericalParameters.EIGENVALUE_CUTOFF) -> bool:
        """
        Check if a matrix is a valid density matrix.
        
        Args:
            matrix: Matrix to validate
            tolerance: Numerical error tolerance
            
        Returns:
            bool: True if it's a valid density matrix
        """
        # Check if Hermitian
        if not np.allclose(matrix, matrix.conj().T, atol=tolerance):
            logger.warning("Density matrix is not Hermitian")
            return False
            
        # Check if trace is 1
        trace = np.trace(matrix).real
        if not np.isclose(trace, 1.0, atol=tolerance):
            logger.warning(f"Density matrix trace is {trace}, not 1.0")
            return False
            
        # Check if positive semidefinite
        eigenvalues = np.linalg.eigvalsh(matrix)
        if np.any(eigenvalues < -tolerance):
            logger.warning("Density matrix is not positive semidefinite")
            return False
            
        return True
    
    def apply_gate(self, gate: str, target_qubits: Union[List[int], int], 
                  control_qubits: Optional[Union[List[int], int]] = None,
                  params: Optional[Union[List[float], float]] = None) -> bool:
        """
        Apply a quantum gate to the state.
        
        Args:
            gate: Name of the gate to apply
            target_qubits: Indices of target qubits
            control_qubits: Optional indices of control qubits
            params: Optional parameters for parameterized gates
            
        Returns:
            bool: True if gate was successfully applied
        """
        # Validate qubit indices
        if isinstance(target_qubits, int):
            target_qubits = [target_qubits]
            
        if isinstance(control_qubits, int):
            control_qubits = [control_qubits]
            
        for q in target_qubits or []:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Target qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        for q in control_qubits or []:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Control qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        # Check for overlap between control and target qubits
        if control_qubits:
            for q in control_qubits:
                if q in target_qubits:
                    logger.error(f"Qubit {q} cannot be both control and target")
                    return False

        # Record the gate application in history
        gate_record = {
            "gate": gate,
            "target_qubits": target_qubits.copy() if target_qubits else None,
            "control_qubits": control_qubits.copy() if control_qubits else None,
            "params": params.copy() if isinstance(params, list) else params,
            "time": np.datetime64('now')
        }
        self.gate_history.append(gate_record)
        
        # Convert parameters to list if needed
        if params is not None and not isinstance(params, list):
            params = [params]
            
        # Apply gate based on type
        result = False
        
        # Single-qubit gates
        if gate.upper() in ["X", "X_GATE", "PAULIX", "PAULIX_GATE"]:
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_pauli_x())
            
        elif gate.upper() in ["Y", "Y_GATE", "PAULIY", "PAULIY_GATE"]:
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_pauli_y())
            
        elif gate.upper() in ["Z", "Z_GATE", "PAULIZ", "PAULIZ_GATE"]:
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_pauli_z())
            
        elif gate.upper() in ["H", "H_GATE", "HADAMARD", "HADAMARD_GATE"]:
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_hadamard())
            
        elif gate.upper() in ["S", "S_GATE", "PHASE"]:
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_s_gate())
            
        elif gate.upper() in ["T", "T_GATE"]:
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_t_gate())
            
        # Parameterized single-qubit gates
        elif gate.upper() in ["RX", "RX_GATE"]:
            if params is None or len(params) != 1:
                logger.error(f"RX gate requires 1 parameter (rotation angle), got {params}")
                return False
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_rx_gate(params[0]))
            
        elif gate.upper() in ["RY", "RY_GATE"]:
            if params is None or len(params) != 1:
                logger.error(f"RY gate requires 1 parameter (rotation angle), got {params}")
                return False
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_ry_gate(params[0]))
            
        elif gate.upper() in ["RZ", "RZ_GATE"]:
            if params is None or len(params) != 1:
                logger.error(f"RZ gate requires 1 parameter (rotation angle), got {params}")
                return False
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_rz_gate(params[0]))
            
        elif gate.upper() in ["P", "P_GATE", "PHASE_SHIFT"]:
            if params is None or len(params) != 1:
                logger.error(f"Phase gate requires 1 parameter (phase angle), got {params}")
                return False
            result = self._apply_single_qubit_gate(target_qubits[0], self._get_phase_gate(params[0]))
            
        # Two-qubit gates
        elif gate.upper() in ["CNOT", "CX", "CONTROLLED_X"]:
            if not control_qubits or len(control_qubits) != 1 or len(target_qubits) != 1:
                logger.error(f"CNOT gate requires 1 control and 1 target qubit")
                return False
            result = self._apply_controlled_gate(control_qubits[0], target_qubits[0], self._get_pauli_x())
            
        elif gate.upper() in ["CY", "CONTROLLED_Y"]:
            if not control_qubits or len(control_qubits) != 1 or len(target_qubits) != 1:
                logger.error(f"CY gate requires 1 control and 1 target qubit")
                return False
            result = self._apply_controlled_gate(control_qubits[0], target_qubits[0], self._get_pauli_y())
            
        elif gate.upper() in ["CZ", "CONTROLLED_Z"]:
            if not control_qubits or len(control_qubits) != 1 or len(target_qubits) != 1:
                logger.error(f"CZ gate requires 1 control and 1 target qubit")
                return False
            result = self._apply_controlled_gate(control_qubits[0], target_qubits[0], self._get_pauli_z())
            
        elif gate.upper() in ["SWAP", "SWAP_GATE"]:
            if len(target_qubits) != 2:
                logger.error(f"SWAP gate requires 2 target qubits, got {len(target_qubits)}")
                return False
            result = self._apply_swap_gate(target_qubits[0], target_qubits[1])
            
        # Three-qubit gates
        elif gate.upper() in ["TOFFOLI", "CCNOT", "CONTROLLED_CONTROLLED_NOT"]:
            if not control_qubits or len(control_qubits) != 2 or len(target_qubits) != 1:
                logger.error(f"Toffoli gate requires 2 control qubits and 1 target qubit")
                return False
            result = self._apply_toffoli_gate(control_qubits[0], control_qubits[1], target_qubits[0])
            
        elif gate.upper() in ["CSWAP", "FREDKIN", "CONTROLLED_SWAP"]:
            if not control_qubits or len(control_qubits) != 1 or len(target_qubits) != 2:
                logger.error(f"CSWAP gate requires 1 control qubit and 2 target qubits")
                return False
            result = self._apply_controlled_swap(control_qubits[0], target_qubits[0], target_qubits[1])
            
        # Multi-qubit operations
        elif gate.upper() in ["QFT", "QFT_GATE", "QUANTUM_FOURIER_TRANSFORM"]:
            result = self._apply_qft(target_qubits)
            
        elif gate.upper() in ["IQFT", "INVERSE_QFT", "INVERSE_QUANTUM_FOURIER_TRANSFORM"]:
            result = self._apply_inverse_qft(target_qubits)
            
        else:
            logger.warning(f"Unknown gate: {gate}")
            result = False
        
        if result:
            self._invalidate_caches()
            self._update_properties()
            self.last_update_time = np.datetime64('now')
        
        return result
    
    def _apply_single_qubit_gate(self, qubit: int, matrix: np.ndarray) -> bool:
        """
        Apply a single-qubit gate to the specified qubit.
        
        Args:
            qubit: Index of the qubit to apply the gate to
            matrix: 2x2 unitary matrix representing the gate
            
        Returns:
            bool: True if successful
        """
        if qubit < 0 or qubit >= self.num_qubits:
            logger.error(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        # For each basis state, apply the gate to the specified qubit
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Iterate through all basis states
        for i in range(self.dimension):
            # Check if the qubit is 0 or 1 in this basis state
            bit_val = (i >> qubit) & 1
            
            # Compute the index with qubit set to 0 and 1
            idx0 = i & ~(1 << qubit)  # Clear the qubit
            idx1 = i | (1 << qubit)   # Set the qubit
            
            if bit_val == 0:
                # |ψ⟩ = α|0⟩ + β|1⟩ => U|ψ⟩ = U00α|0⟩ + U01β|1⟩
                new_state[idx0] += matrix[0, 0] * self.state_vector[i]
                new_state[idx1] += matrix[1, 0] * self.state_vector[i]
            else:
                # |ψ⟩ = α|0⟩ + β|1⟩ => U|ψ⟩ = U10α|0⟩ + U11β|1⟩
                new_state[idx0] += matrix[0, 1] * self.state_vector[i]
                new_state[idx1] += matrix[1, 1] * self.state_vector[i]
        
        # Update the state vector
        self.state_vector = new_state
        
        return True
    
    def _apply_controlled_gate(self, control_qubit: int, target_qubit: int, matrix: np.ndarray) -> bool:
        """
        Apply a controlled single-qubit gate.
        
        Args:
            control_qubit: Index of the control qubit
            target_qubit: Index of the target qubit
            matrix: 2x2 unitary matrix representing the target gate
            
        Returns:
            bool: True if successful
        """
        if control_qubit < 0 or control_qubit >= self.num_qubits:
            logger.error(f"Control qubit index {control_qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        if target_qubit < 0 or target_qubit >= self.num_qubits:
            logger.error(f"Target qubit index {target_qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        if control_qubit == target_qubit:
            logger.error(f"Control and target qubits cannot be the same")
            return False
            
        # Create new state vector - start with zeros
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # Process each basis state
        for i in range(self.dimension):
            # Check if the control qubit is 1
            if (i >> control_qubit) & 1 == 1:
                # Control is 1, apply the gate to target
                target_bit = (i >> target_qubit) & 1
                
                # Compute indices with target qubit flipped
                if target_bit == 0:
                    # Target is 0, will become 1 after gate
                    new_idx = i | (1 << target_qubit)
                else:
                    # Target is 1, will become 0 after gate
                    new_idx = i & ~(1 << target_qubit)
                
                # For general gate, apply the matrix
                # For X gate specifically, this just flips the bit
                if target_bit == 0:
                    # |0⟩ target
                    new_state[i & ~(1 << target_qubit)] += matrix[0, 0] * self.state_vector[i]
                    new_state[i | (1 << target_qubit)] += matrix[1, 0] * self.state_vector[i]
                else:
                    # |1⟩ target
                    new_state[i & ~(1 << target_qubit)] += matrix[0, 1] * self.state_vector[i]
                    new_state[i | (1 << target_qubit)] += matrix[1, 1] * self.state_vector[i]
            else:
                # Control is 0, no change
                new_state[i] = self.state_vector[i]
        
        # Update the state vector
        self.state_vector = new_state
        
        return True
    
    def _apply_swap_gate(self, qubit1: int, qubit2: int) -> bool:
        """
        Apply a SWAP gate between two qubits.
        
        Args:
            qubit1: First qubit index
            qubit2: Second qubit index
            
        Returns:
            bool: True if successful
        """
        if qubit1 < 0 or qubit1 >= self.num_qubits:
            logger.error(f"Qubit index {qubit1} out of range [0, {self.num_qubits-1}]")
            return False
            
        if qubit2 < 0 or qubit2 >= self.num_qubits:
            logger.error(f"Qubit index {qubit2} out of range [0, {self.num_qubits-1}]")
            return False
            
        if qubit1 == qubit2:
            logger.warning(f"SWAP gate on the same qubit {qubit1} has no effect")
            return True
            
        # Create new state vector
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # For each basis state, swap the values of the two qubits
        for i in range(self.dimension):
            # Extract the values of the two qubits
            val1 = (i >> qubit1) & 1
            val2 = (i >> qubit2) & 1
            
            if val1 == val2:
                # If both qubits have the same value, no change
                new_state[i] = self.state_vector[i]
            else:
                # Compute the index with swapped qubit values
                swapped_idx = i ^ (1 << qubit1) ^ (1 << qubit2)
                new_state[swapped_idx] = self.state_vector[i]
        
        # Update the state vector
        self.state_vector = new_state
        
        return True
    
    def _apply_toffoli_gate(self, control1: int, control2: int, target: int) -> bool:
        """
        Apply a Toffoli (CCNOT) gate.
        
        Args:
            control1: First control qubit index
            control2: Second control qubit index
            target: Target qubit index
            
        Returns:
            bool: True if successful
        """
        if control1 < 0 or control1 >= self.num_qubits:
            logger.error(f"Control qubit index {control1} out of range [0, {self.num_qubits-1}]")
            return False
            
        if control2 < 0 or control2 >= self.num_qubits:
            logger.error(f"Control qubit index {control2} out of range [0, {self.num_qubits-1}]")
            return False
            
        if target < 0 or target >= self.num_qubits:
            logger.error(f"Target qubit index {target} out of range [0, {self.num_qubits-1}]")
            return False
            
        if control1 == control2 or control1 == target or control2 == target:
            logger.error(f"Control and target qubits must be distinct")
            return False
            
        # Create new state vector
        new_state = self.state_vector.copy()
        
        # Apply X gate on target only when both control qubits are |1⟩
        for i in range(self.dimension):
            # Check if both control qubits are 1
            if ((i >> control1) & 1) == 1 and ((i >> control2) & 1) == 1:
                # Flip the target qubit
                flipped_idx = i ^ (1 << target)
                new_state[i], new_state[flipped_idx] = new_state[flipped_idx], new_state[i]
        
        # Update the state vector
        self.state_vector = new_state
        
        return True
    
    def _apply_controlled_swap(self, control: int, target1: int, target2: int) -> bool:
        """
        Apply a controlled SWAP (Fredkin) gate.
        
        Args:
            control: Control qubit index
            target1: First target qubit index
            target2: Second target qubit index
            
        Returns:
            bool: True if successful
        """
        if control < 0 or control >= self.num_qubits:
            logger.error(f"Control qubit index {control} out of range [0, {self.num_qubits-1}]")
            return False
            
        if target1 < 0 or target1 >= self.num_qubits:
            logger.error(f"Target qubit index {target1} out of range [0, {self.num_qubits-1}]")
            return False
            
        if target2 < 0 or target2 >= self.num_qubits:
            logger.error(f"Target qubit index {target2} out of range [0, {self.num_qubits-1}]")
            return False
            
        if control == target1 or control == target2 or target1 == target2:
            logger.error(f"Control and target qubits must be distinct")
            return False
            
        # Create new state vector
        new_state = self.state_vector.copy()
        
        # Apply SWAP only when control qubit is |1⟩
        for i in range(self.dimension):
            # Check if control qubit is 1
            if ((i >> control) & 1) == 1:
                # Extract the values of the two target qubits
                val1 = (i >> target1) & 1
                val2 = (i >> target2) & 1
                
                if val1 != val2:
                    # Compute the index with swapped qubit values
                    swapped_idx = i ^ (1 << target1) ^ (1 << target2)
                    new_state[i], new_state[swapped_idx] = new_state[swapped_idx], new_state[i]
        
        # Update the state vector
        self.state_vector = new_state
        
        return True
    
    def _apply_qft(self, qubits: Optional[List[int]] = None) -> bool:
        """
        Apply Quantum Fourier Transform to specified qubits.
        
        Args:
            qubits: List of qubit indices, or None for all qubits
            
        Returns:
            bool: True if successful
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        n = len(qubits)
        if n == 0:
            return True  # No qubits to transform
            
        # Create indices mapping for the qubits we're applying QFT to
        # This lets us apply QFT to non-contiguous or out-of-order qubits
        indices = []
        for i in range(2**n):
            # Convert i to binary and map to the actual indices
            index = 0
            for j in range(n):
                if (i >> j) & 1:
                    index |= (1 << qubits[j])
            indices.append(index)
            
        # Apply QFT to the selected subspace
        omega = np.exp(2j * np.pi / (2**n))
        qft_matrix = np.zeros((2**n, 2**n), dtype=np.complex128)
        
        for i in range(2**n):
            for j in range(2**n):
                qft_matrix[i, j] = omega**(i * j) / np.sqrt(2**n)
                
        # Apply the QFT matrix to the selected subspace
        subspace = np.array([self.state_vector[i] for i in indices])
        transformed = np.dot(qft_matrix, subspace)
        
        # Update the state vector
        new_state = self.state_vector.copy()
        for i, idx in enumerate(indices):
            new_state[idx] = transformed[i]
            
        self.state_vector = new_state
        
        return True
    
    def _apply_inverse_qft(self, qubits: Optional[List[int]] = None) -> bool:
        """
        Apply inverse Quantum Fourier Transform to specified qubits.
        
        Args:
            qubits: List of qubit indices, or None for all qubits
            
        Returns:
            bool: True if successful
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        n = len(qubits)
        if n == 0:
            return True  # No qubits to transform
            
        # Create indices mapping for the qubits we're applying inverse QFT to
        indices = []
        for i in range(2**n):
            # Convert i to binary and map to the actual indices
            index = 0
            for j in range(n):
                if (i >> j) & 1:
                    index |= (1 << qubits[j])
            indices.append(index)
            
        # Apply inverse QFT to the selected subspace
        omega = np.exp(-2j * np.pi / (2**n))  # Note the negative sign for inverse
        iqft_matrix = np.zeros((2**n, 2**n), dtype=np.complex128)
        
        for i in range(2**n):
            for j in range(2**n):
                iqft_matrix[i, j] = omega**(i * j) / np.sqrt(2**n)
                
        # Apply the inverse QFT matrix to the selected subspace
        subspace = np.array([self.state_vector[i] for i in indices])
        transformed = np.dot(iqft_matrix, subspace)
        
        # Update the state vector
        new_state = self.state_vector.copy()
        for i, idx in enumerate(indices):
            new_state[idx] = transformed[i]
            
        self.state_vector = new_state
        
        return True
    
    def measure(self, qubits: Optional[Union[List[int], int]] = None, 
                basis: Optional[str] = None) -> Dict[str, Any]:
        """
        Measure the quantum state and collapse it according to the measurement outcome.
        
        Args:
            qubits: Specific qubits to measure, or None for all qubits
            basis: Measurement basis, default is "Z_basis" (computational)
            
        Returns:
            dict: Measurement result with outcome, probabilities, etc.
        """
        # Default to measuring all qubits in computational basis
        if qubits is None:
            qubits = list(range(self.num_qubits))
        elif isinstance(qubits, int):
            qubits = [qubits]
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return {"error": f"Invalid qubit index {q}"}
                
        # Determine basis and apply basis transformation if needed
        basis = basis or "Z_basis"
        if basis != "Z_basis":
            # Apply basis transformation
            self._apply_basis_transformation(qubits, basis)
        
        # Compute outcome probabilities based on state vector
        probabilities = np.abs(self.state_vector)**2
        
        # Select an outcome based on probabilities
        outcome_idx = np.random.choice(self.dimension, p=probabilities)
        
        # Convert to binary string representation for the full state
        full_outcome = format(outcome_idx, f'0{self.num_qubits}b')
        
        # Extract just the measured qubits
        measured_outcome = ''.join(full_outcome[self.num_qubits - 1 - q] for q in sorted(qubits))
        
        # Collapse the state based on the measurement
        self._collapse_state_vector(qubits, full_outcome)
        
        # Create probability distribution for measured qubits
        measured_probs = {}
        for i in range(2**len(qubits)):
            # Convert to binary string
            bitstring = format(i, f'0{len(qubits)}b')
            # Calculate probability by summing over all matching states
            prob = 0.0
            for j in range(self.dimension):
                full_state = format(j, f'0{self.num_qubits}b')
                # Extract the relevant qubits
                measured_bits = ''.join(full_state[self.num_qubits - 1 - q] for q in sorted(qubits))
                if measured_bits == bitstring:
                    prob += probabilities[j]
            if prob > NumericalParameters.EIGENVALUE_CUTOFF:  # Only include non-zero probabilities
                measured_probs[bitstring] = float(prob)
        
        # Create result dictionary
        result = {
            "outcome": measured_outcome,
            "probabilities": measured_probs,
            "value": int(measured_outcome, 2),
            "qubits": qubits.copy(),
            "basis": basis,
            "time": np.datetime64('now')
        }
        
        # Record in measurement history
        self.measurement_history.append(result)
        self.measurement_count += 1
        
        # If basis was changed, transform back to computational basis
        if basis != "Z_basis":
            self._apply_inverse_basis_transformation(qubits, basis)
        
        # Update properties
        self._invalidate_caches()
        self._update_properties()
        self.last_update_time = np.datetime64('now')
        
        return result
    
    def _apply_basis_transformation(self, qubits: List[int], basis: str) -> bool:
        """
        Apply transformation to change the measurement basis.
        
        Args:
            qubits: List of qubit indices to transform
            basis: Target basis ("X_basis", "Y_basis", "Bell_basis", etc.)
            
        Returns:
            bool: True if successful
        """
        for q in qubits:
            if basis == "X_basis":
                # Apply Hadamard to measure in X basis
                self._apply_single_qubit_gate(q, self._get_hadamard())
            elif basis == "Y_basis":
                # Apply S† then H to measure in Y basis
                # S† = [ 1  0 ]
                #      [ 0 -i ]
                s_dagger = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
                self._apply_single_qubit_gate(q, s_dagger)
                self._apply_single_qubit_gate(q, self._get_hadamard())
            elif basis == "Bell_basis" and len(qubits) == 2:
                # Apply inverse Bell transform: CNOT then H
                self._apply_controlled_gate(qubits[0], qubits[1], self._get_pauli_x())
                self._apply_single_qubit_gate(qubits[0], self._get_hadamard())
                return True
            elif basis != "Z_basis":
                logger.warning(f"Unsupported basis: {basis}, using Z_basis")
                return False
                
        return True
    
    def _apply_inverse_basis_transformation(self, qubits: List[int], basis: str) -> bool:
        """
        Apply inverse transformation to return to computational basis.
        
        Args:
            qubits: List of qubit indices to transform
            basis: Source basis ("X_basis", "Y_basis", "Bell_basis", etc.)
            
        Returns:
            bool: True if successful
        """
        for q in qubits:
            if basis == "X_basis":
                # Apply Hadamard to return to Z basis
                self._apply_single_qubit_gate(q, self._get_hadamard())
            elif basis == "Y_basis":
                # Apply H then S to return to Z basis
                self._apply_single_qubit_gate(q, self._get_hadamard())
                self._apply_single_qubit_gate(q, self._get_s_gate())
            elif basis == "Bell_basis" and len(qubits) == 2:
                # Apply Bell transform: H then CNOT
                self._apply_single_qubit_gate(qubits[0], self._get_hadamard())
                self._apply_controlled_gate(qubits[0], qubits[1], self._get_pauli_x())
                return True
            elif basis != "Z_basis":
                logger.warning(f"Unsupported basis: {basis}, using Z_basis")
                return False
                
        return True
    
    def _collapse_state_vector(self, qubits: List[int], outcome: str) -> bool:
        """
        Collapse the state vector based on a measurement outcome.
        
        Args:
            qubits: List of qubit indices measured
            outcome: Full binary string outcome for all qubits
            
        Returns:
            bool: True if successful
        """
        if len(outcome) != self.num_qubits:
            logger.error(f"Outcome length {len(outcome)} doesn't match number of qubits {self.num_qubits}")
            return False
            
        # Create a mask for the measured qubits
        mask = 0
        for q in qubits:
            mask |= (1 << q)
            
        # Get the value of the measured qubits based on the outcome
        measured_value = 0
        for q in qubits:
            bit_pos = self.num_qubits - 1 - q
            if outcome[bit_pos] == '1':
                measured_value |= (1 << q)
                
        # Create new state vector with only the states that match the measurement
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        norm_squared = 0.0
        
        for i in range(self.dimension):
            # Check if the measured qubits match the outcome
            if (i & mask) == measured_value:
                new_state[i] = self.state_vector[i]
                norm_squared += np.abs(self.state_vector[i])**2
                
        # Normalize the new state
        if norm_squared < NumericalParameters.EIGENVALUE_CUTOFF:
            logger.error(f"State collapse failed: zero probability for outcome {outcome}")
            return False
            
        self.state_vector = new_state / np.sqrt(norm_squared)
        
        return True
    
    def entangle_with(self, other_state: 'QuantumState', 
                       qubits1: Optional[List[int]] = None,
                       qubits2: Optional[List[int]] = None,
                       method: str = "direct") -> bool:
        """
        Entangle this state with another quantum state.
        
        Args:
            other_state: The quantum state to entangle with
            qubits1: Qubits from this state to entangle (default: [0])
            qubits2: Qubits from other state to entangle (default: [0])
            method: Entanglement method/protocol
            
        Returns:
            bool: True if entanglement was successful
        """
        if other_state.name == self.name:
            logger.error("Cannot entangle a state with itself")
            return False
            
        # Default to first qubit if not specified
        if qubits1 is None:
            qubits1 = [0]
        if qubits2 is None:
            qubits2 = [0]
            
        # Validate qubit indices
        for q in qubits1:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        for q in qubits2:
            if q < 0 or q >= other_state.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {other_state.num_qubits-1}]")
                return False
                
        # Check if number of qubits match
        if len(qubits1) != len(qubits2):
            logger.error(f"Number of qubits must match for entanglement")
            return False
            
        # Handle different entanglement methods
        if method.lower() == "direct":
            # Direct entanglement just marks the states as entangled
            # This is a simplified approach as actual entanglement would require
            # creating a joint state space
            
            # Mark both states as entangled
            self.is_entangled = True
            other_state.is_entangled = True
            
            # Add to each other's entangled_with sets
            self.entangled_with.add(other_state.name)
            other_state.entangled_with.add(self.name)
            
            # Apply Hadamard and CNOT for simple entanglement preparation
            # (simplified since we're not actually joining the state spaces)
            for i in range(len(qubits1)):
                # Hadamard on first qubit to create superposition
                if i == 0:
                    self._apply_single_qubit_gate(qubits1[i], self._get_hadamard())
            
            # Record in metadata
            if "entangled_with" not in self.metadata:
                self.metadata["entangled_with"] = []
            if other_state.name not in self.metadata["entangled_with"]:
                self.metadata["entangled_with"].append(other_state.name)
                
            if "entangled_with" not in other_state.metadata:
                other_state.metadata["entangled_with"] = []
            if self.name not in other_state.metadata["entangled_with"]:
                other_state.metadata["entangled_with"].append(self.name)
                
            # Record entangled qubits
            if "entangled_qubits" not in self.metadata:
                self.metadata["entangled_qubits"] = {}
            self.metadata["entangled_qubits"][other_state.name] = qubits1
            
            if "entangled_qubits" not in other_state.metadata:
                other_state.metadata["entangled_qubits"] = {}
            other_state.metadata["entangled_qubits"][self.name] = qubits2
            
        elif method.lower() in ["bell", "epr"]:
            # Prepare Bell state entanglement
            if len(qubits1) == 1 and len(qubits2) == 1:
                # Simplified Bell state preparation
                self._apply_single_qubit_gate(qubits1[0], self._get_hadamard())
                
                # Mark both states as entangled
                self.is_entangled = True
                other_state.is_entangled = True
                
                # Add to each other's entangled_with sets and metadata
                self.entangled_with.add(other_state.name)
                other_state.entangled_with.add(self.name)
                
                if "entanglement_type" not in self.metadata:
                    self.metadata["entanglement_type"] = {}
                self.metadata["entanglement_type"][other_state.name] = "Bell"
                
                if "entanglement_type" not in other_state.metadata:
                    other_state.metadata["entanglement_type"] = {}
                other_state.metadata["entanglement_type"][self.name] = "Bell"
                
                # Record entangled qubits
                if "entangled_qubits" not in self.metadata:
                    self.metadata["entangled_qubits"] = {}
                self.metadata["entangled_qubits"][other_state.name] = qubits1
                
                if "entangled_qubits" not in other_state.metadata:
                    other_state.metadata["entangled_qubits"] = {}
                other_state.metadata["entangled_qubits"][self.name] = qubits2
            else:
                logger.error(f"Bell entanglement requires exactly 1 qubit from each state")
                return False
                
        elif method.lower() in ["ghz"]:
            # Prepare GHZ state entanglement
            if len(qubits1) >= 1 and len(qubits2) >= 1:
                # Simplified GHZ state preparation
                self._apply_single_qubit_gate(qubits1[0], self._get_hadamard())
                
                # Mark both states as entangled
                self.is_entangled = True
                other_state.is_entangled = True
                
                # Add to each other's entangled_with sets and metadata
                self.entangled_with.add(other_state.name)
                other_state.entangled_with.add(self.name)
                
                if "entanglement_type" not in self.metadata:
                    self.metadata["entanglement_type"] = {}
                self.metadata["entanglement_type"][other_state.name] = "GHZ"
                
                if "entanglement_type" not in other_state.metadata:
                    other_state.metadata["entanglement_type"] = {}
                other_state.metadata["entanglement_type"][self.name] = "GHZ"
                
                # Record entangled qubits
                if "entangled_qubits" not in self.metadata:
                    self.metadata["entangled_qubits"] = {}
                self.metadata["entangled_qubits"][other_state.name] = qubits1
                
                if "entangled_qubits" not in other_state.metadata:
                    other_state.metadata["entangled_qubits"] = {}
                other_state.metadata["entangled_qubits"][self.name] = qubits2
            else:
                logger.error(f"GHZ entanglement requires at least 1 qubit from each state")
                return False
                
        elif method.lower() in ["w"]:
            # Prepare W state entanglement
            if len(qubits1) >= 1 and len(qubits2) >= 1:
                # Simplified W state preparation
                self._apply_single_qubit_gate(qubits1[0], self._get_hadamard())
                
                # Mark both states as entangled
                self.is_entangled = True
                other_state.is_entangled = True
                
                # Add to each other's entangled_with sets and metadata
                self.entangled_with.add(other_state.name)
                other_state.entangled_with.add(self.name)
                
                if "entanglement_type" not in self.metadata:
                    self.metadata["entanglement_type"] = {}
                self.metadata["entanglement_type"][other_state.name] = "W"
                
                if "entanglement_type" not in other_state.metadata:
                    other_state.metadata["entanglement_type"] = {}
                other_state.metadata["entanglement_type"][self.name] = "W"
                
                # Record entangled qubits
                if "entangled_qubits" not in self.metadata:
                    self.metadata["entangled_qubits"] = {}
                self.metadata["entangled_qubits"][other_state.name] = qubits1
                
                if "entangled_qubits" not in other_state.metadata:
                    other_state.metadata["entangled_qubits"] = {}
                other_state.metadata["entangled_qubits"][self.name] = qubits2
            else:
                logger.error(f"W entanglement requires at least 1 qubit from each state")
                return False
        else:
            logger.error(f"Unknown entanglement method: {method}")
            return False
        
        # Update state type
        if self.state_type != StateType.ENTANGLED:
            self.state_type = StateType.ENTANGLED
            self.metadata["state_type"] = self.state_type.value
            
        if other_state.state_type != StateType.ENTANGLED:
            other_state.state_type = StateType.ENTANGLED
            other_state.metadata["state_type"] = other_state.state_type.value
            
        # Update properties
        self._update_properties()
        other_state._update_properties()
        
        return True
    
    def teleport_to(self, destination_state: 'QuantumState',
                    source_qubit: int = 0,
                    destination_qubit: int = 0,
                    protocol: str = "standard") -> bool:
        """
        Teleport a qubit state to another quantum state.
        
        Args:
            destination_state: Target state to teleport to
            source_qubit: Qubit index in this state to teleport
            destination_qubit: Qubit index in destination to teleport to
            protocol: Teleportation protocol to use
            
        Returns:
            bool: True if teleportation was successful
        """
        # Validate qubit indices
        if source_qubit < 0 or source_qubit >= self.num_qubits:
            logger.error(f"Source qubit index {source_qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        if destination_qubit < 0 or destination_qubit >= destination_state.num_qubits:
            logger.error(f"Destination qubit index {destination_qubit} out of range [0, {destination_state.num_qubits-1}]")
            return False
            
        # Check if the states are entangled (required for teleportation)
        if destination_state.name not in self.entangled_with:
            logger.error(f"States must be entangled for teleportation")
            return False
        
        # Extract the state of the source qubit
        source_state = self._extract_qubit_state(source_qubit)
        
        # Different protocols have different requirements and steps
        if protocol.lower() in ["standard", "standard_protocol"]:
            # Standard teleportation: measure source qubit in Bell basis
            result = self.measure([source_qubit], basis="Bell_basis")
            
            # Apply correction based on measurement outcome
            outcome = result["outcome"]
            if outcome == "0":  # |Φ+⟩ outcome
                # No correction needed
                pass
            elif outcome == "1":  # |Φ-⟩ outcome
                # Apply Z correction
                destination_state._apply_single_qubit_gate(destination_qubit, self._get_pauli_z())
            elif outcome == "2":  # |Ψ+⟩ outcome
                # Apply X correction
                destination_state._apply_single_qubit_gate(destination_qubit, self._get_pauli_x())
            elif outcome == "3":  # |Ψ-⟩ outcome
                # Apply X and Z corrections
                destination_state._apply_single_qubit_gate(destination_qubit, self._get_pauli_x())
                destination_state._apply_single_qubit_gate(destination_qubit, self._get_pauli_z())
                
            # Set the destination qubit to the teleported state
            destination_state._set_qubit_state(destination_qubit, source_state[0], source_state[1])
            
        else:
            logger.error(f"Unsupported teleportation protocol: {protocol}")
            return False
            
        # Record the teleportation
        teleport_record = {
            "source_qubit": source_qubit,
            "destination_qubit": destination_qubit,
            "destination_state": destination_state.name,
            "protocol": protocol,
            "time": np.datetime64('now')
        }
        
        if "teleportation_history" not in self.metadata:
            self.metadata["teleportation_history"] = []
        self.metadata["teleportation_history"].append(teleport_record)
        
        if "teleportation_history" not in destination_state.metadata:
            destination_state.metadata["teleportation_history"] = []
        destination_state.metadata["teleportation_history"].append({
            "source_state": self.name,
            "source_qubit": source_qubit,
            "destination_qubit": destination_qubit,
            "protocol": protocol,
            "time": np.datetime64('now')
        })
        
        # Update properties
        self._invalidate_caches()
        destination_state._invalidate_caches()
        self._update_properties()
        destination_state._update_properties()
        
        logger.info(f"Teleported qubit {source_qubit} from {self.name} to qubit {destination_qubit} in {destination_state.name} using {protocol}")
        
        return True
    
    def reset(self, qubits: Optional[List[int]] = None) -> bool:
        """
        Reset specified qubits or the entire state to |0...0>.
        
        Args:
            qubits: List of qubit indices to reset, or None for all qubits
            
        Returns:
            bool: True if reset was successful
        """
        if qubits is None:
            # Reset entire state
            self.state_vector = np.zeros(self.dimension, dtype=np.complex128)
            self.state_vector[0] = 1.0
            self._invalidate_caches()
            self._update_properties()
            return True
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return False
                
        # For partial reset, we'll use an efficient approach based on measurement and collapse
        # to avoid computing the full projector matrix
        
        # For each qubit, measure and collapse to |0⟩
        for q in qubits:
            # Apply measurement-like collapse
            collapsed = self._reset_qubit(q)
            if not collapsed:
                logger.warning(f"Failed to reset qubit {q}")
                
        self._invalidate_caches()
        self._update_properties()
        
        return True
    
    def _reset_qubit(self, qubit: int) -> bool:
        """
        Reset a single qubit to |0⟩ while preserving superposition of other qubits.
        
        Args:
            qubit: Qubit index to reset
            
        Returns:
            bool: True if successful
        """
        if qubit < 0 or qubit >= self.num_qubits:
            logger.error(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        # Create new state vector
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        norm_squared = 0.0
        
        # For each basis state
        for i in range(self.dimension):
            # Check if the qubit is already in state |0⟩
            if ((i >> qubit) & 1) == 0:
                new_state[i] = self.state_vector[i]
                norm_squared += np.abs(self.state_vector[i])**2
            else:
                # Otherwise, map to the corresponding state with qubit = 0
                j = i & ~(1 << qubit)  # Clear the qubit
                new_state[j] += self.state_vector[i]
                norm_squared += np.abs(self.state_vector[i])**2
                
        # Normalize the new state
        if norm_squared < NumericalParameters.EIGENVALUE_CUTOFF:
            # If norm is too small, the state is effectively |1⟩
            # Create a deterministic reset by setting qubit to |0⟩
            for i in range(self.dimension):
                if ((i >> qubit) & 1) == 1:
                    j = i & ~(1 << qubit)  # Clear the qubit
                    new_state[j] = self.state_vector[i]
                    
            # Renormalize
            new_norm = np.linalg.norm(new_state)
            if new_norm < NumericalParameters.EIGENVALUE_CUTOFF:
                logger.error(f"Failed to reset qubit {qubit}: could not generate valid state")
                return False
                
            self.state_vector = new_state / new_norm
        else:
            self.state_vector = new_state / np.sqrt(norm_squared)
            
        return True
    
    def _extract_qubit_state(self, qubit: int) -> Tuple[complex, complex]:
        """
        Extract the state of a single qubit as (alpha, beta) where |ψ⟩ = α|0⟩ + β|1⟩.
        
        Args:
            qubit: Qubit index to extract
            
        Returns:
            Tuple[complex, complex]: (alpha, beta) coefficients
            
        Note:
            This is only exact if the qubit is not entangled with other qubits.
            For entangled qubits, it returns the reduced density matrix coefficients.
        """
        if qubit < 0 or qubit >= self.num_qubits:
            logger.error(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
            return (1.0, 0.0)  # Default to |0⟩
            
        if self.num_qubits == 1:
            # For a single-qubit system, return the state vector directly
            return (self.state_vector[0], self.state_vector[1])
            
        # For multi-qubit systems, compute the reduced density matrix
        rho = self._get_reduced_density_matrix(qubit)
        
        # Extract the coefficients
        alpha = np.sqrt(rho[0, 0])
        beta = rho[0, 1] / alpha if abs(alpha) > NumericalParameters.EIGENVALUE_CUTOFF else 0.0
        
        # If the qubit is entangled, the resulting state won't be pure
        # We'll return an approximation based on the density matrix
        return (alpha, beta)
    
    def _set_qubit_state(self, qubit: int, alpha: complex, beta: complex) -> bool:
        """
        Set the state of a single qubit to |ψ⟩ = α|0⟩ + β|1⟩.
        
        Args:
            qubit: Qubit index to set
            alpha: Coefficient for |0⟩
            beta: Coefficient for |1⟩
            
        Returns:
            bool: True if successful
            
        Note:
            This only works correctly if the qubit is not entangled with other qubits.
            For entangled qubits, it will attempt to modify the qubit while preserving
            entanglement relationships.
        """
        if qubit < 0 or qubit >= self.num_qubits:
            logger.error(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
            return False
            
        # Normalize the coefficients
        norm = np.sqrt(np.abs(alpha)**2 + np.abs(beta)**2)
        if norm < NumericalParameters.EIGENVALUE_CUTOFF:
            logger.error(f"Invalid qubit state: zero norm")
            return False
            
        alpha /= norm
        beta /= norm
        
        if self.num_qubits == 1:
            # For a single-qubit system, set the state vector directly
            self.state_vector[0] = alpha
            self.state_vector[1] = beta
            self._invalidate_caches()
            self._update_properties()
            return True
            
        # For multi-qubit systems, update the state vector while preserving
        # entanglement relationships as much as possible
        
        # Create new state vector
        new_state = np.zeros(self.dimension, dtype=np.complex128)
        
        # For each basis state
        for i in range(self.dimension):
            if ((i >> qubit) & 1) == 0:
                # If the qubit is |0⟩, multiply by alpha
                new_state[i] = self.state_vector[i] * alpha / self.state_vector[i & ~(1 << qubit)] if abs(self.state_vector[i & ~(1 << qubit)]) > NumericalParameters.EIGENVALUE_CUTOFF else 0.0
            else:
                # If the qubit is |1⟩, multiply by beta
                new_state[i] = self.state_vector[i] * beta / self.state_vector[i & ~(1 << qubit)] if abs(self.state_vector[i & ~(1 << qubit)]) > NumericalParameters.EIGENVALUE_CUTOFF else 0.0
                
        # Normalize the new state
        norm = np.linalg.norm(new_state)
        if norm < NumericalParameters.EIGENVALUE_CUTOFF:
            logger.error(f"Failed to set qubit {qubit} state: generated state has zero norm")
            return False
            
        self.state_vector = new_state / norm
        self._invalidate_caches()
        self._update_properties()
        
        return True
    
    def get_bloch_vector(self, qubit: int = 0) -> Tuple[float, float, float]:
        """
        Calculate the Bloch sphere coordinates for a single qubit.
        
        Args:
            qubit: Index of the qubit to analyze
            
        Returns:
            Tuple[float, float, float]: (x, y, z) coordinates on the Bloch sphere
            
        Note:
            Only applies to single qubits or specific qubits in a multi-qubit system.
        """
        if qubit < 0 or qubit >= self.num_qubits:
            logger.error(f"Qubit index {qubit} out of range [0, {self.num_qubits-1}]")
            return (0.0, 0.0, 0.0)
            
        # Check cache first
        if qubit in self._bloch_vector_cache:
            return self._bloch_vector_cache[qubit]
            
        if self.num_qubits == 1:
            # For single qubit, use the state vector directly
            rho = self.get_density_matrix()
        else:
            # For multi-qubit, get reduced density matrix for the target qubit
            rho = self._get_reduced_density_matrix(qubit)
            
        # Bloch vector components from density matrix
        # x = Tr(ρX) = ρ01 + ρ10 = 2*Re(ρ01)
        # y = Tr(ρY) = i(ρ01 - ρ10) = 2*Im(ρ01)
        # z = Tr(ρZ) = ρ00 - ρ11
        x = 2.0 * rho[0, 1].real
        y = 2.0 * rho[0, 1].imag
        z = rho[0, 0] - rho[1, 1]
        
        # Ensure we're within the Bloch sphere due to numerical precision
        length = np.sqrt(x**2 + y**2 + z**2)
        if length > 1.0:
            x /= length
            y /= length
            z /= length
            
        # Cache the result
        self._bloch_vector_cache[qubit] = (x, y, z)
        
        return (x, y, z)
    
    def _get_reduced_density_matrix(self, qubit: int) -> np.ndarray:
        """
        Calculate the reduced density matrix for a specific qubit.
        
        Args:
            qubit: Index of the qubit to get reduced density matrix for
            
        Returns:
            np.ndarray: 2x2 reduced density matrix
        """
        # Check cache first
        if qubit in self._reduced_density_cache:
            return self._reduced_density_cache[qubit].copy()
            
        # Get the full density matrix
        full_rho = self.get_density_matrix()
        
        # Initialize the reduced density matrix
        rho_reduced = np.zeros((2, 2), dtype=np.complex128)
        
        # We'll use the partial trace formula to calculate the reduced density matrix
        # For each combination of 0 and 1 for the target qubit
        for i in range(2):
            for j in range(2):
                # Sum over all other qubits
                rho_reduced[i, j] = 0.0
                for k in range(2**(self.num_qubits - 1)):
                    # Create indices with target qubit set to i or j, and other qubits set to k
                    # First convert k to binary and insert i or j at the qubit position
                    idx_i = 0
                    idx_j = 0
                    bit_pos = 0
                    
                    for q in range(self.num_qubits):
                        if q == qubit:
                            # Set target qubit to i or j
                            if i == 1:
                                idx_i |= (1 << q)
                            if j == 1:
                                idx_j |= (1 << q)
                        else:
                            # Set other qubits according to k
                            if (k >> bit_pos) & 1:
                                idx_i |= (1 << q)
                                idx_j |= (1 << q)
                            bit_pos += 1
                    
                    rho_reduced[i, j] += full_rho[idx_i, idx_j]
        
        # Cache the result
        self._reduced_density_cache[qubit] = rho_reduced.copy()
        
        return rho_reduced
    
    def _update_properties(self) -> None:
        """
        Update internal properties like coherence, entropy, etc.
        """
        # Calculate coherence
        self.coherence = self._calculate_coherence()
        self.metadata["coherence"] = float(self.coherence)
        
        # Calculate von Neumann entropy
        self.entropy = self._calculate_entropy()
        self.metadata["entropy"] = float(self.entropy)
        
        # Update phase and metadata
        self.phase = np.angle(self.state_vector[0]) if abs(self.state_vector[0]) > NumericalParameters.EIGENVALUE_CUTOFF else 0.0
        self.metadata["phase"] = float(self.phase)
        
        # Update state type
        if self.is_entangled and self.state_type != StateType.ENTANGLED:
            self.state_type = StateType.ENTANGLED
            self.metadata["state_type"] = self.state_type.value
        
        # Update timestamps
        self.last_update_time = np.datetime64('now')
        self.metadata["last_updated"] = str(self.last_update_time)
    
    def _calculate_coherence(self) -> float:
        """
        Calculate the coherence of the quantum state.
        
        Returns:
            float: Coherence value in [0.0, 1.0]
            
        Note:
            For pure states, this is the off-diagonal energy of the density matrix.
            For mixed states, we use the L1-norm of coherence.
        """
        # Check if coherence has been fixed by the VM
        if hasattr(self, '_coherence_fixed') and self._coherence_fixed:
            return getattr(self, '_fixed_coherence_value', 0.95)
            
        # Get density matrix
        rho = self.get_density_matrix()
        
        # L1 norm coherence: sum of absolute values of off-diagonal elements
        coherence = 0.0
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    coherence += np.abs(rho[i, j])
                    
        # Normalize to [0, 1] range
        # Maximum L1 coherence for an n-dimensional system is n(n-1)/2
        max_coherence = self.dimension * (self.dimension - 1) / 2
        if max_coherence > 0:
            coherence /= max_coherence
            
        return float(coherence)
    
    def _calculate_entropy(self) -> float:
        """
        Calculate the von Neumann entropy of the quantum state.
        
        Returns:
            float: Entropy value in [0.0, 1.0]
            
        Note:
            S(ρ) = -Tr(ρ log₂ ρ) = -∑ᵢ λᵢ log₂ λᵢ
            where λᵢ are the eigenvalues of ρ
        """
        # Check if entropy has been fixed by the VM
        if hasattr(self, '_entropy_fixed') and self._entropy_fixed:
            return getattr(self, '_fixed_entropy_value', 0.05)
            
        # Get density matrix
        rho = self.get_density_matrix()
        
        # Calculate eigenvalues
        eigenvalues = np.linalg.eigvalsh(rho)
        
        # Calculate entropy: -∑ᵢ λᵢ log₂ λᵢ
        entropy = 0.0
        for eig in eigenvalues:
            if eig > NumericalParameters.EIGENVALUE_CUTOFF:  # Avoid log(0)
                entropy -= eig * np.log2(eig)
                
        # Normalize to [0, 1] range
        # Maximum entropy for an n-dimensional system is log₂(n)
        max_entropy = np.log2(self.dimension)
        if max_entropy > 0:
            entropy /= max_entropy
            
        return float(entropy)
    
    def _invalidate_caches(self) -> None:
        """
        Invalidate all cached values.
        """
        self._density_matrix_cache = None
        self._bloch_vector_cache = {}
        self._reduced_density_cache = {}
    
    def get_register_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the quantum state.
        
        Returns:
            Dict[str, Any]: State information including properties, metadata, and history
        """
        info = {
            "name": self.name,
            "id": self.id,
            "num_qubits": self.num_qubits,
            "dimension": self.dimension,
            "state_type": self.state_type.value,
            "is_entangled": self.is_entangled,
            "entangled_with": list(self.entangled_with),
            "coherence": float(self.coherence),
            "entropy": float(self.entropy),
            "phase": float(self.phase),
            "creation_time": str(self.creation_time),
            "last_update_time": str(self.last_update_time),
            "metadata": self.metadata.copy(),
            "measurement_count": len(self.measurement_history),
            "gate_count": len(self.gate_history)
        }
        
        return info
    
    def export_to_dict(self, include_state_vector: bool = True) -> Dict[str, Any]:
        """
        Export the state to a serializable dictionary.
        
        Args:
            include_state_vector: Whether to include the full state vector
            
        Returns:
            Dict[str, Any]: Serializable state representation
        """
        export_data = {
            "name": self.name,
            "id": self.id,
            "num_qubits": self.num_qubits,
            "state_type": self.state_type.value,
            "is_entangled": self.is_entangled,
            "entangled_with": list(self.entangled_with),
            "coherence": float(self.coherence),
            "entropy": float(self.entropy),
            "phase": float(self.phase),
            "creation_time": str(self.creation_time),
            "last_update_time": str(self.last_update_time),
            "metadata": self.metadata.copy(),
            "measurement_history": self.measurement_history.copy(),
            "gate_history": self.gate_history.copy()
        }
        
        if include_state_vector:
            # Convert complex state vector to serializable form
            sv_real = self.state_vector.real.tolist()
            sv_imag = self.state_vector.imag.tolist()
            export_data["state_vector"] = {
                "real": sv_real,
                "imag": sv_imag
            }
            
        return export_data
    
    def import_from_dict(self, data: Dict[str, Any]) -> bool:
        """
        Import state from a dictionary.
        
        Args:
            data: Dictionary containing state data
            
        Returns:
            bool: True if successful
        """
        try:
            # Basic validation
            if "name" not in data or "num_qubits" not in data:
                logger.error(f"Invalid state data: missing required fields")
                return False
                
            # Update fields
            if "id" in data:
                self.id = data["id"]
                
            if "state_type" in data:
                try:
                    self.state_type = StateType(data["state_type"])
                except ValueError:
                    logger.warning(f"Unknown state type: {data['state_type']}, keeping current")
                    
            if "is_entangled" in data:
                self.is_entangled = bool(data["is_entangled"])
                
            if "entangled_with" in data:
                self.entangled_with = set(data["entangled_with"])
                
            if "coherence" in data:
                self.coherence = float(data["coherence"])
                
            if "entropy" in data:
                self.entropy = float(data["entropy"])
                
            if "phase" in data:
                self.phase = float(data["phase"])
                
            if "metadata" in data:
                self.metadata = data["metadata"].copy()
                
            if "measurement_history" in data:
                self.measurement_history = data["measurement_history"].copy()
                
            if "gate_history" in data:
                self.gate_history = data["gate_history"].copy()
                
            # Import state vector if available
            if "state_vector" in data:
                sv_data = data["state_vector"]
                if "real" in sv_data and "imag" in sv_data:
                    real = np.array(sv_data["real"])
                    imag = np.array(sv_data["imag"])
                    state_vector = real + 1j * imag
                    
                    if len(state_vector) != self.dimension:
                        logger.error(f"State vector dimension mismatch: expected {self.dimension}, got {len(state_vector)}")
                        return False
                        
                    self.state_vector = state_vector
                    self._invalidate_caches()
                    
            # Update properties
            self._update_properties()
            
            return True
        
        except Exception as e:
            logger.error(f"Error importing state data: {str(e)}")
            return False
    
    def calculate_fidelity(self, other_state: 'QuantumState') -> float:
        """
        Calculate the fidelity between this state and another quantum state.
        
        Args:
            other_state: The state to compare with
            
        Returns:
            float: Fidelity between the states [0.0, 1.0]
            
        Note:
            F(ρ, σ) = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))² for mixed states
            F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|² for pure states
        """
        if self.num_qubits != other_state.num_qubits:
            logger.error(f"Cannot calculate fidelity: qubit count mismatch ({self.num_qubits} vs {other_state.num_qubits})")
            return 0.0
            
        # For pure states, use the simpler formula
        if self.metadata.get("is_pure", True) and other_state.metadata.get("is_pure", True):
            # F(|ψ⟩, |φ⟩) = |⟨ψ|φ⟩|²
            inner_product = np.vdot(self.state_vector, other_state.state_vector)
            fidelity = np.abs(inner_product)**2
        else:
            # For mixed states, use the full formula
            # F(ρ, σ) = Tr(sqrt(sqrt(ρ) σ sqrt(ρ)))²
            rho = self.get_density_matrix()
            sigma = other_state.get_density_matrix()
            
            # Calculate sqrt(ρ)
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
            # Zero out negative eigenvalues due to numerical precision
            eigenvalues = np.maximum(eigenvalues, 0.0)
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            sqrt_rho = np.dot(eigenvectors, np.dot(np.diag(sqrt_eigenvalues), eigenvectors.conj().T))
            
            # Calculate sqrt(sqrt(ρ) σ sqrt(ρ))
            inner = np.dot(sqrt_rho, np.dot(sigma, sqrt_rho))
            eigenvalues_inner = np.linalg.eigvalsh(inner)
            # Zero out negative eigenvalues due to numerical precision
            eigenvalues_inner = np.maximum(eigenvalues_inner, 0.0)
            trace = np.sum(np.sqrt(eigenvalues_inner))
            
            fidelity = trace**2
            
        # Record in history
        record = {
            "time": np.datetime64('now'),
            "other_state": other_state.name,
            "fidelity": float(fidelity)
        }
        self.fidelity_history.append(record)
        
        return float(fidelity)
    
    def is_close_to(self, other_state: 'QuantumState', tolerance: float = 1e-8) -> bool:
        """
        Check if this state is approximately equal to another state.
        
        Args:
            other_state: The state to compare with
            tolerance: Tolerance for numerical comparison
            
        Returns:
            bool: True if states are close
        """
        if self.num_qubits != other_state.num_qubits:
            return False
            
        # Check state vector closeness
        return np.allclose(self.state_vector, other_state.state_vector, atol=tolerance)
    
    # Gate matrix helper methods
    def _get_pauli_x(self) -> np.ndarray:
        """Return the Pauli X gate matrix."""
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    
    def _get_pauli_y(self) -> np.ndarray:
        """Return the Pauli Y gate matrix."""
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    
    def _get_pauli_z(self) -> np.ndarray:
        """Return the Pauli Z gate matrix."""
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    def _get_hadamard(self) -> np.ndarray:
        """Return the Hadamard gate matrix."""
        return np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    
    def _get_s_gate(self) -> np.ndarray:
        """Return the S (phase) gate matrix."""
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    
    def _get_t_gate(self) -> np.ndarray:
        """Return the T gate matrix."""
        return np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
    
    def _get_rx_gate(self, theta: float) -> np.ndarray:
        """
        Return the RX (rotation around X-axis) gate matrix.
        
        Args:
            theta: Rotation angle in radians
        """
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ], dtype=np.complex128)
    
    def _get_ry_gate(self, theta: float) -> np.ndarray:
        """
        Return the RY (rotation around Y-axis) gate matrix.
        
        Args:
            theta: Rotation angle in radians
        """
        cos = np.cos(theta / 2)
        sin = np.sin(theta / 2)
        return np.array([
            [cos, -sin],
            [sin, cos]
        ], dtype=np.complex128)
    
    def _get_rz_gate(self, theta: float) -> np.ndarray:
        """
        Return the RZ (rotation around Z-axis) gate matrix.
        
        Args:
            theta: Rotation angle in radians
        """
        phase = np.exp(-1j * theta / 2)
        return np.array([
            [phase, 0],
            [0, phase.conjugate()]
        ], dtype=np.complex128)
    
    def _get_phase_gate(self, phi: float) -> np.ndarray:
        """
        Return the phase gate matrix.
        
        Args:
            phi: Phase angle in radians
        """
        return np.array([
            [1, 0],
            [0, np.exp(1j * phi)]
        ], dtype=np.complex128)
    
    def calculate_expectation_value(self, observable: Union[str, np.ndarray], qubits: Optional[List[int]] = None) -> complex:
        """
        Calculate the expectation value of an observable.
        
        Args:
            observable: Observable operator (matrix or Pauli string)
            qubits: Qubits to apply the observable to, or None for all
            
        Returns:
            complex: Expectation value ⟨ψ|O|ψ⟩
        """
        if qubits is None:
            qubits = list(range(self.num_qubits))
            
        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return 0.0
                
        # Handle different observable formats
        if isinstance(observable, str):
            # Pauli string like "XYZ" means tensor product X⊗Y⊗Z
            if len(observable) != len(qubits):
                logger.error(f"Pauli string length {len(observable)} doesn't match qubit count {len(qubits)}")
                return 0.0
                
            # Build the observable matrix
            obs_matrix = np.array([[1]], dtype=np.complex128)
            
            for i, pauli in enumerate(observable):
                if pauli.upper() == 'I':
                    pauli_matrix = np.eye(2, dtype=np.complex128)
                elif pauli.upper() == 'X':
                    pauli_matrix = self._get_pauli_x()
                elif pauli.upper() == 'Y':
                    pauli_matrix = self._get_pauli_y()
                elif pauli.upper() == 'Z':
                    pauli_matrix = self._get_pauli_z()
                else:
                    logger.error(f"Invalid Pauli operator: {pauli}")
                    return 0.0
                    
                obs_matrix = np.kron(obs_matrix, pauli_matrix)
                
            # Calculate ⟨ψ|O|ψ⟩
            expectation = np.vdot(self.state_vector, np.dot(obs_matrix, self.state_vector))
            
        elif isinstance(observable, np.ndarray):
            # Direct matrix form
            if observable.shape != (self.dimension, self.dimension):
                logger.error(f"Observable matrix shape {observable.shape} doesn't match expected dimension {(self.dimension, self.dimension)}")
                return 0.0
                
            # Calculate ⟨ψ|O|ψ⟩
            expectation = np.vdot(self.state_vector, np.dot(observable, self.state_vector))
            
        else:
            logger.error(f"Unsupported observable type: {type(observable)}")
            return 0.0
            
        return expectation
    
    def partial_trace(self, qubits_to_trace_out: List[int]) -> 'QuantumState':
        """
        Perform partial trace over specified qubits, returning a new quantum state.
        
        Args:
            qubits_to_trace_out: List of qubit indices to trace out
            
        Returns:
            QuantumState: New state representing reduced density matrix
        """
        # Validate qubit indices
        for q in qubits_to_trace_out:
            if q < 0 or q >= self.num_qubits:
                logger.error(f"Qubit index {q} out of range [0, {self.num_qubits-1}]")
                return None
                
        # Ensure we're not tracing out all qubits
        if len(qubits_to_trace_out) >= self.num_qubits:
            logger.error(f"Cannot trace out all qubits")
            return None
            
        # Determine which qubits to keep
        qubits_to_keep = [q for q in range(self.num_qubits) if q not in qubits_to_trace_out]
        num_qubits_kept = len(qubits_to_keep)
        
        # Get the full density matrix
        rho = self.get_density_matrix()
        
        # Calculate the reduced density matrix
        dim_keep = 2**num_qubits_kept
        rho_reduced = np.zeros((dim_keep, dim_keep), dtype=np.complex128)
        
        # Use the partial trace formula
        for i in range(dim_keep):
            for j in range(dim_keep):
                # Convert i, j to binary strings for the kept qubits
                i_bin = format(i, f'0{num_qubits_kept}b')
                j_bin = format(j, f'0{num_qubits_kept}b')
                
                # Sum over all possible values for traced-out qubits
                for k in range(2**len(qubits_to_trace_out)):
                    # Convert k to binary for traced-out qubits
                    k_bin = format(k, f'0{len(qubits_to_trace_out)}b')
                    
                    # Create full binary strings for the i, j indices
                    i_full = ['0'] * self.num_qubits
                    j_full = ['0'] * self.num_qubits
                    
                    # Fill in the kept qubits
                    for idx, q in enumerate(qubits_to_keep):
                        i_full[q] = i_bin[idx]
                        j_full[q] = j_bin[idx]
                        
                    # Fill in the traced-out qubits with the same values
                    for idx, q in enumerate(qubits_to_trace_out):
                        i_full[q] = k_bin[idx]
                        j_full[q] = k_bin[idx]
                        
                    # Convert binary strings to integers
                    i_idx = int(''.join(i_full), 2)
                    j_idx = int(''.join(j_full), 2)
                    
                    # Add to the reduced density matrix
                    rho_reduced[i, j] += rho[i_idx, j_idx]
        
        # Create a new quantum state for the result
        result = QuantumState(f"{self.name}_reduced", num_qubits_kept, state_type=StateType.MIXED)
        result.set_density_matrix(rho_reduced)
        
        return result