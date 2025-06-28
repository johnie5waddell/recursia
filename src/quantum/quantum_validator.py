#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Quantum Validator for Recursia - FIXED VERSION

This module implements the validator for quantum operations in the Recursia language,
providing static validation for quantum gate applications, measurements, entanglement,
teleportation, and qubit specifications. It integrates with the symbol table and type system
to ensure quantum semantic correctness during the compilation phase.

The QuantumValidator is a critical component of the Organic Simulation Hypothesis (OSH)
paradigm implementation, enforcing quantum consistency and physical plausibility.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple, Union, Any
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
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()

from src.core.data_classes import SemanticError
from src.core.symbol_table import SymbolTable

# Setup logging
logger = logging.getLogger(__name__)

class QuantumValidator:
    """
    Validates quantum operations in the Recursia language against the symbol table.
    
    Performs semantic validation of:
    - Quantum states and their types
    - Gate applications and compatibility
    - Measurement correctness
    - Entanglement relationships and protocols
    - Teleportation prerequisites and qubit specifications
    - Expression references to quantum states
    - OSH-aligned coherence, entropy, and recursion properties
    
    Integrates with the symbol table to resolve identifiers and validate
    type compatibility across quantum operations.
    """
    
    def __init__(self, symbol_table: SymbolTable):
        """
        Initialize the quantum validator with a symbol table.
        
        Args:
            symbol_table: The symbol table containing state declarations
        """
        self.symbol_table = symbol_table
        self.errors: List[SemanticError] = []
        
        # Register standard gates
        self.valid_gates = {
            # Single-qubit gates
            "H_gate", "X_gate", "Y_gate", "Z_gate", "S_gate", "T_gate", "P_gate", "I_gate",
            "Hadamard_gate", "PauliX_gate", "PauliY_gate", "PauliZ_gate", "PhaseShift_gate",
            "SqrtX_gate", "SqrtY_gate", "SqrtZ_gate", "SqrtW_gate", "SqrtNOT_gate",
            "PHASE_gate", "phase_gate",  # Add PHASE gate variants
            # Short forms
            "H", "X", "Y", "Z", "S", "T", "P", "I", "PHASE",
            
            # Parameterized single-qubit gates
            "RX_gate", "RY_gate", "RZ_gate", "U_gate", "U1_gate", "U2_gate", "U3_gate",
            
            # Two-qubit gates
            "CNOT_gate", "CX_gate", "CZ_gate", "SWAP_gate", "ControlledPhaseShift_gate", 
            "ControlledZ_gate", "AdjacentControlledPhaseShift_gate", "ControlledSWAP_gate",
            
            # Three-qubit gates
            "TOFFOLI_gate", "CCNOT_gate", "CSWAP_gate",
            
            # Multi-qubit algorithms
            "QFT_gate", "InverseQFT_gate", "Oracle_gate", "Grover_gate", "Shor_gate",
            "VQE_gate", "QAOA_gate", "Trotter_gate", "RandomUnitary_gate",
            
            # Physics-inspired gates
            "Ising_gate", "Heisenberg_gate", "FermiHubbard_gate"
        }
        
        # Register standard measurement bases
        self.valid_basis = {
            "standard_basis", "Z_basis", "X_basis", "Y_basis", "Bell_basis", 
            "GHZ_basis", "W_basis", "Magic_basis", "computational_basis", 
            "hadamard_basis", "pauli_basis", "circular_basis"
        }
        
        # Register valid entanglement protocols
        self.valid_entanglement_protocols = {
            "direct_protocol", "CNOT_protocol", "Hadamard_protocol", "EPR_protocol",
            "GHZ_protocol", "W_protocol", "cluster_protocol", "graph_state_protocol",
            "AKLT_protocol", "kitaev_honeycomb_protocol", "tensor_network_protocol"
        }
        
        # Register valid teleportation protocols
        self.valid_teleportation_protocols = {
            "standard_protocol", "dense_coding_protocol", "superdense_protocol",
            "entanglement_swapping_protocol", "quantum_repeater_protocol",
            "teleportation_circuit_protocol", "remote_state_preparation_protocol"
        }
        
        # Valid quantum state types
        self.quantum_state_types = {
            "quantum_type", "superposition_type", "entangled_type", "mixed_type",
            "field_type", "waveform_type"
        }
        
        # OSH-specific type and property constraints
        self.osh_property_ranges = {
            "state_coherence": (0.0, 1.0),
            "state_entropy": (0.0, 1.0),
            "state_recursion_depth": (0, 10),  # Maximum recursion depth for OSH
            "observer_collapse_threshold": (0.0, 1.0)
        }

    def _extract_numeric_value(self, value) -> Union[int, float, None]:
        """
        Extract the numeric value from various value types.
        
        Args:
            value: The value to extract from (could be NumberLiteral, int, float, str, etc.)
            
        Returns:
            Union[int, float, None]: The extracted numeric value or None if extraction fails
        """
        try:
            # If it's already a number
            if isinstance(value, (int, float)):
                return value
            
            # If it's a NumberLiteral object
            if hasattr(value, 'value'):
                if isinstance(value.value, (int, float)):
                    return value.value
                elif isinstance(value.value, str):
                    # Try to parse the string value
                    try:
                        if '.' in value.value or 'e' in value.value.lower():
                            return float(value.value)
                        else:
                            return int(value.value)
                    except ValueError:
                        logger.warning(f"Could not parse numeric value from string: {value.value}")
                        return None
            
            # If it's a string representation of a number
            if isinstance(value, str):
                try:
                    if '.' in value or 'e' in value.lower():
                        return float(value)
                    else:
                        return int(value)
                except ValueError:
                    logger.warning(f"Could not parse numeric value from string: {value}")
                    return None
            
            # If it has an expression attribute
            if hasattr(value, 'expression'):
                return self._extract_numeric_value(value.expression)
            
            logger.warning(f"Could not extract numeric value from: {type(value)} - {value}")
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting numeric value: {e}")
            return None
    
    def validate_quantum_state(self, state_name: str, location: Tuple[int, int, int]) -> bool:
        """
        Validate that a named state exists and is a valid quantum state.
        
        This method checks if:
        1. The state exists in the symbol table
        2. The state has a type compatible with quantum operations
        3. The state meets OSH-aligned type requirements
        
        Args:
            state_name: Name of the state to validate
            location: Source location (line, column, length) for error reporting
            
        Returns:
            bool: True if state is valid, False otherwise
        """
        state_def = self.symbol_table.get_state(state_name)
        if state_def is None:
            # Check if it's a function parameter (could be a state reference at runtime)
            var_def = self.symbol_table.get_variable(state_name)
            if var_def is not None:
                # This is a variable/parameter that might contain a state reference
                # Allow it for now - runtime will validate the actual reference
                return True
            
            self.errors.append(SemanticError(
                f"Undefined quantum state '{state_name}'",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        # Check if the state type is a quantum type
        state_type = "quantum_type"  # Default
        if hasattr(state_def, "state_type") and state_def.state_type is not None:
            state_type = state_def.state_type
        elif hasattr(state_def, "type_name") and state_def.type_name is not None:
            state_type = state_def.type_name
        
        if state_type not in self.quantum_state_types:
            self.errors.append(SemanticError(
                f"State '{state_name}' has type '{state_type}' which is not a quantum state type. Valid types: {', '.join(self.quantum_state_types)}",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
        
        # Validate OSH-specific properties
        return self._validate_osh_properties(state_def, state_name, location)
    
    def _get_state_num_qubits(self, state_def) -> int:
        """
        Extract the number of qubits from a state definition.
        
        Args:
            state_def: State definition from symbol table
            
        Returns:
            int: Number of qubits, defaulting to 1 if not found
        """
        # First check if num_qubits is directly available
        if hasattr(state_def, 'num_qubits') and state_def.num_qubits is not None:
            numeric_value = self._extract_numeric_value(state_def.num_qubits)
            if numeric_value is not None:
                return int(numeric_value)
        
        # Check fields for state_qubits
        if hasattr(state_def, 'fields') and state_def.fields:
            for field_name, field_value in state_def.fields.items():
                if field_name == "state_qubits":
                    numeric_value = self._extract_numeric_value(field_value)
                    if numeric_value is not None:
                        return int(numeric_value)
        
        # Default to 1 qubit if not found
        return 1
    
    def _validate_osh_properties(self, state_def, state_name: str, location: Tuple[int, int, int]) -> bool:
        """
        Validate Organic Simulation Hypothesis (OSH) properties of a quantum state.
        
        Checks coherence, entropy, and recursive mechanics integration to ensure
        the state conforms to OSH principles of consciousness and recursion.
        
        Args:
            state_def: State definition from symbol table
            state_name: Name of the state
            location: Source location for error reporting
            
        Returns:
            bool: True if OSH properties are valid, False otherwise
        """
        if not hasattr(state_def, "fields"):
            return True
        
        # Check each OSH property against its allowed range
        for prop_name, (min_val, max_val) in self.osh_property_ranges.items():
            if prop_name in state_def.fields:
                prop_val = state_def.fields[prop_name]
                numeric_value = self._extract_numeric_value(prop_val)
                
                if numeric_value is not None:
                    if numeric_value < min_val or numeric_value > max_val:
                        self.errors.append(SemanticError(
                            f"OSH property '{prop_name}' value {numeric_value} for state '{state_name}' must be between {min_val} and {max_val}",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                else:
                    self.errors.append(SemanticError(
                        f"Invalid OSH property '{prop_name}' value for state '{state_name}'. Must be a number between {min_val} and {max_val}",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
        
        # Validate coherence and entropy relationship for OSH coherence dynamics
        if "state_coherence" in state_def.fields and "state_entropy" in state_def.fields:
            coherence_val = self._extract_numeric_value(state_def.fields["state_coherence"])
            entropy_val = self._extract_numeric_value(state_def.fields["state_entropy"])
            
            if coherence_val is not None and entropy_val is not None:
                # In OSH theory, high coherence with high entropy is physically implausible
                if coherence_val > 0.8 and entropy_val > 0.8:
                    logger.warning(
                        f"OSH warning: State '{state_name}' has both high coherence ({coherence_val}) and high entropy ({entropy_val}), which is physically implausible under the Organic Simulation Hypothesis"
                    )
        
        # Validate recursive boundary properties
        if "state_recursion_boundary" in state_def.fields:
            boundary_val = state_def.fields["state_recursion_boundary"]
            if hasattr(boundary_val, "value") and boundary_val.value == True:
                # Check if the state has necessary boundary properties
                if "state_recursion_parent" not in state_def.fields:
                    logger.warning(
                        f"OSH warning: Recursive boundary state '{state_name}' should define 'state_recursion_parent'"
                    )
                
                if "state_recursion_depth" not in state_def.fields:
                    logger.warning(
                        f"OSH warning: Recursive boundary state '{state_name}' should define 'state_recursion_depth'"
                    )
        
        return True
    
    def validate_gate_application(self, stmt: ApplyGateStatement) -> bool:
        """
        Validate a gate application statement.
        
        Performs comprehensive checking of gate syntax, target state compatibility,
        qubit indices, control qubits, parameters, and OSH coherence implications.
        
        Args:
            stmt: The gate application AST node
            
        Returns:
            bool: True if the gate application is valid
        """
        # Check if the gate type is valid
        if stmt.gate_type not in self.valid_gates:
            self.errors.append(SemanticError(
                f"Unknown gate type '{stmt.gate_type}'. Valid gates include: {', '.join(sorted(list(self.valid_gates))[:5])}...",
                "current_file",
                stmt.location[0],
                stmt.location[1],
                "quantum"
            ))
            return False
            
        # Validate the target state(s) - handle both strings and IdentifierExpression objects
        if isinstance(stmt.target, list):
            targets = []
            for t in stmt.target:
                if hasattr(t, 'name'):  # IdentifierExpression
                    targets.append(t.name)
                else:  # string
                    targets.append(t)
        else:
            if hasattr(stmt.target, 'name'):  # IdentifierExpression
                targets = [stmt.target.name]
            else:  # string
                targets = [stmt.target]
        
        target_states = []
        for target in targets:
            is_valid = self.validate_quantum_state(target, stmt.location)
            if not is_valid:
                return False
            target_states.append(target)
            
        # For single target gates, get the state definition to extract number of qubits
        if len(targets) == 1:
            state_def = self.symbol_table.get_state(targets[0])
            num_qubits = self._get_state_num_qubits(state_def)
            
        # Validate gate arguments
        if hasattr(stmt, "args") and stmt.args:
            for i, arg in enumerate(stmt.args):
                arg_type = self._get_expression_type(arg)
                if arg_type is None or arg_type.name not in ["number_type", "complex_type"]:
                    self.errors.append(SemanticError(
                        f"Gate argument must be a number or complex value, got '{arg_type.name if arg_type else 'unknown'}'",
                        "current_file",
                        stmt.location[0],
                        stmt.location[1],
                        "quantum"
                    ))
                    return False
                    
        # Validate qubit specification if provided
        if stmt.qubit_spec:
            is_valid = self._validate_qubit_specification(
                num_qubits, stmt.qubit_spec, stmt.target, stmt.location
            )
            if not is_valid:
                return False
                
        # Validate control specification if provided
        if stmt.control_spec:
            # Use the first target for control validation (multi-qubit gates handle this differently)
            primary_target = targets[0] if targets else None
            is_valid = self._validate_control_specification(
                stmt.control_spec, primary_target, stmt.location
            )
            if not is_valid:
                return False
                
        # Validate parameters specification if provided
        if stmt.params_spec:
            is_valid = self._validate_params_specification(
                stmt.params_spec, stmt.gate_type, stmt.location
            )
            if not is_valid:
                return False
                
        # Check compatibility of the gate with the number of qubits and controls
        is_valid = self._validate_gate_compatibility(
            stmt.gate_type, 
            stmt.qubit_spec, 
            stmt.control_spec, 
            stmt.params_spec, 
            stmt.location
        )
        if not is_valid:
            return False
            
        # Validate multi-target gate requirements
        if not self._validate_multi_target_gate(stmt.gate_type, targets, stmt.location):
            return False
            
        # Validate OSH coherence impact for certain gates
        if stmt.gate_type in ["H_gate", "Hadamard_gate", "QFT_gate", "Grover_gate"]:
            # These gates significantly increase superposition and may affect system coherence
            if hasattr(state_def, "fields") and "state_coherence" in state_def.fields:
                coherence_val = self._extract_numeric_value(state_def.fields["state_coherence"])
                if coherence_val is not None and coherence_val < 0.3:
                    logger.warning(
                        f"OSH warning: Applying high-superposition gate '{stmt.gate_type}' to low-coherence state '{stmt.target}' may lead to quantum decoherence"
                    )
            
        return True
    
    def validate_measure(self, stmt: MeasureStatement) -> bool:
        """
        Validate a measurement statement.
        
        Verifies measurement target state, qubit selection, basis validity,
        and destination variable compatibility. Also checks OSH-related 
        observer effects and collapse thresholds.
        
        Args:
            stmt: The measurement AST node
            
        Returns:
            bool: True if the measurement is valid
        """
        # Validate the target state
        is_valid = self.validate_quantum_state(stmt.target, stmt.location)
        if not is_valid:
            return False
            
        # Get the state definition to extract number of qubits
        state_def = self.symbol_table.get_state(stmt.target)
        num_qubits = self._get_state_num_qubits(state_def)
            
        # Validate qubit specification if provided
        if stmt.qubits is not None:
            # Handle different types of stmt.qubits properly
            try:
                # If it's already a QubitSpec object, use it directly
                if isinstance(stmt.qubits, QubitSpec):
                    qubit_spec = stmt.qubits
                # Handle primitive types
                elif isinstance(stmt.qubits, int):
                    qubit_spec = QubitSpec(
                        kind="single",
                        value=stmt.qubits,
                        location=stmt.location
                    )
                elif isinstance(stmt.qubits, list):
                    qubit_spec = QubitSpec(
                        kind="list",
                        value=stmt.qubits,
                        location=stmt.location
                    )
                else:
                    # For any other type, default to qubit 0
                    qubit_spec = QubitSpec(
                        kind="single",
                        value=0,
                        location=stmt.location
                    )
                    logger.warning(
                        f"Unrecognized qubit specification type: {type(stmt.qubits)}. Using default qubit 0."
                    )
                
                is_valid = self._validate_qubit_specification(
                    num_qubits, qubit_spec, stmt.target, stmt.location
                )
                if not is_valid:
                    return False
                    
            except Exception as e:
                # If creating the QubitSpec fails, log a detailed error
                self.errors.append(SemanticError(
                    f"Invalid qubit specification: {str(e)}",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
        
        # Validate measurement basis if provided
        if stmt.basis and stmt.basis not in self.valid_basis:
            # Check if it's a string literal or identifier reference
            if isinstance(stmt.basis, str) and not stmt.basis.startswith('"') and not stmt.basis.startswith("'"):
                # It might be a variable reference
                basis_var = self.symbol_table.get_variable(stmt.basis)
                if basis_var is None:
                    self.errors.append(SemanticError(
                        f"Unknown measurement basis '{stmt.basis}'. Valid bases include: {', '.join(sorted(list(self.valid_basis))[:5])}...",
                        "current_file",
                        stmt.location[0],
                        stmt.location[1],
                        "quantum"
                    ))
                    return False
            else:
                self.errors.append(SemanticError(
                    f"Unknown measurement basis '{stmt.basis}'. Valid bases include: {', '.join(sorted(list(self.valid_basis))[:5])}...",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
                    
        # Validate destination if provided
        if stmt.destination:
            # If destination is a new variable, that's fine
            # If it's an existing variable, it should be compatible
            dest_var = self.symbol_table.get_variable(stmt.destination)
            if dest_var is not None:
                # Check if the variable type is compatible with measurement result
                var_type = dest_var.var_type
                if var_type and var_type.name not in ["any_type", "number_type", "string_type", "array_type", "object_type"]:
                    self.errors.append(SemanticError(
                        f"Measurement destination variable '{stmt.destination}' has incompatible type '{var_type.name}'. Expected 'number_type', 'string_type', 'array_type', or 'object_type'",
                        "current_file",
                        stmt.location[0],
                        stmt.location[1],
                        "quantum"
                    ))
                    return False
        
        # Validate measurement against observer properties for OSH compliance
        if hasattr(state_def, "fields"):
            # Check for coherence and measure relationship
            if "state_coherence" in state_def.fields:
                coherence_val = self._extract_numeric_value(state_def.fields["state_coherence"])
                if coherence_val is not None and coherence_val < 0.1:
                    logger.warning(
                        f"OSH warning: Measuring state '{stmt.target}' with very low coherence ({coherence_val}) may yield unstable results"
                    )
            
            # Check if measurement has observer dependency
            if "state_observer_dependency" in state_def.fields:
                observer_dep = state_def.fields["state_observer_dependency"]
                if hasattr(observer_dep, "value") and observer_dep.value == True:
                    # State is observer-dependent, check if an observer is specified
                    if not hasattr(stmt, "observer") or not stmt.observer:
                        logger.warning(
                            f"OSH warning: Measuring observer-dependent state '{stmt.target}' without specifying an observer may yield inconsistent results under OSH"
                        )
            
            # Check for measurement collapse properties
            if "state_collapse_resistant" in state_def.fields:
                resist_val = state_def.fields["state_collapse_resistant"]
                if hasattr(resist_val, "value") and resist_val.value == True:
                    # The state is marked as resistant to collapse
                    logger.warning(
                        f"OSH warning: State '{stmt.target}' is marked as collapse-resistant, measurement may have limited effect"
                    )
        
        # All validations passed
        return True

    def validate_entangle(self, stmt: EntangleStatement) -> bool:
        """
        Validate an entanglement statement.
        
        Performs comprehensive validation of entanglement operations including 
        state compatibility, qubit specifications, protocols, and OSH-aligned 
        coherence dynamics checks.
        
        Args:
            stmt: The entanglement AST node
            
        Returns:
            bool: True if the entanglement is valid
        """
        # Get direct target names if available, handling IdentifierExpression objects
        target1_name = stmt.target1
        target2_name = stmt.target2
        
        # Handle case where target is an IdentifierExpression instead of string
        if hasattr(target1_name, 'name'):
            target1_name = target1_name.name
        if hasattr(target2_name, 'name'):
            target2_name = target2_name.name
        
        # Additional safety check - ensure targets are strings
        if target1_name is not None and not isinstance(target1_name, str):
            target1_name = str(target1_name)
        if target2_name is not None and not isinstance(target2_name, str):
            target2_name = str(target2_name)
        
        # Handle potential expression-based targets
        if target1_name is None and stmt.target1_expr:
            target1_name = self._resolve_expression_to_state(stmt.target1_expr)
            if target1_name is None:
                self.errors.append(SemanticError(
                    f"First entanglement target must be a valid quantum state reference",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
        
        if target2_name is None and stmt.target2_expr:
            target2_name = self._resolve_expression_to_state(stmt.target2_expr)
            if target2_name is None:
                self.errors.append(SemanticError(
                    f"Second entanglement target must be a valid quantum state reference",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
                
        # Validate both target states
        if not target1_name or not target2_name:
            self.errors.append(SemanticError(
                f"Entanglement requires two target states",
                "current_file",
                stmt.location[0],
                stmt.location[1],
                "quantum"
            ))
            return False
            
        is_valid1 = self.validate_quantum_state(target1_name, stmt.location)
        is_valid2 = self.validate_quantum_state(target2_name, stmt.location)
        if not is_valid1 or not is_valid2:
            return False
            
        # Check that states are different
        if target1_name == target2_name:
            self.errors.append(SemanticError(
                f"Cannot entangle a state with itself",
                "current_file",
                stmt.location[0],
                stmt.location[1],
                "quantum"
            ))
            return False
            
        # Validate qubit specifications if provided
        state1_def = self.symbol_table.get_state(target1_name)
        state2_def = self.symbol_table.get_state(target2_name)
        
        num_qubits1 = self._get_state_num_qubits(state1_def)
        num_qubits2 = self._get_state_num_qubits(state2_def)
                    
        if stmt.target1_qubits:
            is_valid = self._validate_qubit_specification(
                num_qubits1, stmt.target1_qubits, target1_name, stmt.location
            )
            if not is_valid:
                return False
                
        if stmt.target2_qubits:
            is_valid = self._validate_qubit_specification(
                num_qubits2, stmt.target2_qubits, target2_name, stmt.location
            )
            if not is_valid:
                return False
                
        # Validate entanglement protocol if provided
        if stmt.protocol and stmt.protocol not in self.valid_entanglement_protocols:
            # Check if it's a string literal or variable reference
            if isinstance(stmt.protocol, str) and not stmt.protocol.startswith('"') and not stmt.protocol.startswith("'"):
                protocol_var = self.symbol_table.get_variable(stmt.protocol)
                if protocol_var is None:
                    self.errors.append(SemanticError(
                        f"Unknown entanglement protocol '{stmt.protocol}'. Valid protocols include: {', '.join(sorted(list(self.valid_entanglement_protocols))[:5])}...",
                        "current_file",
                        stmt.location[0],
                        stmt.location[1],
                        "quantum"
                    ))
                    return False
            else:
                self.errors.append(SemanticError(
                    f"Unknown entanglement protocol '{stmt.protocol}'. Valid protocols include: {', '.join(sorted(list(self.valid_entanglement_protocols))[:5])}...",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
        
        # If protocol is specified, validate protocol-specific requirements
        if stmt.protocol and isinstance(stmt.protocol, str):
            is_valid = self._validate_entanglement_protocol(
                stmt.protocol, target1_name, target2_name, stmt.location
            )
            if not is_valid:
                return False
                
        # Validate entanglement type if provided
        if stmt.entanglement_type:
            # Check if specific entanglement types are compatible with state types
            if stmt.entanglement_type == "GHZ" and (num_qubits1 + num_qubits2 < 3):
                self.errors.append(SemanticError(
                    f"GHZ entanglement requires at least 3 qubits total, but states '{target1_name}' and '{target2_name}' only have {num_qubits1 + num_qubits2} qubits",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
                
            if stmt.entanglement_type == "W" and (num_qubits1 + num_qubits2 < 3):
                self.errors.append(SemanticError(
                    f"W entanglement requires at least 3 qubits total, but states '{target1_name}' and '{target2_name}' only have {num_qubits1 + num_qubits2} qubits",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
                
        # Update entanglement relationships in state definitions if both are resolved
        if state1_def and state2_def:
            # Add mutual entanglement relationship
            state1_entangled = state1_def.fields.get("state_entangled_with", None)
            state2_entangled = state2_def.fields.get("state_entangled_with", None)
            
            # These operations are for static analysis only - runtime will handle actual entanglement
            if state1_entangled is None:
                # Create new entangled_with field
                state1_def.fields["state_entangled_with"] = {"value": [target2_name]}
            elif isinstance(state1_entangled, dict) and "value" in state1_entangled:
                if isinstance(state1_entangled["value"], list):
                    # Convert existing items to strings for comparison to avoid unhashable type errors
                    existing_names = []
                    for item in state1_entangled["value"]:
                        if hasattr(item, 'name'):
                            existing_names.append(item.name)
                        else:
                            existing_names.append(str(item))
                    
                    if target2_name not in existing_names:
                        state1_entangled["value"].append(target2_name)
                
            if state2_entangled is None:
                # Create new entangled_with field
                state2_def.fields["state_entangled_with"] = {"value": [target1_name]}
            elif isinstance(state2_entangled, dict) and "value" in state2_entangled:
                if isinstance(state2_entangled["value"], list):
                    # Convert existing items to strings for comparison to avoid unhashable type errors
                    existing_names = []
                    for item in state2_entangled["value"]:
                        if hasattr(item, 'name'):
                            existing_names.append(item.name)
                        else:
                            existing_names.append(str(item))
                    
                    if target1_name not in existing_names:
                        state2_entangled["value"].append(target1_name)
            
            # For OSH-aligned simulations, mark both states as entangled type 
            state1_def.fields["state_type"] = {"value": "entangled_type"}
            state2_def.fields["state_type"] = {"value": "entangled_type"}
            
            # Update coherence values based on OSH entanglement principle
            if "state_coherence" in state1_def.fields and "state_coherence" in state2_def.fields:
                coherence1_val = self._extract_numeric_value(state1_def.fields["state_coherence"])
                coherence2_val = self._extract_numeric_value(state2_def.fields["state_coherence"])
                
                if coherence1_val is not None and coherence2_val is not None:
                    avg_coherence = (coherence1_val + coherence2_val) / 2.0
                    if avg_coherence < 0.4:
                        logger.warning(
                            f"OSH warning: Entangling states with low average coherence ({avg_coherence:.2f}) may result in rapid decoherence"
                        )
        
        # Check for recursion boundary crossing in OSH model
        if state1_def and state2_def and hasattr(state1_def, "fields") and hasattr(state2_def, "fields"):
            if "state_recursion_depth" in state1_def.fields and "state_recursion_depth" in state2_def.fields:
                depth1_val = self._extract_numeric_value(state1_def.fields["state_recursion_depth"])
                depth2_val = self._extract_numeric_value(state2_def.fields["state_recursion_depth"])
                
                if depth1_val is not None and depth2_val is not None:
                    if abs(depth1_val - depth2_val) > 1:
                        logger.warning(
                            f"OSH warning: Entangling states across recursion depths ({depth1_val} and {depth2_val}) may cause simulation instability"
                        )
            
        return True
    
    def _validate_entanglement_protocol(self, protocol: str, target1: str, target2: str, 
                                 location: Tuple[int, int, int]) -> bool:
        """
        Validate protocol-specific requirements for entanglement.
        
        Each protocol has specific requirements for state types, qubit counts,
        and coherence capabilities.
        
        Args:
            protocol: Entanglement protocol name
            target1: First target state name
            target2: Second target state name
            location: Source location for error reporting
            
        Returns:
            bool: True if protocol requirements are met, False otherwise
        """
        state1_def = self.symbol_table.get_state(target1)
        state2_def = self.symbol_table.get_state(target2)
        
        # Get qubit counts from state definitions
        qubits1 = self._get_state_num_qubits(state1_def)
        qubits2 = self._get_state_num_qubits(state2_def)
        
        # Protocol-specific validations
        if protocol == "GHZ_protocol":
            # GHZ protocol requires at least 3 qubits total
            total_qubits = qubits1 + qubits2
            if total_qubits < 3:
                self.errors.append(SemanticError(
                    f"GHZ protocol requires at least 3 qubits total, but states '{target1}' and '{target2}' only have {total_qubits} qubits",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
        
        elif protocol == "W_protocol":
            # W protocol requires at least 3 qubits total
            total_qubits = qubits1 + qubits2
            if total_qubits < 3:
                self.errors.append(SemanticError(
                    f"W protocol requires at least 3 qubits total, but states '{target1}' and '{target2}' only have {total_qubits} qubits",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
        
        elif protocol == "cluster_protocol" or protocol == "graph_state_protocol":
            # These protocols work better with larger qubit counts
            total_qubits = qubits1 + qubits2
            if total_qubits < 4:
                logger.warning(
                    f"OSH warning: Protocol '{protocol}' works best with at least 4 qubits, but only {total_qubits} qubits are available"
                )
        
        elif protocol == "AKLT_protocol" or protocol == "kitaev_honeycomb_protocol":
            # These protocols model specific physics systems and have OSH implications
            # Check for coherence properties
            for state_def, state_name in [(state1_def, target1), (state2_def, target2)]:
                if state_def and hasattr(state_def, "fields"):
                    if "state_coherence" in state_def.fields:
                        coherence_val = self._extract_numeric_value(state_def.fields["state_coherence"])
                        if coherence_val is not None and coherence_val < 0.5:
                            logger.warning(
                                f"OSH warning: State '{state_name}' has low coherence ({coherence_val}), which may lead to unstable behavior with '{protocol}'"
                            )
            
            # Check qubit count for honeycomb - needs minimum of 6 total qubits
            if protocol == "kitaev_honeycomb_protocol" and qubits1 + qubits2 < 6:
                logger.warning(
                    f"OSH warning: '{protocol}' requires at least 6 qubits for proper simulation, but only {qubits1 + qubits2} qubits are available"
                )
        
        return True
    
    def validate_teleport(self, stmt: TeleportStatement) -> bool:
        """
        Validate a teleportation statement.
        
        Performs thorough verification of teleportation operations including
        source/destination existence, qubit specifications, protocols, and 
        OSH-aligned requirements like entanglement preconditions.
        
        Args:
            stmt: The teleportation AST node
            
        Returns:
            bool: True if the teleportation is valid
        """
        # Validate source and destination states
        is_valid_source = self.validate_quantum_state(stmt.source, stmt.location)
        is_valid_dest = self.validate_quantum_state(stmt.destination, stmt.location)
        if not is_valid_source or not is_valid_dest:
            return False
            
        # FIXED: Allow teleportation within the same register if different qubits are specified
        # Get the qubit specifications to determine if they're actually different
        source_qubit_ids = self._get_qubit_indices_from_spec(stmt.source_qubits, 100) if stmt.source_qubits else [0]
        dest_qubit_ids = self._get_qubit_indices_from_spec(stmt.destination_qubits, 100) if stmt.destination_qubits else [0]
        
        # Check if source and destination are the same register AND the same qubits
        if stmt.source == stmt.destination and source_qubit_ids == dest_qubit_ids:
            self.errors.append(SemanticError(
                f"Source and destination qubits for teleportation must be different (cannot teleport qubit to itself)",
                "current_file",
                stmt.location[0],
                stmt.location[1],
                "quantum"
            ))
            return False
            
        # Validate qubit specifications if provided
        source_def = self.symbol_table.get_state(stmt.source)
        destination_def = self.symbol_table.get_state(stmt.destination)
        
        num_qubits_source = self._get_state_num_qubits(source_def)
        num_qubits_dest = self._get_state_num_qubits(destination_def)
        
        if stmt.source_qubits:
            is_valid = self._validate_qubit_specification(
                num_qubits_source, stmt.source_qubits, stmt.source, stmt.location
            )
            if not is_valid:
                return False
                
        if stmt.destination_qubits:
            is_valid = self._validate_qubit_specification(
                num_qubits_dest, stmt.destination_qubits, stmt.destination, stmt.location
            )
            if not is_valid:
                return False
                
        # Validate teleportation protocol if provided
        if stmt.protocol and stmt.protocol not in self.valid_teleportation_protocols:
            # Check if it's a string literal or variable reference
            if isinstance(stmt.protocol, str) and not stmt.protocol.startswith('"') and not stmt.protocol.startswith("'"):
                protocol_var = self.symbol_table.get_variable(stmt.protocol)
                if protocol_var is None:
                    self.errors.append(SemanticError(
                        f"Unknown teleportation protocol '{stmt.protocol}'. Valid protocols include: {', '.join(sorted(list(self.valid_teleportation_protocols))[:5])}...",
                        "current_file",
                        stmt.location[0],
                        stmt.location[1],
                        "quantum"
                    ))
                    return False
            else:
                self.errors.append(SemanticError(
                    f"Unknown teleportation protocol '{stmt.protocol}'. Valid protocols include: {', '.join(sorted(list(self.valid_teleportation_protocols))[:5])}...",
                    "current_file",
                    stmt.location[0],
                    stmt.location[1],
                    "quantum"
                ))
                return False
                
        # Check if the states are entangled (required for teleportation between different registers)
        if stmt.source != stmt.destination and source_def and destination_def:
            source_entangled = source_def.fields.get("state_entangled_with", {}).get("value", [])
            dest_entangled = destination_def.fields.get("state_entangled_with", {}).get("value", [])
            
            if isinstance(source_entangled, list) and isinstance(dest_entangled, list):
                # Convert entangled lists to strings for comparison to avoid unhashable type errors
                source_entangled_names = []
                for item in source_entangled:
                    if hasattr(item, 'name'):
                        source_entangled_names.append(item.name)
                    else:
                        source_entangled_names.append(str(item))
                
                dest_entangled_names = []
                for item in dest_entangled:
                    if hasattr(item, 'name'):
                        dest_entangled_names.append(item.name)
                    else:
                        dest_entangled_names.append(str(item))
                
                if stmt.destination not in source_entangled_names or stmt.source not in dest_entangled_names:
                    # Not an error, but a warning - teleportation requires entanglement
                    logger.warning(
                        f"OSH warning: Teleportation from '{stmt.source}' to '{stmt.destination}' may not work: states are not entangled. Consider using 'entangle' first."
                    )
            
            # Validate OSH coherence requirements
            for state_def, state_name in [(source_def, stmt.source), (destination_def, stmt.destination)]:
                if hasattr(state_def, "fields") and "state_coherence" in state_def.fields:
                    coherence_val = self._extract_numeric_value(state_def.fields["state_coherence"])
                    if coherence_val is not None and coherence_val < 0.5:
                        logger.warning(
                            f"OSH warning: State '{state_name}' has low coherence ({coherence_val}), which may cause teleportation errors"
                        )
            
            # Validate protocol-specific requirements
            if stmt.protocol:
                if stmt.protocol == "dense_coding_protocol" or stmt.protocol == "superdense_protocol":
                    # These protocols require specific qubit constraints
                    if num_qubits_source < 2 or num_qubits_dest < 2:
                        logger.warning(
                            f"OSH warning: Protocol '{stmt.protocol}' works best with at least 2 qubits per state"
                        )
                
                elif stmt.protocol == "quantum_repeater_protocol":
                    # Quantum repeaters need multiple entangled pairs
                    if not source_entangled_names or not dest_entangled_names or len(source_entangled_names) < 2 or len(dest_entangled_names) < 2:
                        logger.warning(
                            f"OSH warning: '{stmt.protocol}' typically requires multiple entangled states for proper operation"
                        )
                
                elif stmt.protocol == "entanglement_swapping_protocol":
                    # Check for intermediate state (required for swapping)
                    intermediate_found = False
                    for ent_state in source_entangled_names:
                        if ent_state != stmt.destination and ent_state in dest_entangled_names:
                            intermediate_found = True
                            break
                    
                    if not intermediate_found:
                        logger.warning(
                            f"OSH warning: '{stmt.protocol}' requires an intermediate entangled state shared by both source and destination"
                        )
        
        return True
    
    def _validate_qubit_specification(self, num_qubits: int, qubit_spec: QubitSpec, 
                                    state_name: str, location: Tuple[int, int, int]) -> bool:
        """
        Validate a qubit specification for compatibility with a quantum state.
        
        Verifies that the qubit indices are within range and follow correct format
        for the specified state.
        
        Args:
            num_qubits: Total number of qubits in the state
            qubit_spec: Qubit specification to validate
            state_name: Name of the target state
            location: Source location for error reporting
            
        Returns:
            bool: True if the qubit specification is valid
        """
        if qubit_spec.kind == "single":
            # Get the actual value, handling various value types
            qubit_value = self._extract_numeric_value(qubit_spec.value)
            
            if qubit_value is None:
                self.errors.append(SemanticError(
                    f"Invalid qubit index: could not extract numeric value from {qubit_spec.value}",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            qubit_value = int(qubit_value)
            if qubit_value < 0 or qubit_value >= num_qubits:
                self.errors.append(SemanticError(
                    f"Qubit index {qubit_value} out of range [0, {num_qubits-1}] for state '{state_name}'",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
        elif qubit_spec.kind == "list":
            # List of qubit indices
            if isinstance(qubit_spec.value, list):
                # Create a list to track seen indices for duplicate detection
                seen_indices = []
                has_duplicates = False
                
                for idx in qubit_spec.value:
                    # Extract numeric value
                    idx_value = self._extract_numeric_value(idx)
                    
                    if idx_value is None:
                        self.errors.append(SemanticError(
                            f"Invalid qubit index: could not extract numeric value from {idx}",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                        
                    idx_value = int(idx_value)
                    
                    if idx_value < 0 or idx_value >= num_qubits:
                        self.errors.append(SemanticError(
                            f"Qubit index {idx_value} out of range [0, {num_qubits-1}] for state '{state_name}'",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                    
                    # Check for duplicates manually
                    if idx_value in seen_indices:
                        has_duplicates = True
                    else:
                        seen_indices.append(idx_value)
                
                # Check for duplicate indices
                if has_duplicates:
                    self.errors.append(SemanticError(
                        f"Duplicate qubit indices in qubit specification for state '{state_name}'",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
        elif qubit_spec.kind == "range":
            # Range of qubit indices (start, end, step)
            if isinstance(qubit_spec.value, tuple) and len(qubit_spec.value) >= 2:
                start_val = self._extract_numeric_value(qubit_spec.value[0])
                end_val = self._extract_numeric_value(qubit_spec.value[1])
                
                if start_val is None or end_val is None:
                    self.errors.append(SemanticError(
                        f"Invalid range values in qubit specification",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                
                start = int(start_val)
                end = int(end_val)
                
                if start < 0 or start >= num_qubits:
                    self.errors.append(SemanticError(
                        f"Range start {start} out of range [0, {num_qubits-1}] for state '{state_name}'",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                if end < 0 or end >= num_qubits:
                    self.errors.append(SemanticError(
                        f"Range end {end} out of range [0, {num_qubits-1}] for state '{state_name}'",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                if start > end:
                    self.errors.append(SemanticError(
                        f"Invalid range: start {start} greater than end {end}",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                # Check step if provided
                if len(qubit_spec.value) == 3 and qubit_spec.value[2] is not None:
                    step_val = self._extract_numeric_value(qubit_spec.value[2])
                    
                    if step_val is None:
                        self.errors.append(SemanticError(
                            f"Invalid range step value",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                    
                    step = int(step_val)
                    if step <= 0:
                        self.errors.append(SemanticError(
                            f"Invalid range step {step}: must be a positive integer",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                        
        elif qubit_spec.kind == "all":
            # All qubits - always valid
            pass
            
        else:
            self.errors.append(SemanticError(
                f"Invalid qubit specification kind '{qubit_spec.kind}'. Valid kinds: 'single', 'list', 'range', 'all'",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        return True

    def _validate_control_specification(self, control_spec: ControlSpec, 
                                      state_name: str, location: Tuple[int, int, int]) -> bool:
        """
        Validate a control specification for compatibility with a quantum state.
        
        Verifies control qubit indices are within range and follow proper format.
        
        Args:
            control_spec: Control specification to validate
            state_name: Name of the target state
            location: Source location for error reporting
            
        Returns:
            bool: True if the control specification is valid
        """
        state_def = self.symbol_table.get_state(state_name)
        if state_def is None:
            # State already validated, so this shouldn't happen
            return False
            
        num_qubits = self._get_state_num_qubits(state_def)
                    
        if control_spec.kind == "single":
            # Single control qubit
            control_value = self._extract_numeric_value(control_spec.value)
            
            if control_value is None:
                self.errors.append(SemanticError(
                    f"Invalid control qubit index: could not extract numeric value from {control_spec.value}",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            control_value = int(control_value)
            if control_value < 0 or control_value >= num_qubits:
                self.errors.append(SemanticError(
                    f"Control qubit index {control_value} out of range [0, {num_qubits-1}] for state '{state_name}'",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                    
        elif control_spec.kind == "list":
            # List of control qubits
            if isinstance(control_spec.value, list):
                seen_indices = []
                for idx in control_spec.value:
                    idx_value = self._extract_numeric_value(idx)
                    
                    if idx_value is None:
                        self.errors.append(SemanticError(
                            f"Invalid control qubit index: could not extract numeric value from {idx}",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                        
                    idx_value = int(idx_value)
                    if idx_value < 0 or idx_value >= num_qubits:
                        self.errors.append(SemanticError(
                            f"Control qubit index {idx_value} out of range [0, {num_qubits-1}] for state '{state_name}'",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                    
                    if idx_value in seen_indices:
                        self.errors.append(SemanticError(
                            f"Duplicate control qubit indices for state '{state_name}'",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                    
                    seen_indices.append(idx_value)
                    
        elif control_spec.kind == "range":
            # Range of control qubits
            if isinstance(control_spec.value, tuple) and len(control_spec.value) >= 2:
                start_val = self._extract_numeric_value(control_spec.value[0])
                end_val = self._extract_numeric_value(control_spec.value[1])
                
                if start_val is None or end_val is None:
                    self.errors.append(SemanticError(
                        f"Invalid control range values",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                
                start = int(start_val)
                end = int(end_val)
                
                if start < 0 or start >= num_qubits:
                    self.errors.append(SemanticError(
                        f"Control range start {start} out of range [0, {num_qubits-1}] for state '{state_name}'",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                if end < 0 or end >= num_qubits:
                    self.errors.append(SemanticError(
                        f"Control range end {end} out of range [0, {num_qubits-1}] for state '{state_name}'",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                if start > end:
                    self.errors.append(SemanticError(
                        f"Invalid control range: start {start} greater than end {end}",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                # Check step if provided
                if len(control_spec.value) == 3 and control_spec.value[2] is not None:
                    step_val = self._extract_numeric_value(control_spec.value[2])
                    
                    if step_val is None or int(step_val) <= 0:
                        self.errors.append(SemanticError(
                            f"Invalid control range step: must be positive",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                        
        elif control_spec.kind == "anti":
            # Anti-control (control on |0⟩ instead of |1⟩)
            control_value = self._extract_numeric_value(control_spec.value)
            
            if control_value is None:
                self.errors.append(SemanticError(
                    f"Invalid anti-control qubit index: could not extract numeric value from {control_spec.value}",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            control_value = int(control_value)
            if control_value < 0 or control_value >= num_qubits:
                self.errors.append(SemanticError(
                    f"Anti-control qubit index {control_value} out of range [0, {num_qubits-1}] for state '{state_name}'",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                    
        else:
            self.errors.append(SemanticError(
                f"Invalid control specification kind '{control_spec.kind}'. Valid kinds: 'single', 'list', 'range', 'anti'",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        return True
    
    def _validate_params_specification(self, params_spec: ParamsSpec, 
                                      gate_type: str, location: Tuple[int, int, int]) -> bool:
        """
        Validate a parameters specification for a quantum gate.
        
        Verifies that parameters match the expected number and types for a given gate.
        
        Args:
            params_spec: Parameters specification to validate
            gate_type: Type of the gate these parameters apply to
            location: Source location for error reporting
            
        Returns:
            bool: True if the parameters specification is valid
        """
        # Check if gate requires parameters
        parameterized_gates = {
            "RX_gate", "RY_gate", "RZ_gate", "P_gate", "PhaseShift_gate",
            "U_gate", "U1_gate", "U2_gate", "U3_gate",
            "ControlledPhaseShift_gate", "VQE_gate", "QAOA_gate",
            "Trotter_gate", "RandomUnitary_gate", "Ising_gate", "Heisenberg_gate", 
            "FermiHubbard_gate"
        }
        
        if gate_type not in parameterized_gates:
            self.errors.append(SemanticError(
                f"Gate '{gate_type}' does not accept parameters. Parameterized gates include: {', '.join(sorted(list(parameterized_gates))[:5])}...",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        # Validate parameter count
        min_params, max_params = self._get_gate_parameter_count(gate_type)
        
        if min_params is not None and len(params_spec.expressions) < min_params:
            self.errors.append(SemanticError(
                f"Gate '{gate_type}' requires at least {min_params} parameter(s), got {len(params_spec.expressions)}",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        if max_params is not None and len(params_spec.expressions) > max_params:
            self.errors.append(SemanticError(
                f"Gate '{gate_type}' accepts at most {max_params} parameter(s), got {len(params_spec.expressions)}",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        # Validate parameter types
        for i, expr in enumerate(params_spec.expressions):
            param_type = self._get_expression_type(expr)
            if param_type is None or param_type.name not in ["number_type", "complex_type"]:
                self.errors.append(SemanticError(
                    f"Gate parameter must be a number or complex value, got '{param_type.name if param_type else 'unknown'}'",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
        return True
    
    def _validate_gate_compatibility(self, gate_type: str, qubit_spec: Optional[QubitSpec],
                                   control_spec: Optional[ControlSpec], 
                                   params_spec: Optional[ParamsSpec],
                                   location: Tuple[int, int, int]) -> bool:
        """
        Validate compatibility of gate type with specifications.
        
        Ensures the gate is used with appropriate qubits, controls and parameters,
        following both quantum circuit rules and OSH coherence principles.
        
        Args:
            gate_type: Type of the gate
            qubit_spec: Specification of target qubits
            control_spec: Specification of control qubits
            params_spec: Specification of gate parameters
            location: Source location for error reporting
            
        Returns:
            bool: True if the gate is compatible with the specifications
        """
        # Define gate categories - sync with parser GATE_TYPES and _validate_multi_target_gate
        single_qubit_gates = {
            "H_gate", "X_gate", "Y_gate", "Z_gate", "S_gate", "T_gate", "P_gate", "I_gate",
            "Hadamard_gate", "PauliX_gate", "PauliY_gate", "PauliZ_gate", "PhaseShift_gate",
            "SqrtX_gate", "SqrtY_gate", "SqrtZ_gate", "SqrtW_gate", "SqrtNOT_gate",
            "PHASE_gate", "phase_gate",
            "RX_gate", "RY_gate", "RZ_gate", "U_gate", "U1_gate", "U2_gate", "U3_gate",
            "H", "X", "Y", "Z", "S", "T", "P", "I", "PHASE"
        }
        
        two_qubit_gates = {
            "CNOT_gate", "CX_gate", "CZ_gate", "SWAP_gate"
        }
        
        three_qubit_gates = {
            "TOFFOLI_gate", "CCNOT_gate", "CSWAP_gate"
        }
        
        multi_qubit_gates = {
            "QFT_gate", "InverseQFT_gate", "U_gate", "U2_gate", "U3_gate",
            "Oracle_gate", "Grover_gate", "Shor_gate", "VQE_gate", "QAOA_gate",
            "Trotter_gate", "RandomUnitary_gate", "Ising_gate", "Heisenberg_gate", 
            "FermiHubbard_gate"
        }
        
        # Validate gate category constraints
        if gate_type in single_qubit_gates:
            # Single-qubit gates should target exactly one qubit
            if qubit_spec is None:
                pass  # Valid - no explicit qubit specification means operate on the entire state
            elif qubit_spec and qubit_spec.kind == "single":
                pass  # Valid
            elif qubit_spec and qubit_spec.kind == "list" and isinstance(qubit_spec.value, list) and len(qubit_spec.value) == 1:
                pass  # Valid
            else:
                self.errors.append(SemanticError(
                    f"Gate '{gate_type}' requires exactly one target qubit",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            # And should not have control qubits (unless it's a controlled version)
            if control_spec and not gate_type.startswith("C"):
                self.errors.append(SemanticError(
                    f"Gate '{gate_type}' does not accept control qubits",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
        elif gate_type in two_qubit_gates:
            if gate_type in ["CNOT_gate", "CX_gate", "CZ_gate"]:
                # For multi-target gates parsed as target lists, skip detailed validation here
                # The _validate_multi_target_gate method will handle the proper validation
                if qubit_spec is None and control_spec is None:
                    pass  # Multi-target syntax, will be validated by _validate_multi_target_gate
                # Controlled gates need one target and one control (explicit syntax)
                elif qubit_spec and qubit_spec.kind == "single":
                    pass  # Valid target
                elif qubit_spec and qubit_spec.kind == "list" and isinstance(qubit_spec.value, list) and len(qubit_spec.value) == 1:
                    pass  # Valid target
                else:
                    self.errors.append(SemanticError(
                        f"Gate '{gate_type}' requires exactly one target qubit",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                # Only validate control spec if we're not using multi-target syntax
                if qubit_spec is not None or control_spec is not None:
                    if control_spec and control_spec.kind == "single":
                        pass  # Valid control
                    elif control_spec and control_spec.kind == "list" and isinstance(control_spec.value, list) and len(control_spec.value) == 1:
                        pass  # Valid control
                    else:
                        self.errors.append(SemanticError(
                            f"Gate '{gate_type}' requires exactly one control qubit",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                    
            elif gate_type == "SWAP_gate":
                # For multi-target gates parsed as target lists, skip detailed validation here
                if qubit_spec is None and control_spec is None:
                    pass  # Multi-target syntax, will be validated by _validate_multi_target_gate
                # SWAP needs two targets and no controls (explicit syntax)
                elif qubit_spec and qubit_spec.kind == "list" and isinstance(qubit_spec.value, list) and len(qubit_spec.value) == 2:
                    pass  # Valid
                else:
                    self.errors.append(SemanticError(
                        f"Gate '{gate_type}' requires exactly two target qubits",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                if control_spec:
                    self.errors.append(SemanticError(
                        f"Gate '{gate_type}' does not accept control qubits",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
        elif gate_type in three_qubit_gates:
            if gate_type in ["TOFFOLI_gate", "CCNOT_gate"]:
                # For multi-target gates parsed as target lists, skip detailed validation here
                if qubit_spec is None and control_spec is None:
                    pass  # Multi-target syntax, will be validated by _validate_multi_target_gate
                # Toffoli/CCNOT needs one target and two controls (explicit syntax)
                elif qubit_spec and qubit_spec.kind == "single":
                    pass  # Valid target
                elif qubit_spec and qubit_spec.kind == "list" and isinstance(qubit_spec.value, list) and len(qubit_spec.value) == 1:
                    pass  # Valid target
                else:
                    self.errors.append(SemanticError(
                        f"Gate '{gate_type}' requires exactly one target qubit",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                # Only validate control spec if we're not using multi-target syntax
                if qubit_spec is not None or control_spec is not None:
                    if control_spec and control_spec.kind == "list" and isinstance(control_spec.value, list) and len(control_spec.value) == 2:
                        pass  # Valid controls
                    else:
                        self.errors.append(SemanticError(
                            f"Gate '{gate_type}' requires exactly two control qubits",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
                    
            elif gate_type == "CSWAP_gate":
                # For multi-target gates parsed as target lists, skip detailed validation here
                if qubit_spec is None and control_spec is None:
                    pass  # Multi-target syntax, will be validated by _validate_multi_target_gate
                # CSWAP needs two targets and one control (explicit syntax)
                elif qubit_spec and qubit_spec.kind == "list" and isinstance(qubit_spec.value, list) and len(qubit_spec.value) == 2:
                    pass  # Valid targets
                else:
                    self.errors.append(SemanticError(
                        f"Gate '{gate_type}' requires exactly two target qubits",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                    
                # Only validate control spec if we're not using multi-target syntax
                if qubit_spec is not None or control_spec is not None:
                    if control_spec and control_spec.kind == "single":
                        pass  # Valid control
                    elif control_spec and control_spec.kind == "list" and isinstance(control_spec.value, list) and len(control_spec.value) == 1:
                        pass  # Valid control
                    else:
                        self.errors.append(SemanticError(
                            f"Gate '{gate_type}' requires exactly one control qubit",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
        
        elif gate_type in multi_qubit_gates:
            # Multi-qubit gates like QFT need at least one qubit specified
            if not qubit_spec:
                self.errors.append(SemanticError(
                    f"Gate '{gate_type}' requires at least one target qubit to be specified",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            # QFT, InverseQFT need consecutive qubits
            if gate_type in ["QFT_gate", "InverseQFT_gate"]:
                if qubit_spec.kind == "range":
                    # Range is good for consecutive qubits
                    pass
                elif qubit_spec.kind == "list" and isinstance(qubit_spec.value, list):
                    # Check if the list contains consecutive qubits
                    qubit_indices = []
                    for val in qubit_spec.value:
                        numeric_val = self._extract_numeric_value(val)
                        if numeric_val is not None:
                            qubit_indices.append(int(numeric_val))
                    
                    sorted_qubits = sorted(qubit_indices)
                    if sorted_qubits != list(range(min(sorted_qubits), max(sorted_qubits) + 1)):
                        self.errors.append(SemanticError(
                            f"Gate '{gate_type}' requires consecutive qubits",
                            "current_file",
                            location[0],
                            location[1],
                            "quantum"
                        ))
                        return False
            
            # Add OSH-specific gate validation for recursive gates
            if gate_type in ["Trotter_gate", "Ising_gate", "Heisenberg_gate"]:
                # These gates have special OSH coherence implications
                if not params_spec or len(params_spec.expressions) < 1:
                    self.errors.append(SemanticError(
                        f"OSH-aligned gate '{gate_type}' requires at least one parameter for coherence management",
                        "current_file",
                        location[0],
                        location[1],
                        "quantum"
                    ))
                    return False
                        
        # Algorithm gates like Grover, Shor, VQE might have special requirements
        if gate_type in ["Grover_gate", "Shor_gate"]:
            # These usually need more qubits for meaningful operation
            if qubit_spec and qubit_spec.kind == "list" and isinstance(qubit_spec.value, list) and len(qubit_spec.value) < 3:
                logger.warning(
                    f"OSH warning: Gate '{gate_type}' typically requires more qubits for meaningful operation"
                )
        
        # For all gate types, check that control and target qubits don't overlap
        if qubit_spec and control_spec:
            qubit_indices = self._get_qubit_indices_from_spec(qubit_spec, 100)  # Use a large number since we already validated ranges
            control_indices = self._get_qubit_indices_from_spec(control_spec, 100)
            
            overlap = set(qubit_indices).intersection(set(control_indices))
            if overlap:
                self.errors.append(SemanticError(
                    f"Target and control qubits cannot overlap (indices {sorted(list(overlap))})",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
        return True
    
    def _resolve_expression_to_state(self, expr: Expression) -> Optional[str]:
        """
        Try to resolve an expression to a quantum state name.
        
        For expressions that might reference quantum states indirectly,
        this function attempts to resolve them to a concrete state name.
        
        Args:
            expr: The expression to resolve
            
        Returns:
            Optional[str]: The state name if resolved, None otherwise
        """
        # Check if expression is a direct reference with a name attribute
        if hasattr(expr, 'name'):
            # Direct variable reference
            var = self.symbol_table.get_variable(expr.name)
            if var is None:
                # Check if it's a direct state reference
                state = self.symbol_table.get_state(expr.name)
                if state is not None:
                    return expr.name
                return None
                
            # Check if the variable is a state reference
            var_type = var.var_type
            if var_type and var_type.name in self.quantum_state_types:
                # It's a state variable
                return expr.name
            elif var_type and var_type.name == "string_type":
                # It might be a string variable containing a state name
                return None  # Can't resolve string variables statically
                
        elif isinstance(expr, FunctionCallExpression):
            # Function call that might return a state
            function_def = self.symbol_table.get_function(expr.function)
            if function_def is None:
                return None
                
            # Check the return type of the function
            return_type = function_def.return_type
            if return_type and return_type.name in self.quantum_state_types:
                # Function returns a state, but we can't determine which one statically
                return None
                
        elif isinstance(expr, QuantumExpression):
            # Quantum expressions might return state references
            if expr.kind in ["ket", "bra"] and len(expr.args) > 0:
                # Try to resolve the argument
                return self._resolve_expression_to_state(expr.args[0])
                
        return None

    def _validate_quantum_state_expression(self, expr: Expression, location: Tuple[int, int, int]) -> bool:
        """
        Validate if an expression refers to a quantum state.
        
        Verifies that an expression can be resolved to a valid quantum state
        either directly or indirectly.
        
        Args:
            expr: The expression to validate
            location: Source location for error reporting
            
        Returns:
            bool: True if the expression refers to a valid quantum state
        """
        # Direct identifier - must be a state or a variable referencing a state
        if hasattr(expr, 'name'):
            # Check if it's a declared state
            state = self.symbol_table.get_state(expr.name)
            if state is not None:
                return True
                
            # Check if it's a variable that might refer to a state
            var = self.symbol_table.get_variable(expr.name)
            if var is not None:
                var_type = var.var_type
                if var_type and var_type.name in self.quantum_state_types:
                    return True
                    
            # Neither a state nor a suitable variable
            self.errors.append(SemanticError(
                f"Expression '{expr.name}' does not refer to a quantum state",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        # Function call - check if the function returns a quantum state
        elif isinstance(expr, FunctionCallExpression):
            function_def = self.symbol_table.get_function(expr.function)
            if function_def is None:
                self.errors.append(SemanticError(
                    f"Undefined function '{expr.function}'",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            # Check the return type
            return_type = function_def.return_type
            if return_type and return_type.name in self.quantum_state_types:
                return True
                
            self.errors.append(SemanticError(
                f"Function '{expr.function}' does not return a quantum state",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        # Quantum expressions may produce quantum states
        elif isinstance(expr, QuantumExpression):
            if expr.kind in ["ket", "schmidt_decomposition"]:
                return True
                
            self.errors.append(SemanticError(
                f"Quantum expression '{expr.kind}' does not produce a quantum state reference",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        # Other expressions - not valid quantum state references
        self.errors.append(SemanticError(
            f"Expression does not refer to a quantum state",
            "current_file",
            location[0],
            location[1],
            "quantum"
        ))
        return False

    def _get_expression_type(self, expr: Expression) -> Optional[TypeAnnotation]:
        """
        Get the type of an expression through inference.
        
        Enhanced to better handle quantum-specific expressions including
        superpositions, states, and quantum operators.
        
        Args:
            expr: The expression to analyze
            
        Returns:
            Optional[TypeAnnotation]: The type annotation, or None if not determinable
        """
        try:
            from src.core.type_checker import TypeChecker  # Import at function level to avoid circular imports
            
            # Initialize type checker with our symbol table
            type_checker = TypeChecker(self.symbol_table)
            
            # Special handling for quantum expressions
            if isinstance(expr, QuantumExpression):
                # Return appropriate type based on quantum expression kind
                if expr.kind in ["bra", "ket"]:
                    return TypeAnnotation(name="state_vector_type")
                elif expr.kind in ["braket", "expectation", "fidelity"]:
                    return TypeAnnotation(name="complex_type")
                elif expr.kind in ["trace", "entropy", "purity"]:
                    return TypeAnnotation(name="number_type")
                elif expr.kind in ["eigenvalues"]:
                    return TypeAnnotation(name="vector_type")
                elif expr.kind in ["eigenvectors", "schmidt_decomposition"]:
                    return TypeAnnotation(name="matrix_type")
                elif expr.kind in ["tensor_product"]:
                    return TypeAnnotation(name="tensor_type")
                elif expr.kind in ["partial_trace"]:
                    return TypeAnnotation(name="density_matrix_type")
            
            # Use the proper type checker to infer expression type
            inferred_type = type_checker.get_expression_type(expr)
            
            # Log warning if type cannot be determined
            if inferred_type is None:
                logger.warning(f"Could not determine type for expression at {expr.location}")
                
            return inferred_type
        except ImportError:
            # If type checker is not available, provide basic type inference
            if isinstance(expr, NumberLiteral):
                return TypeAnnotation(name="number_type")
            elif hasattr(expr, 'value') and isinstance(expr.value, str):
                return TypeAnnotation(name="string_type")
            else:
                return None
    
    def _get_qubit_indices_from_spec(self, qubit_spec: Union[QubitSpec, ControlSpec], num_qubits: int) -> List[int]:
        """
        Extract the list of qubit indices from a qubit specification.
        
        Converts any qubit specification format to a concrete list of indices.
        
        Args:
            qubit_spec: The qubit specification
            num_qubits: The total number of qubits available
            
        Returns:
            List[int]: List of qubit indices
        """
        if qubit_spec.kind == "single":
            qubit_value = self._extract_numeric_value(qubit_spec.value)
            if qubit_value is not None:
                return [int(qubit_value)]
            else:
                return []
                
        elif qubit_spec.kind == "list":
            if isinstance(qubit_spec.value, list):
                indices = []
                for idx in qubit_spec.value:
                    idx_value = self._extract_numeric_value(idx)
                    if idx_value is not None:
                        indices.append(int(idx_value))
                return indices
            else:
                return []
                
        elif qubit_spec.kind == "range":
            if isinstance(qubit_spec.value, tuple) and len(qubit_spec.value) >= 2:
                start_val = self._extract_numeric_value(qubit_spec.value[0])
                end_val = self._extract_numeric_value(qubit_spec.value[1])
                step_val = 1
                
                if len(qubit_spec.value) == 3 and qubit_spec.value[2] is not None:
                    step_val = self._extract_numeric_value(qubit_spec.value[2])
                    
                if start_val is not None and end_val is not None and step_val is not None:
                    return list(range(int(start_val), int(end_val) + 1, int(step_val)))
                else:
                    return []
            else:
                return []
                
        elif qubit_spec.kind == "all":
            return list(range(num_qubits))
            
        elif qubit_spec.kind == "anti":  # For control specs
            qubit_value = self._extract_numeric_value(qubit_spec.value)
            if qubit_value is not None:
                return [int(qubit_value)]
            else:
                return []
                
        return []

    def _get_gate_parameter_count(self, gate_type: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get the expected number of parameters for a gate type.
        
        Returns the minimum and maximum number of parameters expected for a gate.
        
        Args:
            gate_type: The gate type to check
            
        Returns:
            Tuple[Optional[int], Optional[int]]: (min_params, max_params) - minimum and maximum allowed parameters
        """
        # Single-parameter gates
        if gate_type in ["RX_gate", "RY_gate", "RZ_gate", "P_gate", "PhaseShift_gate", 
                        "U1_gate", "ControlledPhaseShift_gate"]:
            return (1, 1)
            
        # Two-parameter gates
        elif gate_type in ["U2_gate"]:
            return (2, 2)
            
        # Three-parameter gates
        elif gate_type in ["U3_gate", "U_gate"]:
            return (3, 3)
            
        # Variable parameter gates with minimum requirements
        elif gate_type in ["VQE_gate", "QAOA_gate"]:
            return (1, None)  # At least 1 parameter, no maximum
            
        elif gate_type in ["Trotter_gate"]:
            return (2, None)  # At least 2 parameters (time step, iterations)
            
        elif gate_type in ["Ising_gate", "Heisenberg_gate", "FermiHubbard_gate"]:
            return (1, None)  # At least 1 parameter (coupling strength)
            
        elif gate_type in ["RandomUnitary_gate"]:
            return (0, 1)  # 0 or 1 parameter (seed)
            
        # Gates that don't accept parameters
        else:
            return (0, 0)
        
    def _validate_quantum_expression(self, expr: QuantumExpression, location: Tuple[int, int, int]) -> bool:
        """
        Validate a quantum expression (bra, ket, braket, etc.).
        
        Verifies that quantum expression formats, arguments, and types are valid.
        
        Args:
            expr: The quantum expression to validate
            location: Source location for error reporting
            
        Returns:
            bool: True if the quantum expression is valid
        """
        valid_kinds = [
            "bra", "ket", "braket", "expectation", "tensor_product", 
            "trace", "partial_trace", "fidelity", "entropy",
            "purity", "schmidt_decomposition", "eigenvalues", "eigenvectors"
        ]
        
        if expr.kind not in valid_kinds:
            self.errors.append(SemanticError(
                f"Unknown quantum expression kind '{expr.kind}'. Valid kinds: {', '.join(valid_kinds)}",
                "current_file",
                location[0], 
                location[1],
                "quantum"
            ))
            return False
            
        # Validate arguments based on expression kind
        if expr.kind == "bra" or expr.kind == "ket":
            # Single argument expressions
            if len(expr.args) != 1:
                self.errors.append(SemanticError(
                    f"{expr.kind} expression requires exactly one argument",
                    "current_file", 
                    location[0],
                    location[1], 
                    "quantum"
                ))
                return False
                
            # Verify the argument refers to a quantum state
            return self._validate_quantum_state_expression(expr.args[0], location)
            
        elif expr.kind in ["braket", "expectation", "tensor_product", "fidelity", "partial_trace"]:
            # Two argument expressions
            if len(expr.args) != 2:
                self.errors.append(SemanticError(
                    f"{expr.kind} expression requires exactly two arguments",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            # Both arguments must be quantum states or operators
            valid_arg1 = self._validate_quantum_state_expression(expr.args[0], location)
            valid_arg2 = self._validate_quantum_state_expression(expr.args[1], location)
            return valid_arg1 and valid_arg2
            
        elif expr.kind in ["trace", "entropy", "purity", "eigenvalues", "eigenvectors", "schmidt_decomposition"]:
            # Single argument expressions that operate on matrices
            if len(expr.args) != 1:
                self.errors.append(SemanticError(
                    f"{expr.kind} expression requires exactly one argument",
                    "current_file",
                    location[0],
                    location[1],
                    "quantum"
                ))
                return False
                
            # The argument must be a quantum state or operator
            return self._validate_quantum_state_expression(expr.args[0], location)
            
        return True
    
    def _validate_multi_target_gate(self, gate_type: str, targets: List[str], location: Tuple[int, int, int]) -> bool:
        """
        Validate that multi-target gates have the correct number of targets.
        
        Args:
            gate_type: Type of the gate
            targets: List of target states
            location: Source location for error reporting
            
        Returns:
            bool: True if the gate has the correct number of targets
        """
        # Define required number of targets for each gate type
        single_target_gates = {
            "H_gate", "X_gate", "Y_gate", "Z_gate", "S_gate", "T_gate", "P_gate", "I_gate",
            "Hadamard_gate", "PauliX_gate", "PauliY_gate", "PauliZ_gate", "PhaseShift_gate",
            "SqrtX_gate", "SqrtY_gate", "SqrtZ_gate", "SqrtW_gate", "SqrtNOT_gate",
            "PHASE_gate", "phase_gate",
            "RX_gate", "RY_gate", "RZ_gate", "U_gate", "U1_gate", "U2_gate", "U3_gate",
            "H", "X", "Y", "Z", "S", "T", "P", "I", "PHASE"
        }
        
        two_target_gates = {
            "CNOT_gate", "CX_gate", "CZ_gate", "SWAP_gate", "ControlledPhaseShift_gate",
            "ControlledZ_gate", "AdjacentControlledPhaseShift_gate",
            "CNOT", "CX", "CZ", "SWAP"
        }
        
        three_target_gates = {
            "TOFFOLI_gate", "CCNOT_gate", "CSWAP_gate", "ControlledSWAP_gate",
            "TOFFOLI", "CCNOT", "CSWAP"
        }
        
        num_targets = len(targets)
        
        if gate_type in single_target_gates and num_targets != 1:
            self.errors.append(SemanticError(
                f"Gate '{gate_type}' requires exactly one target qubit",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
        elif gate_type in two_target_gates and num_targets != 2:
            self.errors.append(SemanticError(
                f"Gate '{gate_type}' requires exactly two target qubits",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
        elif gate_type in three_target_gates and num_targets != 3:
            self.errors.append(SemanticError(
                f"Gate '{gate_type}' requires exactly three target qubits",
                "current_file",
                location[0],
                location[1],
                "quantum"
            ))
            return False
            
        return True