from typing import Any, Dict, List, Optional, Set, Tuple, Union
import time
import logging
from datetime import datetime

import numpy as np

from src.physics.coherence import CoherenceManager


logger = logging.getLogger(__name__)

class StateRegistry:
    """
    Registry for quantum states in the Recursia runtime
    
    Manages the lifecycle, properties, and relationships of quantum states
    in the system, providing a centralized repository for state information 
    and facilitating operations on states across the runtime.
    """
    
    def __init__(self):
        """Initialize the state registry"""
        self.states: Dict[str, Dict[str, Any]] = {}
        self.state_types: Dict[str, Dict[str, Any]] = {}
        self.state_fields: Dict[str, Dict[str, Any]] = {}
        self.state_vectors: Dict[str, np.ndarray] = {}
        self.density_matrices: Dict[str, np.ndarray] = {}
        self.entanglement_relationships: Dict[Tuple[str, str], float] = {}
        self.memory_manager = None
        self.recursive_mechanics = None
        self.coherence_manager = CoherenceManager()
        self.event_system = None
        
        # Register default state types
        self._register_default_state_types()
        
        # Performance tracking
        self.stats = {
            'states_created': 0,
            'states_deleted': 0,
            'fields_set': 0,
            'fields_get': 0,
            'entanglements_created': 0,
            'entanglements_broken': 0
        }
    
    def _register_default_state_types(self):
        """Register the default state types supported by Recursia"""
        default_types = {
            'quantum_type': {
                'description': 'Standard quantum state with quantum properties',
                'default_fields': {
                    'coherence': 1.0,
                    'entropy': 0.0,
                    'is_entangled': False,
                    'entangled_with': set(),
                    'measurement_basis': 'computational_basis'
                }
            },
            'entity_type': {
                'description': 'Entity capable of complex behavior and interactions',
                'default_fields': {
                    'coherence': 0.8,
                    'entropy': 0.2,
                    'is_entangled': False,
                    'entangled_with': set(),
                    'behavior_pattern': 'standard',
                    'interaction_mode': 'reactive'
                }
            },
            'superposition_type': {
                'description': 'Quantum state optimized for superposition properties',
                'default_fields': {
                    'coherence': 1.0,
                    'entropy': 0.0,
                    'is_entangled': False,
                    'superposition_complexity': 1.0,
                    'phase_stability': 0.9
                }
            },
            'entangled_type': {
                'description': 'Quantum state optimized for entanglement properties',
                'default_fields': {
                    'coherence': 0.9,
                    'entropy': 0.1,
                    'is_entangled': False,
                    'entanglement_capacity': 1.0,
                    'entanglement_fidelity': 0.95
                }
            },
            'field_type': {
                'description': 'Quantum field with spatial properties',
                'default_fields': {
                    'coherence': 0.7,
                    'entropy': 0.3,
                    'is_entangled': False,
                    'field_strength': 1.0,
                    'field_symmetry': 'isotropic',
                    'field_dimensions': [10, 10, 10]
                }
            },
            'mixed_type': {
                'description': 'Mixed quantum state with classical and quantum properties',
                'default_fields': {
                    'coherence': 0.5,
                    'entropy': 0.5,
                    'is_entangled': False,
                    'classical_probability': 0.5,
                    'quantum_probability': 0.5
                }
            },
            'measurement_type': {
                'description': 'State representing measurement outcomes',
                'default_fields': {
                    'coherence': 0.0,
                    'entropy': 1.0,
                    'is_entangled': False,
                    'outcome_probability': {},
                    'most_recent_outcome': None,
                    'measurement_count': 0
                }
            }
        }
        
        for type_name, type_info in default_types.items():
            self.state_types[type_name] = type_info
    
    def create_state(self, name: str, state_type: str = 'quantum_type', 
                    num_qubits: Optional[int] = None, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new quantum state
        
        Args:
            name (str): State name
            state_type (str): State type from registered types
            num_qubits (int, optional): Number of qubits for the state
            properties (dict, optional): Additional properties to set
            
        Returns:
            bool: True if state was created successfully
        """
        if name in self.states:
            logger.warning(f"State '{name}' already exists, cannot create")
            return False
        
        # Check if state type exists
        if state_type not in self.state_types:
            logger.warning(f"Unknown state type '{state_type}', defaulting to 'quantum_type'")
            state_type = 'quantum_type'
        
        # Create state record
        created_at = time.time()
        self.states[name] = {
            'name': name,
            'type': state_type,
            'num_qubits': num_qubits,
            'created_at': created_at,
            'created_datetime': datetime.fromtimestamp(created_at),
            'version': 1,
            'modified_at': created_at,
            'modified_datetime': datetime.fromtimestamp(created_at),
            'id': len(self.states) + 1,  # Assign unique ID
            # OSH default values for validation
            'initial_coherence': 0.95,
            'initial_entropy': 0.05,
            'coherence': 0.95,  # Current coherence
            'entropy': 0.05     # Current entropy
        }
        
        # Initialize fields with defaults from the state type
        self.state_fields[name] = {}
        type_defaults = self.state_types[state_type].get('default_fields', {})
        for field_name, default_value in type_defaults.items():
            self.state_fields[name][field_name] = default_value
        
        # Apply any provided properties
        if properties:
            for field_name, value in properties.items():
                self.state_fields[name][field_name] = value
                
                # Handle alternative naming conventions - override defaults if provided
                if field_name == 'state_coherence':
                    self.state_fields[name]['coherence'] = value
                elif field_name == 'state_entropy':
                    self.state_fields[name]['entropy'] = value
                elif field_name == 'decoherence_time':
                    # Store decoherence time in milliseconds
                    self.state_fields[name]['decoherence_time'] = value
        
        # Calculate OSH decoherence time if not provided
        if 'decoherence_time' not in self.state_fields[name]:
            # OSH predicts 1-5ms decoherence times
            # Default to 3ms (middle of range)
            coherence = self.state_fields[name].get('coherence', 0.95)
            num_qubits_factor = 1.0 / np.sqrt(num_qubits if num_qubits else 1)
            coherence_factor = coherence ** 2
            # Base time is 3ms, modified by system parameters
            decoherence_time_ms = 3.0 * num_qubits_factor * coherence_factor
            # Clamp to OSH range [1ms, 5ms]
            decoherence_time_ms = max(1.0, min(5.0, decoherence_time_ms))
            self.state_fields[name]['decoherence_time'] = decoherence_time_ms / 1000.0  # Store in seconds
        
        # Create state vector if number of qubits is specified
        if num_qubits is not None and num_qubits > 0:
            # Initialize to |0...0⟩ state
            state_vector_size = 2 ** num_qubits
            state_vector = np.zeros(state_vector_size, dtype=complex)
            state_vector[0] = 1.0  # |0...0⟩ state
            self.state_vectors[name] = state_vector
            
            # Also initialize density matrix
            density_matrix = np.zeros((state_vector_size, state_vector_size), dtype=complex)
            density_matrix[0, 0] = 1.0  # |0...0⟩⟨0...0| density matrix
            self.density_matrices[name] = density_matrix
            
            # Allocate memory if memory manager is set
            if self.memory_manager:
                try:
                    memory_id = self.memory_manager.allocate_quantum_state(num_qubits, name)
                    self.states[name]['memory_id'] = memory_id
                except Exception as e:
                    logger.error(f"Failed to allocate memory for state '{name}': {e}")
        
        # Register with recursive mechanics if available
        if self.recursive_mechanics:
            parent_state = None
            level = 0
            # Check if properties specify a parent state or recursion level
            if properties:
                parent_state = properties.get('parent_state')
                level = properties.get('recursion_level', 0)
            self.recursive_mechanics.register_system(name, level, parent_state)
        
        # Emit creation event if event system is available
        if self.event_system:
            self.event_system.emit_quantum_event(
                'state_creation_event',
                name,
                {
                    'state_type': state_type,
                    'num_qubits': num_qubits,
                    'properties': properties
                }
            )
        
        # Update stats
        self.stats['states_created'] += 1
        
        logger.info(f"Created state '{name}' of type '{state_type}' with {num_qubits} qubits - Total states: {len(self.states)}")
        return True
    
    def get_state(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a state by name
        
        Args:
            name (str): State name
            
        Returns:
            dict: State information, or None if not found
        """
        state = self.states.get(name)
        if not state:
            return None
        
        # Create a copy with fields
        result = state.copy()
        fields = self.state_fields.get(name, {}).copy()
        result['fields'] = fields
        
        # Also expose key fields directly for API compatibility
        # These are the fields the API server expects at the top level
        if 'coherence' in fields:
            result['coherence'] = fields['coherence']
        if 'state_coherence' in fields:
            result['state_coherence'] = fields['state_coherence']
        if 'entropy' in fields:
            result['entropy'] = fields['entropy'] 
        if 'state_entropy' in fields:
            result['state_entropy'] = fields['state_entropy']
        if 'state_qubits' in fields:
            result['state_qubits'] = fields['state_qubits']
            
        # Ensure num_qubits is properly exposed
        if 'num_qubits' not in result and 'state_qubits' in fields:
            result['num_qubits'] = fields['state_qubits']
        
        return result
    
    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all states in the registry
        
        Returns:
            dict: Dictionary of state names to state information
        """
        result = {}
        for name, state in self.states.items():
            state_info = state.copy()
            fields = self.state_fields.get(name, {}).copy()
            state_info['fields'] = fields
            
            # Also expose key fields directly for API compatibility
            # These are the fields the API server expects at the top level
            if 'coherence' in fields:
                state_info['coherence'] = fields['coherence']
            if 'state_coherence' in fields:
                state_info['state_coherence'] = fields['state_coherence']
            if 'entropy' in fields:
                state_info['entropy'] = fields['entropy'] 
            if 'state_entropy' in fields:
                state_info['state_entropy'] = fields['state_entropy']
            if 'state_qubits' in fields:
                state_info['state_qubits'] = fields['state_qubits']
                
            # Ensure num_qubits is properly exposed
            if 'num_qubits' not in state_info and 'state_qubits' in fields:
                state_info['num_qubits'] = fields['state_qubits']
                
            result[name] = state_info
        
        return result
    
    def get_states_by_type(self, state_type: str) -> List[str]:
        """
        Get all states of a specific type
        
        Args:
            state_type (str): State type
            
        Returns:
            list: List of state names
        """
        return [name for name, state in self.states.items() if state.get('type') == state_type]
    
    def get_entangled_states(self) -> List[Tuple[str, str, float]]:
        """
        Get all entangled state pairs
        
        Returns:
            list: List of (state1, state2, entanglement_strength) tuples
        """
        result = []
        for (state1, state2), strength in self.entanglement_relationships.items():
            result.append((state1, state2, strength))
        
        return result
    
    def set_field(self, state_name: str, field_name: str, value: Any) -> bool:
        """
        Set a field value for a state
        
        Args:
            state_name (str): State name
            field_name (str): Field name
            value: Field value
            
        Returns:
            bool: True if field was set successfully
        """
        if state_name not in self.states:
            logger.warning(f"Cannot set field for non-existent state: {state_name}")
            return False
        
        # Get previous value for change detection
        previous_value = None
        if state_name in self.state_fields and field_name in self.state_fields[state_name]:
            previous_value = self.state_fields[state_name][field_name]
        
        # Set field
        self.state_fields.setdefault(state_name, {})[field_name] = value
        
        # Update state version and modification time
        self.states[state_name]['version'] += 1
        current_time = time.time()
        self.states[state_name]['modified_at'] = current_time
        self.states[state_name]['modified_datetime'] = datetime.fromtimestamp(current_time)
        
        # Handle special fields
        if field_name == 'is_entangled' and value is True:
            # Make sure entangled_with field exists
            if 'entangled_with' not in self.state_fields[state_name]:
                self.state_fields[state_name]['entangled_with'] = []
        
        # Replace the current entangled_with handling in set_field with:
        elif field_name == 'entangled_with':
            # Update entanglement relationships
            if value and isinstance(value, (str, list)):
                entangled_states = [value] if isinstance(value, str) else value
                
                # Get previous entangled states for comparison
                previous_entangled = self.state_fields.get(state_name, {}).get('entangled_with', [])
                if not isinstance(previous_entangled, list):
                    previous_entangled = []
                
                # Break entanglement with states no longer in the list
                for other_state in previous_entangled:
                    if other_state not in entangled_states:
                        self._update_entanglement_links(state_name, other_state, remove=True)
                        self.stats['entanglements_broken'] += 1
                
                # Add entanglement with new states
                for other_state in entangled_states:
                    if other_state in self.states and other_state not in previous_entangled:
                        self._update_entanglement_links(state_name, other_state)
                        self.stats['entanglements_created'] += 1
                
                # Set is_entangled to True as well
                self.state_fields[state_name]['is_entangled'] = len(entangled_states) > 0
                
                # Update stats
                if previous_value is None or (isinstance(previous_value, list) and len(previous_value) == 0):
                    self.stats['entanglements_created'] += len(entangled_states)
        
        # Emit field change event if event system is available
        if self.event_system and previous_value != value:
            field_event_types = {
                'coherence': 'coherence_change_event',
                'entropy': 'entropy_change_event',
                'is_entangled': 'entanglement_status_event'
            }
            
            if field_name in field_event_types:
                self.event_system.emit_quantum_event(
                    field_event_types[field_name],
                    state_name,
                    {
                        'field_name': field_name,
                        'previous_value': previous_value,
                        'new_value': value,
                        'state_type': self.states[state_name]['type']
                    }
                )
        
        # Update stats
        self.stats['fields_set'] += 1
        
        return True
    
    def get_field(self, state_name: str, field_name: str) -> Optional[Any]:
        """
        Get a field value for a state
        
        Args:
            state_name (str): State name
            field_name (str): Field name
            
        Returns:
            Field value, or None if not found
        """
        if state_name not in self.states:
            return None
        
        # Check if the field exists directly
        if state_name in self.state_fields and field_name in self.state_fields[state_name]:
            # Update stats
            self.stats['fields_get'] += 1
            return self.state_fields[state_name][field_name]
        
        # Check for derived fields that can be computed
        if field_name == 'coherence' and state_name in self.state_vectors:
            # Calculate coherence on-demand
            state_vector = self.state_vectors[state_name]
            coherence = self._calculate_coherence(state_vector)
            
            # Cache the result
            self.state_fields.setdefault(state_name, {})['coherence'] = coherence
            
            # Update stats
            self.stats['fields_get'] += 1
            
            return coherence
        
        elif field_name == 'entropy' and state_name in self.density_matrices:
            # Calculate entropy on-demand
            density_matrix = self.density_matrices[state_name]
            entropy = self._calculate_entropy(density_matrix)
            
            # Cache the result
            self.state_fields.setdefault(state_name, {})['entropy'] = entropy
            
            # Update stats
            self.stats['fields_get'] += 1
            
            return entropy
        
        # Field not found
        return None
    
    def _calculate_coherence(self, state_vector=None, density_matrix=None) -> float:
        """
        Calculate coherence using a sophisticated approach.
        Can use either state vector or density matrix input.
        
        Args:
            state_vector: Optional quantum state vector
            density_matrix: Optional density matrix
            
        Returns:
            float: Coherence value between 0.0 and 1.0
        """
        # Use coherence manager if available
        if self.coherence_manager:
            try:
                if density_matrix is not None:
                    return self.coherence_manager.calculate_coherence_from_density(density_matrix)
                elif state_vector is not None:
                    return self.coherence_manager.calculate_coherence(state_vector)
            except Exception as e:
                logger.debug(f"CoherenceManager failed, using fallback: {e}")
                # Continue to fallback implementation
        
        # Fallback implementation if coherence_manager is not available or failed
        if density_matrix is None and state_vector is not None:
            # Construct density matrix from state vector
            density_matrix = np.outer(state_vector, np.conjugate(state_vector))
        
        if density_matrix is None:
            return 0.0
        
        # Calculate l1-norm coherence
        dim = density_matrix.shape[0]
        
        # Use faster vectorized operations
        diag_indices = np.diag_indices(dim)
        off_diag_mask = np.ones((dim, dim), dtype=bool)
        off_diag_mask[diag_indices] = False
        
        off_diag_sum = np.sum(np.abs(density_matrix[off_diag_mask]))
        
        # Calculate reference basis coherence
        normalization = dim * (dim - 1)
        if normalization == 0:
            return 0.0
        
        # Calculate robustness of coherence - cache eigenvalues for entropy calculation
        eigenvalues = np.linalg.eigvalsh(density_matrix)
        purity = np.sum(eigenvalues ** 2)  # Tr(ρ²)
        
        # Weight between l1-norm and purity-based coherence
        l1_coherence = off_diag_sum / normalization
        purity_factor = (purity - 1/dim) / (1 - 1/dim)  # Normalized purity
        
        # Combined coherence measure
        coherence = 0.7 * l1_coherence + 0.3 * purity_factor
        return min(1.0, max(0.0, coherence))

    def _calculate_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculate von Neumann entropy of a density matrix
        
        Args:
            density_matrix: Quantum density matrix
            
        Returns:
            float: Entropy value between 0.0 and 1.0
        """
        try:
            # Calculate eigenvalues of density matrix
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            
            # Optimize by using vectorized operations with a mask
            mask = eigenvalues > 1e-10
            valid_eigenvalues = eigenvalues[mask]
            
            if len(valid_eigenvalues) == 0:
                return 0.0
            
            # Calculate von Neumann entropy: -Tr(ρ log ρ)
            entropy = -np.sum(valid_eigenvalues * np.log2(valid_eigenvalues))
            
            # Normalize to get a value between 0 and 1
            max_entropy = np.log2(len(density_matrix))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            return min(1.0, max(0.0, normalized_entropy))
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            return 0.0
        
    def delete_state(self, name: str) -> bool:
        """
        Delete a state
        
        Args:
            name (str): State name
            
        Returns:
            bool: True if state was deleted successfully
        """
        if name not in self.states:
            logger.warning(f"Cannot delete non-existent state: {name}")
            return False
        
        # Capture state information for event
        state_info = self.get_state(name)
        
        # Clean up entanglement relationships
        entangled_with = self.state_fields.get(name, {}).get('entangled_with', [])
        if isinstance(entangled_with, list):
            for other_state in entangled_with:
                # Remove this state from other state's entangled_with list
                if other_state in self.state_fields and 'entangled_with' in self.state_fields[other_state]:
                    other_entangled = self.state_fields[other_state]['entangled_with']
                    if isinstance(other_entangled, list) and name in other_entangled:
                        other_entangled.remove(name)
                        self.state_fields[other_state]['entangled_with'] = other_entangled
                        self.state_fields[other_state]['is_entangled'] = len(other_entangled) > 0
                
                # Remove entanglement relationship
                key = tuple(sorted([name, other_state]))
                if key in self.entanglement_relationships:
                    del self.entanglement_relationships[key]
                    self.stats['entanglements_broken'] += 1
        
        # Remove from memory manager if allocated
        memory_id = self.states[name].get('memory_id')
        if memory_id is not None and self.memory_manager:
            try:
                self.memory_manager.deallocate(memory_id)
            except Exception as e:
                logger.error(f"Failed to deallocate memory for state '{name}': {e}")
        
        # Delete state data
        del self.states[name]
        if name in self.state_fields:
            del self.state_fields[name]
        if name in self.state_vectors:
            del self.state_vectors[name]
        if name in self.density_matrices:
            del self.density_matrices[name]
        
        # Emit destruction event if event system is available
        if self.event_system:
            self.event_system.emit_quantum_event(
                'state_destruction_event',
                name,
                state_info
            )
        
        # Update stats
        self.stats['states_deleted'] += 1
        
        logger.info(f"Deleted state '{name}'")
        return True
    
    def register_state_type(self, name: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a state type
        
        Args:
            name (str): Type name
            properties (dict, optional): Type properties
            
        Returns:
            bool: True if type was registered successfully
        """
        if name in self.state_types:
            logger.warning(f"State type '{name}' already exists, cannot register")
            return False
        
        self.state_types[name] = properties or {}
        logger.info(f"Registered state type '{name}'")
        return True
    
    def get_state_type(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a state type by name
        
        Args:
            name (str): Type name
            
        Returns:
            dict: Type properties, or None if not found
        """
        return self.state_types.get(name)
    
    def get_all_state_types(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all registered state types
        
        Returns:
            dict: Dictionary of type names to type properties
        """
        return self.state_types.copy()
    
    def set_state_vector(self, name: str, state_vector: np.ndarray) -> bool:
        """
        Set the state vector for a quantum state
        
        Args:
            name (str): State name
            state_vector: Quantum state vector
            
        Returns:
            bool: True if successful
        """
        if name not in self.states:
            logger.warning(f"Cannot set state vector for non-existent state: {name}")
            return False
        
        # Ensure state vector is properly normalized
        norm = np.linalg.norm(state_vector)
        if abs(norm - 1.0) > 1e-6:
            logger.warning(f"State vector for '{name}' not normalized, normalizing")
            state_vector = state_vector / norm
        
        # Update state vector
        self.state_vectors[name] = state_vector
        
        # Update corresponding density matrix
        self.density_matrices[name] = np.outer(state_vector, np.conjugate(state_vector))
        
        # Update coherence and entropy
        coherence = self._calculate_coherence(state_vector)
        entropy = self._calculate_entropy(self.density_matrices[name])
        
        self.state_fields.setdefault(name, {})['coherence'] = coherence
        self.state_fields.setdefault(name, {})['entropy'] = entropy
        
        # Update state version
        self.states[name]['version'] += 1
        current_time = time.time()
        self.states[name]['modified_at'] = current_time
        self.states[name]['modified_datetime'] = datetime.fromtimestamp(current_time)
        
        return True
    
    def get_state_vector(self, name: str) -> Optional[np.ndarray]:
        """
        Get the state vector for a quantum state
        
        Args:
            name (str): State name
            
        Returns:
            np.ndarray: State vector, or None if not found
        """
        return self.state_vectors.get(name)
    
    def _register_with_recursive_mechanics(self, name, properties):
        """
        Register a state with the recursive mechanics system
        
        Args:
            name (str): State name
            properties (dict): State properties
        """
        if not self.recursive_mechanics:
            return
        
        parent_state = None
        level = 0
        recursion_boundary = False
        
        # Check if properties specify recursive parameters
        if properties:
            parent_state = properties.get('parent_state')
            level = properties.get('recursion_level', 0)
            recursion_boundary = properties.get('recursion_boundary', False)
        
        # Register the system
        self.recursive_mechanics.register_system(name, level, parent_state)
        
        # Set up recursion boundary if specified
        if recursion_boundary and parent_state:
            try:
                self.recursive_mechanics.set_boundary_condition(
                    level, level+1, 
                    properties.get('boundary_condition', 'standard')
                )
            except Exception as e:
                logger.warning(f"Failed to set recursion boundary: {e}")
                
    def set_density_matrix(self, name: str, density_matrix: np.ndarray) -> bool:
        """
        Set the density matrix for a quantum state
        
        Args:
            name (str): State name
            density_matrix: Quantum density matrix
            
        Returns:
            bool: True if successful
        """
        if name not in self.states:
            logger.warning(f"Cannot set density matrix for non-existent state: {name}")
            return False
        
        # Check if density matrix is valid (Hermitian, trace 1, positive semidefinite)
        try:
            # Check Hermitian
            if not np.allclose(density_matrix, density_matrix.conj().T):
                logger.warning(f"Density matrix for '{name}' not Hermitian, correcting")
                density_matrix = 0.5 * (density_matrix + density_matrix.conj().T)
            
            # Check trace 1
            trace = np.trace(density_matrix).real
            if abs(trace - 1.0) > 1e-6:
                logger.warning(f"Density matrix for '{name}' has trace {trace}, normalizing")
                density_matrix = density_matrix / trace
            
            # Check positive semidefinite
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            if np.any(eigenvalues < -1e-6):
                logger.warning(f"Density matrix for '{name}' not positive semidefinite, correcting")
                # Project onto positive semidefinite cone
                D, V = np.linalg.eigh(density_matrix)
                D = np.maximum(D, 0)
                D = D / np.sum(D)  # Renormalize
                density_matrix = V @ np.diag(D) @ V.conj().T
        except Exception as e:
            logger.error(f"Error validating density matrix for '{name}': {e}")
            return False
        
        # Update density matrix
        self.density_matrices[name] = density_matrix
        
        # If pure state, also update state vector
        if np.allclose(np.trace(density_matrix @ density_matrix).real, 1.0, atol=1e-6):
            # Pure state, extract state vector
            eigenvalues, eigenvectors = np.linalg.eigh(density_matrix)
            max_idx = np.argmax(eigenvalues)
            state_vector = eigenvectors[:, max_idx]
            
            # Fix global phase
            # Convention: First non-zero element should be real and positive
            for i in range(len(state_vector)):
                if abs(state_vector[i]) > 1e-6:
                    phase = np.angle(state_vector[i])
                    state_vector = state_vector * np.exp(-1j * phase)
                    break
            
            self.state_vectors[name] = state_vector

            # Update corresponding density matrix
            self.density_matrices[name] = np.outer(state_vector, np.conjugate(state_vector))

            # Update coherence and entropy
            self._update_state_coherence_and_entropy(name)
        else:
            # Mixed state, remove state vector if it exists
            if name in self.state_vectors:
                del self.state_vectors[name]
        
        # Update coherence and entropy
        coherence = self._calculate_coherence(self.state_vectors.get(name, np.array([1.0])))
        entropy = self._calculate_entropy(density_matrix)
        
        self.state_fields.setdefault(name, {})['coherence'] = coherence
        self.state_fields.setdefault(name, {})['entropy'] = entropy
        
        # Update state version
        self.states[name]['version'] += 1
        current_time = time.time()
        self.states[name]['modified_at'] = current_time
        self.states[name]['modified_datetime'] = datetime.fromtimestamp(current_time)
        
        return True
    
    def get_density_matrix(self, name: str) -> Optional[np.ndarray]:
        """
        Get the density matrix for a quantum state
        
        Args:
            name (str): State name
            
        Returns:
            np.ndarray: Density matrix, or None if not found
        """
        return self.density_matrices.get(name)
    
    def entangle_states(self, state1: str, state2: str, strength: float = 1.0,
                        qubits1: Optional[Union[int, List[int]]] = None,
                        qubits2: Optional[Union[int, List[int]]] = None) -> bool:
        """
        Entangle two quantum states
        
        Args:
            state1 (str): First state name
            state2 (str): Second state name
            strength (float): Entanglement strength (0.0 to 1.0)
            qubits1 (int or list, optional): Qubit(s) from first state to entangle
            qubits2 (int or list, optional): Qubit(s) from second state to entangle
            
        Returns:
            bool: True if states were entangled successfully
        """
        if state1 not in self.states:
            logger.warning(f"Cannot entangle: state '{state1}' does not exist")
            return False
        
        if state2 not in self.states:
            logger.warning(f"Cannot entangle: state '{state2}' does not exist")
            return False
        
        if state1 == state2:
            logger.warning(f"Cannot entangle state '{state1}' with itself")
            return False
        
        # Update entanglement links
        if not self._update_entanglement_links(state1, state2, strength):
            return False
        
        # Emit entanglement event if event system is available
        if self.event_system:
            self.event_system.emit_entanglement_event(
                state1, state2, 'entanglement_creation_event',
                {
                    'strength': strength,
                    'qubits1': qubits1,
                    'qubits2': qubits2
                }
            )
        
        # Update stats
        self.stats['entanglements_created'] += 1
        
        logger.info(f"Entangled states '{state1}' and '{state2}' with strength {strength}")
        return True

    def break_entanglement(self, state1: str, state2: str) -> bool:
        """
        Break entanglement between two quantum states
        
        Args:
            state1 (str): First state name
            state2 (str): Second state name
            
        Returns:
            bool: True if entanglement was broken successfully
        """
        if state1 not in self.states or state2 not in self.states:
            return False
        
        # Check if states are entangled
        key = tuple(sorted([state1, state2]))
        if key not in self.entanglement_relationships:
            return False
        
        # Get current strength for event
        strength = self.entanglement_relationships[key]
        
        # Update entanglement links (removal)
        if not self._update_entanglement_links(state1, state2, remove=True):
            return False
        
        # Emit entanglement breaking event if event system is available
        if self.event_system:
            self.event_system.emit_entanglement_event(
                state1, state2, 'entanglement_breaking_event',
                {
                    'previous_strength': strength
                }
            )
        
        # Update stats
        self.stats['entanglements_broken'] += 1
        
        logger.info(f"Broke entanglement between states '{state1}' and '{state2}'")
        return True

    def break_entanglement(self, state1: str, state2: str) -> bool:
        """
        Break entanglement between two quantum states
        
        Args:
            state1 (str): First state name
            state2 (str): Second state name
            
        Returns:
            bool: True if entanglement was broken successfully
        """
        if state1 not in self.states or state2 not in self.states:
            return False
        
        # Check if states are entangled
        key = tuple(sorted([state1, state2]))
        if key not in self.entanglement_relationships:
            return False
        
        # Remove entanglement relationship
        strength = self.entanglement_relationships.pop(key)
        
        # Update entangled_with fields
        if state1 in self.state_fields and 'entangled_with' in self.state_fields[state1]:
            entangled = self.state_fields[state1]['entangled_with']
            if isinstance(entangled, list) and state2 in entangled:
                entangled.remove(state2)
                self.state_fields[state1]['entangled_with'] = entangled
                self.state_fields[state1]['is_entangled'] = len(entangled) > 0
        
        if state2 in self.state_fields and 'entangled_with' in self.state_fields[state2]:
            entangled = self.state_fields[state2]['entangled_with']
            if isinstance(entangled, list) and state1 in entangled:
                entangled.remove(state1)
                self.state_fields[state2]['entangled_with'] = entangled
                self.state_fields[state2]['is_entangled'] = len(entangled) > 0
        
        # Update state versions
        current_time = time.time()
        
        self.states[state1]['version'] += 1
        self.states[state1]['modified_at'] = current_time
        self.states[state1]['modified_datetime'] = datetime.fromtimestamp(current_time)
        
        self.states[state2]['version'] += 1
        self.states[state2]['modified_at'] = current_time
        self.states[state2]['modified_datetime'] = datetime.fromtimestamp(current_time)
        
        # Emit entanglement breaking event if event system is available
        if self.event_system:
            self.event_system.emit_entanglement_event(
                state1, state2, 'entanglement_breaking_event',
                {
                    'previous_strength': strength
                }
            )
        
        # Update stats
        self.stats['entanglements_broken'] += 1
        
        logger.info(f"Broke entanglement between states '{state1}' and '{state2}'")
        return True
    
    def get_entanglement_strength(self, state1: str, state2: str) -> float:
        """
        Get the entanglement strength between two states
        
        Args:
            state1 (str): First state name
            state2 (str): Second state name
            
        Returns:
            float: Entanglement strength (0.0 if not entangled)
        """
        key = tuple(sorted([state1, state2]))
        return self.entanglement_relationships.get(key, 0.0)
    
    def get_entangled_states(self, state_name: str) -> List[Tuple[str, float]]:
        """
        Get all states entangled with a given state
        
        Args:
            state_name (str): State name
            
        Returns:
            list: List of (state_name, strength) tuples
        """
        result = []
        
        # Get entangled_with list from state fields
        entangled_with = self.state_fields.get(state_name, {}).get('entangled_with', [])
        if not isinstance(entangled_with, list):
            return result
        
        # Get strength for each entangled state
        for other_state in entangled_with:
            key = tuple(sorted([state_name, other_state]))
            strength = self.entanglement_relationships.get(key, 0.0)
            result.append((other_state, strength))
        
        return result
    
    def create_composite_state(self, name: str, component_states: List[str], 
                              state_type: str = 'composite_type') -> bool:
        """
        Create a composite state from component states
        
        Args:
            name (str): New state name
            component_states (list): List of component state names
            state_type (str): State type for the composite
            
        Returns:
            bool: True if composite state was created successfully
        """
        # Check if all component states exist
        for state in component_states:
            if state not in self.states:
                logger.warning(f"Cannot create composite: component state '{state}' does not exist")
                return False
        
        # Get total number of qubits from component states
        total_qubits = sum(self.states[state].get('num_qubits', 0) for state in component_states)
        
        # Create the composite state
        if not self.create_state(name, state_type, total_qubits):
            return False
        
        # Set component states field
        self.state_fields[name]['component_states'] = component_states
        
        # For quantum states, we would need to construct the tensor product of the component states
        # This is a complex operation that would be implemented in the quantum backend
        
        logger.info(f"Created composite state '{name}' from components: {component_states}")
        return True
    
    def set_recursive_mechanics(self, recursive_mechanics) -> None:
        """
        Set the recursive mechanics module for recursive state operations
        
        Args:
            recursive_mechanics: RecursiveMechanics instance
        """
        self.recursive_mechanics = recursive_mechanics
    
    def export_state(self, state_name: str) -> Optional[Dict[str, Any]]:
        """
        Export a state's complete data for serialization
        
        Args:
            state_name (str): State name
            
        Returns:
            dict: State export data, or None if state not found
        """
        if state_name not in self.states:
            return None
        
        # Get basic state info
        state_export = self.states[state_name].copy()
        
        # Add fields
        state_export['fields'] = self.state_fields.get(state_name, {}).copy()
        
        # Add state vector if available (convert to list for JSON serialization)
        if state_name in self.state_vectors:
            state_vector = self.state_vectors[state_name]
            state_export['state_vector'] = [complex(x.real, x.imag) for x in state_vector]
        
        # Add density matrix if available (convert to list of lists for JSON serialization)
        if state_name in self.density_matrices:
            density_matrix = self.density_matrices[state_name]
            state_export['density_matrix'] = [
                [complex(x.real, x.imag) for x in row] for row in density_matrix
            ]
        
        # Add entanglement info
        state_export['entanglement'] = {
            'is_entangled': self.state_fields.get(state_name, {}).get('is_entangled', False),
            'entangled_with': self.get_entangled_states(state_name)
        }
        
        return state_export
    
    def import_state(self, state_data: Dict[str, Any]) -> bool:
        """
        Import a state from exported data
        
        Args:
            state_data (dict): State data to import
            
        Returns:
            bool: True if state was imported successfully
        """
        if not state_data or not isinstance(state_data, dict) or 'name' not in state_data:
            logger.warning("Invalid state data for import")
            return False
        
        name = state_data['name']
        
        # Check if state already exists
        if name in self.states:
            logger.warning(f"State '{name}' already exists, cannot import")
            return False
        
        # Create the state
        state_type = state_data.get('type', 'quantum_type')
        num_qubits = state_data.get('num_qubits')
        
        if not self.create_state(name, state_type, num_qubits):
            return False
        
        # Import fields
        if 'fields' in state_data and isinstance(state_data['fields'], dict):
            for field_name, value in state_data['fields'].items():
                self.state_fields[name][field_name] = value
        
        # Import state vector if available
        if 'state_vector' in state_data and isinstance(state_data['state_vector'], list):
            try:
                state_vector = np.array(
                    [complex(x) if isinstance(x, (int, float)) else complex(x[0], x[1])
                     for x in state_data['state_vector']],
                    dtype=complex
                )
                self.state_vectors[name] = state_vector
            except Exception as e:
                logger.warning(f"Error importing state vector for '{name}': {e}")
        
        # Import density matrix if available
        if 'density_matrix' in state_data and isinstance(state_data['density_matrix'], list):
            try:
                density_matrix = np.array(
                    [[complex(x) if isinstance(x, (int, float)) else complex(x[0], x[1])
                      for x in row] for row in state_data['density_matrix']],
                    dtype=complex
                )
                self.density_matrices[name] = density_matrix
            except Exception as e:
                logger.warning(f"Error importing density matrix for '{name}': {e}")
        
        # Import entanglement info
        if 'entanglement' in state_data and isinstance(state_data['entanglement'], dict):
            entanglement = state_data['entanglement']
            
            # Set is_entangled field
            is_entangled = entanglement.get('is_entangled', False)
            self.state_fields[name]['is_entangled'] = is_entangled
            
            # Add entanglement relationships
            entangled_with = entanglement.get('entangled_with', [])
            if isinstance(entangled_with, list):
                for other_state, strength in entangled_with:
                    if other_state in self.states:
                        self.entangle_states(name, other_state, strength)
        
        logger.info(f"Imported state '{name}'")
        return True
    
    def merge_states(self, state1: str, state2: str, new_name: str, 
                    merge_type: str = 'superposition') -> bool:
        """
        Merge two quantum states into a new state
        
        Args:
            state1 (str): First state name
            state2 (str): Second state name
            new_name (str): Name for the merged state
            merge_type (str): Type of merging ('superposition', 'mixture', 'tensor_product')
            
        Returns:
            bool: True if states were merged successfully
        """
        if state1 not in self.states:
            logger.warning(f"Cannot merge: state '{state1}' does not exist")
            return False
        
        if state2 not in self.states:
            logger.warning(f"Cannot merge: state '{state2}' does not exist")
            return False
        
        if new_name in self.states:
            logger.warning(f"Cannot merge: target state '{new_name}' already exists")
            return False
        
        # Get state vectors for quantum states
        if merge_type in ('superposition', 'mixture') and (
            state1 not in self.state_vectors or state2 not in self.state_vectors):
            logger.warning(f"Cannot {merge_type}: state vectors not available")
            return False
        
        # Perform the merge based on the type
        if merge_type == 'superposition':
            # Equal superposition of the two states
            sv1 = self.state_vectors[state1]
            sv2 = self.state_vectors[state2]
            
            # Check if states have the same dimension
            if len(sv1) != len(sv2):
                logger.warning(f"Cannot superpose: states have different dimensions ({len(sv1)} vs {len(sv2)})")
                return False
            
            # Create equal superposition
            new_state_vector = (sv1 + sv2) / np.sqrt(2)
            
            # Create new state
            num_qubits = self.states[state1].get('num_qubits')
            if not self.create_state(new_name, 'superposition_type', num_qubits):
                return False
            
            # Set state vector
            self.state_vectors[new_name] = new_state_vector
            
            # Set density matrix
            self.density_matrices[new_name] = np.outer(new_state_vector, np.conjugate(new_state_vector))
            
            # Update fields
            self.state_fields[new_name]['merged_from'] = [state1, state2]
            self.state_fields[new_name]['merge_type'] = 'superposition'
            
        elif merge_type == 'mixture':
            # Equal mixture of the two states (mixed state)
            dm1 = self.density_matrices[state1]
            dm2 = self.density_matrices[state2]
            
            # Check if states have the same dimension
            if dm1.shape[0] != dm2.shape[0]:
                logger.warning(f"Cannot mix: states have different dimensions ({dm1.shape[0]} vs {dm2.shape[0]})")
                return False
            
            # Create equal mixture
            new_density_matrix = (dm1 + dm2) / 2
            
            # Create new state
            num_qubits = self.states[state1].get('num_qubits')
            if not self.create_state(new_name, 'mixed_type', num_qubits):
                return False
            
            # Set density matrix
            self.density_matrices[new_name] = new_density_matrix
            
            # Update fields
            self.state_fields[new_name]['merged_from'] = [state1, state2]
            self.state_fields[new_name]['merge_type'] = 'mixture'
            
        elif merge_type == 'tensor_product':
            # Tensor product of the two states
            num_qubits1 = self.states[state1].get('num_qubits', 0)
            num_qubits2 = self.states[state2].get('num_qubits', 0)
            
            if num_qubits1 <= 0 or num_qubits2 <= 0:
                logger.warning(f"Cannot tensor: invalid qubit counts ({num_qubits1}, {num_qubits2})")
                return False
            
            # Create new state
            total_qubits = num_qubits1 + num_qubits2
            if not self.create_state(new_name, 'composite_type', total_qubits):
                return False
            
            # If we have state vectors, compute tensor product
            if state1 in self.state_vectors and state2 in self.state_vectors:
                sv1 = self.state_vectors[state1]
                sv2 = self.state_vectors[state2]
                
                # Compute tensor product
                new_state_vector = np.kron(sv1, sv2)
                self.state_vectors[new_name] = new_state_vector
                
                # Set density matrix
                self.density_matrices[new_name] = np.outer(new_state_vector, np.conjugate(new_state_vector))
            
            # If we have density matrices, compute tensor product
            elif state1 in self.density_matrices and state2 in self.density_matrices:
                dm1 = self.density_matrices[state1]
                dm2 = self.density_matrices[state2]
                
                # Compute tensor product
                new_density_matrix = np.kron(dm1, dm2)
                self.density_matrices[new_name] = new_density_matrix
            
            # Update fields
            self.state_fields[new_name]['merged_from'] = [state1, state2]
            self.state_fields[new_name]['merge_type'] = 'tensor_product'
        
        else:
            logger.warning(f"Unsupported merge type: {merge_type}")
            return False
        
        logger.info(f"Merged states '{state1}' and '{state2}' into '{new_name}' using {merge_type}")
        return True
    
    def clone_state(self, source_name: str, new_name: str) -> bool:
        """
        Clone a quantum state to a new state (classical copy, not quantum cloning)
        
        Args:
            source_name (str): Source state name
            new_name (str): New state name
            
        Returns:
            bool: True if state was cloned successfully
        """
        if source_name not in self.states:
            logger.warning(f"Cannot clone: source state '{source_name}' does not exist")
            return False
        
        if new_name in self.states:
            logger.warning(f"Cannot clone: target state '{new_name}' already exists")
            return False
        
        # Get source state info
        source_state = self.states[source_name]
        
        # Create new state with same properties
        if not self.create_state(
            new_name, 
            source_state.get('type', 'quantum_type'), 
            source_state.get('num_qubits')
        ):
            return False
        
        # Copy fields from source state
        if source_name in self.state_fields:
            # Copy all fields except entanglement-related ones
            for field_name, value in self.state_fields[source_name].items():
                if field_name not in ('is_entangled', 'entangled_with'):
                    self.state_fields[new_name][field_name] = value
        
        # Copy state vector if available
        if source_name in self.state_vectors:
            self.state_vectors[new_name] = self.state_vectors[source_name].copy()
        
        # Copy density matrix if available
        if source_name in self.density_matrices:
            self.density_matrices[new_name] = self.density_matrices[source_name].copy()
        
        # Add clone-specific fields
        self.state_fields[new_name]['cloned_from'] = source_name
        self.state_fields[new_name]['cloned_at'] = time.time()
        
        logger.info(f"Cloned state '{source_name}' to '{new_name}'")
        return True
    
    def _initialize_state_record(self, name, state_type, num_qubits):
        """
        Initialize a state record with metadata and default fields
        
        Args:
            name (str): State name
            state_type (str): State type from registered types
            num_qubits (int, optional): Number of qubits for the state
            
        Returns:
            bool: True if initialization was successful
        """
        created_at = time.time()
        self.states[name] = {
            'name': name,
            'type': state_type,
            'num_qubits': num_qubits,
            'created_at': created_at,
            'created_datetime': datetime.fromtimestamp(created_at),
            'version': 1,
            'modified_at': created_at,
            'modified_datetime': datetime.fromtimestamp(created_at),
            'id': len(self.states) + 1  # Assign unique ID
        }
        
        # Initialize fields with defaults from the state type
        self.state_fields[name] = {}
        type_defaults = self.state_types[state_type].get('default_fields', {})
        for field_name, default_value in type_defaults.items():
            self.state_fields[name][field_name] = default_value
        
        return True

    def _update_entanglement_links(self, state1, state2, strength=1.0, remove=False):
        """
        Update entanglement relationships between two states
        
        Args:
            state1 (str): First state name
            state2 (str): Second state name
            strength (float): Entanglement strength (0.0 to 1.0)
            remove (bool): If True, remove the entanglement instead of adding it
            
        Returns:
            bool: True if operation was successful
        """
        key = tuple(sorted([state1, state2]))
        current_time = time.time()
        
        if remove:
            # Remove entanglement
            if key in self.entanglement_relationships:
                del self.entanglement_relationships[key]
                
            # Update entangled_with fields
            for state_name, other_state in [(state1, state2), (state2, state1)]:
                if state_name in self.state_fields and 'entangled_with' in self.state_fields[state_name]:
                    entangled = self.state_fields[state_name]['entangled_with']
                    if isinstance(entangled, list) and other_state in entangled:
                        entangled.remove(other_state)
                        self.state_fields[state_name]['entangled_with'] = entangled
                        self.state_fields[state_name]['is_entangled'] = len(entangled) > 0
        else:
            # Add entanglement
            self.entanglement_relationships[key] = max(0.0, min(1.0, strength))
            
            # Update entangled_with fields
            for state_name, other_state in [(state1, state2), (state2, state1)]:
                entangled_with = self.state_fields.setdefault(state_name, {}).get('entangled_with', [])
                if isinstance(entangled_with, list) and other_state not in entangled_with:
                    entangled_with.append(other_state)
                    self.state_fields[state_name]['entangled_with'] = entangled_with
                    self.state_fields[state_name]['is_entangled'] = True
        
        # Update state versions
        for state_name in [state1, state2]:
            if state_name in self.states:
                self.states[state_name]['version'] += 1
                self.states[state_name]['modified_at'] = current_time
                self.states[state_name]['modified_datetime'] = datetime.fromtimestamp(current_time)
        
        return True

    def _update_state_coherence_and_entropy(self, state_name):
        """
        Update coherence and entropy values for a state
        
        Args:
            state_name (str): State name
            
        Returns:
            Tuple[float, float]: Coherence and entropy values
        """
        coherence = None
        entropy = None
        
        # Calculate coherence from state vector if available
        if state_name in self.state_vectors:
            state_vector = self.state_vectors[state_name]
            coherence = self._calculate_coherence(state_vector=state_vector)
        
        # Calculate entropy from density matrix if available
        if state_name in self.density_matrices:
            density_matrix = self.density_matrices[state_name]
            entropy = self._calculate_entropy(density_matrix)
            
            # If coherence not calculated from state vector, calculate from density matrix
            if coherence is None:
                coherence = self._calculate_coherence(density_matrix=density_matrix)
        
        # Update fields
        if coherence is not None:
            self.state_fields.setdefault(state_name, {})['coherence'] = coherence
            
        if entropy is not None:
            self.state_fields.setdefault(state_name, {})['entropy'] = entropy
            
        return coherence, entropy

    def _partial_trace(self, density_matrix: np.ndarray, num_qubits: int, trace_qubits: List[int]) -> np.ndarray:
        """
        Perform partial trace on a density matrix
        
        Args:
            density_matrix: Full system density matrix
            num_qubits: Total number of qubits in the system
            trace_qubits: Indices of qubits to trace out
            
        Returns:
            np.ndarray: Reduced density matrix
        """
        if not trace_qubits:
            return density_matrix
        
        # Sort qubits to trace
        trace_qubits = sorted(trace_qubits)
        
        # Calculate dimensions
        dim_full = 2 ** num_qubits
        dim_trace = 2 ** len(trace_qubits)
        dim_keep = 2 ** (num_qubits - len(trace_qubits))
        
        # Identify which qubits to keep
        keep_qubits = [i for i in range(num_qubits) if i not in trace_qubits]
        
        # Calculate permutation to move trace qubits to the end
        perm = keep_qubits + trace_qubits
        perm_inv = [0] * num_qubits
        for i, p in enumerate(perm):
            perm_inv[p] = i
        
        # Permute the density matrix if necessary
        if perm != list(range(num_qubits)):
            # Convert to multi-index form
            tensor_shape = [2] * (2 * num_qubits)
            reshaped_dm = density_matrix.reshape(tensor_shape)
            
            # Apply permutation
            perm_indices = perm + [num_qubits + p for p in perm]
            permuted_dm = np.transpose(reshaped_dm, perm_indices)
            
            # Convert back to matrix form
            permuted_dm = permuted_dm.reshape((dim_full, dim_full))
        else:
            permuted_dm = density_matrix
        
        # Perform the partial trace
        reduced_dm = np.zeros((dim_keep, dim_keep), dtype=complex)
        for i in range(dim_trace):
            # Calculate offset in the full density matrix
            i_offset = i * dim_keep
            for j in range(dim_keep):
                for k in range(dim_keep):
                    reduced_dm[j, k] += permuted_dm[j + i_offset, k + i_offset]
        
        return reduced_dm

    def partial_trace(self, state_name: str, subsystem: str, trace_qubits: List[int]) -> bool:
        """
        Perform partial trace on a quantum state, creating a new state
        
        Args:
            state_name (str): Source state name
            subsystem (str): Name for the resulting subsystem state
            trace_qubits (list): Indices of qubits to trace out
            
        Returns:
            bool: True if partial trace was successful
        """
        if state_name not in self.states:
            logger.warning(f"Cannot partial trace: state '{state_name}' does not exist")
            return False
        
        if subsystem in self.states:
            logger.warning(f"Cannot partial trace: target state '{subsystem}' already exists")
            return False
        
        # Get density matrix of the state
        if state_name not in self.density_matrices:
            logger.warning(f"Cannot partial trace: density matrix for '{state_name}' not available")
            return False
        
        density_matrix = self.density_matrices[state_name]
        num_qubits = self.states[state_name].get('num_qubits', 0)
        
        if num_qubits <= 0:
            logger.warning(f"Cannot partial trace: invalid qubit count ({num_qubits})")
            return False
        
        # Validate trace_qubits
        for qubit in trace_qubits:
            if not 0 <= qubit < num_qubits:
                logger.warning(f"Cannot partial trace: qubit index {qubit} out of range [0, {num_qubits-1}]")
                return False
        
        # Calculate remaining qubits after trace
        remaining_qubits = num_qubits - len(trace_qubits)
        if remaining_qubits <= 0:
            logger.warning(f"Cannot partial trace: tracing out all qubits")
            return False
        
        try:
                # Use the improved partial trace implementation
            density_matrix = self.density_matrices[state_name]
            num_qubits = self.states[state_name].get('num_qubits', 0)
            result_dm = self._partial_trace(density_matrix, num_qubits, trace_qubits)
            
            # Create the subsystem state
            remaining_qubits = num_qubits - len(trace_qubits)
            if not self.create_state(subsystem, 'quantum_type', remaining_qubits):
                return False
            
            # Set density matrix
            self.density_matrices[subsystem] = result_dm
            
            # Create the subsystem state
            if not self.create_state(subsystem, 'quantum_type', remaining_qubits):
                return False
            
            # Set density matrix
            self.density_matrices[subsystem] = result_dm
            
            # If pure state, extract state vector
            if np.allclose(np.trace(result_dm @ result_dm).real, 1.0, atol=1e-6):
                # Pure state, extract state vector
                eigenvalues, eigenvectors = np.linalg.eigh(result_dm)
                max_idx = np.argmax(eigenvalues)
                state_vector = eigenvectors[:, max_idx]
                
                # Fix global phase
                for i in range(len(state_vector)):
                    if abs(state_vector[i]) > 1e-6:
                        phase = np.angle(state_vector[i])
                        state_vector = state_vector * np.exp(-1j * phase)
                        break
                
                self.state_vectors[subsystem] = state_vector
            
            # Update fields
            self.state_fields[subsystem]['derived_from'] = state_name
            self.state_fields[subsystem]['traced_qubits'] = trace_qubits
            
            logger.info(f"Performed partial trace on '{state_name}', created subsystem '{subsystem}'")
            return True
            
        except Exception as e:
            logger.error(f"Error performing partial trace: {e}")
            return False
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the states in the registry
        
        Returns:
            dict: State statistics
        """
        num_states = len(self.states)
        stats = {
            'total_states': num_states,
            'states_by_type': {},
            'total_qubits': 0,
            'entangled_pairs': len(self.entanglement_relationships),
            'average_coherence': 0.0,
            'average_entropy': 0.0
        }
        
        # Count states by type
        for state in self.states.values():
            state_type = state.get('type', 'unknown')
            stats['states_by_type'][state_type] = stats['states_by_type'].get(state_type, 0) + 1
            stats['total_qubits'] += state.get('num_qubits', 0)
        
        # Calculate average coherence and entropy
        total_coherence = 0.0
        total_entropy = 0.0
        count = 0
        
        for state_name in self.states:
            coherence = self.state_fields.get(state_name, {}).get('coherence')
            entropy = self.state_fields.get(state_name, {}).get('entropy')
            
            if coherence is not None:
                total_coherence += coherence
                count += 1
            
            if entropy is not None:
                total_entropy += entropy
        
        if count > 0:
            stats['average_coherence'] = total_coherence / count
            stats['average_entropy'] = total_entropy / count
        
        # Add performance stats
        stats.update(self.stats)
        
        return stats
    
    def reset(self) -> None:
        """
        Reset the registry, clearing all states
        """
        # Delete all states from memory manager
        if self.memory_manager:
            for state_name, state in self.states.items():
                memory_id = state.get('memory_id')
                if memory_id is not None:
                    try:
                        self.memory_manager.deallocate(memory_id)
                    except Exception as e:
                        logger.error(f"Failed to deallocate memory for state '{state_name}': {e}")
        
        # Clear data structures
        self.states = {}
        self.state_fields = {}
        self.state_vectors = {}
        self.density_matrices = {}
        self.entanglement_relationships = {}
        
        # Reset stats
        self.stats = {
            'states_created': 0,
            'states_deleted': 0,
            'fields_set': 0,
            'fields_get': 0,
            'entanglements_created': 0,
            'entanglements_broken': 0
        }
        
        logger.info("State registry reset")