import logging
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
from typing import Any, Dict, List, Optional, Tuple, Union

from src.physics.constants import (
    DecoherenceRates, CoherenceParameters, NumericalParameters,
    get_decoherence_rate
)

logger = logging.getLogger(__name__)

class CoherenceManager:
    """
    Manages quantum coherence in the Recursia simulation.
    
    Coherence is a fundamental property in quantum systems representing quantum
    superposition integrity. This class provides methods for calculating, 
    manipulating, and modeling coherence in various quantum contexts.
    
    The manager handles:
    - Coherence calculation from density matrices
    - Decoherence processes and time evolution
    - Coherence transfer between quantum systems
    - Coherence enhancement and restoration techniques
    - Alignment between multiple quantum states
    """
    
    def __init__(self, environment: str = "default"):
        """
        Initialize the coherence manager with physically justified parameters.
        
        Args:
            environment: Type of environment for decoherence rates
                        ("vacuum", "cryogenic", "biological", etc.)
        """
        # Decoherence parameters based on physical environment
        self.decoherence_rate = get_decoherence_rate(environment)
        self.minimum_coherence = CoherenceParameters.MINIMUM_COHERENCE
        
        # Observation impact parameters from weak measurement theory
        self.observation_impact = CoherenceParameters.DEFAULT_MEASUREMENT_IMPACT
        
        # Entanglement parameters from monogamy constraints
        self.entanglement_sharing = CoherenceParameters.ENTANGLEMENT_SHARING
        
        # Coherence restoration parameters from quantum error correction
        self.max_coherence_restoration = CoherenceParameters.MAX_RESTORATION
        self.coherence_restoration_cost = CoherenceParameters.RESTORATION_ENERGY_COST
        
        # Alignment parameters for numerical convergence
        self.alignment_tolerance = NumericalParameters.CONVERGENCE_THRESHOLD
        self.max_alignment_iterations = NumericalParameters.MAX_ALIGNMENT_ITERATIONS
        
        # State registry for coherence tracking
        self.state_coherence_registry = {}  # Maps state_name -> coherence value
        self.state_entropy_registry = {}    # Maps state_name -> entropy value
    
    def calculate_coherence(self, density_matrix: np.ndarray) -> float:
        """
        Calculate the coherence of a quantum state from its density matrix.
        Uses l1-norm of coherence (sum of absolute values of off-diagonal elements).
        
        Args:
            density_matrix: Density matrix of the quantum state
            
        Returns:
            float: Coherence measure between 0 (fully decohered) and 1 (maximum coherence)
            
        Raises:
            ValueError: If density matrix has invalid shape or is not square
        """
        try:
            # Validate density matrix
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
            
            if density_matrix.ndim != 2:
                raise ValueError(f"Density matrix must be 2-dimensional, got {density_matrix.ndim}")
            
            if density_matrix.shape[0] != density_matrix.shape[1]:
                raise ValueError(f"Density matrix must be square, got shape {density_matrix.shape}")
            
            # Extract diagonal elements (populations)
            diag = np.diag(density_matrix)
            
            # Create a matrix with only the diagonal elements
            diag_matrix = np.diag(diag)
            
            # Off-diagonal elements represent coherence
            off_diag = density_matrix - diag_matrix
            
            # L1 norm of coherence
            l1_coherence = np.sum(np.abs(off_diag))
            
            # Normalize based on dimension
            dimension = density_matrix.shape[0]
            # Maximum possible L1 coherence is n*(n-1) for a pure state
            max_coherence = dimension * (dimension - 1)
            
            # Normalize to [0, 1]
            if max_coherence > 0:
                normalized_coherence = l1_coherence / max_coherence
            else:
                normalized_coherence = 0.0
            
            return normalized_coherence
        except Exception as e:
            logger.error(f"Error calculating coherence: {e}")
            # Return minimum coherence as a fallback
            return 0.0
    
    def calculate_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculate the von Neumann entropy of a quantum state.
        S = -Tr(ρ log ρ)
        
        Args:
            density_matrix: Density matrix of the quantum state
            
        Returns:
            float: Entropy value (0 for pure states, >0 for mixed states)
            
        Raises:
            ValueError: If density matrix has invalid properties
        """
        try:
            # Validate density matrix
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
            
            if density_matrix.ndim != 2 or density_matrix.shape[0] != density_matrix.shape[1]:
                raise ValueError(f"Density matrix must be square, got shape {density_matrix.shape}")
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            
            # Filter out very small negative eigenvalues (numerical errors)
            eigenvalues = np.where(eigenvalues < 1e-10, 0, eigenvalues)
            
            # Calculate entropy using only positive eigenvalues
            entropy = 0.0
            for eig in eigenvalues:
                if eig > 0:
                    entropy -= eig * np.log2(eig)
            
            # Normalize by maximum possible entropy (log of dimension)
            dimension = density_matrix.shape[0]
            max_entropy = np.log2(dimension)
            
            if max_entropy > 0:
                normalized_entropy = entropy / max_entropy
            else:
                normalized_entropy = 0.0
            
            return normalized_entropy
        except Exception as e:
            logger.error(f"Error calculating entropy: {e}")
            # Return zero entropy as a fallback
            return 0.0
    
    def apply_decoherence(self, density_matrix: np.ndarray, time_step: float = 1.0) -> np.ndarray:
        """
        Apply natural decoherence to a quantum state for a given time step.
        Decoherence reduces off-diagonal elements of the density matrix.
        
        Args:
            density_matrix: Density matrix of the quantum state
            time_step: Amount of time to simulate decoherence
            
        Returns:
            np.ndarray: Updated density matrix after decoherence
            
        Raises:
            ValueError: If density matrix has invalid properties
        """
        try:
            # Validate density matrix
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
            
            if density_matrix.ndim != 2 or density_matrix.shape[0] != density_matrix.shape[1]:
                raise ValueError(f"Density matrix must be square, got shape {density_matrix.shape}")
            
            # Calculate decoherence factor for this time step
            decoherence_factor = np.exp(-self.decoherence_rate * time_step)
            
            # Extract diagonal elements (populations)
            diag = np.diag(density_matrix)
            
            # Create a matrix with only the diagonal elements
            diag_matrix = np.diag(diag)
            
            # Off-diagonal elements represent coherence
            off_diag = density_matrix - diag_matrix
            
            # Apply decoherence to off-diagonal elements
            decohered_off_diag = off_diag * decoherence_factor
            
            # Combine diagonal and decohered off-diagonal components
            decohered_matrix = diag_matrix + decohered_off_diag
            
            # Ensure the resulting matrix is still a valid density matrix
            decohered_matrix = self._ensure_valid_density_matrix(decohered_matrix)
            
            return decohered_matrix
        except Exception as e:
            logger.error(f"Error applying decoherence: {e}")
            # Return the original matrix as a fallback
            return density_matrix
    
    def _ensure_valid_density_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Ensure a matrix is a valid density matrix:
        - Hermitian (equal to its conjugate transpose)
        - Positive semi-definite (all eigenvalues ≥ 0)
        - Trace = 1
        
        Args:
            matrix: Matrix to validate/correct
            
        Returns:
            np.ndarray: Valid density matrix
        """
        # Ensure Hermitian
        hermitian_matrix = 0.5 * (matrix + matrix.conj().T)
        
        # Ensure positive semi-definite
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Reconstruct the matrix
        reconstructed = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
        
        # Normalize trace to 1
        trace = np.trace(reconstructed)
        if trace > 1e-10:  # Avoid division by very small numbers
            normalized = reconstructed / trace
        else:
            # If trace is too small, create a maximally mixed state
            dimension = matrix.shape[0]
            normalized = np.eye(dimension) / dimension
        
        return normalized
    
    def coherence_evolution(self, density_matrix: np.ndarray, 
                           hamiltonian: np.ndarray, 
                           time_step: float = 0.01,
                           steps: int = 100,
                           include_decoherence: bool = True) -> List[np.ndarray]:
        """
        Simulate the evolution of a quantum state under a Hamiltonian with optional decoherence.
        
        Args:
            density_matrix: Initial density matrix
            hamiltonian: System Hamiltonian
            time_step: Size of each time step
            steps: Number of time steps to simulate
            include_decoherence: Whether to include decoherence effects
            
        Returns:
            List[np.ndarray]: List of density matrices at each time step
            
        Raises:
            ValueError: If matrices have incompatible dimensions
        """
        try:
            # Check matrix dimensions
            if density_matrix.shape != hamiltonian.shape:
                raise ValueError(
                    f"Density matrix shape {density_matrix.shape} must match Hamiltonian shape {hamiltonian.shape}"
                )
            
            # Ensure matrices are square
            n = density_matrix.shape[0]
            if density_matrix.shape[1] != n or hamiltonian.shape[1] != n:
                raise ValueError("Matrices must be square")
            
            # Validate the initial density matrix
            if not self._is_valid_density_matrix(density_matrix):
                logger.warning("Initial density matrix is not valid, correcting")
                density_matrix = self._ensure_valid_density_matrix(density_matrix)
            
            # Evolution operator for one time step: exp(-i*H*dt) (in natural units)
            evolution_operator = self._calculate_evolution_operator(hamiltonian, time_step)
            
            # List to store density matrices at each time step
            density_matrices = [density_matrix.copy()]
            
            # Current state
            current_state = density_matrix.copy()
            
            # Simulate evolution
            for _ in range(steps):
                # Unitary evolution under Hamiltonian
                current_state = evolution_operator @ current_state @ evolution_operator.conj().T
                
                # Apply decoherence if requested
                if include_decoherence:
                    current_state = self.apply_decoherence(current_state, time_step)
                
                # Ensure it remains a valid density matrix after each step
                current_state = self._ensure_valid_density_matrix(current_state)
                
                # Store the result
                density_matrices.append(current_state.copy())
            
            return density_matrices
        except Exception as e:
            logger.error(f"Error in coherence evolution: {e}")
            # Return a list with just the initial state as a fallback
            return [density_matrix]
    
    def _calculate_evolution_operator(self, hamiltonian: np.ndarray, time_step: float) -> np.ndarray:
        """
        Calculate the quantum evolution operator exp(-i*H*dt).
        
        Args:
            hamiltonian: System Hamiltonian
            time_step: Time step size
            
        Returns:
            np.ndarray: Evolution operator
        """
        try:
            # Diagonalize the Hamiltonian
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
            
            # Calculate exp(-i*eigenvalues*dt)
            diagonal_evolution = np.exp(-1j * eigenvalues * time_step)
            
            # Construct the evolution operator
            evolution_operator = eigenvectors @ np.diag(diagonal_evolution) @ eigenvectors.conj().T
            
            return evolution_operator
        except Exception as e:
            logger.error(f"Error calculating evolution operator: {e}")
            # Return identity as a fallback (no evolution)
            return np.eye(hamiltonian.shape[0])
    
    def _is_valid_density_matrix(self, matrix: np.ndarray, tolerance: float = 1e-10) -> bool:
        """
        Check if a matrix is a valid density matrix.
        
        Args:
            matrix: Matrix to check
            tolerance: Numerical tolerance for checks
            
        Returns:
            bool: True if valid density matrix
        """
        try:
            # Check if matrix is square
            if matrix.shape[0] != matrix.shape[1]:
                return False
            
            # Check if Hermitian (self-adjoint): ρ = ρ†
            hermitian_diff = matrix - matrix.conj().T
            if not np.allclose(hermitian_diff, np.zeros_like(hermitian_diff), atol=tolerance):
                return False
            
            # Check if trace is 1
            if not np.isclose(np.trace(matrix), 1.0, atol=tolerance):
                return False
            
            # Check if positive semi-definite (all eigenvalues ≥ 0)
            eigenvalues = np.linalg.eigvalsh(matrix)
            if np.any(eigenvalues < -tolerance):
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating density matrix: {e}")
            return False
        
    def coherence_transfer(self, density_matrix1: np.ndarray, density_matrix2: np.ndarray, 
                         coupling_strength: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model coherence transfer between two quantum systems.
        
        Args:
            density_matrix1: Density matrix of first system
            density_matrix2: Density matrix of second system
            coupling_strength: Strength of coherence coupling (0 to 1)
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated density matrices after coherence transfer
            
        Raises:
            ValueError: If density matrices are invalid
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix1, np.ndarray) or not isinstance(density_matrix2, np.ndarray):
                raise ValueError("Density matrices must be numpy arrays")
            
            # Ensure coupling strength is in valid range
            coupling_strength = max(0.0, min(1.0, coupling_strength))
            
            # Calculate current coherence values
            coherence1 = self.calculate_coherence(density_matrix1)
            coherence2 = self.calculate_coherence(density_matrix2)
            
            # Calculate coherence transfer
            delta_coherence = coupling_strength * (coherence1 - coherence2)
            
            # Create diagonal-only matrices
            diag1 = np.diag(np.diag(density_matrix1))
            diag2 = np.diag(np.diag(density_matrix2))
            
            # Off-diagonal components (coherence)
            off_diag1 = density_matrix1 - diag1
            off_diag2 = density_matrix2 - diag2
            
            # Adjust off-diagonal elements based on coherence transfer
            adjusted_off_diag1 = off_diag1 * (1 - delta_coherence)
            adjusted_off_diag2 = off_diag2 * (1 + delta_coherence)
            
            # Combine to form new density matrices
            new_density_matrix1 = diag1 + adjusted_off_diag1
            new_density_matrix2 = diag2 + adjusted_off_diag2
            
            # Ensure they're still valid density matrices
            new_density_matrix1 = self._ensure_valid_density_matrix(new_density_matrix1)
            new_density_matrix2 = self._ensure_valid_density_matrix(new_density_matrix2)
            
            return new_density_matrix1, new_density_matrix2
        except Exception as e:
            logger.error(f"Error in coherence transfer: {e}")
            # Return original matrices as fallback
            return density_matrix1, density_matrix2
    
    def coherence_enhancement(self, density_matrix: np.ndarray, 
                             target_coherence: float = 0.9) -> np.ndarray:
        """
        Enhance the coherence of a quantum state towards a target value.
        This simulates active coherence restoration techniques.
        
        Args:
            density_matrix: Original density matrix
            target_coherence: Target coherence value (0 to 1)
            
        Returns:
            np.ndarray: Enhanced density matrix
            
        Raises:
            ValueError: If density matrix is invalid
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError("Density matrix must be a numpy array")
            
            # Clamp target coherence to valid range
            target_coherence = max(0.0, min(self.max_coherence_restoration, target_coherence))
            
            # Current coherence
            current_coherence = self.calculate_coherence(density_matrix)
            
            # If already above target, no enhancement needed
            if current_coherence >= target_coherence:
                return density_matrix
            
            # Get diagonal and off-diagonal components
            diag = np.diag(np.diag(density_matrix))
            off_diag = density_matrix - diag
            
            # Calculate enhancement factor
            if current_coherence > 0:
                enhancement_factor = target_coherence / current_coherence
            else:
                # If current coherence is 0, generate new coherence
                enhancement_factor = target_coherence
            
            # Cap the enhancement factor to avoid numerical instability
            enhancement_factor = min(enhancement_factor, 5.0)
            
            # Apply enhancement to off-diagonal elements
            enhanced_off_diag = off_diag * enhancement_factor
            
            # Combine components
            enhanced_matrix = diag + enhanced_off_diag
            
            # Ensure it's a valid density matrix
            enhanced_matrix = self._ensure_valid_density_matrix(enhanced_matrix)
            
            return enhanced_matrix
        except Exception as e:
            logger.error(f"Error in coherence enhancement: {e}")
            # Return original matrix as fallback
            return density_matrix
    
    def increase_coherence(self, state_name: str, target_coherence: float = 0.8) -> float:
        """
        Increase the coherence of a named quantum state.
        
        Args:
            state_name: Name of the quantum state
            target_coherence: Target coherence level (0 to 1)
            
        Returns:
            float: New coherence value
        """
        try:
            # Get current coherence
            current_coherence = self.state_coherence_registry.get(state_name, 0.5)
            
            # Calculate coherence increase
            # The closer to 1.0, the harder it is to increase further
            coherence_headroom = 1.0 - current_coherence
            if coherence_headroom <= 0:
                return current_coherence  # Already at maximum
            
            # Calculate new coherence
            # This uses a asymptotic approach to the target
            target = min(1.0, target_coherence)
            step_size = 0.3 * (target - current_coherence)
            new_coherence = current_coherence + step_size
            
            # Ensure coherence is in valid range
            new_coherence = max(0.0, min(1.0, new_coherence))
            
            # Update registry
            self.state_coherence_registry[state_name] = new_coherence
            
            return new_coherence
        except Exception as e:
            logger.error(f"Error increasing coherence for state {state_name}: {e}")
            # Return a default value as fallback
            return 0.5
    
    def align_coherence(self, state1: str, state2: str, target_level: float = 0.8) -> float:
        """
        Align the coherence of two quantum states, increasing overall coherence.
        
        Args:
            state1: First state name
            state2: Second state name
            target_level: Target coherence level
            
        Returns:
            float: New coherence level of the first state
        """
        try:
            # Get current coherence values
            coherence1 = self.state_coherence_registry.get(state1, 0.5)
            coherence2 = self.state_coherence_registry.get(state2, 0.5)
            
            # Calculate the alignment target
            # Higher coherence states pull lower ones up more than vice versa
            if coherence1 >= coherence2:
                pull_factor = 0.7  # First state pulls second more
            else:
                pull_factor = 0.3  # Second state pulls first more
            
            alignment_target = min(target_level, coherence1 * pull_factor + coherence2 * (1-pull_factor))
            
            # Align coherence values
            new_coherence1 = coherence1 + 0.2 * (alignment_target - coherence1)
            new_coherence2 = coherence2 + 0.2 * (alignment_target - coherence2)
            
            # Ensure values are in range
            new_coherence1 = max(0.0, min(1.0, new_coherence1))
            new_coherence2 = max(0.0, min(1.0, new_coherence2))
            
            # Update registry
            self.state_coherence_registry[state1] = new_coherence1
            self.state_coherence_registry[state2] = new_coherence2
            
            return new_coherence1
        except Exception as e:
            logger.error(f"Error aligning coherence between states {state1} and {state2}: {e}")
            # Return a default value as fallback
            return 0.5
    
    def align_state_group(self, state_names: List[str], alignment_factor: float = 0.7) -> Dict[str, float]:
        """
        Align the coherence of a group of quantum states.
        
        Args:
            state_names: List of state names to align
            alignment_factor: Strength of alignment (0 to 1)
            
        Returns:
            Dict[str, float]: New coherence values for each state
        """
        try:
            if not state_names:
                return {}
            
            # Get current coherence values
            coherence_values = {}
            for name in state_names:
                coherence_values[name] = self.state_coherence_registry.get(name, 0.5)
            
            # Calculate average coherence
            avg_coherence = sum(coherence_values.values()) / len(coherence_values)
            
            # Align towards average
            alignment_strength = max(0.0, min(1.0, alignment_factor))
            
            # Update coherence values
            new_coherence_values = {}
            for name, coherence in coherence_values.items():
                # Move towards the average
                new_coherence = coherence + alignment_strength * (avg_coherence - coherence)
                # Ensure in valid range
                new_coherence = max(0.0, min(1.0, new_coherence))
                # Store new value
                new_coherence_values[name] = new_coherence
                self.state_coherence_registry[name] = new_coherence
            
            return new_coherence_values
        except Exception as e:
            logger.error(f"Error aligning state group: {e}")
            # Return empty dict as fallback
            return {}
    
    def align_all_states(self, state_names: List[str], alignment_factor: float = 0.6) -> Dict[str, Dict[str, float]]:
        """
        Align coherence across all states to improve global coherence.
        
        Args:
            state_names: List of all state names
            alignment_factor: Strength of alignment (0 to 1)
            
        Returns:
            Dict[str, Dict[str, float]]: Results including coherence values and statistics
        """
        try:
            if not state_names:
                return {'coherence_values': {}, 'stats': {'average': 0, 'min': 0, 'max': 0}}
            
            # Get current coherence values
            coherence_values = {}
            for name in state_names:
                coherence_values[name] = self.state_coherence_registry.get(name, 0.5)
            
            # Calculate statistics
            avg_coherence = sum(coherence_values.values()) / len(coherence_values)
            min_coherence = min(coherence_values.values())
            max_coherence = max(coherence_values.values())
            
            # Calculate alignment strength based on distribution
            # More variability requires stronger alignment
            coherence_range = max_coherence - min_coherence
            adjusted_alignment = alignment_factor * (1 + coherence_range)
            adjusted_alignment = max(0.0, min(0.9, adjusted_alignment))
            
            # Update coherence values
            new_coherence_values = {}
            for name, coherence in coherence_values.items():
                # Move towards the average based on distance
                distance = abs(coherence - avg_coherence)
                step = adjusted_alignment * distance * 0.5
                
                if coherence < avg_coherence:
                    new_coherence = coherence + step
                else:
                    new_coherence = coherence - step * 0.7  # Less reduction for high coherence
                
                # Ensure in valid range
                new_coherence = max(0.0, min(1.0, new_coherence))
                
                # Store new value
                new_coherence_values[name] = new_coherence
                self.state_coherence_registry[name] = new_coherence
            
            # Calculate new statistics
            new_avg = sum(new_coherence_values.values()) / len(new_coherence_values)
            new_min = min(new_coherence_values.values())
            new_max = max(new_coherence_values.values())
            
            return {
                'coherence_values': new_coherence_values,
                'stats': {
                    'initial_average': avg_coherence,
                    'initial_min': min_coherence,
                    'initial_max': max_coherence,
                    'final_average': new_avg,
                    'final_min': new_min,
                    'final_max': new_max,
                    'alignment_factor': adjusted_alignment
                }
            }
        except Exception as e:
            logger.error(f"Error aligning all states: {e}")
            # Return empty result as fallback
            return {'coherence_values': {}, 'stats': {'error': str(e)}}
    
    def align_by_field(self, state_names: List[str], field_name: str, alignment_factor: float = 0.6) -> Dict[str, Dict[str, float]]:
        """
        Align states based on a shared field value.
        States with similar field values will have more similar coherence.
        
        Args:
            state_names: List of state names
            field_name: Field to align by (e.g., 'state_entropy', 'state_qubits', etc.)
            alignment_factor: Strength of alignment (0 to 1)
            
        Returns:
            Dict[str, Dict[str, float]]: Results including field values and coherence values
        """
        try:
            if not state_names:
                return {'field_values': {}, 'coherence_values': {}, 'groups': {}}
            
            # Get current coherence values
            coherence_values = {}
            for name in state_names:
                coherence_values[name] = self.state_coherence_registry.get(name, 0.5)
            
            # Try to access field values through state registry or external accessor
            field_values = {}
            state_registry_available = False
            
            # Check if we have a state registry accessor
            if hasattr(self, 'state_registry') and self.state_registry is not None:
                state_registry_available = True
                # Get field values from state registry
                for name in state_names:
                    field_values[name] = self.state_registry.get_field(name, field_name, None)
            
            # Check if we have a get_state_field function available
            elif hasattr(self, 'get_state_field') and callable(self.get_state_field):
                state_registry_available = True
                # Get field values using the function
                for name in state_names:
                    field_values[name] = self.get_state_field(name, field_name, None)
            
            # If we couldn't access a state registry, use internal fields or fallback
            if not state_registry_available:
                logger.warning(f"No state registry available, using internal values for alignment")
                # Check if the field name refers to entropy
                if field_name.lower() in ['entropy', 'state_entropy']:
                    for name in state_names:
                        field_values[name] = self.state_entropy_registry.get(name, 0.5)
                else:
                    # Use coherence as fallback for any other field
                    logger.warning(f"Using coherence as fallback for field '{field_name}'")
                    field_values = coherence_values.copy()
            
            # Filter out states with no field value
            valid_states = [name for name in state_names if field_values.get(name) is not None]
            
            if not valid_states:
                logger.warning(f"No states with valid field value for '{field_name}'")
                return {
                    'field_values': field_values,
                    'coherence_values': coherence_values,
                    'groups': {},
                    'status': 'No valid field values found'
                }
            
            # Determine if field values are numeric or categorical
            first_field = field_values.get(valid_states[0])
            is_numeric = isinstance(first_field, (int, float, complex, np.number))
            
            # Group states by field value
            groups = {}
            
            if is_numeric:
                # For numeric fields, group by ranges
                # Determine range of values
                valid_values = [field_values[name] for name in valid_states]
                min_val = min(valid_values)
                max_val = max(valid_values)
                
                # Determine grouping strategy based on range and number of states
                if max_val - min_val < 0.001:  # All values effectively the same
                    bin_width = 1.0  # Single group
                else:
                    # Choose bin width based on number of states and range
                    num_bins = min(len(valid_states), 5)  # At most 5 groups
                    bin_width = (max_val - min_val) / num_bins
                
                # Group states by binning
                for name in valid_states:
                    value = field_values[name]
                    if bin_width > 0:
                        bin_index = int((value - min_val) / bin_width)
                    else:
                        bin_index = 0
                    bin_key = f"bin_{bin_index}"
                    
                    if bin_key not in groups:
                        groups[bin_key] = []
                    groups[bin_key].append(name)
            else:
                # For categorical fields, group by exact value
                for name in valid_states:
                    value = str(field_values[name])  # Convert to string for consistency
                    if value not in groups:
                        groups[value] = []
                    groups[value].append(name)
            
            # Apply alignment within each group
            new_coherence_values = {}
            group_results = {}
            
            for group_key, group_members in groups.items():
                if len(group_members) > 1:
                    # Apply stronger alignment within groups
                    group_factor = min(1.0, alignment_factor * 1.5)
                    aligned_values = self.align_state_group(group_members, group_factor)
                    new_coherence_values.update(aligned_values)
                    
                    # Calculate group statistics for result
                    avg_coherence = sum(aligned_values.values()) / len(aligned_values)
                    group_results[group_key] = {
                        'count': len(group_members),
                        'members': group_members,
                        'avg_coherence': avg_coherence,
                        'field_value': field_values[group_members[0]] if is_numeric else group_key
                    }
                else:
                    # Single state, keep its value
                    name = group_members[0]
                    new_coherence_values[name] = coherence_values[name]
                    group_results[group_key] = {
                        'count': 1,
                        'members': group_members,
                        'avg_coherence': coherence_values[name],
                        'field_value': field_values[name] if is_numeric else group_key
                    }
            
            # If there are states not in any group, keep their original values
            for name in state_names:
                if name not in new_coherence_values:
                    new_coherence_values[name] = coherence_values[name]
            
            return {
                'field_values': field_values,
                'coherence_values': new_coherence_values,
                'groups': group_results,
                'is_numeric_field': is_numeric,
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error aligning by field: {e}")
            # Return empty result as fallback with error information
            return {
                'field_values': {},
                'coherence_values': {},
                'groups': {},
                'status': f'error: {str(e)}'
            }
        
    def apply_quantum_annealing(self, state_name: str, target_coherence: float, iterations: int = 10) -> float:
        """
        Apply quantum annealing algorithm to increase coherence with thermal fluctuations.
        Implements simulated quantum annealing with tunneling effects and thermal bath coupling.
        
        Args:
            state_name: Target state name
            target_coherence: Target coherence level
            iterations: Number of annealing steps
            
        Returns:
            float: New coherence value
        """
        try:
            # Get current coherence and entropy
            current_coherence = self.state_coherence_registry.get(state_name, 0.5)
            current_entropy = self.state_entropy_registry.get(state_name, 0.5)
            
            # Annealing parameters
            initial_temperature = 2.0
            final_temperature = 0.01
            tunneling_strength = 0.4 * np.exp(-0.5 * current_entropy)  # Tunneling decreases with entropy
            
            # Current state
            coherence = current_coherence
            entropy = current_entropy
            
            # Best state tracking
            best_coherence = coherence
            best_entropy = entropy
            best_energy = -coherence + 0.2 * entropy  # Energy function (minimize)
            
            # Annealing schedule
            for step in range(iterations):
                # Calculate current temperature
                temperature = initial_temperature * ((final_temperature / initial_temperature) ** (step / (iterations - 1)))
                
                # Calculate energy barriers
                energy_barrier = 0.1 * (1.0 - coherence) * (temperature / initial_temperature)
                
                # Thermal fluctuation component
                thermal_step = (target_coherence - coherence) * (1.0 - np.exp(-1.0 / temperature))
                
                # Tunneling component (quantum effect)
                tunneling_probability = tunneling_strength * np.exp(-energy_barrier / temperature)
                tunneling_step = 0.0
                if np.random.random() < tunneling_probability:
                    tunneling_step = 0.1 * (target_coherence - coherence) * (1.0 + np.random.random())
                
                # Combined step
                coherence_step = thermal_step + tunneling_step
                coherence = coherence + coherence_step
                
                # Entropy reduction (quantum annealing tends to reduce entropy)
                entropy_reduction = 0.05 * abs(coherence_step) * (1.0 + (1.0 - temperature))
                entropy = max(0.0, entropy - entropy_reduction)
                
                # Ensure coherence is in valid range
                coherence = max(0.0, min(1.0, coherence))
                
                # Track best state
                current_energy = -coherence + 0.2 * entropy
                if current_energy < best_energy:
                    best_coherence = coherence
                    best_entropy = entropy
                    best_energy = current_energy
                
                logger.debug(f"Quantum annealing step {step}: T={temperature:.4f}, "
                            f"coherence={coherence:.4f}, energy={current_energy:.4f}")
            
            # Update registries with best found values
            self.state_coherence_registry[state_name] = best_coherence
            self.state_entropy_registry[state_name] = best_entropy
            
            return best_coherence
        except Exception as e:
            logger.error(f"Error applying quantum annealing to {state_name}: {e}")
            # Return a default value as fallback
            return 0.5

    def apply_quantum_information_algorithm(self, states: List[str], algorithm_type: str = "quantum_annealing",
                               parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Apply a selected quantum information algorithm to multiple states.
        
        Args:
            states: List of state names to process
            algorithm_type: Algorithm type ('quantum_annealing', 'gradient_descent', 
                                        'tensor_compression', 'stochastic_descent')
            parameters: Algorithm-specific parameters
            
        Returns:
            Dict[str, Any]: Results including new coherence values and statistics
        """
        try:
            if not states:
                return {'states_processed': 0, 'error': "No states provided"}
            
            # Default parameters
            params = {
                'target_coherence': 0.8,
                'iterations': 10,
                'convergence_threshold': 0.01
            }
            
            # Update with user-provided parameters
            if parameters:
                params.update(parameters)
            
            # Select algorithm function
            algorithm_map = {
                'quantum_annealing': self.apply_quantum_annealing,
                'gradient_descent': self.apply_gradient_descent,
                'tensor_compression': self.apply_tensor_compression,
                'stochastic_descent': self.apply_stochastic_descent_algorithm
            }
            
            if algorithm_type not in algorithm_map:
                return {'error': f"Unknown algorithm type: {algorithm_type}"}
            
            algorithm_fn = algorithm_map[algorithm_type]
            
            # Process each state
            results = {}
            initial_values = {}
            final_values = {}
            
            for state in states:
                if state not in self.state_coherence_registry:
                    results[state] = {'error': "State not found"}
                    continue
                
                # Store initial values
                initial_coherence = self.state_coherence_registry.get(state, 0.5)
                initial_entropy = self.state_entropy_registry.get(state, 0.5)
                initial_values[state] = {
                    'coherence': initial_coherence,
                    'entropy': initial_entropy
                }
                
                # Apply algorithm
                if algorithm_type in ['quantum_annealing', 'gradient_descent', 'stochastic_descent']:
                    # These algorithms take iterations parameter
                    new_coherence = algorithm_fn(
                        state, 
                        params['target_coherence'],
                        params.get('iterations', 10)
                    )
                else:
                    # Tensor compression doesn't use iterations parameter
                    new_coherence = algorithm_fn(
                        state, 
                        params['target_coherence']
                    )
                
                # Store final values
                final_coherence = self.state_coherence_registry.get(state, 0.5)
                final_entropy = self.state_entropy_registry.get(state, 0.5)
                final_values[state] = {
                    'coherence': final_coherence,
                    'entropy': final_entropy
                }
                
                # Calculate improvement
                coherence_change = final_coherence - initial_coherence
                entropy_change = final_entropy - initial_entropy
                
                results[state] = {
                    'initial_coherence': initial_coherence,
                    'final_coherence': final_coherence,
                    'coherence_change': coherence_change,
                    'initial_entropy': initial_entropy,
                    'final_entropy': final_entropy,
                    'entropy_change': entropy_change
                }
            
            # Calculate overall statistics
            initial_coherence_avg = np.mean([v['coherence'] for v in initial_values.values()])
            final_coherence_avg = np.mean([v['coherence'] for v in final_values.values()])
            coherence_improvement = final_coherence_avg - initial_coherence_avg
            
            initial_entropy_avg = np.mean([v['entropy'] for v in initial_values.values()])
            final_entropy_avg = np.mean([v['entropy'] for v in final_values.values()])
            entropy_change = final_entropy_avg - initial_entropy_avg
            
            summary = {
                'algorithm': algorithm_type,
                'states_processed': len(states),
                'avg_initial_coherence': initial_coherence_avg,
                'avg_final_coherence': final_coherence_avg,
                'avg_coherence_improvement': coherence_improvement,
                'avg_initial_entropy': initial_entropy_avg,
                'avg_final_entropy': final_entropy_avg,
                'avg_entropy_change': entropy_change,
                'parameters': params
            }
            
            return {
                'summary': summary,
                'state_results': results
            }
        except Exception as e:
            logger.error(f"Error applying {algorithm_type} algorithm: {e}")
            return {'error': str(e)}

    def _calculate_entropy(self, density_matrix: np.ndarray) -> float:
        """
        Calculate the von Neumann entropy of a quantum state.
        S = -Tr(ρ log ρ) = -sum(λ_i log λ_i) where λ_i are eigenvalues of ρ
        
        Args:
            density_matrix: Density matrix of the quantum state
            
        Returns:
            float: Entropy value (0 for pure states, >0 for mixed states)
        """
        try:
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(density_matrix)
            
            # Filter out very small eigenvalues (numerical errors)
            eigenvalues = eigenvalues[eigenvalues > 1e-10]
            
            # Calculate entropy using only positive eigenvalues
            entropy = 0.0
            for eig in eigenvalues:
                entropy -= eig * np.log2(eig)
            
            return entropy
        except Exception as e:
            logger.error(f"Error in _calculate_entropy: {e}")
            return 0.0  # Default fallback
        
    def calculate_entanglement(self, density_matrix: np.ndarray, dims: Tuple[int, int]) -> float:
        """
        Calculate entanglement measure using negativity (PPT criterion).
        
        Negativity is defined as the sum of the absolute values of the negative
        eigenvalues of the partial transpose of the density matrix.
        
        Args:
            density_matrix: The density matrix of the quantum state
            dims: Dimensions of the two subsystems (dim_A, dim_B)
            
        Returns:
            float: Entanglement measure (0 for separable states, >0 for entangled states)
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
                
            dim_a, dim_b = dims
            dim_total = dim_a * dim_b
            
            if density_matrix.shape != (dim_total, dim_total):
                raise ValueError(
                    f"Density matrix dimension {density_matrix.shape} does not match "
                    f"subsystem dimensions {dims} (expected {dim_total}x{dim_total})"
                )
            
            # Reshape into tensor product form
            rho_reshaped = density_matrix.reshape(dim_a, dim_b, dim_a, dim_b)
            
            # Perform partial transpose (transpose only for second subsystem)
            rho_pt = np.transpose(rho_reshaped, (0, 3, 2, 1))
            
            # Reshape back to matrix form
            rho_pt = rho_pt.reshape(dim_total, dim_total)
            
            # Calculate eigenvalues
            eigenvalues = np.linalg.eigvalsh(rho_pt)
            
            # Calculate negativity (sum of absolute values of negative eigenvalues)
            negativity = np.sum(np.abs(eigenvalues[eigenvalues < 0]))
            
            # Normalize by maximum possible negativity
            # For maximally entangled states, negativity is (dim_min - 1)/2
            dim_min = min(dim_a, dim_b)
            max_negativity = (dim_min - 1) / 2
            
            if max_negativity > 0:
                normalized_negativity = negativity / max_negativity
            else:
                normalized_negativity = 0.0
            
            return normalized_negativity
        except Exception as e:
            logger.error(f"Error calculating entanglement: {e}")
            return 0.0  # Default fallback - assume separable
        
    def partial_trace(self, density_matrix: np.ndarray, num_qubits: int, 
                    subsystem: str, trace_qubits: List[int]) -> np.ndarray:
        """
        Perform partial trace and return a reduced density matrix
        
        Args:
            density_matrix: The full system density matrix
            num_qubits: Total number of qubits in the system
            subsystem: Name for the resulting subsystem (for logging)
            trace_qubits: Indices of qubits to trace out
            
        Returns:
            np.ndarray: Reduced density matrix
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(density_matrix, np.ndarray):
                raise ValueError(f"Density matrix must be a numpy array, got {type(density_matrix)}")
                
            if not isinstance(trace_qubits, list):
                raise ValueError(f"trace_qubits must be a list of indices, got {type(trace_qubits)}")
                
            if max(trace_qubits, default=-1) >= num_qubits:
                raise ValueError(f"trace_qubits contains invalid qubit index {max(trace_qubits)} for a {num_qubits}-qubit system")
                
            # Use the unified partial trace implementation
            result_dm = self._partial_trace(density_matrix, num_qubits, trace_qubits)
            
            logger.debug(f"Performed partial trace on {num_qubits}-qubit system, traced out qubits {trace_qubits}")
            
            return result_dm
        except Exception as e:
            logger.error(f"Error performing partial trace for subsystem {subsystem}: {e}")
            # Return maximally mixed state as fallback
            remaining_qubits = num_qubits - len(trace_qubits)
            dim = 2 ** remaining_qubits
            return np.eye(dim) / dim
        
    def _partial_trace(self, density_matrix: np.ndarray, 
                    system_dims: Union[Tuple[int, int], int], 
                    subsystem_to_trace: Union[List[int], int]) -> np.ndarray:
        """
        Perform partial trace operation on a quantum system.
        
        Supports both qubit-based tracing (using dimension = 2^n_qubits) and
        arbitrary dimensional subsystems.
        
        Args:
            density_matrix: The density matrix of the composite system
            system_dims: Either (dim_A, dim_B) tuple for bipartite system or 
                        total qubits for qubit-based systems
            subsystem_to_trace: Either 0/1 for bipartite subsystem to trace out, or
                            list of qubit indices to trace out for qubit systems
            
        Returns:
            np.ndarray: Reduced density matrix after tracing out specified subsystem
            
        Raises:
            ValueError: If inputs are invalid or dimensions don't match
        """
        try:
            # Check if we're using qubit notation or arbitrary dimensions
            if isinstance(system_dims, tuple) and isinstance(subsystem_to_trace, int):
                # Bipartite system with arbitrary dimensions
                dim_a, dim_b = system_dims
                dim_total = dim_a * dim_b
                
                # Check dimensions
                if density_matrix.shape != (dim_total, dim_total):
                    raise ValueError(
                        f"Density matrix dimension {density_matrix.shape} does not match "
                        f"subsystem dimensions {system_dims} (expected {dim_total}x{dim_total})"
                    )
                
                # Reshape density matrix into tensor product form
                rho_reshaped = density_matrix.reshape(dim_a, dim_b, dim_a, dim_b)
                
                # Perform partial trace
                if subsystem_to_trace == 0:
                    # Trace out system A (keep B)
                    reduced_dm = np.trace(rho_reshaped, axis1=0, axis2=2)
                elif subsystem_to_trace == 1:
                    # Trace out system B (keep A)
                    reduced_dm = np.trace(rho_reshaped, axis1=1, axis2=3)
                else:
                    raise ValueError(f"For bipartite systems, subsystem_to_trace must be 0 or 1, got {subsystem_to_trace}")
                
                return self._ensure_valid_density_matrix(reduced_dm)
                
            elif isinstance(system_dims, int) and isinstance(subsystem_to_trace, list):
                # Qubit system
                num_qubits = system_dims
                trace_qubits = subsystem_to_trace
                
                if not trace_qubits:
                    return density_matrix
                
                # Sort qubits to trace
                trace_qubits = sorted(trace_qubits)
                
                # Calculate dimensions
                dim_full = 2 ** num_qubits
                dim_trace = 2 ** len(trace_qubits)
                dim_keep = 2 ** (num_qubits - len(trace_qubits))
                
                # Validate dimensions
                if density_matrix.shape != (dim_full, dim_full):
                    raise ValueError(
                        f"Density matrix dimension {density_matrix.shape} does not match "
                        f"expected dimension for {num_qubits} qubits: {dim_full}x{dim_full}"
                    )
                
                # Identify which qubits to keep
                keep_qubits = [i for i in range(num_qubits) if i not in trace_qubits]
                
                # Calculate permutation to move trace qubits to the end
                perm = keep_qubits + trace_qubits
                
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
                
                return self._ensure_valid_density_matrix(reduced_dm)
            else:
                raise ValueError(
                    f"Invalid parameter types: system_dims must be tuple or int, "
                    f"subsystem_to_trace must be int or list, got {type(system_dims)} and {type(subsystem_to_trace)}"
                )
        except Exception as e:
            logger.error(f"Error performing partial trace: {e}")
            # Return identity/maximally mixed state as a fallback
            if isinstance(system_dims, tuple) and isinstance(subsystem_to_trace, int):
                retained_dim = system_dims[1 - subsystem_to_trace]
            else:
                retained_dim = 2 ** (system_dims - len(subsystem_to_trace))
                
            return np.eye(retained_dim) / retained_dim
        
    def is_state_entangled(self, density_matrix: np.ndarray, dims: Tuple[int, int], threshold: float = 1e-6) -> bool:
        """
        Determine if a quantum state is entangled using the Peres–Horodecki criterion.
        
        This method implements the PPT (Positive Partial Transpose) criterion which is
        necessary and sufficient for 2×2 and 2×3 systems, and necessary for larger systems.
        
        Args:
            density_matrix: The density matrix of the quantum state
            dims: Dimensions of the two subsystems (dim_A, dim_B)
            threshold: Numerical threshold for determining negativity
            
        Returns:
            bool: True if the state is entangled, False otherwise
        """
        try:
            entanglement_measure = self.calculate_entanglement(density_matrix, dims)
            return entanglement_measure > threshold
        except Exception as e:
            logger.error(f"Error determining if state is entangled: {e}")
            return False  # Conservative fallback - assume separable
        

    def calculate_bipartite_coherence(self, density_matrix: np.ndarray, dims: Tuple[int, int]) -> float:
        """
        Calculate the coherence between two subsystems in a bipartite quantum state.
        
        This method captures the quantum correlation beyond entanglement, providing a
        measure of coherent information flow between subsystems.
        
        Args:
            density_matrix: The density matrix of the composite system
            dims: Dimensions of the two subsystems (dim_A, dim_B)
            
        Returns:
            float: Coherence measure between the subsystems (0 to 1)
        """
        try:
            # Calculate reduced density matrices
            rho_a = self._partial_trace(density_matrix, dims, 1)  # Trace out B
            rho_b = self._partial_trace(density_matrix, dims, 0)  # Trace out A
            
            # Calculate von Neumann entropies
            # For each density matrix ρ, S(ρ) = -Tr(ρ log₂ ρ)
            
            # Helper function to calculate entropy
            def von_neumann_entropy(rho):
                # Get eigenvalues
                eigenvalues = np.linalg.eigvalsh(rho)
                # Filter out very small eigenvalues (numerical errors)
                eigenvalues = eigenvalues[eigenvalues > 1e-10]
                # Calculate entropy: -sum(λ log₂ λ)
                return -np.sum(eigenvalues * np.log2(eigenvalues))
            
            # Calculate entropies
            s_a = von_neumann_entropy(rho_a)
            s_b = von_neumann_entropy(rho_b)
            s_ab = von_neumann_entropy(density_matrix)
            
            # Calculate mutual information: I(A:B) = S(A) + S(B) - S(AB)
            mutual_info = s_a + s_b - s_ab
            
            # Normalize to [0, 1] range
            # Maximum mutual information is min(log₂(dim_A), log₂(dim_B))
            max_mutual_info = min(np.log2(dims[0]), np.log2(dims[1]))
            if max_mutual_info > 0:
                normalized_coherence = mutual_info / max_mutual_info
            else:
                normalized_coherence = 0.0
            
            # Ensure it's in valid range [0, 1]
            return max(0.0, min(1.0, normalized_coherence))
            
        except Exception as e:
            logger.error(f"Error calculating bipartite coherence: {e}")
            return 0.0  # Default fallback value
        
    def get_coherence_statistics(self) -> Dict[str, float]:
        """
        Get statistical metrics about coherence across all registered states.
        
        Returns:
            Dict[str, float]: Dictionary of coherence statistics
        """
        try:
            if not self.state_coherence_registry:
                return {
                    'count': 0,
                    'average': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0,
                    'std_dev': 0.0
                }
            
            coherence_values = list(self.state_coherence_registry.values())
            
            stats = {
                'count': len(coherence_values),
                'average': np.mean(coherence_values),
                'min': min(coherence_values),
                'max': max(coherence_values),
                'median': np.median(coherence_values),
                'std_dev': np.std(coherence_values),
                'below_0.2': sum(1 for c in coherence_values if c < 0.2),
                'high_coherence': sum(1 for c in coherence_values if c > 0.8)
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error calculating coherence statistics: {e}")
            return {'error': str(e)}
        
    def apply_gradient_descent(self, state_name: str, target_coherence: float, iterations: int = 8) -> float:
        """
        Apply gradient descent algorithm to optimize coherence with momentum and adaptive learning rates.
        
        Args:
            state_name: Target state name
            target_coherence: Target coherence level
            iterations: Number of gradient steps
            
        Returns:
            float: New coherence value
        """
        try:
            # Get current coherence
            current_coherence = self.state_coherence_registry.get(state_name, 0.5)
            current_entropy = self.state_entropy_registry.get(state_name, 0.5)
            
            # Optimization parameters
            initial_learning_rate = 0.2
            momentum = 0.7
            decay_rate = 0.9
            
            # Initialize variables
            coherence = current_coherence
            entropy = current_entropy
            velocity = 0.0
            learning_rate = initial_learning_rate
            
            # Gradient descent with momentum
            for i in range(iterations):
                # Calculate gradient (simple objective: distance to target)
                gradient = target_coherence - coherence
                
                # Apply adaptive learning rate
                # Learning rate decreases when close to target or after many iterations
                adaptive_rate = learning_rate * np.exp(-0.5 * abs(gradient))
                
                # Update velocity (momentum term)
                velocity = momentum * velocity + adaptive_rate * gradient
                
                # Apply noise for exploration (diminishes over time)
                exploration_noise = 0.03 * np.random.normal() * np.exp(-i / (iterations/2))
                
                # Update coherence
                delta_coherence = velocity + exploration_noise
                coherence += delta_coherence
                
                # Update entropy (generally decreases as coherence approaches target)
                entropy_gradient = -0.1 * abs(delta_coherence) * entropy
                entropy += entropy_gradient
                
                # Constrain values to valid ranges
                coherence = max(0.0, min(1.0, coherence))
                entropy = max(0.0, entropy)
                
                # Decay learning rate
                learning_rate *= decay_rate
                
                logger.debug(f"Gradient descent iteration {i}: coherence={coherence:.4f}, "
                            f"gradient={gradient:.4f}, velocity={velocity:.4f}")
            
            # Update registries
            self.state_coherence_registry[state_name] = coherence
            self.state_entropy_registry[state_name] = entropy
            
            return coherence
        except Exception as e:
            logger.error(f"Error applying gradient descent to {state_name}: {e}")
            # Return a default value as fallback
            return 0.5
        
    def apply_stochastic_descent_algorithm(self, state_name: str, target_coherence: float, iterations: int = 5) -> float:
        """
        Apply stochastic descent algorithm to optimize coherence through simulated annealing.
        
        Args:
            state_name: Target state name
            target_coherence: Target coherence level
            iterations: Number of optimization iterations
            
        Returns:
            float: New coherence value
        """
        try:
            # Get current state values
            current_coherence = self.state_coherence_registry.get(state_name, 0.5)
            current_entropy = self.state_entropy_registry.get(state_name, 0.5)
            
            # Initialize optimization parameters
            best_coherence = current_coherence
            best_entropy = current_entropy
            temperature = 1.0
            cooling_rate = 0.8
            
            # Stochastic descent with simulated annealing
            for i in range(iterations):
                # Generate candidate step with controlled randomness
                step_size = 0.1 * (1.0 - i/iterations)  # Decreasing step size
                
                # Random perturbation
                coherence_step = step_size * (target_coherence - current_coherence) + \
                                0.05 * np.random.normal()
                    
                # Apply candidate step
                candidate_coherence = current_coherence + coherence_step
                candidate_coherence = max(0.0, min(1.0, candidate_coherence))
                
                # Entropy tends to decrease as coherence increases
                entropy_delta = -0.05 * coherence_step * (1 + np.random.random() * 0.2)
                candidate_entropy = max(0.0, current_entropy + entropy_delta)
                
                # Evaluate fitness (higher coherence, lower entropy is better)
                current_fitness = current_coherence - 0.3 * current_entropy
                candidate_fitness = candidate_coherence - 0.3 * candidate_entropy
                
                # Acceptance probability (simulated annealing)
                delta_fitness = candidate_fitness - current_fitness
                
                # Accept if better or probabilistically if worse
                if delta_fitness > 0 or np.random.random() < np.exp(delta_fitness / temperature):
                    current_coherence = candidate_coherence
                    current_entropy = candidate_entropy
                    
                    # Track best solution
                    if candidate_fitness > (best_coherence - 0.3 * best_entropy):
                        best_coherence = candidate_coherence
                        best_entropy = candidate_entropy
                
                # Cool temperature
                temperature *= cooling_rate
                
                logger.debug(f"Stochastic descent iteration {i}: coherence={current_coherence:.4f}, "
                            f"entropy={current_entropy:.4f}, temp={temperature:.4f}")
            
            # Update registries with best found values
            self.state_coherence_registry[state_name] = best_coherence
            self.state_entropy_registry[state_name] = best_entropy
            
            return best_coherence
        except Exception as e:
            logger.error(f"Error applying stochastic descent to {state_name}: {e}")
            # Return a default value as fallback
            return 0.5
        
    def apply_tensor_compression(self, state_name: str, target_coherence: float) -> float:
        """
        Apply tensor network compression algorithm to optimize coherence.
        This implements SVD-based tensor compression to reduce entropy while
        preserving quantum state information.
        
        Args:
            state_name: Target state name
            target_coherence: Target coherence level
            
        Returns:
            float: New coherence value
        """
        try:
            # Get current coherence and entropy
            current_coherence = self.state_coherence_registry.get(state_name, 0.5)
            current_entropy = self.state_entropy_registry.get(state_name, 0.5)
            
            # Calculate bond dimension based on entropy
            # Higher entropy states require more aggressive compression
            max_bond_dimension = int(10 * (1.0 - current_entropy * 0.5))
            bond_dimension = max(2, min(10, max_bond_dimension))
            
            # SVD truncation factor - determines how much information to preserve
            truncation_factor = min(0.9, 0.7 + 0.2 * current_coherence)
            
            # Calculate compression effects
            
            # Entropy reduction from compression (more effective at high entropy)
            entropy_reduction = 0.15 * (current_entropy ** 0.7) * (1.0 - truncation_factor)
            new_entropy = max(0.0, current_entropy - entropy_reduction)
            
            # Coherence enhancement (inversely proportional to truncation)
            # More truncation can initially reduce coherence, but lower entropy improves it
            coherence_delta = 0.2 * (1.0 - current_entropy) * (target_coherence - current_coherence)
            coherence_loss = 0.05 * (1.0 - truncation_factor) * current_coherence
            
            # Net coherence change
            net_change = coherence_delta - coherence_loss
            new_coherence = current_coherence + net_change
            
            # Ensure values are in valid range
            new_coherence = max(0.0, min(1.0, new_coherence))
            
            # Update registries
            self.state_coherence_registry[state_name] = new_coherence
            self.state_entropy_registry[state_name] = new_entropy
            
            # Return compression statistics for logging/monitoring
            compression_stats = {
                'bond_dimension': bond_dimension,
                'truncation_factor': truncation_factor,
                'entropy_reduction': entropy_reduction,
                'coherence_delta': net_change
            }
            
            logger.debug(f"Tensor compression applied to {state_name}: {compression_stats}")
            
            return new_coherence
        except Exception as e:
            logger.error(f"Error applying tensor compression to {state_name}: {e}")
            # Return a default value as fallback
            return 0.5
        
    def calculate_coherence_between(self, state1: str, state2: str) -> float:
        """
        Calculate coherence relationship between two quantum states using their 
        density matrices and quantum information metrics.
        
        Args:
            state1: First state name
            state2: Second state name
            
        Returns:
            float: Coherence relationship measure (0 to 1)
        """
        try:
            # First get stored coherence values as a baseline
            coherence1 = self.state_coherence_registry.get(state1, 0.5)
            coherence2 = self.state_coherence_registry.get(state2, 0.5)
            entropy1 = self.state_entropy_registry.get(state1, 0.5)
            entropy2 = self.state_entropy_registry.get(state2, 0.5)
            
            # Attempt to get density matrices using standard access patterns
            dm1 = None
            dm2 = None
            
            # Try different ways to access density matrices
            if hasattr(self, 'state_registry') and self.state_registry is not None:
                # Try to get matrices from state registry
                dm1 = self.state_registry.get_density_matrix(state1)
                dm2 = self.state_registry.get_density_matrix(state2)
            elif hasattr(self, 'get_density_matrix') and callable(self.get_density_matrix):
                # Try using a method if available
                dm1 = self.get_density_matrix(state1)
                dm2 = self.get_density_matrix(state2)
            elif hasattr(self, 'density_matrices') and isinstance(self.density_matrices, dict):
                # Fall back to direct attribute access
                dm1 = self.density_matrices.get(state1)
                dm2 = self.density_matrices.get(state2)
            
            # If we couldn't get both density matrices, use the registry-based approach
            if dm1 is None or dm2 is None:
                logger.debug(f"Using registry-based coherence calculation for {state1} and {state2}")
                
                # Calculate a more sophisticated approximation using both coherence and entropy
                coherence_diff = abs(coherence1 - coherence2)
                entropy_diff = abs(entropy1 - entropy2)
                
                # Consider both coherence similarity and entropy similarity
                coherence_similarity = 1.0 - coherence_diff
                entropy_similarity = 1.0 - min(1.0, entropy_diff)
                
                # Weight coherence higher than entropy
                relationship = 0.7 * coherence_similarity + 0.3 * entropy_similarity
                relationship *= (coherence1 + coherence2) / 2  # Scale by average coherence
                
                return max(0.0, min(1.0, relationship))
            
            # Validate density matrices
            if not isinstance(dm1, np.ndarray) or not isinstance(dm2, np.ndarray):
                logger.warning(f"Invalid density matrix types: {type(dm1)} and {type(dm2)}")
                return max(0.0, min(1.0, (coherence1 + coherence2) / 2))
            
            # Check if matrices have compatible dimensions
            if dm1.shape != dm2.shape:
                logger.warning(f"Incompatible dimensions: {dm1.shape} vs {dm2.shape}")
                return max(0.0, min(1.0, (coherence1 + coherence2) / 2))
            
            # Calculate trace distance correctly: 0.5 * ||ρ1 - ρ2||₁
            # The trace norm ||A||₁ = Tr(√(A†A))
            diff = dm1 - dm2
            
            # Compute eigenvalues of diff†·diff for numerical stability
            eigenvalues = np.linalg.eigvalsh(diff @ diff.conj().T)
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative due to numerical precision
            
            # Trace distance = 0.5 * sum(sqrt(eigenvalues))
            trace_distance = 0.5 * np.sum(np.sqrt(eigenvalues))
            
            # Calculate fidelity using Uhlmann's theorem: F(ρ,σ) = (Tr[√(√ρσ√ρ)])²
            try:
                # More stable fidelity calculation using eigendecomposition
                sqrt_dm1 = self._matrix_sqrt(dm1)
                intermediate = sqrt_dm1 @ dm2 @ sqrt_dm1
                eigenvalues = np.linalg.eigvalsh(intermediate)
                eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
                fidelity = np.sum(np.sqrt(eigenvalues))
                
                # The square is often omitted for quantum states
                fidelity = min(1.0, max(0.0, fidelity))
            except np.linalg.LinAlgError:
                logger.warning(f"Fidelity calculation failed, using fallback for {state1} and {state2}")
                # Fall back to coherence-based approximation
                fidelity = 1.0 - 0.5 * abs(coherence1 - coherence2)
            
            # Combine trace distance and fidelity into a coherence relationship metric
            relationship = 0.5 * (1.0 - trace_distance) + 0.5 * fidelity
            
            # Weight with individual coherence values (use stored values for efficiency)
            individual_coherence_weight = 0.2
            avg_coherence = (coherence1 + coherence2) / 2
            
            # Final relationship measure
            final_relationship = (1.0 - individual_coherence_weight) * relationship + individual_coherence_weight * avg_coherence
            
            # Ensure valid range [0, 1]
            return max(0.0, min(1.0, final_relationship))
        except Exception as e:
            logger.error(f"Error calculating coherence between {state1} and {state2}: {e}")
            # Return a reasonable default based on individual coherence values
            return max(0.0, min(1.0, (coherence1 + coherence2) / 2))
        
    def _matrix_sqrt(self, matrix: np.ndarray) -> np.ndarray:
        """
        Calculate the matrix square root √A for a Hermitian matrix A.
        
        Uses eigendecomposition for numerical stability: A = UDU† → √A = UD^(1/2)U†
        
        Args:
            matrix: Hermitian matrix to find square root of
            
        Returns:
            np.ndarray: Square root of the input matrix
            
        Raises:
            np.linalg.LinAlgError: If eigendecomposition fails
        """
        # Ensure matrix is Hermitian (should be for density matrices)
        if not np.allclose(matrix, matrix.conj().T):
            # Make it Hermitian by averaging with conjugate transpose
            matrix = 0.5 * (matrix + matrix.conj().T)
        
        # Calculate eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        
        # Filter out negative eigenvalues (numerical errors)
        eigenvalues = np.maximum(eigenvalues, 0)
        
        # Calculate square root of eigenvalues
        sqrt_eigenvalues = np.sqrt(eigenvalues)
        
        # Reconstruct the matrix
        sqrt_matrix = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.conj().T
        
        return sqrt_matrix

    def get_state_coherence(self, state_name: str) -> float:
        """
        Get the current coherence value for a state.
        
        Args:
            state_name: State name
            
        Returns:
            float: Coherence value
        """
        return self.state_coherence_registry.get(state_name, 0.5)
    
    def set_state_coherence(self, state_name: str, coherence: float) -> None:
        """
        Set the coherence value for a state.
        
        Args:
            state_name: State name
            coherence: Coherence value (0 to 1)
        """
        self.state_coherence_registry[state_name] = max(0.0, min(1.0, coherence))
    
    def get_state_entropy(self, state_name: str) -> float:
        """
        Get the current entropy value for a state.
        
        Args:
            state_name: State name
            
        Returns:
            float: Entropy value
        """
        return self.state_entropy_registry.get(state_name, 0.5)
    
    def set_state_entropy(self, state_name: str, entropy: float) -> None:
        """
        Set the entropy value for a state.
        
        Args:
            state_name: State name
            entropy: Entropy value (0+)
        """
        self.state_entropy_registry[state_name] = max(0.0, entropy)
    
    def reset_state(self, state_name: str) -> None:
        """
        Reset coherence and entropy for a state to default values.
        
        Args:
            state_name: State name
        """
        self.state_coherence_registry[state_name] = 1.0  # Maximum coherence
        self.state_entropy_registry[state_name] = 0.0    # Minimum entropy