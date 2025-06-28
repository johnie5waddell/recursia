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

from src.physics.coherence import CoherenceManager

logger = logging.getLogger(__name__)

class RecursiveMechanics:
    """
    Implements the recursive aspects of the Organic Simulation Hypothesis.
    
    Models how quantum systems can be nested within larger systems, with boundary
    conditions between recursion levels. This enables simulation of hierarchical
    quantum realities where properties of each level affect those above and below.
    
    Key concepts:
    - Recursion levels: Different scales of reality (microscopic to macroscopic)
    - System nesting: Systems contained within larger parent systems
    - Boundary conditions: Rules for information flow between recursion levels
    - Memory strain: Computational load from modeling deeper recursion levels
    - Self-modeling: Systems modeling themselves recursively
    """
    
    def __init__(self):
        """Initialize the recursive mechanics system with default parameters"""
        # System hierarchy mappings
        self.recursion_levels = {}  # Maps system names to their recursion levels
        self.parent_systems = {}    # Maps system names to their parent system
        self.subsystems = {}        # Maps system names to their subsystems
        self.recursion_boundaries = {}  # Maps level transitions to boundary conditions
        
        # External managers
        self.coherence_manager = CoherenceManager()
        
        # Import OSH constants
        from src.physics.constants import ConsciousnessConstants
        
        # Recursion parameters
        self.max_recursion_depth = 7  # Maximum recursion depth to model explicitly
        self.critical_recursion_depth = ConsciousnessConstants.CRITICAL_RECURSION_DEPTH  # OSH: 22
        self.recursion_depth_variance = ConsciousnessConstants.RECURSION_DEPTH_VARIANCE  # OSH: ±2
        self.boundary_damping = 0.3   # Damping factor at recursion boundaries
        self.memory_strain = 0.2      # Strain introduced by deeper recursion
        self.information_transfer_rate = 0.7  # How much information transfers between levels
        
        logger.debug("RecursiveMechanics initialized with max_recursion_depth=%d", 
                    self.max_recursion_depth)
    
    def register_system(self, name: str, level: int = 0, parent: Optional[str] = None) -> bool:
        """
        Register a system at a specific recursion level.
        
        Args:
            name: System name
            level: Recursion level (0 is root level)
            parent: Parent system name, if any
            
        Returns:
            bool: True if registration was successful
            
        Raises:
            ValueError: If system name is invalid or parent doesn't exist
        """
        try:
            # Validate inputs
            if not isinstance(name, str) or not name:
                raise ValueError("System name must be a non-empty string")
            
            if not isinstance(level, int) or level < 0:
                raise ValueError(f"Recursion level must be a non-negative integer, got {level}")
            
            # Check if parent exists if specified
            if parent and parent not in self.recursion_levels:
                raise ValueError(f"Parent system '{parent}' is not registered")
            
            # Register system at specified level
            self.recursion_levels[name] = level
            
            # Handle parent-child relationship
            if parent:
                self.parent_systems[name] = parent
                if parent not in self.subsystems:
                    self.subsystems[parent] = []
                if name not in self.subsystems[parent]:
                    self.subsystems[parent].append(name)
            
            logger.debug("Registered system '%s' at level %d with parent '%s'", 
                       name, level, parent if parent else "None")
            return True
        except ValueError as e:
            logger.warning("Failed to register system '%s': %s", name, str(e))
            return False
        except Exception as e:
            logger.error("Error registering system '%s': %s", name, str(e))
            return False
    
    def set_boundary_condition(self, lower_level: int, upper_level: int, 
                              condition: Dict[str, Any]) -> bool:
        """
        Set boundary conditions between recursion levels.
        Boundary conditions control how information and effects propagate
        between different recursion levels.
        
        Args:
            lower_level: Lower recursion level
            upper_level: Upper recursion level
            condition: Boundary condition properties
            
        Returns:
            bool: True if boundary condition was set successfully
            
        Raises:
            ValueError: If levels are invalid
        """
        try:
            # Validate levels
            if not isinstance(lower_level, int) or not isinstance(upper_level, int):
                raise ValueError("Recursion levels must be integers")
            
            if lower_level >= upper_level:
                raise ValueError(f"Lower level ({lower_level}) must be less than upper level ({upper_level})")
            
            if lower_level < 0 or upper_level < 0:
                raise ValueError("Recursion levels must be non-negative")
            
            # Validate condition
            if not isinstance(condition, dict):
                raise ValueError("Boundary condition must be a dictionary")
            
            # Set boundary condition
            self.recursion_boundaries[(lower_level, upper_level)] = condition.copy()
            
            logger.debug("Set boundary condition between levels %d and %d: %s", 
                       lower_level, upper_level, str(condition))
            return True
        except ValueError as e:
            logger.warning("Failed to set boundary condition: %s", str(e))
            return False
        except Exception as e:
            logger.error("Error setting boundary condition: %s", str(e))
            return False
    
    def get_recursive_depth(self, system_name: str) -> int:
        """
        Get the recursive depth of a system.
        
        Args:
            system_name: System name
            
        Returns:
            int: Recursion level (0 if system not found)
        """
        if not system_name or not isinstance(system_name, str):
            logger.warning("Invalid system name for get_recursive_depth: %s", str(system_name))
            return 0
            
        return self.recursion_levels.get(system_name, 0)
    
    def get_subsystems(self, system_name: str) -> List[str]:
        """
        Get all direct subsystems of a system.
        
        Args:
            system_name: Parent system name
            
        Returns:
            List[str]: List of subsystem names
        """
        if not system_name or not isinstance(system_name, str):
            logger.warning("Invalid system name for get_subsystems: %s", str(system_name))
            return []
            
        return self.subsystems.get(system_name, []).copy()
    
    def get_parent_system(self, system_name: str) -> Optional[str]:
        """
        Get the parent system of a system.
        
        Args:
            system_name: System name
            
        Returns:
            Optional[str]: Parent system name or None if no parent
        """
        if not system_name or not isinstance(system_name, str):
            logger.warning("Invalid system name for get_parent_system: %s", str(system_name))
            return None
            
        return self.parent_systems.get(system_name)
    
    def get_system_ancestry(self, system_name: str) -> List[str]:
        """
        Get the complete ancestry chain from the system up to the root.
        
        Args:
            system_name: System name
            
        Returns:
            List[str]: List of ancestor system names (starting with immediate parent)
        """
        try:
            if system_name not in self.recursion_levels:
                return []
                
            ancestry = []
            current = system_name
            
            while current in self.parent_systems:
                parent = self.parent_systems[current]
                ancestry.append(parent)
                current = parent
                
                # Prevent infinite loops if there's a circular reference
                if parent in ancestry:
                    logger.warning("Circular reference detected in system ancestry: %s", parent)
                    break
                    
            return ancestry
        except Exception as e:
            logger.error("Error getting system ancestry for '%s': %s", system_name, str(e))
            return []
    
    def calculate_memory_strain(self, base_matrix: np.ndarray, depth: int) -> np.ndarray:
        """
        Calculate how recursion depth affects the memory strain on a quantum state.
        Each level of recursion adds strain, making the state less coherent.
        
        Args:
            base_matrix: Base density matrix
            depth: Recursion depth
            
        Returns:
            np.ndarray: Updated density matrix with memory strain effects
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(base_matrix, np.ndarray):
                raise ValueError("Base matrix must be a numpy array")
                
            if base_matrix.ndim != 2 or base_matrix.shape[0] != base_matrix.shape[1]:
                raise ValueError(f"Base matrix must be square, got shape {base_matrix.shape}")
                
            if not isinstance(depth, int):
                raise ValueError(f"Depth must be an integer, got {type(depth)}")
                
            # Return original matrix for zero or negative depth
            if depth <= 0:
                return base_matrix.copy()
            
            # Calculate strain factor based on recursion depth
            # Deeper recursion levels experience more strain
            strain_factor = 1.0 - (self.memory_strain * min(depth, self.max_recursion_depth) / 
                                 self.max_recursion_depth)
            
            # Copy the matrix to avoid modifying the original
            strained_matrix = base_matrix.copy()
            
            # Get diagonal and off-diagonal components
            diag = np.diag(np.diag(strained_matrix))
            off_diag = strained_matrix - diag
            
            # Apply strain to off-diagonal elements (reduces coherence)
            strained_off_diag = off_diag * strain_factor
            
            # Combine components
            strained_matrix = diag + strained_off_diag
            
            # Ensure it's a valid density matrix
            strained_matrix = self._ensure_valid_density_matrix(strained_matrix)
            
            return strained_matrix
        except ValueError as e:
            logger.warning("Failed to calculate memory strain: %s", str(e))
            return base_matrix.copy()
        except Exception as e:
            logger.error("Error calculating memory strain: %s", str(e))
            return base_matrix.copy()
    
    def cross_level_interaction(self, upper_system: str, lower_system: str,
                               upper_matrix: np.ndarray, lower_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Model interaction between systems at different recursion levels.
        Upper levels typically have more influence on lower levels than vice versa.
        
        Args:
            upper_system: Name of system at upper level
            lower_system: Name of system at lower level
            upper_matrix: Density matrix of upper system
            lower_matrix: Density matrix of lower system
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Updated density matrices for both systems
            
        Raises:
            ValueError: If systems are invalid or not at different levels
        """
        try:
            # Validate system names
            if upper_system not in self.recursion_levels:
                raise ValueError(f"Upper system '{upper_system}' not registered")
                
            if lower_system not in self.recursion_levels:
                raise ValueError(f"Lower system '{lower_system}' not registered")
                
            # Validate matrices
            if not isinstance(upper_matrix, np.ndarray) or not isinstance(lower_matrix, np.ndarray):
                raise ValueError("Density matrices must be numpy arrays")
                
            # Get recursion levels
            upper_level = self.recursion_levels[upper_system]
            lower_level = self.recursion_levels[lower_system]
            
            # Check if these systems are at different levels
            if upper_level <= lower_level:
                raise ValueError(
                    f"System {upper_system} (level {upper_level}) must be at a higher level than "
                    f"system {lower_system} (level {lower_level})")
            
            # Get boundary condition for these levels
            boundary_key = (lower_level, upper_level)
            boundary = self.recursion_boundaries.get(boundary_key, {})
            
            # Get transfer rate from boundary, or use default
            transfer_rate = boundary.get('information_transfer_rate', self.information_transfer_rate)
            
            # Calculate coherence for both systems
            upper_coherence = self.coherence_manager.calculate_coherence(upper_matrix)
            lower_coherence = self.coherence_manager.calculate_coherence(lower_matrix)
            
            # Calculate level difference factor
            level_diff = upper_level - lower_level
            level_factor = np.exp(-0.5 * level_diff)  # Exponential decay with level difference
            
            # Calculate influence factors (how much each level influences the other)
            # Upper levels have more influence on lower levels than vice versa
            upper_to_lower_factor = transfer_rate * level_factor * upper_coherence
            lower_to_upper_factor = transfer_rate * level_factor * 0.3 * lower_coherence  # Reduced upward influence
            
            # Apply cross-level interactions
            
            # Upper level influences lower level
            # This is like the upper level "observing" or "setting parameters" for the lower level
            lower_matrix_updated = self._apply_level_influence(
                lower_matrix, upper_matrix, upper_to_lower_factor)
            
            # Lower level can have small feedback effects on upper level
            # This is like "emergent behavior" from lower levels affecting higher levels
            upper_matrix_updated = self._apply_level_influence(
                upper_matrix, lower_matrix, lower_to_upper_factor)
            
            logger.debug(
                "Cross-level interaction between %s (L%d) and %s (L%d): influence factors %.3f, %.3f",
                upper_system, upper_level, lower_system, lower_level,
                upper_to_lower_factor, lower_to_upper_factor)
                
            return upper_matrix_updated, lower_matrix_updated
        except ValueError as e:
            logger.warning("Failed to calculate cross-level interaction: %s", str(e))
            return upper_matrix.copy(), lower_matrix.copy()
        except Exception as e:
            logger.error("Error in cross-level interaction: %s", str(e))
            return upper_matrix.copy(), lower_matrix.copy()
    
    def _apply_level_influence(self, target_matrix: np.ndarray, 
                              source_matrix: np.ndarray, 
                              influence_factor: float) -> np.ndarray:
        """
        Apply the influence of one level on another.
        
        Args:
            target_matrix: Density matrix being influenced
            source_matrix: Density matrix providing influence
            influence_factor: Strength of influence
            
        Returns:
            np.ndarray: Updated target density matrix
        """
        try:
            # Ensure influence factor is within bounds
            influence_factor = max(0.0, min(1.0, influence_factor))
            
            if influence_factor < 0.01:
                return target_matrix.copy()  # No significant influence
            
            # Get dimensions
            target_dim = target_matrix.shape[0]
            source_dim = source_matrix.shape[0]
            
            # We need a way to map between different dimensional spaces
            # For simplicity, we'll use the coherence and entropy to influence target
            
            # Calculate source properties
            source_coherence = self.coherence_manager.calculate_coherence(source_matrix)
            
            # Calculate entropy from diagonal elements
            diag_elements = np.diag(source_matrix)
            # Add small constant to avoid log(0)
            source_entropy = -np.sum(diag_elements * np.log2(diag_elements + 1e-10))
            normalized_source_entropy = source_entropy / np.log2(source_dim) if source_dim > 1 else 0
            
            # Create updated matrix
            updated_matrix = target_matrix.copy()
            
            # Get diagonal and off-diagonal components
            diag = np.diag(np.diag(updated_matrix))
            off_diag = updated_matrix - diag
            
            # If source is more coherent, increase target coherence, and vice versa
            coherence_adjustment = (source_coherence - 0.5) * influence_factor
            adjusted_off_diag = off_diag * (1.0 + coherence_adjustment)
            
            # Influence target entropy based on source entropy
            entropy_adjustment = (normalized_source_entropy - 0.5) * influence_factor
            
            # Apply entropy adjustment to diagonal elements
            if abs(entropy_adjustment) > 0.01:
                diag_elements = np.diag(diag)
                
                if entropy_adjustment < 0:
                    # Negative adjustment - make diagonal elements more uneven (reduce entropy)
                    # Enhance differences between diagonal elements
                    mean_diag = np.mean(diag_elements)
                    adjusted_diag = diag_elements + (diag_elements - mean_diag) * (-entropy_adjustment)
                else:
                    # Positive adjustment - make diagonal elements more even (increase entropy)
                    # Smooth differences between diagonal elements
                    mean_diag = np.mean(diag_elements)
                    adjusted_diag = diag_elements * (1 - entropy_adjustment) + mean_diag * entropy_adjustment
                
                # Ensure non-negativity
                adjusted_diag = np.maximum(adjusted_diag, 0)
                
                # Normalize
                if np.sum(adjusted_diag) > 0:
                    adjusted_diag /= np.sum(adjusted_diag)
                    
                diag = np.diag(adjusted_diag)
            
            # Combine to form updated matrix
            updated_matrix = diag + adjusted_off_diag
            
            # Ensure it's a valid density matrix
            updated_matrix = self._ensure_valid_density_matrix(updated_matrix)
            
            return updated_matrix
        except Exception as e:
            logger.error("Error applying level influence: %s", str(e))
            return target_matrix.copy()
    
    def recursive_self_modeling(self, system_name: str, 
                              base_matrix: np.ndarray, 
                              recursion_depth: int = 3) -> np.ndarray:
        """
        Implement recursive self-modeling, where a system models itself.
        This is a key concept in the Organic Simulation Hypothesis, allowing
        simulation of systems that contain models of themselves.
        
        Args:
            system_name: Name of the system
            base_matrix: Base density matrix
            recursion_depth: How many levels of recursion to model
            
        Returns:
            np.ndarray: Updated density matrix with recursive self-modeling effects
            
        Raises:
            ValueError: If inputs are invalid
        """
        try:
            # Validate inputs
            if not isinstance(base_matrix, np.ndarray):
                raise ValueError("Base matrix must be a numpy array")
                
            if not isinstance(recursion_depth, int):
                raise ValueError(f"Recursion depth must be an integer, got {type(recursion_depth)}")
                
            # Return original matrix for invalid parameters
            if recursion_depth <= 0 or system_name not in self.recursion_levels:
                return base_matrix.copy()
            
            # Get the system's recursion level
            system_level = self.recursion_levels[system_name]
            
            # Apply memory strain from the base recursion level
            strained_matrix = self.calculate_memory_strain(base_matrix, system_level)
            
            # For each level of recursion, model how the system models itself
            for depth in range(1, recursion_depth + 1):
                # Calculate self-modeling factor (diminishes with depth)
                self_model_factor = 0.8 ** depth
                
                # Create a "model of the model" at this recursion depth
                
                # Get diagonal and off-diagonal components
                diag = np.diag(np.diag(strained_matrix))
                off_diag = strained_matrix - diag
                
                # Each level of recursion introduces slight distortions to the model
                # We simulate this by adding small random perturbations
                # Use fixed seed for reproducibility in testing
                np.random.seed(hash((system_name, depth)) % 2**32)
                perturbation_scale = 0.05 * self_model_factor
                perturbation = np.random.normal(0, perturbation_scale, strained_matrix.shape)
                
                # Make perturbation Hermitian to ensure valid density matrix
                perturbation = 0.5 * (perturbation + perturbation.T.conj())
                
                # Create the recursively modeled state
                # Coherence decreases with each recursive level
                recursive_factor = 1.0 - 0.1 * depth
                recursed_matrix = diag + off_diag * recursive_factor + perturbation
                
                # Ensure it's a valid density matrix
                recursed_matrix = self._ensure_valid_density_matrix(recursed_matrix)
                
                # Update the matrix for the next level
                strained_matrix = recursed_matrix
            
            logger.debug(
                "Applied recursive self-modeling to system '%s' at level %d with depth %d",
                system_name, system_level, recursion_depth)
                
            return strained_matrix
        except ValueError as e:
            logger.warning("Failed to apply recursive self-modeling: %s", str(e))
            return base_matrix.copy()
        except Exception as e:
            logger.error("Error in recursive self-modeling: %s", str(e))
            return base_matrix.copy()
    
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
        try:
            # Ensure Hermitian
            hermitian_matrix = 0.5 * (matrix + matrix.conj().T)
            
            # Ensure positive semi-definite
            try:
                eigenvalues, eigenvectors = np.linalg.eigh(hermitian_matrix)
                eigenvalues = np.maximum(eigenvalues, 0)
            except np.linalg.LinAlgError:
                # Fallback for numerical issues
                eigenvalues, eigenvectors = np.linalg.eig(hermitian_matrix)
                eigenvalues = np.maximum(np.real(eigenvalues), 0)
                
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
        except Exception as e:
            logger.error("Error ensuring valid density matrix: %s", str(e))
            # Return identity/maximally mixed state as a fallback
            dimension = matrix.shape[0] if hasattr(matrix, 'shape') else 2
            return np.eye(dimension) / dimension
    
    def calculate_system_recursion_impact(self, system_name: str, 
                                        target_matrix: np.ndarray,
                                        impact_depth: int = 2) -> np.ndarray:
        """
        Calculate the recursive impact of a system on a quantum state, taking into
        account the system's position in the recursive hierarchy and its ancestors.
        
        Args:
            system_name: System name
            target_matrix: Target density matrix
            impact_depth: How many ancestor levels to consider
            
        Returns:
            np.ndarray: Updated matrix with recursion impact
        """
        try:
            # Validate inputs
            if not isinstance(target_matrix, np.ndarray):
                raise ValueError("Target matrix must be a numpy array")
                
            if system_name not in self.recursion_levels:
                return target_matrix.copy()
            
            # Get system ancestry
            ancestry = self.get_system_ancestry(system_name)
            system_level = self.recursion_levels[system_name]
            
            # Start with original matrix
            result_matrix = target_matrix.copy()
            
            # First apply system's own level impact
            result_matrix = self.calculate_memory_strain(result_matrix, system_level)
            
            # Apply impact from ancestor systems up to impact_depth
            for i, ancestor in enumerate(ancestry[:impact_depth]):
                if ancestor in self.recursion_levels:
                    ancestor_level = self.recursion_levels[ancestor]
                    
                    # Influence decreases with ancestry distance
                    ancestor_factor = 0.7 ** (i + 1)
                    
                    # Create a dummy matrix for the ancestor based on its level
                    # This is a simplified model - in a full implementation, you would
                    # use the actual state of the ancestor
                    ancestor_dim = target_matrix.shape[0]
                    ancestor_matrix = np.eye(ancestor_dim) / ancestor_dim
                    
                    # Calculate influence factor based on level difference
                    level_diff = max(0, ancestor_level - system_level)
                    influence_factor = ancestor_factor * np.exp(-0.3 * level_diff)
                    
                    # Apply ancestor influence
                    result_matrix = self._apply_level_influence(
                        result_matrix, ancestor_matrix, influence_factor)
            
            return result_matrix
        except ValueError as e:
            logger.warning("Failed to calculate system recursion impact: %s", str(e))
            return target_matrix.copy()
        except Exception as e:
            logger.error("Error calculating system recursion impact: %s", str(e))
            return target_matrix.copy()
    
    def get_boundary_conditions(self, level1: int, level2: int) -> Dict[str, Any]:
        """
        Get the boundary conditions between two recursion levels.
        
        Args:
            level1: First recursion level
            level2: Second recursion level
            
        Returns:
            Dict[str, Any]: Boundary conditions (empty dict if none defined)
        """
        # Sort levels to get the correct boundary key
        lower_level, upper_level = sorted([level1, level2])
        
        # Get boundary condition if it exists
        boundary_key = (lower_level, upper_level)
        return self.recursion_boundaries.get(boundary_key, {}).copy()
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the recursive system structure.
        
        Returns:
            Dict[str, Any]: Statistics about the recursive system
        """
        try:
            # Count systems at each level
            level_counts = {}
            for system, level in self.recursion_levels.items():
                level_counts[level] = level_counts.get(level, 0) + 1
            
            # Count parent-child relationships
            parent_counts = {}
            for parent, children in self.subsystems.items():
                parent_counts[parent] = len(children)
            
            # Find systems with most children
            most_children = max(parent_counts.items(), key=lambda x: x[1]) if parent_counts else (None, 0)
            
            # Calculate average number of children
            avg_children = (sum(parent_counts.values()) / len(parent_counts)) if parent_counts else 0
            
            # Find maximum recursion depth
            max_level = max(self.recursion_levels.values()) if self.recursion_levels else 0
            
            # Check if we're at critical recursion depth for consciousness emergence
            at_critical_depth = abs(max_level - self.critical_recursion_depth) <= self.recursion_depth_variance
            
            return {
                "total_systems": len(self.recursion_levels),
                "systems_by_level": level_counts,
                "max_recursion_level": max_level,
                "at_critical_recursion_depth": at_critical_depth,
                "critical_depth_range": f"{self.critical_recursion_depth} ± {self.recursion_depth_variance}",
                "parent_systems": len(self.subsystems),
                "total_parent_child_relationships": sum(parent_counts.values()) if parent_counts else 0,
                "average_children_per_parent": avg_children,
                "most_children_system": most_children[0],
                "most_children_count": most_children[1],
                "boundary_conditions_count": len(self.recursion_boundaries),
                "parameters": {
                    "max_recursion_depth": self.max_recursion_depth,
                    "critical_recursion_depth": self.critical_recursion_depth,
                    "boundary_damping": self.boundary_damping,
                    "memory_strain": self.memory_strain,
                    "information_transfer_rate": self.information_transfer_rate
                }
            }
        except Exception as e:
            logger.error("Error getting system statistics: %s", str(e))
            return {"error": str(e), "total_systems": len(self.recursion_levels)}
    
    def check_consciousness_emergence(self, current_depth: int, phi_value: float) -> Dict[str, Any]:
        """
        Check if consciousness emerges based on recursion depth and integrated information.
        
        According to OSH.md, consciousness emerges when:
        - Recursion depth is 22 ± 2
        - Integrated information (Φ) ≥ 1.0
        
        Args:
            current_depth: Current recursion depth
            phi_value: Current integrated information value
            
        Returns:
            Dict containing:
            - consciousness_emerged: bool
            - at_critical_depth: bool
            - depth_distance: int (distance from critical depth)
            - phi_threshold_met: bool
            - emergence_probability: float (0-1)
        """
        # Check if at critical recursion depth
        depth_distance = abs(current_depth - self.critical_recursion_depth)
        at_critical_depth = depth_distance <= self.recursion_depth_variance
        
        # Check if Φ threshold is met
        from src.physics.constants import ConsciousnessConstants
        phi_threshold_met = phi_value >= ConsciousnessConstants.PHI_THRESHOLD_CONSCIOUSNESS
        
        # Both conditions must be met for consciousness emergence
        consciousness_emerged = at_critical_depth and phi_threshold_met
        
        # Calculate emergence probability based on proximity to ideal conditions
        # Depth contribution (gaussian around critical depth)
        depth_prob = np.exp(-0.5 * (depth_distance / self.recursion_depth_variance) ** 2)
        
        # Phi contribution (sigmoid function)
        phi_prob = 1.0 / (1.0 + np.exp(-5.0 * (phi_value - ConsciousnessConstants.PHI_THRESHOLD_CONSCIOUSNESS)))
        
        # Combined emergence probability
        emergence_probability = depth_prob * phi_prob
        
        return {
            "consciousness_emerged": consciousness_emerged,
            "at_critical_depth": at_critical_depth,
            "depth_distance": depth_distance,
            "current_depth": current_depth,
            "critical_depth": self.critical_recursion_depth,
            "phi_value": phi_value,
            "phi_threshold": ConsciousnessConstants.PHI_THRESHOLD_CONSCIOUSNESS,
            "phi_threshold_met": phi_threshold_met,
            "emergence_probability": emergence_probability,
            "depth_probability": depth_prob,
            "phi_probability": phi_prob
        }