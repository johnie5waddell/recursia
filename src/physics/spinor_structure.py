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
from typing import List, Tuple, Optional, Literal


class SpinorStructure:
    """
    Comprehensive spinor structure implementation for relativistic quantum field theory simulations.
    
    Provides matrix representations for the Dirac algebra in various representations:
    - Dirac (standard) representation
    - Weyl representation 
    - Chiral representation
    
    The class includes alpha, beta, and gamma matrices necessary for spinor field evolution,
    and utilities for momentum operators and spatial derivatives.
    """
    
    def __init__(
        self, 
        dimension: int = 4, 
        representation: Literal["dirac", "weyl", "chiral"] = "dirac"
    ):
        """
        Initialize spinor structure with specified dimension and representation.
        
        Args:
            dimension: Spinor dimension (4 for Dirac, 2 for Weyl/Pauli)
            representation: Representation type ("dirac", "weyl", "chiral")
            
        Raises:
            ValueError: If dimension or representation is not supported
        """
        if dimension not in [2, 4]:
            raise ValueError(f"Dimension {dimension} not supported. Use 2 for Pauli/Weyl or 4 for Dirac.")
            
        if representation not in ["dirac", "weyl", "chiral"]:
            raise ValueError(f"Representation '{representation}' not supported. Use 'dirac', 'weyl', or 'chiral'.")
        
        self.dimension = dimension
        self.representation = representation
        
        # Initialize Pauli matrices for all representations
        self.sigma = self._get_pauli_matrices()
        
        # Initialize matrices based on representation
        self._initialize_matrices()
        
    def _get_pauli_matrices(self) -> List[np.ndarray]:
        """Return standard Pauli matrices as a list [σx, σy, σz]."""
        sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        return [sigma_x, sigma_y, sigma_z]
    
    def _initialize_matrices(self) -> None:
        """Initialize matrices for the selected representation."""
        if self.dimension == 2:
            # For 2D spinors, use Pauli matrices directly
            self.alpha = self.sigma
            self.beta = np.eye(2, dtype=np.complex128)
            self.gamma = self._get_pauli_gamma_matrices()
            return
            
        # For 4D spinors, use representation-specific matrices
        if self.representation == "dirac":
            self.alpha, self.beta = self._get_dirac_matrices()
            self.gamma = self._get_dirac_gamma_matrices()
        elif self.representation == "weyl":
            self.alpha, self.beta = self._get_weyl_matrices()
            self.gamma = self._get_weyl_gamma_matrices()
        elif self.representation == "chiral":
            self.alpha, self.beta = self._get_chiral_matrices()
            self.gamma = self._get_chiral_gamma_matrices()
        else:
            # Should never reach here due to validation, but as fallback
            self.alpha, self.beta = self._get_dirac_matrices()
            self.gamma = self._get_dirac_gamma_matrices()
    
    def _get_dirac_matrices(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Return alpha and beta matrices in the standard Dirac representation.
        
        Alpha matrices: α_i = [[0, σ_i], [σ_i, 0]]
        Beta matrix: β = [[I, 0], [0, -I]]
        
        Returns:
            Tuple containing:
                - List of alpha matrices [α₁, α₂, α₃]
                - Beta matrix β
        """
        # Identity and zero matrices
        I_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)
        
        # Alpha matrices
        alpha = []
        for sigma in self.sigma:
            alpha_matrix = np.block([
                [zero_2, sigma],
                [sigma, zero_2]
            ])
            alpha.append(alpha_matrix)
        
        # Beta matrix
        beta = np.block([
            [I_2, zero_2],
            [zero_2, -I_2]
        ])
        
        return alpha, beta
    
    def _get_weyl_matrices(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Return alpha and beta matrices in the Weyl representation.
        
        In Weyl representation, the beta matrix becomes off-diagonal:
        Alpha matrices: Similar structure to Dirac
        Beta matrix: β = [[0, I], [I, 0]]
        
        Returns:
            Tuple containing:
                - List of alpha matrices [α₁, α₂, α₃]
                - Beta matrix β
        """
        # Identity and zero matrices
        I_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)
        
        # Alpha matrices - similar structure to Dirac
        alpha = []
        for i, sigma in enumerate(self.sigma):
            alpha_matrix = np.block([
                [zero_2, sigma],
                [sigma, zero_2]
            ])
            alpha.append(alpha_matrix)
        
        # Beta matrix in Weyl representation
        beta = np.block([
            [zero_2, I_2],
            [I_2, zero_2]
        ])
        
        return alpha, beta
    
    def _get_chiral_matrices(self) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Return alpha and beta matrices in the chiral representation.
        
        In chiral representation:
        Alpha matrices: α_i = [[σ_i, 0], [0, -σ_i]]
        Beta matrix: β = [[0, I], [I, 0]]
        
        Returns:
            Tuple containing:
                - List of alpha matrices [α₁, α₂, α₃]
                - Beta matrix β
        """
        # Identity and zero matrices
        I_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)
        
        # Alpha matrices in chiral representation
        alpha = []
        for sigma in self.sigma:
            alpha_matrix = np.block([
                [sigma, zero_2],
                [zero_2, -sigma]
            ])
            alpha.append(alpha_matrix)
        
        # Beta matrix (same as Weyl representation)
        beta = np.block([
            [zero_2, I_2],
            [I_2, zero_2]
        ])
        
        return alpha, beta
    
    def _get_dirac_gamma_matrices(self) -> List[np.ndarray]:
        """
        Return gamma matrices in the Dirac representation.
        
        In Dirac representation:
        - γ⁰ = β = [[I, 0], [0, -I]]
        - γⁱ = β·αⁱ = [[0, σⁱ], [-σⁱ, 0]] for i=1,2,3
        - γ⁵ = iγ⁰γ¹γ²γ³ = [[0, I], [I, 0]]
        
        Returns:
            List of gamma matrices [γ⁰, γ¹, γ², γ³, γ⁵]
        """
        # Identity and zero matrices
        I_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)
        
        # γ⁰ = β
        gamma_0 = self.beta
        
        # Spatial gamma matrices: γⁱ = β·αⁱ
        gamma_spatial = []
        for alpha_i in self.alpha:
            gamma_i = np.matmul(self.beta, alpha_i)
            gamma_spatial.append(gamma_i)
        
        # γ⁵ = iγ⁰γ¹γ²γ³ (or direct construction)
        gamma_5 = np.block([
            [zero_2, I_2],
            [I_2, zero_2]
        ])
        
        # Combine all gamma matrices
        gamma = [gamma_0] + gamma_spatial + [gamma_5]
        return gamma
    
    def _get_weyl_gamma_matrices(self) -> List[np.ndarray]:
        """
        Return gamma matrices in the Weyl representation.
        
        In Weyl representation:
        - γ⁰ = β = [[0, I], [I, 0]]
        - γⁱ = β·αⁱ
        - γ⁵ = diagonal([[I, 0], [0, -I]])
        
        Returns:
            List of gamma matrices [γ⁰, γ¹, γ², γ³, γ⁵]
        """
        # Identity and zero matrices
        I_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)
        
        # γ⁰ = β
        gamma_0 = self.beta
        
        # Spatial gamma matrices: γⁱ = β·αⁱ
        gamma_spatial = []
        for alpha_i in self.alpha:
            gamma_i = np.matmul(self.beta, alpha_i)
            gamma_spatial.append(gamma_i)
        
        # γ⁵ in Weyl representation
        gamma_5 = np.block([
            [I_2, zero_2],
            [zero_2, -I_2]
        ])
        
        # Combine all gamma matrices
        gamma = [gamma_0] + gamma_spatial + [gamma_5]
        return gamma
    
    def _get_chiral_gamma_matrices(self) -> List[np.ndarray]:
        """
        Return gamma matrices in the chiral representation.
        
        In chiral representation:
        - γ⁰ = β = [[0, I], [I, 0]]
        - γⁱ = β·αⁱ
        - γ⁵ = diagonal([[-I, 0], [0, I]])
        
        Returns:
            List of gamma matrices [γ⁰, γ¹, γ², γ³, γ⁵]
        """
        # Identity and zero matrices
        I_2 = np.eye(2, dtype=np.complex128)
        zero_2 = np.zeros((2, 2), dtype=np.complex128)
        
        # γ⁰ = β
        gamma_0 = self.beta
        
        # Spatial gamma matrices: γⁱ = β·αⁱ
        gamma_spatial = []
        for alpha_i in self.alpha:
            gamma_i = np.matmul(self.beta, alpha_i)
            gamma_spatial.append(gamma_i)
        
        # γ⁵ in chiral representation
        gamma_5 = np.block([
            [-I_2, zero_2],
            [zero_2, I_2]
        ])
        
        # Combine all gamma matrices
        gamma = [gamma_0] + gamma_spatial + [gamma_5]
        return gamma
    
    def _get_pauli_gamma_matrices(self) -> List[np.ndarray]:
        """
        Return simplified gamma matrices for 2D spinors.
        
        For 2D spinors, we use a simplified representation:
        - γ⁰ = σz
        - γ¹ = iσy
        - γ² = iσx
        
        Returns:
            List of gamma matrices [γ⁰, γ¹, γ²]
        """
        gamma_0 = self.sigma[2]  # σz
        gamma_1 = 1j * self.sigma[1]  # iσy
        gamma_2 = 1j * self.sigma[0]  # iσx
        
        return [gamma_0, gamma_1, gamma_2]
    
    def get_momentum_operator(
        self, 
        component: int, 
        grid: np.ndarray, 
        dx: float,
        boundary_type: str = "central"
    ) -> np.ndarray:
        """
        Calculate momentum operator -iħ∇ for the specified component.
        
        Implements numerical differentiation with various boundary handling options.
        
        Args:
            component: Spatial dimension index (0, 1, 2 for x, y, z)
            grid: Field array to operate on
            dx: Grid spacing in the component direction
            boundary_type: Boundary condition type ('central', 'periodic', 'dirichlet', 'neumann')
            
        Returns:
            np.ndarray: Momentum operator -i∇ applied to grid
            
        Raises:
            ValueError: If component is invalid or boundary_type is not supported
        """
        if component >= grid.ndim:
            raise ValueError(f"Component {component} exceeds grid dimensions {grid.ndim}")
            
        if boundary_type not in ["central", "periodic", "dirichlet", "neumann"]:
            raise ValueError(f"Boundary type '{boundary_type}' not supported")
        
        # Create gradient array
        grad = np.zeros_like(grid, dtype=np.complex128)
        
        # Define slices for indexing
        slice_forward = [slice(None)] * grid.ndim
        slice_backward = [slice(None)] * grid.ndim
        slice_center = [slice(None)] * grid.ndim
        
        slice_forward[component] = slice(1, None)
        slice_center[component] = slice(None, -1)
        slice_backward[component] = slice(None, -1)
        
        # Create slices for boundaries
        left_slice = [slice(None)] * grid.ndim
        left_slice[component] = 0
        
        right_slice = [slice(None)] * grid.ndim
        right_slice[component] = -1
        
        # Central differences for interior points
        interior_slice = [slice(None)] * grid.ndim
        interior_slice[component] = slice(1, -1)
        grad[tuple(interior_slice)] = (grid[tuple(slice_forward)][tuple(interior_slice)] - 
                                     grid[tuple(slice_backward)][tuple(interior_slice)]) / (2 * dx)
        
        # Apply boundary conditions
        if boundary_type == "central":
            # Forward difference at left boundary
            left_next_slice = [slice(None)] * grid.ndim
            left_next_slice[component] = 1
            grad[tuple(left_slice)] = (grid[tuple(left_next_slice)] - 
                                     grid[tuple(left_slice)]) / dx
            
            # Backward difference at right boundary
            right_prev_slice = [slice(None)] * grid.ndim
            right_prev_slice[component] = -2
            grad[tuple(right_slice)] = (grid[tuple(right_slice)] - 
                                      grid[tuple(right_prev_slice)]) / dx
                                      
        elif boundary_type == "periodic":
            # Connect boundaries (periodic)
            # Last point uses first point and vice versa
            left_wrap_slice = [slice(None)] * grid.ndim
            left_wrap_slice[component] = -1
            
            right_wrap_slice = [slice(None)] * grid.ndim
            right_wrap_slice[component] = 0
            
            grad[tuple(left_slice)] = (grid[tuple(left_next_slice)] - 
                                     grid[tuple(left_wrap_slice)]) / (2 * dx)
            
            right_next_slice = [slice(None)] * grid.ndim
            right_next_slice[component] = 0  # Wrap to start
            
            grad[tuple(right_slice)] = (grid[tuple(right_next_slice)] - 
                                      grid[tuple(right_prev_slice)]) / (2 * dx)
                                      
        elif boundary_type == "dirichlet":
            # Zero boundary gradient - assumes boundary is fixed
            grad[tuple(left_slice)] = 0
            grad[tuple(right_slice)] = 0
            
        elif boundary_type == "neumann":
            # Enforce zero derivative at boundary
            left_next_slice = [slice(None)] * grid.ndim
            left_next_slice[component] = 1
            
            right_prev_slice = [slice(None)] * grid.ndim
            right_prev_slice[component] = -2
            
            # Second-order one-sided approximation
            grad[tuple(left_slice)] = (-3 * grid[tuple(left_slice)] + 
                                     4 * grid[tuple(left_next_slice)] - 
                                     grid[tuple(left_next_slice)]) / (2 * dx)
            
            grad[tuple(right_slice)] = (3 * grid[tuple(right_slice)] - 
                                      4 * grid[tuple(right_prev_slice)] + 
                                      grid[tuple(right_prev_slice)]) / (2 * dx)
        
        # Return momentum operator: -i∇
        return -1j * grad

    def get_dirac_operator(
        self, 
        grid: np.ndarray,
        dx: float,
        mass: float = 0.0,
        boundary_type: str = "central"
    ) -> np.ndarray:
        """
        Calculate the Dirac operator (i∂̸ - m) applied to a spinor field.
        
        The Dirac operator is: iγⁱ∂ᵢ - m = iγ⁰∂₀ + iγⁱ∂ᵢ - m
        For spatial-only calculations: iγⁱ∂ᵢ - m = iγ¹∂₁ + iγ²∂₂ + iγ³∂₃ - m
        
        Args:
            grid: Spinor field array to operate on
            dx: Grid spacing (assumed uniform across dimensions)
            mass: Mass parameter (default 0 for massless case)
            boundary_type: Boundary condition for derivatives
            
        Returns:
            np.ndarray: Result of Dirac operator applied to grid
        """
        # Number of spatial dimensions
        spatial_dims = min(grid.ndim, 3)
        
        # Initialize result
        result = np.zeros_like(grid, dtype=np.complex128)
        
        # Apply spatial derivatives
        for d in range(spatial_dims):
            # Get momentum for this component: -i∂ᵢ
            p_operator = self.get_momentum_operator(d, grid, dx, boundary_type)
            
            # Apply gamma matrix: γⁱ(-i∂ᵢ) = iγⁱ∂ᵢ
            # Note: gamma[d+1] because gamma[0] is temporal component
            if d+1 < len(self.gamma):
                if self.dimension == 4:
                    # For 4D spinors
                    result += np.tensordot(self.gamma[d+1], p_operator, axes=0)
                else:
                    # For 2D spinors
                    result += np.tensordot(self.gamma[d], p_operator, axes=0)
        
        # Apply mass term: -m
        if mass != 0:
            if self.dimension == 4:
                # For 4D spinors, beta = gamma[0]
                result -= mass * np.tensordot(self.gamma[0], grid, axes=0)
            else:
                # For 2D spinors
                result -= mass * grid
                
        return result