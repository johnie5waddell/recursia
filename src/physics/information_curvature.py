"""
Information Curvature Implementation
====================================

Implements the information-geometric curvature tensor calculations according to OSH.md.
This module calculates how information density curves spacetime through the
information-gravity coupling.

Key equation from OSH.md:
R_μν ~ α∇_μ∇_ν I

Where:
- R_μν: Ricci curvature tensor
- α = 8π: Information-gravity coupling constant
- I: Information density field
- ∇_μ: Covariant derivative

Author: OSH Implementation Team
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
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from scipy.ndimage import laplace
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

from .constants import FieldParameters, SPEED_OF_LIGHT, GRAVITATIONAL_CONSTANT

logger = logging.getLogger(__name__)


@dataclass
class InformationCurvatureTensor:
    """Results of information curvature calculation."""
    ricci_tensor: np.ndarray  # R_μν tensor components
    scalar_curvature: float   # R = g^μν R_μν
    information_density: np.ndarray  # I(x,y,z,t)
    gradient_field: np.ndarray  # ∇I
    coupling_strength: float  # α = 8π
    
    def get_einstein_tensor(self) -> np.ndarray:
        """Calculate Einstein tensor G_μν = R_μν - (1/2)g_μν R."""
        # For simplicity, using flat metric g_μν = η_μν
        metric = np.eye(4)  # Minkowski metric in (+,-,-,-) signature
        metric[0, 0] = 1
        metric[1, 1] = -1
        metric[2, 2] = -1
        metric[3, 3] = -1
        
        einstein = self.ricci_tensor - 0.5 * metric * self.scalar_curvature
        return einstein


class InformationCurvatureCalculator:
    """
    Calculates spacetime curvature induced by information density fields.
    
    This implements the core OSH prediction that information curves spacetime
    analogously to mass-energy in general relativity.
    """
    
    def __init__(self):
        """Initialize the information curvature calculator."""
        # Information-gravity coupling constant from OSH.md
        self.alpha = FieldParameters.INFORMATION_GRAVITY_COUPLING  # α = 8π
        
        # Numerical parameters
        self.grid_spacing = 1.0  # Spatial grid spacing
        self.time_spacing = 0.01  # Temporal grid spacing
        
        logger.info(f"Initialized InformationCurvatureCalculator with α = {self.alpha}")
    
    def calculate_curvature_from_information(
        self,
        information_field: np.ndarray,
        spatial_dimensions: Tuple[int, int, int],
        time_index: int = 0
    ) -> InformationCurvatureTensor:
        """
        Calculate Ricci curvature tensor from information density field.
        
        Implements: R_μν ~ α∇_μ∇_ν I
        
        Args:
            information_field: Information density field I(x,y,z,t)
            spatial_dimensions: (nx, ny, nz) grid dimensions
            time_index: Current time slice
            
        Returns:
            InformationCurvatureTensor with calculated components
        """
        nx, ny, nz = spatial_dimensions
        
        # Ensure information field has correct shape
        if information_field.shape != (nx, ny, nz):
            logger.warning(f"Reshaping information field from {information_field.shape} to {spatial_dimensions}")
            information_field = np.reshape(information_field, spatial_dimensions)
        
        # Calculate spatial gradients ∇I
        grad_x = np.gradient(information_field, axis=0) / self.grid_spacing
        grad_y = np.gradient(information_field, axis=1) / self.grid_spacing
        grad_z = np.gradient(information_field, axis=2) / self.grid_spacing
        
        # For time component, we need temporal history
        # For now, approximate ∂I/∂t = 0 (static approximation)
        grad_t = np.zeros_like(information_field)
        
        # Assemble 4-gradient ∇_μ I
        gradient_4d = np.stack([grad_t, grad_x, grad_y, grad_z], axis=-1)
        
        # Calculate second derivatives ∇_μ∇_ν I
        # This gives us the 4x4 Ricci tensor at each spatial point
        ricci_components = np.zeros((nx, ny, nz, 4, 4))
        
        # Time-time component
        ricci_components[:, :, :, 0, 0] = self.alpha * laplace(grad_t) / self.grid_spacing**2
        
        # Time-space components (symmetric)
        for i, grad_i in enumerate([grad_x, grad_y, grad_z], 1):
            ricci_components[:, :, :, 0, i] = self.alpha * np.gradient(grad_t, axis=i-1) / self.grid_spacing
            ricci_components[:, :, :, i, 0] = ricci_components[:, :, :, 0, i]
        
        # Space-space components
        spatial_grads = [grad_x, grad_y, grad_z]
        for i in range(3):
            for j in range(3):
                if i == j:
                    # Diagonal: ∇²I_i
                    ricci_components[:, :, :, i+1, j+1] = self.alpha * laplace(spatial_grads[i]) / self.grid_spacing**2
                else:
                    # Off-diagonal: ∂²I/∂x_i∂x_j
                    ricci_components[:, :, :, i+1, j+1] = self.alpha * np.gradient(spatial_grads[i], axis=j) / self.grid_spacing
        
        # Calculate scalar curvature R = g^μν R_μν
        # Using Minkowski metric for contraction
        scalar_curvature_field = (
            ricci_components[:, :, :, 0, 0] -  # g^00 = 1
            ricci_components[:, :, :, 1, 1] -  # g^11 = -1
            ricci_components[:, :, :, 2, 2] -  # g^22 = -1
            ricci_components[:, :, :, 3, 3]    # g^33 = -1
        )
        
        # Average scalar curvature
        scalar_curvature = np.mean(scalar_curvature_field)
        
        # For visualization/analysis, we often want the Ricci tensor at the center point
        center_x, center_y, center_z = nx // 2, ny // 2, nz // 2
        ricci_tensor_center = ricci_components[center_x, center_y, center_z]
        
        return InformationCurvatureTensor(
            ricci_tensor=ricci_tensor_center,
            scalar_curvature=float(scalar_curvature),
            information_density=information_field,
            gradient_field=gradient_4d,
            coupling_strength=self.alpha
        )
    
    def calculate_curvature_from_quantum_state(
        self,
        quantum_states: Dict[str, Any],
        grid_size: Tuple[int, int, int] = (32, 32, 32)
    ) -> InformationCurvatureTensor:
        """
        Calculate curvature from quantum state information.
        
        Args:
            quantum_states: Dictionary of quantum states with their properties
            grid_size: Spatial discretization grid
            
        Returns:
            InformationCurvatureTensor with calculated components
        """
        nx, ny, nz = grid_size
        
        # Create information density field from quantum states
        information_field = np.zeros(grid_size)
        
        # Map quantum states to spatial grid
        # This is a simplified mapping - in practice would use more sophisticated methods
        for i, (state_name, state) in enumerate(quantum_states.items()):
            # Handle state as dictionary
            if isinstance(state, dict):
                coherence = state.get('coherence', 0.0)
                entropy = state.get('entropy', 0.0)
                # Information density proportional to coherence and inversely to entropy
                info_density = coherence * (1.0 - entropy + 1e-6)
            elif hasattr(state, 'coherence') and hasattr(state, 'entropy'):
                # Fallback for objects with attributes
                info_density = state.coherence * (1.0 - state.entropy + 1e-6)
            else:
                continue
                
            # Map to spatial location (simple grid assignment)
            x = (i * 7) % nx  # Pseudo-random distribution
            y = (i * 13) % ny
            z = (i * 23) % nz
            
            # Add Gaussian blob of information
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    for dz in range(-2, 3):
                        xi = (x + dx) % nx
                        yi = (y + dy) % ny
                        zi = (z + dz) % nz
                        
                        distance = np.sqrt(dx**2 + dy**2 + dz**2)
                        weight = np.exp(-distance**2 / 2.0)
                        
                        information_field[xi, yi, zi] += info_density * weight
        
        # Normalize information field
        if np.max(information_field) > 0:
            information_field /= np.max(information_field)
        
        # Calculate curvature from information field
        return self.calculate_curvature_from_information(
            information_field,
            grid_size
        )
    
    def calculate_gravitational_wave_strain(
        self,
        curvature_tensor: InformationCurvatureTensor,
        distance: float,
        frequency: float = 100.0  # Hz
    ) -> float:
        """
        Calculate gravitational wave strain from information curvature.
        
        This implements the OSH prediction that rapid changes in information
        density can produce gravitational waves.
        
        Args:
            curvature_tensor: Calculated curvature tensor
            distance: Distance to source (meters)
            frequency: Characteristic frequency of information changes (Hz)
            
        Returns:
            Dimensionless strain amplitude h
        """
        # Extract spatial components of Ricci tensor (source of GWs)
        ricci_spatial = curvature_tensor.ricci_tensor[1:, 1:]
        
        # Calculate quadrupole moment from curvature
        # Q ~ ∫ R_ij x^i x^j d³x (simplified)
        quadrupole_amplitude = np.sum(np.abs(ricci_spatial)) * self.grid_spacing**3
        
        # Gravitational wave strain formula
        # h ~ (G/c⁴) * (d²Q/dt²) / r
        # With d²Q/dt² ~ ω²Q for oscillating source
        
        omega = 2 * np.pi * frequency
        strain = (GRAVITATIONAL_CONSTANT / SPEED_OF_LIGHT**4) * \
                (omega**2 * quadrupole_amplitude) / distance
        
        # Apply information-gravity coupling enhancement
        strain *= np.sqrt(self.alpha / (8 * np.pi))  # Enhancement factor
        
        return float(strain)
    
    def visualize_curvature(
        self,
        curvature_tensor: InformationCurvatureTensor,
        slice_index: int = None
    ) -> Dict[str, np.ndarray]:
        """
        Prepare curvature data for visualization.
        
        Args:
            curvature_tensor: Calculated curvature tensor
            slice_index: Z-slice to visualize (None for center)
            
        Returns:
            Dictionary of 2D arrays for visualization
        """
        if slice_index is None:
            slice_index = curvature_tensor.information_density.shape[2] // 2
        
        # Extract 2D slices
        info_slice = curvature_tensor.information_density[:, :, slice_index]
        
        # Calculate curvature invariants for visualization
        # Kretschmann scalar: K = R_μνρσ R^μνρσ
        kretschmann = np.sum(curvature_tensor.ricci_tensor**2)
        
        # Gradient magnitude
        grad_magnitude = np.sqrt(
            np.sum(curvature_tensor.gradient_field[:, :, slice_index, 1:]**2, axis=-1)
        )
        
        return {
            'information_density': info_slice,
            'gradient_magnitude': grad_magnitude,
            'scalar_curvature': curvature_tensor.scalar_curvature,
            'kretschmann_scalar': kretschmann,
            'coupling_strength': curvature_tensor.coupling_strength
        }


def calculate_information_curvature_coupling(
    integrated_information: float,
    spatial_extent: float,
    time_scale: float = 1.0
) -> float:
    """
    Quick calculation of curvature-information coupling strength.
    
    Args:
        integrated_information: Φ value (bits)
        spatial_extent: Characteristic size (meters)
        time_scale: Characteristic time (seconds)
        
    Returns:
        Coupling strength in geometric units
    """
    # Information density
    info_density = integrated_information / (spatial_extent**3)
    
    # Apply coupling constant
    alpha = FieldParameters.INFORMATION_GRAVITY_COUPLING  # 8π
    
    # Curvature scale ~ α * I / L²
    curvature_scale = alpha * info_density / spatial_extent**2
    
    return float(curvature_scale)