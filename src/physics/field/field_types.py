"""
Recursia Field Type System - Enterprise Implementation

This module defines all quantum field types, their properties, validation logic,
and type-specific behaviors for the Recursia physics engine. It provides a
comprehensive type system for scalar, vector, spinor, tensor, gauge, and
composite field types with full OSH integration.

Author: Johnie Waddell
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import numpy as np
import logging
from collections import defaultdict
import threading

from src.core.data_classes import FieldCategory, FieldProperties

logger = logging.getLogger(__name__)


@dataclass
class FieldConfiguration:
    """Configuration for field initialization and evolution."""
    grid_size: Tuple[int, ...]
    dt: float = 0.01
    boundary_conditions: str = "periodic"
    initial_conditions: Optional[Dict[str, Any]] = None
    coupling_constants: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        if self.initial_conditions is None:
            self.initial_conditions = {}
        if self.coupling_constants is None:
            self.coupling_constants = {}


class FieldTypeDefinition(ABC):
    """Abstract base class for field type definitions."""
    
    def __init__(self, name: str, category: FieldCategory):
        self.name = name
        self.category = category
        self.properties = FieldProperties()
        self._lock = threading.RLock()
    
    @abstractmethod
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        """Get the default shape for this field type given grid dimensions."""
        pass
    
    @abstractmethod
    def validate_values(self, values: np.ndarray) -> bool:
        """Validate field values for this type."""
        pass
    
    @abstractmethod
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """Initialize field values with appropriate defaults."""
        pass
    
    @abstractmethod
    def get_component_names(self) -> List[str]:
        """Get names of field components."""
        pass
    
    @abstractmethod
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate energy density for this field type."""
        pass
    
    def get_evolution_operator_shape(self, grid_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Get shape of evolution operator for this field type."""
        field_shape = self.get_default_shape(grid_shape)
        return field_shape
    
    def supports_coupling(self, other_type: 'FieldTypeDefinition') -> bool:
        """Check if this field type can couple with another."""
        # Default coupling rules - can be overridden
        if self.category == other_type.category:
            return True
        
        # Cross-category coupling rules
        coupling_matrix = {
            (FieldCategory.SCALAR, FieldCategory.VECTOR): True,
            (FieldCategory.VECTOR, FieldCategory.TENSOR): True,
            (FieldCategory.GAUGE, FieldCategory.SPINOR): True,
            (FieldCategory.COMPOSITE, FieldCategory.SCALAR): True,
            (FieldCategory.COMPOSITE, FieldCategory.VECTOR): True,
        }
        
        pair = (self.category, other_type.category)
        reverse_pair = (other_type.category, self.category)
        
        return coupling_matrix.get(pair, False) or coupling_matrix.get(reverse_pair, False)


class ScalarFieldType(FieldTypeDefinition):
    """Real scalar field φ."""
    
    def __init__(self):
        super().__init__("scalar_field", FieldCategory.SCALAR)
        self.properties.evolution_type = "wave_equation"
        self.properties.spin = 0.0
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (grid_dimensions,)
        return tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return np.isreal(values).all() and np.isfinite(values).all()
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        initialization = kwargs.get('initialization', 'zero')
        
        if initialization == 'zero':
            return np.zeros(shape, dtype=np.float64)
        elif initialization == 'gaussian':
            amplitude = kwargs.get('amplitude', 1.0)
            sigma = kwargs.get('sigma', shape[0] // 4)
            center = kwargs.get('center', tuple(s // 2 for s in shape))
            
            field = np.zeros(shape, dtype=np.float64)
            if len(shape) == 1:
                x = np.arange(shape[0])
                field = amplitude * np.exp(-((x - center[0]) ** 2) / (2 * sigma ** 2))
            elif len(shape) == 2:
                x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
                r_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
                field = amplitude * np.exp(-r_sq / (2 * sigma ** 2))
            elif len(shape) == 3:
                x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), 
                                     np.arange(shape[2]), indexing='ij')
                r_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2 + (z - center[2]) ** 2
                field = amplitude * np.exp(-r_sq / (2 * sigma ** 2))
            
            return field
        elif initialization == 'random':
            amplitude = kwargs.get('amplitude', 0.1)
            return amplitude * np.random.randn(*shape)
        elif initialization == 'plane_wave':
            k = kwargs.get('k', [1.0])
            amplitude = kwargs.get('amplitude', 1.0)
            phase = kwargs.get('phase', 0.0)
            
            if len(shape) == 1:
                x = np.arange(shape[0]) * self.properties.spatial_step
                return amplitude * np.cos(k[0] * x + phase)
            else:
                # Multi-dimensional plane wave
                coords = np.meshgrid(*[np.arange(s) * self.properties.spatial_step for s in shape], indexing='ij')
                k_dot_r = sum(k[i] * coords[i] for i in range(min(len(k), len(coords))))
                return amplitude * np.cos(k_dot_r + phase)
        else:
            raise ValueError(f"Unknown initialization type: {initialization}")
    
    def get_component_names(self) -> List[str]:
        return ["φ"]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate energy density: (1/2)(∂φ/∂t)² + (1/2)∇φ² + V(φ)"""
        energy = 0.5 * values ** 2  # Potential energy (assuming harmonic)
        
        if derivatives:
            if 'time' in derivatives:
                energy += 0.5 * derivatives['time'] ** 2
            
            # Kinetic energy from spatial derivatives
            for i, deriv_key in enumerate(['dx', 'dy', 'dz']):
                if deriv_key in derivatives:
                    energy += 0.5 * derivatives[deriv_key] ** 2
        
        return energy


class ComplexScalarFieldType(FieldTypeDefinition):
    """Complex scalar field φ = φ_r + iφ_i."""
    
    def __init__(self):
        super().__init__("complex_scalar_field", FieldCategory.SCALAR)
        self.properties.evolution_type = "schrodinger_equation"
        self.properties.spin = 0.0
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (grid_dimensions,)
        return tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return np.isfinite(values).all()
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        initialization = kwargs.get('initialization', 'zero')
        
        if initialization == 'zero':
            return np.zeros(shape, dtype=np.complex128)
        elif initialization == 'gaussian_wave_packet':
            amplitude = kwargs.get('amplitude', 1.0)
            sigma = kwargs.get('sigma', shape[0] // 4)
            k0 = kwargs.get('k0', 1.0)
            center = kwargs.get('center', tuple(s // 2 for s in shape))
            
            if len(shape) == 1:
                x = np.arange(shape[0]) * self.properties.spatial_step
                envelope = np.exp(-((x - center[0] * self.properties.spatial_step) ** 2) / (2 * sigma ** 2))
                wave = np.exp(1j * k0 * x)
                return amplitude * envelope * wave
            else:
                # Multi-dimensional Gaussian wave packet
                coords = np.meshgrid(*[np.arange(s) * self.properties.spatial_step for s in shape], indexing='ij')
                r_sq = sum((coords[i] - center[i] * self.properties.spatial_step) ** 2 for i in range(len(coords)))
                envelope = np.exp(-r_sq / (2 * sigma ** 2))
                k0_vec = kwargs.get('k0', [k0] * len(shape))
                k_dot_r = sum(k0_vec[i] * coords[i] for i in range(len(coords)))
                wave = np.exp(1j * k_dot_r)
                return amplitude * envelope * wave
        elif initialization == 'plane_wave':
            k = kwargs.get('k', [1.0])
            amplitude = kwargs.get('amplitude', 1.0)
            phase = kwargs.get('phase', 0.0)
            
            coords = np.meshgrid(*[np.arange(s) * self.properties.spatial_step for s in shape], indexing='ij')
            k_dot_r = sum(k[i] * coords[i] for i in range(min(len(k), len(coords))))
            return amplitude * np.exp(1j * (k_dot_r + phase))
        else:
            return self._real_initialization(shape, **kwargs).astype(np.complex128)
    
    def _real_initialization(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        """Helper for real-valued initializations."""
        scalar_field = ScalarFieldType()
        scalar_field.properties = self.properties
        return scalar_field.initialize_values(shape, **kwargs)
    
    def get_component_names(self) -> List[str]:
        return ["φ_real", "φ_imag"]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate energy density for complex scalar field."""
        energy = np.real(values * np.conj(values))  # |φ|²
        
        if derivatives:
            # Add kinetic energy terms
            for deriv_key in ['dx', 'dy', 'dz']:
                if deriv_key in derivatives:
                    energy += np.real(derivatives[deriv_key] * np.conj(derivatives[deriv_key]))
        
        return energy


class VectorFieldType(FieldTypeDefinition):
    """3-component vector field A⃗."""
    
    def __init__(self):
        super().__init__("vector_field", FieldCategory.VECTOR)
        self.properties.evolution_type = "wave_equation"
        self.properties.spin = 1.0
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (3, grid_dimensions)
        return (3,) + tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return (values.shape[0] == 3 and np.isreal(values).all() and 
                np.isfinite(values).all())
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        if shape[0] != 3:
            raise ValueError("Vector field must have 3 components")
        
        initialization = kwargs.get('initialization', 'zero')
        
        if initialization == 'zero':
            return np.zeros(shape, dtype=np.float64)
        elif initialization == 'random':
            amplitude = kwargs.get('amplitude', 0.1)
            return amplitude * np.random.randn(*shape)
        elif initialization == 'dipole':
            # Electric dipole field
            moment = kwargs.get('moment', [0.0, 0.0, 1.0])
            center = kwargs.get('center', tuple(s // 2 for s in shape[1:]))
            
            field = np.zeros(shape, dtype=np.float64)
            if len(shape) == 3:  # 1D case
                x = np.arange(shape[1]) - center[0]
                r = np.abs(x) + 1e-10  # Avoid division by zero
                field[2, :] = moment[2] / (r ** 2)  # Simplified 1D dipole
            else:
                # Multi-dimensional dipole field
                coords = np.meshgrid(*[np.arange(s) - c for s, c in zip(shape[1:], center)], indexing='ij')
                r_vec = np.array(coords)
                r_mag = np.sqrt(np.sum(r_vec ** 2, axis=0)) + 1e-10
                
                # Dipole field: E = (3(p⃗·r̂)r̂ - p⃗) / r³
                p = np.array(moment)
                for i in range(3):
                    field[i] = p[i] / (r_mag ** 3)  # Simplified
            
            return field
        elif initialization == 'uniform':
            direction = kwargs.get('direction', [1.0, 0.0, 0.0])
            amplitude = kwargs.get('amplitude', 1.0)
            
            field = np.zeros(shape, dtype=np.float64)
            for i in range(3):
                field[i] = amplitude * direction[i]
            
            return field
        else:
            raise ValueError(f"Unknown initialization type: {initialization}")
    
    def get_component_names(self) -> List[str]:
        return ["A_x", "A_y", "A_z"]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate energy density: (1/2)|A⃗|²"""
        energy = 0.5 * np.sum(values ** 2, axis=0)
        
        if derivatives:
            # Add gradient energy
            for deriv_key in ['dx', 'dy', 'dz']:
                if deriv_key in derivatives:
                    energy += 0.5 * np.sum(derivatives[deriv_key] ** 2, axis=0)
        
        return energy


class SpinorFieldType(FieldTypeDefinition):
    """4-component Dirac spinor field ψ."""
    
    def __init__(self):
        super().__init__("spinor_field", FieldCategory.SPINOR)
        self.properties.evolution_type = "dirac_equation"
        self.properties.spin = 0.5
        self.properties.mass = 1.0  # Default fermion mass
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (4, grid_dimensions)
        return (4,) + tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return (values.shape[0] == 4 and np.isfinite(values).all())
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        if shape[0] != 4:
            raise ValueError("Spinor field must have 4 components")
        
        initialization = kwargs.get('initialization', 'zero')
        
        if initialization == 'zero':
            return np.zeros(shape, dtype=np.complex128)
        elif initialization == 'spin_up':
            field = np.zeros(shape, dtype=np.complex128)
            field[0] = 1.0  # |↑⟩ state
            return field
        elif initialization == 'spin_down':
            field = np.zeros(shape, dtype=np.complex128)
            field[1] = 1.0  # |↓⟩ state
            return field
        elif initialization == 'plane_wave_spinor':
            amplitude = kwargs.get('amplitude', 1.0)
            k = kwargs.get('k', [1.0])
            spin_state = kwargs.get('spin_state', 'up')
            
            coords = np.meshgrid(*[np.arange(s) * self.properties.spatial_step for s in shape[1:]], indexing='ij')
            k_dot_r = sum(k[i] * coords[i] for i in range(min(len(k), len(coords))))
            plane_wave = amplitude * np.exp(1j * k_dot_r)
            
            field = np.zeros(shape, dtype=np.complex128)
            if spin_state == 'up':
                field[0] = plane_wave
            elif spin_state == 'down':
                field[1] = plane_wave
            else:
                # Superposition
                field[0] = plane_wave / np.sqrt(2)
                field[1] = plane_wave / np.sqrt(2)
            
            return field
        else:
            return np.zeros(shape, dtype=np.complex128)
    
    def get_component_names(self) -> List[str]:
        return ["ψ₁", "ψ₂", "ψ₃", "ψ₄"]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate energy density for Dirac spinor."""
        # Probability density |ψ†ψ|
        energy = np.real(np.sum(np.conj(values) * values, axis=0))
        
        if derivatives and self.properties.mass > 0:
            # Add mass term
            energy += self.properties.mass * energy
        
        return energy


class TensorFieldType(FieldTypeDefinition):
    """Rank-2 tensor field (e.g., graviton, stress tensor)."""
    
    def __init__(self):
        super().__init__("tensor_field", FieldCategory.TENSOR)
        self.properties.evolution_type = "wave_equation"
        self.properties.spin = 2.0
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (3, 3, grid_dimensions)
        return (3, 3) + tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return (values.shape[0] == 3 and values.shape[1] == 3 and 
                np.isreal(values).all() and np.isfinite(values).all())
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        if shape[0] != 3 or shape[1] != 3:
            raise ValueError("Tensor field must have 3x3 components")
        
        initialization = kwargs.get('initialization', 'zero')
        
        if initialization == 'zero':
            return np.zeros(shape, dtype=np.float64)
        elif initialization == 'identity':
            field = np.zeros(shape, dtype=np.float64)
            field[0, 0] = 1.0
            field[1, 1] = 1.0
            field[2, 2] = 1.0
            return field
        elif initialization == 'symmetric_random':
            amplitude = kwargs.get('amplitude', 0.1)
            field = amplitude * np.random.randn(*shape)
            # Make symmetric
            for i in range(shape[0]):
                for j in range(i + 1, shape[1]):
                    field[j, i] = field[i, j]
            return field
        else:
            return np.zeros(shape, dtype=np.float64)
    
    def get_component_names(self) -> List[str]:
        return [f"T_{i}{j}" for i in range(3) for j in range(3)]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate energy density for tensor field."""
        # Trace of tensor squared
        energy = np.zeros(values.shape[2:])
        for i in range(3):
            for j in range(3):
                energy += 0.5 * values[i, j] ** 2
        
        return energy


class GaugeFieldType(FieldTypeDefinition):
    """4-vector gauge field A^μ."""
    
    def __init__(self):
        super().__init__("gauge_field", FieldCategory.GAUGE)
        self.properties.evolution_type = "maxwell_equations"
        self.properties.spin = 1.0
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (4, grid_dimensions)
        return (4,) + tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return (values.shape[0] == 4 and np.isreal(values).all() and 
                np.isfinite(values).all())
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        if shape[0] != 4:
            raise ValueError("Gauge field must have 4 components")
        
        initialization = kwargs.get('initialization', 'zero')
        
        if initialization == 'zero':
            return np.zeros(shape, dtype=np.float64)
        elif initialization == 'coulomb_gauge':
            # ∇·A = 0, A₀ = φ (scalar potential)
            field = np.zeros(shape, dtype=np.float64)
            # Set scalar potential
            potential = kwargs.get('potential', 1.0)
            field[0] = potential
            return field
        elif initialization == 'electromagnetic_wave':
            amplitude = kwargs.get('amplitude', 1.0)
            k = kwargs.get('k', [1.0])
            polarization = kwargs.get('polarization', [0.0, 1.0, 0.0])  # y-polarized
            
            coords = np.meshgrid(*[np.arange(s) * self.properties.spatial_step for s in shape[1:]], indexing='ij')
            k_dot_r = sum(k[i] * coords[i] for i in range(min(len(k), len(coords))))
            wave = amplitude * np.cos(k_dot_r)
            
            field = np.zeros(shape, dtype=np.float64)
            # Set vector potential components
            for i in range(1, 4):  # A₁, A₂, A₃
                if i - 1 < len(polarization):
                    field[i] = polarization[i - 1] * wave
            
            return field
        else:
            return np.zeros(shape, dtype=np.float64)
    
    def get_component_names(self) -> List[str]:
        return ["A₀", "A₁", "A₂", "A₃"]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate electromagnetic energy density."""
        # Simple approximation: |E|² + |B|²
        energy = 0.5 * np.sum(values[1:4] ** 2, axis=0)  # Vector potential contribution
        
        if derivatives:
            # Add field strength tensor contributions
            for deriv_key in ['dx', 'dy', 'dz']:
                if deriv_key in derivatives:
                    energy += 0.5 * np.sum(derivatives[deriv_key][1:4] ** 2, axis=0)
        
        return energy


class CompositeFieldType(FieldTypeDefinition):
    """Composite field built from multiple component fields."""
    
    def __init__(self, component_types: List[FieldTypeDefinition], name: str = "composite_field"):
        super().__init__(name, FieldCategory.COMPOSITE)
        self.component_types = component_types
        self.properties.evolution_type = "composite"
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        # Total size is sum of all component shapes
        total_components = 0
        for comp_type in self.component_types:
            comp_shape = comp_type.get_default_shape(grid_dimensions)
            total_components += comp_shape[0] if len(comp_shape) > 1 else 1
        
        if isinstance(grid_dimensions, int):
            return (total_components, grid_dimensions)
        return (total_components,) + tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        # Validate each component separately
        offset = 0
        for comp_type in self.component_types:
            comp_shape = comp_type.get_default_shape(values.shape[1:])
            comp_size = comp_shape[0] if len(comp_shape) > 1 else 1
            
            comp_values = values[offset:offset + comp_size]
            if not comp_type.validate_values(comp_values):
                return False
            
            offset += comp_size
        
        return True
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        field = np.zeros(shape, dtype=np.complex128)  # Use complex to accommodate all types
        
        offset = 0
        for i, comp_type in enumerate(self.component_types):
            comp_grid_dims = shape[1:] if len(shape) > 1 else shape[0]
            comp_shape = comp_type.get_default_shape(comp_grid_dims)
            comp_size = comp_shape[0] if len(comp_shape) > 1 else 1
            
            comp_kwargs = kwargs.get(f'component_{i}', {})
            comp_values = comp_type.initialize_values(comp_shape, **comp_kwargs)
            
            field[offset:offset + comp_size] = comp_values
            offset += comp_size
        
        return field
    
    def get_component_names(self) -> List[str]:
        names = []
        for i, comp_type in enumerate(self.component_types):
            comp_names = comp_type.get_component_names()
            names.extend([f"{comp_type.name}_{name}" for name in comp_names])
        return names
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        energy = np.zeros(values.shape[1:])
        
        offset = 0
        for comp_type in self.component_types:
            comp_shape = comp_type.get_default_shape(values.shape[1:])
            comp_size = comp_shape[0] if len(comp_shape) > 1 else 1
            
            comp_values = values[offset:offset + comp_size]
            comp_derivatives = {}
            if derivatives:
                for key, deriv in derivatives.items():
                    comp_derivatives[key] = deriv[offset:offset + comp_size]
            
            energy += comp_type.calculate_energy_density(comp_values, comp_derivatives)
            offset += comp_size
        
        return energy


class ProbabilityFieldType(FieldTypeDefinition):
    """Normalized probabilistic field."""
    
    def __init__(self):
        super().__init__("probability_field", FieldCategory.PROBABILITY)
        self.properties.evolution_type = "diffusion_equation"
    
    def get_default_shape(self, grid_dimensions: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
        if isinstance(grid_dimensions, int):
            return (grid_dimensions,)
        return tuple(grid_dimensions)
    
    def validate_values(self, values: np.ndarray) -> bool:
        return (np.all(values >= 0) and np.isfinite(values).all() and 
                np.abs(np.sum(values) - 1.0) < 1e-6)
    
    def initialize_values(self, shape: Tuple[int, ...], **kwargs) -> np.ndarray:
        initialization = kwargs.get('initialization', 'uniform')
        
        if initialization == 'uniform':
            field = np.ones(shape, dtype=np.float64)
            field /= np.sum(field)  # Normalize
            return field
        elif initialization == 'gaussian':
            sigma = kwargs.get('sigma', shape[0] // 4)
            center = kwargs.get('center', tuple(s // 2 for s in shape))
            
            if len(shape) == 1:
                x = np.arange(shape[0])
                field = np.exp(-((x - center[0]) ** 2) / (2 * sigma ** 2))
            else:
                coords = np.meshgrid(*[np.arange(s) for s in shape], indexing='ij')
                r_sq = sum((coords[i] - center[i]) ** 2 for i in range(len(coords)))
                field = np.exp(-r_sq / (2 * sigma ** 2))
            
            field /= np.sum(field)  # Normalize
            return field
        elif initialization == 'delta':
            center = kwargs.get('center', tuple(s // 2 for s in shape))
            field = np.zeros(shape, dtype=np.float64)
            field[center] = 1.0
            return field
        else:
            raise ValueError(f"Unknown initialization type: {initialization}")
    
    def get_component_names(self) -> List[str]:
        return ["P"]
    
    def calculate_energy_density(self, values: np.ndarray, 
                                derivatives: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        """Calculate entropy density: -P log P"""
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        return -values * np.log(values + epsilon)


class FieldTypeRegistry:
    """Registry for all field types with thread-safe operations."""
    
    def __init__(self):
        self._types: Dict[str, FieldTypeDefinition] = {}
        self._lock = threading.RLock()
        self._register_builtin_types()
    
    def _register_builtin_types(self):
        """Register all built-in field types."""
        builtin_types = [
            ScalarFieldType(),
            ComplexScalarFieldType(),
            VectorFieldType(),
            SpinorFieldType(),
            TensorFieldType(),
            GaugeFieldType(),
            ProbabilityFieldType()
        ]
        
        for field_type in builtin_types:
            self._types[field_type.name] = field_type
    
    def register_type(self, field_type: FieldTypeDefinition) -> bool:
        """Register a new field type."""
        with self._lock:
            if field_type.name in self._types:
                logger.warning(f"Field type {field_type.name} already registered, overwriting")
            
            self._types[field_type.name] = field_type
            logger.info(f"Registered field type: {field_type.name}")
            return True
    
    def get_type(self, name: str) -> Optional[FieldTypeDefinition]:
        """Get a field type by name."""
        with self._lock:
            return self._types.get(name)
    
    def list_types(self) -> List[str]:
        """List all registered field type names."""
        with self._lock:
            return list(self._types.keys())
    
    def get_types_by_category(self, category: FieldCategory) -> List[FieldTypeDefinition]:
        """Get all field types in a category."""
        with self._lock:
            return [ft for ft in self._types.values() if ft.category == category]
    
    def create_composite_type(self, component_names: List[str], 
                             composite_name: str) -> Optional[CompositeFieldType]:
        """Create a composite field type from existing types."""
        with self._lock:
            component_types = []
            for name in component_names:
                if name not in self._types:
                    logger.error(f"Component type {name} not found")
                    return None
                component_types.append(self._types[name])
            
            composite = CompositeFieldType(component_types, composite_name)
            self.register_type(composite)
            return composite
    
    def validate_type_compatibility(self, type1_name: str, type2_name: str) -> bool:
        """Check if two field types can couple."""
        with self._lock:
            type1 = self._types.get(type1_name)
            type2 = self._types.get(type2_name)
            
            if not type1 or not type2:
                return False
            
            return type1.supports_coupling(type2)
    
    def get_coupling_strength(self, type1_name: str, type2_name: str) -> float:
        """Get default coupling strength between two field types."""
        if not self.validate_type_compatibility(type1_name, type2_name):
            return 0.0
        
        # Default coupling strengths based on category
        coupling_matrix = {
            (FieldCategory.SCALAR, FieldCategory.SCALAR): 1.0,
            (FieldCategory.SCALAR, FieldCategory.VECTOR): 0.8,
            (FieldCategory.VECTOR, FieldCategory.VECTOR): 0.9,
            (FieldCategory.GAUGE, FieldCategory.SPINOR): 1.0,
            (FieldCategory.COMPOSITE, FieldCategory.SCALAR): 0.5,
        }
        
        with self._lock:
            type1 = self._types[type1_name]
            type2 = self._types[type2_name]
            
            pair = (type1.category, type2.category)
            reverse_pair = (type2.category, type1.category)
            
            return coupling_matrix.get(pair, coupling_matrix.get(reverse_pair, 0.5))


# Global registry instance
field_type_registry = FieldTypeRegistry()


def get_field_type(name: str) -> Optional[FieldTypeDefinition]:
    """Convenience function to get a field type."""
    return field_type_registry.get_type(name)


def register_field_type(field_type: FieldTypeDefinition) -> bool:
    """Convenience function to register a field type."""
    return field_type_registry.register_type(field_type)


def list_field_types() -> List[str]:
    """Convenience function to list all field types."""
    return field_type_registry.list_types()


def create_composite_field_type(component_names: List[str], 
                               composite_name: str) -> Optional[CompositeFieldType]:
    """Convenience function to create composite field type."""
    return field_type_registry.create_composite_type(component_names, composite_name)