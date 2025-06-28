"""
field_compute.py - Recursia Field Computational Engine

This module provides the core numerical infrastructure for evolving quantum fields
across space and time. It encapsulates PDE solvers, differential operators, 
stability checking, and high-performance computational backends for the OSH framework.

Supports:
- Wave, diffusion, Schrödinger, Maxwell, and Dirac equation solvers
- High-performance Laplacian and gradient operators
- Spectral and finite difference derivative engines
- Custom boundary conditions and CFL-based stability
- Implicit/explicit integration schemes
- GPU acceleration and sparse matrix optimizations
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.fft import fft, ifft, fft2, ifft2, fftn, ifftn
from scipy.integrate import solve_ivp
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum
import threading
import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

from src.core.data_classes import BoundaryCondition, ComputationalParameters, IntegrationScheme, NumericalMethod

try:
    import cupy as cp # type: ignore
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class DifferentialOperator(ABC):
    """Abstract base class for differential operators."""
    
    def __init__(self, params: ComputationalParameters):
        self.params = params
        self._cache = {}
        self._lock = threading.RLock()
    
    @abstractmethod
    def construct_matrix(self, shape: Tuple[int, ...], **kwargs) -> Union[np.ndarray, sp.spmatrix]:
        """Construct the differential operator matrix."""
        pass
    
    @abstractmethod
    def apply(self, field: np.ndarray, **kwargs) -> np.ndarray:
        """Apply the operator to a field."""
        pass
    
    def clear_cache(self):
        """Clear operator cache."""
        with self._lock:
            self._cache.clear()


class LaplacianOperator(DifferentialOperator):
    """Laplacian operator for various dimensions and boundary conditions."""
    
    def construct_matrix(self, shape: Tuple[int, ...], **kwargs) -> Union[np.ndarray, sp.spmatrix]:
        """Construct Laplacian matrix for given shape."""
        cache_key = (shape, self.params.boundary_condition, self.params.finite_difference_order)
        
        with self._lock:
            if cache_key in self._cache and self.params.cache_operators:
                return self._cache[cache_key]
        
        if len(shape) == 1:
            matrix = self._construct_1d_laplacian(shape[0])
        elif len(shape) == 2:
            matrix = self._construct_2d_laplacian(shape)
        elif len(shape) == 3:
            matrix = self._construct_3d_laplacian(shape)
        else:
            raise ValueError(f"Unsupported dimension: {len(shape)}")
        
        with self._lock:
            if self.params.cache_operators and len(self._cache) < self.params.max_cache_size:
                self._cache[cache_key] = matrix
        
        return matrix
    
    def _construct_1d_laplacian(self, n: int) -> sp.spmatrix:
        """Construct 1D Laplacian matrix."""
        dx2 = self.params.dx ** 2
        
        if self.params.finite_difference_order == 2:
            # Second-order central difference
            diagonals = [np.ones(n-1), -2*np.ones(n), np.ones(n-1)]
            offsets = [-1, 0, 1]
        elif self.params.finite_difference_order == 4:
            # Fourth-order central difference
            diagonals = [
                -np.ones(n-2)/12, 4*np.ones(n-1)/3, -5*np.ones(n)/2,
                4*np.ones(n-1)/3, -np.ones(n-2)/12
            ]
            offsets = [-2, -1, 0, 1, 2]
        else:
            raise ValueError(f"Unsupported finite difference order: {self.params.finite_difference_order}")
        
        matrix = sp.diags(diagonals, offsets, shape=(n, n), format='csr') / dx2
        
        # Apply boundary conditions
        if self.params.boundary_condition == BoundaryCondition.PERIODIC:
            self._apply_periodic_bc_1d(matrix, n)
        elif self.params.boundary_condition == BoundaryCondition.DIRICHLET:
            # Homogeneous Dirichlet (zero at boundaries)
            matrix[0, :] = 0
            matrix[-1, :] = 0
        elif self.params.boundary_condition == BoundaryCondition.NEUMANN:
            # Zero derivative at boundaries
            self._apply_neumann_bc_1d(matrix, n)
        
        return matrix
    
    def _construct_2d_laplacian(self, shape: Tuple[int, int]) -> sp.spmatrix:
        """Construct 2D Laplacian matrix."""
        ny, nx = shape
        
        # Construct 1D operators
        Lx = self._construct_1d_laplacian(nx)
        Ly = self._construct_1d_laplacian(ny)
        
        # Create identity matrices
        Ix = sp.identity(nx)
        Iy = sp.identity(ny)
        
        # 2D Laplacian = Ly ⊗ Ix + Iy ⊗ Lx
        L2d = sp.kron(Iy, Lx) + sp.kron(Ly, Ix)
        
        return L2d.tocsr()
    
    def _construct_3d_laplacian(self, shape: Tuple[int, int, int]) -> sp.spmatrix:
        """Construct 3D Laplacian matrix."""
        nz, ny, nx = shape
        
        # Construct 1D operators
        Lx = self._construct_1d_laplacian(nx)
        Ly = self._construct_1d_laplacian(ny)
        Lz = self._construct_1d_laplacian(nz)
        
        # Create identity matrices
        Ix = sp.identity(nx)
        Iy = sp.identity(ny)
        Iz = sp.identity(nz)
        
        # 3D Laplacian
        L3d = (sp.kron(sp.kron(Iz, Iy), Lx) + 
               sp.kron(sp.kron(Iz, Ly), Ix) + 
               sp.kron(sp.kron(Lz, Iy), Ix))
        
        return L3d.tocsr()
    
    def _apply_periodic_bc_1d(self, matrix: sp.spmatrix, n: int):
        """Apply periodic boundary conditions to 1D matrix."""
        if self.params.finite_difference_order == 2:
            matrix[0, -1] = 1 / self.params.dx**2
            matrix[-1, 0] = 1 / self.params.dx**2
        elif self.params.finite_difference_order == 4:
            matrix[0, -2:] = [-1/12, 4/3] / self.params.dx**2
            matrix[1, -1] = -1/12 / self.params.dx**2
            matrix[-1, :2] = [4/3, -1/12] / self.params.dx**2
            matrix[-2, 0] = -1/12 / self.params.dx**2
    
    def _apply_neumann_bc_1d(self, matrix: sp.spmatrix, n: int):
        """Apply Neumann boundary conditions to 1D matrix."""
        # Zero derivative: use one-sided differences
        matrix[0, 0] = -1 / self.params.dx**2
        matrix[0, 1] = 1 / self.params.dx**2
        matrix[-1, -2] = 1 / self.params.dx**2
        matrix[-1, -1] = -1 / self.params.dx**2
    
    def apply(self, field: np.ndarray, **kwargs) -> np.ndarray:
        """Apply Laplacian operator to field."""
        if self.params.method == NumericalMethod.SPECTRAL:
            return self._apply_spectral(field)
        else:
            matrix = self.construct_matrix(field.shape)
            field_flat = field.flatten()
            result_flat = matrix.dot(field_flat)
            return result_flat.reshape(field.shape)
    
    def _apply_spectral(self, field: np.ndarray) -> np.ndarray:
        """Apply Laplacian using spectral method."""
        if len(field.shape) == 1:
            return self._apply_spectral_1d(field)
        elif len(field.shape) == 2:
            return self._apply_spectral_2d(field)
        elif len(field.shape) == 3:
            return self._apply_spectral_3d(field)
        else:
            raise ValueError(f"Unsupported dimension: {len(field.shape)}")
    
    def _apply_spectral_1d(self, field: np.ndarray) -> np.ndarray:
        """1D spectral Laplacian."""
        n = field.shape[0]
        k = 2 * np.pi * np.fft.fftfreq(n, self.params.dx)
        field_hat = fft(field)
        laplacian_hat = -(k**2) * field_hat
        return np.real(ifft(laplacian_hat))
    
    def _apply_spectral_2d(self, field: np.ndarray) -> np.ndarray:
        """2D spectral Laplacian."""
        ny, nx = field.shape
        kx = 2 * np.pi * np.fft.fftfreq(nx, self.params.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, self.params.dy)
        
        Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
        k2 = Kx**2 + Ky**2
        
        field_hat = fft2(field)
        laplacian_hat = -k2 * field_hat
        return np.real(ifft2(laplacian_hat))
    
    def _apply_spectral_3d(self, field: np.ndarray) -> np.ndarray:
        """3D spectral Laplacian."""
        nz, ny, nx = field.shape
        kx = 2 * np.pi * np.fft.fftfreq(nx, self.params.dx)
        ky = 2 * np.pi * np.fft.fftfreq(ny, self.params.dy)
        kz = 2 * np.pi * np.fft.fftfreq(nz, self.params.dz)
        
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = Kx**2 + Ky**2 + Kz**2
        
        field_hat = fftn(field)
        laplacian_hat = -k2 * field_hat
        return np.real(ifftn(laplacian_hat))


class GradientOperator(DifferentialOperator):
    """Gradient operator for various dimensions and boundary conditions."""
    
    def construct_matrix(self, shape: Tuple[int, ...], component: int = 0, **kwargs) -> Union[np.ndarray, sp.spmatrix]:
        """Construct gradient matrix for given shape and component."""
        cache_key = (shape, component, self.params.boundary_condition, self.params.finite_difference_order)
        
        with self._lock:
            if cache_key in self._cache and self.params.cache_operators:
                return self._cache[cache_key]
        
        if len(shape) == 1:
            matrix = self._construct_1d_gradient(shape[0])
        elif len(shape) == 2:
            matrix = self._construct_2d_gradient(shape, component)
        elif len(shape) == 3:
            matrix = self._construct_3d_gradient(shape, component)
        else:
            raise ValueError(f"Unsupported dimension: {len(shape)}")
        
        with self._lock:
            if self.params.cache_operators and len(self._cache) < self.params.max_cache_size:
                self._cache[cache_key] = matrix
        
        return matrix
    
    def _construct_1d_gradient(self, n: int) -> sp.spmatrix:
        """Construct 1D gradient matrix."""
        dx = self.params.dx
        
        if self.params.finite_difference_order == 2:
            # Second-order central difference
            diagonals = [-np.ones(n-1)/2, np.ones(n-1)/2]
            offsets = [-1, 1]
        elif self.params.finite_difference_order == 4:
            # Fourth-order central difference
            diagonals = [
                np.ones(n-2)/12, -2*np.ones(n-1)/3, 
                2*np.ones(n-1)/3, -np.ones(n-2)/12
            ]
            offsets = [-2, -1, 1, 2]
        else:
            raise ValueError(f"Unsupported finite difference order: {self.params.finite_difference_order}")
        
        matrix = sp.diags(diagonals, offsets, shape=(n, n), format='csr') / dx
        
        # Apply boundary conditions
        if self.params.boundary_condition == BoundaryCondition.PERIODIC:
            self._apply_periodic_bc_gradient_1d(matrix, n)
        elif self.params.boundary_condition == BoundaryCondition.DIRICHLET:
            # Forward/backward differences at boundaries
            matrix[0, :3] = [-3, 4, -1] / (2 * dx)
            matrix[-1, -3:] = [1, -4, 3] / (2 * dx)
        
        return matrix
    
    def _construct_2d_gradient(self, shape: Tuple[int, int], component: int) -> sp.spmatrix:
        """Construct 2D gradient matrix for specified component."""
        ny, nx = shape
        
        if component == 0:  # x-component
            Gx = self._construct_1d_gradient(nx)
            Iy = sp.identity(ny)
            return sp.kron(Iy, Gx)
        elif component == 1:  # y-component
            Gy = self._construct_1d_gradient(ny)
            Ix = sp.identity(nx)
            return sp.kron(Gy, Ix)
        else:
            raise ValueError(f"Invalid component {component} for 2D gradient")
    
    def _construct_3d_gradient(self, shape: Tuple[int, int, int], component: int) -> sp.spmatrix:
        """Construct 3D gradient matrix for specified component."""
        nz, ny, nx = shape
        
        Ix = sp.identity(nx)
        Iy = sp.identity(ny)
        Iz = sp.identity(nz)
        
        if component == 0:  # x-component
            Gx = self._construct_1d_gradient(nx)
            return sp.kron(sp.kron(Iz, Iy), Gx)
        elif component == 1:  # y-component
            Gy = self._construct_1d_gradient(ny)
            return sp.kron(sp.kron(Iz, Gy), Ix)
        elif component == 2:  # z-component
            Gz = self._construct_1d_gradient(nz)
            return sp.kron(sp.kron(Gz, Iy), Ix)
        else:
            raise ValueError(f"Invalid component {component} for 3D gradient")
    
    def _apply_periodic_bc_gradient_1d(self, matrix: sp.spmatrix, n: int):
        """Apply periodic boundary conditions to 1D gradient matrix."""
        dx = self.params.dx
        if self.params.finite_difference_order == 2:
            matrix[0, -1] = -1 / (2 * dx)
            matrix[-1, 0] = 1 / (2 * dx)
        elif self.params.finite_difference_order == 4:
            matrix[0, -2:] = [-1/12, 2/3] / dx
            matrix[1, -1] = 1/12 / dx
            matrix[-1, :2] = [-2/3, 1/12] / dx
            matrix[-2, 0] = -1/12 / dx
    
    def apply(self, field: np.ndarray, **kwargs) -> np.ndarray:
        """Apply gradient operator to field."""
        if self.params.method == NumericalMethod.SPECTRAL:
            return self._apply_spectral(field)
        else:
            ndim = len(field.shape)
            gradient = np.zeros((ndim,) + field.shape)
            
            for i in range(ndim):
                matrix = self.construct_matrix(field.shape, component=i)
                field_flat = field.flatten()
                gradient_flat = matrix.dot(field_flat)
                gradient[i] = gradient_flat.reshape(field.shape)
            
            return gradient
    
    def _apply_spectral(self, field: np.ndarray) -> np.ndarray:
        """Apply gradient using spectral method."""
        ndim = len(field.shape)
        gradient = np.zeros((ndim,) + field.shape, dtype=complex)
        
        if ndim == 1:
            n = field.shape[0]
            k = 2j * np.pi * np.fft.fftfreq(n, self.params.dx)
            field_hat = fft(field)
            gradient[0] = np.real(ifft(k * field_hat))
            
        elif ndim == 2:
            ny, nx = field.shape
            kx = 2j * np.pi * np.fft.fftfreq(nx, self.params.dx)
            ky = 2j * np.pi * np.fft.fftfreq(ny, self.params.dy)
            
            field_hat = fft2(field)
            gradient[0] = np.real(ifft2(kx[None, :] * field_hat))
            gradient[1] = np.real(ifft2(ky[:, None] * field_hat))
            
        elif ndim == 3:
            nz, ny, nx = field.shape
            kx = 2j * np.pi * np.fft.fftfreq(nx, self.params.dx)
            ky = 2j * np.pi * np.fft.fftfreq(ny, self.params.dy)
            kz = 2j * np.pi * np.fft.fftfreq(nz, self.params.dz)
            
            field_hat = fftn(field)
            gradient[0] = np.real(ifftn(kx[None, None, :] * field_hat))
            gradient[1] = np.real(ifftn(ky[None, :, None] * field_hat))
            gradient[2] = np.real(ifftn(kz[:, None, None] * field_hat))
        
        return gradient


class SpectralOperator(DifferentialOperator):
    """High-accuracy spectral differential operator using FFT."""
    
    def construct_matrix(self, shape: Tuple[int, ...], **kwargs) -> Union[np.ndarray, sp.spmatrix]:
        """Spectral operators don't use explicit matrices."""
        raise NotImplementedError("Spectral operators apply directly to fields")
    
    def apply(self, field: np.ndarray, derivative_order: int = 1, component: int = 0, **kwargs) -> np.ndarray:
        """Apply spectral derivative operator."""
        if derivative_order == 1:
            return self._apply_first_derivative(field, component)
        elif derivative_order == 2:
            return self._apply_second_derivative(field, component)
        else:
            return self._apply_nth_derivative(field, derivative_order, component)
    
    def _apply_first_derivative(self, field: np.ndarray, component: int = 0) -> np.ndarray:
        """Apply first derivative using spectral method."""
        ndim = len(field.shape)
        
        if ndim == 1:
            n = field.shape[0]
            k = 2j * np.pi * np.fft.fftfreq(n, self.params.dx)
            field_hat = fft(field)
            return np.real(ifft(k * field_hat))
            
        elif ndim == 2:
            ny, nx = field.shape
            field_hat = fft2(field)
            
            if component == 0:  # x-derivative
                kx = 2j * np.pi * np.fft.fftfreq(nx, self.params.dx)
                return np.real(ifft2(kx[None, :] * field_hat))
            elif component == 1:  # y-derivative
                ky = 2j * np.pi * np.fft.fftfreq(ny, self.params.dy)
                return np.real(ifft2(ky[:, None] * field_hat))
                
        elif ndim == 3:
            nz, ny, nx = field.shape
            field_hat = fftn(field)
            
            if component == 0:  # x-derivative
                kx = 2j * np.pi * np.fft.fftfreq(nx, self.params.dx)
                return np.real(ifftn(kx[None, None, :] * field_hat))
            elif component == 1:  # y-derivative
                ky = 2j * np.pi * np.fft.fftfreq(ny, self.params.dy)
                return np.real(ifftn(ky[None, :, None] * field_hat))
            elif component == 2:  # z-derivative
                kz = 2j * np.pi * np.fft.fftfreq(nz, self.params.dz)
                return np.real(ifftn(kz[:, None, None] * field_hat))
        
        raise ValueError(f"Invalid component {component} for {ndim}D field")
    
    def _apply_second_derivative(self, field: np.ndarray, component: int = 0) -> np.ndarray:
        """Apply second derivative using spectral method."""
        ndim = len(field.shape)
        
        if ndim == 1:
            n = field.shape[0]
            k = 2j * np.pi * np.fft.fftfreq(n, self.params.dx)
            field_hat = fft(field)
            return np.real(ifft(-(k**2) * field_hat))
            
        elif ndim == 2:
            ny, nx = field.shape
            field_hat = fft2(field)
            
            if component == 0:  # d²/dx²
                kx = 2 * np.pi * np.fft.fftfreq(nx, self.params.dx)
                return np.real(ifft2(-(kx[None, :]**2) * field_hat))
            elif component == 1:  # d²/dy²
                ky = 2 * np.pi * np.fft.fftfreq(ny, self.params.dy)
                return np.real(ifft2(-(ky[:, None]**2) * field_hat))
                
        elif ndim == 3:
            nz, ny, nx = field.shape
            field_hat = fftn(field)
            
            if component == 0:  # d²/dx²
                kx = 2 * np.pi * np.fft.fftfreq(nx, self.params.dx)
                return np.real(ifftn(-(kx[None, None, :]**2) * field_hat))
            elif component == 1:  # d²/dy²
                ky = 2 * np.pi * np.fft.fftfreq(ny, self.params.dy)
                return np.real(ifftn(-(ky[None, :, None]**2) * field_hat))
            elif component == 2:  # d²/dz²
                kz = 2 * np.pi * np.fft.fftfreq(nz, self.params.dz)
                return np.real(ifftn(-(kz[:, None, None]**2) * field_hat))
        
        raise ValueError(f"Invalid component {component} for {ndim}D field")
    
    def _apply_nth_derivative(self, field: np.ndarray, n: int, component: int = 0) -> np.ndarray:
        """Apply nth derivative using spectral method."""
        ndim = len(field.shape)
        
        if ndim == 1:
            npts = field.shape[0]
            k = 2j * np.pi * np.fft.fftfreq(npts, self.params.dx)
            field_hat = fft(field)
            return np.real(ifft((k**n) * field_hat))
            
        elif ndim == 2:
            ny, nx = field.shape
            field_hat = fft2(field)
            
            if component == 0:  # dⁿ/dxⁿ
                kx = 2j * np.pi * np.fft.fftfreq(nx, self.params.dx)
                return np.real(ifft2((kx[None, :]**n) * field_hat))
            elif component == 1:  # dⁿ/dyⁿ
                ky = 2j * np.pi * np.fft.fftfreq(ny, self.params.dy)
                return np.real(ifft2((ky[:, None]**n) * field_hat))
                
        elif ndim == 3:
            nz, ny, nx = field.shape
            field_hat = fftn(field)
            
            if component == 0:  # dⁿ/dxⁿ
                kx = 2j * np.pi * np.fft.fftfreq(nx, self.params.dx)
                return np.real(ifftn((kx[None, None, :]**n) * field_hat))
            elif component == 1:  # dⁿ/dyⁿ
                ky = 2j * np.pi * np.fft.fftfreq(ny, self.params.dy)
                return np.real(ifftn((ky[None, :, None]**n) * field_hat))
            elif component == 2:  # dⁿ/dzⁿ
                kz = 2j * np.pi * np.fft.fftfreq(nz, self.params.dz)
                return np.real(ifftn((kz[:, None, None]**n) * field_hat))
        
        raise ValueError(f"Invalid component {component} for {ndim}D field")


class PDESolver(ABC):
    """Abstract base class for PDE solvers."""
    
    def __init__(self, params: ComputationalParameters):
        self.params = params
        self.statistics = {
            'total_steps': 0,
            'successful_steps': 0,
            'stability_violations': 0,
            'average_step_time': 0.0,
            'total_time': 0.0
        }
        self._step_times = []
    
    @abstractmethod
    def evolve(self, field: np.ndarray, dt: Optional[float] = None, **kwargs) -> np.ndarray:
        """Evolve field forward by one time step."""
        pass
    
    @abstractmethod
    def check_stability(self, field: np.ndarray, dt: float, **kwargs) -> bool:
        """Check CFL stability condition."""
        pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get solver statistics."""
        stats = self.statistics.copy()
        if self._step_times:
            stats['average_step_time'] = np.mean(self._step_times)
            stats['total_time'] = np.sum(self._step_times)
        return stats
    
    def reset_statistics(self):
        """Reset solver statistics."""
        self.statistics = {
            'total_steps': 0,
            'successful_steps': 0,
            'stability_violations': 0,
            'average_step_time': 0.0,
            'total_time': 0.0
        }
        self._step_times.clear()
    
    def _record_step(self, success: bool, step_time: float):
        """Record step statistics."""
        self.statistics['total_steps'] += 1
        if success:
            self.statistics['successful_steps'] += 1
        self._step_times.append(step_time)
        
        # Keep only recent step times to avoid memory growth
        if len(self._step_times) > 1000:
            self._step_times = self._step_times[-500:]


class WaveEquationSolver(PDESolver):
    """Solver for the wave equation using Verlet integration."""
    
    def __init__(self, params: ComputationalParameters, wave_speed: float = 1.0):
        super().__init__(params)
        self.wave_speed = wave_speed
        self.laplacian_op = LaplacianOperator(params)
        self.prev_field = None
        self.current_field = None
    
    def evolve(self, field: np.ndarray, dt: Optional[float] = None, **kwargs) -> np.ndarray:
        """Evolve wave equation using Verlet integration."""
        start_time = time.time()
        dt = dt or self.params.dt
        
        try:
            # Check stability
            if not self.check_stability(field, dt):
                self.statistics['stability_violations'] += 1
                if self.params.adaptive_timestep:
                    dt = self._compute_stable_timestep(field)
                    logger.warning(f"Reduced timestep to {dt} for stability")
            
            if self.prev_field is None:
                # First step: use forward Euler to bootstrap
                laplacian = self.laplacian_op.apply(field)
                self.prev_field = field.copy()
                self.current_field = field + dt * np.zeros_like(field)  # Initial velocity assumed zero
                next_field = field + dt * self.current_field + 0.5 * (dt**2) * (self.wave_speed**2) * laplacian
            else:
                # Verlet integration: u^{n+1} = 2u^n - u^{n-1} + c²dt²∇²u^n
                laplacian = self.laplacian_op.apply(self.current_field)
                next_field = (2 * self.current_field - self.prev_field + 
                             (self.wave_speed * dt)**2 * laplacian)
            
            # Update history
            self.prev_field = self.current_field.copy() if self.current_field is not None else field.copy()
            self.current_field = next_field.copy()
            
            step_time = time.time() - start_time
            self._record_step(True, step_time)
            
            return next_field
            
        except Exception as e:
            step_time = time.time() - start_time
            self._record_step(False, step_time)
            logger.error(f"Wave equation solver failed: {e}")
            raise
    
    def check_stability(self, field: np.ndarray, dt: float, **kwargs) -> bool:
        """Check CFL condition for wave equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        cfl = self.wave_speed * dt / dx_min
        return cfl <= self.params.cfl_safety_factor * self.params.max_cfl
    
    def _compute_stable_timestep(self, field: np.ndarray) -> float:
        """Compute stable timestep based on CFL condition."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        return self.params.cfl_safety_factor * self.params.max_cfl * dx_min / self.wave_speed


class DiffusionEquationSolver(PDESolver):
    """Solver for the diffusion equation."""
    
    def __init__(self, params: ComputationalParameters, diffusion_coefficient: float = 1.0):
        super().__init__(params)
        self.diffusion_coefficient = diffusion_coefficient
        self.laplacian_op = LaplacianOperator(params)
    
    def evolve(self, field: np.ndarray, dt: Optional[float] = None, **kwargs) -> np.ndarray:
        """Evolve diffusion equation."""
        start_time = time.time()
        dt = dt or self.params.dt
        
        try:
            # Check stability
            if not self.check_stability(field, dt):
                self.statistics['stability_violations'] += 1
                if self.params.adaptive_timestep:
                    dt = self._compute_stable_timestep(field)
                    logger.warning(f"Reduced timestep to {dt} for stability")
            
            if self.params.integration_scheme == IntegrationScheme.EULER:
                next_field = self._euler_step(field, dt)
            elif self.params.integration_scheme == IntegrationScheme.CRANK_NICOLSON:
                next_field = self._crank_nicolson_step(field, dt)
            elif self.params.integration_scheme == IntegrationScheme.RK4:
                next_field = self._rk4_step(field, dt)
            else:
                raise ValueError(f"Unsupported integration scheme: {self.params.integration_scheme}")
            
            step_time = time.time() - start_time
            self._record_step(True, step_time)
            
            return next_field
            
        except Exception as e:
            step_time = time.time() - start_time
            self._record_step(False, step_time)
            logger.error(f"Diffusion solver failed: {e}")
            raise
    
    def _euler_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Forward Euler step."""
        laplacian = self.laplacian_op.apply(field)
        return field + dt * self.diffusion_coefficient * laplacian
    
    def _crank_nicolson_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Crank-Nicolson implicit step."""
        laplacian_matrix = self.laplacian_op.construct_matrix(field.shape)
        
        # (I - 0.5*D*dt*L) * u^{n+1} = (I + 0.5*D*dt*L) * u^n
        I = sp.identity(field.size)
        A = I - 0.5 * self.diffusion_coefficient * dt * laplacian_matrix
        b_matrix = I + 0.5 * self.diffusion_coefficient * dt * laplacian_matrix
        
        rhs = b_matrix.dot(field.flatten())
        solution = spsolve(A, rhs)
        
        return solution.reshape(field.shape)
    
    def _rk4_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Fourth-order Runge-Kutta step."""
        def f(u):
            return self.diffusion_coefficient * self.laplacian_op.apply(u)
        
        k1 = dt * f(field)
        k2 = dt * f(field + 0.5 * k1)
        k3 = dt * f(field + 0.5 * k2)
        k4 = dt * f(field + k3)
        
        return field + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def check_stability(self, field: np.ndarray, dt: float, **kwargs) -> bool:
        """Check stability condition for diffusion equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        stability_number = self.diffusion_coefficient * dt / (dx_min**2)
        
        if self.params.integration_scheme == IntegrationScheme.EULER:
            return stability_number <= 0.5  # Forward Euler stability limit
        else:
            return stability_number <= 2.0  # More lenient for implicit methods
    
    def _compute_stable_timestep(self, field: np.ndarray) -> float:
        """Compute stable timestep for diffusion equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        if self.params.integration_scheme == IntegrationScheme.EULER:
            return 0.4 * dx_min**2 / self.diffusion_coefficient
        else:
            return 1.0 * dx_min**2 / self.diffusion_coefficient


class SchrodingerEquationSolver(PDESolver):
    """Solver for the Schrödinger equation."""
    
    def __init__(self, params: ComputationalParameters, hbar: float = 1.0, mass: float = 1.0):
        super().__init__(params)
        self.hbar = hbar
        self.mass = mass
        self.laplacian_op = LaplacianOperator(params)
    
    def evolve(self, field: np.ndarray, dt: Optional[float] = None, potential: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Evolve Schrödinger equation."""
        start_time = time.time()
        dt = dt or self.params.dt
        potential = potential if potential is not None else np.zeros_like(field.real)
        
        try:
            if not self.check_stability(field, dt, potential=potential):
                self.statistics['stability_violations'] += 1
                if self.params.adaptive_timestep:
                    dt = self._compute_stable_timestep(field, potential)
                    logger.warning(f"Reduced timestep to {dt} for stability")
            
            if self.params.integration_scheme == IntegrationScheme.SPLIT_OPERATOR:
                next_field = self._split_operator_step(field, dt, potential)
            elif self.params.integration_scheme == IntegrationScheme.CRANK_NICOLSON:
                next_field = self._crank_nicolson_step(field, dt, potential)
            else:
                next_field = self._rk4_step(field, dt, potential)
            
            step_time = time.time() - start_time
            self._record_step(True, step_time)
            
            return next_field
            
        except Exception as e:
            step_time = time.time() - start_time
            self._record_step(False, step_time)
            logger.error(f"Schrödinger solver failed: {e}")
            raise
    
    def _split_operator_step(self, field: np.ndarray, dt: float, potential: np.ndarray) -> np.ndarray:
        """Split-operator method for Schrödinger equation."""
        # First half-step with kinetic operator in Fourier space
        field_fft = fftn(field)
        
        # Construct k-space grid
        ndim = len(field.shape)
        if ndim == 1:
            n = field.shape[0]
            k = 2 * np.pi * np.fft.fftfreq(n, self.params.dx)
            k2 = k**2
        elif ndim == 2:
            ny, nx = field.shape
            kx = 2 * np.pi * np.fft.fftfreq(nx, self.params.dx)
            ky = 2 * np.pi * np.fft.fftfreq(ny, self.params.dy)
            Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
            k2 = Kx**2 + Ky**2
        elif ndim == 3:
            nz, ny, nx = field.shape
            kx = 2 * np.pi * np.fft.fftfreq(nx, self.params.dx)
            ky = 2 * np.pi * np.fft.fftfreq(ny, self.params.dy)
            kz = 2 * np.pi * np.fft.fftfreq(nz, self.params.dz)
            Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
            k2 = Kx**2 + Ky**2 + Kz**2
        
        # Apply kinetic evolution
        kinetic_phase = np.exp(-1j * self.hbar * k2 * dt / (4 * self.mass))
        field_fft *= kinetic_phase
        field_half = ifftn(field_fft)
        
        # Full step with potential operator
        potential_phase = np.exp(-1j * potential * dt / self.hbar)
        field_pot = field_half * potential_phase
        
        # Second half-step with kinetic operator
        field_fft = fftn(field_pot)
        field_fft *= kinetic_phase
        
        return ifftn(field_fft)
    
    def _crank_nicolson_step(self, field: np.ndarray, dt: float, potential: np.ndarray) -> np.ndarray:
        """Crank-Nicolson step for Schrödinger equation."""
        laplacian_matrix = self.laplacian_op.construct_matrix(field.shape)
        
        # Hamiltonian: H = -ℏ²/(2m)∇² + V
        H = (-self.hbar**2 / (2 * self.mass)) * laplacian_matrix + sp.diags(potential.flatten())
        
        I = sp.identity(field.size)
        A = I + 1j * dt * H / (2 * self.hbar)
        B = I - 1j * dt * H / (2 * self.hbar)
        
        rhs = B.dot(field.flatten())
        solution = spsolve(A, rhs)
        
        return solution.reshape(field.shape)
    
    def _rk4_step(self, field: np.ndarray, dt: float, potential: np.ndarray) -> np.ndarray:
        """RK4 step for Schrödinger equation."""
        def f(psi):
            laplacian = self.laplacian_op.apply(psi)
            return (-1j/self.hbar) * ((-self.hbar**2/(2*self.mass)) * laplacian + potential * psi)
        
        k1 = dt * f(field)
        k2 = dt * f(field + 0.5 * k1)
        k3 = dt * f(field + 0.5 * k2)
        k4 = dt * f(field + k3)
        
        return field + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def check_stability(self, field: np.ndarray, dt: float, potential: Optional[np.ndarray] = None, **kwargs) -> bool:
        """Check stability for Schrödinger equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        kinetic_dt = self.hbar * dt / (self.mass * dx_min**2)
        
        # Check kinetic stability
        kinetic_stable = kinetic_dt <= 1.0
        
        # Check potential stability if provided
        potential_stable = True
        if potential is not None:
            max_potential = np.max(np.abs(potential))
            potential_dt = max_potential * dt / self.hbar
            potential_stable = potential_dt <= np.pi
        
        return kinetic_stable and potential_stable
    
    def _compute_stable_timestep(self, field: np.ndarray, potential: np.ndarray) -> float:
        """Compute stable timestep for Schrödinger equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        
        # Kinetic constraint
        dt_kinetic = 0.5 * self.mass * dx_min**2 / self.hbar
        
        # Potential constraint
        if potential is not None and np.max(np.abs(potential)) > 0:
            dt_potential = 0.5 * self.hbar / np.max(np.abs(potential))
            return min(dt_kinetic, dt_potential)
        
        return dt_kinetic


class MaxwellEquationsSolver(PDESolver):
    """Solver for Maxwell's equations in vacuum."""
    
    def __init__(self, params: ComputationalParameters, c: float = 1.0):
        super().__init__(params)
        self.c = c  # Speed of light
        self.gradient_op = GradientOperator(params)
    
    def evolve(self, field: np.ndarray, dt: Optional[float] = None, **kwargs) -> np.ndarray:
        """Evolve Maxwell equations using FDTD."""
        start_time = time.time()
        dt = dt or self.params.dt
        
        try:
            if not self.check_stability(field, dt):
                self.statistics['stability_violations'] += 1
                if self.params.adaptive_timestep:
                    dt = self._compute_stable_timestep(field)
                    logger.warning(f"Reduced timestep to {dt} for stability")
            
            # Field format: [Ex, Ey, Ez, Bx, By, Bz]
            if len(field.shape) == 2:  # 2D case
                next_field = self._evolve_2d(field, dt)
            elif len(field.shape) == 3:  # 3D case
                next_field = self._evolve_3d(field, dt)
            else:
                raise ValueError(f"Unsupported field dimension: {len(field.shape)}")
            
            step_time = time.time() - start_time
            self._record_step(True, step_time)
            
            return next_field
            
        except Exception as e:
            step_time = time.time() - start_time
            self._record_step(False, step_time)
            logger.error(f"Maxwell solver failed: {e}")
            raise
    
    def _evolve_2d(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Evolve 2D Maxwell equations (TM mode)."""
        # Field components: [Ez, Hx, Hy]
        Ez = field[0]
        Hx = field[1]
        Hy = field[2]
        
        # ∂Ez/∂t = c²(∂Hy/∂x - ∂Hx/∂y)
        dHy_dx = self.gradient_op.apply(Hy.reshape(Hy.shape), component=0)[0]
        dHx_dy = self.gradient_op.apply(Hx.reshape(Hx.shape), component=1)[1]
        dEz_dt = self.c**2 * (dHy_dx - dHx_dy)
        
        # ∂Hx/∂t = -∂Ez/∂y
        dEz_dy = self.gradient_op.apply(Ez.reshape(Ez.shape), component=1)[1]
        dHx_dt = -dEz_dy
        
        # ∂Hy/∂t = ∂Ez/∂x
        dEz_dx = self.gradient_op.apply(Ez.reshape(Ez.shape), component=0)[0]
        dHy_dt = dEz_dx
        
        # Update using RK4
        return self._rk4_maxwell_2d(field, dt, dEz_dt, dHx_dt, dHy_dt)
    
    def _evolve_3d(self, field: np.ndarray, dt: float) -> np.ndarray:
        """Evolve 3D Maxwell equations."""
        # Field components: [Ex, Ey, Ez, Bx, By, Bz]
        E = field[:3]
        B = field[3:]
        
        # Compute curl operations
        curl_E = self._compute_curl(E)
        curl_B = self._compute_curl(B)
        
        # Maxwell equations:
        # ∂E/∂t = c²∇×B
        # ∂B/∂t = -∇×E
        dE_dt = self.c**2 * curl_B
        dB_dt = -curl_E
        
        # RK4 update
        field_dot = np.concatenate([dE_dt, dB_dt], axis=0)
        return self._rk4_step_maxwell(field, dt, field_dot)
    
    def _compute_curl(self, vector_field: np.ndarray) -> np.ndarray:
        """Compute curl of vector field."""
        if len(vector_field.shape) == 3:  # 2D field
            Fx, Fy = vector_field[0], vector_field[1]
            
            dFy_dx = self.gradient_op.apply(Fy, component=0)[0]
            dFx_dy = self.gradient_op.apply(Fx, component=1)[1]
            
            curl_z = dFy_dx - dFx_dy
            return np.array([np.zeros_like(curl_z), np.zeros_like(curl_z), curl_z])
            
        elif len(vector_field.shape) == 4:  # 3D field
            Fx, Fy, Fz = vector_field[0], vector_field[1], vector_field[2]
            
            grad_Fx = self.gradient_op.apply(Fx)
            grad_Fy = self.gradient_op.apply(Fy)
            grad_Fz = self.gradient_op.apply(Fz)
            
            curl_x = grad_Fz[1] - grad_Fy[2]  # ∂Fz/∂y - ∂Fy/∂z
            curl_y = grad_Fx[2] - grad_Fz[0]  # ∂Fx/∂z - ∂Fz/∂x
            curl_z = grad_Fy[0] - grad_Fx[1]  # ∂Fy/∂x - ∂Fx/∂y
            
            return np.array([curl_x, curl_y, curl_z])
    
    def _rk4_maxwell_2d(self, field: np.ndarray, dt: float, dEz_dt: np.ndarray, 
                       dHx_dt: np.ndarray, dHy_dt: np.ndarray) -> np.ndarray:
        """RK4 step for 2D Maxwell equations."""
        def field_derivative(f):
            Ez, Hx, Hy = f[0], f[1], f[2]
            
            dHy_dx = self.gradient_op.apply(Hy, component=0)[0]
            dHx_dy = self.gradient_op.apply(Hx, component=1)[1]
            dEz = self.c**2 * (dHy_dx - dHx_dy)
            
            dEz_dy = self.gradient_op.apply(Ez, component=1)[1]
            dHx = -dEz_dy
            
            dEz_dx = self.gradient_op.apply(Ez, component=0)[0]
            dHy = dEz_dx
            
            return np.array([dEz, dHx, dHy])
        
        k1 = dt * field_derivative(field)
        k2 = dt * field_derivative(field + 0.5 * k1)
        k3 = dt * field_derivative(field + 0.5 * k2)
        k4 = dt * field_derivative(field + k3)
        
        return field + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _rk4_step_maxwell(self, field: np.ndarray, dt: float, field_dot: np.ndarray) -> np.ndarray:
        """RK4 step for Maxwell equations."""
        def f(f_field):
            E = f_field[:3]
            B = f_field[3:]
            curl_E = self._compute_curl(E)
            curl_B = self._compute_curl(B)
            dE_dt = self.c**2 * curl_B
            dB_dt = -curl_E
            return np.concatenate([dE_dt, dB_dt], axis=0)
        
        k1 = dt * f(field)
        k2 = dt * f(field + 0.5 * k1)
        k3 = dt * f(field + 0.5 * k2)
        k4 = dt * f(field + k3)
        
        return field + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def check_stability(self, field: np.ndarray, dt: float, **kwargs) -> bool:
        """Check CFL condition for Maxwell equations."""
        if len(field.shape) == 2:  # 2D
            dx_min = min(self.params.dx, self.params.dy)
        else:  # 3D
            dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        
        cfl = self.c * dt / dx_min
        return cfl <= self.params.cfl_safety_factor * self.params.max_cfl
    
    def _compute_stable_timestep(self, field: np.ndarray) -> float:
        """Compute stable timestep for Maxwell equations."""
        if len(field.shape) == 2:  # 2D
            dx_min = min(self.params.dx, self.params.dy)
        else:  # 3D
            dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        
        return self.params.cfl_safety_factor * self.params.max_cfl * dx_min / self.c


class DiracEquationSolver(PDESolver):
    """Solver for the Dirac equation."""
    
    def __init__(self, params: ComputationalParameters, hbar: float = 1.0, c: float = 1.0, mass: float = 1.0):
        super().__init__(params)
        self.hbar = hbar
        self.c = c
        self.mass = mass
        self.gradient_op = GradientOperator(params)
        
        # Dirac gamma matrices (standard representation)
        self.gamma0 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=complex)
        self.gamma1 = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [-1, 0, 0, 0]], dtype=complex)
        self.gamma2 = np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, 1j, 0, 0], [-1j, 0, 0, 0]], dtype=complex)
        self.gamma3 = np.array([[0, 0, 1, 0], [0, 0, 0, -1], [-1, 0, 0, 0], [0, 1, 0, 0]], dtype=complex)
        
        self.gamma_matrices = [self.gamma0, self.gamma1, self.gamma2, self.gamma3]
    
    def evolve(self, field: np.ndarray, dt: Optional[float] = None, **kwargs) -> np.ndarray:
        """Evolve Dirac equation."""
        start_time = time.time()
        dt = dt or self.params.dt
        
        try:
            if not self.check_stability(field, dt):
                self.statistics['stability_violations'] += 1
                if self.params.adaptive_timestep:
                    dt = self._compute_stable_timestep(field)
                    logger.warning(f"Reduced timestep to {dt} for stability")
            
            # Dirac equation: (iγᵘ∂ᵘ - mc/ℏ)ψ = 0
            # or i∂ψ/∂t = (cγ⁰γⁱ∂ᵢ + mc²γ⁰/ℏ)ψ
            next_field = self._rk4_step(field, dt)
            
            step_time = time.time() - start_time
            self._record_step(True, step_time)
            
            return next_field
            
        except Exception as e:
            step_time = time.time() - start_time
            self._record_step(False, step_time)
            logger.error(f"Dirac solver failed: {e}")
            raise
    
    def _rk4_step(self, field: np.ndarray, dt: float) -> np.ndarray:
        """RK4 step for Dirac equation."""
        def f(psi):
            return self._dirac_operator(psi) / (1j * self.hbar)
        
        k1 = dt * f(field)
        k2 = dt * f(field + 0.5 * k1)
        k3 = dt * f(field + 0.5 * k2)
        k4 = dt * f(field + k3)
        
        return field + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _dirac_operator(self, psi: np.ndarray) -> np.ndarray:
        """Apply Dirac operator (cγ⁰γⁱ∂ᵢ + mc²γ⁰/ℏ)ψ."""
        ndim = len(psi.shape) - 1  # Last dimension is spinor components
        result = np.zeros_like(psi)
        
        # Spatial derivative terms
        for i in range(min(ndim, 3)):  # Up to 3 spatial dimensions
            grad_psi = self.gradient_op.apply(psi, component=i)
            
            # Apply γ⁰γⁱ to each spinor component
            for alpha in range(4):
                for beta in range(4):
                    gamma_product = self.gamma0[alpha, beta] * self.gamma_matrices[i+1][beta, :]
                    result[..., alpha] += self.c * np.sum(gamma_product * grad_psi, axis=-1)
        
        # Mass term: mc²γ⁰ψ/ℏ
        mass_term = (self.mass * self.c**2 / self.hbar) * np.einsum('ij,...j->...i', self.gamma0, psi)
        result += mass_term
        
        return result
    
    def check_stability(self, field: np.ndarray, dt: float, **kwargs) -> bool:
        """Check stability for Dirac equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        
        # CFL-like condition for Dirac equation
        cfl = self.c * dt / dx_min
        return cfl <= self.params.cfl_safety_factor * self.params.max_cfl
    
    def _compute_stable_timestep(self, field: np.ndarray) -> float:
        """Compute stable timestep for Dirac equation."""
        dx_min = min(self.params.dx, self.params.dy, self.params.dz)
        return self.params.cfl_safety_factor * self.params.max_cfl * dx_min / self.c


class FieldComputeEngine:
    """Central engine for field computation and PDE solving."""
    
    def __init__(self, params: Optional[ComputationalParameters] = None):
        self.params = params or ComputationalParameters()
        self._solvers = {}
        self._operators = {}
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(max_workers=self.params.num_threads)
        
        # Performance tracking
        self.statistics = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_compute_time': 0.0,
            'average_compute_time': 0.0
        }
        
        # Register default solvers
        self._register_default_solvers()
        self._register_default_operators()
    
    def _register_default_solvers(self):
        """Register default PDE solvers."""
        self._solvers = {
            'wave': lambda **kwargs: WaveEquationSolver(self.params, **kwargs),
            'diffusion': lambda **kwargs: DiffusionEquationSolver(self.params, **kwargs),
            'schrodinger': lambda **kwargs: SchrodingerEquationSolver(self.params, **kwargs),
            'maxwell': lambda **kwargs: MaxwellEquationsSolver(self.params, **kwargs),
            'dirac': lambda **kwargs: DiracEquationSolver(self.params, **kwargs)
        }
    
    def _register_default_operators(self):
        """Register default differential operators."""
        self._operators = {
            'laplacian': LaplacianOperator(self.params),
            'gradient': GradientOperator(self.params),
            'spectral': SpectralOperator(self.params)
        }
    
    def get_solver(self, field_type: str, **kwargs) -> PDESolver:
        """Get solver for field type."""
        with self._lock:
            if field_type not in self._solvers:
                raise ValueError(f"Unknown field type: {field_type}")
            
            solver = self._solvers[field_type](**kwargs)
            return solver
    
    def get_operator(self, operator_type: str) -> DifferentialOperator:
        """Get differential operator."""
        with self._lock:
            if operator_type not in self._operators:
                raise ValueError(f"Unknown operator type: {operator_type}")
            
            return self._operators[operator_type]
    
    def evolve_field(self, field: np.ndarray, field_type: str, dt: Optional[float] = None, **kwargs) -> np.ndarray:
        """Evolve field using appropriate solver."""
        start_time = time.time()
        
        try:
            solver = self.get_solver(field_type, **kwargs)
            result = solver.evolve(field, dt, **kwargs)
            
            # Update statistics
            compute_time = time.time() - start_time
            with self._lock:
                self.statistics['total_evolutions'] += 1
                self.statistics['successful_evolutions'] += 1
                self.statistics['total_compute_time'] += compute_time
                self.statistics['average_compute_time'] = (
                    self.statistics['total_compute_time'] / self.statistics['total_evolutions']
                )
            
            return result
            
        except Exception as e:
            compute_time = time.time() - start_time
            with self._lock:
                self.statistics['total_evolutions'] += 1
                self.statistics['total_compute_time'] += compute_time
            
            logger.error(f"Field evolution failed: {e}")
            raise
    
    def calculate_derivatives(self, field: np.ndarray, derivative_orders: List[int] = [1, 2]) -> Dict[str, np.ndarray]:
        """Calculate derivatives of field."""
        results = {}
        
        for order in derivative_orders:
            if order == 1:
                grad_op = self.get_operator('gradient')
                results['gradient'] = grad_op.apply(field)
            elif order == 2:
                laplacian_op = self.get_operator('laplacian')
                results['laplacian'] = laplacian_op.apply(field)
            else:
                # Use spectral operator for higher orders
                spectral_op = self.get_operator('spectral')
                results[f'derivative_order_{order}'] = spectral_op.apply(field, derivative_order=order)
        
        return results
    
    def register_solver(self, name: str, solver_factory: Callable):
        """Register custom solver."""
        with self._lock:
            self._solvers[name] = solver_factory
    
    def register_operator(self, name: str, operator: DifferentialOperator):
        """Register custom operator."""
        with self._lock:
            self._operators[name] = operator
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            stats = self.statistics.copy()
            
            # Add solver statistics
            for name in self._solvers:
                try:
                    solver = self._solvers[name]()
                    stats[f'{name}_solver'] = solver.get_statistics()
                except:
                    pass
            
            return stats
    
    def clear_cache(self):
        """Clear all operator caches."""
        with self._lock:
            for operator in self._operators.values():
                operator.clear_cache()
            
            self.statistics['cache_hits'] = 0
            self.statistics['cache_misses'] = 0
    
    def shutdown(self):
        """Shutdown thread pool."""
        self._thread_pool.shutdown(wait=True)
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass


# Global compute engine instance
_global_compute_engine = None
_global_lock = threading.Lock()


def get_compute_engine() -> FieldComputeEngine:
    """Get global compute engine instance."""
    global _global_compute_engine
    
    if _global_compute_engine is None:
        with _global_lock:
            if _global_compute_engine is None:
                _global_compute_engine = FieldComputeEngine()
    
    return _global_compute_engine


def evolve_field(field: np.ndarray, field_type: str, dt: Optional[float] = None, **kwargs) -> np.ndarray:
    """Convenience function to evolve field using global engine."""
    engine = get_compute_engine()
    return engine.evolve_field(field, field_type, dt, **kwargs)


def calculate_field_derivatives(field: np.ndarray, orders: List[int] = [1, 2]) -> Dict[str, np.ndarray]:
    """Convenience function to calculate derivatives using global engine."""
    engine = get_compute_engine()
    return engine.calculate_derivatives(field, orders)


def set_global_compute_parameters(params: ComputationalParameters):
    """Set global computation parameters."""
    global _global_compute_engine
    
    with _global_lock:
        if _global_compute_engine is not None:
            _global_compute_engine.shutdown()
        _global_compute_engine = FieldComputeEngine(params)


# GPU acceleration utilities (if available)
if CUPY_AVAILABLE:
    def to_gpu(array: np.ndarray) -> cp.ndarray:
        """Transfer array to GPU."""
        return cp.asarray(array)
    
    def from_gpu(array: cp.ndarray) -> np.ndarray:
        """Transfer array from GPU."""
        return cp.asnumpy(array)
else:
    def to_gpu(array: np.ndarray) -> np.ndarray:
        """Fallback when GPU not available."""
        return array
    
    def from_gpu(array: np.ndarray) -> np.ndarray:
        """Fallback when GPU not available."""
        return array


# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress numpy warnings for better performance
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')