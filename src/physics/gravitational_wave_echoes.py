"""
Gravitational Wave Echo Simulation Module

Implements OSH predictions for post-ringdown echoes in gravitational wave signatures
from black hole mergers. Based on the theoretical framework where echoes arise from
memory field compression feedback and information curvature resonance.

This module simulates gravitational wave signals with OSH-specific echo patterns
that emerge from the recursive memory structure of spacetime near extreme curvature.
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
import scipy.signal
import scipy.special
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime
import json

# Import existing OSH components
from .field.field_types import FieldConfiguration
from ..quantum.quantum_state import QuantumState

# Physical constants
G = 6.67430e-11  # m³/(kg⋅s²) - Gravitational constant
C = 299792458.0  # m/s - Speed of light
SOLAR_MASS = 1.98892e30  # kg - Solar mass
SCHWARZSCHILD_RADIUS_PER_SOLAR_MASS = 2.95325e3  # m - rs = 2GM/c²

logger = logging.getLogger(__name__)


class MergerType(Enum):
    """Types of compact object mergers."""
    BBH = "binary_black_hole"
    BNS = "binary_neutron_star"
    NSBH = "neutron_star_black_hole"
    PRIMORDIAL = "primordial_black_hole"


class EchoMechanism(Enum):
    """OSH-specific echo generation mechanisms."""
    MEMORY_COMPRESSION = "memory_compression"
    INFORMATION_CURVATURE = "information_curvature"
    RSP_RESONANCE = "rsp_resonance"
    COHERENCE_REFLECTION = "coherence_reflection"
    ENTROPY_BARRIER = "entropy_barrier"


@dataclass
class BinaryParameters:
    """Parameters for binary system."""
    mass1: float  # Solar masses
    mass2: float  # Solar masses
    spin1: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Dimensionless spin
    spin2: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Dimensionless spin
    eccentricity: float = 0.0
    inclination: float = 0.0  # radians
    distance: float = 100.0  # Mpc
    
    @property
    def total_mass(self) -> float:
        """Total mass in solar masses."""
        return self.mass1 + self.mass2
    
    @property
    def chirp_mass(self) -> float:
        """Chirp mass in solar masses."""
        eta = self.mass1 * self.mass2 / (self.total_mass ** 2)
        return self.total_mass * (eta ** 0.6)
    
    @property
    def final_mass(self) -> float:
        """Estimated final black hole mass using fitting formulas."""
        # Simplified fitting formula - full version would include spins
        eta = self.mass1 * self.mass2 / (self.total_mass ** 2)
        return self.total_mass * (1 - 0.057 * eta)
    
    @property
    def final_spin(self) -> float:
        """Estimated final black hole spin magnitude."""
        # Simplified estimate - full version would be more complex
        eta = self.mass1 * self.mass2 / (self.total_mass ** 2)
        return 0.69 * eta  # Typical value for non-spinning merger


@dataclass
class OSHEchoParameters:
    """Parameters controlling OSH-specific echo properties."""
    memory_strain_threshold: float = 0.85  # Threshold for memory field response
    information_curvature_coupling: float = 0.3  # Coupling to information geometry
    rsp_amplification: float = 2.5  # RSP effect on echo amplitude
    coherence_decay_time: float = 0.1  # seconds - Coherence decay timescale
    entropy_barrier_height: float = 0.95  # Entropy threshold for echo generation
    echo_delay_factor: float = 1.1  # Multiplicative factor for echo delays
    max_echo_orders: int = 5  # Maximum number of echo orders to compute
    quantum_noise_level: float = 1e-23  # Quantum noise floor


@dataclass
class GravitationalWaveSignal:
    """Container for gravitational wave strain data."""
    times: np.ndarray  # Time array in seconds
    strain_plus: np.ndarray  # Plus polarization strain
    strain_cross: np.ndarray  # Cross polarization strain
    frequency: np.ndarray  # Instantaneous frequency
    amplitude: np.ndarray  # Strain amplitude envelope
    phase: np.ndarray  # Gravitational wave phase
    
    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return self.times[-1] - self.times[0]
    
    @property
    def sampling_rate(self) -> float:
        """Sampling rate in Hz."""
        return 1.0 / (self.times[1] - self.times[0])


class GravitationalWaveEchoSimulator:
    """
    Simulates gravitational wave echoes based on OSH predictions.
    
    This simulator generates realistic gravitational wave signals including
    OSH-specific echo patterns arising from memory field compression and
    information curvature effects near black hole horizons.
    """
    
    def __init__(self, 
                 runtime: Optional[Any] = None,
                 memory_field: Optional[Any] = None):
        """
        Initialize the gravitational wave echo simulator.
        
        Args:
            runtime: Runtime with VM execution context for metrics
            memory_field: Memory field physics instance
        """
        self.runtime = runtime
        self.memory_field = memory_field
        
        # Waveform generation parameters
        self.sampling_rate = 4096.0  # Hz - Standard LIGO rate
        self.segment_duration = 32.0  # seconds - Typical analysis segment
        
        # Cache for computed waveforms
        self._waveform_cache = {}
        
        # Initialize echo analysis components
        self._initialize_echo_analyzers()
        
        logger.info("GravitationalWaveEchoSimulator initialized")
    
    def _initialize_echo_analyzers(self) -> None:
        """Initialize specialized analyzers for echo detection."""
        # Matched filter bank for echo templates
        self.echo_templates = self._generate_echo_template_bank()
        
        # Wavelet transform parameters for time-frequency analysis
        self.wavelet_scales = np.logspace(-3, 0, 64)  # Cover relevant frequencies
        
        # Information geometry analyzer
        self.info_geometry_grid_size = 128
    
    def _get_metrics_from_vm(self) -> Dict[str, float]:
        """Get current OSH metrics from VM execution context."""
        if self.runtime and hasattr(self.runtime, 'execution_context'):
            if hasattr(self.runtime.execution_context, 'current_metrics'):
                metrics = self.runtime.execution_context.current_metrics
                return {
                    'integrated_information': metrics.information_density,
                    'kolmogorov_complexity': metrics.kolmogorov_complexity,
                    'rsp': metrics.rsp,
                    'information_curvature': metrics.information_curvature,
                    'coherence': metrics.coherence,
                    'entropy': metrics.entropy
                }
        
        # Fallback values
        return {
            'integrated_information': 0.5,
            'kolmogorov_complexity': 10.0,
            'rsp': 5.0,
            'information_curvature': 0.01,
            'coherence': 0.95,
            'entropy': 0.05
        }
        
    def simulate_merger_with_echoes(self,
                                   binary_params: BinaryParameters,
                                   osh_params: Optional[OSHEchoParameters] = None,
                                   include_noise: bool = True) -> GravitationalWaveSignal:
        """
        Simulate a complete gravitational wave signal with OSH echoes.
        
        Args:
            binary_params: Binary system parameters
            osh_params: OSH-specific echo parameters
            include_noise: Whether to include detector noise
            
        Returns:
            Complete gravitational wave signal with echoes
        """
        if osh_params is None:
            osh_params = OSHEchoParameters()
        
        # Generate base waveform (inspiral-merger-ringdown)
        base_signal = self._generate_base_waveform(binary_params)
        
        # Calculate OSH metrics at merger
        osh_metrics = self._calculate_merger_osh_metrics(binary_params, base_signal)
        
        # Generate echo train based on OSH physics
        echo_signal = self._generate_osh_echoes(
            base_signal, binary_params, osh_params, osh_metrics
        )
        
        # Combine base signal with echoes
        combined_signal = self._combine_signals(base_signal, echo_signal)
        
        # Add detector noise if requested
        if include_noise:
            combined_signal = self._add_detector_noise(combined_signal)
        
        # Calculate additional diagnostic quantities
        self._compute_echo_diagnostics(combined_signal, osh_metrics)
        
        return combined_signal
    
    def _generate_base_waveform(self, params: BinaryParameters) -> GravitationalWaveSignal:
        """Generate the base inspiral-merger-ringdown waveform."""
        # Time array
        dt = 1.0 / self.sampling_rate
        merger_time = self.segment_duration * 0.75  # Merger at 3/4 point
        times = np.arange(0, self.segment_duration, dt)
        
        # Frequency evolution (simplified post-Newtonian)
        t_merge = times[np.argmin(np.abs(times - merger_time))]
        tau = t_merge - times  # Time to merger
        
        # Inspiral phase
        f_isco = C**3 / (6**(3/2) * np.pi * G * params.total_mass * SOLAR_MASS)
        inspiral_mask = tau > 0
        
        # Frequency evolution during inspiral
        frequency = np.zeros_like(times)
        frequency[inspiral_mask] = (
            (256 / (5 * np.pi)) * 
            (G * params.chirp_mass * SOLAR_MASS / C**3) ** (5/3) * 
            tau[inspiral_mask] ** (-3/8)
        ) ** (-1)
        
        # Amplitude evolution
        amplitude = np.zeros_like(times)
        strain_const = (4 / params.distance) * (
            G * params.chirp_mass * SOLAR_MASS / C**2
        ) ** (5/6) * (np.pi / C) ** (2/3)
        
        amplitude[inspiral_mask] = strain_const * frequency[inspiral_mask] ** (2/3)
        
        # Phase evolution
        phase = np.zeros_like(times)
        phase[inspiral_mask] = -2 * np.pi * np.cumsum(frequency[inspiral_mask]) * dt
        
        # Merger and ringdown
        merger_mask = (tau <= 0) & (tau > -0.01)  # 10ms merger
        ringdown_mask = tau <= -0.01
        
        # Ringdown parameters
        f_qnm = self._calculate_qnm_frequency(params)
        tau_qnm = self._calculate_qnm_damping_time(params)
        
        # Smooth transition through merger
        if np.any(merger_mask):
            frequency[merger_mask] = np.linspace(
                frequency[inspiral_mask][-1] if np.any(inspiral_mask) else f_isco,
                f_qnm,
                np.sum(merger_mask)
            )
            # Peak amplitude at merger
            amplitude[merger_mask] = strain_const * 1.5
        
        # Exponential ringdown
        if np.any(ringdown_mask):
            ringdown_times = times[ringdown_mask] - times[merger_mask][-1]
            frequency[ringdown_mask] = f_qnm
            amplitude[ringdown_mask] = (
                amplitude[merger_mask][-1] * 
                np.exp(-ringdown_times / tau_qnm)
            )
            phase[ringdown_mask] = (
                phase[merger_mask][-1] + 
                2 * np.pi * f_qnm * ringdown_times
            )
        
        # Generate strain polarizations
        strain_plus = amplitude * np.cos(phase)
        strain_cross = amplitude * np.sin(phase)
        
        return GravitationalWaveSignal(
            times=times,
            strain_plus=strain_plus,
            strain_cross=strain_cross,
            frequency=frequency,
            amplitude=amplitude,
            phase=phase
        )
    
    def _calculate_merger_osh_metrics(self,
                                     params: BinaryParameters,
                                     signal: GravitationalWaveSignal) -> Dict[str, Any]:
        """Calculate OSH metrics at merger time."""
        # Find merger time (peak amplitude)
        merger_idx = np.argmax(signal.amplitude)
        merger_time = signal.times[merger_idx]
        
        # Create memory region for merger
        merger_region = self.memory_field.create_region(
            name=f"merger_{datetime.now().timestamp()}",
            capacity_qubits=int(params.total_mass * 10),  # Scale with mass
            temperature=self._hawking_temperature(params.final_mass),
            volume=self._horizon_volume(params.final_mass)
        )
        
        # Calculate information density field around horizon
        info_field = self._calculate_horizon_information_field(params)
        
        # Get OSH metrics from VM
        vm_metrics = self._get_metrics_from_vm()
        
        # Use VM metrics with merger-specific adjustments
        integrated_info = vm_metrics['integrated_information'] * params.total_mass / 10.0
        complexity = vm_metrics['kolmogorov_complexity'] * (1 + np.log10(params.total_mass))
        
        # Near-horizon entropy flux approaches zero
        entropy_flux = 0.1 / params.final_mass  # Inverse mass scaling
        
        # Calculate RSP using VM values
        rsp = integrated_info * complexity / max(entropy_flux, 1e-10)
        
        # Information curvature near horizon
        info_curvature = vm_metrics['information_curvature'] * params.final_spin
        
        return {
            "merger_time": merger_time,
            "merger_idx": merger_idx,
            "integrated_info": integrated_info,
            "complexity": complexity,
            "entropy_flux": entropy_flux,
            "rsp": rsp,
            "info_curvature": info_curvature,
            "info_field": info_field,
            "memory_region": merger_region,
            "horizon_radius": self._schwarzschild_radius(params.final_mass),
            "hawking_temperature": self._hawking_temperature(params.final_mass)
        }
    
    def _generate_osh_echoes(self,
                            base_signal: GravitationalWaveSignal,
                            params: BinaryParameters,
                            osh_params: OSHEchoParameters,
                            osh_metrics: Dict[str, Any]) -> GravitationalWaveSignal:
        """Generate echo train based on OSH physics."""
        times = base_signal.times
        merger_idx = osh_metrics["merger_idx"]
        
        # Initialize echo signal arrays
        echo_plus = np.zeros_like(base_signal.strain_plus)
        echo_cross = np.zeros_like(base_signal.strain_cross)
        
        # Only generate echoes after merger
        post_merger_mask = times > times[merger_idx]
        if not np.any(post_merger_mask):
            return GravitationalWaveSignal(
                times=times,
                strain_plus=echo_plus,
                strain_cross=echo_cross,
                frequency=np.zeros_like(times),
                amplitude=np.zeros_like(times),
                phase=np.zeros_like(times)
            )
        
        # Extract ringdown portion for echo template
        ringdown_plus = base_signal.strain_plus[merger_idx:]
        ringdown_cross = base_signal.strain_cross[merger_idx:]
        ringdown_times = times[merger_idx:] - times[merger_idx]
        
        # Generate echo train
        for echo_order in range(1, osh_params.max_echo_orders + 1):
            # Calculate echo delay based on OSH physics
            echo_delay = self._calculate_echo_delay(
                params, osh_params, osh_metrics, echo_order
            )
            
            # Calculate echo amplitude modification
            echo_amplitude_factor = self._calculate_echo_amplitude(
                params, osh_params, osh_metrics, echo_order
            )
            
            # Calculate phase shift from information geometry
            echo_phase_shift = self._calculate_echo_phase_shift(
                params, osh_params, osh_metrics, echo_order
            )
            
            # Apply time delay and amplitude/phase modifications
            echo_start_idx = merger_idx + int(echo_delay * self.sampling_rate)
            
            if echo_start_idx < len(times):
                # Calculate how much of the echo fits in the time window
                echo_length = min(
                    len(ringdown_plus),
                    len(times) - echo_start_idx
                )
                
                if echo_length > 0:
                    # Apply echo with modifications
                    echo_slice = slice(echo_start_idx, echo_start_idx + echo_length)
                    original_slice = slice(0, echo_length)
                    
                    # Add echo with amplitude factor and phase shift
                    echo_plus[echo_slice] += (
                        echo_amplitude_factor * 
                        ringdown_plus[original_slice] * 
                        np.cos(echo_phase_shift)
                    )
                    echo_cross[echo_slice] += (
                        echo_amplitude_factor * 
                        ringdown_cross[original_slice] * 
                        np.cos(echo_phase_shift)
                    )
                    
                    # Add information curvature modulation
                    if osh_metrics["info_curvature"] > 0:
                        modulation = self._calculate_curvature_modulation(
                            times[echo_slice] - times[echo_start_idx],
                            osh_metrics["info_curvature"],
                            echo_order
                        )
                        echo_plus[echo_slice] *= modulation
                        echo_cross[echo_slice] *= modulation
        
        # Calculate echo signal properties
        echo_amplitude = np.sqrt(echo_plus**2 + echo_cross**2)
        echo_phase = np.angle(echo_plus + 1j * echo_cross)
        
        # Estimate instantaneous frequency from phase
        echo_frequency = np.zeros_like(echo_phase)
        echo_frequency[1:] = np.diff(echo_phase) * self.sampling_rate / (2 * np.pi)
        echo_frequency[0] = echo_frequency[1]  # Extrapolate first point
        
        return GravitationalWaveSignal(
            times=times,
            strain_plus=echo_plus,
            strain_cross=echo_cross,
            frequency=echo_frequency,
            amplitude=echo_amplitude,
            phase=echo_phase
        )
    
    def _calculate_echo_delay(self,
                             params: BinaryParameters,
                             osh_params: OSHEchoParameters,
                             osh_metrics: Dict[str, Any],
                             echo_order: int) -> float:
        """Calculate echo time delay based on OSH physics."""
        # Base delay from light crossing time near horizon
        r_horizon = osh_metrics["horizon_radius"]
        base_delay = 2 * r_horizon / C  # Round trip time
        
        # Modify based on RSP value (high RSP = longer delay)
        rsp_factor = 1.0
        if osh_metrics["rsp"] != float('inf'):
            rsp_factor = 1.0 + np.log10(1 + osh_metrics["rsp"] / 1000)
        else:
            # For infinite RSP (black hole), use mass-based scaling
            rsp_factor = 2.0 * np.log10(params.final_mass)
        
        # Information curvature contribution
        curvature_factor = 1.0 + osh_params.information_curvature_coupling * osh_metrics["info_curvature"]
        
        # Echo order scaling (later echoes have longer delays)
        order_factor = osh_params.echo_delay_factor ** (echo_order - 1)
        
        # Total delay
        total_delay = base_delay * rsp_factor * curvature_factor * order_factor
        
        # Add stochastic component from quantum uncertainty
        quantum_jitter = np.random.normal(0, base_delay * 0.01)
        
        return max(total_delay + quantum_jitter, base_delay)
    
    def _calculate_echo_amplitude(self,
                                 params: BinaryParameters,
                                 osh_params: OSHEchoParameters,
                                 osh_metrics: Dict[str, Any],
                                 echo_order: int) -> float:
        """Calculate echo amplitude modification based on OSH physics."""
        # Base amplitude reduction per echo
        base_reduction = 0.3  # 30% of previous echo
        
        # Memory strain effect
        memory_factor = 1.0
        if hasattr(osh_metrics["memory_region"], "entropy"):
            # Higher entropy = more amplitude reduction
            memory_entropy = osh_metrics["memory_region"].entropy
            memory_factor = np.exp(-memory_entropy * osh_params.memory_strain_threshold)
        
        # RSP amplification (high RSP = stronger echoes)
        rsp_factor = 1.0
        if osh_metrics["rsp"] != float('inf'):
            rsp_factor = 1.0 + osh_params.rsp_amplification * np.tanh(osh_metrics["rsp"] / 1000)
        else:
            # Maximum amplification for infinite RSP
            rsp_factor = 1.0 + osh_params.rsp_amplification
        
        # Coherence decay
        coherence_factor = np.exp(-echo_order / osh_params.coherence_decay_time)
        
        # Information curvature enhancement near horizon
        curvature_enhancement = 1.0 + osh_metrics["info_curvature"] * 0.5
        
        # Total amplitude factor
        amplitude_factor = (
            base_reduction ** echo_order *
            memory_factor *
            rsp_factor *
            coherence_factor *
            curvature_enhancement
        )
        
        # Ensure amplitude doesn't exceed original signal
        return min(amplitude_factor, 1.0)
    
    def _calculate_echo_phase_shift(self,
                                   params: BinaryParameters,
                                   osh_params: OSHEchoParameters,
                                   osh_metrics: Dict[str, Any],
                                   echo_order: int) -> float:
        """Calculate echo phase shift from information geometry."""
        # Base phase shift from information curvature
        base_phase = osh_metrics["info_curvature"] * np.pi
        
        # Memory field contribution
        memory_phase = 0.0
        if "memory_region" in osh_metrics:
            # Phase shift from memory coherence
            memory_phase = np.pi * (1 - osh_metrics["memory_region"].fidelity)
        
        # Echo order dependence
        order_phase = echo_order * np.pi / 4
        
        # Total phase shift modulo 2π
        total_phase = (base_phase + memory_phase + order_phase) % (2 * np.pi)
        
        return total_phase
    
    def _calculate_curvature_modulation(self,
                                       time_array: np.ndarray,
                                       info_curvature: float,
                                       echo_order: int) -> np.ndarray:
        """Calculate time-dependent modulation from information curvature."""
        # Oscillation frequency proportional to curvature
        mod_frequency = info_curvature * 100  # Hz
        
        # Damped oscillation
        damping_time = 0.01 * echo_order  # Faster damping for later echoes
        envelope = np.exp(-time_array / damping_time)
        
        # Modulation pattern
        modulation = 1.0 + 0.2 * envelope * np.sin(2 * np.pi * mod_frequency * time_array)
        
        return modulation
    
    def _combine_signals(self,
                        base_signal: GravitationalWaveSignal,
                        echo_signal: GravitationalWaveSignal) -> GravitationalWaveSignal:
        """Combine base waveform with echo train."""
        # Simple addition for strain
        combined_plus = base_signal.strain_plus + echo_signal.strain_plus
        combined_cross = base_signal.strain_cross + echo_signal.strain_cross
        
        # Recalculate amplitude and phase
        combined_amplitude = np.sqrt(combined_plus**2 + combined_cross**2)
        combined_phase = np.angle(combined_plus + 1j * combined_cross)
        
        # Frequency is more complex - use instantaneous frequency
        combined_frequency = np.zeros_like(combined_phase)
        combined_frequency[1:] = np.diff(combined_phase) * self.sampling_rate / (2 * np.pi)
        combined_frequency[0] = base_signal.frequency[0]
        
        # Smooth frequency to remove artifacts
        from scipy.ndimage import gaussian_filter1d
        combined_frequency = gaussian_filter1d(combined_frequency, sigma=5)
        
        return GravitationalWaveSignal(
            times=base_signal.times,
            strain_plus=combined_plus,
            strain_cross=combined_cross,
            frequency=combined_frequency,
            amplitude=combined_amplitude,
            phase=combined_phase
        )
    
    def _add_detector_noise(self, signal: GravitationalWaveSignal) -> GravitationalWaveSignal:
        """Add realistic detector noise to the signal."""
        # Advanced LIGO noise curve (simplified)
        frequencies = np.fft.fftfreq(len(signal.times), 1/self.sampling_rate)
        frequencies = frequencies[:len(frequencies)//2]  # Positive frequencies
        
        # Noise PSD model (simplified Advanced LIGO)
        f_min = 10.0  # Hz
        noise_psd = np.zeros_like(frequencies)
        mask = frequencies > f_min
        
        # Seismic wall
        noise_psd[mask] = 1e-22 * (f_min / frequencies[mask]) ** 4
        
        # Thermal noise
        f_thermal = 50.0  # Hz
        thermal_mask = frequencies > f_thermal
        noise_psd[thermal_mask] += 3e-24
        
        # Shot noise
        f_shot = 200.0  # Hz
        shot_mask = frequencies > f_shot
        noise_psd[shot_mask] += 2e-23 * (frequencies[shot_mask] / f_shot) ** 2
        
        # Generate colored noise
        noise_amplitude = np.sqrt(noise_psd * self.sampling_rate / 2)
        noise_phase = np.random.uniform(0, 2*np.pi, len(frequencies))
        
        # Construct noise in frequency domain
        noise_fft = np.zeros(len(signal.times), dtype=complex)
        noise_fft[:len(frequencies)] = noise_amplitude * np.exp(1j * noise_phase)
        noise_fft[-len(frequencies)+1:] = np.conj(noise_fft[1:len(frequencies)])[::-1]
        
        # Transform to time domain
        noise_time = np.real(np.fft.ifft(noise_fft))
        
        # Add noise to both polarizations
        noisy_plus = signal.strain_plus + noise_time
        noisy_cross = signal.strain_cross + noise_time * np.random.normal(1.0, 0.1)  # Slightly different noise
        
        return GravitationalWaveSignal(
            times=signal.times,
            strain_plus=noisy_plus,
            strain_cross=noisy_cross,
            frequency=signal.frequency,
            amplitude=signal.amplitude,
            phase=signal.phase
        )
    
    def _compute_echo_diagnostics(self,
                                 signal: GravitationalWaveSignal,
                                 osh_metrics: Dict[str, Any]) -> None:
        """Compute diagnostic quantities for echo analysis."""
        # Store diagnostics in signal object
        signal.diagnostics = {
            "osh_metrics": osh_metrics,
            "echo_snr": self._calculate_echo_snr(signal, osh_metrics["merger_idx"]),
            "echo_times": self._detect_echo_times(signal, osh_metrics["merger_idx"]),
            "spectral_features": self._analyze_echo_spectrum(signal),
            "information_content": self._calculate_information_content(signal)
        }
    
    def _calculate_echo_snr(self, signal: GravitationalWaveSignal, merger_idx: int) -> float:
        """Calculate signal-to-noise ratio of echo components."""
        # Estimate noise level from pre-merger data
        pre_merger = signal.strain_plus[:merger_idx//2]
        noise_std = np.std(pre_merger)
        
        # Echo region is post-merger
        post_merger = signal.strain_plus[merger_idx:]
        
        # Simple SNR estimate
        if noise_std > 0:
            echo_snr = np.max(np.abs(post_merger)) / noise_std
        else:
            echo_snr = 0.0
        
        return echo_snr
    
    def _detect_echo_times(self, 
                          signal: GravitationalWaveSignal, 
                          merger_idx: int) -> List[float]:
        """Detect echo arrival times using matched filtering."""
        # Use ringdown as template
        ringdown_length = min(int(0.1 * self.sampling_rate), len(signal.times) - merger_idx)
        template = signal.strain_plus[merger_idx:merger_idx + ringdown_length]
        
        # Matched filter with post-merger signal
        post_merger_signal = signal.strain_plus[merger_idx:]
        
        if len(post_merger_signal) > len(template):
            correlation = scipy.signal.correlate(post_merger_signal, template, mode='valid')
            
            # Find peaks in correlation
            peaks, properties = scipy.signal.find_peaks(
                np.abs(correlation),
                height=0.1 * np.max(np.abs(correlation)),
                distance=int(0.01 * self.sampling_rate)  # Minimum 10ms between echoes
            )
            
            # Convert to times
            echo_times = signal.times[merger_idx + peaks]
            return echo_times.tolist()
        
        return []
    
    def _analyze_echo_spectrum(self, signal: GravitationalWaveSignal) -> Dict[str, Any]:
        """Analyze spectral features of echoes."""
        # Compute spectrogram
        f, t, Sxx = scipy.signal.spectrogram(
            signal.strain_plus,
            fs=self.sampling_rate,
            nperseg=256,
            noverlap=240
        )
        
        # Find dominant frequencies
        dominant_freq_idx = np.argmax(Sxx, axis=0)
        dominant_frequencies = f[dominant_freq_idx]
        
        return {
            "frequencies": f.tolist(),
            "times": t.tolist(),
            "power_spectrum": Sxx.tolist(),
            "dominant_frequencies": dominant_frequencies.tolist()
        }
    
    def _calculate_information_content(self, signal: GravitationalWaveSignal) -> float:
        """Calculate information content of the signal using compression."""
        # Convert to bytes for compression estimate
        signal_bytes = signal.strain_plus.tobytes()
        
        # Simple entropy estimate
        import zlib
        compressed_size = len(zlib.compress(signal_bytes))
        original_size = len(signal_bytes)
        
        # Information content estimate
        compression_ratio = compressed_size / original_size
        information_content = -np.log2(compression_ratio) * original_size / 1024  # In kilobits
        
        return information_content
    
    # Helper methods for physical calculations
    
    def _schwarzschild_radius(self, mass_solar: float) -> float:
        """Calculate Schwarzschild radius in meters."""
        return 2 * G * mass_solar * SOLAR_MASS / C**2
    
    def _hawking_temperature(self, mass_solar: float) -> float:
        """Calculate Hawking temperature in Kelvin."""
        from scipy.constants import hbar, k as kb
        return hbar * C**3 / (8 * np.pi * G * mass_solar * SOLAR_MASS * kb)
    
    def _horizon_volume(self, mass_solar: float) -> float:
        """Calculate horizon volume in m³."""
        r_s = self._schwarzschild_radius(mass_solar)
        return (4/3) * np.pi * r_s**3
    
    def _calculate_qnm_frequency(self, params: BinaryParameters) -> float:
        """Calculate quasi-normal mode frequency."""
        # Simplified formula for (l,m,n) = (2,2,0) mode
        M_final = params.final_mass * SOLAR_MASS
        a_final = params.final_spin
        
        # Dimensionless frequency
        omega_220 = 1.5251 - 1.1568 * (1 - a_final)**0.1292
        
        # Convert to Hz
        f_qnm = omega_220 * C**3 / (2 * np.pi * G * M_final)
        
        return f_qnm
    
    def _calculate_qnm_damping_time(self, params: BinaryParameters) -> float:
        """Calculate quasi-normal mode damping time."""
        M_final = params.final_mass * SOLAR_MASS
        a_final = params.final_spin
        
        # Quality factor
        Q = 0.7000 + 1.4187 * (1 - a_final)**(-0.4990)
        
        # Damping time
        tau = Q / (np.pi * self._calculate_qnm_frequency(params))
        
        return tau
    
    def _generate_echo_template_bank(self) -> List[np.ndarray]:
        """Generate bank of echo templates for matched filtering."""
        templates = []
        
        # Generate templates with different damping times and frequencies
        template_length = int(0.1 * self.sampling_rate)  # 100ms templates
        times = np.arange(template_length) / self.sampling_rate
        
        for damping_factor in [0.5, 1.0, 2.0]:
            for freq_factor in [0.8, 1.0, 1.2]:
                # Base frequency around 250 Hz (typical for stellar mass BH)
                frequency = 250.0 * freq_factor
                damping_time = 0.01 * damping_factor
                
                # Damped sinusoid
                template = np.exp(-times / damping_time) * np.sin(2 * np.pi * frequency * times)
                templates.append(template)
        
        return templates
    
    def _calculate_horizon_information_field(self, params: BinaryParameters) -> np.ndarray:
        """Calculate information density field near horizon."""
        # Create radial grid around horizon
        r_horizon = self._schwarzschild_radius(params.final_mass)
        grid_size = self.info_geometry_grid_size
        
        # Radial coordinates from 0.9 to 2.0 horizon radii
        r = np.linspace(0.9 * r_horizon, 2.0 * r_horizon, grid_size)
        theta = np.linspace(0, np.pi, grid_size)
        
        # 2D information field (axisymmetric)
        R, Theta = np.meshgrid(r, theta)
        
        # Information density increases near horizon
        # Using OSH model where information ~ 1/(r - r_horizon)
        epsilon = 0.01 * r_horizon  # Regularization
        info_field = 1.0 / (R - r_horizon + epsilon)
        
        # Normalize
        info_field = info_field / np.max(info_field)
        
        # Add angular dependence from spin
        if params.final_spin > 0:
            # Kerr metric modification
            spin_modulation = 1.0 + 0.5 * params.final_spin * np.cos(Theta)
            info_field *= spin_modulation
        
        return info_field
    
    def _generate_horizon_connectivity(self, params: BinaryParameters) -> Dict[str, List[str]]:
        """Generate connectivity graph for horizon quantum states."""
        # Simplified model with radial shells
        n_shells = min(10, int(params.final_mass))
        connectivity = {}
        
        for i in range(n_shells):
            shell_name = f"shell_{i}"
            # Each shell connects to adjacent shells
            connections = []
            if i > 0:
                connections.append(f"shell_{i-1}")
            if i < n_shells - 1:
                connections.append(f"shell_{i+1}")
            
            connectivity[shell_name] = connections
        
        return connectivity
    
    def _generate_horizon_state_values(self,
                                      params: BinaryParameters,
                                      signal: GravitationalWaveSignal,
                                      merger_idx: int) -> Dict[str, complex]:
        """Generate quantum state values near horizon at merger."""
        n_shells = min(10, int(params.final_mass))
        state_values = {}
        
        # Merger amplitude sets overall scale
        merger_amplitude = signal.amplitude[merger_idx]
        
        for i in range(n_shells):
            # Amplitude decreases with distance from horizon
            shell_amplitude = merger_amplitude * np.exp(-i / 2.0)
            
            # Phase accumulates with shell number
            shell_phase = i * np.pi / 4
            
            # Complex amplitude
            state_values[f"shell_{i}"] = shell_amplitude * np.exp(1j * shell_phase)
        
        return state_values
    
    def analyze_echo_evidence(self, signal: GravitationalWaveSignal) -> Dict[str, Any]:
        """
        Analyze gravitational wave signal for evidence of OSH echoes.
        
        Args:
            signal: Gravitational wave signal to analyze
            
        Returns:
            Dictionary containing echo evidence metrics
        """
        # Find merger time
        merger_idx = np.argmax(signal.amplitude)
        merger_time = signal.times[merger_idx]
        
        # Analyze post-merger signal
        post_merger_signal = signal.strain_plus[merger_idx:]
        post_merger_times = signal.times[merger_idx:] - merger_time
        
        # Matched filter analysis with echo templates
        echo_correlations = []
        for template in self.echo_templates:
            if len(post_merger_signal) > len(template):
                correlation = scipy.signal.correlate(
                    post_merger_signal[:len(template) * 5],  # Look in 5x template length
                    template,
                    mode='valid'
                )
                echo_correlations.append(np.max(np.abs(correlation)))
        
        # Wavelet analysis for time-frequency features
        if len(post_merger_signal) > 256:
            cwt_matrix = scipy.signal.cwt(
                post_merger_signal[:1024] if len(post_merger_signal) > 1024 else post_merger_signal,
                scipy.signal.ricker,
                self.wavelet_scales
            )
            
            # Look for repeating patterns
            pattern_score = self._calculate_pattern_score(cwt_matrix)
        else:
            pattern_score = 0.0
        
        # Statistical tests
        echo_times = self._detect_echo_times(signal, merger_idx)
        
        # Echo spacing analysis
        if len(echo_times) > 1:
            echo_spacings = np.diff(echo_times)
            spacing_regularity = 1.0 - np.std(echo_spacings) / (np.mean(echo_spacings) + 1e-10)
        else:
            spacing_regularity = 0.0
        
        # Calculate overall echo evidence score
        evidence_score = self._calculate_evidence_score(
            echo_correlations, pattern_score, spacing_regularity, len(echo_times)
        )
        
        return {
            "evidence_score": evidence_score,
            "echo_times": echo_times,
            "n_echoes_detected": len(echo_times),
            "max_correlation": max(echo_correlations) if echo_correlations else 0.0,
            "pattern_score": pattern_score,
            "spacing_regularity": spacing_regularity,
            "statistical_significance": self._calculate_statistical_significance(evidence_score)
        }
    
    def _calculate_pattern_score(self, cwt_matrix: np.ndarray) -> float:
        """Calculate score for repeating patterns in wavelet transform."""
        # Look for self-similarity in the wavelet transform
        n_scales, n_times = cwt_matrix.shape
        
        if n_times < 100:
            return 0.0
        
        # Compute autocorrelation along time axis for each scale
        pattern_scores = []
        for scale_idx in range(n_scales):
            scale_data = cwt_matrix[scale_idx, :]
            autocorr = np.correlate(scale_data, scale_data, mode='same')
            
            # Look for peaks in autocorrelation (excluding zero lag)
            center = len(autocorr) // 2
            side_autocorr = np.abs(autocorr[:center-10])  # Exclude near-zero lags
            
            if len(side_autocorr) > 0 and np.max(np.abs(scale_data)) > 0:
                pattern_scores.append(np.max(side_autocorr) / np.max(np.abs(autocorr)))
        
        return np.mean(pattern_scores) if pattern_scores else 0.0
    
    def _calculate_evidence_score(self,
                                 correlations: List[float],
                                 pattern_score: float,
                                 spacing_regularity: float,
                                 n_echoes: int) -> float:
        """Calculate overall echo evidence score."""
        # Weighted combination of evidence
        correlation_score = np.mean(correlations) if correlations else 0.0
        
        # Normalize components to [0, 1]
        correlation_weight = 0.3
        pattern_weight = 0.3
        spacing_weight = 0.2
        count_weight = 0.2
        
        # More echoes = stronger evidence (sigmoid scaling)
        count_score = np.tanh(n_echoes / 3.0)
        
        evidence_score = (
            correlation_weight * correlation_score +
            pattern_weight * pattern_score +
            spacing_weight * spacing_regularity +
            count_weight * count_score
        )
        
        return float(np.clip(evidence_score, 0, 1))
    
    def _calculate_statistical_significance(self, evidence_score: float) -> float:
        """Calculate statistical significance of echo detection."""
        # Map evidence score to p-value estimate
        # This is simplified - real implementation would use proper statistics
        if evidence_score < 0.1:
            return 1.0  # No significance
        elif evidence_score < 0.3:
            return 0.1
        elif evidence_score < 0.5:
            return 0.05
        elif evidence_score < 0.7:
            return 0.01
        else:
            return 0.001
    
    def save_waveform(self, signal: GravitationalWaveSignal, filename: str) -> None:
        """Save gravitational wave signal to file."""
        data = {
            "times": signal.times.tolist(),
            "strain_plus": signal.strain_plus.tolist(),
            "strain_cross": signal.strain_cross.tolist(),
            "frequency": signal.frequency.tolist(),
            "amplitude": signal.amplitude.tolist(),
            "phase": signal.phase.tolist()
        }
        
        if hasattr(signal, 'diagnostics'):
            # Convert numpy arrays in diagnostics to lists
            data["diagnostics"] = self._serialize_diagnostics(signal.diagnostics)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved waveform to {filename}")
    
    def _serialize_diagnostics(self, diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """Convert diagnostic data to JSON-serializable format."""
        serialized = {}
        
        for key, value in diagnostics.items():
            if isinstance(value, np.ndarray):
                serialized[key] = value.tolist()
            elif isinstance(value, dict):
                serialized[key] = self._serialize_diagnostics(value)
            elif isinstance(value, (list, tuple)):
                serialized[key] = list(value)
            elif isinstance(value, (int, float, str, bool, type(None))):
                serialized[key] = value
            else:
                # Convert to string for complex objects
                serialized[key] = str(value)
        
        return serialized