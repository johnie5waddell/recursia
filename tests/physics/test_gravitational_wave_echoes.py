"""
Test suite for gravitational wave echo simulation.

Tests the OSH-predicted echo patterns in gravitational waves from black hole mergers.
Validates physics calculations, echo generation, and analysis algorithms.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import scipy.signal

from src.physics.gravitational_wave_echoes import (
    GravitationalWaveEchoSimulator,
    BinaryParameters,
    OSHEchoParameters,
    GravitationalWaveSignal,
    MergerType,
    EchoMechanism
)
from src.physics.osh_metrics_calculator import OSHMetricsCalculator
from src.physics.memory_field_proper import MemoryFieldProper


class TestBinaryParameters:
    """Test binary system parameter calculations."""
    
    def test_total_mass_calculation(self):
        """Test total mass is correctly calculated."""
        params = BinaryParameters(mass1=30, mass2=25)
        assert params.total_mass == 55
    
    def test_chirp_mass_calculation(self):
        """Test chirp mass calculation."""
        params = BinaryParameters(mass1=30, mass2=30)
        # For equal masses, chirp mass = total_mass * 0.5^0.6 ≈ 0.659 * total_mass
        expected = 60 * 0.25**0.6
        assert abs(params.chirp_mass - expected) < 0.1
    
    def test_final_mass_estimate(self):
        """Test final black hole mass estimate."""
        params = BinaryParameters(mass1=30, mass2=30)
        # For equal masses, η = 0.25, so final mass ≈ 0.9857 * total_mass
        expected = 60 * (1 - 0.057 * 0.25)
        assert abs(params.final_mass - expected) < 0.01
    
    def test_final_spin_estimate(self):
        """Test final black hole spin estimate."""
        params = BinaryParameters(mass1=30, mass2=30)
        # For equal masses, η = 0.25
        expected = 0.69 * 0.25
        assert abs(params.final_spin - expected) < 0.01


class TestGravitationalWaveSignal:
    """Test gravitational wave signal container."""
    
    def test_signal_duration(self):
        """Test duration calculation."""
        times = np.linspace(0, 2, 1000)
        signal = GravitationalWaveSignal(
            times=times,
            strain_plus=np.zeros_like(times),
            strain_cross=np.zeros_like(times),
            frequency=np.zeros_like(times),
            amplitude=np.zeros_like(times),
            phase=np.zeros_like(times)
        )
        assert signal.duration == 2.0
    
    def test_sampling_rate(self):
        """Test sampling rate calculation."""
        times = np.linspace(0, 1, 4096)
        signal = GravitationalWaveSignal(
            times=times,
            strain_plus=np.zeros_like(times),
            strain_cross=np.zeros_like(times),
            frequency=np.zeros_like(times),
            amplitude=np.zeros_like(times),
            phase=np.zeros_like(times)
        )
        assert abs(signal.sampling_rate - 4095) < 1  # Close to 4096 Hz


class TestGravitationalWaveEchoSimulator:
    """Test the main echo simulator."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return GravitationalWaveEchoSimulator()
    
    @pytest.fixture
    def binary_params(self):
        """Standard binary parameters."""
        return BinaryParameters(
            mass1=30,
            mass2=30,
            distance=100,
            spin1=np.array([0, 0, 0.3]),
            spin2=np.array([0, 0, -0.1])
        )
    
    @pytest.fixture
    def osh_params(self):
        """Standard OSH parameters."""
        return OSHEchoParameters(
            memory_strain_threshold=0.85,
            information_curvature_coupling=0.3,
            rsp_amplification=2.5,
            coherence_decay_time=0.1,
            max_echo_orders=3
        )
    
    def test_simulator_initialization(self, simulator):
        """Test simulator initializes correctly."""
        assert simulator.sampling_rate == 4096.0
        assert simulator.segment_duration == 32.0
        assert simulator.osh_calculator is not None
        assert simulator.memory_field is not None
        assert len(simulator.echo_templates) > 0
    
    def test_base_waveform_generation(self, simulator, binary_params):
        """Test base waveform has correct properties."""
        signal = simulator._generate_base_waveform(binary_params)
        
        # Check signal properties
        assert len(signal.times) == int(simulator.sampling_rate * simulator.segment_duration)
        assert len(signal.strain_plus) == len(signal.times)
        assert len(signal.strain_cross) == len(signal.times)
        
        # Check merger occurs
        merger_idx = np.argmax(signal.amplitude)
        assert merger_idx > len(signal.times) // 2  # Merger after midpoint
        
        # Check frequency evolution
        inspiral_freq = signal.frequency[:merger_idx]
        assert np.all(np.diff(inspiral_freq[inspiral_freq > 0]) > 0)  # Increasing frequency
        
        # Check amplitude peaks at merger
        assert signal.amplitude[merger_idx] == np.max(signal.amplitude)
        
        # Check ringdown decay
        ringdown_amp = signal.amplitude[merger_idx + 100:]  # 100 samples after merger
        if len(ringdown_amp) > 10:
            assert np.all(np.diff(ringdown_amp[:10]) < 0)  # Decreasing amplitude
    
    def test_osh_metrics_calculation(self, simulator, binary_params):
        """Test OSH metrics are calculated correctly."""
        signal = simulator._generate_base_waveform(binary_params)
        metrics = simulator._calculate_merger_osh_metrics(binary_params, signal)
        
        # Check all required metrics are present
        assert 'merger_time' in metrics
        assert 'merger_idx' in metrics
        assert 'integrated_info' in metrics
        assert 'complexity' in metrics
        assert 'entropy_flux' in metrics
        assert 'rsp' in metrics
        assert 'info_curvature' in metrics
        assert 'memory_region' in metrics
        
        # Check metric values are reasonable
        assert metrics['integrated_info'] >= 0
        assert metrics['complexity'] >= 0
        assert metrics['entropy_flux'] > 0
        assert metrics['rsp'] > 0  # Should be positive for black holes
        assert metrics['info_curvature'] >= 0
        
        # Check horizon radius calculation
        expected_radius = 2.95e3 * binary_params.final_mass  # In meters
        assert abs(metrics['horizon_radius'] - expected_radius) < 100
    
    def test_echo_generation(self, simulator, binary_params, osh_params):
        """Test echo signal generation."""
        base_signal = simulator._generate_base_waveform(binary_params)
        osh_metrics = simulator._calculate_merger_osh_metrics(binary_params, base_signal)
        
        echo_signal = simulator._generate_osh_echoes(
            base_signal, binary_params, osh_params, osh_metrics
        )
        
        # Check echo signal has same length as base
        assert len(echo_signal.times) == len(base_signal.times)
        
        # Check echoes only appear after merger
        merger_idx = osh_metrics['merger_idx']
        pre_merger = echo_signal.strain_plus[:merger_idx]
        assert np.allclose(pre_merger, 0)  # No echoes before merger
        
        # Check post-merger has non-zero signal
        post_merger = echo_signal.strain_plus[merger_idx:]
        if osh_params.max_echo_orders > 0:
            assert not np.allclose(post_merger, 0)  # Should have echoes
    
    def test_echo_delay_calculation(self, simulator, binary_params, osh_params):
        """Test echo delay calculations follow OSH predictions."""
        osh_metrics = {
            'horizon_radius': 88575,  # 30 M_sun black hole
            'rsp': 1000,
            'info_curvature': 0.5
        }
        
        # Test first echo
        delay1 = simulator._calculate_echo_delay(binary_params, osh_params, osh_metrics, 1)
        base_delay = 2 * osh_metrics['horizon_radius'] / 3e8  # Light crossing time
        assert delay1 > base_delay  # Should be longer than light crossing
        
        # Test echo order scaling
        delay2 = simulator._calculate_echo_delay(binary_params, osh_params, osh_metrics, 2)
        assert delay2 > delay1  # Later echoes have longer delays
        
        # Check delay factor scaling
        expected_ratio = osh_params.echo_delay_factor
        actual_ratio = delay2 / delay1
        assert abs(actual_ratio - expected_ratio) < 0.5  # Approximate due to other factors
    
    def test_echo_amplitude_calculation(self, simulator, binary_params, osh_params):
        """Test echo amplitude follows OSH predictions."""
        osh_metrics = {
            'rsp': 1000,
            'info_curvature': 0.5,
            'memory_region': Mock(entropy=0.1, fidelity=0.95)
        }
        
        # Test amplitude decreases with echo order
        amp1 = simulator._calculate_echo_amplitude(binary_params, osh_params, osh_metrics, 1)
        amp2 = simulator._calculate_echo_amplitude(binary_params, osh_params, osh_metrics, 2)
        amp3 = simulator._calculate_echo_amplitude(binary_params, osh_params, osh_metrics, 3)
        
        assert 0 < amp3 < amp2 < amp1 <= 1  # Decreasing amplitude
        
        # Test RSP amplification effect
        osh_params_low_rsp = OSHEchoParameters(rsp_amplification=0.5)
        amp_low = simulator._calculate_echo_amplitude(
            binary_params, osh_params_low_rsp, osh_metrics, 1
        )
        assert amp_low < amp1  # Lower amplification = lower amplitude
    
    def test_full_simulation_with_echoes(self, simulator, binary_params, osh_params):
        """Test complete simulation produces valid waveform."""
        signal = simulator.simulate_merger_with_echoes(
            binary_params, osh_params, include_noise=False
        )
        
        # Check signal is valid
        assert signal is not None
        assert len(signal.times) > 0
        assert not np.any(np.isnan(signal.strain_plus))
        assert not np.any(np.isnan(signal.strain_cross))
        
        # Check diagnostics were computed
        assert hasattr(signal, 'diagnostics')
        assert 'echo_times' in signal.diagnostics
        assert 'echo_snr' in signal.diagnostics
    
    def test_echo_detection(self, simulator, binary_params, osh_params):
        """Test echo detection algorithm."""
        # Generate signal with known echoes
        signal = simulator.simulate_merger_with_echoes(
            binary_params, osh_params, include_noise=False
        )
        
        # Analyze for echoes
        analysis = simulator.analyze_echo_evidence(signal)
        
        # Check analysis results
        assert 'evidence_score' in analysis
        assert 'echo_times' in analysis
        assert 'n_echoes_detected' in analysis
        
        # Should detect at least one echo with these parameters
        if osh_params.max_echo_orders > 0:
            assert analysis['n_echoes_detected'] > 0
            assert analysis['evidence_score'] > 0.1
    
    def test_noise_addition(self, simulator, binary_params):
        """Test detector noise is added correctly."""
        # Generate clean signal
        clean_signal = simulator._generate_base_waveform(binary_params)
        
        # Add noise
        noisy_signal = simulator._add_detector_noise(clean_signal)
        
        # Check noise was added
        assert not np.array_equal(noisy_signal.strain_plus, clean_signal.strain_plus)
        
        # Check noise characteristics
        noise = noisy_signal.strain_plus - clean_signal.strain_plus
        noise_std = np.std(noise)
        assert 1e-24 < noise_std < 1e-21  # Reasonable noise level
    
    def test_information_curvature_field(self, simulator, binary_params):
        """Test information field calculation near horizon."""
        info_field = simulator._calculate_horizon_information_field(binary_params)
        
        # Check field shape
        assert info_field.shape == (simulator.info_geometry_grid_size, 
                                   simulator.info_geometry_grid_size)
        
        # Check field properties
        assert np.all(info_field >= 0)  # Non-negative
        assert np.max(info_field) == 1.0  # Normalized
        
        # Check increases near horizon (center of grid)
        center = info_field.shape[0] // 2
        assert info_field[center, 0] > info_field[center, -1]  # Higher at inner radius
    
    def test_schwarzschild_radius(self, simulator):
        """Test Schwarzschild radius calculation."""
        # Test for 1 solar mass
        r_s = simulator._schwarzschild_radius(1.0)
        expected = 2.95325e3  # meters
        assert abs(r_s - expected) < 1
        
        # Test scaling with mass
        r_s_10 = simulator._schwarzschild_radius(10.0)
        assert abs(r_s_10 - 10 * r_s) < 1
    
    def test_qnm_frequency(self, simulator, binary_params):
        """Test quasi-normal mode frequency calculation."""
        f_qnm = simulator._calculate_qnm_frequency(binary_params)
        
        # Check reasonable frequency range for stellar mass black holes
        assert 50 < f_qnm < 1000  # Hz
        
        # Test scaling with mass (inverse relationship)
        params_heavy = BinaryParameters(mass1=60, mass2=60)
        f_qnm_heavy = simulator._calculate_qnm_frequency(params_heavy)
        assert f_qnm_heavy < f_qnm  # Lower frequency for higher mass
    
    def test_hawking_temperature(self, simulator):
        """Test Hawking temperature calculation."""
        # Test for stellar mass black hole
        T_H = simulator._hawking_temperature(30.0)
        
        # Should be extremely cold
        assert T_H < 1e-6  # Kelvin
        
        # Test inverse mass relationship
        T_H_light = simulator._hawking_temperature(10.0)
        assert T_H_light > T_H  # Higher temperature for lower mass


class TestEchoAnalysis:
    """Test echo analysis algorithms."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator instance."""
        return GravitationalWaveEchoSimulator()
    
    def test_pattern_score_calculation(self, simulator):
        """Test pattern recognition in wavelet transform."""
        # Create synthetic signal with repeating pattern
        t = np.linspace(0, 1, 1000)
        signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz
        
        # Add echo
        echo_delay = 0.1
        echo_idx = int(echo_delay * 1000)
        signal[echo_idx:2*echo_idx] += 0.5 * signal[:echo_idx]
        
        # Compute wavelet transform
        cwt_matrix = scipy.signal.cwt(signal, scipy.signal.ricker, simulator.wavelet_scales)
        
        # Calculate pattern score
        score = simulator._calculate_pattern_score(cwt_matrix)
        
        # Should detect some pattern
        assert 0 < score < 1
    
    def test_evidence_score_calculation(self, simulator):
        """Test overall evidence score calculation."""
        # Test with strong evidence
        score_strong = simulator._calculate_evidence_score(
            correlations=[0.8, 0.7, 0.6],
            pattern_score=0.7,
            spacing_regularity=0.9,
            n_echoes=3
        )
        assert score_strong > 0.6
        
        # Test with weak evidence
        score_weak = simulator._calculate_evidence_score(
            correlations=[0.2, 0.1],
            pattern_score=0.1,
            spacing_regularity=0.2,
            n_echoes=1
        )
        assert score_weak < 0.3
        
        # Test with no evidence
        score_none = simulator._calculate_evidence_score(
            correlations=[],
            pattern_score=0,
            spacing_regularity=0,
            n_echoes=0
        )
        assert score_none < 0.1
    
    def test_statistical_significance(self, simulator):
        """Test p-value estimation."""
        # High evidence should give low p-value
        p_high = simulator._calculate_statistical_significance(0.8)
        assert p_high < 0.01
        
        # Low evidence should give high p-value
        p_low = simulator._calculate_statistical_significance(0.1)
        assert p_low > 0.5
        
        # Check monotonic relationship
        p_values = [simulator._calculate_statistical_significance(s) 
                   for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
        assert all(p_values[i] >= p_values[i+1] for i in range(len(p_values)-1))


class TestIntegration:
    """Integration tests for the complete system."""
    
    @pytest.fixture
    def simulator(self):
        """Create simulator with real components."""
        osh_calc = OSHMetricsCalculator()
        memory_field = MemoryFieldProper()
        return GravitationalWaveEchoSimulator(osh_calc, memory_field)
    
    def test_gw150914_like_system(self, simulator):
        """Test simulation of GW150914-like system."""
        # GW150914 parameters
        params = BinaryParameters(
            mass1=36,
            mass2=29,
            distance=410,
            spin1=np.array([0, 0, 0.3]),
            spin2=np.array([0, 0, -0.1])
        )
        
        osh_params = OSHEchoParameters(
            max_echo_orders=5,
            rsp_amplification=3.0
        )
        
        # Run simulation
        signal = simulator.simulate_merger_with_echoes(
            params, osh_params, include_noise=True
        )
        
        # Verify signal properties
        assert signal is not None
        assert signal.duration > 1.0  # At least 1 second
        
        # Check for echoes
        analysis = simulator.analyze_echo_evidence(signal)
        assert analysis['n_echoes_detected'] >= 0  # May or may not detect in noise
    
    def test_extreme_mass_ratio(self, simulator):
        """Test extreme mass ratio system."""
        params = BinaryParameters(
            mass1=100,
            mass2=1,
            distance=1000
        )
        
        # Should still produce valid waveform
        signal = simulator.simulate_merger_with_echoes(params)
        assert not np.any(np.isnan(signal.strain_plus))
    
    def test_high_spin_system(self, simulator):
        """Test rapidly spinning black holes."""
        params = BinaryParameters(
            mass1=25,
            mass2=25,
            spin1=np.array([0, 0, 0.95]),
            spin2=np.array([0, 0, 0.95]),
            distance=500
        )
        
        osh_params = OSHEchoParameters(
            information_curvature_coupling=0.5,  # Higher coupling for spinning BHs
            max_echo_orders=7
        )
        
        signal = simulator.simulate_merger_with_echoes(params, osh_params)
        
        # High spin should affect echo pattern
        if hasattr(signal, 'diagnostics'):
            assert len(signal.diagnostics.get('echo_times', [])) > 0


class TestSerialization:
    """Test data serialization for API."""
    
    @pytest.fixture
    def simulator(self):
        return GravitationalWaveEchoSimulator()
    
    def test_waveform_serialization(self, simulator, tmp_path):
        """Test saving waveform to file."""
        # Generate simple signal
        params = BinaryParameters(mass1=10, mass2=10, distance=100)
        signal = simulator.simulate_merger_with_echoes(params)
        
        # Save to file
        filename = tmp_path / "test_waveform.json"
        simulator.save_waveform(signal, str(filename))
        
        # Check file exists and contains data
        assert filename.exists()
        
        import json
        with open(filename) as f:
            data = json.load(f)
        
        assert 'times' in data
        assert 'strain_plus' in data
        assert 'strain_cross' in data
        assert len(data['times']) == len(data['strain_plus'])


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for echo simulation."""
    
    def test_simulation_speed(self, benchmark):
        """Benchmark full simulation speed."""
        simulator = GravitationalWaveEchoSimulator()
        params = BinaryParameters(mass1=30, mass2=30)
        
        # Benchmark simulation
        result = benchmark(simulator.simulate_merger_with_echoes, params)
        assert result is not None
    
    def test_echo_analysis_speed(self, benchmark):
        """Benchmark echo analysis speed."""
        simulator = GravitationalWaveEchoSimulator()
        
        # Generate test signal
        params = BinaryParameters(mass1=30, mass2=30)
        signal = simulator.simulate_merger_with_echoes(params)
        
        # Benchmark analysis
        result = benchmark(simulator.analyze_echo_evidence, signal)
        assert 'evidence_score' in result