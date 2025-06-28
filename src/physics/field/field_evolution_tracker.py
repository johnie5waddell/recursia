"""
field_evolution_tracker.py - Recursia Field Evolution Tracker

Comprehensive, OSH-aware temporal tracking system for quantum field evolution.
Provides advanced compression, change detection, trend analysis, and scientific
diagnostics aligned with the Organic Simulation Hypothesis framework.

Part of the Recursia Quantum Programming Language
"""

import logging
import threading
from threading import RLock
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
import numpy as np
import json
import pickle
import gzip
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

# Scientific computing imports
try:
    import scipy.signal
    import scipy.fft
    import scipy.stats
    import scipy.interpolate
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("SciPy not available - some advanced features will be limited")

# Recursia core imports
from src.core.data_classes import ChangeDetectionResult, ChangeType, CompressionMethod, DeltaRecord, EvolutionSnapshot, OSHMetrics, TrendAnalysis, TrendDirection
from src.core.utils import (
    global_error_manager, performance_profiler, colorize_text
)


class CompressionEngine:
    """Handles compression and decompression of field deltas."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CompressionEngine")
        self._compression_stats = defaultdict(lambda: {
            'count': 0, 'total_ratio': 0.0, 'total_time': 0.0
        })
        self._lock = RLock()

    def compress_delta(self, current_field: np.ndarray, previous_field: np.ndarray,
                      method: CompressionMethod = CompressionMethod.ADAPTIVE) -> DeltaRecord:
        """Compress the difference between two field states."""
        start_time = time.perf_counter()
        
        try:
            # Calculate delta
            delta = current_field - previous_field
            original_size = delta.nbytes
            
            # Choose compression method
            if method == CompressionMethod.ADAPTIVE:
                method = self._select_optimal_compression(delta)
            
            # Apply compression
            compressed_data, compression_ratio = self._apply_compression(delta, method)
            
            # Create delta record
            record = DeltaRecord(
                field_id="",  # Set by caller
                time_from=0.0,  # Set by caller
                time_to=0.0,    # Set by caller
                compression_method=method,
                compressed_data=compressed_data,
                compression_ratio=compression_ratio,
                base_field_shape=current_field.shape,
                metadata={
                    'original_size': original_size,
                    'compressed_size': len(compressed_data),
                    'delta_stats': {
                        'mean': float(np.mean(delta)),
                        'std': float(np.std(delta)),
                        'max_abs': float(np.max(np.abs(delta))),
                        'sparsity': float(np.count_nonzero(delta) / delta.size)
                    }
                }
            )
            
            # Update statistics
            compression_time = time.perf_counter() - start_time
            self._update_compression_stats(method, compression_ratio, compression_time)
            
            self.logger.debug(f"Compressed delta using {method.value}: "
                            f"{compression_ratio:.2f}x ratio in {compression_time:.4f}s")
            
            return record
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            global_error_manager.runtime_error(str(e))
            raise

    def decompress_delta(self, record: DeltaRecord, base_field: np.ndarray) -> np.ndarray:
        """Decompress a delta record and apply to base field."""
        try:
            # Decompress data
            delta = self._apply_decompression(record.compressed_data, 
                                            record.compression_method,
                                            record.base_field_shape)
            
            # Apply delta to base field
            result = base_field + delta
            
            self.logger.debug(f"Decompressed delta using {record.compression_method.value}")
            return result
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            global_error_manager.runtime_error(str(e))
            raise

    def _select_optimal_compression(self, delta: np.ndarray) -> CompressionMethod:
        """Select optimal compression method based on delta characteristics."""
        sparsity = np.count_nonzero(delta) / delta.size
        variance = np.var(delta)
        
        if sparsity < 0.1:
            return CompressionMethod.SPARSE
        elif variance > 1.0 and SCIPY_AVAILABLE:
            return CompressionMethod.WAVELET
        elif delta.size > 10000:
            return CompressionMethod.GZIP
        else:
            return CompressionMethod.DELTA

    def _apply_compression(self, delta: np.ndarray, method: CompressionMethod) -> Tuple[bytes, float]:
        """Apply specified compression method."""
        original_size = delta.nbytes
        
        if method == CompressionMethod.DELTA:
            # Simple delta compression
            compressed = gzip.compress(delta.tobytes())
            
        elif method == CompressionMethod.SPARSE:
            # Sparse matrix compression
            threshold = np.std(delta) * 0.1
            mask = np.abs(delta) > threshold
            sparse_data = {
                'mask': mask,
                'values': delta[mask],
                'shape': delta.shape,
                'dtype': str(delta.dtype)
            }
            compressed = gzip.compress(pickle.dumps(sparse_data))
            
        elif method == CompressionMethod.WAVELET and SCIPY_AVAILABLE:
            # FFT-based wavelet compression
            fft_data = scipy.fft.fftn(delta)
            # Keep only significant coefficients
            threshold = np.percentile(np.abs(fft_data), 90)
            fft_data[np.abs(fft_data) < threshold] = 0
            compressed = gzip.compress(pickle.dumps({
                'fft_data': fft_data,
                'shape': delta.shape,
                'dtype': str(delta.dtype)
            }))
            
        elif method == CompressionMethod.GZIP:
            # Raw gzip compression
            compressed = gzip.compress(delta.tobytes())
            
        else:
            # Fallback to delta
            compressed = gzip.compress(delta.tobytes())
        
        compression_ratio = original_size / len(compressed) if compressed else 1.0
        return compressed, compression_ratio

    def _apply_decompression(self, compressed_data: bytes, method: CompressionMethod,
                           shape: Tuple[int, ...]) -> np.ndarray:
        """Apply specified decompression method."""
        if method == CompressionMethod.DELTA or method == CompressionMethod.GZIP:
            # Simple decompression
            decompressed = gzip.decompress(compressed_data)
            return np.frombuffer(decompressed).reshape(shape)
            
        elif method == CompressionMethod.SPARSE:
            # Sparse decompression
            sparse_data = pickle.loads(gzip.decompress(compressed_data))
            result = np.zeros(sparse_data['shape'], dtype=sparse_data['dtype'])
            result[sparse_data['mask']] = sparse_data['values']
            return result
            
        elif method == CompressionMethod.WAVELET and SCIPY_AVAILABLE:
            # FFT decompression
            fft_info = pickle.loads(gzip.decompress(compressed_data))
            return scipy.fft.ifftn(fft_info['fft_data']).real.astype(fft_info['dtype'])
            
        else:
            # Fallback
            decompressed = gzip.decompress(compressed_data)
            return np.frombuffer(decompressed).reshape(shape)

    def _update_compression_stats(self, method: CompressionMethod, ratio: float, time_taken: float):
        """Update compression statistics."""
        with self._lock:
            stats = self._compression_stats[method.value]
            stats['count'] += 1
            stats['total_ratio'] += ratio
            stats['total_time'] += time_taken

    def get_compression_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get compression performance statistics."""
        with self._lock:
            result = {}
            for method, stats in self._compression_stats.items():
                if stats['count'] > 0:
                    result[method] = {
                        'count': stats['count'],
                        'avg_ratio': stats['total_ratio'] / stats['count'],
                        'avg_time': stats['total_time'] / stats['count'],
                        'total_time': stats['total_time']
                    }
            return result


class ChangeDetector:
    """Detects significant changes in field evolution."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ChangeDetector")
        self._detection_thresholds = {
            'sudden_change': 3.0,  # Standard deviations
            'oscillation_correlation': 0.7,
            'phase_transition_metrics': 2,  # Number of metrics changing
            'coherence_collapse': 0.3,  # Absolute change threshold
            'entropy_cascade': 0.2,
            'rsp_instability': 0.5,
            'energy_spike': 5.0
        }

    def detect_changes(self, snapshots: List[EvolutionSnapshot]) -> List[ChangeDetectionResult]:
        """Detect changes in a series of field snapshots."""
        if len(snapshots) < 3:
            return []

        results = []
        
        try:
            # Extract time series data
            times = np.array([s.time_point for s in snapshots])
            coherence = np.array([s.osh_metrics.coherence if s.osh_metrics else 0.0 for s in snapshots])
            entropy = np.array([s.osh_metrics.entropy if s.osh_metrics else 0.0 for s in snapshots])
            energy = np.array([s.energy_density or 0.0 for s in snapshots])
            rsp = np.array([s.osh_metrics.rsp if s.osh_metrics else 0.0 for s in snapshots])

            # Detect sudden changes
            results.extend(self._detect_sudden_changes(snapshots, times, coherence, entropy, energy))
            
            # Detect oscillatory behavior
            results.extend(self._detect_oscillations(snapshots, times, coherence, energy))
            
            # Detect phase transitions
            results.extend(self._detect_phase_transitions(snapshots, times, coherence, entropy, rsp))
            
            # Detect specific OSH phenomena
            results.extend(self._detect_osh_phenomena(snapshots, times, coherence, entropy, rsp, energy))
            
            self.logger.debug(f"Detected {len(results)} changes in {len(snapshots)} snapshots")
            
        except Exception as e:
            self.logger.error(f"Change detection failed: {e}")
            global_error_manager.runtime_error(str(e))

        return results

    def _detect_sudden_changes(self, snapshots: List[EvolutionSnapshot], times: np.ndarray,
                              coherence: np.ndarray, entropy: np.ndarray, 
                              energy: np.ndarray) -> List[ChangeDetectionResult]:
        """Detect sudden spikes or drops in field metrics."""
        results = []
        threshold = self._detection_thresholds['sudden_change']
        
        for metric_name, metric_data in [('coherence', coherence), ('entropy', entropy), ('energy', energy)]:
            if len(metric_data) < 3:
                continue
                
            # Calculate rolling statistics
            window_size = min(5, len(metric_data) // 2)
            if window_size < 2:
                continue
                
            try:
                # Simple rolling mean and std
                for i in range(window_size, len(metric_data)):
                    window = metric_data[max(0, i-window_size):i]
                    mean_val = np.mean(window)
                    std_val = np.std(window)
                    
                    if std_val > 0:
                        z_score = abs(metric_data[i] - mean_val) / std_val
                        if z_score > threshold:
                            results.append(ChangeDetectionResult(
                                field_id=snapshots[i].field_id,
                                change_type=ChangeType.SUDDEN,
                                time_detected=times[i],
                                confidence=min(z_score / threshold, 1.0),
                                magnitude=z_score * std_val,
                                description=f"Sudden {metric_name} change: {z_score:.2f}σ deviation"
                            ))
            except Exception as e:
                self.logger.warning(f"Sudden change detection failed for {metric_name}: {e}")
                
        return results

    def _detect_oscillations(self, snapshots: List[EvolutionSnapshot], times: np.ndarray,
                           coherence: np.ndarray, energy: np.ndarray) -> List[ChangeDetectionResult]:
        """Detect oscillatory patterns in field evolution."""
        results = []
        
        if not SCIPY_AVAILABLE or len(coherence) < 10:
            return results
            
        try:
            # Autocorrelation analysis
            for metric_name, metric_data in [('coherence', coherence), ('energy', energy)]:
                if np.std(metric_data) > 0:
                    autocorr = np.correlate(metric_data - np.mean(metric_data), 
                                          metric_data - np.mean(metric_data), mode='full')
                    autocorr = autocorr[autocorr.size // 2:]
                    autocorr = autocorr / autocorr[0]
                    
                    # Look for periodic peaks
                    if len(autocorr) > 5:
                        peaks, _ = scipy.signal.find_peaks(autocorr[1:], height=self._detection_thresholds['oscillation_correlation'])
                        if len(peaks) > 0:
                            period = peaks[0] + 1
                            confidence = float(autocorr[period])
                            results.append(ChangeDetectionResult(
                                field_id=snapshots[0].field_id,
                                change_type=ChangeType.OSCILLATORY,
                                time_detected=times[-1],
                                confidence=confidence,
                                magnitude=np.std(metric_data),
                                description=f"Oscillatory {metric_name} with period ~{period} steps"
                            ))
        except Exception as e:
            self.logger.warning(f"Oscillation detection failed: {e}")
            
        return results

    def _detect_phase_transitions(self, snapshots: List[EvolutionSnapshot], times: np.ndarray,
                                 coherence: np.ndarray, entropy: np.ndarray, 
                                 rsp: np.ndarray) -> List[ChangeDetectionResult]:
        """Detect phase transitions where multiple metrics change simultaneously."""
        results = []
        threshold = self._detection_thresholds['phase_transition_metrics']
        
        if len(snapshots) < 5:
            return results
            
        try:
            # Look for points where multiple metrics change significantly
            for i in range(2, len(snapshots) - 2):
                changes = 0
                change_magnitudes = []
                
                # Check coherence change
                coh_change = abs(coherence[i] - coherence[i-1])
                if coh_change > 0.1:
                    changes += 1
                    change_magnitudes.append(coh_change)
                
                # Check entropy change
                ent_change = abs(entropy[i] - entropy[i-1])
                if ent_change > 0.1:
                    changes += 1
                    change_magnitudes.append(ent_change)
                
                # Check RSP change
                rsp_change = abs(rsp[i] - rsp[i-1])
                if rsp_change > 0.2:
                    changes += 1
                    change_magnitudes.append(rsp_change)
                
                if changes >= threshold:
                    avg_magnitude = np.mean(change_magnitudes) if change_magnitudes else 0.0
                    results.append(ChangeDetectionResult(
                        field_id=snapshots[i].field_id,
                        change_type=ChangeType.PHASE_TRANSITION,
                        time_detected=times[i],
                        confidence=min(changes / 3.0, 1.0),
                        magnitude=avg_magnitude,
                        description=f"Phase transition: {changes} metrics changed simultaneously"
                    ))
        except Exception as e:
            self.logger.warning(f"Phase transition detection failed: {e}")
            
        return results

    def _detect_osh_phenomena(self, snapshots: List[EvolutionSnapshot], times: np.ndarray,
                             coherence: np.ndarray, entropy: np.ndarray, 
                             rsp: np.ndarray, energy: np.ndarray) -> List[ChangeDetectionResult]:
        """Detect OSH-specific phenomena."""
        results = []
        
        try:
            # Coherence collapse detection
            coh_drops = np.diff(coherence)
            collapse_indices = np.where(coh_drops < -self._detection_thresholds['coherence_collapse'])[0]
            for idx in collapse_indices:
                if idx + 1 < len(snapshots):
                    results.append(ChangeDetectionResult(
                        field_id=snapshots[idx + 1].field_id,
                        change_type=ChangeType.COHERENCE_COLLAPSE,
                        time_detected=times[idx + 1],
                        confidence=min(abs(coh_drops[idx]) / self._detection_thresholds['coherence_collapse'], 1.0),
                        magnitude=abs(coh_drops[idx]),
                        description=f"Coherence collapse: {coh_drops[idx]:.3f} drop"
                    ))
            
            # Entropy cascade detection
            ent_spikes = np.diff(entropy)
            cascade_indices = np.where(ent_spikes > self._detection_thresholds['entropy_cascade'])[0]
            for idx in cascade_indices:
                if idx + 1 < len(snapshots):
                    results.append(ChangeDetectionResult(
                        field_id=snapshots[idx + 1].field_id,
                        change_type=ChangeType.ENTROPY_CASCADE,
                        time_detected=times[idx + 1],
                        confidence=min(ent_spikes[idx] / self._detection_thresholds['entropy_cascade'], 1.0),
                        magnitude=ent_spikes[idx],
                        description=f"Entropy cascade: {ent_spikes[idx]:.3f} increase"
                    ))
            
            # RSP instability detection
            rsp_changes = np.abs(np.diff(rsp))
            unstable_indices = np.where(rsp_changes > self._detection_thresholds['rsp_instability'])[0]
            for idx in unstable_indices:
                if idx + 1 < len(snapshots):
                    results.append(ChangeDetectionResult(
                        field_id=snapshots[idx + 1].field_id,
                        change_type=ChangeType.RSP_INSTABILITY,
                        time_detected=times[idx + 1],
                        confidence=min(rsp_changes[idx] / self._detection_thresholds['rsp_instability'], 1.0),
                        magnitude=rsp_changes[idx],
                        description=f"RSP instability: {rsp_changes[idx]:.3f} fluctuation"
                    ))
            
            # Energy spike detection
            energy_z_scores = np.abs((energy - np.mean(energy)) / (np.std(energy) + 1e-10))
            spike_indices = np.where(energy_z_scores > self._detection_thresholds['energy_spike'])[0]
            for idx in spike_indices:
                if idx < len(snapshots):
                    results.append(ChangeDetectionResult(
                        field_id=snapshots[idx].field_id,
                        change_type=ChangeType.ENERGY_SPIKE,
                        time_detected=times[idx],
                        confidence=min(energy_z_scores[idx] / self._detection_thresholds['energy_spike'], 1.0),
                        magnitude=energy_z_scores[idx],
                        description=f"Energy spike: {energy_z_scores[idx]:.2f}σ above normal"
                    ))
                    
        except Exception as e:
            self.logger.warning(f"OSH phenomena detection failed: {e}")
            
        return results


class TrendAnalyzer:
    """Analyzes trends in field evolution data."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TrendAnalyzer")

    def analyze_trends(self, snapshots: List[EvolutionSnapshot]) -> TrendAnalysis:
        """Perform comprehensive trend analysis on field snapshots."""
        if len(snapshots) < 3:
            return TrendAnalysis(
                field_id=snapshots[0].field_id if snapshots else "",
                analysis_time=time.time(),
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                confidence=0.0
            )

        try:
            # Extract metrics
            times = np.array([s.time_point for s in snapshots])
            coherence = np.array([s.osh_metrics.coherence if s.osh_metrics else 0.0 for s in snapshots])
            entropy = np.array([s.osh_metrics.entropy if s.osh_metrics else 0.0 for s in snapshots])
            energy = np.array([s.energy_density or 0.0 for s in snapshots])
            
            # Calculate basic statistics
            statistics = self._calculate_statistics(coherence, entropy, energy)
            
            # Determine trend direction and strength
            trend_direction, trend_strength, confidence = self._analyze_trend_direction(
                times, coherence, entropy, energy
            )
            
            # Calculate autocorrelation
            autocorr = self._calculate_autocorrelation(coherence) if SCIPY_AVAILABLE else None
            
            # Find dominant frequencies
            frequencies = self._find_dominant_frequencies(coherence, times) if SCIPY_AVAILABLE else None
            
            # Generate prediction
            prediction = self._generate_prediction(times, coherence, entropy, energy)
            
            result = TrendAnalysis(
                field_id=snapshots[0].field_id,
                analysis_time=time.time(),
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                confidence=confidence,
                statistics=statistics,
                autocorrelation=autocorr,
                dominant_frequencies=frequencies,
                prediction=prediction
            )
            
            self.logger.debug(f"Trend analysis completed: {trend_direction.value} "
                            f"(strength: {trend_strength:.3f}, confidence: {confidence:.3f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            global_error_manager.runtime_error(str(e))
            return TrendAnalysis(
                field_id=snapshots[0].field_id,
                analysis_time=time.time(),
                trend_direction=TrendDirection.STABLE,
                trend_strength=0.0,
                confidence=0.0
            )

    def _calculate_statistics(self, coherence: np.ndarray, entropy: np.ndarray, 
                            energy: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics for the metrics."""
        try:
            return {
                'coherence_mean': float(np.mean(coherence)),
                'coherence_std': float(np.std(coherence)),
                'coherence_min': float(np.min(coherence)),
                'coherence_max': float(np.max(coherence)),
                'coherence_skew': float(scipy.stats.skew(coherence)) if SCIPY_AVAILABLE else 0.0,
                'coherence_kurtosis': float(scipy.stats.kurtosis(coherence)) if SCIPY_AVAILABLE else 0.0,
                'entropy_mean': float(np.mean(entropy)),
                'entropy_std': float(np.std(entropy)),
                'entropy_min': float(np.min(entropy)),
                'entropy_max': float(np.max(entropy)),
                'entropy_skew': float(scipy.stats.skew(entropy)) if SCIPY_AVAILABLE else 0.0,
                'entropy_kurtosis': float(scipy.stats.kurtosis(entropy)) if SCIPY_AVAILABLE else 0.0,
                'energy_mean': float(np.mean(energy)),
                'energy_std': float(np.std(energy)),
                'energy_min': float(np.min(energy)),
                'energy_max': float(np.max(energy)),
                'coherence_entropy_correlation': float(np.corrcoef(coherence, entropy)[0, 1]) if len(coherence) > 1 else 0.0,
                'coherence_energy_correlation': float(np.corrcoef(coherence, energy)[0, 1]) if len(coherence) > 1 else 0.0
            }
        except Exception as e:
            self.logger.warning(f"Statistics calculation failed: {e}")
            return {}

    def _analyze_trend_direction(self, times: np.ndarray, coherence: np.ndarray,
                               entropy: np.ndarray, energy: np.ndarray) -> Tuple[TrendDirection, float, float]:
        """Analyze the overall trend direction of the metrics."""
        try:
            # Calculate linear trends for each metric
            coh_slope, coh_r2 = self._calculate_linear_trend(times, coherence)
            ent_slope, ent_r2 = self._calculate_linear_trend(times, entropy)
            eng_slope, eng_r2 = self._calculate_linear_trend(times, energy)
            
            # Weight by R² values
            weights = np.array([coh_r2, ent_r2, eng_r2])
            slopes = np.array([coh_slope, ent_slope, eng_slope])
            
            if np.sum(weights) > 0:
                weighted_slope = np.average(slopes, weights=weights)
                confidence = float(np.mean(weights))
            else:
                weighted_slope = np.mean(slopes)
                confidence = 0.1
            
            trend_strength = abs(weighted_slope)
            
            # Determine trend direction
            if trend_strength < 0.01:
                direction = TrendDirection.STABLE
            elif weighted_slope > 0.05:
                direction = TrendDirection.INCREASING
            elif weighted_slope < -0.05:
                direction = TrendDirection.DECREASING
            else:
                # Check for oscillations
                coh_oscillations = self._detect_oscillations_simple(coherence)
                if coh_oscillations > 0.3:
                    direction = TrendDirection.OSCILLATING
                elif trend_strength > 0.1:
                    direction = TrendDirection.CHAOTIC
                else:
                    direction = TrendDirection.STABLE
            
            return direction, trend_strength, confidence
            
        except Exception as e:
            self.logger.warning(f"Trend direction analysis failed: {e}")
            return TrendDirection.STABLE, 0.0, 0.0

    def _calculate_linear_trend(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Calculate linear trend slope and R²."""
        if len(x) < 2 or np.all(y == y[0]):
            return 0.0, 0.0
            
        try:
            slope, intercept, r_value, _, _ = scipy.stats.linregress(x, y) if SCIPY_AVAILABLE else (0, 0, 0, 0, 0)
            return float(slope), float(r_value**2) if SCIPY_AVAILABLE else 0.0
        except Exception:
            return 0.0, 0.0

    def _detect_oscillations_simple(self, data: np.ndarray) -> float:
        """Simple oscillation detection based on zero crossings."""
        if len(data) < 4:
            return 0.0
            
        try:
            # Detrend data
            detrended = data - np.mean(data)
            
            # Count zero crossings
            zero_crossings = np.sum(np.diff(np.signbit(detrended)))
            
            # Normalize by length
            oscillation_score = zero_crossings / len(data)
            
            return float(oscillation_score)
        except Exception:
            return 0.0

    def _calculate_autocorrelation(self, data: np.ndarray) -> np.ndarray:
        """Calculate autocorrelation function."""
        if not SCIPY_AVAILABLE or len(data) < 5:
            return np.array([])
            
        try:
            # Center the data
            centered = data - np.mean(data)
            
            # Calculate autocorrelation
            autocorr = np.correlate(centered, centered, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Normalize
            if autocorr[0] != 0:
                autocorr = autocorr / autocorr[0]
            
            return autocorr[:min(len(autocorr), 20)]  # Return first 20 lags
        except Exception as e:
            self.logger.warning(f"Autocorrelation calculation failed: {e}")
            return np.array([])

    def _find_dominant_frequencies(self, data: np.ndarray, times: np.ndarray) -> List[float]:
        """Find dominant frequencies in the data."""
        if not SCIPY_AVAILABLE or len(data) < 10:
            return []
            
        try:
            # Calculate sampling rate
            dt = np.mean(np.diff(times)) if len(times) > 1 else 1.0
            fs = 1.0 / dt if dt > 0 else 1.0
            
            # Compute power spectral density
            frequencies, psd = scipy.signal.periodogram(data, fs=fs)
            
            # Find peaks in PSD
            peaks, properties = scipy.signal.find_peaks(psd, height=np.max(psd) * 0.1)
            
            # Sort by power and return top frequencies
            peak_powers = psd[peaks]
            sorted_indices = np.argsort(peak_powers)[::-1]
            
            dominant_freqs = []
            for idx in sorted_indices[:5]:  # Top 5 frequencies
                freq = frequencies[peaks[idx]]
                if freq > 0:  # Exclude DC component
                    dominant_freqs.append(float(freq))
            
            return dominant_freqs
        except Exception as e:
            self.logger.warning(f"Frequency analysis failed: {e}")
            return []

    def _generate_prediction(self, times: np.ndarray, coherence: np.ndarray,
                           entropy: np.ndarray, energy: np.ndarray) -> Dict[str, Any]:
        """Generate short-term predictions for the metrics."""
        if len(times) < 3:
            return {}
            
        try:
            prediction = {}
            
            # Simple linear extrapolation for next few time steps
            dt = np.mean(np.diff(times)) if len(times) > 1 else 1.0
            future_times = np.array([times[-1] + dt, times[-1] + 2*dt, times[-1] + 3*dt])
            
            for metric_name, metric_data in [('coherence', coherence), ('entropy', entropy), ('energy', energy)]:
                try:
                    # Fit linear trend
                    slope, intercept, r_value, _, std_err = scipy.stats.linregress(times, metric_data) if SCIPY_AVAILABLE else (0, np.mean(metric_data), 0, 0, np.std(metric_data))
                    
                    # Predict future values
                    future_values = slope * future_times + intercept
                    
                    # Calculate confidence based on R² and standard error
                    confidence = float(r_value**2) if SCIPY_AVAILABLE else 0.1
                    uncertainty = float(std_err * 1.96) if SCIPY_AVAILABLE else float(np.std(metric_data))  # 95% CI
                    
                    prediction[metric_name] = {
                        'future_times': future_times.tolist(),
                        'predicted_values': future_values.tolist(),
                        'confidence': confidence,
                        'uncertainty': uncertainty,
                        'trend_slope': float(slope) if SCIPY_AVAILABLE else 0.0
                    }
                except Exception as e:
                    self.logger.warning(f"Prediction failed for {metric_name}: {e}")
                    
            return prediction
        except Exception as e:
            self.logger.warning(f"Prediction generation failed: {e}")
            return {}


class FieldEvolutionTracker:
    """Main class for tracking field evolution with comprehensive analytics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the field evolution tracker."""
        self.logger = logging.getLogger(f"{__name__}.FieldEvolutionTracker")
        
        # Configuration
        default_config = {
            'max_history_size': 1000,
            'compression_enabled': True,
            'change_detection_enabled': True,
            'trend_analysis_enabled': True,
            'periodic_snapshot_interval': 10,
            'change_detection_threshold': 0.1,
            'memory_limit_mb': 500,
            'enable_interpolation': True,
            'export_format': 'json'  # json, pickle, or both
        }
        self.config = {**default_config, **(config or {})}
        
        # Thread safety
        self._lock = RLock()
        
        # Storage for field histories
        self._field_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['max_history_size'])
        )
        self._full_snapshots: Dict[str, Dict[int, EvolutionSnapshot]] = defaultdict(dict)
        self._delta_records: Dict[str, List[DeltaRecord]] = defaultdict(list)
        
        # Change detection results
        self._change_detection_results: Dict[str, List[ChangeDetectionResult]] = defaultdict(list)
        self._trend_analyses: Dict[str, TrendAnalysis] = {}
        
        # Statistics and monitoring
        self._tracking_stats = {
            'total_snapshots': 0,
            'total_fields': 0,
            'memory_usage_mb': 0.0,
            'compression_ratio_avg': 1.0,
            'last_cleanup_time': time.time()
        }
        
        # Subsystem components
        self.compression_engine = CompressionEngine()
        self.change_detector = ChangeDetector()
        self.trend_analyzer = TrendAnalyzer()
        
        # Background processing
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="FieldEvolution")
        self._shutdown = False
        
        self.logger.info("FieldEvolutionTracker initialized with configuration")

    def record_field_state(self, field_id: str, field_values: np.ndarray, 
                          time_point: float, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Record a field state snapshot with full analytics."""
        try:
            with performance_profiler.timed_context("field_evolution_record"):
                # Create OSH metrics (placeholder - would integrate with actual OSH system)
                osh_metrics = self._calculate_osh_metrics(field_values)
                
                # Create snapshot
                snapshot = EvolutionSnapshot(
                    field_id=field_id,
                    time_point=time_point,
                    field_values=field_values.copy(),
                    osh_metrics=osh_metrics,
                    metadata=metadata or {},
                    evolution_index=len(self._field_history[field_id])
                )
                
                with self._lock:
                    # Get previous snapshot for delta compression
                    previous_snapshot = None
                    if self._field_history[field_id]:
                        previous_snapshot = self._field_history[field_id][-1]
                    
                    # Add to history
                    self._field_history[field_id].append(snapshot)
                    
                    # Create compressed delta if enabled and previous exists
                    if (self.config['compression_enabled'] and previous_snapshot and 
                        previous_snapshot.field_values.shape == field_values.shape):
                        try:
                            delta_record = self.compression_engine.compress_delta(
                                field_values, previous_snapshot.field_values
                            )
                            delta_record.field_id = field_id
                            delta_record.time_from = previous_snapshot.time_point
                            delta_record.time_to = time_point
                            self._delta_records[field_id].append(delta_record)
                        except Exception as e:
                            self.logger.warning(f"Delta compression failed for {field_id}: {e}")
                    
                    # Store periodic full snapshots
                    if (snapshot.evolution_index % self.config['periodic_snapshot_interval'] == 0):
                        self._full_snapshots[field_id][snapshot.evolution_index] = snapshot
                    
                    # Update statistics
                    self._tracking_stats['total_snapshots'] += 1
                    if field_id not in [s.field_id for history in self._field_history.values() for s in history]:
                        self._tracking_stats['total_fields'] += 1
                
                # Background analytics
                if self.config['change_detection_enabled'] or self.config['trend_analysis_enabled']:
                    self._executor.submit(self._perform_background_analysis, field_id)
                
                self.logger.debug(f"Recorded field state for {field_id} at t={time_point:.6f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to record field state for {field_id}: {e}")
            global_error_manager.runtime_error(str(e))
            return False

    def get_field_at_time(self, field_id: str, time_point: float) -> Optional[np.ndarray]:
        """Reconstruct field at arbitrary time point using interpolation."""
        if not self.config['enable_interpolation']:
            return None
            
        try:
            with self._lock:
                history = self._field_history.get(field_id, deque())
                if len(history) < 2:
                    return None
                
                # Find surrounding snapshots
                times = [s.time_point for s in history]
                if time_point <= times[0]:
                    return history[0].field_values.copy()
                elif time_point >= times[-1]:
                    return history[-1].field_values.copy()
                
                # Find bracketing snapshots
                for i in range(len(times) - 1):
                    if times[i] <= time_point <= times[i + 1]:
                        # Linear interpolation
                        t0, t1 = times[i], times[i + 1]
                        f0, f1 = history[i].field_values, history[i + 1].field_values
                        
                        alpha = (time_point - t0) / (t1 - t0) if t1 != t0 else 0.0
                        interpolated = f0 * (1 - alpha) + f1 * alpha
                        
                        self.logger.debug(f"Interpolated field {field_id} at t={time_point:.6f}")
                        return interpolated
                
                return None
                
        except Exception as e:
            self.logger.error(f"Field interpolation failed for {field_id}: {e}")
            global_error_manager.runtime_error(str(e))
            return None

    def analyze_field_trends(self, field_id: str) -> Optional[TrendAnalysis]:
        """Perform comprehensive trend analysis on field history."""
        if not self.config['trend_analysis_enabled']:
            return None
            
        try:
            with self._lock:
                history = list(self._field_history.get(field_id, deque()))
                
            if len(history) < 3:
                return None
                
            analysis = self.trend_analyzer.analyze_trends(history)
            
            with self._lock:
                self._trend_analyses[field_id] = analysis
                
            self.logger.debug(f"Trend analysis completed for {field_id}: "
                            f"{analysis.trend_direction.value} with strength {analysis.trend_strength:.3f}")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed for {field_id}: {e}")
            global_error_manager.runtime_error(str(e))
            return None

    def get_change_detection_results(self, field_id: str) -> List[ChangeDetectionResult]:
        """Get change detection results for a field."""
        with self._lock:
            return list(self._change_detection_results.get(field_id, []))

    def find_significant_changes(self, field_id: str, threshold: float = 0.1) -> List[Tuple[float, ChangeType]]:
        """Find significant changes above threshold."""
        results = []
        try:
            changes = self.get_change_detection_results(field_id)
            for change in changes:
                if change.magnitude >= threshold:
                    results.append((change.time_detected, change.change_type))
        except Exception as e:
            self.logger.error(f"Finding significant changes failed for {field_id}: {e}")
        return results

    def get_field_statistics(self, field_id: str) -> Dict[str, Any]:
        """Get comprehensive statistics for a field."""
        try:
            with self._lock:
                history = list(self._field_history.get(field_id, deque()))
                
            if not history:
                return {}
                
            # Basic statistics
            stats = {
                'field_id': field_id,
                'snapshot_count': len(history),
                'time_span': history[-1].time_point - history[0].time_point if len(history) > 1 else 0.0,
                'first_recorded': history[0].time_point,
                'last_recorded': history[-1].time_point,
            }
            
            # OSH metrics statistics
            if history[0].osh_metrics:
                coherence_values = [s.osh_metrics.coherence for s in history if s.osh_metrics]
                entropy_values = [s.osh_metrics.entropy for s in history if s.osh_metrics]
                rsp_values = [s.osh_metrics.rsp for s in history if s.osh_metrics]
                
                if coherence_values:
                    stats.update({
                        'coherence_mean': float(np.mean(coherence_values)),
                        'coherence_std': float(np.std(coherence_values)),
                        'coherence_min': float(np.min(coherence_values)),
                        'coherence_max': float(np.max(coherence_values)),
                        'entropy_mean': float(np.mean(entropy_values)),
                        'entropy_std': float(np.std(entropy_values)),
                        'rsp_mean': float(np.mean(rsp_values)),
                        'rsp_std': float(np.std(rsp_values))
                    })
            
            # Change detection summary
            changes = self.get_change_detection_results(field_id)
            if changes:
                change_types = [c.change_type for c in changes]
                stats['change_count'] = len(changes)
                stats['change_types'] = list(set(ct.value for ct in change_types))
                stats['avg_change_magnitude'] = float(np.mean([c.magnitude for c in changes]))
            
            # Trend analysis summary
            trend = self._trend_analyses.get(field_id)
            if trend:
                stats['trend_direction'] = trend.trend_direction.value
                stats['trend_strength'] = trend.trend_strength
                stats['trend_confidence'] = trend.confidence
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Statistics calculation failed for {field_id}: {e}")
            global_error_manager.runtime_error(str(e))
            return {}

    def get_field_history_summary(self, field_id: str) -> List[Dict[str, Any]]:
        """Get summary of field history snapshots."""
        try:
            with self._lock:
                history = list(self._field_history.get(field_id, deque()))
                
            summary = []
            for snapshot in history:
                entry = {
                    'time_point': snapshot.time_point,
                    'evolution_index': snapshot.evolution_index,
                    'field_shape': snapshot.field_values.shape,
                    'energy_density': snapshot.energy_density,
                    'gradient_magnitude': snapshot.gradient_magnitude
                }
                
                if snapshot.osh_metrics:
                    entry.update({
                        'coherence': snapshot.osh_metrics.coherence,
                        'entropy': snapshot.osh_metrics.entropy,
                        'strain': snapshot.osh_metrics.strain,
                        'rsp': snapshot.osh_metrics.rsp
                    })
                    
                summary.append(entry)
                
            return summary
            
        except Exception as e:
            self.logger.error(f"History summary failed for {field_id}: {e}")
            return []

    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get overall tracking system statistics."""
        try:
            with self._lock:
                stats = self._tracking_stats.copy()
                
            # Add compression statistics
            compression_stats = self.compression_engine.get_compression_statistics()
            if compression_stats:
                stats['compression'] = compression_stats
                
            # Calculate memory usage
            total_memory = 0
            field_counts = {}
            
            with self._lock:
                for field_id, history in self._field_history.items():
                    field_counts[field_id] = len(history)
                    for snapshot in history:
                        total_memory += snapshot.field_values.nbytes
                        
                for field_id, deltas in self._delta_records.items():
                    for delta in deltas:
                        total_memory += delta.get_memory_size()
            
            stats['memory_usage_mb'] = total_memory / (1024 * 1024)
            stats['field_counts'] = field_counts
            stats['active_fields'] = len(self._field_history)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Tracking statistics failed: {e}")
            return {}

    def export_field_history(self, field_id: str, filename: Optional[str] = None) -> str:
        """Export field history to file."""
        try:
            if filename is None:
                timestamp = int(time.time())
                filename = f"field_evolution_{field_id}_{timestamp}.{self.config['export_format']}"
            
            export_data = {
                'field_id': field_id,
                'export_time': time.time(),
                'snapshots': [],
                'change_detection_results': [],
                'trend_analysis': None,
                'statistics': self.get_field_statistics(field_id)
            }
            
            # Export snapshots
            with self._lock:
                history = list(self._field_history.get(field_id, deque()))
                for snapshot in history:
                    snapshot_dict = snapshot.to_dict()
                    # Convert numpy array to list for JSON serialization
                    snapshot_dict['field_values'] = snapshot.field_values.tolist()
                    export_data['snapshots'].append(snapshot_dict)
                
                # Export change detection results
                changes = self._change_detection_results.get(field_id, [])
                for change in changes:
                    change_dict = {
                        'field_id': change.field_id,
                        'change_type': change.change_type.value,
                        'time_detected': change.time_detected,
                        'confidence': change.confidence,
                        'magnitude': change.magnitude,
                        'duration': change.duration,
                        'description': change.description
                    }
                    export_data['change_detection_results'].append(change_dict)
                
                # Export trend analysis
                trend = self._trend_analyses.get(field_id)
                if trend:
                    export_data['trend_analysis'] = {
                        'trend_direction': trend.trend_direction.value,
                        'trend_strength': trend.trend_strength,
                        'confidence': trend.confidence,
                        'statistics': trend.statistics,
                        'dominant_frequencies': trend.dominant_frequencies,
                        'prediction': trend.prediction
                    }
            
            # Write to file
            filepath = Path(filename)
            if self.config['export_format'] == 'json':
                with open(filepath, 'w') as f:
                    json.dump(export_data, f, indent=2)
            elif self.config['export_format'] == 'pickle':
                with open(filepath, 'wb') as f:
                    pickle.dump(export_data, f)
            else:
                # Export both formats
                json_path = filepath.with_suffix('.json')
                pickle_path = filepath.with_suffix('.pkl')
                
                with open(json_path, 'w') as f:
                    json.dump(export_data, f, indent=2)
                with open(pickle_path, 'wb') as f:
                    pickle.dump(export_data, f)
                    
                filename = f"{json_path} and {pickle_path}"
            
            self.logger.info(f"Exported field history for {field_id} to {filename}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Export failed for {field_id}: {e}")
            global_error_manager.runtime_error(str(e))
            raise

    def import_field_history(self, filename: str) -> str:
        """Import field history from file."""
        try:
            filepath = Path(filename)
            
            # Load data
            if filepath.suffix == '.json':
                with open(filepath, 'r') as f:
                    import_data = json.load(f)
            elif filepath.suffix in ['.pkl', '.pickle']:
                with open(filepath, 'rb') as f:
                    import_data = pickle.load(f)
            else:
                raise ValueError(f"Unsupported file format: {filepath.suffix}")
            
            field_id = import_data['field_id']
            
            # Import snapshots
            with self._lock:
                self._field_history[field_id].clear()
                
                for snapshot_dict in import_data['snapshots']:
                    # Reconstruct numpy array
                    field_values = np.array(snapshot_dict['field_values'], 
                                          dtype=snapshot_dict['field_values_dtype'])
                    
                    # Reconstruct OSH metrics
                    osh_metrics = None
                    if snapshot_dict['osh_metrics']:
                        osh_metrics = OSHMetrics()
                        for key, value in snapshot_dict['osh_metrics'].items():
                            setattr(osh_metrics, key, value)
                    
                    snapshot = EvolutionSnapshot(
                        field_id=snapshot_dict['field_id'],
                        time_point=snapshot_dict['time_point'],
                        field_values=field_values,
                        osh_metrics=osh_metrics,
                        metadata=snapshot_dict['metadata'],
                        gradient_magnitude=snapshot_dict['gradient_magnitude'],
                        laplacian_trace=snapshot_dict['laplacian_trace'],
                        energy_density=snapshot_dict['energy_density'],
                        evolution_index=snapshot_dict['evolution_index'],
                        timestamp=snapshot_dict['timestamp']
                    )
                    
                    self._field_history[field_id].append(snapshot)
                
                # Import change detection results
                self._change_detection_results[field_id].clear()
                for change_dict in import_data['change_detection_results']:
                    change = ChangeDetectionResult(
                        field_id=change_dict['field_id'],
                        change_type=ChangeType(change_dict['change_type']),
                        time_detected=change_dict['time_detected'],
                        confidence=change_dict['confidence'],
                        magnitude=change_dict['magnitude'],
                        duration=change_dict['duration'],
                        description=change_dict['description']
                    )
                    self._change_detection_results[field_id].append(change)
                
                # Import trend analysis
                if import_data['trend_analysis']:
                    trend_dict = import_data['trend_analysis']
                    trend = TrendAnalysis(
                        field_id=field_id,
                        analysis_time=time.time(),
                        trend_direction=TrendDirection(trend_dict['trend_direction']),
                        trend_strength=trend_dict['trend_strength'],
                        confidence=trend_dict['confidence'],
                        statistics=trend_dict['statistics'],
                        dominant_frequencies=trend_dict['dominant_frequencies'],
                        prediction=trend_dict['prediction']
                    )
                    self._trend_analyses[field_id] = trend
            
            self.logger.info(f"Imported field history for {field_id} from {filename}")
            return field_id
            
        except Exception as e:
            self.logger.error(f"Import failed from {filename}: {e}")
            global_error_manager.runtime_error(str(e))
            raise

    def clear_history(self, field_id: Optional[str] = None):
        """Clear history for specific field or all fields."""
        try:
            with self._lock:
                if field_id:
                    if field_id in self._field_history:
                        self._field_history[field_id].clear()
                        self._delta_records[field_id].clear()
                        self._change_detection_results[field_id].clear()
                        if field_id in self._trend_analyses:
                            del self._trend_analyses[field_id]
                        if field_id in self._full_snapshots:
                            self._full_snapshots[field_id].clear()
                        self.logger.info(f"Cleared history for field {field_id}")
                else:
                    self._field_history.clear()
                    self._delta_records.clear()
                    self._change_detection_results.clear()
                    self._trend_analyses.clear()
                    self._full_snapshots.clear()
                    self._tracking_stats['total_snapshots'] = 0
                    self._tracking_stats['total_fields'] = 0
                    self.logger.info("Cleared all field histories")
                    
        except Exception as e:
            self.logger.error(f"Clear history failed: {e}")
            global_error_manager.runtime_error(str(e))

    def cleanup(self):
        """Cleanup tracker resources."""
        try:
            self._shutdown = True
            self._executor.shutdown(wait=True)
            self.clear_history()
            self.logger.info("FieldEvolutionTracker cleaned up successfully")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def _calculate_osh_metrics(self, field_values: np.ndarray) -> OSHMetrics:
        """Calculate OSH metrics for field state."""
        try:
            # Basic OSH metrics calculation
            # In a real implementation, this would integrate with the OSH metrics system
            
            # Coherence: measure of phase correlation
            if np.iscomplexobj(field_values):
                phases = np.angle(field_values)
                coherence = float(np.abs(np.mean(np.exp(1j * phases))))
            else:
                coherence = 1.0 - float(np.std(field_values) / (np.mean(np.abs(field_values)) + 1e-10))
            
            # Entropy: Shannon entropy approximation
            if field_values.size > 0:
                probs = np.abs(field_values.flatten())**2
                probs = probs / (np.sum(probs) + 1e-10)
                entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
                entropy = entropy / np.log(len(probs))  # Normalize
            else:
                entropy = 0.0
            
            # Strain: based on gradient magnitude
            try:
                gradients = np.gradient(field_values)
                if isinstance(gradients, list):
                    grad_magnitude = np.sqrt(sum(g**2 for g in gradients))
                else:
                    grad_magnitude = np.abs(gradients)
                strain = float(np.mean(grad_magnitude))
                strain = min(strain, 1.0)  # Normalize
            except:
                strain = 0.0
            
            # RSP: Recursive Simulation Potential
            rsp = coherence * (1.0 - entropy) / (strain + 0.01)
            
            # Create OSH metrics object
            osh_metrics = OSHMetrics()
            osh_metrics.coherence = max(0.0, min(1.0, coherence))
            osh_metrics.entropy = max(0.0, min(1.0, entropy))
            osh_metrics.strain = max(0.0, min(1.0, strain))
            osh_metrics.rsp = max(0.0, rsp)
            osh_metrics.phi = coherence * np.log(field_values.size + 1)  # Integrated information approximation
            osh_metrics.temporal_stability = 1.0 - strain
            osh_metrics.emergence_index = coherence * (1.0 - entropy) * np.sqrt(field_values.size)
            
            return osh_metrics
            
        except Exception as e:
            self.logger.warning(f"OSH metrics calculation failed: {e}")
            # Return default metrics
            osh_metrics = OSHMetrics()
            return osh_metrics

    def _perform_background_analysis(self, field_id: str):
        """Perform background change detection and trend analysis."""
        if self._shutdown:
            return
            
        try:
            with self._lock:
                history = list(self._field_history.get(field_id, deque()))
            
            if len(history) < 3:
                return
            
            # Change detection
            if self.config['change_detection_enabled']:
                try:
                    new_changes = self.change_detector.detect_changes(history[-10:])  # Analyze last 10 snapshots
                    with self._lock:
                        self._change_detection_results[field_id].extend(new_changes)
                        # Keep only recent changes
                        self._change_detection_results[field_id] = self._change_detection_results[field_id][-100:]
                except Exception as e:
                    self.logger.warning(f"Background change detection failed for {field_id}: {e}")
            
            # Trend analysis (less frequent)
            if (self.config['trend_analysis_enabled'] and 
                len(history) % 5 == 0):  # Every 5th snapshot
                try:
                    self.analyze_field_trends(field_id)
                except Exception as e:
                    self.logger.warning(f"Background trend analysis failed for {field_id}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Background analysis failed for {field_id}: {e}")

    def __del__(self):
        """Ensure cleanup on destruction."""
        try:
            self.cleanup()
        except:
            pass


# Global instance management
_global_field_evolution_tracker: Optional[FieldEvolutionTracker] = None
_tracker_lock = threading.Lock()


def get_field_evolution_tracker() -> FieldEvolutionTracker:
    """Get the global field evolution tracker instance."""
    global _global_field_evolution_tracker
    
    with _tracker_lock:
        if _global_field_evolution_tracker is None:
            _global_field_evolution_tracker = FieldEvolutionTracker()
        return _global_field_evolution_tracker


def set_field_evolution_tracker(tracker: FieldEvolutionTracker):
    """Set the global field evolution tracker instance."""
    global _global_field_evolution_tracker
    
    with _tracker_lock:
        if _global_field_evolution_tracker:
            _global_field_evolution_tracker.cleanup()
        _global_field_evolution_tracker = tracker


# Convenience functions for global access
def record_field_evolution(field_id: str, field_values: np.ndarray, 
                          time_point: float, **metadata) -> bool:
    """Record field evolution using global tracker."""
    return get_field_evolution_tracker().record_field_state(field_id, field_values, time_point, metadata)


def get_field_at_time(field_id: str, time_point: float) -> Optional[np.ndarray]:
    """Get field at specific time using global tracker."""
    return get_field_evolution_tracker().get_field_at_time(field_id, time_point)


def analyze_field_changes(field_id: str, threshold: float = 0.1) -> List[Tuple[float, ChangeType]]:
    """Analyze field changes using global tracker."""
    return get_field_evolution_tracker().find_significant_changes(field_id, threshold)


def get_field_evolution_statistics(field_id: str) -> Dict[str, Any]:
    """Get field evolution statistics using global tracker."""
    return get_field_evolution_tracker().get_field_statistics(field_id)


def cleanup_field_evolution_tracker():
    """Cleanup global tracker."""
    global _global_field_evolution_tracker
    
    with _tracker_lock:
        if _global_field_evolution_tracker:
            _global_field_evolution_tracker.cleanup()
            _global_field_evolution_tracker = None


# Module-level logging setup
logging.getLogger(__name__).setLevel(logging.INFO)