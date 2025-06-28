"""
Recursia Field Evolution & Scientific Delta Tracking Engine

This module provides comprehensive temporal field evolution with advanced tracking,
compression, change detection, and interpolation for quantum fields. It serves as
the scientific backbone for OSH validation and empirical research.

Key Features:
- Multi-solver field evolution with adaptive time stepping
- Real-time OSH metrics calculation and validation
- Advanced change detection and phenomena identification
- Compression and delta tracking for scientific reproducibility
- Stability analysis and predictive modeling
- Integration with all core Recursia subsystems
- Enterprise-grade performance monitoring and logging

Author: Johnie Waddell
Version: 1.0.0
"""

import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.stats import entropy as scipy_entropy

# Core Recursia imports
from src.core.data_classes import (
    ChangeDetectionMode, ChangeDetectionResult, ChangeType, CompressionMethod, DeltaRecord, EvolutionConfiguration,
    EvolutionParameters, EvolutionResult, EvolutionSnapshot, EvolutionStatus, OSHMetrics,
    TrendAnalysis, TrendDirection
)

# Physics imports
from src.physics.field.field_compute import get_compute_engine
from src.physics.coherence import CoherenceManager
from src.physics.memory_field import MemoryFieldPhysics
from src.physics.recursive import RecursiveMechanics
from src.physics.physics_event_system import PhysicsEventSystem

# Utilities
from src.core.utils import global_error_manager, performance_profiler
from src.visualization.render_utils import get_comprehensive_metrics_summary

class FieldChangeDetector:
    """Advanced change detection system for field evolution."""
    
    def __init__(self, config: EvolutionConfiguration):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.ChangeDetector")
        self._lock = threading.RLock()
        
        # Detection thresholds based on mode
        self._setup_thresholds()
        
        # History for trend analysis
        self.detection_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
    def _setup_thresholds(self):
        """Configure detection thresholds based on sensitivity mode."""
        base_thresholds = self.config.change_thresholds.copy()
        
        if self.config.change_detection_mode == ChangeDetectionMode.CONSERVATIVE:
            self.thresholds = {k: v * 1.5 for k, v in base_thresholds.items()}
        elif self.config.change_detection_mode == ChangeDetectionMode.SENSITIVE:
            self.thresholds = {k: v * 0.7 for k, v in base_thresholds.items()}
        elif self.config.change_detection_mode == ChangeDetectionMode.HYPERSENSITIVE:
            self.thresholds = {k: v * 0.4 for k, v in base_thresholds.items()}
        else:  # BALANCED
            self.thresholds = base_thresholds
            
        self.logger.debug(f"Detection thresholds set: {self.thresholds}")
    
    def detect_changes(self, field_id: str, current_metrics: OSHMetrics,
                      previous_metrics: Optional[OSHMetrics],
                      field_data: np.ndarray,
                      previous_field: Optional[np.ndarray] = None) -> List[ChangeDetectionResult]:
        """Comprehensive change detection across multiple dimensions."""
        with self._lock:
            changes = []
            current_time = time.time()
            
            if previous_metrics is None:
                self.logger.debug(f"No previous metrics for field {field_id}, skipping detection")
                return changes
            
            # OSH metric changes
            changes.extend(self._detect_osh_changes(
                field_id, current_metrics, previous_metrics, current_time
            ))
            
            # Field structure changes
            if previous_field is not None:
                changes.extend(self._detect_structural_changes(
                    field_id, field_data, previous_field, current_time
                ))
            
            # Pattern changes
            changes.extend(self._detect_pattern_changes(
                field_id, current_metrics, current_time
            ))
            
            # Store detection history
            self.detection_history[field_id].append({
                'time': current_time,
                'changes': changes,
                'metrics': current_metrics
            })
            
            self.logger.debug(f"Detected {len(changes)} changes for field {field_id}")
            return changes
    
    def _detect_osh_changes(self, field_id: str, current: OSHMetrics,
                           previous: OSHMetrics, timestamp: float) -> List[ChangeDetectionResult]:
        """Detect OSH metric-based changes."""
        changes = []
        
        # Coherence changes
        coherence_delta = abs(current.coherence - previous.coherence)
        if coherence_delta > self.thresholds["coherence"]:
            change_type = ChangeType.COHERENCE_COLLAPSE if (
                current.coherence < previous.coherence and coherence_delta > 0.3
            ) else ChangeType.SUDDEN
            
            changes.append(ChangeDetectionResult(
                field_id=field_id,
                change_type=change_type,
                time_detected=timestamp,
                confidence=min(1.0, coherence_delta / self.thresholds["coherence"]),
                magnitude=coherence_delta,
                description=f"Coherence change: {previous.coherence:.3f} → {current.coherence:.3f}",
                metadata={
                    'metric': 'coherence',
                    'delta': coherence_delta,
                    'direction': 'decrease' if current.coherence < previous.coherence else 'increase'
                }
            ))
        
        # Entropy changes
        entropy_delta = abs(current.entropy - previous.entropy)
        if entropy_delta > self.thresholds["entropy"]:
            change_type = ChangeType.ENTROPY_CASCADE if (
                current.entropy > previous.entropy and entropy_delta > 0.2
            ) else ChangeType.SUDDEN
            
            changes.append(ChangeDetectionResult(
                field_id=field_id,
                change_type=change_type,
                time_detected=timestamp,
                confidence=min(1.0, entropy_delta / self.thresholds["entropy"]),
                magnitude=entropy_delta,
                description=f"Entropy change: {previous.entropy:.3f} → {current.entropy:.3f}",
                metadata={
                    'metric': 'entropy',
                    'delta': entropy_delta,
                    'direction': 'increase' if current.entropy > previous.entropy else 'decrease'
                }
            ))
        
        # RSP instability
        if hasattr(current, 'rsp') and hasattr(previous, 'rsp'):
            rsp_delta = abs(current.rsp - previous.rsp)
            if rsp_delta > self.thresholds["rsp"]:
                changes.append(ChangeDetectionResult(
                    field_id=field_id,
                    change_type=ChangeType.RSP_INSTABILITY,
                    time_detected=timestamp,
                    confidence=min(1.0, rsp_delta / self.thresholds["rsp"]),
                    magnitude=rsp_delta,
                    description=f"RSP instability: {previous.rsp:.3f} → {current.rsp:.3f}",
                    metadata={
                        'metric': 'rsp',
                        'delta': rsp_delta,
                        'criticality': 'high' if rsp_delta > 0.5 else 'medium'
                    }
                ))
        
        return changes
    
    def _detect_structural_changes(self, field_id: str, current_field: np.ndarray,
                                 previous_field: np.ndarray, timestamp: float) -> List[ChangeDetectionResult]:
        """Detect structural changes in field data."""
        changes = []
        
        try:
            # Field magnitude changes
            current_magnitude = np.linalg.norm(current_field)
            previous_magnitude = np.linalg.norm(previous_field)
            magnitude_change = abs(current_magnitude - previous_magnitude) / max(previous_magnitude, 1e-10)
            
            if magnitude_change > self.thresholds["energy"]:
                changes.append(ChangeDetectionResult(
                    field_id=field_id,
                    change_type=ChangeType.ENERGY_SPIKE if magnitude_change > 0.5 else ChangeType.SUDDEN,
                    time_detected=timestamp,
                    confidence=min(1.0, magnitude_change / self.thresholds["energy"]),
                    magnitude=magnitude_change,
                    description=f"Field magnitude change: {magnitude_change:.1%}",
                    metadata={
                        'metric': 'field_magnitude',
                        'current_magnitude': current_magnitude,
                        'previous_magnitude': previous_magnitude,
                        'relative_change': magnitude_change
                    }
                ))
            
            # Gradient inversion detection
            if current_field.ndim > 1:
                current_grad = np.gradient(current_field)
                previous_grad = np.gradient(previous_field)
                
                if isinstance(current_grad, list):
                    grad_correlation = np.corrcoef(
                        np.concatenate([g.flatten() for g in current_grad]),
                        np.concatenate([g.flatten() for g in previous_grad])
                    )[0, 1]
                else:
                    grad_correlation = np.corrcoef(current_grad.flatten(), previous_grad.flatten())[0, 1]
                
                if not np.isnan(grad_correlation) and grad_correlation < -0.5:
                    changes.append(ChangeDetectionResult(
                        field_id=field_id,
                        change_type=ChangeType.GRADIENT_INVERSION,
                        time_detected=timestamp,
                        confidence=min(1.0, abs(grad_correlation)),
                        magnitude=abs(grad_correlation),
                        description=f"Gradient inversion detected (correlation: {grad_correlation:.3f})",
                        metadata={
                            'metric': 'gradient_correlation',
                            'correlation': grad_correlation
                        }
                    ))
            
        except Exception as e:
            self.logger.warning(f"Error in structural change detection for {field_id}: {e}")
        
        return changes
    
    def _detect_pattern_changes(self, field_id: str, current_metrics: OSHMetrics,
                              timestamp: float) -> List[ChangeDetectionResult]:
        """Detect pattern-based changes using historical data."""
        changes = []
        
        if field_id not in self.detection_history or len(self.detection_history[field_id]) < 5:
            return changes
        
        # Oscillation detection
        coherence_history = [
            entry['metrics'].coherence 
            for entry in list(self.detection_history[field_id])[-10:]
        ]
        
        if len(coherence_history) >= 6:
            # Simple oscillation detection via autocorrelation
            try:
                autocorr = np.correlate(coherence_history, coherence_history, mode='full')
                autocorr = autocorr[autocorr.size // 2:]
                autocorr = autocorr / autocorr[0]  # Normalize
                
                # Look for periodic peaks
                peaks, _ = signal.find_peaks(autocorr[1:], height=0.3, distance=2)
                
                if len(peaks) > 0:
                    changes.append(ChangeDetectionResult(
                        field_id=field_id,
                        change_type=ChangeType.OSCILLATORY,
                        time_detected=timestamp,
                        confidence=0.7,  # Medium confidence for pattern detection
                        magnitude=float(np.max(autocorr[peaks + 1])),
                        description=f"Oscillatory pattern detected (period ≈ {peaks[0] + 1} steps)",
                        metadata={
                            'pattern': 'oscillatory',
                            'period_estimate': int(peaks[0] + 1),
                            'peak_correlation': float(np.max(autocorr[peaks + 1]))
                        }
                    ))
            except Exception as e:
                self.logger.debug(f"Pattern analysis failed for {field_id}: {e}")
        
        return changes


class CompressionEngine:
    """Advanced field data compression for efficient storage and analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.CompressionEngine")
        self.compression_stats = {
            'total_compressed': 0,
            'total_size_before': 0,
            'total_size_after': 0,
            'method_usage': defaultdict(int)
        }
    
    def compress_delta(self, current_field: np.ndarray, previous_field: np.ndarray,
                      method: CompressionMethod = CompressionMethod.ADAPTIVE) -> DeltaRecord:
        """Compress field difference using specified method."""
        try:
            if method == CompressionMethod.ADAPTIVE:
                method = self._select_optimal_method(current_field, previous_field)
            
            delta_data, compression_ratio = self._apply_compression(
                current_field, previous_field, method
            )
            
            # Update statistics
            self.compression_stats['total_compressed'] += 1
            self.compression_stats['total_size_before'] += current_field.nbytes
            self.compression_stats['total_size_after'] += len(delta_data)
            self.compression_stats['method_usage'][method] += 1
            
            return DeltaRecord(
                field_id="",  # Will be set by caller
                time_from=0.0,  # Will be set by caller
                time_to=0.0,    # Will be set by caller
                compression_method=method,
                compressed_data=delta_data,
                compression_ratio=compression_ratio,
                base_field_shape=current_field.shape,
                metadata={
                    'compression_stats': {
                        'original_size': current_field.nbytes,
                        'compressed_size': len(delta_data),
                        'method_used': method.value
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Compression failed: {e}")
            # Fallback to simple delta
            delta = current_field - previous_field
            delta_bytes = delta.tobytes()
            
            return DeltaRecord(
                field_id="",
                time_from=0.0,
                time_to=0.0,
                compression_method=CompressionMethod.DELTA,
                compressed_data=delta_bytes,
                compression_ratio=1.0,
                base_field_shape=current_field.shape
            )
    
    def _select_optimal_method(self, current_field: np.ndarray, 
                              previous_field: np.ndarray) -> CompressionMethod:
        """Select optimal compression method based on field characteristics."""
        try:
            # Calculate field characteristics
            delta = current_field - previous_field
            sparsity = np.count_nonzero(np.abs(delta) < 1e-10) / delta.size
            max_change = np.max(np.abs(delta))
            mean_change = np.mean(np.abs(delta))
            
            # Decision logic
            if sparsity > 0.8:
                return CompressionMethod.SPARSE
            elif max_change / (mean_change + 1e-10) > 10:
                return CompressionMethod.GZIP
            elif delta.size > 1000:
                return CompressionMethod.WAVELET
            else:
                return CompressionMethod.DELTA
                
        except Exception:
            return CompressionMethod.DELTA
    
    def _apply_compression(self, current_field: np.ndarray, previous_field: np.ndarray,
                          method: CompressionMethod) -> Tuple[bytes, float]:
        """Apply specific compression method."""
        delta = current_field - previous_field
        original_size = delta.nbytes
        
        if method == CompressionMethod.DELTA:
            compressed_data = delta.tobytes()
            
        elif method == CompressionMethod.SPARSE:
            # Sparse compression - store only significant changes
            threshold = np.std(delta) * 0.1
            mask = np.abs(delta) > threshold
            indices = np.where(mask)
            values = delta[mask]
            
            # Store as compressed format
            import pickle
            sparse_data = {
                'indices': indices,
                'values': values,
                'shape': delta.shape,
                'threshold': threshold
            }
            compressed_data = pickle.dumps(sparse_data)
            
        elif method == CompressionMethod.WAVELET:
            # Wavelet compression using FFT approximation
            try:
                fft_data = fft(delta.flatten())
                # Keep only the most significant coefficients
                magnitude = np.abs(fft_data)
                threshold = np.percentile(magnitude, 90)
                fft_data[magnitude < threshold] = 0
                
                import pickle
                compressed_data = pickle.dumps({
                    'fft_data': fft_data,
                    'shape': delta.shape
                })
            except Exception:
                compressed_data = delta.tobytes()
                
        elif method == CompressionMethod.GZIP:
            import gzip
            compressed_data = gzip.compress(delta.tobytes())
            
        else:
            compressed_data = delta.tobytes()
        
        compression_ratio = len(compressed_data) / original_size
        return compressed_data, compression_ratio
    
    def decompress_delta(self, delta_record: DeltaRecord, base_field: np.ndarray) -> np.ndarray:
        """Decompress delta and apply to base field."""
        try:
            method = delta_record.compression_method
            
            if method == CompressionMethod.DELTA:
                delta = np.frombuffer(
                    delta_record.compressed_data, 
                    dtype=base_field.dtype
                ).reshape(delta_record.base_field_shape)
                
            elif method == CompressionMethod.SPARSE:
                import pickle
                sparse_data = pickle.loads(delta_record.compressed_data)
                delta = np.zeros(sparse_data['shape'], dtype=base_field.dtype)
                delta[sparse_data['indices']] = sparse_data['values']
                
            elif method == CompressionMethod.WAVELET:
                import pickle
                data = pickle.loads(delta_record.compressed_data)
                fft_data = data['fft_data']
                delta = np.fft.ifft(fft_data).real.reshape(data['shape'])
                delta = delta.astype(base_field.dtype)
                
            elif method == CompressionMethod.GZIP:
                import gzip
                decompressed = gzip.decompress(delta_record.compressed_data)
                delta = np.frombuffer(decompressed, dtype=base_field.dtype).reshape(
                    delta_record.base_field_shape
                )
                
            else:
                delta = np.frombuffer(
                    delta_record.compressed_data,
                    dtype=base_field.dtype
                ).reshape(delta_record.base_field_shape)
            
            return base_field + delta
            
        except Exception as e:
            self.logger.error(f"Decompression failed: {e}")
            return base_field
    
    def get_compression_statistics(self) -> Dict[str, Any]:
        """Get compression performance statistics."""
        stats = self.compression_stats.copy()
        if stats['total_size_before'] > 0:
            stats['overall_compression_ratio'] = (
                stats['total_size_after'] / stats['total_size_before']
            )
        else:
            stats['overall_compression_ratio'] = 1.0
        
        return stats


class FieldEvolutionEngine:
    """
    Enterprise-grade field evolution engine with comprehensive OSH integration.
    
    This class serves as the central orchestrator for temporal field evolution,
    providing scientific-grade analytics, validation, and monitoring capabilities
    essential for OSH research and validation.
    """
    
    def __init__(self, config: Optional[EvolutionConfiguration] = None, runtime=None):
        self.config = config or EvolutionConfiguration()
        self.logger = self._setup_logging()
        self.runtime = runtime  # Access to VM execution context
        
        # Core components
        self.compute_engine = get_compute_engine()
        # Remove OSHMetricsCalculator - use VM metrics instead
        self.change_detector = FieldChangeDetector(self.config)
        self.compression_engine = CompressionEngine()
        
        # Evolution state
        self.evolution_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config.max_snapshots)
        )
        self.change_log: Dict[str, List[ChangeDetectionResult]] = defaultdict(list)
        self.performance_log: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Thread management
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="FieldEvolution"
        ) if self.config.enable_parallel_processing else None
        
        # Status tracking
        self.status = EvolutionStatus.INITIALIZED
        self._active_evolutions: Set[str] = set()
        
        # Performance monitoring
        self._evolution_stats = {
            'total_evolutions': 0,
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'total_time': 0.0,
            'snapshots_created': 0,
            'changes_detected': 0
        }
        
        self.logger.info("FieldEvolutionEngine initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Configure comprehensive logging system."""
        logger = logging.getLogger(f"{__name__}.FieldEvolutionEngine")
        
        if self.config.detailed_logging:
            logger.setLevel(getattr(logging, self.config.log_level, logging.INFO))
            
            # Add performance logging
            perf_logger = logging.getLogger(f"{__name__}.Performance")
            perf_logger.setLevel(logging.DEBUG)
        
        return logger
    
    def _get_current_metrics(self) -> OSHMetrics:
        """
        Get current OSH metrics from VM execution context.
        This ensures all metrics come from the unified VM calculations.
        """
        if self.runtime and hasattr(self.runtime, 'execution_context'):
            if hasattr(self.runtime.execution_context, 'current_metrics'):
                return self.runtime.execution_context.current_metrics
        
        # Fallback: return default metrics if no runtime context
        return OSHMetrics(
            timestamp=time.time(),
            coherence=0.95,
            entropy=0.05,
            strain=0.0,
            field_energy=0.0,
            rsp=0.0,
            phi=0.0,
            emergence_index=0.0,
            consciousness_quotient=0.0
        )
    
    def evolve_field(self, field_values: np.ndarray, field_type: str,
                    parameters: Optional[EvolutionParameters] = None,
                    field_id: Optional[str] = None,
                    steps: int = 1) -> EvolutionResult:
        """
        Evolve a field forward in time with comprehensive monitoring and analysis.
        
        Args:
            field_values: Current field state
            field_type: Type of field evolution (wave_equation, schrodinger_equation, etc.)
            parameters: Evolution parameters
            field_id: Unique identifier for field tracking
            steps: Number of evolution steps
            
        Returns:
            Comprehensive evolution result with OSH metrics and analysis
        """
        if field_id is None:
            field_id = f"field_{id(field_values)}"
            
        start_time = time.time()
        self.logger.info(f"Starting evolution for field {field_id} ({steps} steps)")
        
        with self._lock:
            self._active_evolutions.add(field_id)
            self.status = EvolutionStatus.RUNNING
        
        try:
            # Initialize result
            result = EvolutionResult(
                success=False,
                final_time=0.0,
                total_steps=0,
                actual_duration=0.0,
                final_field_values={field_id: field_values.copy()},
                field_metadata={field_id: {'type': field_type}},
                final_coherence={},
                final_entropy={},
                final_strain={},
                final_energy={}
            )
            
            # Evolution parameters
            if parameters is None:
                parameters = EvolutionParameters()
            
            if not parameters.validate():
                raise ValueError("Invalid evolution parameters")
            
            # Pre-evolution snapshot
            initial_metrics = self._get_current_metrics()
            self.save_snapshot(field_id, field_values, initial_metrics)
            
            # Evolution loop
            current_field = field_values.copy()
            previous_field = None
            previous_metrics = None
            evolution_statistics = {}
            
            for step in range(steps):
                step_start = time.time()
                
                try:
                    # Store previous state
                    previous_field = current_field.copy()
                    previous_metrics = self._get_current_metrics() if step > 0 else initial_metrics
                    
                    # Evolve field
                    current_field = self.compute_engine.evolve_field(
                        current_field, field_type, dt=parameters.time_step
                    )
                    
                    # Calculate metrics - get from VM execution context
                    current_metrics = self._get_current_metrics()
                    
                    # Change detection
                    if step > 0:
                        changes = self.change_detector.detect_changes(
                            field_id, current_metrics, previous_metrics,
                            current_field, previous_field
                        )
                        
                        if changes:
                            self.change_log[field_id].extend(changes)
                            self._evolution_stats['changes_detected'] += len(changes)
                            
                            self.logger.debug(f"Step {step}: Detected {len(changes)} changes")
                    
                    # Periodic snapshot
                    if step % self.config.snapshot_interval == 0:
                        self.save_snapshot(field_id, current_field, current_metrics)
                    
                    # Update statistics
                    step_time = time.time() - step_start
                    evolution_statistics[f'step_{step}'] = {
                        'time': step_time,
                        'l2_change': float(np.linalg.norm(current_field - previous_field)) if previous_field is not None else 0.0,
                        'coherence': current_metrics.coherence,
                        'entropy': current_metrics.entropy,
                        'energy': current_metrics.field_energy
                    }
                    
                    # Progress logging
                    if step % 10 == 0 or step == steps - 1:
                        self.logger.debug(
                            f"Step {step}/{steps}: coherence={current_metrics.coherence:.3f}, "
                            f"entropy={current_metrics.entropy:.3f}, "
                            f"time={step_time:.4f}s"
                        )
                    
                    result.total_steps = step + 1
                    result.final_time += parameters.time_step
                    
                except Exception as e:
                    self.logger.error(f"Error in evolution step {step}: {e}")
                    result.errors.append(f"Step {step}: {str(e)}")
                    
                    if len(result.errors) > 5:  # Too many errors
                        self.logger.error(f"Too many errors in evolution, stopping")
                        break
            
            # Final metrics and result assembly
            final_metrics = self._get_current_metrics()
            
            result.success = len(result.errors) == 0
            result.actual_duration = time.time() - start_time
            result.final_field_values[field_id] = current_field
            result.final_coherence[field_id] = final_metrics.coherence
            result.final_entropy[field_id] = final_metrics.entropy
            result.final_strain[field_id] = final_metrics.strain
            result.final_energy[field_id] = final_metrics.field_energy
            
            # Performance metrics
            result.average_step_time = sum(
                stats['time'] for stats in evolution_statistics.values()
            ) / max(1, len(evolution_statistics))
            result.total_computation_time = result.actual_duration
            
            # Update global statistics
            self._evolution_stats['total_evolutions'] += 1
            if result.success:
                self._evolution_stats['successful_evolutions'] += 1
            else:
                self._evolution_stats['failed_evolutions'] += 1
            self._evolution_stats['total_time'] += result.actual_duration
            
            self.logger.info(
                f"Evolution completed for field {field_id}: "
                f"success={result.success}, steps={result.total_steps}, "
                f"time={result.actual_duration:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Critical error in field evolution: {e}")
            global_error_manager.error(f"field_evolution", 0, 0, str(e))
            
            # Return error result
            result = EvolutionResult(
                success=False,
                final_time=0.0,
                total_steps=0,
                actual_duration=time.time() - start_time,
                final_field_values={field_id: field_values},
                field_metadata={field_id: {'type': field_type}},
                final_coherence={},
                final_entropy={},
                final_strain={},
                final_energy={},
                errors=[str(e)]
            )
            
            self._evolution_stats['failed_evolutions'] += 1
            return result
            
        finally:
            with self._lock:
                self._active_evolutions.discard(field_id)
                if not self._active_evolutions:
                    self.status = EvolutionStatus.COMPLETED
    
    def save_snapshot(self, field_id: str, field_values: np.ndarray,
                     osh_metrics: OSHMetrics, metadata: Optional[Dict[str, Any]] = None):
        """Save a comprehensive field snapshot with OSH metrics."""
        try:
            snapshot = EvolutionSnapshot(
                field_id=field_id,
                time_point=time.time(),
                field_values=field_values.copy(),
                osh_metrics=osh_metrics,
                metadata=metadata or {},
                evolution_index=len(self.evolution_history[field_id])
            )
            
            with self._lock:
                self.evolution_history[field_id].append(snapshot)
                self._evolution_stats['snapshots_created'] += 1
            
            self.logger.debug(f"Saved snapshot for field {field_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save snapshot for field {field_id}: {e}")
    
    def get_snapshot_summary(self, field_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of field evolution snapshots."""
        with self._lock:
            if field_id not in self.evolution_history:
                return {'error': 'Field not found'}
            
            snapshots = list(self.evolution_history[field_id])
            
            if not snapshots:
                return {'field_id': field_id, 'snapshot_count': 0}
            
            # Calculate summary statistics
            coherence_values = [s.osh_metrics.coherence for s in snapshots]
            entropy_values = [s.osh_metrics.entropy for s in snapshots]
            strain_values = [s.osh_metrics.strain for s in snapshots]
            energy_values = [s.osh_metrics.field_energy for s in snapshots]
            rsp_values = [s.osh_metrics.rsp for s in snapshots]
            
            summary = {
                'field_id': field_id,
                'snapshot_count': len(snapshots),
                'time_span': snapshots[-1].time_point - snapshots[0].time_point,
                'mean_coherence': np.mean(coherence_values),
                'variance_coherence': np.var(coherence_values),
                'mean_entropy': np.mean(entropy_values),
                'variance_entropy': np.var(entropy_values),
                'mean_strain': np.mean(strain_values),
                'variance_strain': np.var(strain_values),
                'mean_energy': np.mean(energy_values),
                'variance_energy': np.var(energy_values),
                'rsp_avg': np.mean(rsp_values),
                'rsp_max': np.max(rsp_values),
                'rsp_min': np.min(rsp_values),
                'changes_detected': len(self.change_log.get(field_id, [])),
                'last_snapshot_time': snapshots[-1].time_point
            }
            
            return summary
    
    def interpolate_field_state(self, field_id: str, target_time: float) -> Optional[np.ndarray]:
        """Interpolate field state at arbitrary time point."""
        with self._lock:
            if field_id not in self.evolution_history:
                self.logger.warning(f"No history for field {field_id}")
                return None
            
            snapshots = list(self.evolution_history[field_id])
            
            if len(snapshots) < 2:
                self.logger.warning(f"Insufficient snapshots for interpolation")
                return snapshots[0].field_values if snapshots else None
            
            # Find surrounding snapshots
            times = [s.time_point for s in snapshots]
            
            if target_time <= times[0]:
                return snapshots[0].field_values
            elif target_time >= times[-1]:
                return snapshots[-1].field_values
            
            # Find interpolation points
            for i in range(len(times) - 1):
                if times[i] <= target_time <= times[i + 1]:
                    # Linear interpolation
                    t0, t1 = times[i], times[i + 1]
                    f0, f1 = snapshots[i].field_values, snapshots[i + 1].field_values
                    
                    alpha = (target_time - t0) / (t1 - t0)
                    interpolated = (1 - alpha) * f0 + alpha * f1
                    
                    self.logger.debug(f"Interpolated field {field_id} at time {target_time}")
                    return interpolated
            
            return None
    
    def analyze_field_evolution(self, field_id: str) -> Optional[TrendAnalysis]:
        """Comprehensive trend analysis of field evolution."""
        with self._lock:
            if field_id not in self.evolution_history:
                return None
            
            snapshots = list(self.evolution_history[field_id])
            
            if len(snapshots) < 3:
                return None
            
            try:
                # Extract time series data
                times = np.array([s.time_point for s in snapshots])
                coherence = np.array([s.osh_metrics.coherence for s in snapshots])
                entropy = np.array([s.osh_metrics.entropy for s in snapshots])
                energy = np.array([s.osh_metrics.field_energy for s in snapshots])
                
                # Trend analysis
                coherence_trend = np.polyfit(times, coherence, 1)[0]
                entropy_trend = np.polyfit(times, entropy, 1)[0]
                
                # Determine overall trend direction
                if abs(coherence_trend) > abs(entropy_trend):
                    if coherence_trend > 0.01:
                        trend_direction = TrendDirection.INCREASING
                    elif coherence_trend < -0.01:
                        trend_direction = TrendDirection.DECREASING
                    else:
                        trend_direction = TrendDirection.STABLE
                else:
                    if entropy_trend > 0.01:
                        trend_direction = TrendDirection.DECREASING  # Higher entropy = degrading
                    elif entropy_trend < -0.01:
                        trend_direction = TrendDirection.INCREASING
                    else:
                        trend_direction = TrendDirection.STABLE
                
                # Calculate trend strength
                trend_strength = max(abs(coherence_trend), abs(entropy_trend))
                
                # Autocorrelation for oscillation detection
                autocorr = None
                dominant_frequencies = None
                
                if len(coherence) >= 10:
                    try:
                        # Simple autocorrelation
                        autocorr = np.correlate(coherence, coherence, mode='full')
                        autocorr = autocorr[autocorr.size // 2:]
                        autocorr = autocorr / autocorr[0]
                        
                        # FFT for frequency analysis
                        dt = np.mean(np.diff(times))
                        freqs = fftfreq(len(coherence), dt)
                        fft_vals = np.abs(fft(coherence))
                        
                        # Find dominant frequencies
                        peaks, _ = signal.find_peaks(fft_vals, height=np.max(fft_vals) * 0.1)
                        dominant_frequencies = [float(freqs[p]) for p in peaks[:5]]
                        
                        # Check for oscillatory behavior
                        if len(peaks) > 0 and np.max(autocorr[1:]) > 0.3:
                            trend_direction = TrendDirection.OSCILLATING
                            
                    except Exception as e:
                        self.logger.debug(f"Frequency analysis failed: {e}")
                
                # Statistics
                statistics = {
                    'mean_coherence': float(np.mean(coherence)),
                    'std_coherence': float(np.std(coherence)),
                    'mean_entropy': float(np.mean(entropy)),
                    'std_entropy': float(np.std(entropy)),
                    'mean_energy': float(np.mean(energy)),
                    'std_energy': float(np.std(energy)),
                    'coherence_trend_slope': float(coherence_trend),
                    'entropy_trend_slope': float(entropy_trend),
                    'correlation_coherence_entropy': float(np.corrcoef(coherence, entropy)[0, 1]) if len(coherence) > 1 else 0.0
                }
                
                # Confidence based on data quality
                confidence = min(1.0, len(snapshots) / 50.0)
                
                return TrendAnalysis(
                    field_id=field_id,
                    analysis_time=time.time(),
                    trend_direction=trend_direction,
                    trend_strength=float(trend_strength),
                    confidence=confidence,
                    statistics=statistics,
                    autocorrelation=autocorr,
                    dominant_frequencies=dominant_frequencies,
                    metadata={
                        'snapshot_count': len(snapshots),
                        'time_span': float(times[-1] - times[0]),
                        'analysis_method': 'polynomial_trend_fft'
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Trend analysis failed for field {field_id}: {e}")
                return None
    
    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution engine statistics."""
        with self._lock:
            stats = self._evolution_stats.copy()
            
            # Add derived metrics
            if stats['total_evolutions'] > 0:
                stats['success_rate'] = stats['successful_evolutions'] / stats['total_evolutions']
                stats['average_evolution_time'] = stats['total_time'] / stats['total_evolutions']
            else:
                stats['success_rate'] = 0.0
                stats['average_evolution_time'] = 0.0
            
            # Add current status
            stats['current_status'] = self.status.value
            stats['active_evolutions'] = len(self._active_evolutions)
            stats['total_fields_tracked'] = len(self.evolution_history)
            
            # Compression statistics
            stats['compression_stats'] = self.compression_engine.get_compression_statistics()
            
            # Add field-specific summaries
            field_summaries = {}
            for field_id in self.evolution_history.keys():
                field_summaries[field_id] = self.get_snapshot_summary(field_id)
            stats['field_summaries'] = field_summaries
            
            return stats
    
    def cleanup(self):
        """Clean up resources and threads."""
        self.logger.info("Cleaning up FieldEvolutionEngine")
        
        with self._lock:
            self.status = EvolutionStatus.TERMINATED
        
        if self._executor:
            self._executor.shutdown(wait=True)
            self.logger.debug("Thread pool executor shut down")
        
        # Clear large data structures
        self.evolution_history.clear()
        self.change_log.clear()
        
        self.logger.info("FieldEvolutionEngine cleanup completed")
    
    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass


# Global instance management
_global_evolution_engine: Optional[FieldEvolutionEngine] = None
_engine_lock = threading.Lock()


def get_field_evolution_engine(config: Optional[EvolutionConfiguration] = None) -> FieldEvolutionEngine:
    """Get or create global field evolution engine instance."""
    global _global_evolution_engine
    
    with _engine_lock:
        if _global_evolution_engine is None:
            _global_evolution_engine = FieldEvolutionEngine(config)
        return _global_evolution_engine


def set_field_evolution_engine(engine: FieldEvolutionEngine):
    """Set global field evolution engine instance."""
    global _global_evolution_engine
    
    with _engine_lock:
        if _global_evolution_engine:
            _global_evolution_engine.cleanup()
        _global_evolution_engine = engine


def evolve_field(field_values: np.ndarray, field_type: str,
                parameters: Optional[EvolutionParameters] = None,
                field_id: Optional[str] = None,
                steps: int = 1) -> EvolutionResult:
    """Convenience function for field evolution using global engine."""
    engine = get_field_evolution_engine()
    return engine.evolve_field(field_values, field_type, parameters, field_id, steps)


def analyze_field_trends(field_id: str) -> Optional[TrendAnalysis]:
    """Convenience function for trend analysis using global engine."""
    engine = get_field_evolution_engine()
    return engine.analyze_field_evolution(field_id)


def get_field_snapshot_summary(field_id: str) -> Dict[str, Any]:
    """Convenience function for snapshot summary using global engine."""
    engine = get_field_evolution_engine()
    return engine.get_snapshot_summary(field_id)


def cleanup_field_evolution_engine():
    """Clean up global field evolution engine."""
    global _global_evolution_engine
    
    with _engine_lock:
        if _global_evolution_engine:
            _global_evolution_engine.cleanup()
            _global_evolution_engine = None


# Ensure cleanup on module exit
import atexit
atexit.register(cleanup_field_evolution_engine)