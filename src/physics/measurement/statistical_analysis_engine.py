"""
Advanced statistical analysis engine for the Recursia quantum simulation system.
Implements comprehensive statistical analysis capabilities for OSH research including:
- Quantum measurement statistical analysis
- OSH-specific validation methods
- Time series analysis for temporal patterns
- Bayesian inference for measurement uncertainty
- Correlation analysis and pattern recognition
- Anomaly detection and outlier analysis
- Confidence interval calculations
- Hypothesis testing for OSH validation
- Multi-dimensional statistical modeling

This module provides the core statistical analysis capabilities needed for rigorous
validation of the Organic Simulation Hypothesis through quantum measurement data.
"""

import datetime
import numpy as np
import logging
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict, deque, Counter
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, wraps
import threading
import json
import hashlib
from abc import ABC, abstractmethod
import traceback
import sys

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")

try:
    from scipy import stats
    from scipy.optimize import minimize
    from scipy.signal import find_peaks, welch
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("SciPy not available - some advanced statistical features will be limited")
    SCIPY_AVAILABLE = False

from src.core.data_classes import (
    AnalysisType, DistributionAnalysis, MeasurementResult, OSHMeasurementMetrics, OSHMetrics, OSHValidationResult, StatisticalConfiguration, StatisticalResult,
    SystemHealthProfile, ComprehensiveMetrics
)

from .measurement_utils import (
    MeasurementError, performance_monitor, cached_computation
)

# Configure logging with detailed format
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('statistical_analysis.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info("INITIALIZING STATISTICAL ANALYSIS ENGINE MODULE")
logger.info("=" * 80)

# Statistical analysis constants
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_SIGNIFICANCE_LEVEL = 0.05
MIN_SAMPLE_SIZE = 3
MAX_SAMPLE_SIZE = 10000
BOOTSTRAP_ITERATIONS = 1000
MONTE_CARLO_ITERATIONS = 10000
ANOMALY_DETECTION_THRESHOLD = 2.5  # Standard deviations
CORRELATION_THRESHOLD = 0.7
OSH_VALIDATION_THRESHOLD = 0.8

# Cache configuration
ANALYSIS_CACHE_SIZE = 512
CACHE_TTL = 600  # 10 minutes

# Performance constants
MAX_CONCURRENT_ANALYSES = 4
ANALYSIS_TIMEOUT = 60.0

logger.info(f"Statistical analysis constants initialized:")
logger.info(f"  DEFAULT_CONFIDENCE_LEVEL: {DEFAULT_CONFIDENCE_LEVEL}")
logger.info(f"  DEFAULT_SIGNIFICANCE_LEVEL: {DEFAULT_SIGNIFICANCE_LEVEL}")
logger.info(f"  MIN_SAMPLE_SIZE: {MIN_SAMPLE_SIZE}")
logger.info(f"  BOOTSTRAP_ITERATIONS: {BOOTSTRAP_ITERATIONS}")
logger.info(f"  ANOMALY_DETECTION_THRESHOLD: {ANOMALY_DETECTION_THRESHOLD}")
logger.info(f"  SCIPY_AVAILABLE: {SCIPY_AVAILABLE}")

class StatisticalAnalysisError(MeasurementError):
    """Base exception for statistical analysis errors."""
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        logger.error(f"StatisticalAnalysisError raised: {message}")
        logger.debug(f"Exception args: {args}, kwargs: {kwargs}")


class InsufficientDataError(StatisticalAnalysisError):
    """Exception raised when insufficient data for analysis."""
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        logger.error(f"InsufficientDataError raised: {message}")


class InvalidParameterError(StatisticalAnalysisError):
    """Exception raised for invalid analysis parameters."""
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        logger.error(f"InvalidParameterError raised: {message}")


class AnalysisExecutionError(StatisticalAnalysisError):
    """Exception raised during analysis execution."""
    def __init__(self, message: str, *args, **kwargs):
        super().__init__(message, *args, **kwargs)
        logger.error(f"AnalysisExecutionError raised: {message}")


def log_function_entry(func):
    """Decorator to log function entry and exit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f">>> ENTERING {func_name}")
        logger.debug(f"    Args count: {len(args)}, Kwargs: {list(kwargs.keys())}")
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.debug(f"<<< EXITING {func_name} successfully in {execution_time:.4f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"<<< EXCEPTION in {func_name} after {execution_time:.4f}s: {str(e)}")
            logger.error(f"    Exception type: {type(e).__name__}")
            logger.error(f"    Traceback: {traceback.format_exc()}")
            raise
    return wrapper


class StatisticalAnalysisEngine:
    """Advanced statistical analysis engine for quantum measurements and OSH research."""
    
    @log_function_entry
    def __init__(self, config: Optional[Union[StatisticalConfiguration, Dict[str, Any]]] = None):
        logger.info("=" * 60)
        logger.info("INITIALIZING STATISTICAL ANALYSIS ENGINE")
        logger.info("=" * 60)
        
        logger.debug(f"Input config type: {type(config)}")
        logger.debug(f"Input config value: {config}")
        
        # Handle both dict and StatisticalConfiguration inputs
        if config is None:
            logger.info("No config provided, using default StatisticalConfiguration")
            self.config = StatisticalConfiguration()
        elif isinstance(config, dict):
            logger.info("Converting dict config to StatisticalConfiguration")
            logger.debug(f"Dict config keys: {list(config.keys())}")
            
            # Convert dict to StatisticalConfiguration, handling missing keys gracefully
            config_kwargs = {}
            key_mappings = {
                'confidence_level': 'confidence_level',
                'enable_cache': 'enable_caching',  # Note: different key name
                'enable_parallel': 'enable_parallel',
                'max_workers': 'max_workers',
                'significance_level': 'significance_level',
                'bootstrap_iterations': 'bootstrap_iterations',
                'monte_carlo_iterations': 'monte_carlo_iterations',
                'anomaly_threshold': 'anomaly_threshold',
                'correlation_threshold': 'correlation_threshold',
                'timeout': 'timeout'
            }
            
            for dict_key, config_key in key_mappings.items():
                if dict_key in config:
                    config_kwargs[config_key] = config[dict_key]
                    logger.debug(f"Mapped {dict_key} -> {config_key}: {config[dict_key]}")
            
            try:
                self.config = StatisticalConfiguration(**config_kwargs)
                logger.info("Successfully created StatisticalConfiguration from dict")
            except Exception as e:
                logger.warning(f"Failed to create StatisticalConfiguration from dict: {e}")
                logger.warning("Using default configuration")
                logger.debug(f"Exception traceback: {traceback.format_exc()}")
                self.config = StatisticalConfiguration()
        else:
            logger.info("Using provided StatisticalConfiguration object")
            self.config = config
        
        logger.info(f"Final configuration:")
        logger.info(f"  confidence_level: {self.config.confidence_level}")
        logger.info(f"  significance_level: {self.config.significance_level}")
        logger.info(f"  enable_caching: {self.config.enable_caching}")
        logger.info(f"  enable_parallel: {self.config.enable_parallel}")
        logger.info(f"  max_workers: {self.config.max_workers}")
        logger.info(f"  bootstrap_iterations: {self.config.bootstrap_iterations}")
        logger.info(f"  monte_carlo_iterations: {self.config.monte_carlo_iterations}")
        logger.info(f"  anomaly_threshold: {self.config.anomaly_threshold}")
        logger.info(f"  correlation_threshold: {self.config.correlation_threshold}")
        logger.info(f"  timeout: {self.config.timeout}")
        
        # Initialize cache and threading
        logger.debug("Initializing cache and threading components")
        self.analysis_cache = {}
        self.cache_lock = threading.RLock()
        logger.debug(f"Cache lock created: {self.cache_lock}")
        
        if self.config.enable_parallel:
            logger.info(f"Initializing ThreadPoolExecutor with {self.config.max_workers} workers")
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        else:
            logger.info("Parallel processing disabled, no ThreadPoolExecutor created")
            self.executor = None
        
        # Performance tracking
        logger.debug("Initializing performance tracking")
        self.analysis_stats = {
            'total_analyses': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'execution_times': deque(maxlen=1000),
            'error_count': 0,
            'analysis_type_counts': defaultdict(int),
            'last_analysis_time': None,
            'average_execution_time': 0.0
        }
        
        logger.info("StatisticalAnalysisEngine initialization complete")
        logger.info("=" * 60)
    
    def __enter__(self):
        logger.debug("StatisticalAnalysisEngine context manager entered")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f"StatisticalAnalysisEngine context manager exiting: {exc_type}")
        if exc_type:
            logger.error(f"Exception in context: {exc_type.__name__}: {exc_val}")
        self.cleanup()
    
    @log_function_entry
    def cleanup(self):
        """Cleanup resources."""
        logger.info("Starting cleanup process")
        
        if self.executor:
            logger.info("Shutting down ThreadPoolExecutor")
            self.executor.shutdown(wait=True)
            logger.debug("ThreadPoolExecutor shutdown complete")
        
        cache_size = len(self.analysis_cache)
        self.clear_cache()
        logger.info(f"Cleared analysis cache ({cache_size} items)")
        
        # Log final statistics
        logger.info("Final performance statistics:")
        stats = self.get_performance_stats()
        for key, value in stats.items():
            logger.info(f"  {key}: {value}")
        
        logger.info("StatisticalAnalysisEngine cleanup complete")
    
    @performance_monitor
    @log_function_entry
    def analyze_measurements(
        self,
        measurements: List[MeasurementResult],
        analysis_types: List[AnalysisType],
        **kwargs
    ) -> Dict[AnalysisType, StatisticalResult]:
        """
        Perform comprehensive statistical analysis on measurement data.
        
        Args:
            measurements: List of measurement results
            analysis_types: Types of analyses to perform
            **kwargs: Additional parameters for specific analyses
            
        Returns:
            Dictionary mapping analysis types to results
            
        Raises:
            InsufficientDataError: If insufficient data for analysis
            StatisticalAnalysisError: If analysis fails
        """
        logger.info("=" * 50)
        logger.info("STARTING COMPREHENSIVE MEASUREMENT ANALYSIS")
        logger.info("=" * 50)
        
        try:
            # Input validation with detailed logging
            logger.debug(f"Input validation:")
            logger.debug(f"  measurements type: {type(measurements)}")
            logger.debug(f"  measurements length: {len(measurements) if measurements else 'None'}")
            logger.debug(f"  analysis_types: {[at.value if hasattr(at, 'value') else str(at) for at in analysis_types]}")
            logger.debug(f"  kwargs keys: {list(kwargs.keys())}")
            
            if not measurements:
                logger.error("No measurements provided for analysis")
                raise InsufficientDataError("No measurements provided for analysis")
            
            if len(measurements) < MIN_SAMPLE_SIZE:
                logger.error(f"Insufficient measurements: {len(measurements)} < {MIN_SAMPLE_SIZE}")
                raise InsufficientDataError(f"Insufficient measurements: {len(measurements)} < {MIN_SAMPLE_SIZE}")
            
            logger.info(f"Processing {len(measurements)} measurements")
            logger.info(f"Requested analysis types: {len(analysis_types)}")
            
            # Log measurement data characteristics
            self._log_measurement_characteristics(measurements)
            
            start_time = time.time()
            results = {}
            successful_analyses = 0
            failed_analyses = 0
            
            # Execute analyses
            for i, analysis_type in enumerate(analysis_types):
                analysis_start_time = time.time()
                logger.info(f"[{i+1}/{len(analysis_types)}] Starting {analysis_type.value} analysis")
                
                try:
                    if analysis_type == AnalysisType.BASIC_STATISTICS:
                        logger.debug("Executing basic statistics analysis")
                        results[analysis_type] = self._analyze_basic_statistics(measurements, **kwargs)
                    elif analysis_type == AnalysisType.DISTRIBUTION_ANALYSIS:
                        logger.debug("Executing distribution analysis")
                        results[analysis_type] = self._analyze_distributions(measurements, **kwargs)
                    elif analysis_type == AnalysisType.CONFIDENCE_INTERVALS:
                        logger.debug("Executing confidence intervals analysis")
                        results[analysis_type] = self._calculate_confidence_intervals(measurements, **kwargs)
                    elif analysis_type == AnalysisType.HYPOTHESIS_TEST:
                        logger.debug("Executing hypothesis test analysis")
                        results[analysis_type] = self._perform_hypothesis_tests(measurements, **kwargs)
                    elif analysis_type == AnalysisType.CORRELATION_ANALYSIS:
                        logger.debug("Executing correlation analysis")
                        results[analysis_type] = self._analyze_correlations(measurements, **kwargs)
                    elif analysis_type == AnalysisType.TIME_SERIES_ANALYSIS:
                        logger.debug("Executing time series analysis")
                        results[analysis_type] = self._analyze_time_series(measurements, **kwargs)
                    elif analysis_type == AnalysisType.BAYESIAN_INFERENCE:
                        logger.debug("Executing Bayesian inference analysis")
                        results[analysis_type] = self._perform_bayesian_inference(measurements, **kwargs)
                    elif analysis_type == AnalysisType.ANOMALY_DETECTION:
                        logger.debug("Executing anomaly detection analysis")
                        results[analysis_type] = self._detect_anomalies(measurements, **kwargs)
                    elif analysis_type == AnalysisType.OSH_METRIC_VALIDATION:
                        logger.debug("Executing OSH metric validation analysis")
                        results[analysis_type] = self._validate_osh_metrics(measurements, **kwargs)
                    elif analysis_type == AnalysisType.MEASUREMENT_QUALITY:
                        logger.debug("Executing measurement quality analysis")
                        results[analysis_type] = self._analyze_measurement_quality(measurements, **kwargs)
                    elif analysis_type == AnalysisType.CONSENSUS_ANALYSIS:
                        logger.debug("Executing consensus analysis")
                        results[analysis_type] = self._analyze_consensus(measurements, **kwargs)
                    elif analysis_type == AnalysisType.PATTERN_CLASSIFICATION:
                        logger.debug("Executing pattern classification analysis")
                        results[analysis_type] = self._classify_patterns(measurements, **kwargs)
                    else:
                        logger.warning(f"Unknown analysis type: {analysis_type}")
                        continue
                    
                    analysis_time = time.time() - analysis_start_time
                    logger.info(f"✓ Completed {analysis_type.value} analysis in {analysis_time:.3f}s")
                    logger.debug(f"  Result success: {results[analysis_type].success}")
                    logger.debug(f"  Result sample size: {results[analysis_type].sample_size}")
                    
                    successful_analyses += 1
                    self.analysis_stats['analysis_type_counts'][analysis_type.value] += 1
                
                except Exception as e:
                    analysis_time = time.time() - analysis_start_time
                    logger.error(f"✗ Failed {analysis_type.value} analysis after {analysis_time:.3f}s: {str(e)}")
                    logger.error(f"  Exception type: {type(e).__name__}")
                    logger.debug(f"  Exception traceback: {traceback.format_exc()}")
                    
                    results[analysis_type] = StatisticalResult(
                        analysis_type=analysis_type,
                        success=False,
                        timestamp=time.time(),
                        execution_time=analysis_time,
                        sample_size=len(measurements),
                        warnings=[f"Analysis failed: {str(e)}"]
                    )
                    failed_analyses += 1
                    self.analysis_stats['error_count'] += 1
            
            execution_time = time.time() - start_time
            self.analysis_stats['total_analyses'] += 1
            self.analysis_stats['execution_times'].append(execution_time)
            self.analysis_stats['last_analysis_time'] = time.time()
            self.analysis_stats['average_execution_time'] = np.mean(self.analysis_stats['execution_times'])
            
            logger.info("=" * 50)
            logger.info("ANALYSIS SUMMARY")
            logger.info("=" * 50)
            logger.info(f"Total execution time: {execution_time:.3f}s")
            logger.info(f"Successful analyses: {successful_analyses}")
            logger.info(f"Failed analyses: {failed_analyses}")
            logger.info(f"Success rate: {successful_analyses/(successful_analyses+failed_analyses)*100:.1f}%")
            logger.info(f"Average time per analysis: {execution_time/len(analysis_types):.3f}s")
            
            return results
        
        except Exception as e:
            logger.error(f"Critical failure in analyze_measurements: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise StatisticalAnalysisError(f"Statistical analysis failed: {str(e)}")
    
    def _log_measurement_characteristics(self, measurements: List[MeasurementResult]):
        """Log characteristics of the measurement data."""
        logger.debug("Analyzing measurement data characteristics:")
        
        try:
            # Basic counts
            total_measurements = len(measurements)
            logger.debug(f"  Total measurements: {total_measurements}")
            
            # Outcome distribution
            outcomes = [m.outcome for m in measurements]
            outcome_counts = Counter(outcomes)
            logger.debug(f"  Unique outcomes: {len(outcome_counts)}")
            logger.debug(f"  Outcome distribution: {dict(outcome_counts)}")
            
            # Basis distribution
            bases = [m.basis for m in measurements]
            basis_counts = Counter(bases)
            logger.debug(f"  Unique bases: {len(basis_counts)}")
            logger.debug(f"  Basis distribution: {dict(basis_counts)}")
            
            # Data completeness
            coherence_before_count = sum(1 for m in measurements if m.coherence_before is not None)
            coherence_after_count = sum(1 for m in measurements if m.coherence_after is not None)
            entropy_before_count = sum(1 for m in measurements if m.entropy_before is not None)
            entropy_after_count = sum(1 for m in measurements if m.entropy_after is not None)
            
            logger.debug(f"  Data completeness:")
            logger.debug(f"    coherence_before: {coherence_before_count}/{total_measurements} ({coherence_before_count/total_measurements*100:.1f}%)")
            logger.debug(f"    coherence_after: {coherence_after_count}/{total_measurements} ({coherence_after_count/total_measurements*100:.1f}%)")
            logger.debug(f"    entropy_before: {entropy_before_count}/{total_measurements} ({entropy_before_count/total_measurements*100:.1f}%)")
            logger.debug(f"    entropy_after: {entropy_after_count}/{total_measurements*100:.1f}%)")
            
            # Time range
            timestamps = [m.timestamp for m in measurements]
            if timestamps:
                time_range = max(timestamps) - min(timestamps)
                logger.debug(f"  Time range: {time_range:.3f}s")
                logger.debug(f"  Earliest timestamp: {min(timestamps)}")
                logger.debug(f"  Latest timestamp: {max(timestamps)}")
            
            # Observer information
            observers = [m.observer for m in measurements if m.observer]
            if observers:
                observer_counts = Counter(observers)
                logger.debug(f"  Unique observers: {len(observer_counts)}")
                logger.debug(f"  Observer distribution: {dict(observer_counts)}")
            else:
                logger.debug(f"  No observer information available")
            
        except Exception as e:
            logger.warning(f"Failed to log measurement characteristics: {str(e)}")
    
    @log_function_entry
    def _analyze_basic_statistics(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Analyze basic statistical properties of measurements."""
        logger.info("Starting basic statistics analysis")
        start_time = time.time()
        
        try:
            # Extract numerical data with detailed logging
            logger.debug("Extracting numerical data from measurements")
            
            probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
            logger.debug(f"Extracted {len(probabilities)} probability values")
            logger.debug(f"Probability range: [{min(probabilities):.4f}, {max(probabilities):.4f}]")
            
            coherence_before = [m.coherence_before for m in measurements if m.coherence_before is not None]
            coherence_after = [m.coherence_after for m in measurements if m.coherence_after is not None]
            entropy_before = [m.entropy_before for m in measurements if m.entropy_before is not None]
            entropy_after = [m.entropy_after for m in measurements if m.entropy_after is not None]
            
            logger.debug(f"Coherence before: {len(coherence_before)} values")
            logger.debug(f"Coherence after: {len(coherence_after)} values")
            logger.debug(f"Entropy before: {len(entropy_before)} values")
            logger.debug(f"Entropy after: {len(entropy_after)} values")
            
            statistics = {}
            warnings = []
            
            # Basic probability statistics
            if probabilities:
                logger.debug("Computing probability statistics")
                prob_stats = {
                    'mean': float(np.mean(probabilities)),
                    'std': float(np.std(probabilities)),
                    'min': float(np.min(probabilities)),
                    'max': float(np.max(probabilities)),
                    'median': float(np.median(probabilities)),
                    'q25': float(np.percentile(probabilities, 25)),
                    'q75': float(np.percentile(probabilities, 75)),
                    'skewness': float(self._calculate_skewness(probabilities)),
                    'kurtosis': float(self._calculate_kurtosis(probabilities))
                }
                statistics['probability'] = prob_stats
                logger.debug(f"Probability statistics: {prob_stats}")
            else:
                logger.warning("No probability data available")
                warnings.append("No probability data available")
            
            # Coherence statistics
            if coherence_before:
                logger.debug("Computing coherence_before statistics")
                coherence_before_stats = {
                    'mean': float(np.mean(coherence_before)),
                    'std': float(np.std(coherence_before)),
                    'min': float(np.min(coherence_before)),
                    'max': float(np.max(coherence_before))
                }
                statistics['coherence_before'] = coherence_before_stats
                logger.debug(f"Coherence before stats: {coherence_before_stats}")
            
            if coherence_after:
                logger.debug("Computing coherence_after statistics")
                coherence_after_stats = {
                    'mean': float(np.mean(coherence_after)),
                    'std': float(np.std(coherence_after)),
                    'min': float(np.min(coherence_after)),
                    'max': float(np.max(coherence_after))
                }
                statistics['coherence_after'] = coherence_after_stats
                logger.debug(f"Coherence after stats: {coherence_after_stats}")
            
            # Entropy statistics
            if entropy_before:
                logger.debug("Computing entropy_before statistics")
                entropy_before_stats = {
                    'mean': float(np.mean(entropy_before)),
                    'std': float(np.std(entropy_before)),
                    'min': float(np.min(entropy_before)),
                    'max': float(np.max(entropy_before))
                }
                statistics['entropy_before'] = entropy_before_stats
                logger.debug(f"Entropy before stats: {entropy_before_stats}")
            
            if entropy_after:
                logger.debug("Computing entropy_after statistics")
                entropy_after_stats = {
                    'mean': float(np.mean(entropy_after)),
                    'std': float(np.std(entropy_after)),
                    'min': float(np.min(entropy_after)),
                    'max': float(np.max(entropy_after))
                }
                statistics['entropy_after'] = entropy_after_stats
                logger.debug(f"Entropy after stats: {entropy_after_stats}")
            
            # Outcome distribution
            logger.debug("Computing outcome distribution")
            outcome_counts = Counter(m.outcome for m in measurements)
            statistics['outcome_distribution'] = dict(outcome_counts)
            statistics['outcome_entropy'] = self._calculate_shannon_entropy(list(outcome_counts.values()))
            logger.debug(f"Outcome distribution: {dict(outcome_counts)}")
            logger.debug(f"Outcome entropy: {statistics['outcome_entropy']:.4f}")
            
            # Basis distribution
            logger.debug("Computing basis distribution")
            basis_counts = Counter(m.basis for m in measurements)
            statistics['basis_distribution'] = dict(basis_counts)
            logger.debug(f"Basis distribution: {dict(basis_counts)}")
            
            # Measurement timing statistics
            logger.debug("Computing timing statistics")
            timestamps = [m.timestamp for m in measurements]
            if len(timestamps) > 1:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                timing_stats = {
                    'total_duration': float(timestamps[-1] - timestamps[0]),
                    'mean_interval': float(np.mean(intervals)),
                    'std_interval': float(np.std(intervals)),
                    'measurement_rate': float(len(measurements) / (timestamps[-1] - timestamps[0]))
                }
                statistics['timing'] = timing_stats
                logger.debug(f"Timing statistics: {timing_stats}")
            else:
                logger.debug("Insufficient timing data for statistics")
            
            execution_time = time.time() - start_time
            logger.info(f"Basic statistics analysis completed in {execution_time:.3f}s")
            logger.debug(f"Generated {len(statistics)} statistical categories")
            logger.debug(f"Warnings generated: {len(warnings)}")
            
            return StatisticalResult(
                analysis_type=AnalysisType.BASIC_STATISTICS,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=statistics,
                warnings=warnings
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Basic statistics analysis failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Basic statistics analysis failed: {str(e)}")
    
    @log_function_entry
    def _analyze_distributions(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Analyze probability distributions of measurement data."""
        logger.info("Starting distribution analysis")
        start_time = time.time()
        
        try:
            probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
            logger.debug(f"Extracted {len(probabilities)} probability values for distribution analysis")
            
            if not probabilities:
                logger.warning("No probability data available for distribution analysis")
                return StatisticalResult(
                    analysis_type=AnalysisType.DISTRIBUTION_ANALYSIS,
                    success=False,
                    timestamp=time.time(),
                    execution_time=time.time() - start_time,
                    sample_size=len(measurements),
                    warnings=["No probability data available for distribution analysis"]
                )
            
            logger.debug(f"Probability data range: [{min(probabilities):.4f}, {max(probabilities):.4f}]")
            logger.debug(f"Probability data mean: {np.mean(probabilities):.4f}")
            logger.debug(f"Probability data std: {np.std(probabilities):.4f}")
            
            distribution_analysis = self._fit_distributions(probabilities)
            
            execution_time = time.time() - start_time
            logger.info(f"Distribution analysis completed in {execution_time:.3f}s")
            logger.debug(f"Best distribution: {distribution_analysis.distribution_type}")
            logger.debug(f"Distribution parameters: {distribution_analysis.parameters}")
            
            return StatisticalResult(
                analysis_type=AnalysisType.DISTRIBUTION_ANALYSIS,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics={'distribution_analysis': distribution_analysis.to_dict()}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Distribution analysis failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Distribution analysis failed: {str(e)}")
    
    @log_function_entry
    def _fit_distributions(self, data: List[float]) -> DistributionAnalysis:
        """Fit various distributions to data and find best fit."""
        logger.debug(f"Fitting distributions to {len(data)} data points")
        
        try:
            data_array = np.array(data)
            logger.debug(f"Data array shape: {data_array.shape}")
            logger.debug(f"Data array statistics: mean={np.mean(data_array):.4f}, std={np.std(data_array):.4f}")
            
            # Test for normality
            normality_tests = {}
            
            if SCIPY_AVAILABLE and len(data) >= 8:
                logger.debug("Running normality tests with SciPy")
                try:
                    # Shapiro-Wilk test
                    shapiro_stat, shapiro_p = stats.shapiro(data_array)
                    normality_tests['shapiro_wilk'] = (float(shapiro_stat), float(shapiro_p))
                    logger.debug(f"Shapiro-Wilk test: stat={shapiro_stat:.4f}, p={shapiro_p:.4f}")
                    
                    # D'Agostino's test
                    if len(data) >= 20:
                        dagostino_stat, dagostino_p = stats.normaltest(data_array)
                        normality_tests['dagostino'] = (float(dagostino_stat), float(dagostino_p))
                        logger.debug(f"D'Agostino test: stat={dagostino_stat:.4f}, p={dagostino_p:.4f}")
                
                except Exception as e:
                    logger.warning(f"Normality tests failed: {str(e)}")
            else:
                logger.debug("Skipping normality tests (insufficient data or no SciPy)")
            
            # Calculate percentiles
            logger.debug("Computing percentiles")
            percentiles = {
                'p1': float(np.percentile(data_array, 1)),
                'p5': float(np.percentile(data_array, 5)),
                'p10': float(np.percentile(data_array, 10)),
                'p25': float(np.percentile(data_array, 25)),
                'p50': float(np.percentile(data_array, 50)),
                'p75': float(np.percentile(data_array, 75)),
                'p90': float(np.percentile(data_array, 90)),
                'p95': float(np.percentile(data_array, 95)),
                'p99': float(np.percentile(data_array, 99))
            }
            logger.debug(f"Percentiles: {percentiles}")
            
            # Detect outliers using IQR method
            logger.debug("Detecting outliers using IQR method")
            q1, q3 = np.percentile(data_array, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = [i for i, val in enumerate(data_array) if val < lower_bound or val > upper_bound]
            logger.debug(f"IQR bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
            logger.debug(f"Found {len(outliers)} outliers at indices: {outliers[:10]}{'...' if len(outliers) > 10 else ''}")
            
            # Fit distributions
            best_distribution = "unknown"
            parameters = {}
            goodness_of_fit = {}
            
            if SCIPY_AVAILABLE and len(data) >= 10:
                logger.debug("Fitting distributions with SciPy")
                try:
                    # Test common distributions
                    distributions = [
                        ('normal', stats.norm),
                        ('uniform', stats.uniform),
                        ('exponential', stats.expon),
                        ('beta', stats.beta),
                        ('gamma', stats.gamma)
                    ]
                    
                    best_aic = float('inf')
                    
                    for dist_name, dist in distributions:
                        logger.debug(f"Fitting {dist_name} distribution")
                        try:
                            # Fit distribution parameters
                            params = dist.fit(data_array)
                            logger.debug(f"  {dist_name} parameters: {params}")
                            
                            # Calculate AIC
                            log_likelihood = np.sum(dist.logpdf(data_array, *params))
                            aic = 2 * len(params) - 2 * log_likelihood
                            
                            goodness_of_fit[dist_name] = {'aic': float(aic), 'log_likelihood': float(log_likelihood)}
                            logger.debug(f"  {dist_name} AIC: {aic:.4f}, log_likelihood: {log_likelihood:.4f}")
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_distribution = dist_name
                                parameters = {f'param_{i}': float(p) for i, p in enumerate(params)}
                                logger.debug(f"  New best distribution: {dist_name} with AIC {aic:.4f}")
                        
                        except Exception as e:
                            logger.debug(f"Failed to fit {dist_name} distribution: {str(e)}")
                            continue
                    
                    logger.info(f"Best fitting distribution: {best_distribution} with AIC {best_aic:.4f}")
                
                except Exception as e:
                    logger.warning(f"Distribution fitting failed: {str(e)}")
            
            else:
                logger.debug("Using fallback distribution characterization")
                # Fallback: simple characterization
                mean_val = np.mean(data_array)
                std_val = np.std(data_array)
                
                if abs(std_val) < 1e-10:
                    best_distribution = "constant"
                    parameters = {'value': float(mean_val)}
                    logger.debug("Detected constant distribution")
                elif np.all(data_array >= 0) and np.all(data_array <= 1):
                    best_distribution = "beta_like"
                    parameters = {'mean': float(mean_val), 'std': float(std_val)}
                    logger.debug("Detected beta-like distribution")
                else:
                    best_distribution = "normal_like"
                    parameters = {'mean': float(mean_val), 'std': float(std_val)}
                    logger.debug("Detected normal-like distribution")
            
            result = DistributionAnalysis(
                distribution_type=best_distribution,
                parameters=parameters,
                goodness_of_fit=goodness_of_fit,
                normality_tests=normality_tests,
                outliers=outliers,
                percentiles=percentiles
            )
            
            logger.debug("Distribution analysis completed successfully")
            return result
        
        except Exception as e:
            logger.error(f"Distribution fitting failed: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    def _calculate_confidence_intervals(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Calculate confidence intervals for measurement statistics."""
        logger.info("Starting confidence interval calculation")
        start_time = time.time()
        
        try:
            confidence_level = kwargs.get('confidence_level', self.config.confidence_level)
            logger.debug(f"Using confidence level: {confidence_level}")
            
            probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
            confidence_intervals = {}
            
            if probabilities:
                logger.debug(f"Computing confidence intervals for {len(probabilities)} probability values")
                
                # Bootstrap confidence intervals for mean
                logger.debug(f"Running bootstrap with {self.config.bootstrap_iterations} iterations")
                bootstrap_means = []
                for i in range(self.config.bootstrap_iterations):
                    if i % (self.config.bootstrap_iterations // 10) == 0:
                        logger.debug(f"Bootstrap progress: {i}/{self.config.bootstrap_iterations}")
                    
                    bootstrap_sample = np.random.choice(probabilities, size=len(probabilities), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                bootstrap_ci = (
                    float(np.percentile(bootstrap_means, lower_percentile)),
                    float(np.percentile(bootstrap_means, upper_percentile))
                )
                confidence_intervals['probability_mean'] = bootstrap_ci
                logger.debug(f"Bootstrap CI for probability mean: {bootstrap_ci}")
                
                # Standard confidence interval for proportion
                mean_prob = np.mean(probabilities)
                n = len(probabilities)
                
                if SCIPY_AVAILABLE:
                    z_score = stats.norm.ppf(1 - alpha / 2)
                    logger.debug(f"Using SciPy z-score: {z_score:.4f}")
                else:
                    # Approximate critical values
                    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
                    z_score = z_scores.get(confidence_level, 1.96)
                    logger.debug(f"Using approximate z-score: {z_score:.4f}")
                
                margin_of_error = z_score * np.sqrt(mean_prob * (1 - mean_prob) / n)
                proportion_ci = (
                    float(max(0, mean_prob - margin_of_error)),
                    float(min(1, mean_prob + margin_of_error))
                )
                confidence_intervals['probability_proportion'] = proportion_ci
                logger.debug(f"Proportion CI for probability: {proportion_ci}")
            
            # Confidence intervals for coherence and entropy if available
            for metric_name in ['coherence_before', 'coherence_after', 'entropy_before', 'entropy_after']:
                values = [getattr(m, metric_name) for m in measurements if getattr(m, metric_name) is not None]
                
                if values and len(values) >= 3:
                    logger.debug(f"Computing CI for {metric_name} with {len(values)} values")
                    
                    # Bootstrap confidence interval
                    bootstrap_means = []
                    bootstrap_iterations = min(self.config.bootstrap_iterations, 1000)  # Limit for performance
                    
                    for _ in range(bootstrap_iterations):
                        bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    alpha = 1 - confidence_level
                    lower_percentile = (alpha / 2) * 100
                    upper_percentile = (1 - alpha / 2) * 100
                    
                    metric_ci = (
                        float(np.percentile(bootstrap_means, lower_percentile)),
                        float(np.percentile(bootstrap_means, upper_percentile))
                    )
                    confidence_intervals[f'{metric_name}_mean'] = metric_ci
                    logger.debug(f"CI for {metric_name}: {metric_ci}")
            
            execution_time = time.time() - start_time
            logger.info(f"Confidence interval calculation completed in {execution_time:.3f}s")
            logger.debug(f"Generated {len(confidence_intervals)} confidence intervals")
            
            return StatisticalResult(
                analysis_type=AnalysisType.CONFIDENCE_INTERVALS,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                confidence_intervals=confidence_intervals,
                metadata={'confidence_level': confidence_level}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Confidence interval calculation failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Confidence interval calculation failed: {str(e)}")
    
    @log_function_entry
    def _perform_hypothesis_tests(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Perform statistical hypothesis tests on measurement data."""
        logger.info("Starting hypothesis testing")
        start_time = time.time()
        
        try:
            p_values = {}
            effect_sizes = {}
            statistics = {}
            warnings = []
            
            # Test for uniform distribution of outcomes
            logger.debug("Testing for uniform distribution of outcomes")
            outcome_counts = Counter(m.outcome for m in measurements)
            logger.debug(f"Outcome counts: {dict(outcome_counts)}")
            
            if len(outcome_counts) > 1:
                observed_frequencies = list(outcome_counts.values())
                expected_frequency = len(measurements) / len(outcome_counts)
                expected_frequencies = [expected_frequency] * len(outcome_counts)
                
                logger.debug(f"Observed frequencies: {observed_frequencies}")
                logger.debug(f"Expected frequencies: {expected_frequencies}")
                
                # Chi-square goodness of fit test
                if SCIPY_AVAILABLE and all(f >= 5 for f in expected_frequencies):
                    logger.debug("Running chi-square goodness of fit test")
                    try:
                        chi2_stat, chi2_p = stats.chisquare(observed_frequencies, expected_frequencies)
                        p_values['outcome_uniformity'] = float(chi2_p)
                        statistics['chi2_statistic'] = float(chi2_stat)
                        
                        logger.debug(f"Chi-square test: stat={chi2_stat:.4f}, p={chi2_p:.4f}")
                        
                        # Effect size (Cramér's V)
                        n = sum(observed_frequencies)
                        k = len(observed_frequencies)
                        cramers_v = np.sqrt(chi2_stat / (n * (k - 1)))
                        effect_sizes['outcome_uniformity_cramers_v'] = float(cramers_v)
                        logger.debug(f"Cramér's V: {cramers_v:.4f}")
                    
                    except Exception as e:
                        warning_msg = f"Chi-square test failed: {str(e)}"
                        logger.warning(warning_msg)
                        warnings.append(warning_msg)
                else:
                    warning_msg = "Chi-square test requirements not met (expected frequencies < 5)"
                    logger.warning(warning_msg)
                    warnings.append(warning_msg)
            
            # Test for coherence preservation
            logger.debug("Testing for coherence preservation")
            coherence_before = [m.coherence_before for m in measurements if m.coherence_before is not None]
            coherence_after = [m.coherence_after for m in measurements if m.coherence_after is not None]
            
            logger.debug(f"Coherence before: {len(coherence_before)} values")
            logger.debug(f"Coherence after: {len(coherence_after)} values")
            
            if coherence_before and coherence_after and len(coherence_before) == len(coherence_after):
                if len(coherence_before) >= 3:
                    logger.debug("Running coherence preservation test")
                    
                    # Paired t-test for coherence change
                    if SCIPY_AVAILABLE:
                        try:
                            t_stat, t_p = stats.ttest_rel(coherence_before, coherence_after)
                            p_values['coherence_preservation'] = float(t_p)
                            statistics['coherence_t_statistic'] = float(t_stat)
                            
                            logger.debug(f"Paired t-test: stat={t_stat:.4f}, p={t_p:.4f}")
                            
                            # Effect size (Cohen's d for paired samples)
                            differences = np.array(coherence_before) - np.array(coherence_after)
                            cohens_d = np.mean(differences) / np.std(differences)
                            effect_sizes['coherence_preservation_cohens_d'] = float(cohens_d)
                            logger.debug(f"Cohen's d: {cohens_d:.4f}")
                        
                        except Exception as e:
                            warning_msg = f"Coherence t-test failed: {str(e)}"
                            logger.warning(warning_msg)
                            warnings.append(warning_msg)
                    else:
                        logger.debug("Running simple sign test for coherence")
                        # Simple sign test
                        differences = np.array(coherence_before) - np.array(coherence_after)
                        positive_changes = np.sum(differences > 0)
                        negative_changes = np.sum(differences < 0)
                        
                        if positive_changes + negative_changes > 0:
                            # Binomial test approximation
                            p_greater = positive_changes / (positive_changes + negative_changes)
                            statistics['coherence_sign_test_ratio'] = float(p_greater)
                            logger.debug(f"Sign test ratio: {p_greater:.4f}")
            
            # Test for measurement timing randomness
            logger.debug("Testing for measurement timing randomness")
            timestamps = [m.timestamp for m in measurements]
            if len(timestamps) > 2:
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                logger.debug(f"Timing intervals: {len(intervals)} values, mean={np.mean(intervals):.4f}")
                
                if len(intervals) >= 3 and SCIPY_AVAILABLE:
                    try:
                        # Test for exponential distribution (random timing)
                        ks_stat, ks_p = stats.kstest(intervals, 'expon', args=(0, np.mean(intervals)))
                        p_values['timing_randomness'] = float(ks_p)
                        statistics['timing_ks_statistic'] = float(ks_stat)
                        logger.debug(f"KS test for timing: stat={ks_stat:.4f}, p={ks_p:.4f}")
                    
                    except Exception as e:
                        warning_msg = f"Timing randomness test failed: {str(e)}"
                        logger.warning(warning_msg)
                        warnings.append(warning_msg)
            
            execution_time = time.time() - start_time
            logger.info(f"Hypothesis testing completed in {execution_time:.3f}s")
            logger.debug(f"Generated {len(p_values)} p-values, {len(effect_sizes)} effect sizes")
            logger.debug(f"Warnings: {len(warnings)}")
            
            return StatisticalResult(
                analysis_type=AnalysisType.HYPOTHESIS_TEST,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=statistics,
                p_values=p_values,
                effect_sizes=effect_sizes,
                warnings=warnings
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Hypothesis testing failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Hypothesis testing failed: {str(e)}")
    
    @log_function_entry
    def _analyze_correlations(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Analyze correlations between measurement variables."""
        logger.info("Starting correlation analysis")
        start_time = time.time()
        
        try:
            # Extract numerical variables
            logger.debug("Extracting numerical variables for correlation analysis")
            variables = {}
            
            # Probability variables
            variables['max_probability'] = [max(m.probabilities.values()) for m in measurements]
            variables['probability_entropy'] = [self._calculate_shannon_entropy(list(m.probabilities.values())) for m in measurements]
            logger.debug(f"Extracted probability variables: max_probability, probability_entropy")
            
            # Coherence variables
            coherence_before = [m.coherence_before for m in measurements if m.coherence_before is not None]
            coherence_after = [m.coherence_after for m in measurements if m.coherence_after is not None]
            
            if coherence_before and len(coherence_before) == len(measurements):
                variables['coherence_before'] = coherence_before
                logger.debug("Added coherence_before variable")
            if coherence_after and len(coherence_after) == len(measurements):
                variables['coherence_after'] = coherence_after
                logger.debug("Added coherence_after variable")
            
            # Entropy variables
            entropy_before = [m.entropy_before for m in measurements if m.entropy_before is not None]
            entropy_after = [m.entropy_after for m in measurements if m.entropy_after is not None]
            
            if entropy_before and len(entropy_before) == len(measurements):
                variables['entropy_before'] = entropy_before
                logger.debug("Added entropy_before variable")
            if entropy_after and len(entropy_after) == len(measurements):
                variables['entropy_after'] = entropy_after
                logger.debug("Added entropy_after variable")
            
            # Time variables
            if len(measurements) > 1:
                timestamps = [m.timestamp for m in measurements]
                variables['timestamp'] = timestamps
                variables['measurement_index'] = list(range(len(measurements)))
                logger.debug("Added time variables: timestamp, measurement_index")
            
            logger.debug(f"Total variables for correlation: {list(variables.keys())}")
            
            # Calculate correlation matrix
            correlations = {}
            p_values = {}
            
            variable_names = list(variables.keys())
            total_pairs = len(variable_names) * (len(variable_names) - 1) // 2
            logger.debug(f"Computing correlations for {total_pairs} variable pairs")
            
            computed_pairs = 0
            for i, var1 in enumerate(variable_names):
                for j, var2 in enumerate(variable_names[i+1:], i+1):
                    computed_pairs += 1
                    if computed_pairs % max(1, total_pairs // 10) == 0:
                        logger.debug(f"Correlation progress: {computed_pairs}/{total_pairs}")
                    
                    try:
                        data1 = np.array(variables[var1])
                        data2 = np.array(variables[var2])
                        
                        if len(data1) == len(data2) and len(data1) >= 3:
                            # Pearson correlation
                            if SCIPY_AVAILABLE:
                                corr_coeff, corr_p = stats.pearsonr(data1, data2)
                                correlations[f'{var1}_vs_{var2}'] = float(corr_coeff)
                                p_values[f'{var1}_vs_{var2}_pearson'] = float(corr_p)
                                
                                # Spearman correlation (rank-based)
                                spearman_coeff, spearman_p = stats.spearmanr(data1, data2)
                                correlations[f'{var1}_vs_{var2}_spearman'] = float(spearman_coeff)
                                p_values[f'{var1}_vs_{var2}_spearman'] = float(spearman_p)
                                
                                logger.debug(f"Correlations {var1} vs {var2}: Pearson={corr_coeff:.3f}, Spearman={spearman_coeff:.3f}")
                            else:
                                # Simple correlation coefficient
                                corr_coeff = np.corrcoef(data1, data2)[0, 1]
                                if not np.isnan(corr_coeff):
                                    correlations[f'{var1}_vs_{var2}'] = float(corr_coeff)
                                    logger.debug(f"Simple correlation {var1} vs {var2}: {corr_coeff:.3f}")
                    
                    except Exception as e:
                        logger.debug(f"Correlation calculation failed for {var1} vs {var2}: {str(e)}")
                        continue
            
            # Identify strong correlations
            strong_correlations = {
                k: v for k, v in correlations.items() 
                if abs(v) >= self.config.correlation_threshold
            }
            
            logger.info(f"Found {len(strong_correlations)} strong correlations (threshold: {self.config.correlation_threshold})")
            for k, v in strong_correlations.items():
                logger.info(f"  {k}: {v:.3f}")
            
            statistics = {
                'correlation_matrix': correlations,
                'strong_correlations': strong_correlations,
                'variable_count': len(variables),
                'correlation_count': len(correlations)
            }
            
            execution_time = time.time() - start_time
            logger.info(f"Correlation analysis completed in {execution_time:.3f}s")
            
            return StatisticalResult(
                analysis_type=AnalysisType.CORRELATION_ANALYSIS,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=statistics,
                p_values=p_values,
                metadata={'correlation_threshold': self.config.correlation_threshold}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Correlation analysis failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Correlation analysis failed: {str(e)}")
    
    @log_function_entry
    def _analyze_time_series(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Analyze time series patterns in measurement data."""
        logger.info("Starting time series analysis")
        start_time = time.time()
        
        try:
            if len(measurements) < 5:
                logger.warning(f"Insufficient data for time series analysis: {len(measurements)} < 5")
                return StatisticalResult(
                    analysis_type=AnalysisType.TIME_SERIES_ANALYSIS,
                    success=False,
                    timestamp=time.time(),
                    execution_time=time.time() - start_time,
                    sample_size=len(measurements),
                    warnings=["Insufficient data for time series analysis"]
                )
            
            # Sort measurements by timestamp
            logger.debug("Sorting measurements by timestamp")
            sorted_measurements = sorted(measurements, key=lambda m: m.timestamp)
            timestamps = [m.timestamp for m in sorted_measurements]
            
            logger.debug(f"Time series spans {timestamps[-1] - timestamps[0]:.3f} seconds")
            logger.debug(f"Average time interval: {(timestamps[-1] - timestamps[0]) / (len(timestamps) - 1):.3f} seconds")
            
            # Analyze different time series
            time_series_results = {}
            
            # Probability time series
            logger.debug("Analyzing probability time series")
            probabilities = [max(m.probabilities.values()) for m in sorted_measurements]
            time_series_results['probability'] = self._analyze_single_time_series(probabilities, timestamps)
            
            # Coherence time series
            coherence_values = [m.coherence_before for m in sorted_measurements if m.coherence_before is not None]
            if len(coherence_values) >= 5:
                logger.debug(f"Analyzing coherence time series ({len(coherence_values)} points)")
                coherence_timestamps = [m.timestamp for m in sorted_measurements if m.coherence_before is not None]
                time_series_results['coherence'] = self._analyze_single_time_series(coherence_values, coherence_timestamps)
            else:
                logger.debug("Insufficient coherence data for time series analysis")
            
            # Entropy time series
            entropy_values = [m.entropy_after for m in sorted_measurements if m.entropy_after is not None]
            if len(entropy_values) >= 5:
                logger.debug(f"Analyzing entropy time series ({len(entropy_values)} points)")
                entropy_timestamps = [m.timestamp for m in sorted_measurements if m.entropy_after is not None]
                time_series_results['entropy'] = self._analyze_single_time_series(entropy_values, entropy_timestamps)
            else:
                logger.debug("Insufficient entropy data for time series analysis")
            
            execution_time = time.time() - start_time
            logger.info(f"Time series analysis completed in {execution_time:.3f}s")
            logger.debug(f"Analyzed {len(time_series_results)} time series")
            
            return StatisticalResult(
                analysis_type=AnalysisType.TIME_SERIES_ANALYSIS,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=time_series_results
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Time series analysis failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Time series analysis failed: {str(e)}")
    
    @log_function_entry
    def _analyze_single_time_series(self, values: List[float], timestamps: List[float]) -> Dict[str, Any]:
        """Analyze a single time series."""
        logger.debug(f"Analyzing single time series with {len(values)} values")
        
        try:
            values_array = np.array(values)
            logger.debug(f"Time series statistics: mean={np.mean(values_array):.4f}, std={np.std(values_array):.4f}")
            
            # Basic trend analysis
            if len(values) >= 3:
                logger.debug("Computing linear trend")
                # Linear trend
                x = np.arange(len(values))
                if SCIPY_AVAILABLE:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values_array)
                    trend_analysis = {
                        'slope': float(slope),
                        'r_squared': float(r_value**2),
                        'p_value': float(p_value),
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                    }
                    logger.debug(f"Linear trend: slope={slope:.4f}, r²={r_value**2:.4f}, p={p_value:.4f}")
                else:
                    # Simple slope calculation
                    slope = (values_array[-1] - values_array[0]) / (len(values_array) - 1)
                    trend_analysis = {
                        'slope': float(slope),
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable'
                    }
                    logger.debug(f"Simple slope: {slope:.4f}")
            else:
                trend_analysis = {'trend_direction': 'insufficient_data'}
                logger.debug("Insufficient data for trend analysis")
            
            # Autocorrelation analysis
            autocorrelation = []
            if len(values) >= 5:
                logger.debug("Computing autocorrelation")
                max_lag = min(len(values) // 2, 10)
                for lag in range(1, max_lag + 1):
                    if lag < len(values):
                        corr = np.corrcoef(values_array[:-lag], values_array[lag:])[0, 1]
                        if not np.isnan(corr):
                            autocorrelation.append(float(corr))
                
                logger.debug(f"Autocorrelation computed for {len(autocorrelation)} lags")
            
            # Change point detection (simple approach)
            change_points = []
            if len(values) >= 10:
                logger.debug("Detecting change points")
                # Look for significant changes in mean
                window_size = max(3, len(values) // 5)
                for i in range(window_size, len(values) - window_size):
                    before_mean = np.mean(values_array[i-window_size:i])
                    after_mean = np.mean(values_array[i:i+window_size])
                    
                    # Significant change threshold
                    if abs(after_mean - before_mean) > 2 * np.std(values_array):
                        change_points.append(i)
                
                logger.debug(f"Found {len(change_points)} change points")
            
            # Stationarity test (simple variance check)
            stationarity = {'is_stationary': True}
            if len(values) >= 10:
                logger.debug("Testing stationarity")
                # Divide into two halves and compare variances
                mid = len(values) // 2
                first_half_var = np.var(values_array[:mid])
                second_half_var = np.var(values_array[mid:])
                
                # If variances differ significantly, likely non-stationary
                if max(first_half_var, second_half_var) > 4 * min(first_half_var, second_half_var):
                    stationarity['is_stationary'] = False
                
                stationarity['first_half_variance'] = float(first_half_var)
                stationarity['second_half_variance'] = float(second_half_var)
                logger.debug(f"Stationarity test: {stationarity}")
            
            result = {
                'trend_analysis': trend_analysis,
                'autocorrelation': autocorrelation,
                'change_points': change_points,
                'stationarity': stationarity,
                'length': len(values),
                'mean': float(np.mean(values_array)),
                'variance': float(np.var(values_array))
            }
            
            logger.debug("Single time series analysis completed")
            return result
        
        except Exception as e:
            logger.error(f"Single time series analysis failed: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    def _perform_bayesian_inference(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Perform Bayesian inference on measurement data."""
        logger.info("Starting Bayesian inference")
        start_time = time.time()
        
        try:
            # Simple Bayesian analysis for measurement probabilities
            probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
            
            if not probabilities:
                logger.warning("No probability data for Bayesian inference")
                return StatisticalResult(
                    analysis_type=AnalysisType.BAYESIAN_INFERENCE,
                    success=False,
                    timestamp=time.time(),
                    execution_time=time.time() - start_time,
                    sample_size=len(measurements),
                    warnings=["No probability data for Bayesian inference"]
                )
            
            # Beta-Binomial model for measurement success probability
            # Prior: Beta(1, 1) (uniform)
            # Update with data
            
            success_threshold = kwargs.get('success_threshold', 0.7)
            logger.debug(f"Using success threshold: {success_threshold}")
            
            successes = sum(1 for p in probabilities if p >= success_threshold)
            failures = len(probabilities) - successes
            
            logger.debug(f"Successes: {successes}, Failures: {failures}")
            
            # Posterior: Beta(1 + successes, 1 + failures)
            alpha_posterior = 1 + successes
            beta_posterior = 1 + failures
            
            logger.debug(f"Posterior parameters: alpha={alpha_posterior}, beta={beta_posterior}")
            
            # Calculate posterior statistics
            posterior_mean = alpha_posterior / (alpha_posterior + beta_posterior)
            posterior_variance = (alpha_posterior * beta_posterior) / ((alpha_posterior + beta_posterior)**2 * (alpha_posterior + beta_posterior + 1))
            
            logger.debug(f"Posterior mean: {posterior_mean:.4f}, variance: {posterior_variance:.6f}")
            
            # Credible interval
            if SCIPY_AVAILABLE:
                credible_interval = stats.beta.interval(self.config.confidence_level, alpha_posterior, beta_posterior)
                logger.debug(f"Credible interval (SciPy): {credible_interval}")
            else:
                # Approximate credible interval
                std_dev = np.sqrt(posterior_variance)
                margin = 1.96 * std_dev  # Approximate 95% interval
                credible_interval = (max(0, posterior_mean - margin), min(1, posterior_mean + margin))
                logger.debug(f"Credible interval (approximate): {credible_interval}")
            
            bayesian_results = {
                'prior_parameters': {'alpha': 1, 'beta': 1},
                'posterior_parameters': {'alpha': float(alpha_posterior), 'beta': float(beta_posterior)},
                'posterior_mean': float(posterior_mean),
                'posterior_variance': float(posterior_variance),
                'credible_interval': (float(credible_interval[0]), float(credible_interval[1])),
                'successes': successes,
                'failures': failures,
                'success_threshold': success_threshold
            }
            
            execution_time = time.time() - start_time
            logger.info(f"Bayesian inference completed in {execution_time:.3f}s")
            logger.debug(f"Posterior mean: {posterior_mean:.4f}")
            
            return StatisticalResult(
                analysis_type=AnalysisType.BAYESIAN_INFERENCE,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=bayesian_results,
                confidence_intervals={'success_probability': credible_interval}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Bayesian inference failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Bayesian inference failed: {str(e)}")
    
    @log_function_entry
    def _detect_anomalies(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Detect anomalies in measurement data."""
        logger.info("Starting anomaly detection")
        start_time = time.time()
        
        try:
            anomalies = {
                'probability_anomalies': [],
                'coherence_anomalies': [],
                'entropy_anomalies': [],
                'timing_anomalies': []
            }
            
            statistics = {}
            
            # Probability anomalies
            logger.debug("Detecting probability anomalies")
            probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
            if probabilities:
                prob_mean = np.mean(probabilities)
                prob_std = np.std(probabilities)
                logger.debug(f"Probability statistics: mean={prob_mean:.4f}, std={prob_std:.4f}")
                
                if prob_std > 0:
                    anomaly_count = 0
                    for i, prob in enumerate(probabilities):
                        z_score = abs(prob - prob_mean) / prob_std
                        if z_score > self.config.anomaly_threshold:
                            anomalies['probability_anomalies'].append({
                                'index': i,
                                'value': float(prob),
                                'z_score': float(z_score),
                                'timestamp': measurements[i].timestamp
                            })
                            anomaly_count += 1
                    
                    logger.debug(f"Found {anomaly_count} probability anomalies")
                    statistics['probability_anomaly_count'] = len(anomalies['probability_anomalies'])
                    statistics['probability_anomaly_rate'] = len(anomalies['probability_anomalies']) / len(measurements)
            
            # Coherence anomalies
            logger.debug("Detecting coherence anomalies")
            coherence_values = [(i, m.coherence_before) for i, m in enumerate(measurements) if m.coherence_before is not None]
            if coherence_values:
                coherence_data = [c[1] for c in coherence_values]
                coherence_mean = np.mean(coherence_data)
                coherence_std = np.std(coherence_data)
                logger.debug(f"Coherence statistics: mean={coherence_mean:.4f}, std={coherence_std:.4f}")
                
                if coherence_std > 0:
                    anomaly_count = 0
                    for i, coherence in coherence_values:
                        z_score = abs(coherence - coherence_mean) / coherence_std
                        if z_score > self.config.anomaly_threshold:
                            anomalies['coherence_anomalies'].append({
                                'index': i,
                                'value': float(coherence),
                                'z_score': float(z_score),
                                'timestamp': measurements[i].timestamp
                            })
                            anomaly_count += 1
                    
                    logger.debug(f"Found {anomaly_count} coherence anomalies")
            
            # Timing anomalies
            logger.debug("Detecting timing anomalies")
            if len(measurements) > 2:
                timestamps = [m.timestamp for m in measurements]
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                
                if len(intervals) >= 3:
                    interval_mean = np.mean(intervals)
                    interval_std = np.std(intervals)
                    logger.debug(f"Timing interval statistics: mean={interval_mean:.4f}, std={interval_std:.4f}")
                    
                    if interval_std > 0:
                        anomaly_count = 0
                        for i, interval in enumerate(intervals):
                            z_score = abs(interval - interval_mean) / interval_std
                            if z_score > self.config.anomaly_threshold:
                                anomalies['timing_anomalies'].append({
                                    'index': i,
                                    'interval': float(interval),
                                    'z_score': float(z_score),
                                    'timestamp': timestamps[i]
                                })
                                anomaly_count += 1
                        
                        logger.debug(f"Found {anomaly_count} timing anomalies")
            
            # Overall anomaly statistics
            total_anomalies = sum(len(anomaly_list) for anomaly_list in anomalies.values())
            statistics['total_anomalies'] = total_anomalies
            statistics['anomaly_rate'] = total_anomalies / len(measurements)
            statistics['anomaly_threshold'] = self.config.anomaly_threshold
            
            logger.info(f"Anomaly detection found {total_anomalies} total anomalies")
            logger.info(f"Anomaly rate: {statistics['anomaly_rate']:.3f}")
            
            execution_time = time.time() - start_time
            logger.info(f"Anomaly detection completed in {execution_time:.3f}s")
            
            return StatisticalResult(
                analysis_type=AnalysisType.ANOMALY_DETECTION,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics={'anomalies': anomalies, 'summary': statistics}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Anomaly detection failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Anomaly detection failed: {str(e)}")
    
    @log_function_entry
    def _validate_osh_metrics(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Validate OSH-specific metrics and patterns."""
        logger.info("Starting OSH metric validation")
        start_time = time.time()
        
        try:
            # Extract OSH-relevant data
            coherence_before = [m.coherence_before for m in measurements if m.coherence_before is not None]
            coherence_after = [m.coherence_after for m in measurements if m.coherence_after is not None]
            entropy_before = [m.entropy_before for m in measurements if m.entropy_before is not None]
            entropy_after = [m.entropy_after for m in measurements if m.entropy_after is not None]
            
            logger.debug(f"OSH data availability:")
            logger.debug(f"  coherence_before: {len(coherence_before)} values")
            logger.debug(f"  coherence_after: {len(coherence_after)} values")
            logger.debug(f"  entropy_before: {len(entropy_before)} values")
            logger.debug(f"  entropy_after: {len(entropy_after)} values")
            
            validation_result = self._perform_osh_validation(
                measurements, coherence_before, coherence_after, entropy_before, entropy_after
            )
            
            execution_time = time.time() - start_time
            logger.info(f"OSH metric validation completed in {execution_time:.3f}s")
            logger.info(f"Overall OSH validation score: {validation_result.overall_score:.3f}")
            
            return StatisticalResult(
                analysis_type=AnalysisType.OSH_METRIC_VALIDATION,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics={'osh_validation': validation_result.to_dict()}
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"OSH metric validation failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"OSH metric validation failed: {str(e)}")
    
    @log_function_entry
    def _perform_osh_validation(
        self, 
        measurements: List[MeasurementResult],
        coherence_before: List[float],
        coherence_after: List[float],
        entropy_before: List[float],
        entropy_after: List[float]
    ) -> OSHValidationResult:
        """Perform comprehensive OSH validation."""
        logger.debug("Performing comprehensive OSH validation")
        
        try:
            # Coherence stability analysis
            coherence_stability_score = 0.0
            if coherence_before and coherence_after and len(coherence_before) == len(coherence_after):
                logger.debug("Analyzing coherence stability")
                coherence_ratios = [ca / max(cb, 1e-10) for cb, ca in zip(coherence_before, coherence_after)]
                coherence_stability_score = np.mean([min(1.0, ratio) for ratio in coherence_ratios])
                logger.debug(f"Coherence stability score: {coherence_stability_score:.4f}")
            else:
                logger.debug("Insufficient coherence data for stability analysis")
            
            # Entropy consistency analysis
            entropy_consistency_score = 0.0
            if entropy_before and entropy_after and len(entropy_before) == len(entropy_after):
                logger.debug("Analyzing entropy consistency")
                entropy_changes = [abs(ea - eb) for eb, ea in zip(entropy_before, entropy_after)]
                entropy_consistency_score = 1.0 - min(1.0, np.mean(entropy_changes))
                logger.debug(f"Entropy consistency score: {entropy_consistency_score:.4f}")
            else:
                logger.debug("Insufficient entropy data for consistency analysis")
            
            # RSP validation (placeholder - would need actual RSP calculations)
            rsp_validation_score = 0.5  # Default neutral score
            logger.debug(f"RSP validation score (placeholder): {rsp_validation_score:.4f}")
            
            # Consciousness emergence analysis
            consciousness_emergence_score = 0.0
            if coherence_before and entropy_after:
                logger.debug("Analyzing consciousness emergence indicators")
                consciousness_indicators = []
                for i, m in enumerate(measurements):
                    if m.coherence_before is not None and m.entropy_after is not None:
                        consciousness_indicator = m.coherence_before * (1.0 - min(1.0, m.entropy_after))
                        consciousness_indicators.append(consciousness_indicator)
                
                if consciousness_indicators:
                    consciousness_emergence_score = np.mean(consciousness_indicators)
                    logger.debug(f"Consciousness emergence score: {consciousness_emergence_score:.4f}")
            else:
                logger.debug("Insufficient data for consciousness emergence analysis")
            
            # Recursive depth analysis
            recursive_depths = [getattr(m, 'recursive_depth', 0) for m in measurements]
            recursive_depth_analysis = {
                'max_depth': max(recursive_depths),
                'mean_depth': np.mean(recursive_depths),
                'depth_distribution': dict(Counter(recursive_depths))
            }
            logger.debug(f"Recursive depth analysis: {recursive_depth_analysis}")
            
            # Observer consensus analysis
            observers = [m.observer for m in measurements if m.observer]
            observer_consensus_analysis = {
                'unique_observers': len(set(observers)),
                'total_observations': len(observers),
                'observer_distribution': dict(Counter(observers))
            }
            logger.debug(f"Observer consensus analysis: {observer_consensus_analysis}")
            
            # Detect anomaly flags
            anomaly_flags = []
            if coherence_stability_score < 0.3:
                anomaly_flags.append("Low coherence stability")
                logger.warning("Low coherence stability detected")
            if entropy_consistency_score < 0.3:
                anomaly_flags.append("High entropy inconsistency")
                logger.warning("High entropy inconsistency detected")
            if consciousness_emergence_score < 0.1:
                anomaly_flags.append("Low consciousness emergence indicators")
                logger.warning("Low consciousness emergence indicators detected")
            
            # Overall validation score
            scores = [coherence_stability_score, entropy_consistency_score, rsp_validation_score, consciousness_emergence_score]
            valid_scores = [s for s in scores if s is not None]
            overall_score = np.mean(valid_scores) if valid_scores else 0.0
            
            # Validation confidence based on data quality
            validation_confidence = min(1.0, len(measurements) / 10.0)  # Higher confidence with more data
            
            logger.debug(f"Overall OSH validation score: {overall_score:.4f}")
            logger.debug(f"Validation confidence: {validation_confidence:.4f}")
            logger.debug(f"Anomaly flags: {anomaly_flags}")
            
            return OSHValidationResult(
                overall_score=float(overall_score),
                coherence_stability_score=float(coherence_stability_score),
                entropy_consistency_score=float(entropy_consistency_score),
                rsp_validation_score=float(rsp_validation_score),
                consciousness_emergence_score=float(consciousness_emergence_score),
                recursive_depth_analysis=recursive_depth_analysis,
                observer_consensus_analysis=observer_consensus_analysis,
                anomaly_flags=anomaly_flags,
                validation_confidence=float(validation_confidence)
            )
        
        except Exception as e:
            logger.error(f"OSH validation failed: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise
    
    @log_function_entry
    def _analyze_measurement_quality(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Analyze the quality of measurements."""
        logger.info("Starting measurement quality analysis")
        start_time = time.time()
        
        try:
            quality_metrics = {}
            
            # Probability distribution quality
            logger.debug("Analyzing probability distribution quality")
            probability_entropies = [self._calculate_shannon_entropy(list(m.probabilities.values())) for m in measurements]
            quality_metrics['probability_entropy'] = {
                'mean': float(np.mean(probability_entropies)),
                'std': float(np.std(probability_entropies)),
                'min': float(np.min(probability_entropies)),
                'max': float(np.max(probability_entropies))
            }
            logger.debug(f"Probability entropy stats: {quality_metrics['probability_entropy']}")
            
            # Measurement completeness
            logger.debug("Analyzing measurement completeness")
            coherence_completeness = sum(1 for m in measurements if m.coherence_before is not None) / len(measurements)
            entropy_completeness = sum(1 for m in measurements if m.entropy_after is not None) / len(measurements)
            
            quality_metrics['data_completeness'] = {
                'coherence_before': float(coherence_completeness),
                'entropy_after': float(entropy_completeness),
                'overall': float((coherence_completeness + entropy_completeness) / 2)
            }
            logger.debug(f"Data completeness: {quality_metrics['data_completeness']}")
            
            # Measurement consistency
            logger.debug("Analyzing measurement consistency")
            outcome_consistency = len(set(m.outcome for m in measurements)) / len(measurements)
            basis_consistency = len(set(m.basis for m in measurements)) / len(measurements)
            
            quality_metrics['consistency'] = {
                'outcome_diversity': float(outcome_consistency),
                'basis_diversity': float(basis_consistency)
            }
            logger.debug(f"Consistency metrics: {quality_metrics['consistency']}")
            
            # Temporal quality
            if len(measurements) > 1:
                logger.debug("Analyzing temporal quality")
                timestamps = [m.timestamp for m in measurements]
                intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
                
                quality_metrics['temporal_quality'] = {
                    'measurement_rate': float(len(measurements) / (timestamps[-1] - timestamps[0])),
                    'interval_coefficient_of_variation': float(interval_cv),
                    'temporal_regularity': float(1.0 / (1.0 + interval_cv))
                }
                logger.debug(f"Temporal quality: {quality_metrics['temporal_quality']}")
            
            # Overall quality score
            logger.debug("Computing overall quality score")
            quality_components = [
                quality_metrics['data_completeness']['overall'],
                1.0 - quality_metrics['consistency']['outcome_diversity'],  # Lower diversity = higher consistency
                quality_metrics.get('temporal_quality', {}).get('temporal_regularity', 0.5)
            ]
            
            overall_quality = np.mean(quality_components)
            quality_metrics['overall_quality_score'] = float(overall_quality)
            logger.debug(f"Overall quality score: {overall_quality:.4f}")
            
            execution_time = time.time() - start_time
            logger.info(f"Measurement quality analysis completed in {execution_time:.3f}s")
            
            return StatisticalResult(
                analysis_type=AnalysisType.MEASUREMENT_QUALITY,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=quality_metrics
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Measurement quality analysis failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Measurement quality analysis failed: {str(e)}")
    
    @log_function_entry
    def _analyze_consensus(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Analyze consensus patterns in measurements."""
        logger.info("Starting consensus analysis")
        start_time = time.time()
        
        try:
            # Group by observer
            logger.debug("Grouping measurements by observer")
            observer_measurements = defaultdict(list)
            for m in measurements:
                if m.observer:
                    observer_measurements[m.observer].append(m)
            
            logger.debug(f"Found {len(observer_measurements)} unique observers")
            for obs, meas in observer_measurements.items():
                logger.debug(f"  {obs}: {len(meas)} measurements")
            
            consensus_analysis = {}
            
            if len(observer_measurements) >= 2:
                logger.debug("Analyzing consensus between multiple observers")
                # Analyze outcome consensus between observers
                observer_outcomes = {obs: [m.outcome for m in meas] for obs, meas in observer_measurements.items()}
                
                # Pairwise agreement
                observer_pairs = []
                observers = list(observer_outcomes.keys())
                
                for i, obs1 in enumerate(observers):
                    for obs2 in observers[i+1:]:
                        outcomes1 = observer_outcomes[obs1]
                        outcomes2 = observer_outcomes[obs2]
                        
                        # Find common measurement indices (if any)
                        common_outcomes = []
                        min_length = min(len(outcomes1), len(outcomes2))
                        
                        for j in range(min_length):
                            if outcomes1[j] == outcomes2[j]:
                                common_outcomes.append(1)
                            else:
                                common_outcomes.append(0)
                        
                        if common_outcomes:
                            agreement_rate = np.mean(common_outcomes)
                            observer_pairs.append({
                                'observer1': obs1,
                                'observer2': obs2,
                                'agreement_rate': float(agreement_rate),
                                'common_measurements': len(common_outcomes)
                            })
                            logger.debug(f"Agreement between {obs1} and {obs2}: {agreement_rate:.3f}")
                
                consensus_analysis['pairwise_agreements'] = observer_pairs
                
                if observer_pairs:
                    mean_agreement = np.mean([pair['agreement_rate'] for pair in observer_pairs])
                    consensus_analysis['mean_agreement_rate'] = float(mean_agreement)
                    consensus_analysis['consensus_strength'] = float(mean_agreement)
                    logger.info(f"Mean observer agreement rate: {mean_agreement:.3f}")
                else:
                    consensus_analysis['consensus_strength'] = 0.0
                    logger.warning("No pairwise agreements could be computed")
            
            else:
                consensus_analysis['consensus_strength'] = 1.0  # Single observer = perfect consensus
                consensus_analysis['note'] = "Single observer or insufficient multi-observer data"
                logger.debug("Single observer detected - perfect consensus assumed")
            
            # Outcome distribution analysis
            logger.debug("Analyzing outcome distribution")
            all_outcomes = [m.outcome for m in measurements]
            outcome_counts = Counter(all_outcomes)
            most_common_outcome = outcome_counts.most_common(1)[0]
            
            consensus_analysis['outcome_distribution'] = dict(outcome_counts)
            consensus_analysis['most_common_outcome'] = most_common_outcome[0]
            consensus_analysis['most_common_frequency'] = float(most_common_outcome[1] / len(measurements))
            
            logger.debug(f"Most common outcome: {most_common_outcome[0]} ({consensus_analysis['most_common_frequency']:.3f})")
            
            execution_time = time.time() - start_time
            logger.info(f"Consensus analysis completed in {execution_time:.3f}s")
            logger.info(f"Consensus strength: {consensus_analysis.get('consensus_strength', 'N/A')}")
            
            return StatisticalResult(
                analysis_type=AnalysisType.CONSENSUS_ANALYSIS,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=consensus_analysis
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Consensus analysis failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Consensus analysis failed: {str(e)}")
    
    @log_function_entry
    def _classify_patterns(self, measurements: List[MeasurementResult], **kwargs) -> StatisticalResult:
        """Classify patterns in measurement data."""
        logger.info("Starting pattern classification")
        start_time = time.time()
        
        try:
            patterns = {}
            
            # Temporal patterns
            if len(measurements) >= 5:
                logger.debug("Analyzing temporal patterns")
                # Sort by timestamp
                sorted_measurements = sorted(measurements, key=lambda m: m.timestamp)
                probabilities = [max(m.probabilities.values()) for m in sorted_measurements]
                
                # Identify trend patterns
                if len(probabilities) >= 3:
                    logger.debug("Computing trend patterns")
                    # Simple trend classification
                    first_third = np.mean(probabilities[:len(probabilities)//3])
                    last_third = np.mean(probabilities[-len(probabilities)//3:])
                    
                    if last_third > first_third * 1.1:
                        temporal_pattern = "increasing"
                    elif last_third < first_third * 0.9:
                        temporal_pattern = "decreasing"
                    else:
                        temporal_pattern = "stable"
                    
                    patterns['temporal_trend'] = temporal_pattern
                    patterns['trend_strength'] = float(abs(last_third - first_third) / first_third)
                    logger.debug(f"Temporal trend: {temporal_pattern}, strength: {patterns['trend_strength']:.3f}")
            
            # Oscillation patterns
            if len(measurements) >= 6:
                logger.debug("Analyzing oscillation patterns")
                probabilities = [max(m.probabilities.values()) for m in sorted(measurements, key=lambda m: m.timestamp)]
                
                # Simple oscillation detection
                direction_changes = 0
                for i in range(1, len(probabilities) - 1):
                    if ((probabilities[i] > probabilities[i-1] and probabilities[i] > probabilities[i+1]) or
                        (probabilities[i] < probabilities[i-1] and probabilities[i] < probabilities[i+1])):
                        direction_changes += 1
                
                oscillation_score = direction_changes / max(1, len(probabilities) - 2)
                patterns['oscillation_score'] = float(oscillation_score)
                
                if oscillation_score > 0.3:
                    patterns['oscillation_pattern'] = "high"
                elif oscillation_score > 0.1:
                    patterns['oscillation_pattern'] = "moderate"
                else:
                    patterns['oscillation_pattern'] = "low"
                
                logger.debug(f"Oscillation pattern: {patterns['oscillation_pattern']}, score: {oscillation_score:.3f}")
            
            # Clustering patterns
            if len(measurements) >= 10:
                logger.debug("Analyzing clustering patterns")
                # Cluster measurements by outcome probability
                probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
                
                # Simple clustering: high, medium, low probability groups
                high_prob = [p for p in probabilities if p > 0.7]
                medium_prob = [p for p in probabilities if 0.3 <= p <= 0.7]
                low_prob = [p for p in probabilities if p < 0.3]
                
                dominant_cluster = max([('high', len(high_prob)), ('medium', len(medium_prob)), ('low', len(low_prob))], key=lambda x: x[1])[0]
                
                patterns['probability_clusters'] = {
                    'high_probability_count': len(high_prob),
                    'medium_probability_count': len(medium_prob),
                    'low_probability_count': len(low_prob),
                    'dominant_cluster': dominant_cluster
                }
                
                logger.debug(f"Clustering: high={len(high_prob)}, medium={len(medium_prob)}, low={len(low_prob)}")
                logger.debug(f"Dominant cluster: {dominant_cluster}")
            
            # Anomaly patterns
            logger.debug("Analyzing anomaly patterns")
            probabilities = [max(m.probabilities.values()) if m.probabilities else 0.0 for m in measurements]
            if probabilities:
                prob_mean = np.mean(probabilities)
                prob_std = np.std(probabilities)
                
                if prob_std > 0:
                    anomaly_count = sum(1 for p in probabilities if abs(p - prob_mean) > 2 * prob_std)
                    anomaly_rate = anomaly_count / len(probabilities)
                    patterns['anomaly_pattern'] = {
                        'anomaly_count': anomaly_count,
                        'anomaly_rate': float(anomaly_rate),
                        'anomaly_severity': 'high' if anomaly_rate > 0.1 else 'low'
                    }
                    logger.debug(f"Anomaly pattern: {anomaly_count} anomalies ({anomaly_rate:.3f} rate)")
            
            execution_time = time.time() - start_time
            logger.info(f"Pattern classification completed in {execution_time:.3f}s")
            logger.debug(f"Identified {len(patterns)} pattern categories")
            
            return StatisticalResult(
                analysis_type=AnalysisType.PATTERN_CLASSIFICATION,
                success=True,
                timestamp=time.time(),
                execution_time=execution_time,
                sample_size=len(measurements),
                statistics=patterns
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pattern classification failed after {execution_time:.3f}s: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"Pattern classification failed: {str(e)}")
    
    # Utility methods
    
    @log_function_entry
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data."""
        logger.debug(f"Calculating skewness for {len(data)} data points")
        
        try:
            if SCIPY_AVAILABLE:
                skewness = float(stats.skew(data))
                logger.debug(f"Skewness (SciPy): {skewness:.4f}")
                return skewness
            else:
                # Manual calculation
                data_array = np.array(data)
                mean = np.mean(data_array)
                std = np.std(data_array)
                
                if std == 0:
                    logger.debug("Skewness: 0 (zero standard deviation)")
                    return 0.0
                
                skewness = np.mean(((data_array - mean) / std) ** 3)
                logger.debug(f"Skewness (manual): {skewness:.4f}")
                return float(skewness)
        
        except Exception as e:
            logger.warning(f"Skewness calculation failed: {str(e)}")
            return 0.0
    
    @log_function_entry
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data."""
        logger.debug(f"Calculating kurtosis for {len(data)} data points")
        
        try:
            if SCIPY_AVAILABLE:
                kurtosis = float(stats.kurtosis(data))
                logger.debug(f"Kurtosis (SciPy): {kurtosis:.4f}")
                return kurtosis
            else:
                # Manual calculation
                data_array = np.array(data)
                mean = np.mean(data_array)
                std = np.std(data_array)
                
                if std == 0:
                    logger.debug("Kurtosis: 0 (zero standard deviation)")
                    return 0.0
                
                kurtosis = np.mean(((data_array - mean) / std) ** 4) - 3
                logger.debug(f"Kurtosis (manual): {kurtosis:.4f}")
                return float(kurtosis)
        
        except Exception as e:
            logger.warning(f"Kurtosis calculation failed: {str(e)}")
            return 0.0
    
    @log_function_entry
    def _calculate_shannon_entropy(self, values: List[float]) -> float:
        """Calculate Shannon entropy of a probability distribution."""
        logger.debug(f"Calculating Shannon entropy for {len(values)} values")
        
        try:
            if not values or sum(values) == 0:
                logger.debug("Shannon entropy: 0 (empty or zero values)")
                return 0.0
            
            # Normalize to probabilities
            total = sum(values)
            probabilities = [v / total for v in values]
            logger.debug(f"Normalized to probabilities, total: {total}")
            
            # Calculate entropy
            entropy = 0.0
            for p in probabilities:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            logger.debug(f"Shannon entropy: {entropy:.4f}")
            return entropy
        
        except Exception as e:
            logger.warning(f"Shannon entropy calculation failed: {str(e)}")
            return 0.0
    
    @log_function_entry
    def clear_cache(self):
        """Clear analysis cache."""
        logger.debug("Clearing analysis cache")
        
        try:
            with self.cache_lock:
                cache_size = len(self.analysis_cache)
                self.analysis_cache.clear()
                logger.info(f"Cleared {cache_size} items from analysis cache")
                
                # Update cache statistics
                self.analysis_stats['cache_hits'] = 0
                self.analysis_stats['cache_misses'] = 0
        
        except Exception as e:
            logger.error(f"Failed to clear cache: {str(e)}")
    
    @log_function_entry
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        logger.debug("Generating performance statistics")
        
        try:
            total_requests = self.analysis_stats['cache_hits'] + self.analysis_stats['cache_misses']
            hit_rate = self.analysis_stats['cache_hits'] / max(1, total_requests)
            
            avg_execution_time = 0.0
            if self.analysis_stats['execution_times']:
                avg_execution_time = np.mean(self.analysis_stats['execution_times'])
            
            stats = {
                'total_analyses': self.analysis_stats['total_analyses'],
                'cache_hit_rate': hit_rate,
                'cache_hits': self.analysis_stats['cache_hits'],
                'cache_misses': self.analysis_stats['cache_misses'],
                'average_execution_time': avg_execution_time,
                'error_count': self.analysis_stats['error_count'],
                'scipy_available': SCIPY_AVAILABLE,
                'analysis_type_counts': dict(self.analysis_stats['analysis_type_counts']),
                'last_analysis_time': self.analysis_stats['last_analysis_time'],
                'execution_time_std': float(np.std(self.analysis_stats['execution_times'])) if self.analysis_stats['execution_times'] else 0.0,
                'min_execution_time': float(np.min(self.analysis_stats['execution_times'])) if self.analysis_stats['execution_times'] else 0.0,
                'max_execution_time': float(np.max(self.analysis_stats['execution_times'])) if self.analysis_stats['execution_times'] else 0.0,
                'config': {
                    'confidence_level': self.config.confidence_level,
                    'significance_level': self.config.significance_level,
                    'bootstrap_iterations': self.config.bootstrap_iterations,
                    'monte_carlo_iterations': self.config.monte_carlo_iterations,
                    'anomaly_threshold': self.config.anomaly_threshold,
                    'correlation_threshold': self.config.correlation_threshold,
                    'enable_caching': self.config.enable_caching,
                    'enable_parallel': self.config.enable_parallel,
                    'max_workers': self.config.max_workers,
                    'timeout': self.config.timeout
                }
            }
            
            logger.debug("Performance statistics generated successfully")
            logger.debug(f"  Total analyses: {stats['total_analyses']}")
            logger.debug(f"  Cache hit rate: {stats['cache_hit_rate']:.3f}")
            logger.debug(f"  Average execution time: {stats['average_execution_time']:.3f}s")
            logger.debug(f"  Error count: {stats['error_count']}")
            
            return stats
        
        except Exception as e:
            logger.error(f"Failed to generate performance statistics: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return {
                'error': f"Failed to generate stats: {str(e)}",
                'total_analyses': self.analysis_stats.get('total_analyses', 0),
                'error_count': self.analysis_stats.get('error_count', 0)
            }
    
    @log_function_entry
    def analyze_system_health(self, metrics: ComprehensiveMetrics) -> SystemHealthProfile:
        """Analyze system health based on comprehensive metrics."""
        logger.info("Starting system health analysis")
        
        try:
            logger.debug(f"Input metrics timestamp: {metrics.timestamp}")
            logger.debug(f"Core OSH metrics: coherence={metrics.coherence:.3f}, entropy={metrics.entropy:.3f}, strain={metrics.strain:.3f}")
            
            # Component health analysis
            component_health = {}
            
            # Quantum system health
            quantum_health = min(1.0, (metrics.quantum_fidelity + (1.0 - metrics.entropy)) / 2)
            component_health['quantum_system'] = quantum_health
            logger.debug(f"Quantum system health: {quantum_health:.3f}")
            
            # Observer system health
            observer_health = 0.0
            if metrics.observer_count > 0:
                observer_health = min(1.0, (metrics.active_observers / metrics.observer_count) * metrics.observer_consensus)
            component_health['observer_system'] = observer_health
            logger.debug(f"Observer system health: {observer_health:.3f}")
            
            # Memory system health
            memory_health = 1.0 - min(1.0, metrics.memory_strain_max)
            component_health['memory_system'] = memory_health
            logger.debug(f"Memory system health: {memory_health:.3f}")
            
            # Field system health
            field_health = min(1.0, metrics.coherence * (1.0 - metrics.strain))
            component_health['field_system'] = field_health
            logger.debug(f"Field system health: {field_health:.3f}")
            
            # Performance metrics
            performance_metrics = {
                'render_fps': metrics.render_fps,
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'simulation_efficiency': metrics.execution_steps / max(1, metrics.simulation_time) if metrics.simulation_time > 0 else 0.0
            }
            logger.debug(f"Performance metrics: {performance_metrics}")
            
            # Resource utilization
            resource_utilization = {
                'memory_utilization': min(1.0, metrics.memory_usage_mb / 8192),  # Assume 8GB max
                'cpu_utilization': min(1.0, metrics.cpu_usage_percent / 100),
                'quantum_resource_utilization': min(1.0, metrics.total_qubits / 1000),  # Assume 1000 qubit max
                'field_resource_utilization': min(1.0, metrics.field_count / 100)  # Assume 100 field max
            }
            logger.debug(f"Resource utilization: {resource_utilization}")
            
            # Stability indicators
            stability_indicators = {
                'coherence_stability': 1.0 - abs(0.5 - metrics.coherence) * 2,  # Closer to 0.5 is more stable
                'entropy_stability': 1.0 - metrics.entropy,
                'strain_stability': 1.0 - metrics.strain,
                'rsp_stability': min(1.0, metrics.rsp / 10.0)  # Normalize RSP
            }
            logger.debug(f"Stability indicators: {stability_indicators}")
            
            # Generate alerts
            alerts = []
            critical_issues = []
            recommendations = []
            
            if metrics.coherence < 0.1:
                critical_issues.append("Critical: Very low coherence detected")
                recommendations.append("Increase coherence stabilization measures")
            elif metrics.coherence < 0.3:
                alerts.append("Warning: Low coherence levels")
                recommendations.append("Monitor coherence trends closely")
            
            if metrics.entropy > 0.9:
                critical_issues.append("Critical: Very high entropy detected")
                recommendations.append("Implement entropy reduction protocols")
            elif metrics.entropy > 0.7:
                alerts.append("Warning: High entropy levels")
                recommendations.append("Consider entropy management strategies")
            
            if metrics.strain > 0.8:
                critical_issues.append("Critical: High system strain detected")
                recommendations.append("Reduce system load and implement strain relief")
            elif metrics.strain > 0.6:
                alerts.append("Warning: Elevated system strain")
                recommendations.append("Monitor strain levels and optimize resource usage")
            
            if metrics.memory_strain_max > 0.9:
                critical_issues.append("Critical: Memory system under severe strain")
                recommendations.append("Perform immediate memory defragmentation")
            
            if metrics.render_fps < 10:
                alerts.append("Warning: Low rendering performance")
                recommendations.append("Optimize rendering pipeline or reduce visual complexity")
            
            if metrics.cpu_usage_percent > 90:
                alerts.append("Warning: High CPU usage")
                recommendations.append("Optimize computational load distribution")
            
            # Overall health calculation
            health_components = list(component_health.values())
            overall_health = np.mean(health_components) if health_components else 0.0
            
            # Health trend analysis (simplified)
            health_trend = "stable"
            if overall_health > 0.8:
                health_trend = "excellent"
            elif overall_health > 0.6:
                health_trend = "good"
            elif overall_health > 0.4:
                health_trend = "fair"
            elif overall_health > 0.2:
                health_trend = "poor"
            else:
                health_trend = "critical"
            
            # Predictive alerts (basic)
            predictive_alerts = []
            if metrics.coherence < 0.5 and metrics.entropy > 0.5:
                predictive_alerts.append("Prediction: System may experience coherence collapse")
            if metrics.strain > 0.5 and len(metrics.emergent_phenomena) == 0:
                predictive_alerts.append("Prediction: Strain buildup may inhibit emergence")
            if metrics.memory_strain_avg > 0.6:
                predictive_alerts.append("Prediction: Memory fragmentation likely to increase")
            
            health_profile = SystemHealthProfile(
                overall_health=overall_health,
                component_health=component_health,
                performance_metrics=performance_metrics,
                resource_utilization=resource_utilization,
                stability_indicators=stability_indicators,
                alerts=alerts,
                recommendations=recommendations,
                critical_issues=critical_issues,
                health_trend=health_trend,
                predictive_alerts=predictive_alerts,
                timestamp=datetime.datetime.now()
            )
            
            logger.info(f"System health analysis completed:")
            logger.info(f"  Overall health: {overall_health:.3f} ({health_trend})")
            logger.info(f"  Critical issues: {len(critical_issues)}")
            logger.info(f"  Alerts: {len(alerts)}")
            logger.info(f"  Recommendations: {len(recommendations)}")
            
            return health_profile
        
        except Exception as e:
            logger.error(f"System health analysis failed: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            raise AnalysisExecutionError(f"System health analysis failed: {str(e)}")
    
    @log_function_entry
    def validate_measurement_data(self, measurements: List[MeasurementResult]) -> Dict[str, Any]:
        """Validate measurement data for analysis readiness."""
        logger.info(f"Validating {len(measurements)} measurements for analysis")
        
        try:
            validation_results = {
                'is_valid': True,
                'warnings': [],
                'errors': [],
                'statistics': {},
                'recommendations': []
            }
            
            # Basic validation
            if not measurements:
                validation_results['is_valid'] = False
                validation_results['errors'].append("No measurements provided")
                return validation_results
            
            if len(measurements) < MIN_SAMPLE_SIZE:
                validation_results['is_valid'] = False
                validation_results['errors'].append(f"Insufficient sample size: {len(measurements)} < {MIN_SAMPLE_SIZE}")
            
            # Data completeness validation
            total_measurements = len(measurements)
            coherence_before_count = sum(1 for m in measurements if m.coherence_before is not None)
            coherence_after_count = sum(1 for m in measurements if m.coherence_after is not None)
            entropy_before_count = sum(1 for m in measurements if m.entropy_before is not None)
            entropy_after_count = sum(1 for m in measurements if m.entropy_after is not None)
            
            completeness_stats = {
                'total_measurements': total_measurements,
                'coherence_before_completeness': coherence_before_count / total_measurements,
                'coherence_after_completeness': coherence_after_count / total_measurements,
                'entropy_before_completeness': entropy_before_count / total_measurements,
                'entropy_after_completeness': entropy_after_count / total_measurements
            }
            
            validation_results['statistics'].update(completeness_stats)
            
            # Check for sufficient data completeness
            min_completeness = 0.5  # 50% threshold
            if completeness_stats['coherence_before_completeness'] < min_completeness:
                validation_results['warnings'].append(f"Low coherence_before completeness: {completeness_stats['coherence_before_completeness']:.1%}")
            
            if completeness_stats['entropy_after_completeness'] < min_completeness:
                validation_results['warnings'].append(f"Low entropy_after completeness: {completeness_stats['entropy_after_completeness']:.1%}")
            
            # Temporal validation
            timestamps = [m.timestamp for m in measurements]
            if len(set(timestamps)) != len(timestamps):
                validation_results['warnings'].append("Duplicate timestamps detected")
            
            time_range = max(timestamps) - min(timestamps)
            validation_results['statistics']['time_range'] = time_range
            
            if time_range == 0:
                validation_results['warnings'].append("All measurements have the same timestamp")
            elif time_range < 1.0:  # Less than 1 second
                validation_results['warnings'].append(f"Very short time range: {time_range:.3f}s")
            
            # Probability validation
            probability_issues = 0
            for i, m in enumerate(measurements):
                prob_sum = sum(m.probabilities.values())
                if abs(prob_sum - 1.0) > 1e-6:
                    probability_issues += 1
                
                if any(p < 0 or p > 1 for p in m.probabilities.values()):
                    validation_results['errors'].append(f"Invalid probability values in measurement {i}")
            
            if probability_issues > 0:
                validation_results['warnings'].append(f"Probability normalization issues in {probability_issues} measurements")
            
            # Outcome validation
            outcomes = [m.outcome for m in measurements]
            unique_outcomes = len(set(outcomes))
            validation_results['statistics']['unique_outcomes'] = unique_outcomes
            
            if unique_outcomes == 1:
                validation_results['warnings'].append("All measurements have the same outcome")
                validation_results['recommendations'].append("Consider increasing measurement diversity")
            
            # Observer validation
            observers = [m.observer for m in measurements if m.observer]
            if observers:
                unique_observers = len(set(observers))
                validation_results['statistics']['unique_observers'] = unique_observers
                validation_results['statistics']['observer_coverage'] = len(observers) / total_measurements
                
                if unique_observers == 1:
                    validation_results['warnings'].append("All measurements from single observer")
                    validation_results['recommendations'].append("Consider multi-observer measurements for consensus analysis")
            else:
                validation_results['warnings'].append("No observer information available")
            
            # Generate recommendations
            if len(measurements) < 100:
                validation_results['recommendations'].append("Consider collecting more measurements for robust statistical analysis")
            
            if completeness_stats['coherence_before_completeness'] < 0.8:
                validation_results['recommendations'].append("Improve coherence measurement completeness")
            
            if completeness_stats['entropy_after_completeness'] < 0.8:
                validation_results['recommendations'].append("Improve entropy measurement completeness")
            
            logger.info(f"Validation completed:")
            logger.info(f"  Valid: {validation_results['is_valid']}")
            logger.info(f"  Warnings: {len(validation_results['warnings'])}")
            logger.info(f"  Errors: {len(validation_results['errors'])}")
            logger.info(f"  Recommendations: {len(validation_results['recommendations'])}")
            
            return validation_results
        
        except Exception as e:
            logger.error(f"Measurement validation failed: {str(e)}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return {
                'is_valid': False,
                'errors': [f"Validation failed: {str(e)}"],
                'warnings': [],
                'statistics': {},
                'recommendations': []
            }


# Helper functions for standalone use

@log_function_entry
def quick_analysis(measurements: List[MeasurementResult], analysis_type: AnalysisType = AnalysisType.BASIC_STATISTICS) -> StatisticalResult:
    """Quick analysis function for standalone use."""
    logger.info(f"Running quick analysis: {analysis_type.value}")
    
    try:
        with StatisticalAnalysisEngine() as engine:
            results = engine.analyze_measurements(measurements, [analysis_type])
            return results[analysis_type]
    
    except Exception as e:
        logger.error(f"Quick analysis failed: {str(e)}")
        raise


@log_function_entry
def validate_measurements(measurements: List[MeasurementResult]) -> Dict[str, Any]:
    """Quick validation function for standalone use."""
    logger.info(f"Running quick validation for {len(measurements)} measurements")
    
    try:
        with StatisticalAnalysisEngine() as engine:
            return engine.validate_measurement_data(measurements)
    
    except Exception as e:
        logger.error(f"Quick validation failed: {str(e)}")
        raise


# Export key classes and functions
__all__ = [
    # Main engine class
    'StatisticalAnalysisEngine',
    
    # Configuration and result classes
    'StatisticalConfiguration',
    'StatisticalResult',
    'DistributionAnalysis',
    'OSHValidationResult',
    
    # Enums
    'AnalysisType',
    
    # Exceptions
    'StatisticalAnalysisError',
    'InsufficientDataError',
    'InvalidParameterError',
    'AnalysisExecutionError',
    
    # Helper functions
    'quick_analysis',
    'validate_measurements'
]

logger.info("Statistical Analysis Engine module loaded successfully")
logger.info("=" * 80)