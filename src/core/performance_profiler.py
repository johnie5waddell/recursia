"""
Performance Profiler for Recursia VM
====================================

Identifies and tracks performance bottlenecks in the quantum simulation.
"""

import time
import cProfile
import pstats
import io
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance measurement."""
    operation: str
    start_time: float
    end_time: float
    duration: float
    metadata: Dict[str, Any]


class PerformanceProfiler:
    """
    Tracks performance metrics and identifies bottlenecks.
    
    Features:
    - Operation timing
    - Bottleneck detection
    - Performance regression tracking
    - Automatic optimization suggestions
    """
    
    def __init__(self, target_exp_per_sec: float = 0.2):
        """
        Initialize profiler.
        
        Args:
            target_exp_per_sec: Target experiments per second
        """
        self.target_exp_per_sec = target_exp_per_sec
        self.target_ms_per_exp = 1000.0 / target_exp_per_sec
        
        # Metrics storage
        self.operation_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.current_operations: Dict[str, float] = {}
        self.total_experiments = 0
        self.start_time = time.time()
        
        # Bottleneck tracking
        self.bottlenecks: List[str] = []
        self.optimization_suggestions: List[str] = []
        
        # Profile data
        self.profiler = None
        self.is_profiling = False
        
    def start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Start timing an operation."""
        self.current_operations[operation] = time.time()
        
    def end_operation(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.current_operations:
            return 0.0
            
        start_time = self.current_operations.pop(operation)
        duration = time.time() - start_time
        
        # Store metric
        self.operation_times[operation].append(duration)
        
        # Check for bottleneck
        if duration > self.target_ms_per_exp / 1000.0:
            if operation not in self.bottlenecks:
                self.bottlenecks.append(operation)
                logger.warning(f"Performance bottleneck detected: {operation} took {duration*1000:.1f}ms")
                
        return duration
        
    def start_profiling(self):
        """Start detailed profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.is_profiling = True
        logger.info("Performance profiling started")
        
    def stop_profiling(self) -> str:
        """Stop profiling and return results."""
        if not self.is_profiling:
            return "Profiling not active"
            
        self.profiler.disable()
        self.is_profiling = False
        
        # Generate stats
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Top 20 functions
        
        return s.getvalue()
        
    def record_experiment(self):
        """Record completion of an experiment."""
        self.total_experiments += 1
        
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        elapsed_time = time.time() - self.start_time
        actual_exp_per_sec = self.total_experiments / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate operation statistics
        operation_stats = {}
        for operation, times in self.operation_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                total_time = sum(times)
                operation_stats[operation] = {
                    'average_ms': avg_time * 1000,
                    'max_ms': max_time * 1000,
                    'total_ms': total_time * 1000,
                    'count': len(times),
                    'percentage': (total_time / elapsed_time * 100) if elapsed_time > 0 else 0
                }
                
        # Sort by total time
        sorted_ops = sorted(
            operation_stats.items(),
            key=lambda x: x[1]['total_ms'],
            reverse=True
        )
        
        # Generate optimization suggestions
        self._generate_suggestions(sorted_ops, actual_exp_per_sec)
        
        return {
            'target_exp_per_sec': self.target_exp_per_sec,
            'actual_exp_per_sec': actual_exp_per_sec,
            'performance_ratio': actual_exp_per_sec / self.target_exp_per_sec,
            'total_experiments': self.total_experiments,
            'elapsed_time': elapsed_time,
            'bottlenecks': self.bottlenecks,
            'operation_stats': dict(sorted_ops[:10]),  # Top 10 operations
            'suggestions': self.optimization_suggestions
        }
        
    def _generate_suggestions(self, sorted_ops: List, actual_exp_per_sec: float):
        """Generate optimization suggestions based on profiling data."""
        self.optimization_suggestions.clear()
        
        # Check overall performance
        if actual_exp_per_sec < self.target_exp_per_sec * 0.5:
            self.optimization_suggestions.append(
                f"Performance is {actual_exp_per_sec/self.target_exp_per_sec:.1%} of target. "
                "Major optimizations needed."
            )
            
        # Check top operations
        if sorted_ops:
            top_op, top_stats = sorted_ops[0]
            if top_stats['percentage'] > 50:
                self.optimization_suggestions.append(
                    f"Operation '{top_op}' consumes {top_stats['percentage']:.1f}% of time. "
                    "Consider optimizing or caching."
                )
                
            # Specific suggestions
            for op, stats in sorted_ops[:5]:
                if 'iit' in op.lower() or 'phi' in op.lower():
                    if stats['average_ms'] > 100:
                        self.optimization_suggestions.append(
                            f"IIT/Phi calculation taking {stats['average_ms']:.1f}ms. "
                            "Consider using approximation for large systems."
                        )
                elif 'metric' in op.lower():
                    if stats['count'] > self.total_experiments * 10:
                        self.optimization_suggestions.append(
                            f"Metrics updated {stats['count']} times ({stats['count']/self.total_experiments:.1f} per experiment). "
                            "Consider batching or lazy evaluation."
                        )
                elif 'gate' in op.lower():
                    if stats['average_ms'] > 10:
                        self.optimization_suggestions.append(
                            f"Gate operations taking {stats['average_ms']:.1f}ms. "
                            "Consider optimizing quantum backend."
                        )
                        
    def print_report(self):
        """Print performance report to logger."""
        report = self.get_performance_report()
        
        logger.info("=== Performance Report ===")
        logger.info(f"Target: {report['target_exp_per_sec']:.2f} exp/s")
        logger.info(f"Actual: {report['actual_exp_per_sec']:.2f} exp/s "
                   f"({report['performance_ratio']:.1%} of target)")
        logger.info(f"Total experiments: {report['total_experiments']}")
        
        if report['bottlenecks']:
            logger.warning(f"Bottlenecks: {', '.join(report['bottlenecks'])}")
            
        logger.info("\nTop time-consuming operations:")
        for op, stats in list(report['operation_stats'].items())[:5]:
            logger.info(f"  {op}: {stats['total_ms']:.1f}ms total "
                       f"({stats['percentage']:.1f}%), "
                       f"{stats['average_ms']:.1f}ms avg")
                       
        if report['suggestions']:
            logger.info("\nOptimization suggestions:")
            for suggestion in report['suggestions']:
                logger.info(f"  - {suggestion}")


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_performance_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_operation(operation: str):
    """Decorator to profile function execution time."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            profiler = get_performance_profiler()
            profiler.start_operation(operation)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                profiler.end_operation(operation)
        return wrapper
    return decorator