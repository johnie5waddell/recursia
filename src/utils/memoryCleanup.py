"""
Memory Cleanup Utility
Provides functions to clean up memory and reset error states
"""

import gc
import logging
import psutil
import os
from typing import Dict, Any

from src.utils.errorThrottler import reset_throttling, get_throttling_stats


def get_memory_info() -> Dict[str, Any]:
    """Get current memory usage information."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,
        'vms_mb': memory_info.vms / 1024 / 1024,
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }


def force_garbage_collection() -> Dict[str, Any]:
    """Force garbage collection and return stats."""
    memory_before = get_memory_info()
    
    # Force collection
    collected = gc.collect()
    
    # Get stats after
    memory_after = get_memory_info()
    
    return {
        'objects_collected': collected,
        'memory_freed_mb': memory_before['rss_mb'] - memory_after['rss_mb'],
        'memory_before': memory_before,
        'memory_after': memory_after
    }


def reset_error_states():
    """Reset all error throttling states."""
    # Get stats before reset
    stats_before = get_throttling_stats()
    
    # Reset throttling
    reset_throttling()
    
    # Log the reset
    logger = logging.getLogger(__name__)
    logger.info(f"Reset error throttling. Previous stats: {stats_before}")
    
    return stats_before


def cleanup_physics_engine(physics_engine):
    """Clean up physics engine resources."""
    if hasattr(physics_engine, '_phenomena_error_count'):
        physics_engine._phenomena_error_count = 0
    
    # Clear error history if it exists
    if hasattr(physics_engine, 'error_history'):
        physics_engine.error_history.clear()
        
    # Clear performance data if it's large
    if hasattr(physics_engine, 'performance_data'):
        for key, data in physics_engine.performance_data.items():
            if isinstance(data, list) and len(data) > 1000:
                # Keep only last 100 entries
                physics_engine.performance_data[key] = data[-100:]


def cleanup_runtime(runtime):
    """Clean up runtime resources."""
    if hasattr(runtime, '_phenomena_error_count'):
        runtime._phenomena_error_count = 0
        
    # Clear large histories
    if hasattr(runtime, 'metrics_history'):
        if len(runtime.metrics_history) > 1000:
            # Keep only last 100 entries
            runtime.metrics_history = runtime.metrics_history[-100:]
            
    # Force GC on subsystems
    if hasattr(runtime, 'physics_engine'):
        cleanup_physics_engine(runtime.physics_engine)


def perform_full_cleanup(runtime=None, physics_engine=None) -> Dict[str, Any]:
    """
    Perform comprehensive memory cleanup.
    
    Args:
        runtime: Optional runtime instance to clean
        physics_engine: Optional physics engine instance to clean
        
    Returns:
        Cleanup statistics
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive memory cleanup")
    
    # Get initial state
    initial_memory = get_memory_info()
    initial_errors = get_throttling_stats()
    
    # Clean specific instances if provided
    if runtime:
        cleanup_runtime(runtime)
    if physics_engine:
        cleanup_physics_engine(physics_engine)
        
    # Reset error states
    reset_error_states()
    
    # Force garbage collection
    gc_stats = force_garbage_collection()
    
    # Get final state
    final_memory = get_memory_info()
    
    cleanup_stats = {
        'memory': {
            'initial_mb': initial_memory['rss_mb'],
            'final_mb': final_memory['rss_mb'],
            'freed_mb': initial_memory['rss_mb'] - final_memory['rss_mb'],
            'percent_freed': (initial_memory['rss_mb'] - final_memory['rss_mb']) / initial_memory['rss_mb'] * 100
        },
        'errors': {
            'total_errors_before': initial_errors['total_errors'],
            'error_types_before': initial_errors['error_types'],
            'suppressed_before': initial_errors['total_suppressed']
        },
        'gc': gc_stats
    }
    
    logger.info(f"Cleanup complete. Freed {cleanup_stats['memory']['freed_mb']:.2f} MB")
    
    return cleanup_stats


# Auto-cleanup on high memory usage
class MemoryWatcher:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, threshold_percent: float = 80.0):
        self.threshold_percent = threshold_percent
        self.logger = logging.getLogger(f"{__name__}.MemoryWatcher")
        
    def check_and_cleanup(self, runtime=None, physics_engine=None) -> bool:
        """
        Check memory usage and cleanup if above threshold.
        
        Returns:
            True if cleanup was performed
        """
        memory_info = get_memory_info()
        
        if memory_info['percent'] > self.threshold_percent:
            self.logger.warning(
                f"Memory usage at {memory_info['percent']:.1f}% "
                f"(threshold: {self.threshold_percent}%). Triggering cleanup."
            )
            
            stats = perform_full_cleanup(runtime, physics_engine)
            
            self.logger.info(
                f"Cleanup freed {stats['memory']['freed_mb']:.2f} MB. "
                f"Memory now at {stats['memory']['final_mb']:.2f} MB"
            )
            
            return True
            
        return False