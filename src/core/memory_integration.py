"""
Memory Management Integration for Recursia v3
============================================

Integrates the high-level memory management system with the existing
low-level memory pool manager for comprehensive memory optimization.
"""

import logging
from typing import Optional

from src.core.memory_manager import MemoryManager
from src.core.memory_management_system import (
    setup_memory_management,
    get_global_memory_manager,
    MemoryPolicy
)

logger = logging.getLogger(__name__)


def setup_complete_memory_management(runtime: 'RecursiaRuntime') -> None:
    """
    Set up both low-level and high-level memory management.
    
    This integrates:
    1. Low-level memory pool management (MemoryManager)
    2. High-level subsystem cleanup (GlobalMemoryManager)
    
    Args:
        runtime: The RecursiaRuntime instance
    """
    logger.info("Setting up complete memory management system")
    
    # The runtime already has low-level memory management (MemoryManager)
    # which handles memory pools and block allocation
    
    # Now set up high-level memory management for subsystem cleanup
    setup_memory_management(runtime)
    
    # Configure memory policies based on runtime config
    manager = get_global_memory_manager()
    
    # Adjust policies based on runtime configuration
    if hasattr(runtime, 'config'):
        # If running in production mode, be more aggressive with cleanup
        if runtime.config.get('production_mode', False):
            logger.info("Configuring aggressive memory cleanup for production")
            
            # Update quantum states policy
            manager._policies['quantum_states'] = MemoryPolicy(
                max_items=500,  # Reduced from 1000
                max_memory_mb=128,  # Reduced from 256
                cleanup_interval=30.0,  # More frequent cleanup
                cleanup_age_seconds=180.0  # 3 minutes instead of 5
            )
            
            # Update observers policy
            manager._policies['observers'] = MemoryPolicy(
                max_items=200,  # Reduced from 500
                max_memory_mb=64,  # Reduced from 128
                cleanup_interval=30.0,
                cleanup_age_seconds=120.0  # 2 minutes
            )
            
            # Update measurements policy
            manager._policies['measurements'] = MemoryPolicy(
                max_items=5000,  # Reduced from 10000
                max_memory_mb=256,  # Reduced from 512
                cleanup_interval=20.0,  # Very frequent cleanup
                cleanup_age_seconds=60.0  # Only keep 1 minute of data
            )
    
    logger.info("Complete memory management system initialized")


def get_memory_statistics(runtime: 'RecursiaRuntime') -> dict:
    """
    Get comprehensive memory statistics from both systems.
    
    Returns:
        dict: Combined memory statistics
    """
    stats = {}
    
    # Get low-level memory pool statistics
    if hasattr(runtime, 'memory_manager') and runtime.memory_manager:
        pool_stats = {}
        for pool_name, pool in runtime.memory_manager.memory_pools.items():
            pool_stats[pool_name] = pool.get_stats()
        
        stats['memory_pools'] = pool_stats
        stats['total_allocated'] = runtime.memory_manager.total_allocated_memory
        stats['gc_runs'] = runtime.memory_manager.gc_runs
    
    # Get high-level cleanup statistics
    manager = get_global_memory_manager()
    high_level_stats = manager.get_statistics()
    
    stats['cleanup_stats'] = {
        'total_cleanups': high_level_stats['total_cleanups'],
        'emergency_cleanups': high_level_stats['emergency_cleanups'],
        'items_cleaned': high_level_stats['items_cleaned'],
        'last_cleanup_time': high_level_stats['last_cleanup_time']
    }
    
    stats['component_stats'] = high_level_stats['components']
    stats['system_memory'] = high_level_stats['system_memory']
    
    return stats


def optimize_memory_for_long_simulations(runtime: 'RecursiaRuntime') -> None:
    """
    Configure memory management for long-running simulations.
    
    This adjusts policies to prevent memory growth over time.
    """
    logger.info("Optimizing memory management for long simulations")
    
    manager = get_global_memory_manager()
    
    # Very aggressive cleanup for long runs
    long_run_policy = MemoryPolicy(
        max_items=100,
        max_memory_mb=50,
        cleanup_interval=10.0,  # Cleanup every 10 seconds
        cleanup_age_seconds=30.0,  # Only keep 30 seconds of data
        high_water_mark=0.5  # Cleanup at 50% usage
    )
    
    # Apply to all components
    for component_name in manager._components:
        manager._policies[component_name] = long_run_policy
    
    # Also configure low-level memory manager
    if hasattr(runtime, 'memory_manager') and runtime.memory_manager:
        # Enable aggressive garbage collection
        runtime.memory_manager.gc_interval = 5.0  # GC every 5 seconds
        runtime.memory_manager.garbage_collection_threshold = 100  # GC after 100 allocations
    
    logger.info("Memory optimization for long simulations complete")