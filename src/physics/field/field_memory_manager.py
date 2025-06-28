"""
Field Dynamics Memory Manager
============================

Provides automatic memory management for FieldDynamics to prevent memory leaks
during long-running simulations.
"""

import threading
import time
import gc
import logging
from typing import Optional, Dict, Any
from collections import deque

logger = logging.getLogger(__name__)


class FieldMemoryManager:
    """
    Manages memory usage for field dynamics to prevent leaks.
    
    Features:
    - Periodic garbage collection
    - Automatic cleanup of old field data
    - Memory usage monitoring
    - Adaptive cleanup based on memory pressure
    """
    
    def __init__(self, 
                 max_memory_mb: float = 1024,
                 cleanup_interval: float = 60.0,
                 high_water_mark: float = 0.8):
        """
        Initialize memory manager.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            cleanup_interval: Seconds between cleanup cycles
            high_water_mark: Fraction of max memory to trigger aggressive cleanup
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.cleanup_interval = cleanup_interval
        self.high_water_mark = high_water_mark
        
        # Monitoring
        self.memory_history = deque(maxlen=100)
        self.cleanup_count = 0
        self.last_cleanup_time = time.time()
        
        # Thread control
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
    def start(self):
        """Start the memory management thread."""
        with self._lock:
            if not self._running:
                self._running = True
                self._thread = threading.Thread(
                    target=self._memory_monitor_loop,
                    name="FieldMemoryManager",
                    daemon=True
                )
                self._thread.start()
                logger.info("Field memory manager started")
                
    def stop(self):
        """Stop the memory management thread."""
        with self._lock:
            if self._running:
                self._running = False
                if self._thread:
                    self._thread.join(timeout=5.0)
                logger.info("Field memory manager stopped")
                
    def _memory_monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Check memory usage
                memory_usage = self._get_memory_usage()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'usage_bytes': memory_usage,
                    'usage_fraction': memory_usage / self.max_memory_bytes
                })
                
                # Determine if cleanup is needed
                time_since_cleanup = time.time() - self.last_cleanup_time
                memory_fraction = memory_usage / self.max_memory_bytes
                
                needs_cleanup = (
                    time_since_cleanup >= self.cleanup_interval or
                    memory_fraction >= self.high_water_mark
                )
                
                if needs_cleanup:
                    self._perform_cleanup(memory_fraction)
                    self.last_cleanup_time = time.time()
                    self.cleanup_count += 1
                    
                # Sleep before next check
                time.sleep(min(10.0, self.cleanup_interval / 6))
                
            except Exception as e:
                logger.error(f"Error in memory monitor loop: {e}")
                time.sleep(10.0)
                
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss
        except ImportError:
            # Fallback to manual tracking if psutil not available
            return 0
            
    def _perform_cleanup(self, memory_fraction: float):
        """Perform memory cleanup."""
        logger.info(f"Performing field memory cleanup (usage: {memory_fraction:.1%})")
        
        try:
            # Import here to avoid circular dependency
            from src.physics.field.field_dynamics import get_field_dynamics
            
            field_dynamics = get_field_dynamics()
            
            # Get current statistics
            stats = field_dynamics.get_field_statistics()
            
            # Aggressive cleanup if memory pressure is high
            if memory_fraction >= self.high_water_mark:
                logger.warning(f"High memory pressure ({memory_fraction:.1%}), performing aggressive cleanup")
                
                # Delete inactive fields
                inactive_threshold = time.time() - 300  # 5 minutes
                deleted_count = 0
                
                for field_id in field_dynamics.field_registry.list_fields():
                    metadata = field_dynamics.field_registry.get_metadata(field_id)
                    if metadata and metadata.last_update_time < inactive_threshold:
                        if field_dynamics.delete_field(field_id):
                            deleted_count += 1
                            
                logger.info(f"Deleted {deleted_count} inactive fields")
                
                # Clear old evolution history
                field_dynamics._evolution_history.clear()
                
                # Force garbage collection
                gc.collect()
                
            else:
                # Normal cleanup - just trigger garbage collection
                gc.collect(0)  # Quick collection
                
            # Log cleanup results
            new_usage = self._get_memory_usage()
            reduction = memory_fraction - (new_usage / self.max_memory_bytes)
            logger.info(f"Cleanup complete, memory reduced by {reduction:.1%}")
            
        except Exception as e:
            logger.error(f"Error during field cleanup: {e}")
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        recent_history = list(self.memory_history)[-10:] if self.memory_history else []
        
        if recent_history:
            avg_usage = sum(h['usage_bytes'] for h in recent_history) / len(recent_history)
            max_usage = max(h['usage_bytes'] for h in recent_history)
            avg_fraction = avg_usage / self.max_memory_bytes
            max_fraction = max_usage / self.max_memory_bytes
        else:
            avg_usage = max_usage = avg_fraction = max_fraction = 0
            
        return {
            'cleanup_count': self.cleanup_count,
            'average_memory_mb': avg_usage / (1024 * 1024),
            'max_memory_mb': max_usage / (1024 * 1024),
            'average_usage_fraction': avg_fraction,
            'max_usage_fraction': max_fraction,
            'last_cleanup_time': self.last_cleanup_time,
            'is_running': self._running
        }


# Global memory manager instance
_global_memory_manager: Optional[FieldMemoryManager] = None
_manager_lock = threading.Lock()


def get_field_memory_manager() -> FieldMemoryManager:
    """Get the global field memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        with _manager_lock:
            if _global_memory_manager is None:
                _global_memory_manager = FieldMemoryManager()
                _global_memory_manager.start()
                
    return _global_memory_manager


def cleanup_field_memory():
    """Manually trigger field memory cleanup."""
    manager = get_field_memory_manager()
    manager._perform_cleanup(0.0)  # Force cleanup regardless of usage