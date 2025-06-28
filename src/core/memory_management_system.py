"""
Comprehensive Memory Management System for Recursia v3
======================================================

Provides unified memory management for all subsystems to prevent leaks
and ensure production-ready performance.
"""

import threading
import time
import gc
import weakref
import logging
from typing import Dict, Any, Optional, Set, List, Callable
from dataclasses import dataclass, field
from collections import deque
from abc import ABC, abstractmethod
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


@dataclass
class MemoryPolicy:
    """Memory management policy for a subsystem."""
    max_items: int = 10000
    max_memory_mb: float = 512.0
    cleanup_interval: float = 60.0
    high_water_mark: float = 0.8
    cleanup_age_seconds: float = 300.0  # Remove items older than 5 minutes


class MemoryManagedComponent(ABC):
    """Base class for components that need memory management."""
    
    @abstractmethod
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        pass
    
    @abstractmethod
    def cleanup_old_data(self, max_age_seconds: float) -> int:
        """Clean up data older than specified age. Returns items cleaned."""
        pass
    
    @abstractmethod
    def force_cleanup(self, fraction: float = 0.5) -> int:
        """Force cleanup of specified fraction of data. Returns items cleaned."""
        pass


class GlobalMemoryManager:
    """
    Centralized memory management system for all Recursia subsystems.
    
    Features:
    - Automatic registration of managed components
    - Periodic cleanup based on policies
    - Emergency cleanup under memory pressure
    - Performance monitoring and reporting
    """
    
    def __init__(self):
        self._components: Dict[str, MemoryManagedComponent] = {}
        self._policies: Dict[str, MemoryPolicy] = {}
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.RLock()
        self._stats = {
            'total_cleanups': 0,
            'emergency_cleanups': 0,
            'items_cleaned': 0,
            'last_cleanup_time': 0
        }
        
    def register_component(
        self, 
        name: str, 
        component: MemoryManagedComponent, 
        policy: Optional[MemoryPolicy] = None
    ) -> None:
        """Register a component for memory management."""
        with self._lock:
            self._components[name] = component
            self._policies[name] = policy or MemoryPolicy()
            logger.info(f"Registered memory-managed component: {name}")
            
    def unregister_component(self, name: str) -> None:
        """Unregister a component from memory management."""
        with self._lock:
            if name in self._components:
                del self._components[name]
                del self._policies[name]
                logger.info(f"Unregistered component: {name}")
                
    def start(self) -> None:
        """Start the memory management system."""
        with self._lock:
            if not self._running:
                self._running = True
                self._cleanup_thread = threading.Thread(
                    target=self._cleanup_loop,
                    name="GlobalMemoryManager",
                    daemon=True
                )
                self._cleanup_thread.start()
                logger.info("Global memory manager started")
                
    def stop(self) -> None:
        """Stop the memory management system."""
        with self._lock:
            if self._running:
                self._running = False
                if self._cleanup_thread:
                    self._cleanup_thread.join(timeout=5.0)
                logger.info("Global memory manager stopped")
                
    def _cleanup_loop(self) -> None:
        """Main cleanup loop."""
        while self._running:
            try:
                # Check memory pressure
                memory_usage = self._get_system_memory_usage()
                
                # Determine cleanup strategy
                if memory_usage['percent'] > 90:
                    # Emergency cleanup
                    self._emergency_cleanup()
                    self._stats['emergency_cleanups'] += 1
                else:
                    # Regular cleanup based on policies
                    self._regular_cleanup()
                    
                self._stats['total_cleanups'] += 1
                self._stats['last_cleanup_time'] = time.time()
                
                # Sleep until next check
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")
                time.sleep(10.0)
                
    def _get_system_memory_usage(self) -> Dict[str, float]:
        """Get system memory usage."""
        if HAS_PSUTIL:
            try:
                memory = psutil.virtual_memory()
                process = psutil.Process()
                return {
                    'total_mb': memory.total / 1024 / 1024,
                    'available_mb': memory.available / 1024 / 1024,
                    'percent': memory.percent,
                    'process_mb': process.memory_info().rss / 1024 / 1024
                }
            except:
                pass
        
        # Fallback if psutil not available
        return {
            'total_mb': 8192,  # Assume 8GB
            'available_mb': 4096,  # Assume 4GB available
            'percent': 50,  # Assume 50% usage
            'process_mb': 100  # Assume 100MB process
        }
            
    def _regular_cleanup(self) -> None:
        """Perform regular cleanup based on policies."""
        current_time = time.time()
        
        with self._lock:
            for name, component in list(self._components.items()):
                try:
                    policy = self._policies[name]
                    
                    # Get component statistics
                    stats = component.get_memory_usage()
                    
                    # Check if cleanup needed
                    needs_cleanup = False
                    
                    # Check item count
                    if stats.get('item_count', 0) > policy.max_items:
                        needs_cleanup = True
                        
                    # Check memory usage
                    if stats.get('memory_mb', 0) > policy.max_memory_mb * policy.high_water_mark:
                        needs_cleanup = True
                        
                    # Check time since last cleanup
                    last_cleanup = stats.get('last_cleanup_time', 0)
                    if current_time - last_cleanup > policy.cleanup_interval:
                        needs_cleanup = True
                        
                    # Perform cleanup if needed
                    if needs_cleanup:
                        items_cleaned = component.cleanup_old_data(policy.cleanup_age_seconds)
                        self._stats['items_cleaned'] += items_cleaned
                        logger.debug(f"Cleaned {items_cleaned} items from {name}")
                        
                except Exception as e:
                    logger.error(f"Error cleaning component {name}: {e}")
                    
    def _emergency_cleanup(self) -> None:
        """Perform emergency cleanup under high memory pressure."""
        logger.warning("Emergency memory cleanup triggered")
        
        with self._lock:
            for name, component in list(self._components.items()):
                try:
                    # Force cleanup of 50% of data
                    items_cleaned = component.force_cleanup(0.5)
                    self._stats['items_cleaned'] += items_cleaned
                    logger.info(f"Emergency cleaned {items_cleaned} items from {name}")
                except Exception as e:
                    logger.error(f"Error in emergency cleanup of {name}: {e}")
                    
        # Force garbage collection
        gc.collect()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        with self._lock:
            stats = self._stats.copy()
            
            # Add component statistics
            component_stats = {}
            for name, component in self._components.items():
                try:
                    component_stats[name] = component.get_memory_usage()
                except:
                    component_stats[name] = {'error': 'Failed to get stats'}
                    
            stats['components'] = component_stats
            stats['system_memory'] = self._get_system_memory_usage()
            
            return stats


# Global instance
_global_memory_manager: Optional[GlobalMemoryManager] = None
_manager_lock = threading.Lock()


def get_global_memory_manager() -> GlobalMemoryManager:
    """Get the global memory manager instance."""
    global _global_memory_manager
    
    if _global_memory_manager is None:
        with _manager_lock:
            if _global_memory_manager is None:
                _global_memory_manager = GlobalMemoryManager()
                _global_memory_manager.start()
                
    return _global_memory_manager


# Quantum State Memory Management
class QuantumStateMemoryManager(MemoryManagedComponent):
    """Memory management for quantum states."""
    
    def __init__(self, quantum_backend):
        self.backend = quantum_backend
        self.last_cleanup_time = time.time()
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get quantum state memory usage."""
        if not hasattr(self.backend, 'states'):
            return {'item_count': 0, 'memory_mb': 0}
            
        item_count = len(self.backend.states)
        
        # Estimate memory usage (each state ~8KB for small systems)
        memory_mb = item_count * 0.008
        
        return {
            'item_count': item_count,
            'memory_mb': memory_mb,
            'last_cleanup_time': self.last_cleanup_time
        }
        
    def cleanup_old_data(self, max_age_seconds: float) -> int:
        """Clean up old quantum states."""
        if not hasattr(self.backend, 'states'):
            return 0
            
        current_time = time.time()
        cleaned = 0
        
        # Find states to remove
        to_remove = []
        for name, state in self.backend.states.items():
            if hasattr(state, 'creation_time'):
                age = current_time - state.creation_time
                if age > max_age_seconds:
                    to_remove.append(name)
                    
        # Remove old states
        for name in to_remove:
            try:
                del self.backend.states[name]
                cleaned += 1
            except:
                pass
                
        self.last_cleanup_time = current_time
        return cleaned
        
    def force_cleanup(self, fraction: float = 0.5) -> int:
        """Force cleanup of quantum states."""
        if not hasattr(self.backend, 'states'):
            return 0
            
        # Sort states by age and remove oldest
        states_with_age = []
        current_time = time.time()
        
        for name, state in self.backend.states.items():
            age = current_time - getattr(state, 'creation_time', current_time)
            states_with_age.append((age, name))
            
        states_with_age.sort(reverse=True)  # Oldest first
        
        # Remove specified fraction
        to_remove = int(len(states_with_age) * fraction)
        cleaned = 0
        
        for i in range(to_remove):
            try:
                _, name = states_with_age[i]
                del self.backend.states[name]
                cleaned += 1
            except:
                pass
                
        self.last_cleanup_time = current_time
        return cleaned


# Observer Registry Memory Management
class ObserverRegistryMemoryManager(MemoryManagedComponent):
    """Memory management for observer registry."""
    
    def __init__(self, observer_registry):
        self.registry = observer_registry
        self.last_cleanup_time = time.time()
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get observer registry memory usage."""
        if not hasattr(self.registry, 'observers'):
            return {'item_count': 0, 'memory_mb': 0}
            
        item_count = len(self.registry.observers)
        
        # Estimate memory usage
        memory_mb = item_count * 0.002  # ~2KB per observer
        
        return {
            'item_count': item_count,
            'memory_mb': memory_mb,
            'last_cleanup_time': self.last_cleanup_time
        }
        
    def cleanup_old_data(self, max_age_seconds: float) -> int:
        """Clean up old observers."""
        if not hasattr(self.registry, 'observers'):
            return 0
            
        current_time = time.time()
        cleaned = 0
        
        # Find observers to remove
        to_remove = []
        for name, observer in self.registry.observers.items():
            if hasattr(observer, 'creation_time'):
                age = current_time - observer.creation_time
                if age > max_age_seconds:
                    to_remove.append(name)
                    
        # Remove old observers
        for name in to_remove:
            try:
                self.registry.remove_observer(name)
                cleaned += 1
            except:
                pass
                
        self.last_cleanup_time = current_time
        return cleaned
        
    def force_cleanup(self, fraction: float = 0.5) -> int:
        """Force cleanup of observers."""
        if not hasattr(self.registry, 'observers'):
            return 0
            
        # Remove inactive observers first
        cleaned = 0
        observer_names = list(self.registry.observers.keys())
        
        # Sort by activity (if available)
        inactive_observers = []
        for name in observer_names:
            observer = self.registry.observers.get(name)
            if observer and getattr(observer, 'is_active', True) == False:
                inactive_observers.append(name)
                
        # Remove inactive first
        for name in inactive_observers[:int(len(observer_names) * fraction)]:
            try:
                self.registry.remove_observer(name)
                cleaned += 1
            except:
                pass
                
        self.last_cleanup_time = time.time()
        return cleaned


# Measurement History Memory Management
class MeasurementHistoryMemoryManager(MemoryManagedComponent):
    """Memory management for measurement history."""
    
    def __init__(self, runtime):
        self.runtime = runtime
        self.last_cleanup_time = time.time()
        
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get measurement history memory usage."""
        if not hasattr(self.runtime, 'measurement_results'):
            return {'item_count': 0, 'memory_mb': 0}
            
        item_count = len(self.runtime.measurement_results)
        
        # Estimate memory usage
        memory_mb = item_count * 0.001  # ~1KB per measurement
        
        return {
            'item_count': item_count,
            'memory_mb': memory_mb,
            'last_cleanup_time': self.last_cleanup_time
        }
        
    def cleanup_old_data(self, max_age_seconds: float) -> int:
        """Clean up old measurements."""
        if not hasattr(self.runtime, 'measurement_results'):
            return 0
            
        current_time = time.time()
        cleaned = 0
        
        # Keep only recent measurements
        new_results = []
        for result in self.runtime.measurement_results:
            if isinstance(result, dict) and 'timestamp' in result:
                age = current_time - result['timestamp']
                if age <= max_age_seconds:
                    new_results.append(result)
                else:
                    cleaned += 1
            else:
                new_results.append(result)  # Keep if no timestamp
                
        self.runtime.measurement_results = new_results
        self.last_cleanup_time = current_time
        return cleaned
        
    def force_cleanup(self, fraction: float = 0.5) -> int:
        """Force cleanup of measurements."""
        if not hasattr(self.runtime, 'measurement_results'):
            return 0
            
        # Keep only newest measurements
        original_count = len(self.runtime.measurement_results)
        keep_count = int(original_count * (1 - fraction))
        
        # Sort by timestamp if available
        results_with_time = []
        results_without_time = []
        
        for result in self.runtime.measurement_results:
            if isinstance(result, dict) and 'timestamp' in result:
                results_with_time.append(result)
            else:
                results_without_time.append(result)
                
        # Sort by timestamp (newest first)
        results_with_time.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Keep newest
        self.runtime.measurement_results = (
            results_with_time[:keep_count] + 
            results_without_time[:max(0, keep_count - len(results_with_time))]
        )
        
        cleaned = original_count - len(self.runtime.measurement_results)
        self.last_cleanup_time = time.time()
        return cleaned


def setup_memory_management(runtime) -> None:
    """
    Set up memory management for all Recursia subsystems.
    
    Args:
        runtime: The RecursiaRuntime instance
    """
    manager = get_global_memory_manager()
    
    # Register quantum backend
    if hasattr(runtime, 'quantum_backend'):
        quantum_mm = QuantumStateMemoryManager(runtime.quantum_backend)
        manager.register_component(
            'quantum_states',
            quantum_mm,
            MemoryPolicy(max_items=1000, max_memory_mb=256)
        )
        
    # Register observer registry
    if hasattr(runtime, 'observer_registry'):
        observer_mm = ObserverRegistryMemoryManager(runtime.observer_registry)
        manager.register_component(
            'observers',
            observer_mm,
            MemoryPolicy(max_items=500, max_memory_mb=128)
        )
        
    # Register measurement history
    measurement_mm = MeasurementHistoryMemoryManager(runtime)
    manager.register_component(
        'measurements',
        measurement_mm,
        MemoryPolicy(max_items=10000, max_memory_mb=512)
    )
    
    logger.info("Memory management setup complete")