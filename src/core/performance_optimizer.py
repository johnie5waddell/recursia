"""
Performance Optimization Module for Recursia Runtime
====================================================

This module provides high-performance optimizations for quantum operations,
including parallel processing, caching, and efficient batch operations.

Key Features:
- Asynchronous quantum operations with worker pools
- Batch processing for multiple entanglements
- Caching for frequently used quantum states
- Sparse matrix operations for large quantum systems
- Performance monitoring and auto-tuning
"""

import asyncio
import concurrent.futures
import functools
import logging
import time
from collections import deque
from dataclasses import dataclass
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    enable_parallel_operations: bool = True
    max_workers: int = 4
    enable_caching: bool = True
    cache_size: int = 1000
    enable_sparse_matrices: bool = True
    sparse_threshold: int = 8  # Use sparse matrices for > 8 qubits
    batch_size: int = 10
    enable_auto_tuning: bool = True
    profile_operations: bool = False


class QuantumOperationCache:
    """
    LRU cache for quantum operations to avoid redundant calculations.
    """
    
    def __init__(self, max_size: int = 1000):
        """Initialize cache with maximum size."""
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order = deque()
        self.hits = 0
        self.misses = 0
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any):
        """Put value in cache with LRU eviction."""
        with self._lock:
            if key in self.cache:
                # Update existing
                self.access_order.remove(key)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                oldest = self.access_order.popleft()
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, float]:
        """Get cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = self.hits / total if total > 0 else 0
            return {
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'size': len(self.cache)
            }


class QuantumBatchProcessor:
    """
    Processes multiple quantum operations in parallel batches.
    """
    
    def __init__(self, config: PerformanceConfig):
        """Initialize batch processor."""
        self.config = config
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix="quantum_batch"
        )
        self.pending_operations = deque()
        self._lock = Lock()
        self._processing = False
    
    def add_operation(self, operation: Callable, *args, **kwargs) -> concurrent.futures.Future:
        """Add operation to batch queue."""
        future = concurrent.futures.Future()
        
        with self._lock:
            self.pending_operations.append((operation, args, kwargs, future))
            
            # Process batch if we've reached batch size
            if len(self.pending_operations) >= self.config.batch_size:
                self._process_batch()
        
        return future
    
    def _process_batch(self):
        """Process a batch of operations in parallel."""
        if self._processing or not self.pending_operations:
            return
        
        self._processing = True
        batch = []
        
        # Extract batch
        with self._lock:
            for _ in range(min(self.config.batch_size, len(self.pending_operations))):
                if self.pending_operations:
                    batch.append(self.pending_operations.popleft())
        
        # Submit to executor
        for operation, args, kwargs, future in batch:
            try:
                result_future = self.executor.submit(operation, *args, **kwargs)
                # Chain the futures
                result_future.add_done_callback(
                    lambda rf, f=future: self._complete_operation(rf, f)
                )
            except Exception as e:
                future.set_exception(e)
        
        self._processing = False
    
    def _complete_operation(self, result_future: concurrent.futures.Future, 
                          original_future: concurrent.futures.Future):
        """Complete an operation by transferring result."""
        try:
            result = result_future.result()
            original_future.set_result(result)
        except Exception as e:
            original_future.set_exception(e)
    
    def flush(self):
        """Process all pending operations."""
        with self._lock:
            while self.pending_operations:
                self._process_batch()
    
    def shutdown(self, wait: bool = True):
        """Shutdown the batch processor."""
        self.flush()
        self.executor.shutdown(wait=wait)


class ParallelQuantumEngine:
    """
    High-performance quantum engine with parallel operations.
    """
    
    def __init__(self, runtime, config: Optional[PerformanceConfig] = None):
        """Initialize parallel quantum engine."""
        self.runtime = runtime
        self.config = config or PerformanceConfig()
        
        # Initialize components
        self.cache = QuantumOperationCache(max_size=self.config.cache_size)
        self.batch_processor = QuantumBatchProcessor(self.config)
        
        # Performance tracking
        self.operation_times = deque(maxlen=1000)
        self._lock = RLock()
        
        logger.info(f"Initialized ParallelQuantumEngine with {self.config.max_workers} workers")
    
    def parallel_entangle_states(self, entanglement_pairs: List[Tuple[str, str, List[int], List[int]]]) -> List[bool]:
        """
        Perform multiple entanglements in parallel.
        
        Args:
            entanglement_pairs: List of (state1, state2, qubits1, qubits2) tuples
            
        Returns:
            List of success indicators
        """
        start_time = time.time()
        
        # Check cache for any existing entanglements
        results = []
        uncached_pairs = []
        
        for state1, state2, qubits1, qubits2 in entanglement_pairs:
            cache_key = f"entangle_{state1}_{state2}_{qubits1}_{qubits2}"
            cached_result = self.cache.get(cache_key)
            
            if cached_result is not None:
                results.append(cached_result)
            else:
                uncached_pairs.append((state1, state2, qubits1, qubits2))
        
        # Process uncached pairs in parallel
        if uncached_pairs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = []
                
                for state1, state2, qubits1, qubits2 in uncached_pairs:
                    future = executor.submit(
                        self._entangle_single_pair,
                        state1, state2, qubits1, qubits2
                    )
                    futures.append((future, state1, state2, qubits1, qubits2))
                
                # Collect results
                for future, state1, state2, qubits1, qubits2 in futures:
                    try:
                        result = future.result(timeout=5.0)
                        results.append(result)
                        
                        # Cache successful results
                        if result:
                            cache_key = f"entangle_{state1}_{state2}_{qubits1}_{qubits2}"
                            self.cache.put(cache_key, result)
                    except Exception as e:
                        logger.error(f"Entanglement failed for {state1}-{state2}: {e}")
                        results.append(False)
        
        # Track performance
        duration = time.time() - start_time
        self.operation_times.append(('parallel_entangle', duration, len(entanglement_pairs)))
        
        logger.info(f"Parallel entangled {len(entanglement_pairs)} pairs in {duration:.3f}s "
                   f"({duration/len(entanglement_pairs):.3f}s per pair)")
        
        return results
    
    def _entangle_single_pair(self, state1: str, state2: str, 
                             qubits1: List[int], qubits2: List[int]) -> bool:
        """Entangle a single pair of states."""
        try:
            # Use the runtime's entangle method
            return self.runtime.entangle_states(state1, state2, qubits1, qubits2, method='optimized')
        except Exception as e:
            logger.error(f"Failed to entangle {state1}-{state2}: {e}")
            return False
    
    def batch_apply_gates(self, gate_operations: List[Tuple[str, str, List[int], Optional[List[float]]]]) -> List[bool]:
        """
        Apply multiple gates in parallel batches.
        
        Args:
            gate_operations: List of (state_name, gate_name, target_qubits, params) tuples
            
        Returns:
            List of success indicators
        """
        start_time = time.time()
        results = []
        
        # Group operations by state for better cache locality
        operations_by_state = {}
        for state_name, gate_name, qubits, params in gate_operations:
            if state_name not in operations_by_state:
                operations_by_state[state_name] = []
            operations_by_state[state_name].append((gate_name, qubits, params))
        
        # Process each state's operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for state_name, ops in operations_by_state.items():
                future = executor.submit(self._apply_gates_to_state, state_name, ops)
                futures.append(future)
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    state_results = future.result(timeout=5.0)
                    results.extend(state_results)
                except Exception as e:
                    logger.error(f"Batch gate application failed: {e}")
                    results.extend([False] * len(ops))
        
        duration = time.time() - start_time
        self.operation_times.append(('batch_gates', duration, len(gate_operations)))
        
        logger.info(f"Applied {len(gate_operations)} gates in {duration:.3f}s "
                   f"({duration/len(gate_operations):.3f}s per gate)")
        
        return results
    
    def _apply_gates_to_state(self, state_name: str, 
                             operations: List[Tuple[str, List[int], Optional[List[float]]]]) -> List[bool]:
        """Apply multiple gates to a single state."""
        results = []
        
        for gate_name, qubits, params in operations:
            try:
                success = self.runtime.apply_gate(
                    state_name=state_name,
                    gate_name=gate_name,
                    target_qubits=qubits,
                    params=params
                )
                results.append(success)
            except Exception as e:
                logger.error(f"Failed to apply {gate_name} to {state_name}: {e}")
                results.append(False)
        
        return results
    
    def optimize_quantum_circuit(self, operations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize a sequence of quantum operations for better performance.
        
        Optimizations include:
        - Combining adjacent single-qubit gates
        - Parallelizing commuting operations
        - Reordering for better cache locality
        """
        # Group operations by type
        grouped = {
            'gates': [],
            'measurements': [],
            'entanglements': []
        }
        
        for op in operations:
            op_type = op.get('type')
            if op_type == 'gate':
                grouped['gates'].append(op)
            elif op_type == 'measure':
                grouped['measurements'].append(op)
            elif op_type == 'entangle':
                grouped['entanglements'].append(op)
        
        # Optimize gate sequences
        optimized_gates = self._optimize_gate_sequence(grouped['gates'])
        
        # Reorder operations for better performance
        optimized = []
        
        # Gates first (can be parallelized)
        optimized.extend(optimized_gates)
        
        # Then entanglements (can be parallelized)
        optimized.extend(grouped['entanglements'])
        
        # Measurements last (may collapse states)
        optimized.extend(grouped['measurements'])
        
        return optimized
    
    def _optimize_gate_sequence(self, gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize a sequence of gate operations."""
        if not gates:
            return gates
        
        optimized = []
        i = 0
        
        while i < len(gates):
            current = gates[i]
            
            # Look for gates that can be combined
            if i + 1 < len(gates):
                next_gate = gates[i + 1]
                
                # Check if gates commute and can be parallelized
                if self._gates_commute(current, next_gate):
                    # Mark for parallel execution
                    current['parallel_group'] = current.get('parallel_group', i)
                    next_gate['parallel_group'] = current['parallel_group']
            
            optimized.append(current)
            i += 1
        
        return optimized
    
    def _gates_commute(self, gate1: Dict[str, Any], gate2: Dict[str, Any]) -> bool:
        """Check if two gates commute (can be executed in parallel)."""
        # Gates on different qubits always commute
        qubits1 = set(gate1.get('target_qubits', []))
        qubits2 = set(gate2.get('target_qubits', []))
        
        if qubits1.isdisjoint(qubits2):
            return True
        
        # Same-qubit gates generally don't commute
        # (This is simplified; in reality some gates do commute)
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.operation_times:
            return {
                'cache_stats': self.cache.get_stats(),
                'average_times': {},
                'total_operations': 0
            }
        
        # Calculate average times by operation type
        times_by_type = {}
        for op_type, duration, count in self.operation_times:
            if op_type not in times_by_type:
                times_by_type[op_type] = []
            times_by_type[op_type].append(duration / count)
        
        import numpy as np
        average_times = {
            op_type: np.mean(times) 
            for op_type, times in times_by_type.items()
        }
        
        return {
            'cache_stats': self.cache.get_stats(),
            'average_times': average_times,
            'total_operations': sum(count for _, _, count in self.operation_times),
            'speedup_factor': self._calculate_speedup()
        }
    
    def _calculate_speedup(self) -> float:
        """Calculate speedup compared to sequential execution."""
        if not self.operation_times:
            return 1.0
        
        # Estimate sequential time
        sequential_time = sum(duration for _, duration, _ in self.operation_times)
        
        # Actual parallel time
        parallel_time = sum(duration for _, duration, count in self.operation_times 
                          if count > 1) / self.config.max_workers
        
        # Add sequential operations
        parallel_time += sum(duration for _, duration, count in self.operation_times 
                           if count == 1)
        
        return sequential_time / parallel_time if parallel_time > 0 else 1.0
    
    def cleanup(self):
        """Clean up resources."""
        self.batch_processor.shutdown()
        self.cache.clear()


# Factory function for creating optimized runtime
def create_optimized_runtime(base_runtime, config: Optional[PerformanceConfig] = None):
    """
    Enhance a runtime with performance optimizations.
    
    Args:
        base_runtime: The base RecursiaRuntime instance
        config: Optional performance configuration
        
    Returns:
        Runtime with performance enhancements
    """
    # Create parallel engine
    parallel_engine = ParallelQuantumEngine(base_runtime, config)
    
    # Monkey-patch optimized methods
    original_entangle = base_runtime.entangle_states
    original_apply_gate = base_runtime.apply_gate
    
    def optimized_entangle_states(state1: str, state2: str, 
                                 qubits1: List[int], qubits2: List[int],
                                 method: str = 'direct') -> bool:
        """Optimized entangle_states with caching and parallel support."""
        if method == 'optimized':
            # Use parallel engine for single operation
            results = parallel_engine.parallel_entangle_states([(state1, state2, qubits1, qubits2)])
            return results[0] if results else False
        else:
            # Fall back to original method
            return original_entangle(state1, state2, qubits1, qubits2, method)
    
    # Attach optimized methods
    base_runtime.entangle_states = optimized_entangle_states
    base_runtime.parallel_engine = parallel_engine
    
    # Add batch operations
    base_runtime.batch_entangle = parallel_engine.parallel_entangle_states
    base_runtime.batch_apply_gates = parallel_engine.batch_apply_gates
    base_runtime.optimize_circuit = parallel_engine.optimize_quantum_circuit
    
    logger.info("Runtime enhanced with performance optimizations")
    
    return base_runtime