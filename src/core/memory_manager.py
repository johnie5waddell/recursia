import time
import logging
# numpy imported at function level for performance
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
from threading import Lock

logger = logging.getLogger(__name__)

class MemoryBlock:
    """
    Represents a block of allocated memory in the Recursia runtime.
    """
    
    def __init__(self, block_id: int, size: int, pool_name: str, data: Any = None):
        """
        Initialize a memory block
        
        Args:
            block_id: Block identifier
            size: Size of the block in bytes
            pool_name: Name of the memory pool this block belongs to
            data: Optional data stored in this block
        """
        self.id = block_id
        self.size = size
        self.pool_name = pool_name
        self.data = data
        self.creation_time = time.time()
        self.last_access_time = self.creation_time
        self.access_count = 0
        self.is_pinned = False  # Whether this block can be moved during compaction
        self.is_root = False    # Whether this block is a GC root
        self.references: Set[int] = set()  # Set of blocks that reference this block
        self.referenced_by: Set[int] = set()  # Set of blocks that this block references
        
    def access(self) -> None:
        """Update access time and count when block is accessed"""
        self.last_access_time = time.time()
        self.access_count += 1
        
    def pin(self) -> None:
        """Pin the memory block to prevent movement during compaction"""
        self.is_pinned = True
        
    def unpin(self) -> None:
        """Unpin the memory block to allow movement during compaction"""
        self.is_pinned = False
        
    def add_reference(self, block_id: int) -> None:
        """Add a reference to another block"""
        self.references.add(block_id)
        
    def remove_reference(self, block_id: int) -> None:
        """Remove a reference to another block"""
        if block_id in self.references:
            self.references.remove(block_id)
            
    def add_referenced_by(self, block_id: int) -> None:
        """Add a block that references this block"""
        self.referenced_by.add(block_id)
        
    def remove_referenced_by(self, block_id: int) -> None:
        """Remove a block that references this block"""
        if block_id in self.referenced_by:
            self.referenced_by.remove(block_id)
            
    def __repr__(self) -> str:
        return f"MemoryBlock(id={self.id}, size={self.size}, pool={self.pool_name}, refs={len(self.references)})"


class MemoryPool:
    """
    A memory pool that manages a set of memory blocks with similar characteristics.
    Different pools can use different allocation strategies.
    """
    
    def __init__(self, name: str, initial_size: int = 1024 * 1024, 
                 growth_factor: float = 2.0, compaction_threshold: float = 0.3):
        """
        Initialize a memory pool
        
        Args:
            name: Pool name
            initial_size: Initial pool size in bytes
            growth_factor: How much to grow the pool when it's full
            compaction_threshold: Fragmentation threshold to trigger compaction
        """
        self.name = name
        self.size = initial_size
        self.growth_factor = growth_factor
        self.compaction_threshold = compaction_threshold
        self.free_space = initial_size
        self.blocks: Dict[int, Tuple[int, int]] = {}  # block_id -> (offset, size)
        self.free_blocks: List[Tuple[int, int]] = [(0, initial_size)]  # (offset, size)
        self.fragmentation_level = 0.0
        self.last_compaction_time = 0.0
        self.allocation_strategy: Callable[[int], Optional[int]] = self._first_fit_allocate
        self.lock = Lock()  # Thread safety for pool operations
        
    def allocate(self, block_id: int, size: int) -> Optional[int]:
        """
        Allocate memory from this pool
        
        Args:
            block_id: Block identifier
            size: Size to allocate
            
        Returns:
            int: Offset within the pool, or None if allocation failed
        """
        with self.lock:
            # Check if we need to perform compaction
            if self.fragmentation_level > self.compaction_threshold:
                self._compact()
                
            # Try to allocate using the current strategy
            offset = self.allocation_strategy(size)
            
            # If allocation failed, grow the pool and try again
            if offset is None:
                self._grow(size)
                offset = self.allocation_strategy(size)
                
                # If it still failed, we have a problem
                if offset is None:
                    logger.error(f"Failed to allocate {size} bytes in pool {self.name} even after growing")
                    return None
            
            # Record the allocation
            self.blocks[block_id] = (offset, size)
            self.free_space -= size
            
            # Update fragmentation metrics
            self._update_fragmentation_metrics()
            
            return offset
        
    def deallocate(self, block_id: int) -> bool:
        """
        Deallocate a memory block
        
        Args:
            block_id: Block identifier
            
        Returns:
            bool: True if deallocation was successful
        """
        with self.lock:
            if block_id not in self.blocks:
                return False
                
            # Get the block info
            offset, size = self.blocks[block_id]
            
            # Add to free blocks
            self.free_blocks.append((offset, size))
            
            # Remove from allocated blocks
            del self.blocks[block_id]
            
            # Update free space
            self.free_space += size
            
            # Merge adjacent free blocks
            self._merge_free_blocks()
            
            # Update fragmentation metrics
            self._update_fragmentation_metrics()
            
            return True
        
    def _first_fit_allocate(self, size: int) -> Optional[int]:
        """
        First-fit allocation strategy
        
        Args:
            size: Size to allocate
            
        Returns:
            int: Offset within the pool, or None if allocation failed
        """
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size:
                # Use this block
                self.free_blocks.pop(i)
                
                # If there's leftover space, add it back to free blocks
                if block_size > size:
                    self.free_blocks.append((offset + size, block_size - size))
                    
                return offset
                
        return None
        
    def _best_fit_allocate(self, size: int) -> Optional[int]:
        """
        Best-fit allocation strategy
        
        Args:
            size: Size to allocate
            
        Returns:
            int: Offset within the pool, or None if allocation failed
        """
        best_fit_idx = -1
        best_fit_size = float('inf')
        
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size and block_size < best_fit_size:
                best_fit_idx = i
                best_fit_size = block_size
                
        if best_fit_idx != -1:
            # Use this block
            offset, block_size = self.free_blocks.pop(best_fit_idx)
            
            # If there's leftover space, add it back to free blocks
            if block_size > size:
                self.free_blocks.append((offset + size, block_size - size))
                
            return offset
                
        return None
        
    def _worst_fit_allocate(self, size: int) -> Optional[int]:
        """
        Worst-fit allocation strategy
        
        Args:
            size: Size to allocate
            
        Returns:
            int: Offset within the pool, or None if allocation failed
        """
        worst_fit_idx = -1
        worst_fit_size = 0
        
        for i, (offset, block_size) in enumerate(self.free_blocks):
            if block_size >= size and block_size > worst_fit_size:
                worst_fit_idx = i
                worst_fit_size = block_size
                
        if worst_fit_idx != -1:
            # Use this block
            offset, block_size = self.free_blocks.pop(worst_fit_idx)
            
            # If there's leftover space, add it back to free blocks
            if block_size > size:
                self.free_blocks.append((offset + size, block_size - size))
                
            return offset
                
        return None
        
    def _merge_free_blocks(self) -> None:
        """
        Merge adjacent free blocks to reduce fragmentation
        """
        if not self.free_blocks:
            return
            
        # Sort free blocks by offset
        self.free_blocks.sort()
        
        # Merge adjacent blocks
        i = 0
        while i < len(self.free_blocks) - 1:
            current_offset, current_size = self.free_blocks[i]
            next_offset, next_size = self.free_blocks[i + 1]
            
            if current_offset + current_size == next_offset:
                # These blocks are adjacent, merge them
                self.free_blocks[i] = (current_offset, current_size + next_size)
                self.free_blocks.pop(i + 1)
            else:
                i += 1
                
    def _grow(self, min_size: int = 0) -> None:
        """
        Grow the memory pool
        
        Args:
            min_size: Minimum size increment needed
        """
        # Calculate new size
        additional_size = max(min_size, int(self.size * (self.growth_factor - 1)))
        new_size = self.size + additional_size
        
        # Add new free block at the end
        self.free_blocks.append((self.size, additional_size))
        
        # Update size and free space
        self.free_space += additional_size
        self.size = new_size
        
        # Merge adjacent free blocks
        self._merge_free_blocks()
        
        logger.info(f"Pool {self.name} grown from {self.size - additional_size} to {self.size} bytes")
        
    def _compact(self) -> None:
        """
        Compact the memory pool to reduce fragmentation
        """
        # Only perform compaction if it's been a while since the last one
        current_time = time.time()
        if current_time - self.last_compaction_time < 5.0:
            return
            
        logger.info(f"Compacting memory pool {self.name}")
        
        # Sort blocks by offset
        sorted_blocks = sorted([(offset, size, block_id) for block_id, (offset, size) in self.blocks.items()])
        
        # Rebuild the pool from scratch
        self.free_blocks = []
        current_offset = 0
        
        # Re-place all blocks contiguously
        for _, size, block_id in sorted_blocks:
            self.blocks[block_id] = (current_offset, size)
            current_offset += size
            
        # Add the remaining space as a single free block
        if current_offset < self.size:
            self.free_blocks = [(current_offset, self.size - current_offset)]
            
        # Update fragmentation metrics
        self._update_fragmentation_metrics()
        
        # Update compaction timestamp
        self.last_compaction_time = current_time
        
    def _update_fragmentation_metrics(self) -> None:
        """
        Update fragmentation metrics for the pool
        """
        if not self.free_blocks:
            self.fragmentation_level = 0.0
            return
            
        # Calculate fragmentation level
        total_free_space = sum(size for _, size in self.free_blocks)
        largest_free_block = max(size for _, size in self.free_blocks) if self.free_blocks else 0
        
        if total_free_space > 0:
            self.fragmentation_level = 1.0 - (largest_free_block / total_free_space)
        else:
            self.fragmentation_level = 0.0
            
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pool statistics
        
        Returns:
            dict: Pool statistics
        """
        return {
            'name': self.name,
            'size': self.size,
            'free_space': self.free_space,
            'used_space': self.size - self.free_space,
            'utilization': (self.size - self.free_space) / self.size if self.size > 0 else 0,
            'allocated_blocks': len(self.blocks),
            'free_blocks': len(self.free_blocks),
            'fragmentation_level': self.fragmentation_level
        }


class MemoryManager:
    """
    Manages memory allocation and garbage collection for Recursia
    """
    
    # Memory pool types
    STANDARD_POOL = 'standard'
    QUANTUM_POOL = 'quantum'
    STATE_POOL = 'state'
    OBSERVER_POOL = 'observer'
    TEMPORARY_POOL = 'temporary'
    
    # Default minimum allocation size (to avoid wasteful tiny allocations)
    MIN_ALLOCATION_SIZE = 1
    
    # Default GC configuration
    DEFAULT_GC_THRESHOLD = 1000
    DEFAULT_GC_INTERVAL = 30.0
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the memory manager"""
        self.config = config or {}
        
        # Create standard memory pools with configurable sizes
        pool_sizes = self.config.get('pool_sizes', {})
        self.memory_pools = {
            self.STANDARD_POOL: MemoryPool(
                self.STANDARD_POOL, 
                pool_sizes.get(self.STANDARD_POOL, 1024 * 1024)
            ),
            self.QUANTUM_POOL: MemoryPool(
                self.QUANTUM_POOL, 
                pool_sizes.get(self.QUANTUM_POOL, 2 * 1024 * 1024),
                compaction_threshold=0.5
            ),
            self.STATE_POOL: MemoryPool(
                self.STATE_POOL, 
                pool_sizes.get(self.STATE_POOL, 512 * 1024)
            ),
            self.OBSERVER_POOL: MemoryPool(
                self.OBSERVER_POOL, 
                pool_sizes.get(self.OBSERVER_POOL, 256 * 1024)
            ),
            self.TEMPORARY_POOL: MemoryPool(
                self.TEMPORARY_POOL, 
                pool_sizes.get(self.TEMPORARY_POOL, 1024 * 1024),
                growth_factor=1.5
            )
        }
        
        # Configure optimal allocation strategies for different pools
        self.memory_pools[self.STANDARD_POOL].allocation_strategy = self.memory_pools[self.STANDARD_POOL]._best_fit_allocate
        self.memory_pools[self.QUANTUM_POOL].allocation_strategy = self.memory_pools[self.QUANTUM_POOL]._best_fit_allocate
        self.memory_pools[self.TEMPORARY_POOL].allocation_strategy = self.memory_pools[self.TEMPORARY_POOL]._first_fit_allocate
        
        # Track allocations
        self.active_allocations: Dict[int, MemoryBlock] = {}
        self.allocation_count = 0
        self.deallocated_count = 0
        self.total_allocated_memory = 0
        self.gc_runs = 0
        
        # Configure garbage collection
        self.gc_enabled = self.config.get('gc_enabled', True)
        self.garbage_collection_threshold = self.config.get('gc_threshold', self.DEFAULT_GC_THRESHOLD)
        self.gc_stress_level = 0.0  # Increases with allocation pressure
        self.last_gc_time = time.time()
        self.gc_interval = self.config.get('gc_interval', self.DEFAULT_GC_INTERVAL)  # Seconds between automatic GC runs
        self.gc_lock = Lock()  # Thread safety for GC operations
        
        # Memory usage statistics
        self.memory_usage_history = []
        self.max_memory_usage = 0
        self.peak_allocation_rate = 0
        self.allocation_rate_window = []  # For calculating moving average
        
        # Set reference tracking options
        self.track_references = self.config.get('track_references', True)
        
        # Register statistic collection timer
        self.stats_interval = self.config.get('stats_interval', 60.0)  # Collect stats every minute
        self.last_stats_time = time.time()
        
        # LRU cache for temporary pool eviction
        self.lru_block_queue: List[int] = []
        self.memory_budget = self.config.get('memory_budget', 0)  # 0 means unlimited
        
        logger.info("Memory manager initialized with standard pools")
    
    def _attempt_allocation(self, size: int, pool_name: str, fallback_pool: Optional[str] = None) -> Tuple[int, Optional[int]]:
        """
        Helper method to attempt allocation with optional fallback
        
        Args:
            size: Memory size to allocate
            pool_name: Primary pool to try
            fallback_pool: Optional fallback pool
            
        Returns:
            Tuple of (allocation_id, offset or None)
        """
        # Create allocation ID
        allocation_id = self.allocation_count
        self.allocation_count += 1
        
        # Try in the primary pool
        pool = self.memory_pools[pool_name]
        offset = pool.allocate(allocation_id, size)
        
        if offset is not None:
            return allocation_id, offset
            
        # If allocation failed and fallback is provided, try the fallback
        if fallback_pool and fallback_pool != pool_name:
            logger.info(f"Retrying allocation in {fallback_pool} pool")
            pool = self.memory_pools[fallback_pool]
            offset = pool.allocate(allocation_id, size)
            if offset is not None:
                return allocation_id, offset
        
        # Return the ID with None offset to indicate failure
        return allocation_id, None
    
    def allocate(self, size: int, pool_name: str = 'standard', data: Any = None) -> int:
        """
        Allocate memory from a specific pool
        
        Args:
            size (int): Memory size to allocate
            pool_name (str): Memory pool name
            data (Any, optional): Initial data to store
            
        Returns:
            int: Allocation ID
        """
        # Handle invalid size
        if size <= 0:
            logger.warning(f"Attempted to allocate invalid size: {size}")
            size = self.MIN_ALLOCATION_SIZE  # Minimum allocation size
            
        # Ensure pool exists
        if pool_name not in self.memory_pools:
            logger.warning(f"Pool {pool_name} does not exist, using standard pool")
            pool_name = self.STANDARD_POOL
        
        # First allocation attempt with optional fallback to standard pool
        allocation_id, offset = self._attempt_allocation(
            size, 
            pool_name, 
            self.STANDARD_POOL if pool_name != self.STANDARD_POOL else None
        )
                
        if offset is None:
            logger.error(f"Failed to allocate {size} bytes from primary and fallback pools")
            # As a last resort, try garbage collection and retry
            self.collect_garbage(force=True)
            
            # Try again in the original pool
            pool = self.memory_pools[pool_name]
            offset = pool.allocate(allocation_id, size)
            
            if offset is None:
                logger.critical(f"Failed to allocate {size} bytes after garbage collection")
                # Return a special error ID that applications can check
                return -1
        
        # Create memory block
        memory_block = MemoryBlock(allocation_id, size, pool_name, data)
        
        # Register allocation
        self.active_allocations[allocation_id] = memory_block
        
        # Update memory usage
        self.total_allocated_memory += size
        self.max_memory_usage = max(self.max_memory_usage, self.total_allocated_memory)
        
        # Add to LRU queue if it's a temporary block
        if pool_name == self.TEMPORARY_POOL:
            self.lru_block_queue.append(allocation_id)
            # Enforce memory budget if set
            self._enforce_memory_budget()
        
        # Update allocation rate tracking
        current_time = time.time()
        self.allocation_rate_window.append((current_time, size))
        
        # Trim allocation window to last 5 seconds
        while self.allocation_rate_window and self.allocation_rate_window[0][0] < current_time - 5:
            self.allocation_rate_window.pop(0)
            
        # Calculate current allocation rate (bytes/sec)
        if self.allocation_rate_window:
            window_size = current_time - self.allocation_rate_window[0][0]
            if window_size > 0:
                total_window_bytes = sum(s for _, s in self.allocation_rate_window)
                current_rate = total_window_bytes / window_size
                self.peak_allocation_rate = max(self.peak_allocation_rate, current_rate)
        
        # Update GC stress level based on allocation rate and memory usage
        self._update_gc_stress()
        
        # Collect memory usage statistics
        if current_time - self.last_stats_time >= self.stats_interval:
            self._collect_memory_stats()
            self.last_stats_time = current_time
        
        # Check if garbage collection is needed
        if self._check_gc_needed():
            self.collect_garbage()
        
        return allocation_id
    
    def deallocate(self, allocation_id: int) -> bool:
        """
        Deallocate memory
        
        Args:
            allocation_id (int): Allocation ID
            
        Returns:
            bool: True if deallocation was successful
        """
        if allocation_id not in self.active_allocations:
            logger.warning(f"Attempted to deallocate unknown allocation: {allocation_id}")
            return False
        
        # Get allocation info
        allocation = self.active_allocations[allocation_id]
        
        # Clear references
        if self.track_references:
            # Remove references from this block to others
            for ref_id in allocation.references:
                if ref_id in self.active_allocations:
                    self.active_allocations[ref_id].remove_referenced_by(allocation_id)
            
            # Remove references from other blocks to this one
            for ref_id in allocation.referenced_by:
                if ref_id in self.active_allocations:
                    self.active_allocations[ref_id].remove_reference(allocation_id)
        
        # Deallocate from the pool
        pool_name = allocation.pool_name
        if pool_name in self.memory_pools:
            self.memory_pools[pool_name].deallocate(allocation_id)
        
        # Update memory usage
        self.total_allocated_memory -= allocation.size
        
        # Remove from LRU queue if present
        if allocation_id in self.lru_block_queue:
            self.lru_block_queue.remove(allocation_id)
        
        # Remove allocation
        del self.active_allocations[allocation_id]
        self.deallocated_count += 1
        
        # Update GC stress level
        self._update_gc_stress()
        
        return True
    
    def allocate_quantum_state(self, num_qubits: int, name: str = None) -> int:
        """
        Allocate memory for a quantum state
        
        Args:
            num_qubits (int): Number of qubits
            name (str, optional): Optional name for debugging
            
        Returns:
            int: Allocation ID
        """
        # Calculate size needed based on number of qubits
        # For a state vector, we need 2^num_qubits complex numbers
        # Each complex number takes 16 bytes (two doubles)
        if num_qubits < 0:
            logger.error(f"Invalid number of qubits: {num_qubits}")
            return -1
            
        size = 2**num_qubits * 16
        
        # Add metadata size
        metadata_size = 128  # Fixed overhead for metadata
        total_size = size + metadata_size
        
        # Create quantum state data structure
        quantum_data = {
            'type': 'quantum_state',
            'num_qubits': num_qubits,
            'name': name,
            'vector_size': size,
            'allocated_at': time.time()
        }
        
        # Allocate from quantum pool
        allocation_id = self.allocate(total_size, self.QUANTUM_POOL, quantum_data)
        
        if allocation_id >= 0:
            logger.info(f"Allocated quantum state with {num_qubits} qubits, size {total_size} bytes, ID {allocation_id}")
        
        return allocation_id
    
    def allocate_observer(self, properties: Dict[str, Any] = None) -> int:
        """
        Allocate memory for an observer
        
        Args:
            properties (dict, optional): Observer properties
            
        Returns:
            int: Allocation ID
        """
        # Base size for observer structures
        base_size = 256
        
        # Add size for properties
        properties_size = self._calculate_properties_size(properties)
        
        total_size = base_size + properties_size
        
        # Create observer data structure
        observer_data = {
            'type': 'observer',
            'properties': properties or {},
            'allocated_at': time.time()
        }
        
        # Allocate from observer pool
        allocation_id = self.allocate(total_size, self.OBSERVER_POOL, observer_data)
        
        return allocation_id
    
    def _calculate_properties_size(self, properties: Optional[Dict[str, Any]]) -> int:
        """
        Calculate approximate memory size for a dictionary of properties
        
        Args:
            properties: Dictionary of properties to size
            
        Returns:
            int: Estimated size in bytes
        """
        if not properties:
            return 0
            
        properties_size = 0
        for key, value in properties.items():
            # Key size + value size estimation
            key_size = len(key)
            
            if isinstance(value, str):
                value_size = len(value)
            elif isinstance(value, (int, float)):
                value_size = 8
            elif isinstance(value, (list, tuple)):
                # Estimate list size based on length and type
                if value and isinstance(value[0], (int, float)):
                    value_size = len(value) * 8
                elif value and isinstance(value[0], str):
                    value_size = sum(len(s) for s in value)
                else:
                    value_size = len(value) * 16
            elif isinstance(value, dict):
                # Recursively calculate dict size
                value_size = self._calculate_properties_size(value)
            else:
                value_size = 32  # Default size estimate
            
            properties_size += key_size + value_size
            
        return properties_size
    
    def collect_garbage(self, force: bool = False) -> int:
        """
        Perform garbage collection
        
        Args:
            force (bool): Force garbage collection even if thresholds not met
            
        Returns:
            int: Number of allocations freed
        """
        with self.gc_lock:
            if not self.gc_enabled and not force:
                return 0
            
            # Update GC timing
            current_time = time.time()
            self.last_gc_time = current_time
            self.gc_runs += 1
            
            logger.info(f"Running garbage collection cycle #{self.gc_runs}")
            
            # Find blocks that aren't referenced by any other blocks
            # (except for blocks explicitly marked as roots)
            # Do a proper mark and sweep GC algorithm
            freed_count = self._mark_and_sweep_gc()
            
            # For pools that need compaction, compact them
            for pool_name, pool in self.memory_pools.items():
                if pool.fragmentation_level > pool.compaction_threshold:
                    pool._compact()
            
            # Reset GC stress level
            self.gc_stress_level *= 0.5
            
            logger.info(f"Garbage collection freed {freed_count} allocations")
            
            return freed_count
    
    def _mark_and_sweep_gc(self) -> int:
        """
        Perform a mark-and-sweep garbage collection
        
        Returns:
            int: Number of allocations freed
        """
        # Phase 1: Mark all root blocks
        marked_blocks = set()
        
        # First, mark all blocks that are explicitly marked as roots
        for block_id, block in self.active_allocations.items():
            if getattr(block, 'is_root', False):
                marked_blocks.add(block_id)
        
        # Then, do a breadth-first traversal to mark all reachable blocks
        queue = list(marked_blocks)
        while queue:
            block_id = queue.pop(0)
            if block_id in self.active_allocations:
                block = self.active_allocations[block_id]
                for ref_id in block.references:
                    if ref_id not in marked_blocks and ref_id in self.active_allocations:
                        marked_blocks.add(ref_id)
                        queue.append(ref_id)
        
        # Phase 2: Sweep (free all unmarked blocks)
        current_time = time.time()
        candidates = []
        
        for block_id, block in list(self.active_allocations.items()):
            # Skip blocks that are marked (reachable)
            if block_id in marked_blocks:
                continue
                
            # Skip recently allocated blocks (less than 10 seconds old)
            if current_time - block.creation_time < 10:
                continue
            
            # Skip recently accessed blocks (less than 30 seconds ago)
            if current_time - block.last_access_time < 30:
                continue
                
            # Skip blocks that are pinned
            if block.is_pinned:
                continue
                
            candidates.append(block_id)
        
        # Free the identified garbage blocks
        freed_count = 0
        for block_id in candidates:
            if self.deallocate(block_id):
                freed_count += 1
                
        return freed_count
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage
        
        Returns:
            dict: Memory usage statistics
        """
        pool_stats = {name: pool.get_stats() for name, pool in self.memory_pools.items()}
        
        return {
            'total_allocated': self.total_allocated_memory,
            'max_allocated': self.max_memory_usage,
            'active_allocations': len(self.active_allocations),
            'allocation_count': self.allocation_count,
            'deallocated_count': self.deallocated_count,
            'gc_runs': self.gc_runs,
            'gc_stress_level': self.gc_stress_level,
            'peak_allocation_rate': self.peak_allocation_rate,
            'pools': pool_stats,
            'timestamp': time.time()
        }
    
    def store(self, allocation_id: int, data: Any) -> bool:
        """
        Store data in an allocation
        
        Args:
            allocation_id (int): Allocation ID
            data: Data to store
            
        Returns:
            bool: True if storage was successful
        """
        if allocation_id not in self.active_allocations:
            logger.warning(f"Attempted to store data in unknown allocation: {allocation_id}")
            return False
        
        # Update block data
        block = self.active_allocations[allocation_id]
        block.data = data
        block.access()
        
        # Update reference tracking for complex objects if enabled
        if self.track_references:
            self._update_references(allocation_id, data)
            
        # Check if GC is needed after reference update
        if self._check_gc_needed():
            self.collect_garbage()
        
        return True
    
    def load(self, allocation_id: int) -> Any:
        """
        Load data from an allocation
        
        Args:
            allocation_id (int): Allocation ID
            
        Returns:
            The stored data, or None if not found
        """
        if allocation_id not in self.active_allocations:
            logger.warning(f"Attempted to load data from unknown allocation: {allocation_id}")
            return None
        
        # Get data from block
        block = self.active_allocations[allocation_id]
        block.access()  # Update access timestamp
        
        # If this is a temporary block, update its position in the LRU queue
        if block.pool_name == self.TEMPORARY_POOL and allocation_id in self.lru_block_queue:
            self.lru_block_queue.remove(allocation_id)
            self.lru_block_queue.append(allocation_id)
        
        return block.data
    
    def get_allocation_info(self, allocation_id: int) -> Dict[str, Any]:
        """
        Get information about an allocation
        
        Args:
            allocation_id (int): Allocation ID
            
        Returns:
            dict: Allocation information or None if not found
        """
        if allocation_id not in self.active_allocations:
            return None
        
        block = self.active_allocations[allocation_id]
        
        return {
            'id': block.id,
            'size': block.size,
            'pool': block.pool_name,
            'creation_time': block.creation_time,
            'last_access_time': block.last_access_time,
            'access_count': block.access_count,
            'is_pinned': block.is_pinned,
            'is_root': getattr(block, 'is_root', False),
            'reference_count': len(block.referenced_by),
            'references': len(block.references),
            'age': time.time() - block.creation_time
        }
    
    def mark_root(self, allocation_id: int) -> bool:
        """
        Mark an allocation as a GC root (will never be garbage collected automatically)
        
        Args:
            allocation_id (int): Allocation ID
            
        Returns:
            bool: True if successful
        """
        if allocation_id not in self.active_allocations:
            return False
        
        block = self.active_allocations[allocation_id]
        block.is_root = True
        return True
    
    def unmark_root(self, allocation_id: int) -> bool:
        """
        Unmark an allocation as a GC root
        
        Args:
            allocation_id (int): Allocation ID
            
        Returns:
            bool: True if successful
        """
        if allocation_id not in self.active_allocations:
            return False
        
        block = self.active_allocations[allocation_id]
        block.is_root = False
        return True
    
    def create_pool(self, name: str, initial_size: int = 1024 * 1024, 
                    growth_factor: float = 2.0, compaction_threshold: float = 0.3) -> bool:
        """
        Create a new memory pool
        
        Args:
            name (str): Pool name
            initial_size (int): Initial pool size
            growth_factor (float): Growth factor when pool needs to expand
            compaction_threshold (float): Fragmentation threshold for compaction
            
        Returns:
            bool: True if creation was successful
        """
        if name in self.memory_pools:
            logger.warning(f"Pool {name} already exists")
            return False
        
        self.memory_pools[name] = MemoryPool(name, initial_size, growth_factor, compaction_threshold)
        logger.info(f"Created memory pool {name} with size {initial_size}")
        return True
    
    def set_allocation_strategy(self, pool_name: str, strategy: str) -> bool:
        """
        Set allocation strategy for a memory pool
        
        Args:
            pool_name (str): Pool name
            strategy (str): Strategy name ('first_fit', 'best_fit', 'worst_fit')
            
        Returns:
            bool: True if successful
        """
        if pool_name not in self.memory_pools:
            logger.warning(f"Pool {pool_name} does not exist")
            return False
        
        pool = self.memory_pools[pool_name]
        
        strategy_map = {
            'first_fit': pool._first_fit_allocate,
            'best_fit': pool._best_fit_allocate,
            'worst_fit': pool._worst_fit_allocate
        }
        
        if strategy not in strategy_map:
            logger.warning(f"Unknown allocation strategy: {strategy}")
            return False
            
        pool.allocation_strategy = strategy_map[strategy]
        logger.info(f"Set allocation strategy for pool {pool_name} to {strategy}")
        return True
    
    def resize_pool(self, pool_name: str, new_size: int) -> bool:
        """
        Resize a memory pool
        
        Args:
            pool_name (str): Pool name
            new_size (int): New pool size
            
        Returns:
            bool: True if successful
        """
        if pool_name not in self.memory_pools:
            logger.warning(f"Pool {pool_name} does not exist")
            return False
        
        pool = self.memory_pools[pool_name]
        current_size = pool.size
        
        if new_size < current_size:
            # Shrinking - check if we have enough free space
            if pool.free_space < current_size - new_size:
                logger.warning(f"Cannot shrink pool {pool_name}: not enough free space")
                return False
            
            # Compact the pool before shrinking
            pool._compact()
            
            # Adjust the last free block
            if pool.free_blocks and pool.free_blocks[-1][0] + pool.free_blocks[-1][1] == pool.size:
                offset, size = pool.free_blocks[-1]
                reduction = current_size - new_size
                
                if size > reduction:
                    # Reduce the last free block
                    pool.free_blocks[-1] = (offset, size - reduction)
                else:
                    # Remove the last free block
                    pool.free_blocks.pop()
            
            # Update size and free space
            pool.free_space -= (current_size - new_size)
            pool.size = new_size
            
        elif new_size > current_size:
            # Growing - add a new free block at the end
            additional_size = new_size - current_size
            pool.free_blocks.append((current_size, additional_size))
            
            # Update size and free space
            pool.free_space += additional_size
            pool.size = new_size
            
            # Merge adjacent free blocks
            pool._merge_free_blocks()
        
        logger.info(f"Resized pool {pool_name} from {current_size} to {new_size}")
        return True
    
    def allocate_aligned(self, size: int, alignment: int, pool_name: str = 'standard') -> int:
        """
        Allocate memory with specific alignment
        
        Args:
            size (int): Memory size to allocate
            alignment (int): Memory alignment (power of 2)
            pool_name (str): Memory pool name
            
        Returns:
            int: Allocation ID and aligned offset
        """
        # Check if alignment is a power of 2
        if alignment & (alignment - 1) != 0:
            logger.warning(f"Alignment must be a power of 2, got {alignment}")
            alignment = 1 << (alignment.bit_length() - 1)  # Round down to power of 2
        
        # Allocate enough space to ensure we can align
        padded_size = size + alignment - 1
        
        # Allocate the memory
        allocation_id = self.allocate(padded_size, pool_name)
        
        if allocation_id < 0:
            return allocation_id
        
        # Get pool and offset
        block = self.active_allocations[allocation_id]
        pool = self.memory_pools[block.pool_name]
        original_offset = pool.blocks[allocation_id][0]
        
        # Calculate aligned offset
        aligned_offset = (original_offset + alignment - 1) & ~(alignment - 1)
        padding = aligned_offset - original_offset
        
        # If we need to adjust, update the block info
        if padding > 0:
            # Update block offsets in pool
            pool.blocks[allocation_id] = (aligned_offset, size)
            
            # Create a small free block for the padding area if it's big enough
            if padding >= 8:  # Minimum block size
                pool.free_blocks.append((original_offset, padding))
                pool._merge_free_blocks()
        
        # Store the aligned offset in the block's data for future reference
        if isinstance(block.data, dict):
            block.data['aligned_offset'] = aligned_offset
        
        return allocation_id
    
    def _update_gc_stress(self) -> None:
        """
        Update garbage collection stress level based on memory usage
        """
        # Calculate total memory usage percentage across all pools
        total_size = sum(pool.size for pool in self.memory_pools.values())
        total_used = total_size - sum(pool.free_space for pool in self.memory_pools.values())
        usage_percentage = total_used / total_size if total_size > 0 else 0
        
        # Calculate allocation pressure based on recent allocation rate
        allocation_pressure = min(1.0, self.peak_allocation_rate / 1e6)  # Normalize to 1 MB/s
        
        # Update stress level based on usage and allocation rate
        self.gc_stress_level = 0.5 * usage_percentage + 0.3 * allocation_pressure
        
        # Add stress for fragmentation
        fragmentation_factor = max((pool.fragmentation_level for pool in self.memory_pools.values()), default=0.0)
        self.gc_stress_level += 0.2 * fragmentation_factor
        
        # Clamp stress level
        self.gc_stress_level = max(0.0, min(1.0, self.gc_stress_level))
    
    def _collect_memory_stats(self) -> None:
        """
        Collect memory usage statistics
        """
        # Get current memory usage
        memory_usage = self.get_memory_usage()
        
        # Add to history
        self.memory_usage_history.append(memory_usage)
        
        # Keep only the last 100 data points
        if len(self.memory_usage_history) > 100:
            self.memory_usage_history = self.memory_usage_history[-100:]
    
    def _update_references(self, allocation_id: int, data: Any) -> None:
        """
        Update reference tracking for a block based on its data
        
        Args:
            allocation_id (int): Allocation ID
            data: Data stored in the block
        """
        # Get the block
        block = self.active_allocations[allocation_id]
        
        # Clear existing references
        old_references = set(block.references)
        block.references.clear()
        
        # Find new references in the data recursively
        self._scan_object_for_references(allocation_id, data, set())
        
        # Remove old references that are no longer present
        for ref_id in old_references:
            if ref_id not in block.references and ref_id in self.active_allocations:
                self.active_allocations[ref_id].remove_referenced_by(allocation_id)
    
    def _scan_object_for_references(self, allocation_id: int, obj: Any, visited: Set[int]) -> None:
        """
        Recursively scan an object for allocation references
        
        Args:
            allocation_id (int): Allocation ID of the containing block
            obj: Object to scan
            visited: Set of object ids already visited (to prevent cycles)
        """
        # Prevent infinite recursion on cyclic structures
        obj_id = id(obj)
        if obj_id in visited:
            return
        visited.add(obj_id)
        
        # Get the block
        block = self.active_allocations[allocation_id]
        
        if isinstance(obj, dict):
            # Extract references from dict values
            for key, value in obj.items():
                if key == 'allocation_id' and isinstance(value, int) and value in self.active_allocations and value != allocation_id:
                    block.add_reference(value)
                    self.active_allocations[value].add_referenced_by(allocation_id)
                elif isinstance(value, (dict, list, tuple, set)):
                    self._scan_object_for_references(allocation_id, value, visited)
        
        elif isinstance(obj, (list, tuple, set)):
            # Extract references from sequence items
            for item in obj:
                if isinstance(item, dict) and 'allocation_id' in item:
                    ref_id = item['allocation_id']
                    if isinstance(ref_id, int) and ref_id in self.active_allocations and ref_id != allocation_id:
                        block.add_reference(ref_id)
                        self.active_allocations[ref_id].add_referenced_by(allocation_id)
                elif isinstance(item, (dict, list, tuple, set)):
                    self._scan_object_for_references(allocation_id, item, visited)
    
    def _enforce_memory_budget(self) -> None:
        """
        Enforce memory budget by freeing oldest temporary blocks if needed
        """
        if self.memory_budget <= 0:
            return  # No budget limit
            
        # Check if we're over budget
        if self.total_allocated_memory <= self.memory_budget:
            return
            
        # We need to free memory - start with LRU blocks in the temporary pool
        freed = 0
        target_free = self.total_allocated_memory - self.memory_budget
        
        while freed < target_free and self.lru_block_queue:
            # Get the least recently used block
            block_id = self.lru_block_queue[0]
            
            if block_id in self.active_allocations:
                block = self.active_allocations[block_id]
                
                # Only free if in temporary pool and not pinned
                if block.pool_name == self.TEMPORARY_POOL and not block.is_pinned:
                    block_size = block.size
                    
                    if self.deallocate(block_id):
                        freed += block_size
                        logger.info(f"LRU eviction: freed block {block_id} ({block_size} bytes)")
            else:
                # Remove from LRU queue if not in active allocations
                self.lru_block_queue.pop(0)
    
    def _check_gc_needed(self) -> bool:
        """
        Check if garbage collection is needed based on various triggers
        
        Returns:
            bool: True if GC should run
        """
        if not self.gc_enabled:
            return False
            
        current_time = time.time()
        
        # Check various conditions that could trigger GC
        conditions = [
            len(self.active_allocations) > self.garbage_collection_threshold,  # Too many allocations
            self.gc_stress_level > 0.8,  # High memory pressure
            current_time - self.last_gc_time > self.gc_interval  # Time-based interval
        ]
        
        return any(conditions)