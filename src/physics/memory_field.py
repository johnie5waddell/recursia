import logging
import math
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback for numpy operations if needed
    class _NumpyFallback:
        # Add ndarray type for compatibility
        ndarray = list
        def array(self, x): return x
        def zeros(self, shape): return [0] * (shape if isinstance(shape, int) else shape[0])
        def ones(self, shape): return [1] * (shape if isinstance(shape, int) else shape[0])
        @property
        def pi(self): return 3.14159265359
    np = _NumpyFallback()
import time
import heapq
from typing import Dict, List, Optional, Set, Tuple, Union, Any

logger = logging.getLogger(__name__)

class MemoryFieldPhysics:
    """
    Models memory as a physical field in the Organic Simulation Hypothesis.
    
    In the Recursia framework, memory is treated as a physical field with properties
    such as strain, coherence, and entropy. This class implements the dynamics of
    memory regions, their connections, and the information flows between them.
    
    The memory field affects and is affected by quantum operations, observer
    interactions, and recursive processes. It provides a substrate for information
    storage and transfer across the simulation.
    
    Key concepts:
    - Memory strain: Represents load on memory regions (0 to 1)
    - Memory coherence: Represents information integrity in memory (0 to 1)
    - Memory entropy: Represents disorder or information decay (0+)
    - Region connectivity: Represents information pathways between regions
    """
    
    # Constants for merge/split operations
    MERGER_STRAIN_COST = 0.1  # Extra strain from merging process
    COHERENCE_REDUCTION_ON_MERGE = 0.2  # Coherence loss from merging
    ENTROPY_INCREASE_ON_MERGE = 0.2  # Extra entropy from merging
    
    SPLIT_STRAIN_COST = 0.15  # Extra strain from splitting
    COHERENCE_LOSS_ON_SPLIT = 0.25  # Coherence loss from splitting
    ENTROPY_GAIN_ON_SPLIT = 0.3  # Extra entropy from splitting
    
    def __init__(self):
        """Initialize the memory field physics model with default parameters"""
        # Memory region registries
        self.memory_strain = {}         # Maps region name -> strain value (0-1)
        self.memory_coherence = {}      # Maps region name -> coherence value (0-1)
        self.memory_entropy = {}        # Maps region name -> entropy value (0+)
        self.region_connectivity = {}   # Maps region name -> {connected_region -> strength}
        self.region_metadata = {}       # Maps region name -> metadata dictionary
        
        # Field parameters
        self.memory_capacity = 1.0      # Normalized memory capacity (0 to 1)
        self.strain_diffusion_rate = 0.2  # How quickly strain spreads across regions
        self.coherence_coupling = 0.3   # How strongly coherence couples between regions
        self.memory_decay_rate = 0.05   # Rate at which memory naturally decays
        self.baseline_entropy_increase = 0.01  # Natural entropy increase rate
        
        # Advanced parameters
        self.strain_threshold = 0.85    # Threshold for high strain warnings
        self.critical_strain = 0.95     # Threshold for potential region failure
        self.max_coherence_wave_distance = 10.0  # Maximum distance for coherence waves
        self.max_regions = 1000         # Maximum number of regions for performance
        self.max_iterations = 100       # Maximum iterations for simulations
        
        # Add history tracking capability
        self.enable_history = False
        self.max_history_entries = 100
        self.history = {
            'timestamps': [],
            'strain': {},
            'coherence': {},
            'entropy': {}
        }
    
    def enable_history_tracking(self, enabled: bool = True, max_entries: int = 100) -> None:
        """
        Enable or disable tracking of field property history over time.
        
        Args:
            enabled: Whether to enable history tracking
            max_entries: Maximum number of history entries to maintain
        """
        self.enable_history = enabled
        self.max_history_entries = max_entries
        
        # Initialize history structures if not already done
        if not hasattr(self, 'history') or not self.history:
            self.history = {
                'timestamps': [],
                'strain': {},
                'coherence': {},
                'entropy': {}
            }
    
    def _record_history(self) -> None:
        """Record current field state in history if tracking is enabled."""
        if not self.enable_history:
            return
        
        timestamp = self._get_timestamp()
        self.history['timestamps'].append(timestamp)
        
        # Limit history length
        if len(self.history['timestamps']) > self.max_history_entries:
            oldest = self.history['timestamps'].pop(0)
            # Clean up other history entries for this timestamp
            for metric in ['strain', 'coherence', 'entropy']:
                for region in self.history[metric]:
                    if len(self.history[metric][region]) > 0:
                        self.history[metric][region].pop(0)
        
        # Record current values for each region
        for region in self.memory_strain:
            # Initialize region history if needed
            for metric, values in [
                ('strain', self.memory_strain),
                ('coherence', self.memory_coherence),
                ('entropy', self.memory_entropy)
            ]:
                if region not in self.history[metric]:
                    self.history[metric][region] = []
                self.history[metric][region].append(values[region])
    
    def get_region_history(self, region: str, metric: str = 'all') -> Dict[str, Any]:
        """
        Get historical data for a specific region.
        
        Args:
            region: Region name
            metric: 'strain', 'coherence', 'entropy', or 'all'
            
        Returns:
            Dict[str, Any]: Historical data for the region
        """
        if not self.enable_history:
            return {'error': 'History tracking not enabled'}
        
        if region not in self.memory_strain:
            return {'error': f"Region '{region}' not found"}
        
        result = {'timestamps': self.history['timestamps']}
        
        if metric == 'all':
            metrics = ['strain', 'coherence', 'entropy']
        elif metric in ['strain', 'coherence', 'entropy']:
            metrics = [metric]
        else:
            return {'error': f"Invalid metric: {metric}"}
        
        for m in metrics:
            if region in self.history[m]:
                result[m] = self.history[m][region]
        
        return result
    
    def register_memory_region(self, name: str, initial_strain: float = 0.0,
                              initial_coherence: float = 1.0,
                              initial_entropy: float = 0.0,
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register a memory region in the field.
        
        Args:
            name: Region name
            initial_strain: Initial memory strain (0 to 1)
            initial_coherence: Initial memory coherence (0 to 1)
            initial_entropy: Initial memory entropy (0+)
            metadata: Optional metadata for the region
            
        Returns:
            bool: True if registration was successful
            
        Raises:
            ValueError: If name is invalid or region already exists
        """
        try:
            # Validate inputs
            if not isinstance(name, str) or not name:
                raise ValueError("Region name must be a non-empty string")
                
            if name in self.memory_strain:
                logger.warning(f"Region '{name}' already exists")
                return False
                
            # Check maximum regions limit
            if len(self.memory_strain) >= self.max_regions:
                logger.warning(f"Maximum number of regions ({self.max_regions}) reached")
                return False
            
            # Clamp values to valid ranges
            initial_strain = max(0.0, min(1.0, initial_strain))
            initial_coherence = max(0.0, min(1.0, initial_coherence))
            initial_entropy = max(0.0, initial_entropy)
            
            # Register the region
            self.memory_strain[name] = initial_strain
            self.memory_coherence[name] = initial_coherence
            self.memory_entropy[name] = initial_entropy
            self.region_connectivity[name] = {}
            self.region_metadata[name] = metadata or {}
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            logger.info(f"Registered memory region '{name}'")
            return True
        except Exception as e:
            logger.error(f"Error registering memory region '{name}': {e}")
            return False
    
    def connect_regions(self, region1: str, region2: str, strength: float = 1.0) -> bool:
        """
        Connect two memory regions to enable information flow between them.
        
        Args:
            region1: First region name
            region2: Second region name
            strength: Connection strength (0 to 1)
            
        Returns:
            bool: True if connection was successful
            
        Raises:
            ValueError: If regions don't exist or connection is invalid
        """
        try:
            # Validate regions
            if region1 not in self.memory_strain:
                raise ValueError(f"Region '{region1}' not registered")
            if region2 not in self.memory_strain:
                raise ValueError(f"Region '{region2}' not registered")
            if region1 == region2:
                raise ValueError(f"Cannot connect region '{region1}' to itself")
                
            # Ensure connection strength is within bounds
            clamped_strength = max(0.0, min(1.0, strength))
            
            # Connect in both directions
            if region1 not in self.region_connectivity:
                self.region_connectivity[region1] = {}
            if region2 not in self.region_connectivity:
                self.region_connectivity[region2] = {}
            
            self.region_connectivity[region1][region2] = clamped_strength
            self.region_connectivity[region2][region1] = clamped_strength
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            logger.debug(f"Connected regions '{region1}' and '{region2}' with strength {clamped_strength}")
            return True
        except ValueError as e:
            logger.warning(f"Failed to connect regions: {e}")
            return False
        except Exception as e:
            logger.error(f"Error connecting regions '{region1}' and '{region2}': {e}")
            return False
    
    def update_connection_strength(self, region1: str, region2: str, strength: float) -> bool:
        """
        Update the connection strength between two memory regions.
        
        Args:
            region1: First region name
            region2: Second region name
            strength: New connection strength (0 to 1)
            
        Returns:
            bool: True if update was successful
        """
        try:
            # Validate regions
            if region1 not in self.memory_strain:
                raise ValueError(f"Region '{region1}' not registered")
            if region2 not in self.memory_strain:
                raise ValueError(f"Region '{region2}' not registered")
                
            # Check if connection exists
            if region2 not in self.region_connectivity.get(region1, {}):
                logger.info(f"Creating new connection between '{region1}' and '{region2}'")
                return self.connect_regions(region1, region2, strength)
                
            # Ensure strength is within bounds
            clamped_strength = max(0.0, min(1.0, strength))
            
            # Update connection strength in both directions
            self.region_connectivity[region1][region2] = clamped_strength
            self.region_connectivity[region2][region1] = clamped_strength
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            return True
        except Exception as e:
            logger.error(f"Error updating connection strength: {e}")
            return False
    
    def disconnect_regions(self, region1: str, region2: str) -> bool:
        """
        Remove the connection between two memory regions.
        
        Args:
            region1: First region name
            region2: Second region name
            
        Returns:
            bool: True if disconnection was successful
        """
        try:
            # Validate regions
            if region1 not in self.memory_strain or region2 not in self.memory_strain:
                return False
                
            # Remove connections in both directions
            if region2 in self.region_connectivity.get(region1, {}):
                del self.region_connectivity[region1][region2]
            if region1 in self.region_connectivity.get(region2, {}):
                del self.region_connectivity[region2][region1]
                
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
                
            logger.debug(f"Disconnected regions '{region1}' and '{region2}'")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting regions: {e}")
            return False
    
    def add_memory_strain(self, region: str, amount: float) -> float:
        """
        Add memory strain to a region. Strain represents computational
        or information processing load on a region.
        
        Args:
            region: Region name
            amount: Amount of strain to add (positive or negative)
            
        Returns:
            float: New strain value
            
        Raises:
            ValueError: If region doesn't exist
        """
        try:
            # Validate region
            if region not in self.memory_strain:
                raise ValueError(f"Region '{region}' not registered")
                
            # Get current strain
            current_strain = self.memory_strain[region]
            
            # Calculate new strain
            new_strain = current_strain + amount
            
            # Clamp to valid range
            new_strain = max(0.0, min(1.0, new_strain))
            
            # Update strain
            self.memory_strain[region] = new_strain
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            # Log warning if strain is high
            if new_strain > self.strain_threshold and current_strain <= self.strain_threshold:
                logger.warning(f"Region '{region}' has high strain: {new_strain:.2f}")
            
            # Check for critical strain
            if new_strain > self.critical_strain:
                logger.warning(f"Region '{region}' is at critical strain: {new_strain:.2f}")
                # Could trigger emergency defragmentation here
                
            return new_strain
        except ValueError as e:
            logger.warning(str(e))
            return 0.0
        except Exception as e:
            logger.error(f"Error adding memory strain to '{region}': {e}")
            return 0.0
    
    def modify_coherence(self, region: str, amount: float) -> float:
        """
        Modify the coherence of a memory region. Coherence represents
        the integrity and stability of information.
        
        Args:
            region: Region name
            amount: Amount to change coherence (positive or negative)
            
        Returns:
            float: New coherence value
        """
        try:
            # Validate region
            if region not in self.memory_coherence:
                raise ValueError(f"Region '{region}' not registered")
                
            # Get current coherence
            current_coherence = self.memory_coherence[region]
            
            # Calculate new coherence
            new_coherence = current_coherence + amount
            
            # Clamp to valid range
            new_coherence = max(0.0, min(1.0, new_coherence))
            
            # Update coherence
            self.memory_coherence[region] = new_coherence
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            # Log significant changes
            if abs(amount) > 0.2:
                logger.debug(f"Significant coherence change in '{region}': {current_coherence:.2f} -> {new_coherence:.2f}")
                
            return new_coherence
        except ValueError as e:
            logger.warning(str(e))
            return 0.0
        except Exception as e:
            logger.error(f"Error modifying coherence for '{region}': {e}")
            return 0.0
    
    def adjust_entropy(self, region: str, amount: float) -> float:
        """
        Adjust the entropy of a memory region. Entropy represents
        disorder or information decay.
        
        Args:
            region: Region name
            amount: Amount to change entropy (positive or negative)
            
        Returns:
            float: New entropy value
        """
        try:
            # Validate region
            if region not in self.memory_entropy:
                raise ValueError(f"Region '{region}' not registered")
                
            # Get current entropy
            current_entropy = self.memory_entropy[region]
            
            # Calculate new entropy (no upper limit, but must be >= 0)
            new_entropy = max(0.0, current_entropy + amount)
            
            # Update entropy
            self.memory_entropy[region] = new_entropy
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            # Log significant entropy reduction
            if amount < -0.5:
                logger.debug(f"Significant entropy reduction in '{region}': {current_entropy:.2f} -> {new_entropy:.2f}")
                
            return new_entropy
        except ValueError as e:
            logger.warning(str(e))
            return 0.0
        except Exception as e:
            logger.error(f"Error adjusting entropy for '{region}': {e}")
            return 0.0
    
    def get_region_properties(self, region: str) -> Dict[str, Any]:
        """
        Get all properties of a memory region.
        
        Args:
            region: Region name
            
        Returns:
            Dict[str, Any]: Region properties
        """
        try:
            # Validate region
            if region not in self.memory_strain:
                logger.warning(f"Region '{region}' not registered")
                return {}
                
            # Collect properties
            properties = {
                'strain': self.memory_strain.get(region, 0.0),
                'coherence': self.memory_coherence.get(region, 0.0),
                'entropy': self.memory_entropy.get(region, 0.0),
                'connections': list(self.region_connectivity.get(region, {}).keys()),
                'connection_count': len(self.region_connectivity.get(region, {})),
                'metadata': self.region_metadata.get(region, {})
            }
            
            return properties
        except Exception as e:
            logger.error(f"Error getting properties for '{region}': {e}")
            return {}
    
    def get_total_strain(self) -> float:
        """
        Calculate the total strain across all memory regions.
        
        Returns:
            float: Average strain across all regions (0-1), or 0 if no regions exist
        """
        try:
            if not self.memory_strain:
                return 0.0
            
            total_strain = sum(self.memory_strain.values())
            num_regions = len(self.memory_strain)
            
            # Return average strain
            return total_strain / num_regions if num_regions > 0 else 0.0
        except Exception as e:
            logger.error(f"Error calculating total strain: {e}")
            return 0.0
    
    def get_memory_fragments(self) -> List[Dict[str, Any]]:
        """
        Get memory regions as fragments for OSH calculations.
        
        Returns:
            List of memory fragment dictionaries
        """
        fragments = []
        try:
            # Limit to prevent memory issues - max 16 fragments
            regions = list(self.memory_strain.keys())[:16]
            num_regions = len(regions)
            
            if num_regions == 0:
                return []
            
            for i, region_name in enumerate(regions):
                # Calculate position in a compact spherical arrangement
                # Use golden angle for optimal sphere packing
                golden_angle = math.pi * (3.0 - math.sqrt(5.0))  # ~2.39996
                theta = golden_angle * i
                y = 1 - (i / float(num_regions - 1)) * 2  # -1 to +1
                radius_at_y = math.sqrt(1 - y * y)
                
                # Constrain to reasonable bounds
                base_radius = 5.0  # Much smaller than before
                x = math.cos(theta) * radius_at_y * base_radius
                z = math.sin(theta) * radius_at_y * base_radius
                y = y * base_radius + self.memory_strain.get(region_name, 0.0) * 2  # Smaller height variation
                
                fragment = {
                    'name': region_name,
                    'coherence': self.memory_coherence.get(region_name, 0.0),
                    'size': 0.5,  # Smaller size for performance
                    'strain': self.memory_strain.get(region_name, 0.0),
                    'entropy': self.memory_entropy.get(region_name, 0.0),
                    'coupling_strength': 0.5,
                    'connections': min(len(self.region_connectivity.get(region_name, {})), 4),  # Limit connections
                    'position': [
                        round(x, 2),  # Round to reduce precision/memory
                        round(y, 2),
                        round(z, 2)
                    ],
                    'phase': (i * 0.5) % (2 * math.pi)  # Static phase to reduce updates
                }
                fragments.append(fragment)
        except Exception as e:
            logger.error(f"Error getting memory fragments: {e}")
        
        return fragments
    
    def update_field(self, time_step: float = 1.0) -> Dict[str, Dict[str, float]]:
        """
        Update the entire memory field for a time step.
        This applies natural diffusion, decay, and entropy processes.
        
        Args:
            time_step: Time step size
            
        Returns:
            Dict[str, Dict[str, float]]: Changes in each region
        """
        try:
            # Initialize change dictionaries
            strain_changes, coherence_changes, entropy_changes = self._initialize_change_maps()
            
            # Process different types of physics
            self._process_region_interactions(strain_changes, coherence_changes, entropy_changes, time_step)
            self._apply_natural_processes(strain_changes, coherence_changes, entropy_changes, time_step)
            self._apply_field_changes(strain_changes, coherence_changes, entropy_changes)
            
            # Record history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            # Return changes applied
            return {
                'strain': strain_changes,
                'coherence': coherence_changes,
                'entropy': entropy_changes
            }
        except Exception as e:
            logger.debug(f"Error updating memory field: {e}")  # Reduced from error to debug
            return {'strain': {}, 'coherence': {}, 'entropy': {}}
    
    def _initialize_change_maps(self) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Initialize maps to track changes for each region.
        
        Returns:
            Tuple of dictionaries for strain, coherence, and entropy changes
        """
        strain_changes = {}
        coherence_changes = {}
        entropy_changes = {}
        
        for region in self.memory_strain:
            strain_changes[region] = 0.0
            coherence_changes[region] = 0.0
            entropy_changes[region] = 0.0
        
        return strain_changes, coherence_changes, entropy_changes
    
    def _process_region_interactions(self, strain_changes: Dict[str, float],
                                    coherence_changes: Dict[str, float],
                                    entropy_changes: Dict[str, float],
                                    time_step: float) -> None:
        """
        Calculate changes from interactions between regions.
        
        Args:
            strain_changes: Dictionary to store strain changes
            coherence_changes: Dictionary to store coherence changes
            entropy_changes: Dictionary to store entropy changes
            time_step: Time step size
        """
        # Process each region's interactions with connected regions
        for region1 in self.memory_strain:
            # Get connections for this region
            connections = self.region_connectivity.get(region1, {})
            
            for region2, connection_strength in connections.items():
                # Skip if it's the same region (shouldn't happen, but just in case)
                if region1 == region2:
                    continue
                
                # Calculate strain diffusion
                strain_diff = self.memory_strain[region2] - self.memory_strain[region1]
                strain_flow = strain_diff * connection_strength * self.strain_diffusion_rate * time_step
                strain_changes[region1] += strain_flow
                strain_changes[region2] -= strain_flow
                
                # Calculate coherence coupling
                coherence_diff = self.memory_coherence[region2] - self.memory_coherence[region1]
                coherence_flow = coherence_diff * connection_strength * self.coherence_coupling * time_step
                coherence_changes[region1] += coherence_flow
                coherence_changes[region2] -= coherence_flow
                
                # High strain in connected regions increases entropy
                avg_strain = 0.5 * (self.memory_strain[region1] + self.memory_strain[region2])
                entropy_increase = avg_strain * connection_strength * 0.05 * time_step
                entropy_changes[region1] += entropy_increase
                entropy_changes[region2] += entropy_increase
    
    def _apply_natural_processes(self, strain_changes: Dict[str, float],
                                coherence_changes: Dict[str, float],
                                entropy_changes: Dict[str, float],
                                time_step: float) -> None:
        """
        Apply natural processes to all regions.
        
        Args:
            strain_changes: Dictionary to store strain changes
            coherence_changes: Dictionary to store coherence changes
            entropy_changes: Dictionary to store entropy changes
            time_step: Time step size
        """
        for region in self.memory_strain:
            # Natural strain relaxation
            strain_relaxation = self.memory_strain[region] * 0.1 * time_step
            strain_changes[region] -= strain_relaxation
            
            # Natural coherence decay
            coherence_decay = self.memory_coherence[region] * self.memory_decay_rate * time_step
            coherence_changes[region] -= coherence_decay
            
            # Natural entropy increase (proportional to current strain)
            natural_entropy = self.baseline_entropy_increase * (1.0 + self.memory_strain[region]) * time_step
            entropy_changes[region] += natural_entropy
            
            # Strain reduces coherence
            strain_induced_decoherence = self.memory_strain[region] * 0.2 * time_step
            coherence_changes[region] -= strain_induced_decoherence
            
            # High entropy reduces coherence
            entropy_induced_decoherence = self.memory_entropy[region] * 0.01 * time_step
            coherence_changes[region] -= entropy_induced_decoherence
    
    def _apply_field_changes(self, strain_changes: Dict[str, float],
                            coherence_changes: Dict[str, float],
                            entropy_changes: Dict[str, float]) -> None:
        """
        Apply calculated changes to all regions.
        
        Args:
            strain_changes: Dictionary of strain changes to apply
            coherence_changes: Dictionary of coherence changes to apply
            entropy_changes: Dictionary of entropy changes to apply
        """
        for region in self.memory_strain:
            # Update strain (clamped between 0 and 1)
            self.memory_strain[region] = max(0.0, min(1.0, 
                self.memory_strain[region] + strain_changes[region]))
            
            # Update coherence (clamped between 0 and 1)
            self.memory_coherence[region] = max(0.0, min(1.0, 
                self.memory_coherence[region] + coherence_changes[region]))
            
            # Update entropy (only clamped at 0, no upper limit)
            self.memory_entropy[region] = max(0.0, 
                self.memory_entropy[region] + entropy_changes[region])
    
    def merge_regions(self, region1: str, region2: str, new_region_name: str) -> bool:
        """
        Merge two memory regions into a new combined region.
        This represents consolidation of memory areas.
        
        Args:
            region1: First region to merge
            region2: Second region to merge
            new_region_name: Name for the new merged region
            
        Returns:
            bool: True if merge was successful
            
        Raises:
            ValueError: If regions don't exist or merge is invalid
        """
        try:
            # Validate regions
            if region1 not in self.memory_strain:
                raise ValueError(f"Region '{region1}' not registered")
            if region2 not in self.memory_strain:
                raise ValueError(f"Region '{region2}' not registered")
            if region1 == region2:
                raise ValueError(f"Cannot merge region '{region1}' with itself")
            
            if new_region_name in self.memory_strain:
                raise ValueError(f"Region '{new_region_name}' already exists")
            
            # Check if regions are connected
            connection = self.region_connectivity.get(region1, {}).get(region2, 0.0)
            if connection <= 0.0:
                raise ValueError(f"Regions must be connected to merge: '{region1}', '{region2}'")
            
            # Calculate merged properties
            # Strain combines based on weighted average plus merger cost
            merged_strain = (
                (self.memory_strain[region1] + self.memory_strain[region2]) / 2 + 
                self.MERGER_STRAIN_COST
            )
            merged_strain = min(1.0, merged_strain)
            
            # Coherence is reduced during merging due to integration challenges
            merged_coherence = (
                (self.memory_coherence[region1] + self.memory_coherence[region2]) / 2 * 
                (1.0 - self.COHERENCE_REDUCTION_ON_MERGE)
            )
            merged_coherence = max(0.0, min(1.0, merged_coherence))
            
            # Entropy increases during merging
            merged_entropy = (
                (self.memory_entropy[region1] + self.memory_entropy[region2]) / 2 + 
                self.ENTROPY_INCREASE_ON_MERGE
            )
            
            # Combine metadata
            merged_metadata = {}
            merged_metadata.update(self.region_metadata.get(region1, {}))
            merged_metadata.update(self.region_metadata.get(region2, {}))
            merged_metadata['merged_from'] = [region1, region2]
            merged_metadata['merge_timestamp'] = self._get_timestamp()
            
            # Register the new region
            self.register_memory_region(
                new_region_name, 
                initial_strain=merged_strain,
                initial_coherence=merged_coherence,
                initial_entropy=merged_entropy,
                metadata=merged_metadata
            )
            
            # Transfer connections from original regions to the new region
            self._transfer_connections_on_merge(region1, region2, new_region_name)
            
            # Remove the original regions
            self._remove_region(region1)
            self._remove_region(region2)
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            logger.info(f"Merged regions '{region1}' and '{region2}' into '{new_region_name}'")
            return True
        except ValueError as e:
            logger.warning(f"Failed to merge regions: {e}")
            return False
        except Exception as e:
            logger.error(f"Error merging regions: {e}")
            return False
    
    def _transfer_connections_on_merge(self, region1: str, region2: str, new_region: str) -> None:
        """
        Transfer connections from original regions to the merged region.
        
        Args:
            region1: First original region
            region2: Second original region
            new_region: New merged region
        """
        # Initialize connection dict for new region if it doesn't exist
        if new_region not in self.region_connectivity:
            self.region_connectivity[new_region] = {}
        
        # Process all regions
        for region in set(self.region_connectivity.keys()):
            # Skip the regions being merged
            if region == region1 or region == region2 or region == new_region:
                continue
            
            # Get connection strengths to the original regions
            conn1 = self.region_connectivity.get(region, {}).get(region1, 0.0)
            conn2 = self.region_connectivity.get(region, {}).get(region2, 0.0)
            
            if conn1 > 0.0 or conn2 > 0.0:
                # Take the stronger connection, slightly weakened by the merge
                merged_connection = max(conn1, conn2) * 0.9
                
                # Create connection between region and new_region
                self.region_connectivity[region][new_region] = merged_connection
                self.region_connectivity[new_region][region] = merged_connection
    
    def _remove_region(self, region: str) -> None:
        """
        Remove a region from the memory field.
        
        Args:
            region: Region to remove
        """
        try:
            # Remove region properties
            if region in self.memory_strain:
                del self.memory_strain[region]
            if region in self.memory_coherence:
                del self.memory_coherence[region]
            if region in self.memory_entropy:
                del self.memory_entropy[region]
            if region in self.region_metadata:
                del self.region_metadata[region]
            
            # Remove connections to/from this region
            if region in self.region_connectivity:
                del self.region_connectivity[region]
            
            # Remove connections from other regions to this region
            for r in self.region_connectivity:
                if region in self.region_connectivity[r]:
                    del self.region_connectivity[r][region]
        except Exception as e:
            logger.error(f"Error removing region '{region}': {e}")
    
    def split_region(self, region: str, new_region1: str, new_region2: str,
                   split_ratio: float = 0.5) -> Tuple[bool, str, str]:
        """
        Split a memory region into two new regions.
        This represents differentiation or specialization of memory areas.
        
        Args:
            region: Region to split
            new_region1: Name for first new region
            new_region2: Name for second new region
            split_ratio: Ratio for distributing properties (0.0-1.0)
            
        Returns:
            Tuple[bool, str, str]: Success flag and names of created regions
            
        Raises:
            ValueError: If region doesn't exist or split is invalid
        """
        try:
            # Validate region
            if region not in self.memory_strain:
                raise ValueError(f"Region '{region}' not registered")
            
            # Validate new region names
            if new_region1 in self.memory_strain:
                raise ValueError(f"Region '{new_region1}' already exists")
            if new_region2 in self.memory_strain:
                raise ValueError(f"Region '{new_region2}' already exists")
            
            # Ensure split ratio is valid
            split_ratio = max(0.1, min(0.9, split_ratio))
            
            # Calculate split properties
            # Splitting increases strain
            strain1 = min(1.0, self.memory_strain[region] * split_ratio + self.SPLIT_STRAIN_COST)
            strain2 = min(1.0, self.memory_strain[region] * (1.0 - split_ratio) + self.SPLIT_STRAIN_COST)
            
            # Splitting reduces coherence
            coherence1 = max(0.0, self.memory_coherence[region] * split_ratio * (1.0 - self.COHERENCE_LOSS_ON_SPLIT))
            coherence2 = max(0.0, self.memory_coherence[region] * (1.0 - split_ratio) * (1.0 - self.COHERENCE_LOSS_ON_SPLIT))
            
            # Splitting increases entropy
            entropy1 = self.memory_entropy[region] * split_ratio + self.ENTROPY_GAIN_ON_SPLIT
            entropy2 = self.memory_entropy[region] * (1.0 - split_ratio) + self.ENTROPY_GAIN_ON_SPLIT
            
            # Create metadata for new regions
            metadata1 = self.region_metadata.get(region, {}).copy()
            metadata2 = self.region_metadata.get(region, {}).copy()
            
            # Add split information to metadata
            metadata1['split_from'] = region
            metadata1['split_ratio'] = split_ratio
            metadata1['split_timestamp'] = self._get_timestamp()
            metadata1['split_sibling'] = new_region2
            
            metadata2['split_from'] = region
            metadata2['split_ratio'] = 1.0 - split_ratio
            metadata2['split_timestamp'] = self._get_timestamp()
            metadata2['split_sibling'] = new_region1
            
            # Register the new regions
            self.register_memory_region(
                new_region1, 
                initial_strain=strain1,
                initial_coherence=coherence1,
                initial_entropy=entropy1,
                metadata=metadata1
            )
            
            self.register_memory_region(
                new_region2, 
                initial_strain=strain2,
                initial_coherence=coherence2,
                initial_entropy=entropy2,
                metadata=metadata2
            )
            
            # Connect the new regions to each other
            # Split regions maintain a strong connection
            self.connect_regions(new_region1, new_region2, 0.8)
            
            # Distribute connections from the original region
            self._transfer_connections_on_split(region, new_region1, new_region2, split_ratio)
            
            # Remove the original region
            self._remove_region(region)
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            logger.info(f"Split region '{region}' into '{new_region1}' and '{new_region2}'")
            return (True, new_region1, new_region2)
        except ValueError as e:
            logger.warning(f"Failed to split region: {e}")
            return (False, "", "")
        except Exception as e:
            logger.error(f"Error splitting region '{region}': {e}")
            return (False, "", "")
    
    def _transfer_connections_on_split(self, region: str, new_region1: str, 
                                     new_region2: str, split_ratio: float) -> None:
        """
        Transfer connections from original region to split regions.
        
        Args:
            region: Original region
            new_region1: First new region
            new_region2: Second new region
            split_ratio: Split ratio
        """
        try:
            # Distribute connections from the original region
            for connected_region, strength in self.region_connectivity.get(region, {}).items():
                # Connections are distributed based on split ratio
                strength1 = strength * split_ratio * 0.9
                strength2 = strength * (1.0 - split_ratio) * 0.9
                
                # Create connections to the new regions
                if connected_region != new_region1 and connected_region != new_region2:
                    self.connect_regions(new_region1, connected_region, strength1)
                    self.connect_regions(new_region2, connected_region, strength2)
        except Exception as e:
            logger.error(f"Error transferring connections on split: {e}")
    
    def _get_timestamp(self) -> float:
        """
        Get current timestamp for metadata.
        
        Returns:
            float: Current timestamp
        """
        try:
            return time.time()
        except Exception:
            return 0.0
    
    def apply_coherence_wave(self, source_region: str, intensity: float = 1.0,
                           max_distance: float = 5.0) -> Dict[str, float]:
        """
        Apply a coherence wave that propagates from a source region.
        This can increase coherence in connected regions.
        
        Args:
            source_region: Source region for the coherence wave
            intensity: Intensity of the wave (0.0-2.0)
            max_distance: Maximum effective distance for the wave
            
        Returns:
            Dict[str, float]: Coherence change for each affected region
            
        Raises:
            ValueError: If source region doesn't exist
        """
        try:
            # Validate source region
            if source_region not in self.memory_strain:
                raise ValueError(f"Region '{source_region}' not registered")
            
            # Clamp intensity to valid range
            intensity = max(0.0, min(2.0, intensity))
            
            # Source must have sufficient coherence to generate a wave
            source_coherence = self.memory_coherence[source_region]
            if source_coherence < 0.3:
                logger.info(f"Region '{source_region}' has insufficient coherence ({source_coherence:.2f}) to generate wave")
                return {source_region: 0.0}  # Not enough coherence to generate wave
            
            # Calculate effective intensity based on source coherence
            effective_intensity = intensity * source_coherence
            
            # Cap max distance to system limit
            max_distance = min(max_distance, self.max_coherence_wave_distance)
            
            # Source region loses some coherence by generating the wave
            coherence_cost = min(source_coherence, 0.1 * effective_intensity)
            self.memory_coherence[source_region] = max(0.0, source_coherence - coherence_cost)
            
            # Track coherence changes
            coherence_changes = {source_region: -coherence_cost}
            
            # Calculate distances from source to all other regions
            distances = self._calculate_region_distances(source_region)
            
            # Apply coherence wave effects based on distance
            for region, distance in distances.items():
                if distance > max_distance:
                    continue  # Too far to be affected
                
                # Calculate attenuation based on distance
                attenuation = 1.0 - (distance / max_distance)
                
                # Apply coherence increase
                region_strain = self.memory_strain[region]
                region_coherence = self.memory_coherence[region]
                
                # Strain reduces effect of coherence wave
                strain_factor = 1.0 - region_strain * 0.7
                
                # Calculate coherence gain
                coherence_gain = effective_intensity * attenuation * strain_factor * 0.2
                
                # Apply coherence gain
                new_coherence = min(1.0, region_coherence + coherence_gain)
                self.memory_coherence[region] = new_coherence
                
                # Record change
                coherence_changes[region] = coherence_gain
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            return coherence_changes
        except ValueError as e:
            logger.warning(str(e))
            return {source_region: 0.0}
        except Exception as e:
            logger.error(f"Error applying coherence wave from '{source_region}': {e}")
            return {source_region: 0.0}
    
    def _calculate_region_distances(self, source_region: str) -> Dict[str, float]:
        """
        Calculate distances from source region to all other regions using an
        optimized Dijkstra algorithm with a priority queue.
        
        Args:
            source_region: Source region
            
        Returns:
            Dict[str, float]: Mapping of region names to distances
        """
        try:
            # Initialize distances for direct connections
            distances = {}
            for region, strength in self.region_connectivity.get(source_region, {}).items():
                distances[region] = 1.0 / strength if strength > 0 else float('inf')
            
            # Use a priority queue for efficient minimum distance finding
            queue = [(dist, region) for region, dist in distances.items()]
            heapq.heapify(queue)
            
            visited = {source_region}  # Mark source as visited
            
            while queue:
                dist, current = heapq.heappop(queue)
                
                if current in visited:
                    continue
                    
                # Mark as visited
                visited.add(current)
                
                # Update distances through this region
                for neighbor, strength in self.region_connectivity.get(current, {}).items():
                    if neighbor == source_region or neighbor in visited:
                        continue
                    
                    edge_dist = 1.0 / strength if strength > 0 else float('inf')
                    new_dist = dist + edge_dist
                    
                    # Update if this path is shorter
                    if neighbor not in distances or new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(queue, (new_dist, neighbor))
            
            return distances
        except Exception as e:
            logger.error(f"Error calculating region distances: {e}")
            return {}
    
    def calculate_region_distance(self, region1: str, region2: str) -> float:
        """
        Calculate the distance between two regions.
        
        Args:
            region1: First region name
            region2: Second region name
            
        Returns:
            float: Distance between regions (inf if unreachable)
        """
        # If either region doesn't exist, return infinity
        if region1 not in self.memory_strain or region2 not in self.memory_strain:
            return float('inf')
        
        # If they're the same region, distance is 0
        if region1 == region2:
            return 0.0
        
        # Get all distances from region1
        distances = self._calculate_region_distances(region1)
        
        # Return distance to region2, or infinity if unreachable
        return distances.get(region2, float('inf'))
    
    def simulate_memory_resonance(self, regions: List[str], iterations: int = 20) -> Dict[str, List[float]]:
        """
        Simulate memory resonance effects between multiple regions.
        Resonance can lead to synchronization of coherence patterns.
        
        Args:
            regions: List of regions to include in resonance
            iterations: Number of iterations to simulate
            
        Returns:
            Dict[str, List[float]]: Coherence values at each iteration for each region
            
        Raises:
            ValueError: If regions are invalid
        """
        try:
            # Validate regions
            for region in regions:
                if region not in self.memory_strain:
                    raise ValueError(f"Region '{region}' not registered")
            
            if len(regions) < 2:
                raise ValueError("At least two regions required for resonance")
            
            # Cap iterations for performance
            iterations = min(iterations, self.max_iterations)
            
            # Track coherence over time
            coherence_history = {region: [self.memory_coherence[region]] for region in regions}
            
            # Calculate all pairwise connections
            connections = {}
            for i, r1 in enumerate(regions):
                for r2 in regions[i+1:]:
                    strength = self.region_connectivity.get(r1, {}).get(r2, 0.0)
                    if strength > 0.0:
                        connections[(r1, r2)] = strength
            
            # Run simulation
            for _ in range(iterations):
                # Calculate new coherence values
                new_coherence = {region: self.memory_coherence[region] for region in regions}
                
                # Process resonance effects
                for (r1, r2), strength in connections.items():
                    # Get current coherence values
                    c1 = self.memory_coherence[r1]
                    c2 = self.memory_coherence[r2]
                    
                    # Calculate resonance factor
                    # Closer coherence values resonate more strongly
                    coherence_diff = abs(c1 - c2)
                    resonance_factor = (1.0 - coherence_diff) * strength * 0.2
                    
                    # Calculate synchronization pull
                    # Regions pull each other's coherence toward the midpoint
                    midpoint = (c1 + c2) / 2
                    pull1 = (midpoint - c1) * resonance_factor
                    pull2 = (midpoint - c2) * resonance_factor
                    
                    # Update new coherence values
                    new_coherence[r1] += pull1
                    new_coherence[r2] += pull2
                
                # Apply natural processes
                for region in regions:
                    # Strain dampens coherence
                    damping = self.memory_strain[region] * 0.05
                    
                    # Entropy reduces coherence
                    entropy_effect = self.memory_entropy[region] * 0.02
                    
                    # Apply effects
                    new_coherence[region] = max(0.0, min(1.0, new_coherence[region] - damping - entropy_effect))
                    
                    # Update coherence
                    self.memory_coherence[region] = new_coherence[region]
                    
                    # Record history
                    coherence_history[region].append(new_coherence[region])
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            return coherence_history
        except ValueError as e:
            logger.warning(str(e))
            return {}
        except Exception as e:
            logger.error(f"Error simulating memory resonance: {e}")
            return {}
    
    def calculate_field_information_content(self) -> Dict[str, Any]:
        """
        Calculate the total information content of the memory field.
        
        Returns:
            Dict[str, Any]: Information content measures and statistics
        """
        try:
            if not self.memory_strain:
                return {"total_info": 0.0, "region_count": 0, "connection_count": 0}
            
            total_info = 0.0
            region_info = {}
            
            for region in self.memory_strain:
                # Information content is proportional to coherence and inversely proportional to entropy
                coherence = self.memory_coherence.get(region, 0.0)
                entropy = self.memory_entropy.get(region, 0.0)
                strain = self.memory_strain.get(region, 0.0)
                
                # Calculate region information content
                region_info_content = coherence / (1.0 + entropy * 0.5)
                
                # Strain reduces effective information content
                strain_factor = 1.0 - strain * 0.7
                region_info_content *= strain_factor
                
                # Track region info
                region_info[region] = region_info_content
                
                # Add region's information content to total
                total_info += region_info_content
            
            # Calculate connection information
            connection_info = 0.0
            connection_count = 0
            total_strength = 0.0
            
            for r1, connections in self.region_connectivity.items():
                for r2, strength in connections.items():
                    if r1 < r2:  # Count each connection only once
                        connection_count += 1
                        total_strength += strength
                        
                        # Information in connection depends on strength and connected regions' coherence
                        c1 = self.memory_coherence.get(r1, 0.0)
                        c2 = self.memory_coherence.get(r2, 0.0)
                        connection_info += strength * (c1 + c2) / 2 * 0.2
            
            # Calculate average connection strength
            avg_strength = total_strength / connection_count if connection_count > 0 else 0.0
            
            # Total information includes both region and connection information
            total_info += connection_info
            
            return {
                "total_info": total_info,
                "region_info": region_info,
                "connection_info": connection_info,
                "region_count": len(self.memory_strain),
                "connection_count": connection_count,
                "avg_connection_strength": avg_strength,
                "avg_info_per_region": total_info / len(self.memory_strain) if self.memory_strain else 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating field information content: {e}")
            return {"error": str(e), "total_info": 0.0}
    
    def get_field_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the memory field.
        
        Returns:
            Dict[str, Any]: Field statistics
        """
        try:
            if not self.memory_strain:
                return {"region_count": 0}
            
            # Calculate basic stats
            region_count = len(self.memory_strain)
            connection_count = sum(len(connections) for connections in self.region_connectivity.values()) // 2
            
            # Region property statistics
            strain_values = list(self.memory_strain.values())
            coherence_values = list(self.memory_coherence.values())
            entropy_values = list(self.memory_entropy.values())
            
            # Calculate average values
            avg_strain = sum(strain_values) / region_count if region_count > 0 else 0.0
            avg_coherence = sum(coherence_values) / region_count if region_count > 0 else 0.0
            avg_entropy = sum(entropy_values) / region_count if region_count > 0 else 0.0
            
            # Calculate max values
            max_strain = max(strain_values) if strain_values else 0.0
            max_coherence = max(coherence_values) if coherence_values else 0.0
            max_entropy = max(entropy_values) if entropy_values else 0.0
            
            # Calculate min values
            min_strain = min(strain_values) if strain_values else 0.0
            min_coherence = min(coherence_values) if coherence_values else 0.0
            min_entropy = min(entropy_values) if entropy_values else 0.0
            
            # Calculate standard deviations
            if region_count > 1:
                std_strain = np.std(strain_values)
                std_coherence = np.std(coherence_values)
                std_entropy = np.std(entropy_values)
            else:
                std_strain = std_coherence = std_entropy = 0.0
            
            # Connection statistics
            strengths = []
            for r1, connections in self.region_connectivity.items():
                for r2, strength in connections.items():
                    if r1 < r2:  # Count each connection only once
                        strengths.append(strength)
            
            avg_strength = sum(strengths) / len(strengths) if strengths else 0.0
            max_strength = max(strengths) if strengths else 0.0
            min_strength = min(strengths) if strengths else 0.0
            
            # Calculate connectivity information
            avg_connections_per_region = connection_count * 2 / region_count
            
            # Find regions with most connections
            region_connection_counts = {r: len(connections) for r, connections in self.region_connectivity.items()}
            most_connected = max(region_connection_counts.items(), key=lambda x: x[1]) if region_connection_counts else (None, 0)
            
            # Calculate field information content
            info_content = self.calculate_field_information_content()["total_info"]
            
            return {
                "region_count": region_count,
                "connection_count": connection_count,
                
                "strain": {
                    "average": avg_strain,
                    "max": max_strain,
                    "min": min_strain,
                    "std_dev": std_strain
                },
                
                "coherence": {
                    "average": avg_coherence,
                    "max": max_coherence,
                    "min": min_coherence,
                    "std_dev": std_coherence
                },
                
                "entropy": {
                    "average": avg_entropy,
                    "max": max_entropy,
                    "min": min_entropy,
                    "std_dev": std_entropy
                },
                
                "connections": {
                    "average_per_region": avg_connections_per_region,
                    "average_strength": avg_strength,
                    "max_strength": max_strength,
                    "min_strength": min_strength,
                    "most_connected_region": most_connected[0],
                    "most_connected_count": most_connected[1]
                },
                
                "information_content": info_content,
                
                "field_parameters": {
                    "memory_capacity": self.memory_capacity,
                    "strain_diffusion_rate": self.strain_diffusion_rate,
                    "coherence_coupling": self.coherence_coupling,
                    "memory_decay_rate": self.memory_decay_rate,
                    "baseline_entropy_increase": self.baseline_entropy_increase
                },
                
                "history_tracking": {
                    "enabled": self.enable_history,
                    "max_entries": self.max_history_entries,
                    "current_entries": len(self.history['timestamps']) if self.enable_history else 0
                }
            }
        except Exception as e:
            logger.error(f"Error getting field statistics: {e}")
            return {"error": str(e), "region_count": 0 if not self.memory_strain else len(self.memory_strain)}
    
    def defragment_field(self, regions: Optional[List[str]] = None, 
                       defrag_factor: float = 0.5) -> Dict[str, Any]:
        """
        Defragment the memory field to reduce entropy and improve coherence.
        
        Args:
            regions: List of regions to defragment (None for all)
            defrag_factor: Defragmentation factor (0.0-1.0)
            
        Returns:
            Dict[str, Any]: Defragmentation results
        """
        try:
            # Validate and prepare regions list
            if regions is None:
                regions = list(self.memory_strain.keys())
            else:
                # Filter out non-existent regions
                regions = [r for r in regions if r in self.memory_strain]
            
            if not regions:
                return {"regions_processed": 0}
            
            # Clamp defrag factor to valid range
            defrag_factor = max(0.0, min(1.0, defrag_factor))
            
            # Track changes
            results = {
                "regions_processed": len(regions),
                "entropy_reduction": {},
                "coherence_increase": {},
                "total_entropy_reduction": 0.0,
                "total_coherence_increase": 0.0
            }
            
            # Process each region
            for region in regions:
                # Current values
                current_entropy = self.memory_entropy[region]
                current_coherence = self.memory_coherence[region]
                
                # Calculate entropy reduction (proportional to current entropy and defrag factor)
                entropy_reduction = current_entropy * defrag_factor * 0.5
                new_entropy = max(0.0, current_entropy - entropy_reduction)
                
                # Calculate coherence increase (inversely proportional to current coherence)
                coherence_headroom = 1.0 - current_coherence
                coherence_increase = coherence_headroom * defrag_factor * 0.5
                new_coherence = min(1.0, current_coherence + coherence_increase)
                
                # Apply changes
                self.memory_entropy[region] = new_entropy
                self.memory_coherence[region] = new_coherence
                
                # Track changes
                results["entropy_reduction"][region] = entropy_reduction
                results["coherence_increase"][region] = coherence_increase
                results["total_entropy_reduction"] += entropy_reduction
                results["total_coherence_increase"] += coherence_increase
            
            # Calculate averages
            if regions:
                results["average_entropy_reduction"] = results["total_entropy_reduction"] / len(regions)
                results["average_coherence_increase"] = results["total_coherence_increase"] / len(regions)
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            return results
        except Exception as e:
            logger.error(f"Error defragmenting field: {e}")
            return {"error": str(e), "regions_processed": 0}
    
    def optimize_field_topology(self, target_metric: str = "information_content",
                             iterations: int = 10) -> Dict[str, Any]:
        """
        Optimize the field topology by adjusting connections to maximize a target metric.
        
        Args:
            target_metric: Metric to optimize ("information_content", "coherence", "entropy")
            iterations: Number of optimization iterations
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        try:
            # Validate parameters
            iterations = min(iterations, self.max_iterations)
            
            valid_metrics = ["information_content", "coherence", "entropy"]
            if target_metric not in valid_metrics:
                logger.warning(f"Invalid target metric: {target_metric}. Using 'information_content'")
                target_metric = "information_content"
            
            # Track optimization progress
            results = {
                "iterations": iterations,
                "target_metric": target_metric,
                "initial_value": 0.0,
                "final_value": 0.0,
                "changes": []
            }
            
            # Get initial metric value
            if target_metric == "information_content":
                initial_value = self.calculate_field_information_content()["total_info"]
            elif target_metric == "coherence":
                coherence_values = list(self.memory_coherence.values())
                initial_value = sum(coherence_values) / len(coherence_values) if coherence_values else 0.0
            elif target_metric == "entropy":
                entropy_values = list(self.memory_entropy.values())
                initial_value = sum(entropy_values) / len(entropy_values) if entropy_values else 0.0
            
            results["initial_value"] = initial_value
            current_value = initial_value
            
            # Run optimization iterations
            for i in range(iterations):
                # Try different topology adjustments
                best_change = None
                best_change_value = current_value
                
                # Try adjusting different connections
                for r1 in list(self.memory_strain.keys()):
                    for r2 in list(self.memory_strain.keys()):
                        if r1 == r2:
                            continue
                        
                        # Current connection strength
                        current_strength = self.region_connectivity.get(r1, {}).get(r2, 0.0)
                        
                        # Try increasing connection
                        if current_strength < 1.0:
                            # Backup current state
                            old_strength = current_strength
                            
                            # Try new strength
                            new_strength = min(1.0, current_strength + 0.2)
                            self.update_connection_strength(r1, r2, new_strength)
                            
                            # Evaluate new metric
                            if target_metric == "information_content":
                                new_value = self.calculate_field_information_content()["total_info"]
                            elif target_metric == "coherence":
                                coherence_values = list(self.memory_coherence.values())
                                new_value = sum(coherence_values) / len(coherence_values)
                            elif target_metric == "entropy":
                                entropy_values = list(self.memory_entropy.values())
                                new_value = -sum(entropy_values) / len(entropy_values)  # Negative because we want to minimize entropy
                            
                            # Check if this is better
                            if new_value > best_change_value:
                                best_change_value = new_value
                                best_change = ("increase", r1, r2, old_strength, new_strength)
                            
                            # Restore old value for next experiment
                            self.update_connection_strength(r1, r2, old_strength)
                        
                        # Try decreasing connection
                        if current_strength > 0.0:
                            # Backup current state
                            old_strength = current_strength
                            
                            # Try new strength
                            new_strength = max(0.0, current_strength - 0.2)
                            self.update_connection_strength(r1, r2, new_strength)
                            
                            # Evaluate new metric
                            if target_metric == "information_content":
                                new_value = self.calculate_field_information_content()["total_info"]
                            elif target_metric == "coherence":
                                coherence_values = list(self.memory_coherence.values())
                                new_value = sum(coherence_values) / len(coherence_values)
                            elif target_metric == "entropy":
                                entropy_values = list(self.memory_entropy.values())
                                new_value = -sum(entropy_values) / len(entropy_values)
                            
                            # Check if this is better
                            if new_value > best_change_value:
                                best_change_value = new_value
                                best_change = ("decrease", r1, r2, old_strength, new_strength)
                            
                            # Restore old value for next experiment
                            self.update_connection_strength(r1, r2, old_strength)
                
                # Apply the best change if we found one
                if best_change and best_change_value > current_value:
                    action, r1, r2, old_strength, new_strength = best_change
                    self.update_connection_strength(r1, r2, new_strength)
                    
                    current_value = best_change_value
                    results["changes"].append({
                        "iteration": i,
                        "action": action,
                        "region1": r1,
                        "region2": r2,
                        "old_strength": old_strength,
                        "new_strength": new_strength,
                        "new_value": current_value
                    })
                else:
                    # No improvement found
                    results["changes"].append({
                        "iteration": i,
                        "action": "none",
                        "new_value": current_value
                    })
            
            results["final_value"] = current_value
            results["improvement"] = current_value - initial_value
            results["improvement_percent"] = (current_value - initial_value) / initial_value * 100 if initial_value > 0 else 0
            
            # Record in history if tracking is enabled
            if self.enable_history:
                self._record_history()
            
            return results
        except Exception as e:
            logger.error(f"Error optimizing field topology: {e}")
            return {"error": str(e), "iterations": 0}
    
    def reset(self) -> None:
        """
        Reset the memory field to its initial state.
        
        This method clears all memory regions, connections, and history,
        returning the memory field to a clean state as if newly initialized.
        Used during simulation resets and cleanup operations.
        """
        try:
            # Clear all memory region data
            self.memory_strain.clear()
            self.memory_coherence.clear()
            self.memory_entropy.clear()
            self.region_connectivity.clear()
            self.region_metadata.clear()
            
            # Clear history if enabled
            if hasattr(self, 'history') and self.history:
                self.history.clear()
            
            # Reset field parameters to defaults
            self.memory_capacity = 1.0
            self.strain_diffusion_rate = 0.2
            self.coherence_coupling = 0.3
            self.memory_decay_rate = 0.05
            
            logger.debug("Memory field physics reset completed")
            
        except Exception as e:
            logger.error(f"Error resetting memory field physics: {e}")
            # Ensure critical structures are at least empty
            self.memory_strain = {}
            self.memory_coherence = {}
            self.memory_entropy = {}
            self.region_connectivity = {}
            self.region_metadata = {}