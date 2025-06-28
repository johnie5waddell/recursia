"""
Minimum Weight Perfect Matching (MWPM) Decoder
==============================================

Implementation of the optimal decoder for surface codes using minimum weight
perfect matching algorithm. This is the gold standard for surface code decoding
and achieves optimal performance at the cost of higher computational complexity.

Based on:
- Fowler et al., "Surface codes: Towards practical large-scale quantum computation" (2012)
- Edmonds, "Paths, trees, and flowers" (1965) - Original matching algorithm
- NetworkX implementation of Edmonds' blossom algorithm

Time Complexity: O(n³) where n is number of syndrome defects
Space Complexity: O(n²) for the matching graph
"""

import numpy as np
import networkx as nx
import time
from typing import List, Tuple, Dict, Set, Optional
from numba import jit
import logging

from .decoder_interface import BaseDecoder

logger = logging.getLogger(__name__)


@jit(nopython=True)
def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions on surface code lattice."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


@jit(nopython=True) 
def calculate_error_weight(path_length: int, error_rate: float) -> float:
    """Calculate the weight of an error chain based on path length and error rate."""
    return -np.log(error_rate) * path_length


class MWPMDecoder(BaseDecoder):
    """
    Minimum Weight Perfect Matching decoder for surface codes.
    
    This decoder finds the most likely error correction by formulating
    syndrome decoding as a minimum weight perfect matching problem.
    """
    
    def __init__(self, code_distance: int, error_rate: float = 0.001):
        """
        Initialize MWPM decoder for surface code.
        
        Args:
            code_distance: Distance of the surface code (odd integer)
            error_rate: Physical error rate for weight calculation
        """
        super().__init__(code_distance)
        
        if code_distance % 2 == 0:
            raise ValueError("Surface code distance must be odd")
            
        self.error_rate = error_rate
        self.lattice_size = code_distance
        
        # Pre-compute lattice structure
        self.stabilizer_positions = self._generate_stabilizer_positions()
        self.boundary_nodes = self._generate_boundary_nodes()
        
        # Cache for distance calculations
        self._distance_cache = {}
        
        logger.info(f"Initialized MWMP decoder for distance-{code_distance} surface code")
    
    def decode_surface_code(self, syndrome: List[int]) -> List[int]:
        """
        Decode surface code syndrome using minimum weight perfect matching.
        
        Args:
            syndrome: Binary syndrome vector indicating defect locations
            
        Returns:
            List of correction operations for each data qubit
        """
        start_time = time.time()
        
        try:
            # Convert syndrome to defect positions
            defects = self._syndrome_to_defects(syndrome)
            
            if len(defects) == 0:
                # No errors detected
                correction = [0] * self._get_num_data_qubits()
                self._record_decode_attempt(True, time.time() - start_time)
                return correction
            
            # Find minimum weight perfect matching
            matching = self._find_minimum_weight_matching(defects)
            
            # Convert matching to correction operations
            correction = self._matching_to_correction(matching, defects)
            
            decode_time = time.time() - start_time
            self._record_decode_attempt(True, decode_time)
            
            logger.debug(f"MWPM decode completed in {decode_time:.4f}s with {len(defects)} defects")
            return correction
            
        except Exception as e:
            logger.error(f"MWPM decoding failed: {e}")
            # Return identity correction on failure
            correction = [0] * self._get_num_data_qubits()
            self._record_decode_attempt(False, time.time() - start_time)
            return correction
    
    def _syndrome_to_defects(self, syndrome: List[int]) -> List[Tuple[int, int]]:
        """
        Convert binary syndrome vector to defect positions on lattice.
        
        Args:
            syndrome: Binary syndrome vector
            
        Returns:
            List of (row, col) positions where defects occur
        """
        defects = []
        
        for i, syndrome_bit in enumerate(syndrome):
            if syndrome_bit == 1 and i < len(self.stabilizer_positions):
                defects.append(self.stabilizer_positions[i])
                
        return defects
    
    def _find_minimum_weight_matching(self, defects: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Find minimum weight perfect matching of defects.
        
        Uses NetworkX implementation of Edmonds' blossom algorithm.
        
        Args:
            defects: List of defect positions
            
        Returns:
            List of matched defect pairs
        """
        if len(defects) == 0:
            return []
            
        # Ensure even number of defects by adding boundary
        if len(defects) % 2 == 1:
            defects = defects + [self._find_nearest_boundary(defects[-1])]
        
        # Build matching graph
        graph = nx.Graph()
        
        # Add all defects as nodes
        for i, defect in enumerate(defects):
            graph.add_node(i, pos=defect)
        
        # Add edges between all pairs with weights
        for i in range(len(defects)):
            for j in range(i + 1, len(defects)):
                distance = self._calculate_defect_distance(defects[i], defects[j])
                weight = calculate_error_weight(distance, self.error_rate)
                graph.add_edge(i, j, weight=weight)
        
        # Find minimum weight perfect matching
        matching_edges = nx.min_weight_matching(graph, weight='weight')
        
        # Convert edge indices back to defect pairs
        matching = []
        for edge in matching_edges:
            defect1 = defects[edge[0]]
            defect2 = defects[edge[1]]
            matching.append((defect1, defect2))
            
        return matching
    
    def _calculate_defect_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate distance between defects with caching.
        
        Args:
            pos1, pos2: Defect positions
            
        Returns:
            int: Distance between defects
        """
        # Use cache for performance
        cache_key = (min(pos1, pos2), max(pos1, pos2))
        if cache_key in self._distance_cache:
            return self._distance_cache[cache_key]
        
        # Calculate Manhattan distance on torus topology
        distance = manhattan_distance(pos1, pos2)
        
        # Consider periodic boundary conditions for surface code
        dx = abs(pos1[0] - pos2[0])
        dy = abs(pos1[1] - pos2[1])
        
        # Minimum distance considering torus wrap-around
        dx = min(dx, self.lattice_size - dx)
        dy = min(dy, self.lattice_size - dy)
        distance = dx + dy
        
        self._distance_cache[cache_key] = distance
        return distance
    
    def _matching_to_correction(self, matching: List[Tuple[Tuple[int, int], Tuple[int, int]]], 
                               defects: List[Tuple[int, int]]) -> List[int]:
        """
        Convert matching to correction operations on data qubits.
        
        Args:
            matching: List of matched defect pairs
            defects: Original defect positions
            
        Returns:
            List of correction operations (0=no correction, 1=X correction)
        """
        n_data_qubits = self._get_num_data_qubits()
        correction = [0] * n_data_qubits
        
        for defect1, defect2 in matching:
            # Find path between defects
            path = self._find_correction_path(defect1, defect2)
            
            # Apply corrections along path
            for data_qubit_pos in path:
                data_qubit_index = self._position_to_data_qubit_index(data_qubit_pos)
                if 0 <= data_qubit_index < n_data_qubits:
                    correction[data_qubit_index] = 1 - correction[data_qubit_index]  # Flip
        
        return correction
    
    def _find_correction_path(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Find shortest path between two defects on lattice.
        
        Args:
            pos1, pos2: Defect positions
            
        Returns:
            List of data qubit positions along correction path
        """
        path = []
        
        # Simple Manhattan path (can be optimized with A*)
        current_row, current_col = pos1
        target_row, target_col = pos2
        
        # Move horizontally first
        while current_col != target_col:
            if current_col < target_col:
                current_col += 1
            else:
                current_col -= 1
            
            # Add data qubit position if it exists
            data_pos = (current_row, current_col)
            if self._is_data_qubit_position(data_pos):
                path.append(data_pos)
        
        # Then move vertically
        while current_row != target_row:
            if current_row < target_row:
                current_row += 1
            else:
                current_row -= 1
                
            # Add data qubit position if it exists
            data_pos = (current_row, current_col)
            if self._is_data_qubit_position(data_pos):
                path.append(data_pos)
        
        return path
    
    def _generate_stabilizer_positions(self) -> List[Tuple[int, int]]:
        """Generate positions of all stabilizers on the surface code lattice."""
        positions = []
        
        # X-type stabilizers (plaquette centers)
        for row in range(0, self.lattice_size - 1):
            for col in range(0, self.lattice_size - 1):
                if (row + col) % 2 == 0:  # Checkerboard pattern
                    positions.append((row, col))
        
        # Z-type stabilizers (vertex centers)  
        for row in range(0, self.lattice_size - 1):
            for col in range(0, self.lattice_size - 1):
                if (row + col) % 2 == 1:  # Complementary checkerboard
                    positions.append((row, col))
        
        return positions
    
    def _generate_boundary_nodes(self) -> List[Tuple[int, int]]:
        """Generate virtual boundary nodes for odd-parity syndrome correction."""
        boundary = []
        
        # Add boundary nodes around the lattice perimeter
        for i in range(self.lattice_size):
            boundary.extend([
                (-1, i),  # Top boundary
                (self.lattice_size, i),  # Bottom boundary  
                (i, -1),  # Left boundary
                (i, self.lattice_size)  # Right boundary
            ])
        
        return boundary
    
    def _find_nearest_boundary(self, defect: Tuple[int, int]) -> Tuple[int, int]:
        """Find nearest boundary node to a defect."""
        min_distance = float('inf')
        nearest_boundary = self.boundary_nodes[0]
        
        for boundary_node in self.boundary_nodes:
            distance = manhattan_distance(defect, boundary_node)
            if distance < min_distance:
                min_distance = distance
                nearest_boundary = boundary_node
                
        return nearest_boundary
    
    def _get_num_data_qubits(self) -> int:
        """Calculate number of data qubits in surface code."""
        # For distance d surface code: (d^2 + (d-1)^2) data qubits
        return self.lattice_size**2
    
    def _is_data_qubit_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position corresponds to a data qubit."""
        row, col = pos
        return (0 <= row < self.lattice_size and 
                0 <= col < self.lattice_size)
    
    def _position_to_data_qubit_index(self, pos: Tuple[int, int]) -> int:
        """Convert lattice position to data qubit index."""
        row, col = pos
        if not self._is_data_qubit_position(pos):
            return -1
        return row * self.lattice_size + col
    
    def get_decoder_stats(self) -> Dict[str, any]:
        """Get MWPM-specific decoder statistics."""
        base_stats = super().get_decoder_stats()
        base_stats.update({
            'lattice_size': self.lattice_size,
            'num_stabilizers': len(self.stabilizer_positions),
            'num_boundary_nodes': len(self.boundary_nodes),
            'cache_size': len(self._distance_cache),
            'error_rate': self.error_rate
        })
        return base_stats
    
    def clear_cache(self):
        """Clear distance calculation cache to free memory."""
        self._distance_cache.clear()
        logger.debug("Cleared MWPM decoder distance cache")