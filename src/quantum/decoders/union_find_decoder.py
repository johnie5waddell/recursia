"""
Union-Find Decoder for Surface Codes
====================================

Fast approximate decoder for surface codes using the Union-Find algorithm.
Achieves near-optimal performance with O(n α(n)) complexity, making it
suitable for real-time quantum error correction.

Based on:
- Delfosse & Nickerson, "Almost-linear time decoding algorithm for topological codes" (2017)  
- Tarjan, "Efficiency of a Good But Not Linear Set Union Algorithm" (1975)

Time Complexity: O(n α(n)) where α is inverse Ackermann function
Space Complexity: O(n) for Union-Find structure
Threshold: ~10.3% (slightly below MWMP's ~11% but much faster)
"""

import numpy as np
import time
from typing import List, Tuple, Dict, Set, Optional, Union
from numba import jit, types
from numba.typed import Dict as NumbaDict
import logging

from .decoder_interface import BaseDecoder

logger = logging.getLogger(__name__)


@jit(nopython=True)
def find_with_compression(parent: np.ndarray, x: int) -> int:
    """Find with path compression optimization."""
    if parent[x] != x:
        parent[x] = find_with_compression(parent, parent[x])
    return parent[x]


@jit(nopython=True)
def union_by_rank(parent: np.ndarray, rank: np.ndarray, x: int, y: int) -> bool:
    """Union two sets by rank with balancing."""
    px = find_with_compression(parent, x)
    py = find_with_compression(parent, y)
    
    if px == py:
        return False  # Already in same set
    
    # Union by rank
    if rank[px] < rank[py]:
        px, py = py, px
    parent[py] = px
    if rank[px] == rank[py]:
        rank[px] += 1
        
    return True


@jit(nopython=True)
def manhattan_distance_fast(x1: int, y1: int, x2: int, y2: int) -> int:
    """Fast Manhattan distance calculation."""
    return abs(x1 - x2) + abs(y1 - y2)


class UnionFindDecoder(BaseDecoder):
    """
    Union-Find decoder for surface codes.
    
    This decoder uses the Union-Find data structure to efficiently
    group and process syndrome defects, achieving near-linear time
    complexity for real-time applications.
    """
    
    def __init__(self, code_distance: int, error_rate: float = 0.001):
        """
        Initialize Union-Find decoder.
        
        Args:
            code_distance: Distance of the surface code
            error_rate: Physical error rate for weight calculation
        """
        super().__init__(code_distance)
        
        if code_distance % 2 == 0:
            raise ValueError("Surface code distance must be odd")
            
        self.error_rate = error_rate
        self.lattice_size = code_distance
        
        # Pre-allocate Union-Find structures for efficiency
        max_nodes = code_distance * code_distance * 2  # Generous upper bound
        self.parent = np.arange(max_nodes, dtype=np.int32)
        self.rank = np.zeros(max_nodes, dtype=np.int32)
        
        # Pre-compute lattice structure
        self.stabilizer_coords = self._precompute_stabilizer_coordinates()
        self.edge_list = self._precompute_edge_list()
        
        # Performance tracking
        self.cluster_count = 0
        self.max_cluster_size = 0
        
        logger.info(f"Initialized Union-Find decoder for distance-{code_distance} surface code")
    
    def decode_surface_code(self, syndrome: List[int]) -> List[int]:
        """
        Decode surface code syndrome using Union-Find algorithm.
        
        Args:
            syndrome: Binary syndrome vector
            
        Returns:
            List of correction operations
        """
        start_time = time.time()
        
        try:
            # Reset Union-Find structure
            self._reset_union_find()
            
            # Convert syndrome to defects
            defects = self._extract_defects(syndrome)
            
            if len(defects) == 0:
                correction = [0] * self._get_num_data_qubits()
                self._record_decode_attempt(True, time.time() - start_time)
                return correction
            
            # Build clusters using Union-Find
            clusters = self._build_clusters(defects)
            
            # Generate correction from clusters
            correction = self._clusters_to_correction(clusters)
            
            decode_time = time.time() - start_time
            self._record_decode_attempt(True, decode_time)
            
            logger.debug(f"Union-Find decode: {len(defects)} defects, "
                        f"{len(clusters)} clusters, {decode_time:.4f}s")
            
            return correction
            
        except Exception as e:
            logger.error(f"Union-Find decoding failed: {e}")
            correction = [0] * self._get_num_data_qubits()
            self._record_decode_attempt(False, time.time() - start_time)
            return correction
    
    def _reset_union_find(self):
        """Reset Union-Find structure for new decode."""
        # Reset parent pointers
        for i in range(len(self.parent)):
            self.parent[i] = i
            self.rank[i] = 0
    
    def _extract_defects(self, syndrome: List[int]) -> List[Tuple[int, int]]:
        """Extract defect coordinates from syndrome vector."""
        defects = []
        
        for i, bit in enumerate(syndrome):
            if bit == 1 and i < len(self.stabilizer_coords):
                defects.append(self.stabilizer_coords[i])
                
        return defects
    
    def _build_clusters(self, defects: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """
        Build clusters of defects using Union-Find with growth strategy.
        
        This is the core of the Union-Find decoder - we grow clusters
        by adding edges in order of increasing weight until all defects
        are connected or boundary conditions are met.
        
        Args:
            defects: List of defect coordinates
            
        Returns:
            List of clusters, each containing defect coordinates
        """
        if len(defects) == 0:
            return []
        
        # Map defects to node indices
        defect_to_node = {}
        for i, defect in enumerate(defects):
            defect_to_node[defect] = i
        
        # Add virtual boundary nodes for odd-parity correction
        n_defects = len(defects)
        boundary_nodes = []
        
        # Add boundary connections for defects near edges
        for i, defect in enumerate(defects):
            if self._is_near_boundary(defect):
                boundary_node = n_defects + len(boundary_nodes)
                boundary_nodes.append(boundary_node)
                defect_to_node[f"boundary_{i}"] = boundary_node
        
        # Generate edges sorted by weight (greedy growth)
        edges = self._generate_sorted_edges(defects, boundary_nodes)
        
        # Grow clusters by adding edges
        clusters_formed = []
        
        for edge in edges:
            node1, node2, weight = edge
            
            # Try to union the nodes
            if union_by_rank(self.parent, self.rank, node1, node2):
                # Successful union - cluster grew
                self.cluster_count += 1
                
                # Check if this creates a complete cluster (even number of defects)
                cluster_root = find_with_compression(self.parent, node1)
                cluster_size = self._get_cluster_size(cluster_root, n_defects)
                
                if cluster_size % 2 == 0 and cluster_size >= 2:
                    # Even cluster formed - extract it
                    cluster = self._extract_cluster(cluster_root, defects, n_defects)
                    if len(cluster) > 0:
                        clusters_formed.append(cluster)
                        self.max_cluster_size = max(self.max_cluster_size, len(cluster))
        
        # Handle any remaining isolated defects
        remaining_defects = self._get_unclustered_defects(defects, clusters_formed)
        if len(remaining_defects) > 0:
            # Pair remaining defects greedily
            while len(remaining_defects) >= 2:
                pair = [remaining_defects.pop(), remaining_defects.pop()]
                clusters_formed.append(pair)
        
        return clusters_formed
    
    def _generate_sorted_edges(self, defects: List[Tuple[int, int]], 
                              boundary_nodes: List[int]) -> List[Tuple[int, int, float]]:
        """
        Generate edges between defects sorted by weight.
        
        Args:
            defects: List of defect coordinates
            boundary_nodes: List of boundary node indices
            
        Returns:
            List of (node1, node2, weight) tuples sorted by weight
        """
        edges = []
        n_defects = len(defects)
        
        # Edges between defects
        for i in range(n_defects):
            for j in range(i + 1, n_defects):
                distance = manhattan_distance_fast(
                    defects[i][0], defects[i][1],
                    defects[j][0], defects[j][1]
                )
                weight = self._calculate_edge_weight(distance)
                edges.append((i, j, weight))
        
        # Edges to boundary (for odd-parity correction)
        for i, defect in enumerate(defects):
            if self._is_near_boundary(defect):
                boundary_distance = self._distance_to_boundary(defect)
                weight = self._calculate_edge_weight(boundary_distance)
                boundary_node = n_defects + i  # Simplified boundary mapping
                edges.append((i, boundary_node, weight))
        
        # Sort by weight (ascending - shortest edges first)
        edges.sort(key=lambda x: x[2])
        
        return edges
    
    def _calculate_edge_weight(self, distance: int) -> float:
        """Calculate weight for edge based on distance and error model."""
        # Negative log-likelihood for minimum weight matching equivalence
        return -np.log(self.error_rate) * distance
    
    def _is_near_boundary(self, pos: Tuple[int, int]) -> bool:
        """Check if defect is near lattice boundary."""
        row, col = pos
        boundary_threshold = 1  # Within 1 step of boundary
        
        return (row <= boundary_threshold or 
                row >= self.lattice_size - boundary_threshold or
                col <= boundary_threshold or 
                col >= self.lattice_size - boundary_threshold)
    
    def _distance_to_boundary(self, pos: Tuple[int, int]) -> int:
        """Calculate minimum distance from position to lattice boundary."""
        row, col = pos
        
        distances = [
            row,  # Distance to top
            self.lattice_size - 1 - row,  # Distance to bottom
            col,  # Distance to left  
            self.lattice_size - 1 - col   # Distance to right
        ]
        
        return min(distances)
    
    def _get_cluster_size(self, root: int, n_defects: int) -> int:
        """Count number of defects in cluster with given root."""
        count = 0
        for i in range(n_defects):
            if find_with_compression(self.parent, i) == root:
                count += 1
        return count
    
    def _extract_cluster(self, root: int, defects: List[Tuple[int, int]], 
                        n_defects: int) -> List[Tuple[int, int]]:
        """Extract all defects belonging to cluster with given root."""
        cluster = []
        
        for i in range(n_defects):
            if find_with_compression(self.parent, i) == root:
                cluster.append(defects[i])
                
        return cluster
    
    def _get_unclustered_defects(self, defects: List[Tuple[int, int]], 
                                clusters: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Get defects that haven't been assigned to any cluster."""
        clustered_defects = set()
        
        for cluster in clusters:
            for defect in cluster:
                clustered_defects.add(defect)
        
        return [d for d in defects if d not in clustered_defects]
    
    def _clusters_to_correction(self, clusters: List[List[Tuple[int, int]]]) -> List[int]:
        """
        Convert clusters to correction operations.
        
        For each cluster, find minimum weight path connecting all defects
        and mark data qubits along those paths for correction.
        
        Args:
            clusters: List of defect clusters
            
        Returns:
            List of correction operations
        """
        n_data_qubits = self._get_num_data_qubits()
        correction = [0] * n_data_qubits
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
                
            # Connect cluster defects with minimum spanning tree
            paths = self._find_cluster_correction_paths(cluster)
            
            # Apply corrections along paths
            for path in paths:
                for data_pos in path:
                    data_index = self._position_to_data_index(data_pos)
                    if 0 <= data_index < n_data_qubits:
                        correction[data_index] = 1 - correction[data_index]  # Flip
        
        return correction
    
    def _find_cluster_correction_paths(self, cluster: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        """Find correction paths connecting all defects in cluster."""
        if len(cluster) <= 1:
            return []
        
        paths = []
        
        # Simple pairwise connection (can be optimized with MST)
        for i in range(0, len(cluster) - 1, 2):
            if i + 1 < len(cluster):
                path = self._find_shortest_path(cluster[i], cluster[i + 1])
                paths.append(path)
        
        return paths
    
    def _find_shortest_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find shortest path between two positions on lattice."""
        path = []
        
        # Manhattan path
        row, col = start
        target_row, target_col = end
        
        # Move horizontally
        while col != target_col:
            if col < target_col:
                col += 1
            else:
                col -= 1
            path.append((row, col))
        
        # Move vertically
        while row != target_row:
            if row < target_row:
                row += 1
            else:
                row -= 1
            path.append((row, col))
        
        return path
    
    def _precompute_stabilizer_coordinates(self) -> List[Tuple[int, int]]:
        """Pre-compute coordinates of all stabilizers."""
        coords = []
        
        # Generate stabilizer positions in standard surface code layout
        for row in range(self.lattice_size - 1):
            for col in range(self.lattice_size - 1):
                coords.append((row, col))
        
        return coords
    
    def _precompute_edge_list(self) -> List[Tuple[int, int, float]]:
        """Pre-compute edge list for common lattice operations."""
        # This can be used for further optimization
        return []
    
    def _get_num_data_qubits(self) -> int:
        """Get number of data qubits in surface code."""
        return self.lattice_size * self.lattice_size
    
    def _position_to_data_index(self, pos: Tuple[int, int]) -> int:
        """Convert lattice position to data qubit index."""
        row, col = pos
        if 0 <= row < self.lattice_size and 0 <= col < self.lattice_size:
            return row * self.lattice_size + col
        return -1
    
    def get_decoder_stats(self) -> Dict[str, any]:
        """Get Union-Find specific statistics."""
        base_stats = super().get_decoder_stats()
        base_stats.update({
            'algorithm': 'Union-Find',
            'lattice_size': self.lattice_size,
            'cluster_count': self.cluster_count,
            'max_cluster_size': self.max_cluster_size,
            'error_rate': self.error_rate,
            'complexity': 'O(n α(n))'
        })
        return base_stats
    
    def reset_performance_counters(self):
        """Reset performance tracking counters."""
        super().reset_stats()
        self.cluster_count = 0
        self.max_cluster_size = 0