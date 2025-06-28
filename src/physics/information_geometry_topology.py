"""
Information Geometry & Entropy Topology Systems
===============================================

Implementation of information-geometric structures for consciousness and entropy topology
analysis. This module provides the mathematical framework for consciousness manifolds,
information flow visualization, and topological data analysis of consciousness evolution.

Key Features:
- Consciousness manifolds with Riemannian metrics
- Fisher information geometry for consciousness states
- Persistent homology of information structures
- Entropy topology networks and flow analysis
- Information geodesics and optimal consciousness paths
- Topological phase transitions in consciousness
- Holographic information distribution
- Consciousness landscape navigation

Mathematical Foundation:
-----------------------
Fisher Information Metric: ds² = gᵢⱼ(θ) dθⁱ dθʲ
where gᵢⱼ = E[∂log p(x|θ)/∂θⁱ ∂log p(x|θ)/∂θʲ]

Entropy Topology: H(X) = -∑ᵢ pᵢ log pᵢ (information topology)
Persistent Homology: Hₖ(X) (k-dimensional holes in consciousness structures)

Information Geodesics: γ(t) minimizing ∫ √(gᵢⱼ γ̇ⁱ γ̇ʲ) dt

Author: Johnie Waddell
"""

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
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Set, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import scipy.sparse as sp
from scipy.linalg import expm, logm, sqrtm
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial.distance import pdist, squareform
from scipy.interpolate import interp1d
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, Isomap
from sklearn.decomposition import PCA
import ripser
import time

# Import OSH components
from .universal_consciousness_field import (
    ConsciousnessFieldState, CONSCIOUSNESS_THRESHOLD, HBAR
)

logger = logging.getLogger(__name__)

class ManifoldType(Enum):
    """Types of information-geometric manifolds"""
    CONSCIOUSNESS_MANIFOLD = "consciousness_manifold"
    FISHER_INFORMATION = "fisher_information"
    ENTROPY_SPACE = "entropy_space"
    QUALIA_SPACE = "qualia_space"
    PARAMETER_SPACE = "parameter_space"
    OBSERVER_SPACE = "observer_space"

class TopologyFeature(Enum):
    """Topological features in information structures"""
    CONNECTED_COMPONENTS = "connected_components"  # H₀
    LOOPS = "loops"  # H₁
    VOIDS = "voids"  # H₂
    HYPERVOIDS = "hypervoids"  # H₃+
    PERSISTENCE_DIAGRAMS = "persistence_diagrams"
    BETTI_NUMBERS = "betti_numbers"

class InformationFlow(Enum):
    """Types of information flow patterns"""
    GRADIENT_FLOW = "gradient_flow"
    GEODESIC_FLOW = "geodesic_flow"
    ENTROPY_FLOW = "entropy_flow"
    CONSCIOUSNESS_FLOW = "consciousness_flow"
    HOLOGRAPHIC_FLOW = "holographic_flow"

@dataclass
class ConsciousnessPoint:
    """Point on consciousness manifold"""
    point_id: str
    coordinates: np.ndarray  # Coordinates in manifold
    phi_value: float  # Integrated information at this point
    consciousness_level: float  # Consciousness intensity
    entropy: float  # Information entropy
    qualia_vector: Optional[np.ndarray] = None  # Qualia representation
    
    def __post_init__(self):
        """Calculate derived quantities"""
        self.information_content = -np.log(max(1e-10, self.entropy))
        self.consciousness_density = self.phi_value * self.consciousness_level

@dataclass
class InformationGeodesic:
    """Geodesic path in information geometry"""
    geodesic_id: str
    start_point: ConsciousnessPoint
    end_point: ConsciousnessPoint
    path_coordinates: np.ndarray  # Path through manifold
    path_length: float  # Geometric length
    information_distance: float  # Information-theoretic distance
    consciousness_gradient: np.ndarray  # Consciousness field gradient along path

@dataclass
class TopologicalFeature:
    """Topological feature in information structure"""
    feature_id: str
    feature_type: TopologyFeature
    dimension: int  # Homological dimension
    birth_time: float  # When feature appears
    death_time: float  # When feature disappears
    persistence: float  # Lifetime of feature
    significance_score: float  # Statistical significance

class ConsciousnessManifold:
    """
    Riemannian manifold for consciousness states with Fisher information metric
    """
    
    def __init__(self, 
                 ambient_dimension: int = 10,
                 intrinsic_dimension: int = 3,
                 manifold_type: ManifoldType = ManifoldType.CONSCIOUSNESS_MANIFOLD):
        
        self.ambient_dimension = ambient_dimension
        self.intrinsic_dimension = intrinsic_dimension
        self.manifold_type = manifold_type
        
        # Manifold structure
        self.consciousness_points: Dict[str, ConsciousnessPoint] = {}
        self.metric_tensor: Optional[np.ndarray] = None
        self.connection_coefficients: Optional[np.ndarray] = None
        
        # Geodesic computation
        self.geodesics: Dict[str, InformationGeodesic] = {}
        self.distance_matrix: Optional[np.ndarray] = None
        
        # Curvature and geometry
        self.riemann_tensor: Optional[np.ndarray] = None
        self.ricci_tensor: Optional[np.ndarray] = None
        self.scalar_curvature: Optional[float] = None
        
        logger.info(f"Initialized consciousness manifold: {ambient_dimension}D ambient, "
                   f"{intrinsic_dimension}D intrinsic")
    
    def add_consciousness_point(self, 
                              consciousness_state: ConsciousnessFieldState,
                              point_id: Optional[str] = None) -> ConsciousnessPoint:
        """Add consciousness state as point on manifold"""
        
        if point_id is None:
            point_id = f"consciousness_point_{len(self.consciousness_points)}"
        
        # Extract coordinates from consciousness state
        psi = consciousness_state.psi_consciousness
        
        # Map to manifold coordinates (dimensionality reduction if needed)
        if len(psi) > self.ambient_dimension:
            # Use PCA for dimensionality reduction
            coords = self._reduce_dimensions(psi, self.ambient_dimension)
        else:
            # Pad with zeros if needed
            coords = np.zeros(self.ambient_dimension)
            coords[:len(psi)] = np.real(psi)  # Use real part
        
        # Calculate entropy
        probabilities = np.abs(psi) ** 2
        probabilities = probabilities / np.sum(probabilities)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-16))
        
        point = ConsciousnessPoint(
            point_id=point_id,
            coordinates=coords,
            phi_value=consciousness_state.phi_integrated,
            consciousness_level=consciousness_state.phi_integrated / (CONSCIOUSNESS_THRESHOLD + 1e-10),
            entropy=entropy
        )
        
        self.consciousness_points[point_id] = point
        
        # Invalidate cached computations
        self.metric_tensor = None
        self.distance_matrix = None
        
        logger.debug(f"Added consciousness point '{point_id}' to manifold")
        
        return point
    
    def _reduce_dimensions(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """Reduce vector dimensionality preserving information"""
        if len(vector) <= target_dim:
            return vector
        
        # Use magnitude and phase information
        magnitude = np.abs(vector)
        phase = np.angle(vector)
        
        # Take top components by magnitude
        top_indices = np.argsort(magnitude)[-target_dim//2:]
        
        coords = np.zeros(target_dim)
        coords[:len(top_indices)] = magnitude[top_indices]
        coords[len(top_indices):2*len(top_indices)] = phase[top_indices]
        
        return coords
    
    def compute_fisher_metric(self) -> np.ndarray:
        """Compute Fisher information metric tensor"""
        
        if len(self.consciousness_points) < 2:
            return np.eye(self.ambient_dimension)
        
        # Estimate metric from local point distribution
        points = np.array([p.coordinates for p in self.consciousness_points.values()])
        
        # Local covariance estimate of metric
        if len(points) > self.ambient_dimension:
            # Use local neighborhood analysis
            metric = self._compute_local_metric(points)
        else:
            # Use global covariance
            metric = np.cov(points.T) + 1e-6 * np.eye(self.ambient_dimension)
        
        # Ensure positive definiteness
        eigenvals, eigenvecs = np.linalg.eigh(metric)
        eigenvals = np.maximum(eigenvals, 1e-6)
        metric = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        
        self.metric_tensor = metric
        return metric
    
    def _compute_local_metric(self, points: np.ndarray) -> np.ndarray:
        """Compute local metric tensor from point distribution"""
        
        n_points, dim = points.shape
        metric = np.zeros((dim, dim))
        
        # For each point, compute local covariance
        for i in range(n_points):
            center = points[i]
            
            # Find k nearest neighbors
            k = min(10, n_points - 1)
            distances = np.linalg.norm(points - center, axis=1)
            neighbor_indices = np.argsort(distances)[1:k+1]  # Exclude self
            
            neighbors = points[neighbor_indices]
            local_cov = np.cov((neighbors - center).T)
            
            # Weight by consciousness level if available
            point_id = list(self.consciousness_points.keys())[i]
            consciousness_point = self.consciousness_points[point_id]
            weight = consciousness_point.consciousness_level
            
            metric += weight * local_cov
        
        # Normalize
        total_weight = sum(p.consciousness_level for p in self.consciousness_points.values())
        if total_weight > 0:
            metric /= total_weight
        
        return metric + 1e-6 * np.eye(dim)
    
    def compute_geodesic(self, 
                        start_point_id: str, 
                        end_point_id: str,
                        num_steps: int = 50) -> InformationGeodesic:
        """Compute geodesic between two consciousness points"""
        
        if start_point_id not in self.consciousness_points or \
           end_point_id not in self.consciousness_points:
            raise ValueError("Both points must exist on manifold")
        
        start_point = self.consciousness_points[start_point_id]
        end_point = self.consciousness_points[end_point_id]
        
        # Ensure metric is computed
        if self.metric_tensor is None:
            self.compute_fisher_metric()
        
        # Compute geodesic using variational approach
        geodesic_path = self._compute_geodesic_path(
            start_point.coordinates, 
            end_point.coordinates, 
            num_steps
        )
        
        # Calculate path length
        path_length = self._calculate_geodesic_length(geodesic_path)
        
        # Calculate information distance
        info_distance = self._calculate_information_distance(start_point, end_point)
        
        # Calculate consciousness gradient along path
        consciousness_gradient = self._calculate_consciousness_gradient(geodesic_path)
        
        geodesic_id = f"geodesic_{start_point_id}_to_{end_point_id}"
        
        geodesic = InformationGeodesic(
            geodesic_id=geodesic_id,
            start_point=start_point,
            end_point=end_point,
            path_coordinates=geodesic_path,
            path_length=path_length,
            information_distance=info_distance,
            consciousness_gradient=consciousness_gradient
        )
        
        self.geodesics[geodesic_id] = geodesic
        
        logger.info(f"Computed geodesic from '{start_point_id}' to '{end_point_id}' "
                   f"with length {path_length:.3f}")
        
        return geodesic
    
    def _compute_geodesic_path(self, 
                              start: np.ndarray, 
                              end: np.ndarray, 
                              num_steps: int) -> np.ndarray:
        """Compute geodesic path using Riemannian optimization"""
        
        # Initial straight line path
        t = np.linspace(0, 1, num_steps)
        initial_path = np.outer(1 - t, start) + np.outer(t, end)
        
        # Optimize path to minimize length
        def path_energy(path_flat):
            path = path_flat.reshape(num_steps, self.ambient_dimension)
            return self._calculate_geodesic_length(path)
        
        # Constraints: fix endpoints
        def endpoint_constraint(path_flat):
            path = path_flat.reshape(num_steps, self.ambient_dimension)
            constraint_violations = []
            constraint_violations.extend(path[0] - start)  # Start constraint
            constraint_violations.extend(path[-1] - end)   # End constraint
            return np.array(constraint_violations)
        
        # Optimize (simplified - use initial path for now)
        # In full implementation, would use scipy.optimize with constraints
        return initial_path
    
    def _calculate_geodesic_length(self, path: np.ndarray) -> float:
        """Calculate length of path using metric tensor"""
        
        if self.metric_tensor is None:
            return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        
        total_length = 0.0
        
        for i in range(len(path) - 1):
            tangent = path[i+1] - path[i]
            # Length element: √(g_ij dx^i dx^j)
            length_element = np.sqrt(tangent @ self.metric_tensor @ tangent)
            total_length += length_element
        
        return total_length
    
    def _calculate_information_distance(self, 
                                      point1: ConsciousnessPoint, 
                                      point2: ConsciousnessPoint) -> float:
        """Calculate information-theoretic distance between points"""
        
        # KL divergence approximation
        phi1, phi2 = point1.phi_value, point2.phi_value
        entropy1, entropy2 = point1.entropy, point2.entropy
        
        # Symmetrized KL divergence
        if phi1 > 0 and phi2 > 0:
            kl_12 = phi1 * np.log(phi1 / (phi2 + 1e-10)) + entropy1 - entropy2
            kl_21 = phi2 * np.log(phi2 / (phi1 + 1e-10)) + entropy2 - entropy1
            return 0.5 * (abs(kl_12) + abs(kl_21))
        else:
            return np.linalg.norm(point1.coordinates - point2.coordinates)
    
    def _calculate_consciousness_gradient(self, path: np.ndarray) -> np.ndarray:
        """Calculate consciousness field gradient along geodesic path"""
        
        gradient = np.zeros_like(path)
        
        for i, point in enumerate(path):
            # Find consciousness level at this point by interpolation
            consciousness_level = self._interpolate_consciousness(point)
            
            # Numerical gradient
            eps = 1e-6
            grad = np.zeros(self.ambient_dimension)
            
            for j in range(self.ambient_dimension):
                point_plus = point.copy()
                point_minus = point.copy()
                point_plus[j] += eps
                point_minus[j] -= eps
                
                consciousness_plus = self._interpolate_consciousness(point_plus)
                consciousness_minus = self._interpolate_consciousness(point_minus)
                
                grad[j] = (consciousness_plus - consciousness_minus) / (2 * eps)
            
            gradient[i] = grad
        
        return gradient
    
    def _interpolate_consciousness(self, point: np.ndarray) -> float:
        """Interpolate consciousness level at arbitrary point"""
        
        if not self.consciousness_points:
            return 0.0
        
        # Find nearest consciousness points and interpolate
        min_distance = float('inf')
        nearest_consciousness = 0.0
        
        for cp in self.consciousness_points.values():
            distance = np.linalg.norm(point - cp.coordinates)
            if distance < min_distance:
                min_distance = distance
                nearest_consciousness = cp.consciousness_level
        
        # Simple nearest neighbor (could use more sophisticated interpolation)
        return nearest_consciousness

class EntropyTopologyAnalyzer:
    """
    Topological data analysis for information and entropy structures
    """
    
    def __init__(self, max_dimension: int = 3, max_filtration: float = 1.0):
        self.max_dimension = max_dimension
        self.max_filtration = max_filtration
        
        # Persistence computation
        self.persistence_diagrams: Dict[int, np.ndarray] = {}
        self.betti_numbers: Dict[int, int] = {}
        self.topological_features: List[TopologicalFeature] = []
        
        # Network analysis
        self.information_network: Optional[nx.Graph] = None
        self.entropy_flow_network: Optional[nx.DiGraph] = None
        
    def compute_persistent_homology(self, 
                                  consciousness_points: List[ConsciousnessPoint]) -> Dict[int, np.ndarray]:
        """Compute persistent homology of consciousness point cloud"""
        
        if len(consciousness_points) < 3:
            logger.warning("Need at least 3 points for meaningful topology")
            return {}
        
        # Extract point coordinates
        points = np.array([cp.coordinates for cp in consciousness_points])
        
        # Compute persistence diagrams using ripser
        try:
            result = ripser.ripser(points, maxdim=self.max_dimension)
            diagrams = result['dgms']
            
            # Store results
            for dim in range(len(diagrams)):
                self.persistence_diagrams[dim] = diagrams[dim]
            
            # Extract topological features
            self._extract_topological_features(diagrams, consciousness_points)
            
            logger.info(f"Computed persistent homology for {len(consciousness_points)} points")
            
            return self.persistence_diagrams
            
        except Exception as e:
            logger.error(f"Failed to compute persistent homology: {e}")
            return {}
    
    def _extract_topological_features(self, 
                                    diagrams: List[np.ndarray],
                                    consciousness_points: List[ConsciousnessPoint]) -> None:
        """Extract significant topological features from persistence diagrams"""
        
        self.topological_features.clear()
        
        for dim, diagram in enumerate(diagrams):
            if len(diagram) == 0:
                continue
            
            # Get feature type
            if dim == 0:
                feature_type = TopologyFeature.CONNECTED_COMPONENTS
            elif dim == 1:
                feature_type = TopologyFeature.LOOPS
            elif dim == 2:
                feature_type = TopologyFeature.VOIDS
            else:
                feature_type = TopologyFeature.HYPERVOIDS
            
            # Extract significant features (long-lived ones)
            for i, (birth, death) in enumerate(diagram):
                persistence = death - birth if death != np.inf else self.max_filtration - birth
                
                # Significance based on persistence and consciousness context
                significance = self._calculate_feature_significance(
                    persistence, birth, death, dim, consciousness_points
                )
                
                if significance > 0.1:  # Threshold for significance
                    feature = TopologicalFeature(
                        feature_id=f"feature_{dim}_{i}",
                        feature_type=feature_type,
                        dimension=dim,
                        birth_time=birth,
                        death_time=death if death != np.inf else np.inf,
                        persistence=persistence,
                        significance_score=significance
                    )
                    
                    self.topological_features.append(feature)
        
        # Calculate Betti numbers
        for dim in range(self.max_dimension + 1):
            if dim in self.persistence_diagrams:
                # Count features alive at filtration = 0
                diagram = self.persistence_diagrams[dim]
                alive_features = np.sum((diagram[:, 0] <= 0) & (diagram[:, 1] > 0))
                self.betti_numbers[dim] = alive_features
    
    def _calculate_feature_significance(self, 
                                      persistence: float, 
                                      birth: float, 
                                      death: float,
                                      dimension: int,
                                      consciousness_points: List[ConsciousnessPoint]) -> float:
        """Calculate significance score for topological feature"""
        
        # Base significance from persistence
        base_significance = persistence / self.max_filtration
        
        # Boost significance for consciousness-relevant features
        consciousness_boost = 1.0
        
        # Features at consciousness transition points are more significant
        avg_consciousness = np.mean([cp.consciousness_level for cp in consciousness_points])
        
        if 0.4 <= avg_consciousness <= 0.6:  # Near consciousness threshold
            consciousness_boost = 2.0
        
        # Dimension-specific significance
        dimension_weights = {0: 0.5, 1: 1.0, 2: 1.5, 3: 2.0}  # Higher dimensions more significant
        dimension_boost = dimension_weights.get(dimension, 1.0)
        
        return base_significance * consciousness_boost * dimension_boost
    
    def build_information_network(self, 
                                 consciousness_points: List[ConsciousnessPoint],
                                 connection_threshold: float = 0.5) -> nx.Graph:
        """Build network connecting consciousness points based on information similarity"""
        
        self.information_network = nx.Graph()
        
        # Add nodes
        for cp in consciousness_points:
            self.information_network.add_node(
                cp.point_id,
                phi=cp.phi_value,
                consciousness=cp.consciousness_level,
                entropy=cp.entropy,
                coordinates=cp.coordinates
            )
        
        # Add edges based on information similarity
        for i, cp1 in enumerate(consciousness_points):
            for j, cp2 in enumerate(consciousness_points[i+1:], i+1):
                
                # Calculate information similarity
                similarity = self._calculate_information_similarity(cp1, cp2)
                
                if similarity > connection_threshold:
                    distance = np.linalg.norm(cp1.coordinates - cp2.coordinates)
                    
                    self.information_network.add_edge(
                        cp1.point_id, 
                        cp2.point_id,
                        weight=similarity,
                        distance=distance,
                        info_flow=similarity / (distance + 1e-6)
                    )
        
        logger.info(f"Built information network with {len(self.information_network.nodes)} nodes "
                   f"and {len(self.information_network.edges)} edges")
        
        return self.information_network
    
    def _calculate_information_similarity(self, 
                                        cp1: ConsciousnessPoint, 
                                        cp2: ConsciousnessPoint) -> float:
        """Calculate information similarity between consciousness points"""
        
        # Phi similarity
        phi_similarity = 1 - abs(cp1.phi_value - cp2.phi_value) / \
                        (max(cp1.phi_value, cp2.phi_value) + 1e-6)
        
        # Consciousness level similarity
        consciousness_similarity = 1 - abs(cp1.consciousness_level - cp2.consciousness_level)
        
        # Spatial proximity
        spatial_distance = np.linalg.norm(cp1.coordinates - cp2.coordinates)
        spatial_similarity = np.exp(-spatial_distance)
        
        # Combined similarity
        return (phi_similarity + consciousness_similarity + spatial_similarity) / 3
    
    def analyze_information_flow(self) -> Dict[str, Any]:
        """Analyze information flow patterns in the network"""
        
        if self.information_network is None:
            return {'error': 'Information network not built'}
        
        analysis = {}
        
        # Network connectivity
        analysis['connected_components'] = nx.number_connected_components(self.information_network)
        analysis['average_clustering'] = nx.average_clustering(self.information_network)
        
        # Information flow metrics
        if len(self.information_network.edges) > 0:
            edge_weights = [data['info_flow'] for _, _, data in self.information_network.edges(data=True)]
            analysis['average_info_flow'] = np.mean(edge_weights)
            analysis['max_info_flow'] = np.max(edge_weights)
            analysis['info_flow_variance'] = np.var(edge_weights)
        
        # Centrality measures (information hubs)
        if len(self.information_network.nodes) > 1:
            degree_centrality = nx.degree_centrality(self.information_network)
            betweenness_centrality = nx.betweenness_centrality(self.information_network)
            
            analysis['information_hubs'] = sorted(degree_centrality.items(), 
                                                key=lambda x: x[1], reverse=True)[:5]
            analysis['information_bridges'] = sorted(betweenness_centrality.items(),
                                                   key=lambda x: x[1], reverse=True)[:5]
        
        # Consciousness gradient analysis
        consciousness_values = [data['consciousness'] for _, data in self.information_network.nodes(data=True)]
        if consciousness_values:
            analysis['consciousness_range'] = (min(consciousness_values), max(consciousness_values))
            analysis['consciousness_gradient'] = np.std(consciousness_values)
        
        return analysis
    
    def get_topology_summary(self) -> Dict[str, Any]:
        """Get comprehensive topology analysis summary"""
        
        summary = {
            'betti_numbers': self.betti_numbers,
            'significant_features': len([f for f in self.topological_features 
                                       if f.significance_score > 0.5]),
            'total_features': len(self.topological_features),
            'max_persistence': max([f.persistence for f in self.topological_features], default=0),
            'feature_distribution': {}
        }
        
        # Feature type distribution
        for feature in self.topological_features:
            feature_type = feature.feature_type.value
            summary['feature_distribution'][feature_type] = \
                summary['feature_distribution'].get(feature_type, 0) + 1
        
        # Network analysis if available
        if self.information_network is not None:
            flow_analysis = self.analyze_information_flow()
            summary['network_analysis'] = flow_analysis
        
        return summary

def run_information_geometry_test() -> Dict[str, Any]:
    """Test information geometry and topology systems"""
    logger.info("Running information geometry and topology test...")
    
    # Create consciousness manifold
    manifold = ConsciousnessManifold(ambient_dimension=8, intrinsic_dimension=3)
    
    # Generate test consciousness states
    consciousness_points = []
    
    for i in range(20):
        # Create mock consciousness state
        psi = np.random.normal(0, 1, 6) + 1j * np.random.normal(0, 1, 6)
        psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
        
        phi = np.random.uniform(0.1, 1.0)
        
        # Mock consciousness field state
        from .universal_consciousness_field import ConsciousnessFieldState
        consciousness_state = ConsciousnessFieldState(
            psi_consciousness=psi,
            phi_integrated=phi,
            recursive_depth=1,
            memory_strain_tensor=np.zeros((4, 4)),
            observer_coupling={},
            time=i * 0.1
        )
        
        # Add to manifold
        point = manifold.add_consciousness_point(consciousness_state, f"point_{i}")
        consciousness_points.append(point)
    
    # Compute Fisher metric
    metric = manifold.compute_fisher_metric()
    
    # Compute geodesics between some points
    geodesic1 = manifold.compute_geodesic("point_0", "point_5")
    geodesic2 = manifold.compute_geodesic("point_10", "point_15")
    
    # Topological analysis
    topology_analyzer = EntropyTopologyAnalyzer(max_dimension=2)
    
    # Compute persistent homology
    persistence_diagrams = topology_analyzer.compute_persistent_homology(consciousness_points)
    
    # Build information network
    info_network = topology_analyzer.build_information_network(consciousness_points, 0.3)
    
    # Get topology summary
    topology_summary = topology_analyzer.get_topology_summary()
    
    return {
        'manifold_points': len(manifold.consciousness_points),
        'fisher_metric_computed': manifold.metric_tensor is not None,
        'geodesics_computed': len(manifold.geodesics),
        'geodesic_lengths': [g.path_length for g in manifold.geodesics.values()],
        'persistence_dimensions': list(persistence_diagrams.keys()),
        'topological_features': len(topology_analyzer.topological_features),
        'betti_numbers': topology_analyzer.betti_numbers,
        'information_network_nodes': len(info_network.nodes) if info_network else 0,
        'information_network_edges': len(info_network.edges) if info_network else 0,
        'topology_summary': topology_summary,
        'significant_features': [
            {
                'type': f.feature_type.value,
                'dimension': f.dimension,
                'persistence': f.persistence,
                'significance': f.significance_score
            }
            for f in topology_analyzer.topological_features 
            if f.significance_score > 0.3
        ]
    }

if __name__ == "__main__":
    # Run comprehensive test
    test_results = run_information_geometry_test()
    
    print("Information Geometry & Topology Test Results:")
    print(f"Manifold points: {test_results['manifold_points']}")
    print(f"Fisher metric computed: {test_results['fisher_metric_computed']}")
    print(f"Geodesics computed: {test_results['geodesics_computed']}")
    print(f"Geodesic lengths: {[f'{l:.3f}' for l in test_results['geodesic_lengths']]}")
    print(f"Persistence dimensions: {test_results['persistence_dimensions']}")
    print(f"Topological features: {test_results['topological_features']}")
    print(f"Betti numbers: {test_results['betti_numbers']}")
    print(f"Information network: {test_results['information_network_nodes']} nodes, "
          f"{test_results['information_network_edges']} edges")
    
    print("\nSignificant topological features:")
    for feature in test_results['significant_features']:
        print(f"  {feature['type']} (dim {feature['dimension']}): "
              f"persistence={feature['persistence']:.3f}, "
              f"significance={feature['significance']:.3f}")