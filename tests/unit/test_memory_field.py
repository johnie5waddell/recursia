"""
Unit tests for memory field module.
Tests memory strain, defragmentation, field dynamics, and qualia tagging.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.physics.memory_field_proper import (
    MemoryField,
    MemoryStrainTensor,
    MemoryDefragmenter,
    QualiaField,
    MemoryCoherenceInterface,
    MemoryFieldDynamics
)
from src.core.types import RecursiaFloat, RecursiaComplex
from src.physics.field.field_types import FieldConfiguration


class TestMemoryField:
    """Test basic MemoryField functionality."""
    
    @pytest.fixture
    def memory_field(self):
        """Create a test memory field."""
        return MemoryField(
            dimension=(10, 10, 10),
            strain_threshold=0.8,
            coherence_coupling=0.1
        )
        
    def test_field_initialization(self, memory_field):
        """Test proper field initialization."""
        assert memory_field.dimension == (10, 10, 10)
        assert memory_field.field_tensor.shape == (10, 10, 10)
        assert np.all(memory_field.field_tensor >= 0)
        assert memory_field.total_strain == 0.0
        
    def test_memory_allocation(self, memory_field):
        """Test memory allocation in field."""
        # Allocate memory region
        size = 2.0
        position = (5, 5, 5)
        
        allocated = memory_field.allocate_memory(
            position=position,
            size=size,
            coherence=0.9
        )
        
        assert allocated is not None
        assert memory_field.field_tensor[position] > 0
        assert memory_field.total_strain > 0
        
    def test_memory_deallocation(self, memory_field):
        """Test memory deallocation."""
        # First allocate
        position = (3, 3, 3)
        alloc_id = memory_field.allocate_memory(
            position=position,
            size=1.5,
            coherence=0.8
        )
        
        initial_strain = memory_field.total_strain
        
        # Then deallocate
        success = memory_field.deallocate_memory(alloc_id)
        
        assert success
        assert memory_field.total_strain < initial_strain
        assert memory_field.field_tensor[position] < 0.1
        
    def test_strain_accumulation(self, memory_field):
        """Test strain accumulation with multiple allocations."""
        positions = [(i, i, i) for i in range(5)]
        
        for pos in positions:
            memory_field.allocate_memory(
                position=pos,
                size=1.0,
                coherence=0.7
            )
            
        # Strain should accumulate
        assert memory_field.total_strain > len(positions) * 0.5
        
        # Check strain distribution
        strain_map = memory_field.calculate_strain_distribution()
        assert np.max(strain_map) > 0.5
        
    def test_memory_field_overflow(self, memory_field):
        """Test behavior when field reaches capacity."""
        # Fill field beyond threshold
        for i in range(20):
            try:
                memory_field.allocate_memory(
                    position=(i % 10, (i // 10) % 10, 0),
                    size=2.0,
                    coherence=0.6
                )
            except MemoryError:
                # Expected when field is full
                break
                
        assert memory_field.total_strain >= memory_field.strain_threshold
        
    def test_coherence_decay(self, memory_field):
        """Test coherence decay over time."""
        position = (5, 5, 5)
        memory_field.allocate_memory(
            position=position,
            size=1.0,
            coherence=1.0
        )
        
        initial_coherence = memory_field.get_coherence(position)
        
        # Simulate time evolution
        for _ in range(10):
            memory_field.evolve_coherence(time_step=0.1)
            
        final_coherence = memory_field.get_coherence(position)
        
        # Coherence should decay
        assert final_coherence < initial_coherence
        assert final_coherence > 0  # But not to zero immediately


class TestMemoryStrainTensor:
    """Test memory strain tensor calculations."""
    
    def test_strain_tensor_construction(self):
        """Test strain tensor initialization."""
        tensor = MemoryStrainTensor(shape=(8, 8, 8))
        
        assert tensor.shape == (8, 8, 8, 3, 3)  # 3x3 strain at each point
        assert np.all(tensor.components == 0)  # Initially unstrained
        
    def test_local_strain_calculation(self):
        """Test calculation of local strain."""
        tensor = MemoryStrainTensor(shape=(5, 5, 5))
        
        # Apply deformation
        deformation = np.random.randn(5, 5, 5, 3) * 0.1
        tensor.apply_deformation(deformation)
        
        # Calculate strain at center
        strain = tensor.local_strain(position=(2, 2, 2))
        
        assert strain.shape == (3, 3)
        # Strain should be symmetric
        assert np.allclose(strain, strain.T)
        
    def test_strain_energy_density(self):
        """Test strain energy calculations."""
        tensor = MemoryStrainTensor(shape=(4, 4, 4))
        
        # Create non-uniform strain
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    strain_magnitude = np.sqrt(i**2 + j**2 + k**2) / 10
                    tensor.set_local_strain(
                        position=(i, j, k),
                        strain=strain_magnitude * np.eye(3)
                    )
                    
        # Calculate energy density
        energy = tensor.strain_energy_density()
        
        assert energy.shape == (4, 4, 4)
        assert np.all(energy >= 0)
        # Energy should be highest at corners
        assert energy[3, 3, 3] > energy[0, 0, 0]
        
    def test_strain_propagation(self):
        """Test how strain propagates through field."""
        tensor = MemoryStrainTensor(shape=(10, 10, 10))
        
        # Apply point strain
        center = (5, 5, 5)
        tensor.apply_point_strain(
            position=center,
            magnitude=1.0,
            radius=2.0
        )
        
        # Check propagation
        # Strain at center should be highest
        center_strain = tensor.strain_magnitude(center)
        
        # Check neighbors
        for dx, dy, dz in [(1,0,0), (0,1,0), (0,0,1)]:
            neighbor = (center[0]+dx, center[1]+dy, center[2]+dz)
            neighbor_strain = tensor.strain_magnitude(neighbor)
            assert neighbor_strain < center_strain
            assert neighbor_strain > 0  # But non-zero
            
    def test_strain_relaxation(self):
        """Test strain relaxation dynamics."""
        tensor = MemoryStrainTensor(
            shape=(6, 6, 6),
            relaxation_time=1.0
        )
        
        # Create initial strain
        tensor.apply_uniform_strain(magnitude=0.5)
        initial_total = tensor.total_strain()
        
        # Relax
        for _ in range(10):
            tensor.relax(time_step=0.1)
            
        final_total = tensor.total_strain()
        
        # Should relax toward zero
        assert final_total < initial_total
        assert final_total > 0  # But not instantly


class TestMemoryDefragmenter:
    """Test memory defragmentation algorithms."""
    
    @pytest.fixture
    def fragmented_field(self):
        """Create a fragmented memory field."""
        field = MemoryField(dimension=(8, 8, 8))
        
        # Create fragmentation by allocating and deallocating
        allocs = []
        for i in range(10):
            pos = (i % 8, (i // 8) % 8, 0)
            aid = field.allocate_memory(pos, size=0.5, coherence=0.6)
            allocs.append(aid)
            
        # Deallocate every other one
        for i in range(0, 10, 2):
            field.deallocate_memory(allocs[i])
            
        return field
        
    def test_defragmentation_detection(self, fragmented_field):
        """Test detection of fragmentation."""
        defrag = MemoryDefragmenter(fragmented_field)
        
        frag_score = defrag.calculate_fragmentation_score()
        assert frag_score > 0.3  # Should detect fragmentation
        
        # Check fragment identification
        fragments = defrag.identify_fragments()
        assert len(fragments) > 0
        
    def test_memory_compaction(self, fragmented_field):
        """Test memory compaction algorithm."""
        defrag = MemoryDefragmenter(fragmented_field)
        
        initial_frag = defrag.calculate_fragmentation_score()
        initial_coherence = fragmented_field.average_coherence()
        
        # Perform compaction
        defrag.compact_memory()
        
        final_frag = defrag.calculate_fragmentation_score()
        final_coherence = fragmented_field.average_coherence()
        
        # Should reduce fragmentation
        assert final_frag < initial_frag
        # Should preserve or improve coherence
        assert final_coherence >= initial_coherence * 0.9
        
    def test_coherence_aware_defrag(self, fragmented_field):
        """Test coherence-preserving defragmentation."""
        defrag = MemoryDefragmenter(
            fragmented_field,
            preserve_coherence=True
        )
        
        # Mark some regions as high-coherence
        high_coherence_regions = [(2, 2, 0), (4, 4, 0)]
        for pos in high_coherence_regions:
            fragmented_field.set_coherence(pos, 0.95)
            
        # Defragment
        defrag.defragment()
        
        # High coherence regions should be preserved
        for pos in high_coherence_regions:
            assert fragmented_field.get_coherence(pos) > 0.9
            
    def test_incremental_defragmentation(self, fragmented_field):
        """Test incremental defragmentation process."""
        defrag = MemoryDefragmenter(
            fragmented_field,
            incremental=True
        )
        
        initial_frag = defrag.calculate_fragmentation_score()
        
        # Perform incremental steps
        improvements = []
        for _ in range(5):
            improved = defrag.incremental_defrag_step()
            improvements.append(improved)
            
        # Should make progress
        assert any(improvements)
        final_frag = defrag.calculate_fragmentation_score()
        assert final_frag < initial_frag


class TestQualiaField:
    """Test qualia field and tagging functionality."""
    
    def test_qualia_field_creation(self):
        """Test qualia field initialization."""
        qualia = QualiaField(
            dimension=(6, 6, 6),
            n_qualia_types=5
        )
        
        assert qualia.field.shape == (6, 6, 6, 5)
        assert np.all(qualia.field >= 0)
        assert np.all(qualia.field <= 1)
        
    def test_qualia_tagging(self):
        """Test tagging memory with qualia."""
        qualia = QualiaField(dimension=(8, 8, 8))
        
        # Tag a region with specific qualia
        position = (4, 4, 4)
        qualia_vector = np.array([0.8, 0.2, 0.5, 0.1, 0.9])
        
        qualia.tag_memory(
            position=position,
            qualia_type=qualia_vector,
            intensity=0.7
        )
        
        # Retrieve qualia
        retrieved = qualia.get_qualia(position)
        assert np.allclose(retrieved, qualia_vector * 0.7, atol=0.1)
        
    def test_qualia_propagation(self):
        """Test how qualia spreads through field."""
        qualia = QualiaField(
            dimension=(10, 10, 10),
            propagation_rate=0.3
        )
        
        # Tag center with strong qualia
        center = (5, 5, 5)
        qualia.tag_memory(
            position=center,
            qualia_type=np.array([1.0, 0, 0, 0, 0]),
            intensity=1.0
        )
        
        # Propagate
        for _ in range(5):
            qualia.propagate(time_step=0.1)
            
        # Check neighbors have received qualia
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if dx == dy == dz == 0:
                        continue
                    pos = (center[0]+dx, center[1]+dy, center[2]+dz)
                    neighbor_qualia = qualia.get_qualia(pos)
                    assert neighbor_qualia[0] > 0.1  # Some propagation
                    
    def test_qualia_mixing(self):
        """Test mixing of different qualia types."""
        qualia = QualiaField(dimension=(5, 5, 5))
        
        # Tag same position with different qualia
        position = (2, 2, 2)
        
        qualia.tag_memory(
            position=position,
            qualia_type=np.array([1, 0, 0, 0, 0]),
            intensity=0.5
        )
        
        qualia.tag_memory(
            position=position,
            qualia_type=np.array([0, 1, 0, 0, 0]),
            intensity=0.5
        )
        
        # Should mix
        mixed = qualia.get_qualia(position)
        assert mixed[0] > 0.2 and mixed[1] > 0.2
        
    def test_qualia_coherence_correlation(self):
        """Test correlation between qualia and coherence."""
        memory = MemoryField(dimension=(6, 6, 6))
        qualia = QualiaField(dimension=(6, 6, 6))
        
        # Create coherent memory region
        coherent_pos = (3, 3, 3)
        memory.allocate_memory(
            position=coherent_pos,
            size=1.0,
            coherence=0.9
        )
        
        # Tag with qualia
        qualia.tag_memory(
            position=coherent_pos,
            qualia_type=np.array([0, 0, 1, 0, 0]),
            intensity=0.8
        )
        
        # Qualia should be stronger in coherent regions
        qualia_strength = np.linalg.norm(qualia.get_qualia(coherent_pos))
        
        # Compare with low coherence region
        low_coh_pos = (0, 0, 0)
        memory.allocate_memory(
            position=low_coh_pos,
            size=1.0,
            coherence=0.1
        )
        qualia.tag_memory(
            position=low_coh_pos,
            qualia_type=np.array([0, 0, 1, 0, 0]),
            intensity=0.8
        )
        
        low_qualia_strength = np.linalg.norm(qualia.get_qualia(low_coh_pos))
        
        # High coherence should support stronger qualia
        assert qualia_strength > low_qualia_strength * 1.5


class TestMemoryFieldDynamics:
    """Test memory field evolution and dynamics."""
    
    def test_field_evolution_equation(self):
        """Test memory field evolution follows correct dynamics."""
        dynamics = MemoryFieldDynamics(
            dimension=(8, 8, 8),
            diffusion_constant=0.1,
            nonlinearity=0.05
        )
        
        # Create initial condition
        initial_field = dynamics.gaussian_blob(
            center=(4, 4, 4),
            width=1.0,
            amplitude=1.0
        )
        
        # Evolve
        field = initial_field.copy()
        for _ in range(10):
            field = dynamics.evolve(field, time_step=0.01)
            
        # Should diffuse outward
        assert np.max(field) < np.max(initial_field)
        # But conserve total "mass"
        assert abs(np.sum(field) - np.sum(initial_field)) < 0.1
        
    def test_memory_gravity_effects(self):
        """Test gravitational effects from memory strain."""
        dynamics = MemoryFieldDynamics(
            dimension=(10, 10, 10),
            enable_gravity=True,
            gravity_coupling=0.1
        )
        
        # Create high strain region
        strain_field = dynamics.create_strain_source(
            position=(5, 5, 5),
            mass=10.0
        )
        
        # Test particle in field
        test_position = np.array([7.0, 5.0, 5.0])
        acceleration = dynamics.gravitational_acceleration(
            position=test_position,
            strain_field=strain_field
        )
        
        # Should point toward source
        direction = np.array([5.0, 5.0, 5.0]) - test_position
        direction = direction / np.linalg.norm(direction)
        
        assert np.dot(acceleration, direction) > 0
        
    def test_soliton_formation(self):
        """Test formation of stable soliton structures."""
        dynamics = MemoryFieldDynamics(
            dimension=(20, 20, 20),
            nonlinearity=0.2,
            dispersion=0.01
        )
        
        # Initialize soliton profile
        soliton = dynamics.create_soliton(
            center=(10, 10, 10),
            velocity=(1.0, 0, 0),
            amplitude=2.0
        )
        
        # Evolve and check stability
        positions = []
        for t in range(20):
            soliton = dynamics.evolve_soliton(soliton, time_step=0.1)
            peak_pos = dynamics.find_peak_position(soliton)
            positions.append(peak_pos)
            
        # Should move with constant velocity
        velocities = np.diff(positions, axis=0)
        velocity_std = np.std(velocities, axis=0)
        
        assert np.all(velocity_std < 0.1)  # Stable propagation
        

# Edge cases and integration tests
class TestMemoryFieldEdgeCases:
    """Test edge cases and error handling."""
    
    def test_zero_dimension_field(self):
        """Test handling of zero-dimensional fields."""
        with pytest.raises(ValueError):
            MemoryField(dimension=(0, 10, 10))
            
    def test_negative_strain(self):
        """Test handling of negative strain values."""
        field = MemoryField(dimension=(5, 5, 5))
        
        with pytest.raises(ValueError):
            field.set_strain(position=(2, 2, 2), strain=-0.5)
            
    def test_out_of_bounds_access(self):
        """Test boundary handling."""
        field = MemoryField(dimension=(5, 5, 5))
        
        # Should handle gracefully
        result = field.get_coherence((10, 10, 10))
        assert result == 0.0  # Default for out of bounds
        
    def test_extreme_evolution_parameters(self):
        """Test with extreme parameter values."""
        # Very high strain threshold
        field1 = MemoryField(
            dimension=(4, 4, 4),
            strain_threshold=1000.0
        )
        # Should work but never trigger defrag
        assert field1 is not None
        
        # Very fast coherence decay
        field2 = MemoryField(
            dimension=(4, 4, 4),
            coherence_decay_rate=10.0
        )
        field2.allocate_memory((2, 2, 2), 1.0, coherence=1.0)
        field2.evolve_coherence(1.0)
        
        # Should decay to near zero
        assert field2.get_coherence((2, 2, 2)) < 0.01