"""
Pytest configuration and shared fixtures
"""
import pytest
import numpy as np
import logging
import tempfile
import os
from pathlib import Path

# Import core modules for testing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from physics.universal_consciousness_field import UniversalConsciousnessField, ConsciousnessFieldState
from physics.recursive_observer_systems import RecursiveObserverHierarchy, QuantumObserver
from physics.qualia_memory_fields import QualiaMemoryField, QualiaType
from physics.recursive_simulation_architecture import RecursiveSimulationStack
from physics.consciousness_measurement_validation import ConsciousnessTestBattery
from tests import RANDOM_SEED, DEFAULT_TOLERANCE

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup global test environment"""
    np.random.seed(RANDOM_SEED)
    # Create temp directories
    os.makedirs("tests/temp", exist_ok=True)
    yield
    # Cleanup after tests
    import shutil
    if os.path.exists("tests/temp"):
        shutil.rmtree("tests/temp")

@pytest.fixture
def random_state():
    """Provide controlled random state for reproducible tests"""
    return np.random.RandomState(RANDOM_SEED)

@pytest.fixture
def tolerance():
    """Default numerical tolerance for tests"""
    return DEFAULT_TOLERANCE

@pytest.fixture
def sample_quantum_state(random_state):
    """Generate sample quantum state for testing"""
    state = random_state.normal(0, 1, 8) + 1j * random_state.normal(0, 1, 8)
    return state / np.sqrt(np.sum(np.abs(state)**2))

@pytest.fixture
def consciousness_field():
    """Create consciousness field for testing"""
    field = UniversalConsciousnessField(dimensions=32, max_recursion_depth=3)
    initial_psi = np.random.normal(0, 1, 32) + 1j * np.random.normal(0, 1, 32)
    initial_psi = initial_psi / np.sqrt(np.sum(np.abs(initial_psi)**2))
    field.initialize_field(initial_psi)
    return field

@pytest.fixture
def observer_hierarchy():
    """Create observer hierarchy for testing"""
    hierarchy = RecursiveObserverHierarchy(max_hierarchy_depth=3)
    hierarchy.add_observer("test_observer_1", consciousness_level=0.5)
    hierarchy.add_observer("test_observer_2", consciousness_level=0.7)
    return hierarchy

@pytest.fixture
def qualia_memory_field():
    """Create qualia memory field for testing"""
    field = QualiaMemoryField(field_dimensions=(8, 8, 8))
    # Add some test qualia
    field.create_quale("test_red", QualiaType.VISUAL_COLOR, 0.8, 0.6)
    field.create_quale("test_joy", QualiaType.EMOTIONAL_JOY, 0.7, 0.6)
    return field

@pytest.fixture
def simulation_stack():
    """Create simulation stack for testing"""
    return RecursiveSimulationStack(max_depth=3)

@pytest.fixture
def test_battery():
    """Create consciousness test battery"""
    return ConsciousnessTestBattery()

@pytest.fixture
def temp_file():
    """Create temporary file for testing"""
    fd, path = tempfile.mkstemp(dir="tests/temp")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def consciousness_state_minimal():
    """Minimal consciousness state for basic testing"""
    psi = np.array([1, 0, 0, 0], dtype=complex)
    return ConsciousnessFieldState(
        psi_consciousness=psi,
        phi_integrated=0.1,
        recursive_depth=1,
        memory_strain_tensor=np.zeros((4, 4)),
        observer_coupling={},
        time=0.0
    )

@pytest.fixture
def consciousness_state_complex(random_state):
    """Complex consciousness state for advanced testing"""
    psi = random_state.normal(0, 1, 16) + 1j * random_state.normal(0, 1, 16)
    psi = psi / np.sqrt(np.sum(np.abs(psi)**2))
    
    return ConsciousnessFieldState(
        psi_consciousness=psi,
        phi_integrated=0.5,
        recursive_depth=2,
        memory_strain_tensor=random_state.random((4, 4)) * 0.1,
        observer_coupling={"test_observer": 0.3},
        time=1.0
    )