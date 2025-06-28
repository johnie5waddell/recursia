"""
Test Suite for Comprehensive Memory Management System
====================================================

Tests the integration of low-level and high-level memory management.
"""

import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch

from src.core.memory_management_system import (
    GlobalMemoryManager,
    MemoryManagedComponent,
    MemoryPolicy,
    QuantumStateMemoryManager,
    ObserverRegistryMemoryManager,
    MeasurementHistoryMemoryManager,
    setup_memory_management,
    get_global_memory_manager
)
from src.core.memory_integration import (
    setup_complete_memory_management,
    get_memory_statistics,
    optimize_memory_for_long_simulations
)


class TestMemoryManagedComponent:
    """Test base memory managed component."""
    
    def test_abstract_methods(self):
        """Test that abstract methods must be implemented."""
        with pytest.raises(TypeError):
            MemoryManagedComponent()


class TestGlobalMemoryManager:
    """Test the global memory management system."""
    
    def test_initialization(self):
        """Test memory manager initialization."""
        manager = GlobalMemoryManager()
        assert manager._running == False
        assert len(manager._components) == 0
        assert len(manager._policies) == 0
        
    def test_register_component(self):
        """Test component registration."""
        manager = GlobalMemoryManager()
        
        # Create mock component
        component = Mock(spec=MemoryManagedComponent)
        component.get_memory_usage.return_value = {
            'item_count': 10,
            'memory_mb': 1.0,
            'last_cleanup_time': time.time()
        }
        
        # Register component
        manager.register_component('test', component)
        
        assert 'test' in manager._components
        assert 'test' in manager._policies
        assert isinstance(manager._policies['test'], MemoryPolicy)
        
    def test_unregister_component(self):
        """Test component unregistration."""
        manager = GlobalMemoryManager()
        component = Mock(spec=MemoryManagedComponent)
        
        manager.register_component('test', component)
        manager.unregister_component('test')
        
        assert 'test' not in manager._components
        assert 'test' not in manager._policies
        
    def test_start_stop(self):
        """Test starting and stopping the manager."""
        manager = GlobalMemoryManager()
        
        # Start manager
        manager.start()
        assert manager._running == True
        assert manager._cleanup_thread is not None
        assert manager._cleanup_thread.is_alive()
        
        # Stop manager
        manager.stop()
        assert manager._running == False
        time.sleep(0.1)  # Give thread time to stop
        assert not manager._cleanup_thread.is_alive()
        
    def test_regular_cleanup(self):
        """Test regular cleanup based on policies."""
        manager = GlobalMemoryManager()
        
        # Create mock component
        component = Mock(spec=MemoryManagedComponent)
        component.get_memory_usage.return_value = {
            'item_count': 200,  # Over default max of 100
            'memory_mb': 1.0,
            'last_cleanup_time': 0  # Very old
        }
        component.cleanup_old_data.return_value = 50
        
        # Register with aggressive policy
        policy = MemoryPolicy(max_items=100, cleanup_interval=0.1)
        manager.register_component('test', component, policy)
        
        # Perform cleanup
        manager._regular_cleanup()
        
        # Verify cleanup was called
        component.cleanup_old_data.assert_called_once()
        assert manager._stats['items_cleaned'] == 50
        
    def test_emergency_cleanup(self):
        """Test emergency cleanup under high memory pressure."""
        manager = GlobalMemoryManager()
        
        # Create mock components
        component1 = Mock(spec=MemoryManagedComponent)
        component1.force_cleanup.return_value = 100
        
        component2 = Mock(spec=MemoryManagedComponent)
        component2.force_cleanup.return_value = 200
        
        manager.register_component('test1', component1)
        manager.register_component('test2', component2)
        
        # Perform emergency cleanup
        manager._emergency_cleanup()
        
        # Verify both components were cleaned
        component1.force_cleanup.assert_called_once_with(0.5)
        component2.force_cleanup.assert_called_once_with(0.5)
        assert manager._stats['items_cleaned'] == 300
        
    def test_get_statistics(self):
        """Test getting memory statistics."""
        manager = GlobalMemoryManager()
        
        # Create mock component
        component = Mock(spec=MemoryManagedComponent)
        component.get_memory_usage.return_value = {
            'item_count': 10,
            'memory_mb': 1.0
        }
        
        manager.register_component('test', component)
        
        stats = manager.get_statistics()
        
        assert 'total_cleanups' in stats
        assert 'emergency_cleanups' in stats
        assert 'items_cleaned' in stats
        assert 'components' in stats
        assert 'test' in stats['components']
        assert stats['components']['test']['item_count'] == 10


class TestQuantumStateMemoryManager:
    """Test quantum state memory management."""
    
    def test_initialization(self):
        """Test initialization with quantum backend."""
        backend = Mock()
        backend.states = {}
        
        manager = QuantumStateMemoryManager(backend)
        assert manager.backend == backend
        
    def test_get_memory_usage(self):
        """Test getting memory usage for quantum states."""
        backend = Mock()
        backend.states = {
            'state1': Mock(),
            'state2': Mock(),
            'state3': Mock()
        }
        
        manager = QuantumStateMemoryManager(backend)
        usage = manager.get_memory_usage()
        
        assert usage['item_count'] == 3
        assert usage['memory_mb'] == pytest.approx(0.024, rel=0.1)  # 3 * 0.008
        
    def test_cleanup_old_data(self):
        """Test cleaning up old quantum states."""
        backend = Mock()
        
        # Create states with different ages
        current_time = time.time()
        state1 = Mock(creation_time=current_time - 1000)  # Very old
        state2 = Mock(creation_time=current_time - 10)    # Recent
        state3 = Mock(creation_time=current_time - 500)   # Old
        
        backend.states = {
            'state1': state1,
            'state2': state2,
            'state3': state3
        }
        
        manager = QuantumStateMemoryManager(backend)
        cleaned = manager.cleanup_old_data(300)  # 5 minutes
        
        assert cleaned == 2  # state1 and state3 removed
        assert 'state2' in backend.states
        assert 'state1' not in backend.states
        assert 'state3' not in backend.states
        
    def test_force_cleanup(self):
        """Test forced cleanup of quantum states."""
        backend = Mock()
        
        # Create states
        current_time = time.time()
        backend.states = {
            f'state{i}': Mock(creation_time=current_time - i*10)
            for i in range(10)
        }
        
        manager = QuantumStateMemoryManager(backend)
        cleaned = manager.force_cleanup(0.3)  # Remove 30%
        
        assert cleaned == 3
        assert len(backend.states) == 7


class TestObserverRegistryMemoryManager:
    """Test observer registry memory management."""
    
    def test_cleanup_old_data(self):
        """Test cleaning up old observers."""
        registry = Mock()
        registry.observers = {
            'obs1': Mock(creation_time=time.time() - 1000),
            'obs2': Mock(creation_time=time.time() - 10)
        }
        registry.remove_observer = Mock()
        
        manager = ObserverRegistryMemoryManager(registry)
        cleaned = manager.cleanup_old_data(300)
        
        assert cleaned == 1
        registry.remove_observer.assert_called_once_with('obs1')
        
    def test_force_cleanup_inactive_first(self):
        """Test that inactive observers are cleaned up first."""
        registry = Mock()
        registry.observers = {
            'active1': Mock(is_active=True),
            'inactive1': Mock(is_active=False),
            'active2': Mock(is_active=True),
            'inactive2': Mock(is_active=False)
        }
        registry.remove_observer = Mock()
        
        manager = ObserverRegistryMemoryManager(registry)
        cleaned = manager.force_cleanup(0.5)  # Remove 50%
        
        assert cleaned == 2
        # Should remove inactive observers
        assert registry.remove_observer.call_count == 2


class TestMeasurementHistoryMemoryManager:
    """Test measurement history memory management."""
    
    def test_cleanup_old_data(self):
        """Test cleaning up old measurements."""
        runtime = Mock()
        current_time = time.time()
        
        runtime.measurement_results = [
            {'timestamp': current_time - 1000, 'value': 1},  # Old
            {'timestamp': current_time - 10, 'value': 2},    # Recent
            {'value': 3},  # No timestamp - keep
            {'timestamp': current_time - 500, 'value': 4}    # Old
        ]
        
        manager = MeasurementHistoryMemoryManager(runtime)
        cleaned = manager.cleanup_old_data(300)
        
        assert cleaned == 2
        assert len(runtime.measurement_results) == 2
        assert runtime.measurement_results[0]['value'] == 2
        assert runtime.measurement_results[1]['value'] == 3
        
    def test_force_cleanup(self):
        """Test forced cleanup of measurements."""
        runtime = Mock()
        current_time = time.time()
        
        runtime.measurement_results = [
            {'timestamp': current_time - i, 'value': i}
            for i in range(100)
        ]
        
        manager = MeasurementHistoryMemoryManager(runtime)
        cleaned = manager.force_cleanup(0.6)  # Remove 60%
        
        assert cleaned == 60
        assert len(runtime.measurement_results) == 40
        # Should keep newest measurements
        assert runtime.measurement_results[0]['value'] == 0


class TestMemoryIntegration:
    """Test memory management integration with runtime."""
    
    @patch('src.core.memory_integration.setup_memory_management')
    @patch('src.core.memory_integration.get_global_memory_manager')
    def test_setup_complete_memory_management(self, mock_get_manager, mock_setup):
        """Test complete memory management setup."""
        runtime = Mock()
        runtime.config = {'production_mode': True}
        
        manager = Mock()
        manager._policies = {}
        mock_get_manager.return_value = manager
        
        setup_complete_memory_management(runtime)
        
        mock_setup.assert_called_once_with(runtime)
        
        # Check aggressive policies were set
        assert 'quantum_states' in manager._policies
        assert manager._policies['quantum_states'].max_items == 500
        
    def test_get_memory_statistics(self):
        """Test getting comprehensive memory statistics."""
        runtime = Mock()
        runtime.memory_manager = Mock()
        runtime.memory_manager.memory_pools = {
            'standard': Mock(get_stats=lambda: {'size': 1024})
        }
        runtime.memory_manager.total_allocated_memory = 512
        runtime.memory_manager.gc_runs = 5
        
        with patch('src.core.memory_integration.get_global_memory_manager') as mock_get:
            manager = Mock()
            manager.get_statistics.return_value = {
                'total_cleanups': 10,
                'emergency_cleanups': 1,
                'items_cleaned': 100,
                'components': {},
                'system_memory': {}
            }
            mock_get.return_value = manager
            
            stats = get_memory_statistics(runtime)
            
            assert stats['memory_pools']['standard']['size'] == 1024
            assert stats['total_allocated'] == 512
            assert stats['gc_runs'] == 5
            assert stats['cleanup_stats']['total_cleanups'] == 10
            
    def test_optimize_memory_for_long_simulations(self):
        """Test memory optimization for long runs."""
        runtime = Mock()
        runtime.memory_manager = Mock()
        
        with patch('src.core.memory_integration.get_global_memory_manager') as mock_get:
            manager = Mock()
            manager._components = {'comp1': Mock(), 'comp2': Mock()}
            manager._policies = {}
            mock_get.return_value = manager
            
            optimize_memory_for_long_simulations(runtime)
            
            # Check aggressive settings
            assert runtime.memory_manager.gc_interval == 5.0
            assert runtime.memory_manager.garbage_collection_threshold == 100
            
            # Check all components have aggressive policies
            for comp_name in manager._components:
                policy = manager._policies[comp_name]
                assert policy.max_items == 100
                assert policy.cleanup_interval == 10.0


class TestMemoryManagementStressTest:
    """Stress tests for memory management system."""
    
    def test_concurrent_cleanup(self):
        """Test concurrent cleanup operations."""
        manager = GlobalMemoryManager()
        
        # Create multiple components
        components = []
        for i in range(10):
            comp = Mock(spec=MemoryManagedComponent)
            comp.get_memory_usage.return_value = {
                'item_count': 100 + i,
                'memory_mb': 10.0 + i,
                'last_cleanup_time': 0
            }
            comp.cleanup_old_data.return_value = 10
            comp.force_cleanup.return_value = 20
            
            components.append(comp)
            manager.register_component(f'comp{i}', comp)
        
        # Start multiple cleanup threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=manager._regular_cleanup)
            threads.append(t)
            t.start()
        
        # Wait for threads
        for t in threads:
            t.join(timeout=1.0)
        
        # Verify no errors occurred
        assert manager._stats['items_cleaned'] >= 100
        
    def test_memory_pressure_simulation(self):
        """Test behavior under simulated memory pressure."""
        manager = GlobalMemoryManager()
        
        # Mock high memory usage
        with patch.object(manager, '_get_system_memory_usage') as mock_mem:
            mock_mem.return_value = {
                'percent': 95,  # High memory usage
                'total_mb': 8192,
                'available_mb': 400,
                'process_mb': 7000
            }
            
            # Create component
            comp = Mock(spec=MemoryManagedComponent)
            comp.force_cleanup.return_value = 1000
            manager.register_component('test', comp)
            
            # Run one cleanup cycle
            manager._cleanup_loop = Mock(side_effect=manager._emergency_cleanup)
            manager._running = True
            manager._cleanup_loop()
            
            # Verify emergency cleanup was triggered
            comp.force_cleanup.assert_called_once()
            assert manager._stats['emergency_cleanups'] == 1