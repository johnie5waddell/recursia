"""
field_dynamics.py - Recursia Field Dynamics Engine

The central coordinator for all runtime quantum field dynamics in the Recursia system.
Manages field lifecycle, evolution, coupling, and OSH metric computation with enterprise-grade
performance, thread safety, and comprehensive diagnostics for proving the Organic Simulation Hypothesis.

This module integrates quantum field theory, biologically-inspired coupling mechanisms,
recursive strain dynamics, and advanced OSH metrics tracking to provide a complete
field simulation framework aligned with the Organic Simulation Hypothesis.
"""

import numpy as np
import logging
import threading
import time
import uuid
import weakref
from typing import Dict, List, Optional, Any, Tuple, Union, Callable, Set
from dataclasses import dataclass, field as dataclass_field
from collections import defaultdict, deque
from enum import Enum
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from contextlib import contextmanager

# Recursia Core Imports
from src.core.data_classes import CouplingConfiguration, CouplingType, FieldMetadata, FieldState, FieldStatistics, OSHMetrics
from src.physics.field.field_compute import FieldComputeEngine, get_compute_engine
from src.physics.field.field_types import FieldTypeDefinition, get_field_type
from src.physics.field.field_evolution_tracker import FieldEvolutionTracker, get_field_evolution_tracker
from src.physics.coherence import CoherenceManager
from src.physics.memory_field import MemoryFieldPhysics
from src.physics.recursive import RecursiveMechanics
from src.physics.physics_event_system import PhysicsEventSystem
from src.physics.physics_profiler import PhysicsProfiler
from src.core.utils import global_error_manager, performance_profiler

# Configure logging
logger = logging.getLogger(__name__)


class FieldRegistry:
    """Thread-safe registry for all fields and their metadata."""
    
    def __init__(self):
        self._fields: Dict[str, np.ndarray] = {}
        self._metadata: Dict[str, FieldMetadata] = {}
        self._field_types: Dict[str, FieldTypeDefinition] = {}
        self._lock = threading.RLock()
        self._field_name_to_id: Dict[str, str] = {}
        self._active_fields: Set[str] = set()
        logger.info("FieldRegistry initialized")
    
    def register_field(self, field_id: str, name: str, field_values: np.ndarray,
                      field_type: FieldTypeDefinition, metadata: Optional[Dict] = None) -> bool:
        """Register a new field with comprehensive validation."""
        try:
            with self._lock:
                if field_id in self._fields:
                    logger.warning(f"Field {field_id} already exists, updating")
                
                # Validate field values
                if not field_type.validate_values(field_values):
                    raise ValueError(f"Invalid field values for type {field_type.__class__.__name__}")
                
                # Create metadata
                field_metadata = FieldMetadata(
                    field_id=field_id,
                    name=name,
                    field_type=field_type.__class__.__name__,
                    shape=field_values.shape,
                    creation_time=time.time(),
                    last_update_time=time.time(),
                    state=FieldState.CREATED,
                    memory_usage=field_values.nbytes
                )
                
                if metadata:
                    field_metadata.custom_metadata.update(metadata)
                
                # Store field data
                self._fields[field_id] = field_values.copy()
                self._metadata[field_id] = field_metadata
                self._field_types[field_id] = field_type
                self._field_name_to_id[name] = field_id
                self._active_fields.add(field_id)
                
                logger.info(f"Registered field {name} ({field_id}) of type {field_type.__class__.__name__}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to register field {name}: {str(e)}")
            global_error_manager.error("field_dynamics", 0, 0, f"Field registration failed: {str(e)}")
            return False
    
    def get_field(self, identifier: Union[str, int]) -> Optional[np.ndarray]:
        """Get field values by ID or name."""
        try:
            with self._lock:
                field_id = self._resolve_field_id(identifier)
                if field_id in self._fields:
                    return self._fields[field_id].copy()
                return None
        except Exception as e:
            logger.error(f"Failed to get field {identifier}: {str(e)}")
            return None
    
    def update_field(self, identifier: Union[str, int], field_values: np.ndarray) -> bool:
        """Update field values with validation."""
        try:
            with self._lock:
                field_id = self._resolve_field_id(identifier)
                if field_id not in self._fields:
                    logger.error(f"Field {identifier} not found for update")
                    return False
                
                # Validate new values
                field_type = self._field_types[field_id]
                if not field_type.validate_values(field_values):
                    logger.error(f"Invalid field values for update of {identifier}")
                    return False
                
                # Update field and metadata
                self._fields[field_id] = field_values.copy()
                metadata = self._metadata[field_id]
                metadata.last_update_time = time.time()
                metadata.memory_usage = field_values.nbytes
                metadata.shape = field_values.shape
                
                logger.debug(f"Updated field {identifier}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update field {identifier}: {str(e)}")
            return False
    
    def delete_field(self, identifier: Union[str, int]) -> bool:
        """Delete field and cleanup resources."""
        try:
            with self._lock:
                field_id = self._resolve_field_id(identifier)
                if field_id not in self._fields:
                    logger.warning(f"Field {identifier} not found for deletion")
                    return False
                
                # Get metadata for cleanup
                metadata = self._metadata[field_id]
                name = metadata.name
                
                # Remove from all registries
                del self._fields[field_id]
                del self._metadata[field_id]
                del self._field_types[field_id]
                if name in self._field_name_to_id:
                    del self._field_name_to_id[name]
                self._active_fields.discard(field_id)
                
                # Update metadata state
                metadata.update_state(FieldState.DELETED)
                
                logger.info(f"Deleted field {name} ({field_id})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete field {identifier}: {str(e)}")
            return False
    
    def get_metadata(self, identifier: Union[str, int]) -> Optional[FieldMetadata]:
        """Get field metadata."""
        try:
            with self._lock:
                field_id = self._resolve_field_id(identifier)
                return self._metadata.get(field_id)
        except Exception as e:
            logger.error(f"Failed to get metadata for {identifier}: {str(e)}")
            return None
    
    def get_field_type(self, identifier: Union[str, int]) -> Optional[FieldTypeDefinition]:
        """Get field type definition."""
        try:
            with self._lock:
                field_id = self._resolve_field_id(identifier)
                return self._field_types.get(field_id)
        except Exception as e:
            logger.error(f"Failed to get field type for {identifier}: {str(e)}")
            return None
    
    def list_fields(self) -> List[str]:
        """List all active field IDs."""
        with self._lock:
            return list(self._active_fields)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive registry statistics."""
        with self._lock:
            total_memory = sum(metadata.memory_usage for metadata in self._metadata.values())
            total_evolution_steps = sum(metadata.evolution_steps for metadata in self._metadata.values())
            total_evolution_time = sum(metadata.total_evolution_time for metadata in self._metadata.values())
            
            state_counts = defaultdict(int)
            for metadata in self._metadata.values():
                state_counts[metadata.state.value] += 1
            
            return {
                'total_fields': len(self._fields),
                'active_fields': len(self._active_fields),
                'total_memory_usage': total_memory,
                'total_evolution_steps': total_evolution_steps,
                'total_evolution_time': total_evolution_time,
                'state_distribution': dict(state_counts),
                'field_types': list(set(metadata.field_type for metadata in self._metadata.values()))
            }
    
    def _resolve_field_id(self, identifier: Union[str, int]) -> str:
        """Resolve field identifier to field ID."""
        if isinstance(identifier, str):
            if identifier in self._fields:
                return identifier
            elif identifier in self._field_name_to_id:
                return self._field_name_to_id[identifier]
            else:
                raise ValueError(f"Field not found: {identifier}")
        else:
            raise ValueError(f"Invalid field identifier type: {type(identifier)}")


class CouplingManager:
    """Manages all field-to-field and field-to-state coupling interactions."""
    
    def __init__(self, field_registry: FieldRegistry):
        self.field_registry = field_registry
        self._couplings: Dict[str, CouplingConfiguration] = {}
        self._coupling_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.RLock()
        self._active_couplings: Set[str] = set()
        self._coupling_performance: Dict[str, Dict[str, float]] = {}
        logger.info("CouplingManager initialized")
    
    def add_coupling(self, coupling_config: CouplingConfiguration) -> str:
        """Add a new field coupling configuration."""
        try:
            with self._lock:
                coupling_id = f"{coupling_config.source_field_id}_{coupling_config.target_field_id}_{uuid.uuid4().hex[:8]}"
                
                # Validate fields exist
                source_metadata = self.field_registry.get_metadata(coupling_config.source_field_id)
                target_metadata = self.field_registry.get_metadata(coupling_config.target_field_id)
                
                if not source_metadata or not target_metadata:
                    raise ValueError("Source or target field not found")
                
                # Store coupling configuration
                self._couplings[coupling_id] = coupling_config
                self._active_couplings.add(coupling_id)
                self._coupling_performance[coupling_id] = {
                    'total_applications': 0,
                    'total_time': 0.0,
                    'last_application_time': 0.0,
                    'success_rate': 1.0
                }
                
                logger.info(f"Added coupling {coupling_id}: {coupling_config.source_field_id} -> {coupling_config.target_field_id}")
                return coupling_id
                
        except Exception as e:
            logger.error(f"Failed to add coupling: {str(e)}")
            raise
    
    def remove_coupling(self, coupling_id: str) -> bool:
        """Remove a field coupling."""
        try:
            with self._lock:
                if coupling_id in self._couplings:
                    del self._couplings[coupling_id]
                    self._active_couplings.discard(coupling_id)
                    logger.info(f"Removed coupling {coupling_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to remove coupling {coupling_id}: {str(e)}")
            return False
    
    def apply_couplings(self, time_step: float) -> Dict[str, Any]:
        """Apply all active couplings between fields."""
        results = {
            'applied_couplings': 0,
            'failed_couplings': 0,
            'total_time': 0.0,
            'coupling_effects': {}
        }
        
        start_time = time.time()
        
        try:
            with self._lock:
                active_couplings = list(self._active_couplings)
            
            for coupling_id in active_couplings:
                try:
                    coupling_result = self._apply_single_coupling(coupling_id, time_step)
                    if coupling_result['success']:
                        results['applied_couplings'] += 1
                        results['coupling_effects'][coupling_id] = coupling_result
                    else:
                        results['failed_couplings'] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to apply coupling {coupling_id}: {str(e)}")
                    results['failed_couplings'] += 1
            
            results['total_time'] = time.time() - start_time
            logger.debug(f"Applied {results['applied_couplings']} couplings in {results['total_time']:.4f}s")
            
        except Exception as e:
            logger.error(f"Coupling application failed: {str(e)}")
            results['failed_couplings'] = len(self._active_couplings)
        
        return results
    
    def _apply_single_coupling(self, coupling_id: str, time_step: float) -> Dict[str, Any]:
        """Apply a single coupling between two fields."""
        start_time = time.time()
        result = {'success': False, 'effect_magnitude': 0.0, 'error': None}
        
        try:
            coupling_config = self._couplings[coupling_id]
            
            # Get field data and metadata
            source_field = self.field_registry.get_field(coupling_config.source_field_id)
            target_field = self.field_registry.get_field(coupling_config.target_field_id)
            source_metadata = self.field_registry.get_metadata(coupling_config.source_field_id)
            target_metadata = self.field_registry.get_metadata(coupling_config.target_field_id)
            
            if source_field is None or target_field is None:
                result['error'] = "Field data not available"
                return result
            
            # Check if coupling should be active
            if not coupling_config.is_active(source_metadata.osh_metrics, target_metadata.osh_metrics):
                result['success'] = True
                result['effect_magnitude'] = 0.0
                return result
            
            # Apply coupling based on type
            coupling_effect = self._calculate_coupling_effect(
                coupling_config, source_field, target_field, 
                source_metadata.osh_metrics, target_metadata.osh_metrics, time_step
            )
            
            # Update target field
            if coupling_config.symmetry in ['bidirectional', 'source_to_target']:
                updated_target = target_field + coupling_effect['target_effect']
                self.field_registry.update_field(coupling_config.target_field_id, updated_target)
            
            if coupling_config.symmetry in ['bidirectional', 'target_to_source']:
                updated_source = source_field + coupling_effect['source_effect']
                self.field_registry.update_field(coupling_config.source_field_id, updated_source)
            
            # Record coupling history
            self._coupling_history[coupling_id].append({
                'timestamp': time.time(),
                'effect_magnitude': coupling_effect['magnitude'],
                'source_coherence': source_metadata.osh_metrics.coherence,
                'target_coherence': target_metadata.osh_metrics.coherence
            })
            
            result['success'] = True
            result['effect_magnitude'] = coupling_effect['magnitude']
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Coupling {coupling_id} application failed: {str(e)}")
        
        # Update performance metrics
        application_time = time.time() - start_time
        perf_metrics = self._coupling_performance[coupling_id]
        perf_metrics['total_applications'] += 1
        perf_metrics['total_time'] += application_time
        perf_metrics['last_application_time'] = application_time
        perf_metrics['success_rate'] = (perf_metrics['success_rate'] * (perf_metrics['total_applications'] - 1) + 
                                       (1.0 if result['success'] else 0.0)) / perf_metrics['total_applications']
        
        return result
    
    def _calculate_coupling_effect(self, coupling_config: CouplingConfiguration,
                                 source_field: np.ndarray, target_field: np.ndarray,
                                 source_metrics: OSHMetrics, target_metrics: OSHMetrics,
                                 time_step: float) -> Dict[str, Any]:
        """Calculate the coupling effect between two fields."""
        
        # Adaptive strength based on coherence and entropy
        adaptive_strength = coupling_config.strength
        if coupling_config.adaptive:
            coherence_factor = (source_metrics.coherence + target_metrics.coherence) / 2.0
            entropy_factor = 1.0 - (source_metrics.entropy + target_metrics.entropy) / 2.0
            adaptive_strength *= coherence_factor * entropy_factor
        
        # Calculate base coupling effects
        if coupling_config.coupling_type == CouplingType.LINEAR:
            target_effect = adaptive_strength * time_step * source_field
            source_effect = adaptive_strength * time_step * target_field
            
        elif coupling_config.coupling_type == CouplingType.NONLINEAR:
            target_effect = adaptive_strength * time_step * source_field * np.abs(source_field)
            source_effect = adaptive_strength * time_step * target_field * np.abs(target_field)
            
        elif coupling_config.coupling_type == CouplingType.GRADIENT:
            target_effect = adaptive_strength * time_step * np.gradient(source_field, axis=0)
            source_effect = adaptive_strength * time_step * np.gradient(target_field, axis=0)
            
        elif coupling_config.coupling_type == CouplingType.COHERENCE_RESONANCE:
            resonance_factor = np.exp(-np.abs(source_metrics.coherence - target_metrics.coherence))
            target_effect = adaptive_strength * time_step * resonance_factor * source_field
            source_effect = adaptive_strength * time_step * resonance_factor * target_field
            
        elif coupling_config.coupling_type == CouplingType.MEMORY_MEDIATED:
            memory_factor = 1.0 - (source_metrics.strain + target_metrics.strain) / 2.0
            target_effect = adaptive_strength * time_step * memory_factor * source_field
            source_effect = adaptive_strength * time_step * memory_factor * target_field
            
        else:
            # Default to linear coupling
            target_effect = adaptive_strength * time_step * source_field
            source_effect = adaptive_strength * time_step * target_field
        
        # Apply frequency and phase modulation
        if coupling_config.frequency != 1.0:
            phase = 2 * np.pi * coupling_config.frequency * time.time() + coupling_config.phase_offset
            modulation = np.sin(phase)
            target_effect *= modulation
            source_effect *= modulation
        
        # Ensure field shapes match
        if target_effect.shape != target_field.shape:
            target_effect = np.resize(target_effect, target_field.shape)
        if source_effect.shape != source_field.shape:
            source_effect = np.resize(source_effect, source_field.shape)
        
        magnitude = np.sqrt(np.mean(target_effect**2) + np.mean(source_effect**2))
        
        return {
            'target_effect': target_effect,
            'source_effect': source_effect,
            'magnitude': magnitude,
            'adaptive_strength': adaptive_strength
        }
    
    def get_coupling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive coupling statistics."""
        with self._lock:
            total_couplings = len(self._couplings)
            active_couplings = len(self._active_couplings)
            
            coupling_types = defaultdict(int)
            avg_strength = 0.0
            
            for coupling in self._couplings.values():
                coupling_types[coupling.coupling_type.value] += 1
                avg_strength += coupling.strength
            
            if total_couplings > 0:
                avg_strength /= total_couplings
            
            return {
                'total_couplings': total_couplings,
                'active_couplings': active_couplings,
                'coupling_type_distribution': dict(coupling_types),
                'average_coupling_strength': avg_strength,
                'performance_metrics': dict(self._coupling_performance)
            }


class FieldDynamics:
    """
    Central coordinator for all runtime quantum field dynamics in the Recursia system.
    
    This class manages the complete lifecycle of quantum fields including creation, evolution,
    coupling, and analysis. It integrates with all major Recursia subsystems to provide
    comprehensive OSH-aligned field simulation capabilities.
    """
    
    def __init__(self, coherence_manager: Optional[CoherenceManager] = None,
                 memory_field_physics: Optional[MemoryFieldPhysics] = None,
                 recursive_mechanics: Optional[RecursiveMechanics] = None,
                 compute_engine: Optional[FieldComputeEngine] = None,
                 event_system: Optional[PhysicsEventSystem] = None):
        
        # Core subsystem integration
        self.coherence_manager = coherence_manager
        self.memory_field_physics = memory_field_physics
        self.recursive_mechanics = recursive_mechanics
        self.compute_engine = compute_engine or get_compute_engine()
        self.event_system = event_system
        
        # Core components
        self.field_registry = FieldRegistry()
        self.coupling_manager = CouplingManager(self.field_registry)
        self.evolution_tracker = get_field_evolution_tracker()
        
        # Performance and monitoring
        self.profiler = PhysicsProfiler(logger=logger)
        self._performance_metrics = {
            'evolution_times': deque(maxlen=1000),
            'coupling_times': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'error_counts': defaultdict(int)
        }
        
        # Evolution history for OSH analysis
        self._evolution_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Thread safety
        self._lock = threading.RLock()
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="FieldDynamics")
        
        # Configuration
        self._config = {
            'enable_osh_effects': True,
            'enable_coupling': True,
            'enable_evolution_tracking': True,
            'max_evolution_history': 10000,
            'performance_monitoring': True,
            'automatic_cleanup': True
        }
        
        # Start memory manager for automatic cleanup
        from src.physics.field.field_memory_manager import get_field_memory_manager
        self.memory_manager = get_field_memory_manager()
        
        logger.info("FieldDynamics initialized with comprehensive OSH integration")
    
    def create_field(self, name: str, field_type_name: str, grid_shape: Tuple[int, ...],
                    initial_values: Optional[np.ndarray] = None,
                    initialization_params: Optional[Dict[str, Any]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new quantum field with comprehensive validation and OSH integration.
        
        Args:
            name: Human-readable field name
            field_type_name: Type of field (scalar, vector, spinor, etc.)
            grid_shape: Spatial grid dimensions
            initial_values: Optional pre-initialized field values
            initialization_params: Parameters for field initialization
            metadata: Additional metadata for the field
            
        Returns:
            Field ID for subsequent operations
        """
        field_id = f"field_{uuid.uuid4().hex}"
        
        try:
            with self.profiler.timed_step("field_creation"):
                # Get field type definition
                field_type = get_field_type(field_type_name)
                if not field_type:
                    raise ValueError(f"Unknown field type: {field_type_name}")
                
                # Initialize field values
                if initial_values is not None:
                    field_values = initial_values.copy()
                else:
                    init_params = initialization_params or {}
                    field_values = field_type.initialize_values(grid_shape, **init_params)
                
                # Register field
                success = self.field_registry.register_field(
                    field_id, name, field_values, field_type, metadata
                )
                
                if not success:
                    raise RuntimeError("Field registration failed")
                
                # Calculate initial OSH metrics
                initial_metrics = self._calculate_osh_metrics(field_values, field_type, field_id)
                
                # Update field metadata with OSH metrics
                field_metadata = self.field_registry.get_metadata(field_id)
                field_metadata.osh_metrics = initial_metrics
                field_metadata.update_state(FieldState.ACTIVE)
                
                # Record initial state in evolution tracker
                if self._config['enable_evolution_tracking']:
                    self.evolution_tracker.record_field_state(
                        field_id, field_values, 0.0, initial_metrics.to_dict()
                    )
                
                # Emit creation event
                if self.event_system:
                    self.event_system.emit('field_creation_event', {
                        'field_id': field_id,
                        'name': name,
                        'type': field_type_name,
                        'shape': grid_shape,
                        'initial_metrics': initial_metrics.to_dict()
                    })
                
                self.profiler.log_step("field_creation_success", 
                                     field_id=field_id, name=name, type=field_type_name)
                
                return field_id
                
        except Exception as e:
            error_msg = f"Failed to create field {name}: {str(e)}"
            logger.error(error_msg)
            global_error_manager.error("field_dynamics", 0, 0, error_msg)
            
            if self.event_system:
                self.event_system.emit('field_creation_error', {
                    'name': name,
                    'type': field_type_name,
                    'error': str(e)
                })
            
            raise
    
    def evolve_field(self, identifier: Union[str, int], time_step: float,
                    evolution_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Evolve a field forward in time with comprehensive OSH effects.
        
        Args:
            identifier: Field ID or name
            time_step: Time step for evolution
            evolution_params: Additional evolution parameters
            
        Returns:
            Success status
        """
        try:
            with self.profiler.timed_step("field_evolution"):
                # Resolve field ID
                field_metadata = self.field_registry.get_metadata(identifier)
                if not field_metadata:
                    logger.error(f"Field {identifier} not found for evolution")
                    return False
                
                field_id = field_metadata.field_id
                field_metadata.update_state(FieldState.EVOLVING)
                
                # Get field data and type
                field_values = self.field_registry.get_field(field_id)
                field_type = self.field_registry.get_field_type(field_id)
                
                if field_values is None or field_type is None:
                    logger.error(f"Field data not available for {identifier}")
                    return False
                
                # Store pre-evolution state for comparison
                pre_evolution_metrics = field_metadata.osh_metrics
                
                # Apply evolution using compute engine
                evolution_start = time.time()
                evolved_field = self.compute_engine.evolve_field(
                    field_values, 
                    field_type.properties.evolution_type,
                    dt=time_step,
                    **(evolution_params or {})
                )
                evolution_time = time.time() - evolution_start
                
                # Apply OSH effects
                if self._config['enable_osh_effects']:
                    evolved_field = self._apply_osh_effects(
                        evolved_field, field_metadata, field_type, time_step
                    )
                
                # Update field values
                self.field_registry.update_field(field_id, evolved_field)
                
                # Calculate new OSH metrics
                new_metrics = self._calculate_osh_metrics(evolved_field, field_type, field_id)
                
                # Calculate information curvature if enabled
                if self._config.get('enable_information_curvature', True):
                    from src.physics.information_curvature import InformationCurvatureCalculator
                    
                    # Create information density field from integrated information
                    info_density = new_metrics.phi * np.ones_like(evolved_field)
                    
                    # Calculate curvature
                    curvature_calc = InformationCurvatureCalculator()
                    curvature_tensor = curvature_calc.calculate_curvature_from_information(
                        info_density,
                        evolved_field.shape,
                        time_index=field_metadata.evolution_steps
                    )
                    
                    # Store curvature information in metrics
                    new_metrics.information_curvature = curvature_tensor.scalar_curvature
                    
                    # Log significant curvature
                    if abs(curvature_tensor.scalar_curvature) > 0.1:
                        logger.info(f"Significant information curvature detected: {curvature_tensor.scalar_curvature:.4f}")
                        
                        if self.event_system:
                            self.event_system.emit('information_curvature_event', {
                                'field_id': field_id,
                                'scalar_curvature': curvature_tensor.scalar_curvature,
                                'coupling_strength': curvature_tensor.coupling_strength,
                                'information_density': float(np.mean(info_density))
                            })
                
                # Update metadata
                field_metadata.osh_metrics = new_metrics
                field_metadata.evolution_steps += 1
                field_metadata.total_evolution_time += evolution_time
                field_metadata.update_performance('last_evolution_time', evolution_time)
                field_metadata.update_state(FieldState.ACTIVE)
                
                # Record evolution history
                self._evolution_history[field_id].append({
                    'timestamp': time.time(),
                    'step': field_metadata.evolution_steps,
                    'metrics': new_metrics.to_dict(),
                    'evolution_time': evolution_time,
                    'time_step': time_step
                })
                
                # Track evolution in evolution tracker
                if self._config['enable_evolution_tracking']:
                    self.evolution_tracker.record_field_state(
                        field_id, evolved_field, time.time(), new_metrics.to_dict()
                    )
                
                # Update performance metrics
                self._performance_metrics['evolution_times'].append(evolution_time)
                
                # Emit evolution event
                if self.event_system:
                    self.event_system.emit('field_evolution_event', {
                        'field_id': field_id,
                        'evolution_step': field_metadata.evolution_steps,
                        'time_step': time_step,
                        'metrics': new_metrics.to_dict(),
                        'metric_changes': {
                            'coherence_delta': new_metrics.coherence - pre_evolution_metrics.coherence,
                            'entropy_delta': new_metrics.entropy - pre_evolution_metrics.entropy,
                            'strain_delta': new_metrics.strain - pre_evolution_metrics.strain
                        }
                    })
                
                self.profiler.log_step("field_evolution_success", 
                                     field_id=field_id, 
                                     evolution_time=evolution_time,
                                     coherence=new_metrics.coherence)
                
                return True
                
        except Exception as e:
            error_msg = f"Field evolution failed for {identifier}: {str(e)}"
            logger.error(error_msg)
            
            # Update field state to error
            field_metadata = self.field_registry.get_metadata(identifier)
            if field_metadata:
                field_metadata.update_state(FieldState.ERROR)
                field_metadata.add_error(error_msg)
            
            self._performance_metrics['error_counts']['evolution_errors'] += 1
            
            if self.event_system:
                self.event_system.emit('field_evolution_error', {
                    'field_id': identifier,
                    'error': str(e)
                })
            
            return False
    
    def apply_coupling(self, time_step: float) -> bool:
        """Apply all active field couplings."""
        if not self._config['enable_coupling']:
            return True
        
        try:
            with self.profiler.timed_step("coupling_application"):
                coupling_start = time.time()
                results = self.coupling_manager.apply_couplings(time_step)
                coupling_time = time.time() - coupling_start
                
                self._performance_metrics['coupling_times'].append(coupling_time)
                
                if self.event_system:
                    self.event_system.emit('coupling_application_event', {
                        'applied_couplings': results['applied_couplings'],
                        'failed_couplings': results['failed_couplings'],
                        'total_time': coupling_time
                    })
                
                self.profiler.log_step("coupling_application_success",
                                     applied=results['applied_couplings'],
                                     failed=results['failed_couplings'])
                
                return results['failed_couplings'] == 0
                
        except Exception as e:
            logger.error(f"Coupling application failed: {str(e)}")
            self._performance_metrics['error_counts']['coupling_errors'] += 1
            return False
    
    def _apply_osh_effects(self, field_values: np.ndarray, field_metadata: FieldMetadata,
                          field_type: FieldTypeDefinition, time_step: float) -> np.ndarray:
        """Apply OSH-specific effects to field evolution."""
        modified_field = field_values.copy()
        
        try:
            # Apply coherence manager effects
            if self.coherence_manager:
                # Calculate decoherence based on current coherence
                current_coherence = field_metadata.osh_metrics.coherence
                if current_coherence > 0.1:  # Only apply if coherence is significant
                    # Apply decoherence noise
                    noise_strength = (1.0 - current_coherence) * 0.01 * time_step
                    noise = np.random.normal(0, noise_strength, field_values.shape)
                    modified_field += noise
            
            # Apply memory field effects
            if self.memory_field_physics:
                # Get memory strain for this field's region
                field_region = f"field_{field_metadata.field_id}"
                try:
                    strain = self.memory_field_physics.get_region_properties(field_region).get('memory_strain', 0.0)
                    
                    # Apply strain-induced distortion
                    if strain > 0.1:
                        distortion_factor = strain * 0.05 * time_step
                        # Apply random distortion proportional to strain
                        distortion = np.random.laplace(0, distortion_factor, field_values.shape)
                        modified_field += distortion
                        
                        # Update strain in field metadata
                        field_metadata.osh_metrics.strain = strain
                        
                except Exception as e:
                    logger.debug(f"Memory field effect application failed: {str(e)}")
            
            # Apply recursive mechanics effects
            if self.recursive_mechanics:
                try:
                    # Get recursive depth impact
                    system_name = f"field_system_{field_metadata.field_id}"
                    recursive_depth = self.recursive_mechanics.get_recursive_depth(system_name) or 0
                    
                    if recursive_depth > 0:
                        # Apply recursive strain effects
                        recursive_strain = min(recursive_depth * 0.1, 1.0)
                        field_metadata.osh_metrics.recursive_depth_impact = recursive_strain
                        
                        # Apply compression effects from recursive depth
                        compression_factor = 1.0 - recursive_strain * 0.1 * time_step
                        modified_field *= compression_factor
                        
                except Exception as e:
                    logger.debug(f"Recursive mechanics effect application failed: {str(e)}")
            
            # Apply biologically-inspired decoherence
            current_entropy = field_metadata.osh_metrics.entropy
            if current_entropy < 0.9:  # Apply entropy increase
                entropy_increase_rate = 0.01 * time_step
                entropy_noise = np.random.exponential(entropy_increase_rate, field_values.shape)
                modified_field += entropy_noise * (1.0 - current_entropy)
            
        except Exception as e:
            logger.warning(f"OSH effects application partially failed: {str(e)}")
        
        return modified_field
    
    def _calculate_osh_metrics(self, field_values: np.ndarray, field_type: FieldTypeDefinition,
                              field_id: str) -> OSHMetrics:
        """Calculate comprehensive OSH metrics for a field."""
        try:
            # Basic coherence calculation
            if np.iscomplexobj(field_values):
                # For complex fields, use phase coherence
                phases = np.angle(field_values)
                phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
                coherence = phase_coherence
            else:
                # For real fields, use spatial correlation
                mean_val = np.mean(field_values)
                variance = np.var(field_values)
                if variance > 0:
                    coherence = 1.0 - (variance / (np.abs(mean_val) + variance + 1e-10))
                else:
                    coherence = 1.0
            
            coherence = np.clip(coherence, 0.0, 1.0)
            
            # Entropy calculation
            field_probs = np.abs(field_values.flatten())**2
            field_probs = field_probs / (np.sum(field_probs) + 1e-10)
            field_probs = field_probs[field_probs > 1e-10]  # Remove zeros
            if len(field_probs) > 0:
                shannon_entropy = -np.sum(field_probs * np.log(field_probs + 1e-10))
                entropy = shannon_entropy / np.log(len(field_probs))  # Normalize
            else:
                entropy = 0.0
            
            entropy = np.clip(entropy, 0.0, 1.0)
            
            # Energy density calculation
            if hasattr(field_type, 'calculate_energy_density'):
                try:
                    derivatives = self.compute_engine.calculate_derivatives(field_values)
                    energy_density = np.mean(field_type.calculate_energy_density(field_values, derivatives))
                except:
                    energy_density = np.mean(np.abs(field_values)**2)
            else:
                energy_density = np.mean(np.abs(field_values)**2)
            
            # Strain calculation (default to low strain)
            strain = min(entropy * 0.5, 1.0)  # Entropy contributes to strain
            
            # RSP calculation (Recursive Simulation Potential)
            if coherence > 0 and entropy < 1.0:
                information_content = coherence * (1.0 - entropy)
                rsp = information_content * np.log(1.0 + energy_density)
            else:
                rsp = 0.0
            
            # Phi (Integrated Information) approximation
            phi = coherence * (1.0 - entropy) * np.log(1.0 + field_values.size)
            
            # Emergence index
            field_complexity = np.std(field_values) / (np.mean(np.abs(field_values)) + 1e-10)
            emergence_index = coherence * field_complexity * (1.0 - entropy)
            emergence_index = np.clip(emergence_index, 0.0, 1.0)
            
            # Consciousness quotient (advanced OSH metric)
            consciousness_quotient = (phi * rsp) / (entropy + 0.1)
            consciousness_quotient = np.clip(consciousness_quotient / 10.0, 0.0, 1.0)  # Normalize
            
            # Information curvature (second derivative of information density)
            try:
                info_density = -field_probs * np.log(field_probs + 1e-10)
                if len(info_density) > 2:
                    info_density_2d = info_density.reshape(int(np.sqrt(len(info_density))), -1)
                    curvature = np.mean(np.abs(np.gradient(np.gradient(info_density_2d, axis=0), axis=0)))
                else:
                    curvature = 0.0
            except:
                curvature = 0.0
            
            # Temporal stability (will be updated as field evolves)
            temporal_stability = 1.0 - strain
            
            # Memory coherence index (related to field organization)
            memory_coherence_index = coherence * (1.0 - entropy / 2.0)
            
            return OSHMetrics(
                coherence=coherence,
                entropy=entropy,
                strain=strain,
                field_energy=energy_density,  # Using field_energy instead of energy_density
                rsp=rsp,
                phi=phi,
                emergence_index=emergence_index,
                consciousness_quotient=consciousness_quotient,
                information_curvature=curvature,
                temporal_stability=temporal_stability,
                memory_field_coupling=memory_coherence_index,  # Using memory_field_coupling instead
                recursive_depth=0,  # Using recursive_depth (int) instead of recursive_depth_impact
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.warning(f"OSH metrics calculation failed for field {field_id}: {str(e)}")
            return OSHMetrics()  # Return default metrics
    
    def get_field_values(self, identifier: Union[str, int]) -> Optional[np.ndarray]:
        """Get current field values."""
        return self.field_registry.get_field(identifier)
    
    def set_field_values(self, identifier: Union[str, int], field_values: np.ndarray) -> bool:
        """Set field values with validation and metric update."""
        try:
            success = self.field_registry.update_field(identifier, field_values)
            if success:
                # Update OSH metrics
                field_metadata = self.field_registry.get_metadata(identifier)
                field_type = self.field_registry.get_field_type(identifier)
                if field_metadata and field_type:
                    new_metrics = self._calculate_osh_metrics(field_values, field_type, field_metadata.field_id)
                    field_metadata.osh_metrics = new_metrics
            return success
        except Exception as e:
            logger.error(f"Failed to set field values for {identifier}: {str(e)}")
            return False
    
    def delete_field(self, identifier: Union[str, int]) -> bool:
        """Delete a field and cleanup all associated resources."""
        try:
            field_metadata = self.field_registry.get_metadata(identifier)
            if field_metadata:
                field_id = field_metadata.field_id
                
                # Remove from evolution history
                if field_id in self._evolution_history:
                    del self._evolution_history[field_id]
                
                # Remove associated couplings
                couplings_to_remove = []
                for coupling_id, coupling_config in self.coupling_manager._couplings.items():
                    if (coupling_config.source_field_id == field_id or 
                        coupling_config.target_field_id == field_id):
                        couplings_to_remove.append(coupling_id)
                
                for coupling_id in couplings_to_remove:
                    self.coupling_manager.remove_coupling(coupling_id)
                
                # Emit deletion event
                if self.event_system:
                    self.event_system.emit('field_deletion_event', {
                        'field_id': field_id,
                        'name': field_metadata.name
                    })
            
            return self.field_registry.delete_field(identifier)
            
        except Exception as e:
            logger.error(f"Field deletion failed for {identifier}: {str(e)}")
            return False
    
    def add_field_coupling(self, source_field: Union[str, int], target_field: Union[str, int],
                          coupling_type: CouplingType, strength: float = 0.1,
                          **coupling_params) -> Optional[str]:
        """Add a coupling between two fields."""
        try:
            # Resolve field IDs
            source_metadata = self.field_registry.get_metadata(source_field)
            target_metadata = self.field_registry.get_metadata(target_field)
            
            if not source_metadata or not target_metadata:
                logger.error("Source or target field not found for coupling")
                return None
            
            coupling_config = CouplingConfiguration(
                source_field_id=source_metadata.field_id,
                target_field_id=target_metadata.field_id,
                coupling_type=coupling_type,
                strength=strength,
                **coupling_params
            )
            
            coupling_id = self.coupling_manager.add_coupling(coupling_config)
            
            if self.event_system:
                self.event_system.emit('field_coupling_added', {
                    'coupling_id': coupling_id,
                    'source_field': source_metadata.field_id,
                    'target_field': target_metadata.field_id,
                    'coupling_type': coupling_type.value,
                    'strength': strength
                })
            
            return coupling_id
            
        except Exception as e:
            logger.error(f"Failed to add field coupling: {str(e)}")
            return None
    
    def remove_field_coupling(self, coupling_id: str) -> bool:
        """Remove a field coupling."""
        try:
            success = self.coupling_manager.remove_coupling(coupling_id)
            
            if success and self.event_system:
                self.event_system.emit('field_coupling_removed', {
                    'coupling_id': coupling_id
                })
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to remove field coupling {coupling_id}: {str(e)}")
            return False
    
    def get_field_statistics(self) -> FieldStatistics:
        """Get comprehensive field system statistics."""
        try:
            registry_stats = self.field_registry.get_statistics()
            coupling_stats = self.coupling_manager.get_coupling_statistics()
            
            # Calculate average OSH metrics
            all_metrics = []
            for field_id in self.field_registry.list_fields():
                metadata = self.field_registry.get_metadata(field_id)
                if metadata:
                    all_metrics.append(metadata.osh_metrics)
            
            if all_metrics:
                avg_coherence = np.mean([m.coherence for m in all_metrics])
                avg_entropy = np.mean([m.entropy for m in all_metrics])
                avg_strain = np.mean([m.strain for m in all_metrics])
                avg_rsp = np.mean([m.rsp for m in all_metrics])
                avg_phi = np.mean([m.phi for m in all_metrics])
                avg_emergence = np.mean([m.emergence_index for m in all_metrics])
                avg_consciousness = np.mean([m.consciousness_quotient for m in all_metrics])
                
                average_osh_metrics = OSHMetrics(
                    coherence=avg_coherence,
                    entropy=avg_entropy,
                    strain=avg_strain,
                    rsp=avg_rsp,
                    phi=avg_phi,
                    emergence_index=avg_emergence,
                    consciousness_quotient=avg_consciousness
                )
            else:
                average_osh_metrics = OSHMetrics()
            
            # Performance metrics
            perf_metrics = {}
            if self._performance_metrics['evolution_times']:
                perf_metrics['avg_evolution_time'] = np.mean(self._performance_metrics['evolution_times'])
                perf_metrics['max_evolution_time'] = np.max(self._performance_metrics['evolution_times'])
            
            if self._performance_metrics['coupling_times']:
                perf_metrics['avg_coupling_time'] = np.mean(self._performance_metrics['coupling_times'])
                perf_metrics['max_coupling_time'] = np.max(self._performance_metrics['coupling_times'])
            
            return FieldStatistics(
                total_fields=registry_stats['total_fields'],
                active_fields=registry_stats['active_fields'],
                evolving_fields=registry_stats['state_distribution'].get('evolving', 0),
                coupled_fields=coupling_stats['active_couplings'],
                total_memory_usage=registry_stats['total_memory_usage'],
                total_evolution_steps=registry_stats['total_evolution_steps'],
                total_evolution_time=registry_stats['total_evolution_time'],
                average_osh_metrics=average_osh_metrics,
                coupling_statistics=coupling_stats['coupling_type_distribution'],
                performance_metrics=perf_metrics,
                error_counts=dict(self._performance_metrics['error_counts']),
                last_update=time.time()
            )
            
        except Exception as e:
            logger.error(f"Failed to get field statistics: {str(e)}")
            return FieldStatistics()
    
    def get_field_metadata(self, identifier: Union[str, int]) -> Optional[FieldMetadata]:
        """Get field metadata."""
        return self.field_registry.get_metadata(identifier)
    
    def get_field_evolution_history(self, identifier: Union[str, int]) -> List[Dict[str, Any]]:
        """Get evolution history for a field."""
        try:
            field_metadata = self.field_registry.get_metadata(identifier)
            if field_metadata:
                field_id = field_metadata.field_id
                return list(self._evolution_history[field_id])
            return []
        except Exception as e:
            logger.error(f"Failed to get evolution history for {identifier}: {str(e)}")
            return []
    
    def list_fields(self) -> List[Dict[str, Any]]:
        """List all fields with basic information."""
        field_list = []
        try:
            for field_id in self.field_registry.list_fields():
                metadata = self.field_registry.get_metadata(field_id)
                if metadata:
                    field_list.append({
                        'field_id': field_id,
                        'name': metadata.name,
                        'type': metadata.field_type,
                        'shape': metadata.shape,
                        'state': metadata.state.value,
                        'osh_metrics': metadata.osh_metrics.to_dict()
                    })
        except Exception as e:
            logger.error(f"Failed to list fields: {str(e)}")
        
        return field_list
    
    def reset(self):
        """Reset the field dynamics system to initial state."""
        try:
            logger.info("Resetting FieldDynamics system")
            
            with self._lock:
                # Clear all fields
                field_ids = self.field_registry.list_fields().copy()
                for field_id in field_ids:
                    self.delete_field(field_id)
                
                # Clear evolution history
                self._evolution_history.clear()
                
                # Clear performance metrics
                for metric_list in self._performance_metrics.values():
                    if hasattr(metric_list, 'clear'):
                        metric_list.clear()
                
                # Reset error counts
                self._performance_metrics['error_counts'].clear()
                
                # Reset compute engine cache
                self.compute_engine.clear_cache()
                
                logger.info("FieldDynamics system reset completed")
                
        except Exception as e:
            logger.error(f"Failed to reset FieldDynamics: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources and finalize system."""
        try:
            logger.info("Cleaning up FieldDynamics system")
            
            # Get final statistics
            final_stats = self.get_field_statistics()
            logger.info(f"Final statistics: {final_stats.total_fields} fields processed, "
                       f"{final_stats.total_evolution_steps} evolution steps, "
                       f"{final_stats.total_evolution_time:.2f}s total evolution time")
            
            # Shutdown thread pool
            self._thread_pool.shutdown(wait=True)
            
            # Reset system
            self.reset()
            
            logger.info("FieldDynamics cleanup completed")
            
        except Exception as e:
            logger.error(f"FieldDynamics cleanup failed: {str(e)}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        try:
            self.cleanup()
        except:
            pass


# Global field dynamics instance
_global_field_dynamics: Optional[FieldDynamics] = None
_field_dynamics_lock = threading.Lock()


def get_field_dynamics() -> FieldDynamics:
    """Get the global field dynamics instance."""
    global _global_field_dynamics
    
    if _global_field_dynamics is None:
        with _field_dynamics_lock:
            if _global_field_dynamics is None:
                _global_field_dynamics = FieldDynamics()
    
    return _global_field_dynamics


def set_field_dynamics(field_dynamics: FieldDynamics):
    """Set the global field dynamics instance."""
    global _global_field_dynamics
    
    with _field_dynamics_lock:
        if _global_field_dynamics is not None:
            _global_field_dynamics.cleanup()
        _global_field_dynamics = field_dynamics


# Convenience functions for global field dynamics operations
def create_field(name: str, field_type: str, grid_shape: Tuple[int, ...], **kwargs) -> str:
    """Create a field using the global field dynamics instance."""
    return get_field_dynamics().create_field(name, field_type, grid_shape, **kwargs)


def evolve_field(identifier: Union[str, int], time_step: float, **kwargs) -> bool:
    """Evolve a field using the global field dynamics instance."""
    return get_field_dynamics().evolve_field(identifier, time_step, **kwargs)


def get_field_values(identifier: Union[str, int]) -> Optional[np.ndarray]:
    """Get field values using the global field dynamics instance."""
    return get_field_dynamics().get_field_values(identifier)


def apply_field_coupling(time_step: float) -> bool:
    """Apply field couplings using the global field dynamics instance."""
    return get_field_dynamics().apply_coupling(time_step)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.DEBUG)
    
    # Create field dynamics system
    field_dynamics = FieldDynamics()
    
    try:
        # Create a test field
        field_id = field_dynamics.create_field(
            name="test_scalar_field",
            field_type_name="scalar",
            grid_shape=(64, 64),
            initialization_params={'init_type': 'gaussian'}
        )
        
        print(f"Created field: {field_id}")
        
        # Evolve the field
        for step in range(10):
            success = field_dynamics.evolve_field(field_id, 0.01)
            if success:
                metadata = field_dynamics.get_field_metadata(field_id)
                if metadata:
                    metrics = metadata.osh_metrics
                    print(f"Step {step}: Coherence={metrics.coherence:.3f}, "
                          f"Entropy={metrics.entropy:.3f}, RSP={metrics.rsp:.3f}")
        
        # Get final statistics
        stats = field_dynamics.get_field_statistics()
        print(f"\nFinal Statistics:")
        print(f"Total fields: {stats.total_fields}")
        print(f"Evolution steps: {stats.total_evolution_steps}")
        print(f"Average coherence: {stats.average_osh_metrics.coherence:.3f}")
        print(f"Average RSP: {stats.average_osh_metrics.rsp:.3f}")
        
    finally:
        field_dynamics.cleanup()