import datetime
import logging
import math
import time
import traceback
import os
# psutil imported at function level for performance
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable

logger = logging.getLogger(__name__)

class ExecutionContext:
    """
    Execution context for Recursia programs
    
    The execution context maintains the state of a running Recursia program,
    including variables, results, statistics, and execution state. It provides
    a controlled environment for quantum and classical operations.
    
    Core to the Organic Simulation Hypothesis (OSH) implementation, the execution
    context manages recursive scopes, observer effects, memory strain, coherence
    tracking, and the boundary conditions between simulation layers.
    """
    
    # Execution state constants
    STATE_INITIALIZED = 'initialized'
    STATE_RUNNING = 'running'
    STATE_PAUSED = 'paused'
    STATE_COMPLETED = 'completed'
    STATE_ERROR = 'error'
    
    def __init__(self, args: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution context
        
        Args:
            args: Execution arguments including configuration for OSH model parameters
        """
        self.args = args or {}
        self.variables: Dict[str, Any] = {}
        self.variable_scopes: Dict[str, Dict[str, Any]] = {'global': {}}
        self.current_scope: str = 'global'
        self.scope_stack: List[str] = ['global']
        self.scope_metadata: Dict[str, Dict[str, Any]] = {
            'global': {
                'type': 'global',
                'created_at': time.time(),
                'accessed_at': time.time(),
                'access_count': 1,
                'parent': None,
                'children': [],
                'depth': 0,
                'has_return': False,
                'loop_depth': 0,
                'vars_count': 0,
                'memory_usage': 0,
                'coherence': 1.0,
                'strain': 0.0
            }
        }
        self.results: Dict[str, Any] = {}
        
        # Track execution state
        self.state: str = self.STATE_INITIALIZED
        self.error: Optional[Exception] = None
        self.error_info: Optional[Dict[str, Any]] = None
        
        # Statistics tracking
        self.statistics: Dict[str, Any] = {
            'start_time': time.time(),
            'end_time': None,
            'execution_time': 0,
            'gate_count': 0,
            'measurement_count': 0,
            'entanglement_count': 0,
            'instruction_count': 0,
            'memory_usage': 0,
            'memory_peak': 0,
            'max_recursion_depth': 0,
            'current_recursion_depth': 0,
            'scope_count': 1,
            'function_calls': 0,
            'observer_interactions': 0,
            'coherence_operations': 0,
            'quantum_operations': 0,
            'teleportations': 0,
            'recursive_boundary_crossings': 0,
            'critical_strain_events': 0,
            'convergence_events': 0,
            'accumulated_strain': 0.0
        }
        
        # Unified metrics - single source of truth
        from src.core.data_classes import OSHMetrics
        self.current_metrics: OSHMetrics = OSHMetrics(
            information_density=0.0,
            kolmogorov_complexity=1.0,
            entropy=0.05,
            entanglement_entropy=0.1,
            rsp=0.0,
            consciousness_field=0.0,
            phi=0.0,
            coherence=0.95,
            strain=0.0,
            memory_strain=0.0,
            recursive_depth=0,
            observer_influence=0.0,
            timestamp=time.time()
        )
        
        # OSH measurement results storage
        self.osh_measurements: Dict[str, Any] = {
            'integrated_information': 0.0,
            'kolmogorov_complexity': 1.0,
            'entropy_production': 0.0,
            'variational_free_energy': 0.0,
            'collapse_probability': 0.0,
            'field_strain': 0.0,
            'information_curvature': 0.0,
            'recursion_depth': 0,
            'memory_strain': 0.0
        }
        
        # Simulation parameters
        self.simulation_time: float = 0.0  # internal simulation time units
        self.simulation_step: int = 0  # current step/tick
        self.simulation_config: Dict[str, Any] = {
            'time_step': 0.01,
            'max_steps': 10000,
            'auto_save_interval': 100,
            'coherence_threshold': 0.3,
            'strain_threshold': 0.8,
            'enable_observer_effects': True,
            'enable_memory_strain': True,
            'enable_recursive_mechanics': True,
            'recursive_boundary_permeability': 0.5,
            'runtime_optimization_level': 1
        }
        self.simulation_config.update(self.args.get('simulation_config', {}))
        
        # Memory management
        self.memory_allocations: Dict[str, Dict[str, Any]] = {}
        self.memory_pools: Dict[str, Dict[str, Any]] = {
            'standard': {'size': 1024 * 1024, 'used': 0, 'peak': 0},
            'quantum': {'size': 2 * 1024 * 1024, 'used': 0, 'peak': 0},
            'observer': {'size': 256 * 1024, 'used': 0, 'peak': 0},
            'temporary': {'size': 512 * 1024, 'used': 0, 'peak': 0}
        }
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        
        # Logging
        self.log_entries: List[Dict[str, Any]] = []
        self.log_level = self.args.get('log_level', 'info')
        self.log_to_console = self.args.get('log_to_console', True)
        self.log_to_file = self.args.get('log_to_file', False)
        self.log_file_path = self.args.get('log_file_path', 'recursia_execution.log')
        
        # Function call tracking
        self.call_stack: List[Dict[str, Any]] = []
        self.return_values: List[Any] = []
        self.function_stats: Dict[str, Dict[str, Any]] = {}
        
        # Breakpoint management
        self.breakpoints: Set[Tuple[str, int]] = set()  # (filename, line)
        self.breakpoint_hit: bool = False
        self.step_mode: bool = self.args.get('step_mode', False)
        self.trace_mode: bool = self.args.get('trace_mode', False)
        self.break_condition: Optional[Callable[[Dict[str, Any]], bool]] = None
        
        # Observer tracking
        self.current_observer: Optional[str] = None
        self.observed_states: Dict[str, List[str]] = {}  # observer -> [states]
        self.observer_effects: Dict[str, Dict[str, float]] = {}  # observer -> {state: strength}
        self.observer_phases: Dict[str, str] = {}  # observer -> phase
        self.observation_history: List[Dict[str, Any]] = []
        
        # Event listeners
        self.event_listeners: List[int] = []
        
        # Recursive mechanics
        self.recursion_levels: Dict[str, int] = {'global': 0}  # system -> level
        self.recursion_parents: Dict[str, Optional[str]] = {'global': None}  # system -> parent
        self.recursion_boundaries: Dict[Tuple[str, str], Dict[str, float]] = {}  # (lower, upper) -> params
        self.recursion_strain: Dict[str, float] = {'global': 0.0}  # system -> strain
        self.recursive_depth_counters: Dict[str, int] = {}  # operation types -> current depth
        
        # Coherence tracking
        self.state_coherence: Dict[str, float] = {}  # state -> coherence
        self.state_entropy: Dict[str, float] = {}  # state -> entropy
        self.coherence_changes: List[Dict[str, Any]] = []
        
        # Memory field
        self.field_regions: Dict[str, Dict[str, Any]] = {}
        self.field_connections: Dict[Tuple[str, str], float] = {}  # (region1, region2) -> strength
        self.field_strain: Dict[str, float] = {}  # region -> strain
        
        # Execution history
        self.execution_history: List[Dict[str, Any]] = []
        self.execution_snapshots: Dict[int, Dict[str, Any]] = {}  # step -> snapshot
        self.snapshot_interval = self.args.get('snapshot_interval', 100)
        
        # Visualization
        self.visualization_data: Dict[str, Any] = {}
        self.animation_frames: List[Dict[str, Any]] = []
    
    def start_execution(self) -> None:
        """
        Start execution and record the start time
        """
        self.state = self.STATE_RUNNING
        self.statistics['start_time'] = time.time()
        self._update_memory_usage()
        
        # Record initial snapshot
        self._record_execution_snapshot(0)
        
        logger.info("Execution started")
        self.log('info', "Execution started")
    
    def pause_execution(self) -> None:
        """
        Pause execution
        """
        if self.state == self.STATE_RUNNING:
            self.state = self.STATE_PAUSED
            logger.info("Execution paused")
            self.log('info', "Execution paused")
            
            # Record snapshot at pause point
            self._record_execution_snapshot(self.simulation_step, paused=True)
    
    def resume_execution(self) -> None:
        """
        Resume execution
        """
        if self.state == self.STATE_PAUSED:
            self.state = self.STATE_RUNNING
            logger.info("Execution resumed")
            self.log('info', "Execution resumed")
    
    def complete_execution(self) -> None:
        """
        Mark execution as completed and record end time
        """
        self.state = self.STATE_COMPLETED
        self.statistics['end_time'] = time.time()
        self.statistics['execution_time'] = self.statistics['end_time'] - self.statistics['start_time']
        
        # Final memory usage update
        self._update_memory_usage()
        
        # Record final snapshot
        self._record_execution_snapshot(self.simulation_step, final=True)
        
        logger.info(f"Execution completed in {self.statistics['execution_time']:.6f} seconds")
        self.log('info', f"Execution completed in {self.statistics['execution_time']:.6f} seconds")
    
    def set_error(self, error: Union[Exception, str], info: Optional[Dict[str, Any]] = None) -> None:
        """
        Set execution error
        
        Args:
            error: Exception that caused the error or error message string
            info: Additional error information
        """
        self.state = self.STATE_ERROR
        
        # Convert string error to Exception if needed
        if isinstance(error, str):
            self.error = Exception(error)
        else:
            self.error = error
        
        # Initialize error_info properly
        self.error_info = {} if info is None else info.copy()
        
        # Add traceback information 
        self.error_info['traceback'] = traceback.format_exc()
        self.error_info['location'] = self._get_error_location()
        
        # Update timestamps
        self.statistics['end_time'] = time.time()
        self.statistics['execution_time'] = self.statistics['end_time'] - self.statistics['start_time']
        
        # Record error snapshot
        self._record_execution_snapshot(self.simulation_step, error=True)
        
        # Format error message
        error_msg = f"Execution error: {error}"
        if 'location' in self.error_info:
            error_msg += f" at {self.error_info['location']}"
            
        logger.error(error_msg)
        self.log('error', error_msg)
        
    def _get_error_location(self) -> str:
        """
        Get the current code location for error reporting
        
        Returns:
            str: Location string (e.g., "function:line")
        """
        if self.call_stack:
            function = self.call_stack[-1]['name']
            return f"{function}"
        return "global"
    
    # Improved memory_pools integration
    def _update_memory_usage(self) -> int:
        """
        Update memory usage statistics with pool-specific tracking
        
        Returns:
            int: Current memory usage in bytes
        """
        try:
            # Get process memory info
            if self.process:
                memory_info = self.process.memory_info()
                current_usage = memory_info.rss  # Resident Set Size
            else:
                # Fallback if psutil not available
                current_usage = 0
            
            # Update statistics
            self.statistics['memory_usage'] = current_usage
            if current_usage > self.statistics.get('memory_peak', 0):
                self.statistics['memory_peak'] = current_usage
            
            # Update memory pools based on usage distribution
            total_alloc = sum(len(scope) for scope in self.variable_scopes.values())
            if total_alloc > 0:
                # Distribute memory usage across pools proportionally
                standard_pct = 0.60  # Default distribution percentages
                quantum_pct = 0.30
                observer_pct = 0.05
                temp_pct = 0.05
                
                self.memory_pools['standard']['used'] = int(current_usage * standard_pct)
                self.memory_pools['quantum']['used'] = int(current_usage * quantum_pct)
                self.memory_pools['observer']['used'] = int(current_usage * observer_pct)
                self.memory_pools['temporary']['used'] = int(current_usage * temp_pct)
                
                # Update peak values
                for pool_name, pool in self.memory_pools.items():
                    if pool['used'] > pool['peak']:
                        pool['peak'] = pool['used']
                    
            return current_usage
        except Exception as e:
            logger.warning(f"Failed to update memory usage: {e}")
            return 0
    
    # Optimized observer effect decay
    def _decay_observer_effects(self, time_step: float) -> None:
        """
        Apply optimized decay to observer effects using timestamped decay rate
        
        Args:
            time_step: Simulation time step
        """
        if not self.simulation_config.get('enable_observer_effects', True):
            return
            
        decay_rate = 0.05 * time_step  # Base decay rate per time unit
        current_time = self.simulation_time
        
        # Process each observer's effects in bulk
        for observer_name, effects in self.observer_effects.items():
            # Track which state effects to remove (avoid modifying during iteration)
            states_to_remove = []
            
            for state_name, effect_data in effects.items():
                # Convert simple float values to structured data if needed
                if isinstance(effect_data, float):
                    effect_data = {
                        'strength': effect_data,
                        'last_update': current_time - time_step,  # Assume it was just updated
                        'decay_rate': decay_rate
                    }
                    effects[state_name] = effect_data
                
                # Calculate time since last update
                time_elapsed = current_time - effect_data.get('last_update', current_time - time_step)
                
                # Apply decay based on elapsed time (exponential decay)
                strength = effect_data['strength']
                decay = effect_data.get('decay_rate', decay_rate)
                
                # Apply decay using exponential formula rather than linear steps
                new_strength = strength * math.exp(-decay * time_elapsed)
                
                # Update the strength
                effect_data['strength'] = new_strength
                effect_data['last_update'] = current_time
                
                # Schedule for removal if negligible
                if new_strength < 0.01:
                    states_to_remove.append(state_name)
            
            # Remove negligible effects
            for state_name in states_to_remove:
                del effects[state_name]
                
    def _record_execution_snapshot(self, step: int, 
                                  paused: bool = False, 
                                  final: bool = False,
                                  error: bool = False) -> None:
        """
        Record a snapshot of the execution state
        
        Args:
            step: Current simulation step
            paused: Whether execution is paused
            final: Whether this is the final snapshot
            error: Whether this snapshot is due to an error
        """
        # Only record if step is multiple of snapshot interval or special condition
        if not (step % self.snapshot_interval == 0 or paused or final or error):
            return
            
        snapshot = {
            'step': step,
            'simulation_time': self.simulation_time,
            'wall_time': time.time(),
            'state': self.state,
            'current_scope': self.current_scope,
            'scope_stack': self.scope_stack.copy(),
            'call_stack_depth': len(self.call_stack),
            'memory_usage': self.statistics['memory_usage'],
            'instruction_count': self.statistics['instruction_count'],
            'variables_count': sum(len(scope) for scope in self.variable_scopes.values()),
            'current_observer': self.current_observer,
            'observer_count': len(self.observed_states),
            'error': str(self.error) if self.error else None,
            'is_paused': paused,
            'is_final': final,
            'is_error': error
        }
        
        # Add to snapshots
        self.execution_snapshots[step] = snapshot
        
        # Add a limited number of variables (for debugging)
        if self.current_scope in self.variable_scopes:
            # Include only a few key variables to avoid excessive memory usage
            snapshot['variables'] = {
                k: v for k, v in list(self.variable_scopes[self.current_scope].items())[:10]
            }
        
        # For the final snapshot, add summary statistics
        if final:
            snapshot['statistics'] = self.get_statistics()
            snapshot['results'] = self.results
    
    def create_scope(self, name: str) -> str:
        """
        Create a new variable scope
        
        Args:
            name: Scope name or prefix (will be made unique)
            
        Returns:
            str: Created scope name
        """
        # Generate unique scope name
        count = 0
        scope_name = name
        while scope_name in self.variable_scopes:
            count += 1
            scope_name = f"{name}_{count}"
        
        # Create the scope
        self.variable_scopes[scope_name] = {}
        
        # Create scope metadata
        parent_scope = self.current_scope if self.scope_stack else None
        parent_depth = self.scope_metadata.get(parent_scope, {}).get('depth', 0) if parent_scope else 0
        
        self.scope_metadata[scope_name] = {
            'type': 'function' if name.startswith('function_') else 
                   ('loop' if name.startswith(('for_', 'while_')) else 'block'),
            'created_at': time.time(),
            'accessed_at': time.time(),
            'access_count': 1,
            'parent': parent_scope,
            'children': [],
            'depth': parent_depth + 1,
            'has_return': False,
            'loop_depth': self.scope_metadata.get(parent_scope, {}).get('loop_depth', 0),
            'vars_count': 0,
            'memory_usage': 0,
            'coherence': 1.0,
            'strain': 0.0
        }
        
        # Update parent's children list
        if parent_scope:
            self.scope_metadata[parent_scope]['children'].append(scope_name)
        
        # Update statistics
        self.statistics['scope_count'] += 1
        
        logger.debug(f"Created scope: {scope_name}")
        return scope_name
    
    def enter_scope(self, scope_name: str) -> None:
        """
        Enter a variable scope
        
        Args:
            scope_name: Scope to enter
            
        Raises:
            ValueError: If scope doesn't exist
        """
        if scope_name not in self.variable_scopes:
            raise ValueError(f"Unknown scope: {scope_name}")
        
        self.current_scope = scope_name
        self.scope_stack.append(scope_name)
        
        # Update scope metadata
        self.scope_metadata[scope_name]['accessed_at'] = time.time()
        self.scope_metadata[scope_name]['access_count'] += 1
        
        # Update recursion depth tracking
        scope_depth = self.scope_metadata[scope_name]['depth']
        self.statistics['current_recursion_depth'] = scope_depth
        if scope_depth > self.statistics['max_recursion_depth']:
            self.statistics['max_recursion_depth'] = scope_depth
        
        # Apply memory strain based on scope depth according to OSH model
        if self.simulation_config.get('enable_memory_strain', True):
            depth = self.scope_metadata[scope_name]['depth']
            strain_factor = min(0.05 * depth, 0.5)  # Cap at 0.5
            self.scope_metadata[scope_name]['strain'] += strain_factor
            
            # Propagate some strain to parent scopes
            parent = self.scope_metadata[scope_name]['parent']
            while parent:
                # Diminishing effect up the chain
                strain_factor *= 0.5
                self.scope_metadata[parent]['strain'] += strain_factor
                parent = self.scope_metadata[parent]['parent']
            
            # Check for critical strain
            if self.scope_metadata[scope_name]['strain'] > self.simulation_config.get('strain_threshold', 0.8):
                self.statistics['critical_strain_events'] += 1
                self.log('warning', f"Critical memory strain in scope {scope_name}: " +
                       f"{self.scope_metadata[scope_name]['strain']:.2f}")
        
        logger.debug(f"Entered scope: {scope_name}")
    
    def exit_scope(self) -> str:
        """
        Exit the current scope and return to the previous scope
        
        Returns:
            str: Previous scope name
            
        Raises:
            ValueError: If attempting to exit global scope
        """
        if len(self.scope_stack) <= 1:
            raise ValueError("Cannot exit global scope")
        
        previous_scope = self.scope_stack.pop()
        self.current_scope = self.scope_stack[-1]
        
        # Update scope metadata for the scope we're returning to
        self.scope_metadata[self.current_scope]['accessed_at'] = time.time()
        self.scope_metadata[self.current_scope]['access_count'] += 1
        
        # Slightly decrease strain on exit (natural recovery)
        if self.simulation_config.get('enable_memory_strain', True):
            self.scope_metadata[previous_scope]['strain'] = max(
                0, self.scope_metadata[previous_scope]['strain'] - 0.05
            )
            
            # Check if scope was temporary - if so, clean up
            if self.scope_metadata[previous_scope]['type'] == 'block':
                if self.args.get('aggressive_gc', False):
                    # Optional: Remove the scope data to free memory
                    self.variable_scopes.pop(previous_scope, None)
                    # Don't remove metadata as it might be referenced
        
        # Update recursion depth when exiting scope
        if self.current_scope in self.scope_metadata:
            current_depth = self.scope_metadata[self.current_scope]['depth']
            self.statistics['current_recursion_depth'] = current_depth
        
        logger.debug(f"Exited scope: {previous_scope}, returned to: {self.current_scope}")
        return self.current_scope
    
    def set_variable(self, name: str, value: Any, scope: Optional[str] = None) -> None:
        """
        Set a variable in the specified scope or current scope
        
        Args:
            name: Variable name
            value: Variable value
            scope: Scope name (default: current scope)
        """
        target_scope = scope or self.current_scope
        
        if target_scope not in self.variable_scopes:
            raise ValueError(f"Unknown scope: {target_scope}")
        
        # Track previous value for diffing (used in observation)
        previous_value = self.variable_scopes[target_scope].get(name)
        is_new = name not in self.variable_scopes[target_scope]
        
        # Store variable in the appropriate scope
        self.variable_scopes[target_scope][name] = value
        
        # Update scope metadata
        self.scope_metadata[target_scope]['accessed_at'] = time.time()
        if is_new:
            self.scope_metadata[target_scope]['vars_count'] += 1
        
        # Also update the flat variables dictionary for backward compatibility
        self.variables[name] = value
        
        # Apply observer effects if this is a quantum state variable being changed
        # and we have an active observer
        if (self.simulation_config.get('enable_observer_effects', True) and
            self.current_observer and 
            isinstance(value, dict) and
            value.get('type') in ('quantum_state', 'quantum_register')):
            
            # Record the observation
            self.register_observation(self.current_observer, name)
            
            # Note: Actual quantum effects are handled by the quantum system,
            # we just track the relationship here
        
        # Estimate memory usage change
        try:
            import sys
            value_size = sys.getsizeof(value)
            prev_size = sys.getsizeof(previous_value) if previous_value is not None else 0
            size_delta = value_size - prev_size
            
            if size_delta != 0:
                self.scope_metadata[target_scope]['memory_usage'] += size_delta
                
                # Apply small memory strain for large allocations
                if self.simulation_config.get('enable_memory_strain', True) and size_delta > 1024:
                    strain_factor = min(size_delta / (1024 * 1024), 0.1)  # Cap at 0.1
                    self.scope_metadata[target_scope]['strain'] += strain_factor
        except Exception:
            # Ignore errors in memory size estimation
            pass
    
    def get_variable(self, name: str, default: Any = None, search_parent_scopes: bool = True) -> Any:
        """
        Get a variable from the current scope or parent scopes
        
        Args:
            name: Variable name
            default: Default value if not found
            search_parent_scopes: Whether to search parent scopes
            
        Returns:
            Variable value or default
        """
        # Check current scope
        if name in self.variable_scopes[self.current_scope]:
            # Update access metadata
            self.scope_metadata[self.current_scope]['accessed_at'] = time.time()
            return self.variable_scopes[self.current_scope][name]
        
        # Search parent scopes if requested
        if search_parent_scopes:
            # Start from the end of the stack (current scope) and work backward
            for scope_name in reversed(self.scope_stack[:-1]):
                if name in self.variable_scopes[scope_name]:
                    # Update access metadata for the found scope
                    self.scope_metadata[scope_name]['accessed_at'] = time.time()
                    self.scope_metadata[scope_name]['access_count'] += 1
                    return self.variable_scopes[scope_name][name]
        
        # Return default if not found
        return default
    
    def has_variable(self, name: str, scope: Optional[str] = None, search_parent_scopes: bool = True) -> bool:
        """
        Check if a variable exists
        
        Args:
            name: Variable name
            scope: Scope to check (default: current scope)
            search_parent_scopes: Whether to search parent scopes
            
        Returns:
            bool: True if variable exists
        """
        target_scope = scope or self.current_scope
        
        # Check specified scope
        if name in self.variable_scopes.get(target_scope, {}):
            return True
        
        # Search parent scopes if requested
        if search_parent_scopes:
            scope_index = self.scope_stack.index(target_scope) if target_scope in self.scope_stack else -1
            if scope_index >= 0:
                for scope_name in reversed(self.scope_stack[:scope_index]):
                    if name in self.variable_scopes[scope_name]:
                        return True
        
        return False
    
    def add_result(self, name: str, value: Any) -> None:
        """
        Add a result
        
        Args:
            name: Result name
            value: Result value
        """
        self.results[name] = value
        
        # If this is a quantum measurement result, track it
        if isinstance(value, dict) and 'measurement' in name.lower():
            self.statistics['measurement_count'] += 1
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get all results
        
        Returns:
            Dict[str, Any]: Results
        """
        return self.results
    
    def update_statistics(self, stat_name: str, value: Union[int, float] = 1) -> None:
        """
        Update statistics
        
        Args:
            stat_name: Statistic name
            value: Value to add (default: 1)
        """
        if stat_name in self.statistics:
            if isinstance(self.statistics[stat_name], (int, float)):
                self.statistics[stat_name] += value
            else:
                self.statistics[stat_name] = value
        else:
            self.statistics[stat_name] = value
        
        # Special handling for specific statistics
        if stat_name == 'instruction_count':
            # Every N instructions, update memory usage and check for snapshots
            if self.statistics['instruction_count'] % 1000 == 0:
                self._update_memory_usage()
                
                # Record a snapshot periodically
                if self.statistics['instruction_count'] % (10 * self.snapshot_interval) == 0:
                    self._record_execution_snapshot(self.simulation_step)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get execution statistics
        
        Returns:
            Dict[str, Any]: Statistics
        """
        # Ensure end_time is set
        if self.statistics['end_time'] is None:
            self.statistics['end_time'] = time.time()
            self.statistics['execution_time'] = self.statistics['end_time'] - self.statistics['start_time']
        
        # Update memory usage one last time
        self._update_memory_usage()
        
        # Add additional useful statistics
        current_stats = self.statistics.copy()
        
        # Calculate derived statistics
        if current_stats['execution_time'] > 0:
            current_stats['instructions_per_second'] = (
                current_stats['instruction_count'] / current_stats['execution_time']
            )
            
        if self.simulation_time > 0:
            current_stats['wall_time_ratio'] = current_stats['execution_time'] / self.simulation_time
            
        current_stats['scopes_created'] = self.statistics['scope_count']
        current_stats['active_scopes'] = len(self.variable_scopes)
        
        # Add scope statistics
        current_stats['max_scope_depth'] = max(
            metadata['depth'] for metadata in self.scope_metadata.values()
        )
        current_stats['max_strain'] = max(
            metadata['strain'] for metadata in self.scope_metadata.values()
        )
        
        return current_stats
    
    # New helper methods for simulation time effects
    def _apply_natural_decoherence(self, time_step: float) -> None:
        """
        Apply natural decoherence to quantum states over time
        
        Args:
            time_step: Time step to apply
        """
        # Skip if observer effects disabled
        if not self.simulation_config.get('enable_observer_effects', True):
            return
            
        # Apply small natural decoherence to all tracked states
        for state_name in list(self.state_coherence.keys()):
            current = self.state_coherence[state_name]
            if current > 0:
                # Calculate decoherence based on time step
                decoherence_rate = 0.01 * time_step  # 1% per time unit
                new_coherence = max(0, current - decoherence_rate)
                self.state_coherence[state_name] = new_coherence
                
                # Log significant changes
                if abs(new_coherence - current) > 0.1:
                    self.log('debug', f"Natural decoherence of state {state_name}: {current:.2f} -> {new_coherence:.2f}")

    def _apply_memory_strain_recovery(self, time_step: float) -> None:
        """
        Apply natural recovery to memory strain
        
        Args:
            time_step: Time step to apply
        """
        # Skip if memory strain disabled
        if not self.simulation_config.get('enable_memory_strain', True):
            return
            
        # Process scopes in bulk
        recovery_rate = 0.02 * time_step  # 2% per time unit
        
        # Scope strain recovery
        for scope_name in self.scope_metadata:
            current_strain = self.scope_metadata[scope_name]['strain']
            if current_strain > 0:
                new_strain = max(0, current_strain - recovery_rate)
                self.scope_metadata[scope_name]['strain'] = new_strain
                
        # Field region strain recovery
        for region_name in self.field_strain:
            current_strain = self.field_strain[region_name]
            if current_strain > 0:
                new_strain = max(0, current_strain - recovery_rate)
                self.field_strain[region_name] = new_strain
                
    # Buffered file logging
    def log(self, level: str, message: str, source: Optional[str] = None) -> None:
        """
        Add a log entry with buffered file writing
        
        Args:
            level: Log level
            message: Log message
            source: Source of the log message
        """
        # Create log entry
        timestamp = time.time()
        log_entry = {
            'level': level,
            'message': str(message),
            'timestamp': timestamp,
            'datetime': datetime.datetime.fromtimestamp(timestamp),
            'source': source,
            'scope': self.current_scope,
            'simulation_time': self.simulation_time,
            'simulation_step': self.simulation_step
        }
        
        # Add to log entries
        self.log_entries.append(log_entry)
        
        # Print to console if level is high enough and console logging is enabled
        if self.log_to_console:
            level_priorities = {
                'debug': 0,
                'info': 1,
                'warning': 2,
                'error': 3,
                'critical': 4
            }
            
            if level_priorities.get(level, 0) >= level_priorities.get(self.log_level, 1):
                source_prefix = f"[{source}] " if source else ""
                print(f"[{level.upper()}] {source_prefix}{message}")
        
        # Log to logger too
        log_func = getattr(logger, level, logger.info)
        log_func(message)
        
        # Write to file if enabled - use buffered writing
        if self.log_to_file:
            # Initialize file buffer if not exists
            if not hasattr(self, '_log_buffer'):
                self._log_buffer = []
                self._last_flush_time = time.time()
                
            # Add to buffer
            source_str = f" [{source}]" if source else ""
            log_line = f"[{datetime.datetime.fromtimestamp(timestamp)}] [{level.upper()}]{source_str} {message}\n"
            self._log_buffer.append(log_line)
            
            # Flush if buffer gets large or time threshold reached
            buffer_size_threshold = 100  # lines
            time_threshold = 5.0  # seconds
            
            if (len(self._log_buffer) >= buffer_size_threshold or 
                time.time() - self._last_flush_time > time_threshold):
                self._flush_log_buffer()
                
        # Add special flush on error or critical
        if level in ('error', 'critical') and hasattr(self, '_log_buffer'):
            self._flush_log_buffer()
                
    def _flush_log_buffer(self) -> None:
        """
        Flush the log buffer to file
        """
        if not hasattr(self, '_log_buffer') or not self.log_to_file:
            return
            
        try:
            with open(self.log_file_path, 'a') as f:
                f.writelines(self._log_buffer)
            self._log_buffer = []
            self._last_flush_time = time.time()
        except Exception as e:
            logger.warning(f"Failed to write to log file: {e}")
            # Avoid losing logs - keep them in buffer and try again later
            
    # Split large function entry logic
    def enter_function(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Enter a function and push it onto the call stack
        
        Args:
            func_name: Function name
            params: Function parameters
        """
        # Create function context with core information
        function_context = self._create_function_context(func_name, params)
        
        # Update function stats
        self._update_function_stats(func_name, params)
        
        # Push onto call stack and enter scope
        self.call_stack.append(function_context)
        self.enter_scope(function_context['scope'])
        
        # Set parameters as variables in the function scope
        for name, value in params.items():
            self.set_variable(name, value)
        
        # Update recursion metrics and apply OSH effects
        self._update_recursion_metrics()
        self._apply_recursive_mechanics_effects(func_name)
        
        logger.debug(f"Entered function: {func_name}, recursion depth: {len(self.call_stack)}")

    def _create_function_context(self, func_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new function execution context
        
        Args:
            func_name: Function name
            params: Function parameters
            
        Returns:
            Dict[str, Any]: Function context
        """
        return {
            'name': func_name,
            'params': params.copy(),
            'scope': self.create_scope(f"function_{func_name}"),
            'start_time': time.time(),
            'caller_scope': self.current_scope,
            'simulation_time': self.simulation_time,
            'instruction_count': self.statistics['instruction_count'],
            'has_returned': False
        }

    def _update_function_stats(self, func_name: str, params: Dict[str, Any]) -> None:
        """
        Update function statistics on entry
        
        Args:
            func_name: Function name
            params: Function parameters
        """
        # Initialize function stats if needed
        if func_name not in self.function_stats:
            self.function_stats[func_name] = {
                'call_count': 0,
                'total_time': 0,
                'avg_time': 0,
                'min_time': float('inf'),
                'max_time': 0,
                'last_called': time.time(),
                'recursive_calls': 0,
                'param_types': {}
            }
        
        # Update call count
        self.function_stats[func_name]['call_count'] += 1
        self.function_stats[func_name]['last_called'] = time.time()
        
        # Check for recursion - is this function already on the call stack?
        if any(call['name'] == func_name for call in self.call_stack):
            self.function_stats[func_name]['recursive_calls'] += 1
        
        # Track parameter types for statistics
        for param_name, param_value in params.items():
            param_type = type(param_value).__name__
            if param_name not in self.function_stats[func_name]['param_types']:
                self.function_stats[func_name]['param_types'][param_name] = {}
            if param_type not in self.function_stats[func_name]['param_types'][param_name]:
                self.function_stats[func_name]['param_types'][param_name][param_type] = 0
            self.function_stats[func_name]['param_types'][param_name][param_type] += 1

    def _update_recursion_metrics(self) -> None:
        """
        Update recursion depth and related metrics
        """
        # Update recursion depth statistics
        self.statistics['current_recursion_depth'] = len(self.call_stack)
        self.statistics['max_recursion_depth'] = max(
            self.statistics['max_recursion_depth'], 
            self.statistics['current_recursion_depth']
        )
        
        # Update function call count stat
        self.statistics['function_calls'] += 1

    def _apply_recursive_mechanics_effects(self, func_name: str) -> None:
        """
        Apply OSH recursive mechanics effects based on call stack depth
        
        Args:
            func_name: Function name
        """
        # Apply recursive mechanics effects if enabled
        if self.simulation_config.get('enable_recursive_mechanics', True) and self.statistics['current_recursion_depth'] > 2:
            # Track operation in recursion counters
            if 'function' not in self.recursive_depth_counters:
                self.recursive_depth_counters['function'] = 0
            self.recursive_depth_counters['function'] += 1
            
            # Apply strain based on recursion depth
            depth = self.statistics['current_recursion_depth']
            strain_factor = min(0.05 * depth, 0.3)  # Cap at 0.3
            system = 'global'  # Default system
            
            # Update recursion strain
            if system in self.recursion_strain:
                self.recursion_strain[system] += strain_factor
                
                # Check for boundary crossing
                if (self.recursion_strain[system] > 0.8 and 
                    self.recursion_strain.get(system, 0) < 0.8):
                    # Boundary crossing event
                    self.statistics['recursive_boundary_crossings'] += 1
                    self.log('warning', f"Recursive boundary crossed in system {system}")
            
            # Record deep recursion for posterity
            if depth > 10:
                self.log('info', f"Deep recursion detected: {depth} levels in function {func_name}")
                
    # Consistent snapshot interval checking
    def _should_snapshot(self, step: int, force: bool = False) -> bool:
        """
        Determine if a snapshot should be taken at the current step
        
        Args:
            step: Current simulation step
            force: Force a snapshot regardless of interval
            
        Returns:
            bool: True if snapshot should be taken
        """
        if force:
            return True
            
        # Always snapshot at step 0
        if step == 0:
            return True
            
        # Check interval
        return step % self.snapshot_interval == 0

    # Return values stack with depth limit
    def exit_function(self, return_value: Any = None) -> None:
        """
        Exit the current function and pop it from the call stack
        
        Args:
            return_value: Function return value
        """
        if not self.call_stack:
            raise ValueError("Not in a function")
        
        # Pop function context
        function_context = self.call_stack.pop()
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - function_context['start_time']
        instruction_diff = self.statistics['instruction_count'] - function_context['instruction_count']
        
        # Update function statistics
        func_name = function_context['name']
        self.function_stats[func_name]['total_time'] += execution_time
        self.function_stats[func_name]['avg_time'] = (
            self.function_stats[func_name]['total_time'] / 
            self.function_stats[func_name]['call_count']
        )
        self.function_stats[func_name]['min_time'] = min(
            self.function_stats[func_name]['min_time'],
            execution_time
        )
        self.function_stats[func_name]['max_time'] = max(
            self.function_stats[func_name]['max_time'],
            execution_time
        )
        self.function_stats[func_name]['last_return_value_type'] = type(return_value).__name__
        
        # Flag that function has returned
        function_context['has_returned'] = True
        
        # Store return value with depth limit
        max_return_depth = self.args.get('max_return_depth', 100)
        self.return_values.append(return_value)
        
        # Enforce maximum stack depth for return values
        if len(self.return_values) > max_return_depth:
            # Remove oldest return value
            self.return_values.pop(0)
        
        # Mark current scope as having a return
        scope_name = function_context['scope']
        if scope_name in self.scope_metadata:
            self.scope_metadata[scope_name]['has_return'] = True
        
        # Exit the function's scope
        self.exit_scope()
        
        # Return to caller's scope
        if function_context['caller_scope'] != self.current_scope:
            # This is a safety check - should never happen
            logger.warning(f"Scope mismatch: {function_context['caller_scope']} != {self.current_scope}")
            # Force return to caller's scope
            while self.current_scope != function_context['caller_scope'] and len(self.scope_stack) > 1:
                self.exit_scope()
        
        # Update recursion depth
        self.statistics['current_recursion_depth'] = len(self.call_stack)
        
        # Clear recursive depth counter for functions if needed
        if 'function' in self.recursive_depth_counters:
            self.recursive_depth_counters['function'] -= 1
            if self.recursive_depth_counters['function'] <= 0:
                del self.recursive_depth_counters['function']
        
        # Log significant function calls (longer than 1 second or high instruction count)
        if execution_time > 1.0 or instruction_diff > 10000:
            self.log('debug', f"Function {func_name} completed in {execution_time:.6f} seconds, " +
                f"{instruction_diff} instructions")
        
        logger.debug(f"Exited function: {function_context['name']}, execution time: {execution_time:.6f} seconds")
        
    def get_return_value(self) -> Any:
        """
        Get the most recent return value
        
        Returns:
            Any: Return value or None if no return values
        """
        if not self.return_values:
            return None
        return self.return_values[-1]
    
    def set_breakpoint(self, filename: str, line: int) -> None:
        """
        Set a breakpoint
        
        Args:
            filename: Source filename
            line: Line number
        """
        self.breakpoints.add((filename, line))
        self.log('info', f"Breakpoint set at {filename}:{line}")
    
    def clear_breakpoint(self, filename: str, line: int) -> bool:
        """
        Clear a breakpoint
        
        Args:
            filename: Source filename
            line: Line number
            
        Returns:
            bool: True if breakpoint was cleared
        """
        if (filename, line) in self.breakpoints:
            self.breakpoints.remove((filename, line))
            self.log('info', f"Breakpoint cleared at {filename}:{line}")
            return True
        return False
    
    def clear_all_breakpoints(self) -> int:
        """
        Clear all breakpoints
        
        Returns:
            int: Number of breakpoints cleared
        """
        count = len(self.breakpoints)
        self.breakpoints.clear()
        if count > 0:
            self.log('info', f"Cleared {count} breakpoints")
        return count
    
    def check_breakpoint(self, filename: str, line: int) -> bool:
        """
        Check if a breakpoint is hit
        
        Args:
            filename: Source filename
            line: Line number
            
        Returns:
            bool: True if breakpoint is hit
        """
        # Always break if in step mode
        if self.step_mode:
            self.breakpoint_hit = True
            self.log('info', f"Step mode paused at {filename}:{line}")
            return True
            
        # Check for explicit breakpoint
        if (filename, line) in self.breakpoints:
            self.breakpoint_hit = True
            self.log('info', f"Breakpoint hit at {filename}:{line}")
            return True
            
        # Check conditional breakpoint if defined
        if self.break_condition is not None:
            try:
                context = {
                    'filename': filename,
                    'line': line,
                    'scope': self.current_scope,
                    'variables': self.variable_scopes.get(self.current_scope, {}),
                    'call_stack': self.call_stack,
                    'statistics': self.statistics
                }
                if self.break_condition(context):
                    self.breakpoint_hit = True
                    self.log('info', f"Conditional breakpoint hit at {filename}:{line}")
                    return True
            except Exception as e:
                logger.warning(f"Error in breakpoint condition: {e}")
                
        return False
    
    def set_break_condition(self, condition: Callable[[Dict[str, Any]], bool]) -> None:
        """
        Set a conditional breakpoint function
        
        Args:
            condition: Function that takes context dict and returns True if
                       execution should break
        """
        self.break_condition = condition
        self.log('info', "Conditional breakpoint set")
    
    def set_current_observer(self, observer_name: str) -> None:
        """
        Set the current observer
        
        Args:
            observer_name: Observer name
        """
        previous_observer = self.current_observer
        self.current_observer = observer_name
        
        # Track observer phase if not already tracked
        if observer_name not in self.observer_phases:
            self.observer_phases[observer_name] = 'passive'
            
        # If this is a phase change, log it
        if previous_observer != observer_name:
            self.log('debug', f"Current observer changed: {previous_observer} -> {observer_name}")
            
        # Update observer phase to active when set as current
        if self.observer_phases.get(observer_name) == 'passive':
            self.observer_phases[observer_name] = 'active'
            self.log('debug', f"Observer {observer_name} phase changed: passive -> active")
    
    def get_current_observer(self) -> Optional[str]:
        """
        Get the current observer
        
        Returns:
            Optional[str]: Current observer name or None
        """
        return self.current_observer
    
    def set_observer_phase(self, observer_name: str, phase: str) -> None:
        """
        Set the phase of an observer
        
        Args:
            observer_name: Observer name
            phase: Observer phase (passive, active, measuring, analyzing, entangled,
                  learning, collapsed, reset)
        """
        if observer_name not in self.observer_phases:
            self.observer_phases[observer_name] = 'passive'
            
        previous_phase = self.observer_phases[observer_name]
        self.observer_phases[observer_name] = phase
        
        # Log phase change
        if previous_phase != phase:
            self.log('debug', f"Observer {observer_name} phase changed: {previous_phase} -> {phase}")
    
    def get_observer_phase(self, observer_name: str) -> Optional[str]:
        """
        Get the phase of an observer
        
        Args:
            observer_name: Observer name
            
        Returns:
            Optional[str]: Observer phase or None if observer doesn't exist
        """
        return self.observer_phases.get(observer_name)
    
    def register_observation(self, observer_name: str, state_name: str) -> None:
        """
        Register that an observer is observing a state
        
        Args:
            observer_name: Observer name
            state_name: State name
        """
        if observer_name not in self.observed_states:
            self.observed_states[observer_name] = []
        
        if state_name not in self.observed_states[observer_name]:
            self.observed_states[observer_name].append(state_name)
            
        # Create an observation effect with default strength
        if observer_name not in self.observer_effects:
            self.observer_effects[observer_name] = {}
        
        # Initialize with moderate effect
        if state_name not in self.observer_effects[observer_name]:
            self.observer_effects[observer_name][state_name] = 0.5
            
        # Log the observation
        self.log('debug', f"Observer {observer_name} is observing state {state_name}")
        
        # Record in observation history
        observation_record = {
            'observer': observer_name,
            'state': state_name,
            'timestamp': time.time(),
            'simulation_time': self.simulation_time,
            'simulation_step': self.simulation_step,
            'strength': self.observer_effects[observer_name][state_name],
            'observer_phase': self.observer_phases.get(observer_name, 'passive')
        }
        self.observation_history.append(observation_record)
        
        # Update statistics
        self.statistics['observer_interactions'] += 1
        
        # Apply observer effects
        if self.simulation_config.get('enable_observer_effects', True):
            # If observer is in measuring phase, have stronger effect
            if self.observer_phases.get(observer_name) == 'measuring':
                # Potential decoherence effect - to be handled by quantum system
                self.log('debug', f"Observer {observer_name} in measuring phase - potential quantum effects on {state_name}")
            
            # Update observer phase to measuring if actively registering observations
            if self.observer_phases.get(observer_name) == 'active':
                self.observer_phases[observer_name] = 'measuring'
    
    def get_observer_observations(self, observer_name: str) -> List[str]:
        """
        Get all states observed by an observer
        
        Args:
            observer_name: Observer name
            
        Returns:
            List[str]: State names observed by the observer
        """
        return self.observed_states.get(observer_name, []).copy()
    
    def get_state_observers(self, state_name: str) -> List[str]:
        """
        Get all observers observing a state
        
        Args:
            state_name: State name
            
        Returns:
            List[str]: Observer names observing the state
        """
        return [
            observer for observer, states in self.observed_states.items()
            if state_name in states
        ]
    
    def set_observer_effect(self, observer_name: str, state_name: str, strength: float) -> None:
        """
        Set the strength of an observer's effect on a state
        
        Args:
            observer_name: Observer name
            state_name: State name
            strength: Effect strength (0.0 to 1.0)
        """
        # Ensure observer is registered
        if observer_name not in self.observer_effects:
            self.observer_effects[observer_name] = {}
            
        # Apply bounds to strength
        bounded_strength = max(0.0, min(1.0, strength))
        
        # Set the effect
        self.observer_effects[observer_name][state_name] = bounded_strength
    
    def get_observer_effect(self, observer_name: str, state_name: str) -> float:
        """
        Get the strength of an observer's effect on a state
        
        Args:
            observer_name: Observer name
            state_name: State name
            
        Returns:
            float: Effect strength (0.0 to 1.0) or 0.0 if not set
        """
        return self.observer_effects.get(observer_name, {}).get(state_name, 0.0)
    
    def track_quantum_state_coherence(self, state_name: str, coherence_value: float) -> None:
        """
        Track coherence of a quantum state
        
        Args:
            state_name: State name
            coherence_value: Coherence value (0.0 to 1.0)
        """
        # Get previous value for comparison
        previous_value = self.state_coherence.get(state_name, 1.0)
        
        # Bound the value
        bounded_value = max(0.0, min(1.0, coherence_value))
        
        # Store the value
        self.state_coherence[state_name] = bounded_value
        
        # Record change if significant
        if abs(bounded_value - previous_value) > 0.05:
            change_record = {
                'state': state_name,
                'previous': previous_value,
                'current': bounded_value,
                'delta': bounded_value - previous_value,
                'timestamp': time.time(),
                'simulation_time': self.simulation_time,
                'simulation_step': self.simulation_step
            }
            self.coherence_changes.append(change_record)
            
            # Log significant changes
            if abs(bounded_value - previous_value) > 0.2:
                direction = "increased" if bounded_value > previous_value else "decreased"
                self.log('debug', f"Coherence of {state_name} {direction} significantly: " +
                       f"{previous_value:.2f} -> {bounded_value:.2f}")
    
    def track_quantum_state_entropy(self, state_name: str, entropy_value: float) -> None:
        """
        Track entropy of a quantum state
        
        Args:
            state_name: State name
            entropy_value: Entropy value (0.0+)
        """
        # Get previous value for comparison
        previous_value = self.state_entropy.get(state_name, 0.0)
        
        # Ensure non-negative
        bounded_value = max(0.0, entropy_value)
        
        # Store the value
        self.state_entropy[state_name] = bounded_value
        
        # Log significant changes
        if abs(bounded_value - previous_value) > 0.2:
            direction = "increased" if bounded_value > previous_value else "decreased"
            self.log('debug', f"Entropy of {state_name} {direction} significantly: " +
                   f"{previous_value:.2f} -> {bounded_value:.2f}")
            
            # Check for critical entropy (potential decoherence)
            if bounded_value > 0.8 and previous_value <= 0.8:
                self.log('warning', f"Critical entropy reached for {state_name}: {bounded_value:.2f}")
    
    def track_memory_field_strain(self, region_name: str, strain_value: float) -> None:
        """
        Track strain in a memory field region
        
        Args:
            region_name: Region name
            strain_value: Strain value (0.0 to 1.0)
        """
        # Get previous value for comparison
        previous_value = self.field_strain.get(region_name, 0.0)
        
        # Bound the value
        bounded_value = max(0.0, min(1.0, strain_value))
        
        # Store the value
        self.field_strain[region_name] = bounded_value
        
        # Check for critical strain
        if bounded_value > self.simulation_config.get('strain_threshold', 0.8) and previous_value <= self.simulation_config.get('strain_threshold', 0.8):
            self.statistics['critical_strain_events'] += 1
            self.log('warning', f"Critical memory strain in region {region_name}: {bounded_value:.2f}")
    
    # Modularized simulation time advancement
    def advance_simulation_time(self, time_step: Optional[float] = None) -> None:
        """
        Advance the simulation time
        
        Args:
            time_step: Time step to advance (default: from config)
        """
        step = time_step if time_step is not None else self.simulation_config.get('time_step', 0.01)
        
        self.simulation_time += step
        self.simulation_step += 1
        
        # Check if snapshot should be recorded
        if self._should_snapshot(self.simulation_step):
            self._record_execution_snapshot(self.simulation_step)
                
        # Apply simulation-time effects - each in its own method for clarity
        self._apply_natural_decoherence(step)
        self._apply_memory_strain_recovery(step)
        self._decay_observer_effects(step)
        
    def get_context_snapshot(self) -> Dict[str, Any]:
        """
        Get a snapshot of the current execution context
        
        Returns:
            Dict[str, Any]: Context snapshot
        """
        return {
            'state': self.state,
            'current_scope': self.current_scope,
            'scope_stack': self.scope_stack.copy(),
            'call_stack_depth': len(self.call_stack),
            'call_stack': [(func['name'], func.get('start_time', 0)) for func in self.call_stack],
            'statistics': {k: v for k, v in self.get_statistics().items() if k in [
                'instruction_count', 'memory_usage', 'current_recursion_depth', 
                'function_calls', 'observer_interactions'
            ]},
            'simulation': {
                'time': self.simulation_time,
                'step': self.simulation_step
            },
            'current_observer': self.current_observer,
            'observer_count': len(self.observed_states),
            'breakpoint_hit': self.breakpoint_hit,
            'in_function': bool(self.call_stack),
            'active_observers': list(self.observer_phases.keys()),
            'critical_strain': any(
                strain > self.simulation_config.get('strain_threshold', 0.8)
                for strain in self.field_strain.values()
            )
        }
    
    def register_event_listener(self, listener_id: int) -> None:
        """
        Register an event listener ID for tracking
        
        Args:
            listener_id: Event listener ID
        """
        self.event_listeners.append(listener_id)
    
    def register_visualizer_data(self, component: str, data: Any) -> None:
        """
        Register data for visualization
        
        Args:
            component: Visualization component name
            data: Data to visualize
        """
        self.visualization_data[component] = data
        
        # If this is a frame component, add to animation frames
        if component.startswith('frame_'):
            frame_data = {
                'step': self.simulation_step,
                'time': self.simulation_time,
                'component': component,
                'data': data
            }
            self.animation_frames.append(frame_data)
    
    def get_visualizer_data(self, component: Optional[str] = None) -> Any:
        """
        Get visualization data
        
        Args:
            component: Component name (None for all)
            
        Returns:
            Any: Visualization data
        """
        if component:
            return self.visualization_data.get(component)
        return self.visualization_data
    
    def get_animation_frames(self) -> List[Dict[str, Any]]:
        """
        Get all animation frames
        
        Returns:
            List[Dict[str, Any]]: Animation frames
        """
        return self.animation_frames
    
    def register_recursive_system(self, system_name: str, level: int, parent: Optional[str] = None) -> None:
        """
        Register a recursive system
        
        Args:
            system_name: System name
            level: Recursion level (0 = base)
            parent: Parent system name (None for base system)
        """
        self.recursion_levels[system_name] = level
        self.recursion_parents[system_name] = parent
        self.recursion_strain[system_name] = 0.0
        
        # Log registration
        self.log('debug', f"Registered recursive system: {system_name} (level {level})" +
               (f", parent: {parent}" if parent else ""))
    
    def register_recursive_boundary(self, lower_system: str, upper_system: str, permeability: float = 0.5) -> None:
        """
        Register a boundary between recursive systems
        
        Args:
            lower_system: Lower system name
            upper_system: Upper system name
            permeability: Boundary permeability (0.0 to 1.0)
        """
        if lower_system not in self.recursion_levels:
            raise ValueError(f"Unknown recursive system: {lower_system}")
        if upper_system not in self.recursion_levels:
            raise ValueError(f"Unknown recursive system: {upper_system}")
            
        # Validate levels
        if self.recursion_levels[lower_system] >= self.recursion_levels[upper_system]:
            raise ValueError(f"Lower system must have lower level than upper system")
            
        # Register boundary
        self.recursion_boundaries[(lower_system, upper_system)] = {
            'permeability': permeability,
            'created_at': time.time(),
            'crossings': 0,
            'last_modified': time.time()
        }
        
        self.log('debug', f"Registered recursive boundary: {lower_system} <-> {upper_system} " +
               f"(permeability: {permeability:.2f})")
    
    def record_recursive_boundary_crossing(self, lower_system: str, upper_system: str) -> None:
        """
        Record a crossing of a recursive boundary
        
        Args:
            lower_system: Lower system name
            upper_system: Upper system name
        """
        boundary_key = (lower_system, upper_system)
        if boundary_key not in self.recursion_boundaries:
            # Auto-create boundary if needed
            self.register_recursive_boundary(
                lower_system, upper_system, 
                self.simulation_config.get('recursive_boundary_permeability', 0.5)
            )
            
        # Update boundary stats
        self.recursion_boundaries[boundary_key]['crossings'] += 1
        self.recursion_boundaries[boundary_key]['last_modified'] = time.time()
        
        # Update strain
        permeability = self.recursion_boundaries[boundary_key]['permeability']
        strain_factor = 0.1 * (1.0 - permeability)  # Higher strain for lower permeability
        
        if lower_system in self.recursion_strain:
            self.recursion_strain[lower_system] += strain_factor
        
        if upper_system in self.recursion_strain:
            self.recursion_strain[upper_system] += strain_factor * 0.5  # Less effect on upper system
            
        # Update statistics
        self.statistics['recursive_boundary_crossings'] += 1
        
        # Log at appropriate level based on permeability
        if permeability < 0.3:
            self.log('warning', f"Low-permeability recursive boundary crossed: {lower_system} -> {upper_system}")
        else:
            self.log('debug', f"Recursive boundary crossed: {lower_system} -> {upper_system}")
    
    def get_recursive_ancestry(self, system_name: str) -> List[str]:
        """
        Get the ancestry chain of a recursive system
        
        Args:
            system_name: System name
            
        Returns:
            List[str]: System names in order from system to root
        """
        if system_name not in self.recursion_parents:
            return [system_name]
            
        ancestry = [system_name]
        parent = self.recursion_parents[system_name]
        
        while parent is not None:
            ancestry.append(parent)
            parent = self.recursion_parents.get(parent)
            
            # Safety check for circular references
            if len(ancestry) > 100 or parent in ancestry:
                break
                
        return ancestry
    
    def get_recursive_depth(self, system_name: str) -> int:
        """
        Get the recursive depth of a system
        
        Args:
            system_name: System name
            
        Returns:
            int: Recursive depth (0 for base system)
        """
        return self.recursion_levels.get(system_name, 0)
    
    def get_memory_allocation_stats(self) -> Dict[str, Any]:
        """
        Get memory allocation statistics
        
        Returns:
            Dict[str, Any]: Memory statistics
        """
        return {
            'total_used': sum(pool['used'] for pool in self.memory_pools.values()),
            'total_capacity': sum(pool['size'] for pool in self.memory_pools.values()),
            'utilization': sum(pool['used'] for pool in self.memory_pools.values()) / 
                          sum(pool['size'] for pool in self.memory_pools.values()) 
                          if sum(pool['size'] for pool in self.memory_pools.values()) > 0 else 0,
            'allocations': len(self.memory_allocations),
            'pools': {name: pool.copy() for name, pool in self.memory_pools.items()}
        }
    
    # Add cleanup for log buffer
    def cleanup(self) -> None:
        """
        Clean up resources used by the execution context
        """
        # If not already completed or errored, mark as completed
        if self.state not in (self.STATE_COMPLETED, self.STATE_ERROR):
            self.complete_execution()
        
        # Flush any remaining logs
        if hasattr(self, '_log_buffer') and self.log_to_file:
            self._flush_log_buffer()
        
        # Clear event listeners
        old_listeners = self.event_listeners.copy()
        self.event_listeners.clear()
        
        logger.debug(f"Execution context cleaned up, {len(old_listeners)} event listeners removed")
        