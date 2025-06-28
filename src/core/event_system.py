import datetime
import logging
import os
import time
import traceback
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union, TypedDict, Protocol, Tuple

logger = logging.getLogger(__name__)

# Type definitions for enhanced type checking
EventFilterFunc = Callable[[Dict[str, Any]], bool]
EventCallbackFunc = Callable[[Dict[str, Any]], None]


class EventType:
    """Constants for all standard event types"""
    # State events
    STATE_CREATION = "state_creation_event"
    STATE_DESTRUCTION = "state_destruction_event"
    COHERENCE_CHANGE = "coherence_change_event"
    ENTROPY_INCREASE = "entropy_increase_event"
    ENTROPY_DECREASE = "entropy_decrease_event"
    
    # Observer events
    OBSERVATION = "observation_event"
    OBSERVER_PHASE_CHANGE = "observer_phase_change_event"
    OBSERVER_CONSENSUS = "observer_consensus_event"
    
    # Entanglement events
    ENTANGLEMENT_CREATION = "entanglement_creation_event"
    ENTANGLEMENT_BREAKING = "entanglement_breaking_event"
    TELEPORTATION = "teleportation_event"
    
    # Measurement events
    MEASUREMENT = "measurement_event"
    DECOHERENCE = "decoherence_event"
    COLLAPSE = "collapse_event"
    
    # Error and stability events
    QUANTUM_ERROR = "quantum_error_event"
    STABILITY_THRESHOLD = "stability_threshold_event"
    
    # Convergence events
    CONVERGENCE = "convergence_event"
    DIVERGENCE = "divergence_event"
    RESONANCE = "resonance_event" 
    INTERFERENCE = "interference_event"
    
    # Memory events
    MEMORY_STRAIN = "memory_strain_event"
    CRITICAL_STRAIN = "critical_strain_event"
    MEMORY_RESONANCE = "memory_resonance_event"
    DEFRAGMENTATION = "defragmentation_event"
    COHERENCE_WAVE = "coherence_wave_event"
    ALIGNMENT = "alignment_event"
    
    # Boundary events
    RECURSIVE_BOUNDARY = "recursive_boundary_event"
    
    # Simulation events
    SIMULATION_TICK = "simulation_tick_event"
    
    # Field dynamics events
    COUPLING_APPLICATION = "coupling_application_event"
    
    # Hardware events
    HARDWARE_CONNECTION = "hardware_connection_event"
    HARDWARE_DISCONNECTION = "hardware_disconnection_event"
    COLLAPSE_EVENT = "collapse_event"
    TELEPORTATION_EVENT = "teleportation_event" 
    MEMORY_STRAIN = "memory_strain"
    RECURSIVE_BOUNDARY = "recursive_boundary"
    # Set of all standard event types
    ALL_TYPES = {
        STATE_CREATION, STATE_DESTRUCTION, COHERENCE_CHANGE, ENTROPY_INCREASE, ENTROPY_DECREASE,
        OBSERVATION, OBSERVER_PHASE_CHANGE, OBSERVER_CONSENSUS,
        ENTANGLEMENT_CREATION, ENTANGLEMENT_BREAKING, TELEPORTATION,
        MEASUREMENT, DECOHERENCE, COLLAPSE,
        QUANTUM_ERROR, STABILITY_THRESHOLD,
        CONVERGENCE, DIVERGENCE, RESONANCE, INTERFERENCE,
        MEMORY_STRAIN, CRITICAL_STRAIN, MEMORY_RESONANCE, DEFRAGMENTATION, COHERENCE_WAVE, ALIGNMENT,
        RECURSIVE_BOUNDARY,
        SIMULATION_TICK,
        COUPLING_APPLICATION,
        HARDWARE_CONNECTION, HARDWARE_DISCONNECTION, COLLAPSE_EVENT, TELEPORTATION_EVENT,MEMORY_STRAIN,RECURSIVE_BOUNDARY
    }


class EventData(TypedDict, total=False):
    """Type definition for event data structure"""
    id: str
    type: str
    data: Dict[str, Any]
    timestamp: float
    datetime: datetime.datetime
    source: Optional[str]
    related_to: Optional[str]
    recursion_depth: int


class ListenerData(TypedDict):
    """Type definition for listener data structure"""
    event_type: str
    callback: EventCallbackFunc
    filter: Optional[EventFilterFunc]
    priority: int
    created_at: datetime.datetime
    call_count: int
    description: Optional[str]
    last_called_at: Optional[datetime.datetime]
    last_event_handled: Optional[str]


class EventSystemError(Exception):
    """Base exception for event system errors"""
    pass


class InvalidEventTypeError(EventSystemError):
    """Exception raised when an invalid event type is used"""
    pass


class RecursionDepthExceeded(EventSystemError):
    """Exception raised when maximum recursion depth is exceeded"""
    pass


class ListenerExecutionError(EventSystemError):
    """Exception raised when a listener encounters an error"""
    def __init__(self, listener_id: int, original_error: Exception):
        self.listener_id = listener_id
        self.original_error = original_error
        super().__init__(f"Error in listener {listener_id}: {original_error}")


class EventContext:
    """Context manager for pausing/resuming the event system"""
    def __init__(self, event_system: 'EventSystem', pause: bool = True):
        self.event_system = event_system
        self.pause = pause
        self.was_paused = self.event_system.paused
        
    def __enter__(self):
        if self.pause and not self.was_paused:
            self.event_system.pause()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pause and not self.was_paused:
            self.event_system.resume()


class EventRegistry:
    """Manages event history and relationships"""
    def __init__(self, max_history: int = 1000):
        self.history: List[Dict[str, Any]] = []
        self.related_events: Dict[str, List[str]] = {}
        self.max_history = max_history
        
    def add_event(self, event: Dict[str, Any]) -> None:
        """Add an event to history and manage relationships"""
        self.history.append(event)
        
        # Track relationships
        related_to = event.get('related_to')
        if related_to:
            if related_to not in self.related_events:
                self.related_events[related_to] = []
            self.related_events[related_to].append(event['id'])
            
        # Trim history if needed
        self._trim_history()
    
    def _trim_history(self) -> None:
        """Ensure history stays within size limits"""
        if len(self.history) <= self.max_history:
            return
            
        # Remove oldest events
        to_remove = len(self.history) - self.max_history
        removed_events = self.history[:to_remove]
        self.history = self.history[to_remove:]
        
        # Clean up related events for removed events
        for removed_event in removed_events:
            removed_id = removed_event.get('id')
            if removed_id in self.related_events:
                del self.related_events[removed_id]


class ListenerManager:
    """Manages event listeners with lifecycle hooks"""
    def __init__(self):
        self.listeners: Dict[int, Dict[str, Any]] = {}
        self.next_id: int = 0
        
    def add(self, event_type: str, callback: EventCallbackFunc, 
           filter_func: Optional[EventFilterFunc] = None,
           priority: int = 0, description: Optional[str] = None) -> int:
        """Add a new listener"""
        listener_id = self.next_id
        self.next_id += 1
        
        self.listeners[listener_id] = {
            'event_type': event_type,
            'callback': callback,
            'filter': filter_func,
            'priority': priority,
            'created_at': datetime.datetime.now(),
            'call_count': 0,
            'description': description,
            'last_called_at': None,
            'last_event_handled': None
        }
        
        self._on_listener_added(listener_id, event_type)
        return listener_id
        
    def remove(self, listener_id: int) -> bool:
        """Remove a listener"""
        if listener_id not in self.listeners:
            return False
            
        event_type = self.listeners[listener_id]['event_type']
        self._on_listener_removed(listener_id, event_type)
        del self.listeners[listener_id]
        return True
        
    def get_by_event_type(self, event_type: str) -> List[Tuple[int, Dict[str, Any]]]:
        """Get all listeners for a specific event type"""
        result = [(lid, listener) for lid, listener in self.listeners.items()
                if listener['event_type'] == event_type or listener['event_type'] == '*']
                
        # Sort by priority
        result.sort(key=lambda x: x[1]['priority'], reverse=True)
        return result
        
    def update_stats(self, listener_id: int, event_id: Optional[str] = None) -> None:
        """Update listener statistics after execution"""
        if listener_id in self.listeners:
            self.listeners[listener_id]['call_count'] += 1
            self.listeners[listener_id]['last_called_at'] = datetime.datetime.now()
            if event_id:
                self.listeners[listener_id]['last_event_handled'] = event_id
    
    def get_all_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered listeners"""
        return [
            {
                'id': lid,
                'event_type': l['event_type'],
                'priority': l['priority'],
                'created_at': l['created_at'],
                'call_count': l['call_count'],
                'description': l.get('description', ''),
                'last_called_at': l.get('last_called_at'),
                'last_event_handled': l.get('last_event_handled')
            }
            for lid, l in self.listeners.items()
        ]
                
    def _on_listener_added(self, listener_id: int, event_type: str) -> None:
        """Hook called when a listener is added"""
        logger.debug(f"Added listener {listener_id} for event type {event_type}")
        
    def _on_listener_removed(self, listener_id: int, event_type: str) -> None:
        """Hook called when a listener is removed"""
        logger.debug(f"Removed listener {listener_id} for event type {event_type}")


class EventQueryBuilder:
    """Builder pattern for event queries"""
    def __init__(self, event_system: 'EventSystem'):
        self.event_system = event_system
        self.filters: Dict[str, Any] = {}
        
    def of_type(self, event_type: str) -> 'EventQueryBuilder':
        """Filter by event type"""
        self.filters['event_type'] = event_type
        return self
        
    def in_timespan(self, start_time: float, end_time: Optional[float] = None) -> 'EventQueryBuilder':
        """Filter by timespan"""
        self.filters['start_time'] = start_time
        if end_time:
            self.filters['end_time'] = end_time
        return self
        
    def from_source(self, source: str) -> 'EventQueryBuilder':
        """Filter by source"""
        self.filters['source'] = source
        return self
        
    def with_criteria(self, criteria: Dict[str, Any]) -> 'EventQueryBuilder':
        """Filter by custom criteria"""
        self.filters['match_criteria'] = criteria
        return self
        
    def include_related(self, include: bool = True) -> 'EventQueryBuilder':
        """Include related events"""
        self.filters['include_related'] = include
        return self
        
    def include_recursive(self, include: bool = True) -> 'EventQueryBuilder':
        """Include recursive events"""
        self.filters['include_recursive'] = include
        return self
        
    def limit(self, count: int) -> 'EventQueryBuilder':
        """Limit number of results"""
        self.filters['limit'] = count
        return self
        
    def execute(self) -> List[Dict[str, Any]]:
        """Execute the query and return matching events"""
        return self.event_system.get_events(
            event_type=self.filters.get('event_type'),
            limit=self.filters.get('limit'),
            start_time=self.filters.get('start_time'),
            end_time=self.filters.get('end_time'),
            source=self.filters.get('source'),
            include_related=self.filters.get('include_related', False),
            include_recursive=self.filters.get('include_recursive', False),
            match_criteria=self.filters.get('match_criteria')
        )
        
    def count(self) -> int:
        """Count matching events instead of returning them"""
        return self.event_system.count_events(
            event_type=self.filters.get('event_type'),
            time_window=(None if 'start_time' not in self.filters 
                        else time.time() - self.filters['start_time']),
            match_criteria=self.filters.get('match_criteria')
        )


class EventSystem:
    """
    Event system for the Recursia runtime.
    
    Manages events, listeners, and event history for the entire application,
    allowing components to communicate through a publish-subscribe pattern
    with advanced features like event filtering, querying, and relationship tracking.
    """
    
    def __init__(self, max_history: int = 1000, log_events: bool = True):
        """
        Initialize the event system
        
        Args:
            max_history: Maximum number of events to keep in history
            log_events: Whether to log events to the logging system
        """
        # Initialize components
        self.registry = EventRegistry(max_history)
        self.listener_manager = ListenerManager()
        
        # Event tracking
        self.event_types_registry: Set[str] = set(EventType.ALL_TYPES)
        self.paused: bool = False
        self.queued_events: List[Dict[str, Any]] = []
        
        # Configuration
        self.log_events: bool = log_events
        self.max_recursion_depth: int = 5
        self.max_history: int = max_history
        
        # Runtime stats
        self.event_stats: Dict[str, Any] = {
            'total_events': 0,
            'events_by_type': {},
            'listeners_called': 0,
            'errors': 0,
            'cascading_events': 0,
            'recursive_events': 0
        }
        
        # Track recursive event depth to prevent infinite loops
        self.recursion_depth: Dict[str, int] = {}
        
        # Initialize event stats
        for event_type in self.event_types_registry:
            self.event_stats['events_by_type'][event_type] = 0
    
    def register_event_type(self, event_type: str) -> None:
        """
        Register a custom event type
        
        Args:
            event_type: Event type name to register
        """
        self.event_types_registry.add(event_type)
        self.event_stats['events_by_type'][event_type] = 0
    
    def add_listener(self, 
                    event_type: str, 
                    callback: EventCallbackFunc, 
                    filter_func: Optional[EventFilterFunc] = None,
                    priority: int = 0,
                    description: Optional[str] = None) -> int:
        """
        Add an event listener
        
        Args:
            event_type: Event type to listen for (or '*' for all events)
            callback: Callback function to execute when event occurs
            filter_func: Optional function to filter events
            priority: Priority level (higher values execute first)
            description: Optional description of the listener purpose
            
        Returns:
            int: Listener ID for later removal
            
        Raises:
            InvalidEventTypeError: If event_type is not registered
        """
        if event_type != '*' and event_type not in self.event_types_registry:
            raise InvalidEventTypeError(f"Invalid event type: {event_type}")
        
        return self.listener_manager.add(
            event_type, callback, filter_func, priority, description
        )
    
    def remove_listener(self, listener_id: int) -> bool:
        """
        Remove an event listener
        
        Args:
            listener_id: Listener ID to remove
            
        Returns:
            bool: True if listener was removed successfully
        """
        return self.listener_manager.remove(listener_id)
    
    def remove_listeners_by_type(self, event_type: str) -> int:
        """
        Remove all listeners for a specific event type
        
        Args:
            event_type: Event type to remove listeners for
            
        Returns:
            int: Number of listeners removed
        """
        to_remove = [lid for lid, listener in self.listener_manager.listeners.items() 
                     if listener['event_type'] == event_type]
        
        count = 0
        for lid in to_remove:
            if self.remove_listener(lid):
                count += 1
        
        return count
    
    def get_listeners_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered listeners
        
        Returns:
            list: List of listener info dictionaries
        """
        return self.listener_manager.get_all_info()
    
    def _generate_event_id(self, event_type: str, timestamp: float) -> str:
        """
        Generate a unique event ID with virtually no collision chance
        
        Args:
            event_type: Event type
            timestamp: Event timestamp
            
        Returns:
            str: Unique event ID
        """
        unique_parts = [
            event_type,
            f"{timestamp:.6f}",
            str(id(event_type)),
            os.urandom(4).hex()  # Add 8 random hex chars
        ]
        return "_".join(unique_parts)
    
    def emit(self, 
            event_type: str, 
            event_data: Optional[Dict[str, Any]] = None, 
            source: Optional[str] = None,
            related_to: Optional[str] = None) -> int:
        """
        Emit an event
        
        Args:
            event_type: Event type
            event_data: Event data dictionary
            source: Source of the event (e.g. component name)
            related_to: Event ID that this event is related to
            
        Returns:
            int: Number of listeners notified
            
        Raises:
            InvalidEventTypeError: If event_type is not registered
        """
        if event_type not in self.event_types_registry:
            raise InvalidEventTypeError(f"Invalid event type: {event_type}")
        
        # Check recursion depth to prevent infinite event loops
        if event_type in self.recursion_depth:
            self.recursion_depth[event_type] += 1
            if self.recursion_depth[event_type] > self.max_recursion_depth:
                logger.warning(f"Maximum recursion depth reached for event type {event_type}, skipping emission")
                self.recursion_depth[event_type] -= 1
                return 0
        else:
            self.recursion_depth[event_type] = 1

        if event_data is None:
            event_data = {}
        
        # Create event object
        timestamp = time.time()
        event_id = self._generate_event_id(event_type, timestamp)
        
        event = {
            'id': event_id,
            'type': event_type,
            'data': event_data,
            'timestamp': timestamp,
            'datetime': datetime.datetime.fromtimestamp(timestamp),
            'source': source,
            'related_to': related_to,
            'recursion_depth': self.recursion_depth.get(event_type, 1)
        }
        
        # Record event relationship
        if related_to is not None:
            # Track cascading events
            self.event_stats['cascading_events'] += 1
            
            # Track recursive events (same type triggered by same type)
            if any(e.get('type') == event_type for e in self.registry.history 
                  if e.get('id') == related_to):
                self.event_stats['recursive_events'] += 1
        
        # If the system is paused, queue the event
        if self.paused:
            self.queued_events.append(event)
            # Decrement recursion counter before returning
            self.recursion_depth[event_type] -= 1
            return 0
        
        # Add to history
        self.registry.add_event(event)
        
        # Update statistics
        self.event_stats['total_events'] += 1
        self.event_stats['events_by_type'][event_type] = \
            self.event_stats['events_by_type'].get(event_type, 0) + 1
        
        # Log the event if enabled
        if self.log_events:
            logger.info(f"Event emitted: {event_type}" + 
                      (f" from {source}" if source else ""))
            logger.debug(f"Event data: {event_data}")
        
        # Notify listeners
        notified_count = self._process_event_with_listeners(event)
        
        # Decrement recursion counter
        self.recursion_depth[event_type] -= 1
        
        return notified_count
    
    def _process_event_with_listeners(self, event: Dict[str, Any]) -> int:
        """
        Process an event with registered listeners
        
        Args:
            event: Event to process
            
        Returns:
            int: Number of listeners notified
        """
        event_type = event.get('type', 'unknown')
        
        # Get relevant listeners sorted by priority
        relevant_listeners = self.listener_manager.get_by_event_type(event_type)
        
        # Notify listeners
        notified_count = 0
        
        for listener_id, listener in relevant_listeners:
            # Check filter
            if listener['filter'] is not None:
                try:
                    if not listener['filter'](event):
                        continue  # Skip this listener if filter rejects
                except Exception as e:
                    logger.error(f"Error in event filter for listener {listener_id}: {e}")
                    logger.debug(traceback.format_exc())
                    self.event_stats['errors'] += 1
                    continue
            
            # Call callback
            try:
                listener['callback'](event)
                
                # Update listener statistics
                self.listener_manager.update_stats(listener_id, event.get('id'))
                self.event_stats['listeners_called'] += 1
                
                notified_count += 1
            except Exception as e:
                logger.error(f"Error in event listener {listener_id}: {e}")
                logger.debug(traceback.format_exc())
                self.event_stats['errors'] += 1
        
        return notified_count
    
    def emit_quantum_event(self, 
                         event_type: str, 
                         state_name: str,
                         state_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to a quantum state or operation
        
        Args:
            event_type: Event type
            state_name: Name of the quantum state
            state_data: Additional state data
            
        Returns:
            int: Number of listeners notified
        """
        quantum_events = {
            EventType.STATE_CREATION, EventType.STATE_DESTRUCTION, 
            EventType.COHERENCE_CHANGE, EventType.ENTROPY_INCREASE, 
            EventType.ENTROPY_DECREASE, EventType.DECOHERENCE,
            EventType.COLLAPSE, EventType.MEASUREMENT, 
            EventType.QUANTUM_ERROR, EventType.STABILITY_THRESHOLD,
            EventType.CONVERGENCE, EventType.DIVERGENCE
        }
        
        if event_type not in quantum_events:
            raise InvalidEventTypeError(f"Invalid quantum event type: {event_type}")
            
        if state_data is None:
            state_data = {}
        
        event_data = {
            'state_name': state_name,
            'state_data': state_data
        }
        
        return self.emit(event_type, event_data, "quantum_system")
    
    def emit_entanglement_event(self,
                              state1: str,
                              state2: str,
                              event_type: str,
                              entanglement_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to entanglement between quantum states
        
        Args:
            state1: First state name
            state2: Second state name
            event_type: Event type
            entanglement_data: Additional entanglement data
            
        Returns:
            int: Number of listeners notified
        """
        valid_types = {
            EventType.ENTANGLEMENT_CREATION,
            EventType.ENTANGLEMENT_BREAKING,
            EventType.TELEPORTATION
        }
        
        if event_type not in valid_types:
            raise InvalidEventTypeError(f"Invalid entanglement event type: {event_type}")
            
        if entanglement_data is None:
            entanglement_data = {}
            
        event_data = {
            'state1': state1,
            'state2': state2,
            'entanglement_data': entanglement_data
        }
        
        return self.emit(event_type, event_data, "quantum_system")
    
    def emit_observation_event(self,
                             observer_name: str,
                             observed_state: str,
                             observation_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to an observer observing a quantum state
        
        Args:
            observer_name: Observer name
            observed_state: Observed state name
            observation_data: Additional observation data
            
        Returns:
            int: Number of listeners notified
        """
        if observation_data is None:
            observation_data = {}
            
        event_data = {
            'observer_name': observer_name,
            'observed_state': observed_state,
            'observation_data': observation_data
        }
        
        return self.emit(EventType.OBSERVATION, event_data, "observer_system")
    
    def emit_memory_field_event(self,
                              event_type: str,
                              region_name: Optional[str] = None,
                              field_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to memory field physics
        
        Args:
            event_type: Event type related to memory fields
            region_name: Optional name of a specific memory region
            field_data: Additional memory field data
            
        Returns:
            int: Number of listeners notified
        """
        memory_field_events = {
            EventType.MEMORY_STRAIN, EventType.CRITICAL_STRAIN, 
            EventType.MEMORY_RESONANCE, EventType.DEFRAGMENTATION, 
            EventType.COHERENCE_WAVE, EventType.ALIGNMENT
        }
        
        if event_type not in memory_field_events:
            raise InvalidEventTypeError(f"Invalid memory field event type: {event_type}")
            
        if field_data is None:
            field_data = {}
            
        event_data = {
            'region_name': region_name,
            'field_data': field_data
        }
        
        return self.emit(event_type, event_data, "memory_field_system")
    
    def emit_recursive_event(self,
                           system_name: str,
                           related_system: Optional[str] = None,
                           event_type: str = EventType.RECURSIVE_BOUNDARY,
                           recursion_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to recursive mechanics between systems
        
        Args:
            system_name: Name of the system generating the event
            related_system: Optional related system name
            event_type: Event type (default: recursive_boundary_event)
            recursion_data: Additional recursion-related data
            
        Returns:
            int: Number of listeners notified
        """
        valid_types = {EventType.RECURSIVE_BOUNDARY, EventType.OBSERVER_CONSENSUS}
        
        if event_type not in valid_types:
            raise InvalidEventTypeError(f"Invalid recursive event type: {event_type}")
            
        if recursion_data is None:
            recursion_data = {}
            
        event_data = {
            'system_name': system_name,
            'related_system': related_system,
            'recursion_data': recursion_data
        }
        
        return self.emit(event_type, event_data, "recursive_mechanics")
    
    def emit_hardware_event(self,
                          event_type: str,
                          provider: str,
                          device: Optional[str] = None,
                          hardware_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to quantum hardware connection/disconnection
        
        Args:
            event_type: Either hardware_connection_event or hardware_disconnection_event
            provider: Hardware provider name (ibm, rigetti, google, etc.)
            device: Optional specific device name
            hardware_data: Additional hardware-related data
            
        Returns:
            int: Number of listeners notified
        """
        valid_types = {EventType.HARDWARE_CONNECTION, EventType.HARDWARE_DISCONNECTION}
        
        if event_type not in valid_types:
            raise InvalidEventTypeError(f"Invalid hardware event type: {event_type}")
            
        if hardware_data is None:
            hardware_data = {}
            
        event_data = {
            'provider': provider,
            'device': device,
            'hardware_data': hardware_data
        }
        
        return self.emit(event_type, event_data, "hardware_system")
    
    def emit_simulation_event(self,
                            event_type: str = EventType.SIMULATION_TICK,
                            simulation_time: float = 0.0,
                            tick_number: int = 0,
                            simulation_data: Optional[Dict[str, Any]] = None) -> int:
        """
        Emit an event related to simulation progress
        
        Args:
            event_type: Event type
            simulation_time: Current simulation time
            tick_number: Current simulation tick/step number
            simulation_data: Additional simulation data
            
        Returns:
            int: Number of listeners notified
        """
        if simulation_data is None:
            simulation_data = {}
            
        event_data = {
            'simulation_time': simulation_time,
            'tick_number': tick_number,
            'simulation_data': simulation_data
        }
        
        return self.emit(event_type, event_data, "simulation_system")
    
    def pause(self) -> None:
        """
        Pause event processing
        Events will be queued until resume() is called
        """
        self.paused = True
        logger.info("Event system paused")
    
    def resume(self) -> int:
        """
        Resume event processing and process any queued events
        
        Returns:
            int: Number of queued events processed
        """
        if not self.paused:
            return 0
            
        self.paused = False
        
        # Process queued events
        queued_count = len(self.queued_events)
        total_notified = 0
        
        # Make a copy of the queue to prevent infinite loops
        # if events during processing trigger more events
        events_to_process = self.queued_events.copy()
        self.queued_events = []
        
        for event in events_to_process:
            # Add to history
            self.registry.add_event(event)
            
            # Update statistics 
            event_type = event.get('type', 'unknown')
            self.event_stats['total_events'] += 1
            self.event_stats['events_by_type'][event_type] = \
                self.event_stats['events_by_type'].get(event_type, 0) + 1
            
            # Process the event with listeners
            notified = self._process_event_with_listeners(event)
            total_notified += notified
        
        logger.info(f"Event system resumed, processed {queued_count} queued events")
        return total_notified
    
    def event_context(self, pause: bool = True) -> EventContext:
        """Create an event context manager for use with 'with'"""
        return EventContext(self, pause)
    
    def query(self) -> EventQueryBuilder:
        """Create a query builder for fluent event querying"""
        return EventQueryBuilder(self)
    
    def get_events(self, 
                  event_type: Optional[str] = None, 
                  limit: Optional[int] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  source: Optional[str] = None,
                  include_related: bool = False,
                  include_recursive: bool = False,
                  match_criteria: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get events from history with advanced filtering
        
        Args:
            event_type: Event type filter
            limit: Maximum number of events to return
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            source: Filter events by source
            include_related: Whether to include related events
            include_recursive: Whether to include recursive events
            match_criteria: Dictionary of criteria to match in event data
            
        Returns:
            list: List of filtered events
        """
        # Start with all events
        events = self.registry.history.copy()
        
        # Apply filters
        if event_type is not None:
            events = [e for e in events if e['type'] == event_type]
        
        if start_time is not None:
            events = [e for e in events if e['timestamp'] >= start_time]
        
        if end_time is not None:
            events = [e for e in events if e['timestamp'] <= end_time]
        
        if source is not None:
            events = [e for e in events if e.get('source') == source]
        
        # Apply match criteria if provided
        if match_criteria is not None:
            filtered_events = []
            for event in events:
                match = True
                for key, value in match_criteria.items():
                    # Handle nested keys with dot notation (e.g., "data.state_name")
                    keys = key.split('.')
                    event_value = event
                    for k in keys:
                        if isinstance(event_value, dict) and k in event_value:
                            event_value = event_value[k]
                        else:
                            event_value = None
                            break
                    
                    if event_value != value:
                        match = False
                        break
                
                if match:
                    filtered_events.append(event)
            
            events = filtered_events
        
        # Get related events if requested
        if include_related:
            related_events = []
            event_ids = [e.get('id') for e in events if 'id' in e]
            
            for event_id in event_ids:
                if event_id in self.registry.related_events:
                    related_ids = self.registry.related_events[event_id]
                    for related_id in related_ids:
                        # Find matching events in history
                        for history_event in self.registry.history:
                            if history_event.get('id') == related_id and history_event not in events:
                                related_events.append(history_event)
            
            # Add related events
            events.extend(related_events)
        
        # Get recursive events if requested (same type triggered by same type)
        if include_recursive and len(events) > 0:
            recursive_events = []
            for event in events:
                event_id = event.get('id')
                event_type = event.get('type')
                
                if event_id and event_type:
                    # Find events of same type triggered by this event
                    for history_event in self.registry.history:
                        if (history_event.get('type') == event_type and 
                            history_event.get('related_to') == event_id and
                            history_event not in events and
                            history_event not in recursive_events):
                            recursive_events.append(history_event)
            
            # Add recursive events
            events.extend(recursive_events)
            
        # Sort by timestamp
        events.sort(key=lambda e: e.get('timestamp', 0))
        
        # Apply limit
        if limit is not None:
            events = events[-limit:]
        
        return events
    
    def clear_history(self) -> int:
        """
        Clear the event history
        
        Returns:
            int: Number of events cleared
        """
        count = len(self.registry.history)
        self.registry.history = []
        self.registry.related_events = {}
        logger.info(f"Cleared {count} events from history")
        return count
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the event system
        
        Returns:
            dict: System statistics
        """
        # Calculate additional derived statistics
        active_listener_types = set(l['event_type'] for l in self.listener_manager.listeners.values())
        most_common_event_type = max(
            self.event_stats['events_by_type'].items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]
        
        # Calculate average events per second over the last minute
        current_time = time.time()
        one_minute_ago = current_time - 60
        recent_events = [e for e in self.registry.history if e['timestamp'] > one_minute_ago]
        events_per_second = len(recent_events) / 60 if recent_events else 0
        
        return {
            'total_events': self.event_stats['total_events'],
            'events_by_type': self.event_stats['events_by_type'],
            'listeners_called': self.event_stats['listeners_called'],
            'errors': self.event_stats['errors'],
            'cascading_events': self.event_stats['cascading_events'],
            'recursive_events': self.event_stats['recursive_events'],
            'active_listeners': len(self.listener_manager.listeners),
            'active_listener_types': list(active_listener_types),
            'history_size': len(self.registry.history),
            'queued_events': len(self.queued_events),
            'paused': self.paused,
            'most_common_event_type': most_common_event_type,
            'events_per_second': events_per_second
        }
        
    def has_event_occurred(self, 
                         event_type: str, 
                         time_window: Optional[float] = None,
                         match_criteria: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a specific event has occurred within a time window
        
        Args:
            event_type: Type of event to check for
            time_window: Time window in seconds (None for all history)
            match_criteria: Dictionary of criteria to match in event data
            
        Returns:
            bool: True if matching event found
        """
        # Set time window boundary
        if time_window is not None:
            start_time = time.time() - time_window
        else:
            start_time = 0
        
        # Search for matching events
        for event in reversed(self.registry.history):  # Start with most recent
            # Check event type
            if event['type'] != event_type:
                continue
                
            # Check time window
            if event['timestamp'] < start_time:
                return False  # We've gone beyond our time window
                
            # Check match criteria
            if match_criteria is not None:
                match = True
                for key, value in match_criteria.items():
                    # Handle nested keys with dot notation (e.g., "data.state_name")
                    keys = key.split('.')
                    event_value = event
                    for k in keys:
                        if isinstance(event_value, dict) and k in event_value:
                            event_value = event_value[k]
                        else:
                            event_value = None
                            break
                    
                    if event_value != value:
                        match = False
                        break
                
                if match:
                    return True
            else:
                # No criteria specified, any event of this type matches
                return True
        
        return False
    
    def count_events(self,
                   event_type: Optional[str] = None,
                   time_window: Optional[float] = None,
                   match_criteria: Optional[Dict[str, Any]] = None) -> int:
        """
        Count events matching criteria
        
        Args:
            event_type: Type of event to count (None for all types)
            time_window: Time window in seconds (None for all history)
            match_criteria: Dictionary of criteria to match in event data
            
        Returns:
            int: Number of matching events
        """
        # Set time window boundary
        if time_window is not None:
            start_time = time.time() - time_window
        else:
            start_time = 0
        
        # Filter and count events
        count = 0
        for event in self.registry.history:
            # Check event type
            if event_type is not None and event['type'] != event_type:
                continue
                
            # Check time window
            if event['timestamp'] < start_time:
                continue
                
            # Check match criteria
            if match_criteria is not None:
                match = True
                for key, value in match_criteria.items():
                    # Handle nested keys with dot notation
                    keys = key.split('.')
                    event_value = event
                    for k in keys:
                        if isinstance(event_value, dict) and k in event_value:
                            event_value = event_value[k]
                        else:
                            event_value = None
                            break
                    
                    if event_value != value:
                        match = False
                        break
                
                if not match:
                    continue
            
            count += 1
        
        return count
    
    def get_event_rate(self,
                     event_type: Optional[str] = None,
                     time_window: float = 60.0) -> float:
        """
        Calculate the rate of events per second
        
        Args:
            event_type: Type of event to measure (None for all types)
            time_window: Time window in seconds
            
        Returns:
            float: Events per second
        """
        # Get current time
        current_time = time.time()
        start_time = current_time - time_window
        
        # Count events in window
        count = 0
        for event in self.registry.history:
            if event['timestamp'] < start_time:
                continue
                
            if event_type is not None and event['type'] != event_type:
                continue
                
            count += 1
        
        # Calculate rate
        return count / time_window
    
    def get_event_timeline(self, timespan: Optional[float] = None, 
                         group_by: str = "type") -> Dict[str, List[Dict[str, Any]]]:
        """
        Get a timeline of events for visualization
        
        Args:
            timespan: Optional timespan in seconds (None for all history)
            group_by: How to group events ('type', 'source', or None)
            
        Returns:
            dict: Timeline data grouped by the specified field
        """
        # Get events within timespan
        if timespan:
            start_time = time.time() - timespan
            events = [e for e in self.registry.history if e['timestamp'] >= start_time]
        else:
            events = self.registry.history.copy()
        
        # Sort by timestamp
        events.sort(key=lambda e: e['timestamp'])
        
        # Group events
        if group_by is None:
            return {"all": events}
        
        result = {}
        for event in events:
            key = event.get(group_by, "unknown")
            if key not in result:
                result[key] = []
            result[key].append(event)
        
        return result
    
    def get_event_heatmap(self, event_type: Optional[str] = None, 
                        timespan: float = 3600, intervals: int = 60) -> List[int]:
        """
        Get event frequency data for heatmap visualization
        
        Args:
            event_type: Type of event to count (None for all types)
            timespan: Total time span in seconds
            intervals: Number of intervals to divide the timespan into
            
        Returns:
            list: Counts per interval for heatmap visualization
        """
        interval_size = timespan / intervals
        now = time.time()
        result = [0] * intervals
        
        # Filter events by type and timespan
        events = [e for e in self.registry.history 
                 if (event_type is None or e['type'] == event_type)
                 and e['timestamp'] >= now - timespan]
        
        # Count events per interval
        for event in events:
            interval = min(intervals - 1, 
                         int((now - event['timestamp']) / interval_size))
            result[interval] += 1
        
        return result
    
    def _detect_runtime_features(self, runtime) -> Dict[str, bool]:
        """Detect available features in the runtime for conditional hook registration"""
        return {
            'coherence': hasattr(runtime, 'coherence_manager') and runtime.coherence_manager is not None,
            'entanglement': hasattr(runtime, 'entanglement_manager') and runtime.entanglement_manager is not None,
            'observer': hasattr(runtime, 'observer_dynamics') and runtime.observer_dynamics is not None,
            'memory': hasattr(runtime, 'memory_field_physics') and runtime.memory_field_physics is not None,
            'recursive': hasattr(runtime, 'recursive_mechanics') and runtime.recursive_mechanics is not None,
            'hardware': hasattr(runtime, 'hardware_backend') and runtime.hardware_backend is not None,
            'visualization': hasattr(runtime, 'visualization_helper') and runtime.visualization_helper is not None,
        }
    
    def register_hooks_for_runtime(self, runtime) -> List[int]:
        """
        Register standard event hooks for the Recursia runtime
        
        Args:
            runtime: The Recursia runtime instance
            
        Returns:
            list: List of registered listener IDs
        """
        listener_ids = []
        features = self._detect_runtime_features(runtime)
        
        # Factory functions to create hook handlers with proper closure over runtime
        def create_state_change_hook(runtime_ref):
            def on_state_change(event):
                state_name = event['data'].get('state_name')
                state_data = event['data'].get('state_data', {})
                
                # Check coherence/entropy changes to emit specific events
                old_coherence = state_data.get('previous_coherence')
                new_coherence = state_data.get('coherence')
                old_entropy = state_data.get('previous_entropy') 
                new_entropy = state_data.get('entropy')
                
                if old_coherence is not None and new_coherence is not None and old_coherence != new_coherence:
                    self.emit_quantum_event(EventType.COHERENCE_CHANGE, state_name, {
                        'previous': old_coherence,
                        'current': new_coherence,
                        'delta': new_coherence - old_coherence
                    })
                    
                if old_entropy is not None and new_entropy is not None:
                    if new_entropy > old_entropy:
                        self.emit_quantum_event(EventType.ENTROPY_INCREASE, state_name, {
                            'previous': old_entropy,
                            'current': new_entropy,
                            'delta': new_entropy - old_entropy
                        })
                    elif new_entropy < old_entropy:
                        self.emit_quantum_event(EventType.ENTROPY_DECREASE, state_name, {
                            'previous': old_entropy,
                            'current': new_entropy,
                            'delta': old_entropy - new_entropy
                        })
            return on_state_change
        
        def create_observer_interaction_hook(runtime_ref):
            def on_observer_interaction(event):
                observer_data = event['data']
                observer_name = observer_data.get('observer_name')
                observed_state = observer_data.get('observed_state')
                
                # Check if coherence is below collapse threshold
                coherence = observer_data.get('observation_data', {}).get('coherence')
                collapse_threshold = observer_data.get('observation_data', {}).get('collapse_threshold', 0.3)
                
                if coherence is not None and coherence < collapse_threshold:
                    self.emit_quantum_event(EventType.COLLAPSE, observed_state, {
                        'observer': observer_name,
                        'coherence': coherence,
                        'threshold': collapse_threshold
                    })
            return on_observer_interaction
        
        def create_memory_field_hook(runtime_ref):
            def on_memory_field_update(event):
                field_data = event['data'].get('field_data', {})
                region_name = event['data'].get('region_name')
                
                # Check for critical strain events
                strain = field_data.get('strain')
                critical_threshold = field_data.get('critical_threshold', 0.8)
                
                if strain is not None and strain > critical_threshold:
                    self.emit_memory_field_event(EventType.CRITICAL_STRAIN, region_name, {
                        'strain': strain,
                        'threshold': critical_threshold,
                        'timestamp': time.time()
                    })
            return on_memory_field_update
        
        def create_simulation_tick_hook(runtime_ref):
            def on_simulation_tick(event):
                tick_data = event['data']
                tick_number = tick_data.get('tick_number', 0)
                simulation_time = tick_data.get('simulation_time', 0.0)
                
                # Detect emergent phenomena based on system state
                coherence_values = tick_data.get('simulation_data', {}).get('coherence_values', [])
                strain_values = tick_data.get('simulation_data', {}).get('strain_values', [])
                observer_count = tick_data.get('simulation_data', {}).get('active_observers', 0)
                observer_consensus = tick_data.get('simulation_data', {}).get('observer_consensus', 0.0)
                
                # Detect coherence waves via standard deviation spikes
                if coherence_values and len(coherence_values) > 1:
                    try:
                        import numpy as np
                        coherence_std = np.std(coherence_values)
                        if coherence_std > 0.2:  # Threshold for wave detection
                            self.emit_memory_field_event(EventType.COHERENCE_WAVE, None, {
                                'tick': tick_number,
                                'std_deviation': coherence_std,
                                'values': coherence_values
                            })
                    except ImportError:
                        # Fall back to manual calculation if numpy not available
                        if coherence_values:
                            mean = sum(coherence_values) / len(coherence_values)
                            variance = sum((x - mean) ** 2 for x in coherence_values) / len(coherence_values)
                            std_dev = variance ** 0.5
                            if std_dev > 0.2:
                                self.emit_memory_field_event(EventType.COHERENCE_WAVE, None, {
                                    'tick': tick_number,
                                    'std_deviation': std_dev,
                                    'values': coherence_values
                                })
                    except Exception as e:
                        logger.warning(f"Error detecting coherence waves: {e}")
                
                # Detect critical strain events
                if strain_values and any(strain > 0.8 for strain in strain_values):
                    self.emit_memory_field_event(EventType.CRITICAL_STRAIN, None, {
                        'tick': tick_number,
                        'max_strain': max(strain_values),
                        'values': strain_values
                    })
                
                # Detect observer consensus
                if observer_count > 2 and observer_consensus > 0.7:
                    self.emit_recursive_event('simulation', None, EventType.OBSERVER_CONSENSUS, {
                        'tick': tick_number,
                        'observer_count': observer_count,
                        'consensus_level': observer_consensus
                    })
            return on_simulation_tick
        
        def create_recursive_boundary_hook(runtime_ref):
            def on_recursive_boundary(event):
                boundary_data = event['data']
                system_name = boundary_data.get('system_name')
                layer_permeability = boundary_data.get('recursion_data', {}).get('layer_permeability', 1.0)
                
                # Detect recursive layer leakage
                if layer_permeability < 0.3:
                    noise_level = boundary_data.get('recursion_data', {}).get('noise_level', 0.0)
                    if noise_level > 0.5:
                        self.emit_recursive_event(system_name, None, EventType.RECURSIVE_BOUNDARY, {
                            'boundary_breach': True,
                            'permeability': layer_permeability,
                            'noise_level': noise_level
                        })
            return on_recursive_boundary
        
        def create_coherence_change_hook(runtime_ref):
            def on_coherence_change(event):
                state_name = event['data'].get('state_name')
                coherence_data = event['data'].get('state_data', {})
                
                # Record state coherence change
                current = coherence_data.get('current')
                previous = coherence_data.get('previous')
                delta = coherence_data.get('delta')
                
                if delta and abs(delta) > 0.3:  # Significant change
                    if delta > 0:
                        logger.info(f"Significant coherence increase for {state_name}: {previous:.2f} → {current:.2f}")
                    else:
                        logger.info(f"Significant coherence decrease for {state_name}: {previous:.2f} → {current:.2f}")
            return on_coherence_change
        
        def create_entanglement_creation_hook(runtime_ref):
            def on_entanglement_creation(event):
                entanglement_data = event['data']
                state1 = entanglement_data.get('state1')
                state2 = entanglement_data.get('state2')
                
                # Record entanglement creation
                logger.info(f"Entanglement created between {state1} and {state2}")
                
                # Update internal entanglement registry if needed
                strength = entanglement_data.get('entanglement_data', {}).get('strength', 1.0)
                if features['coherence']:
                    try:
                        # Potentially boost coherence based on entanglement
                        runtime.coherence_manager.increase_coherence(state1, 0.1)
                        runtime.coherence_manager.increase_coherence(state2, 0.1)
                    except Exception as e:
                        logger.warning(f"Error updating coherence after entanglement: {e}")
            return on_entanglement_creation
        
        def create_observer_phase_change_hook(runtime_ref):
            def on_observer_phase_change(event):
                observer_data = event['data']
                if 'phase_change' in observer_data:
                    observer_name = observer_data.get('observer_name')
                    old_phase = observer_data.get('previous_phase')
                    new_phase = observer_data.get('current_phase')
                    
                    logger.info(f"Observer {observer_name} phase change: {old_phase} → {new_phase}")
                    
                    # If observer enters 'measuring' phase, trigger special handling
                    if new_phase == 'measuring' and features['observer']:
                        try:
                            observed_states = runtime.observer_registry.get_observer_observations(observer_name)
                            for state in observed_states:
                                self.emit_observation_event(
                                    observer_name, 
                                    state,
                                    {'phase_triggered': True, 'new_phase': new_phase}
                                )
                        except Exception as e:
                            logger.warning(f"Error handling observer phase change: {e}")
            return on_observer_phase_change
        
        def create_recursion_depth_change_hook(runtime_ref):
            def on_recursion_depth_change(event):
                recursion_data = event['data']
                system_name = recursion_data.get('system_name')
                old_depth = recursion_data.get('previous_depth')
                new_depth = recursion_data.get('current_depth')
                
                # Significant recursion depth increases might impact memory strain
                if new_depth > old_depth and features['memory']:
                    try:
                        # Increase memory strain proportionally to depth change
                        strain_increase = (new_depth - old_depth) * 0.1
                        runtime.memory_field_physics.add_memory_strain(system_name, strain_increase)
                        logger.info(f"Increased memory strain for {system_name} due to recursion depth change")
                    except Exception as e:
                        logger.warning(f"Error handling recursion depth change: {e}")
            return on_recursion_depth_change
        
        def create_hardware_connection_hook(runtime_ref):
            def on_hardware_connection(event):
                hardware_data = event['data']
                provider = hardware_data.get('provider')
                device = hardware_data.get('device')
                
                logger.info(f"Connected to quantum hardware: {provider} {device or ''}")
                
                # Validate that runtime is in the right mode for hardware execution
                if hasattr(runtime, 'simulator') and runtime.simulator is not None:
                    logger.warning("Both simulator and hardware backend active - potential conflict")
            return on_hardware_connection
        
        def create_hardware_disconnection_hook(runtime_ref):
            def on_hardware_disconnection(event):
                hardware_data = event['data']
                provider = hardware_data.get('provider')
                
                logger.info(f"Disconnected from quantum hardware: {provider}")
                
                # Check if we should fall back to simulator
                if hasattr(runtime, 'simulator') and runtime.simulator is None:
                    logger.info("Hardware disconnected, consider initializing simulator fallback")
            return on_hardware_disconnection
        
        def create_unified_osh_model_hook(runtime_ref):
            def on_unified_osh_event(event):
                # This is a special handler for the full Organic Simulation Hypothesis model
                # It integrates across all systems for emergent behavior
                event_type = event['type']
                event_data = event['data']
                
                # Coherence/entropy changes may impact memory field
                if event_type in [EventType.COHERENCE_CHANGE, EventType.ENTROPY_INCREASE, EventType.ENTROPY_DECREASE]:
                    state_name = event_data.get('state_name')
                    
                    # Map quantum coherence to memory field coherence
                    if (features['memory'] and hasattr(runtime, 'state_registry') and 
                        state_name is not None):
                        try:
                            quantum_state = runtime.state_registry.get_state(state_name)
                            if quantum_state:
                                # Find or create memory region corresponding to quantum state
                                region_name = f"region_{state_name}"
                                
                                # Adjust memory field coherence based on quantum coherence
                                new_coherence = event_data.get('state_data', {}).get('current')
                                if new_coherence is not None:
                                    runtime.memory_field_physics.modify_coherence(region_name, new_coherence - 0.5)
                                
                                # Adjust memory field entropy based on quantum entropy
                                new_entropy = event_data.get('state_data', {}).get('current')
                                if new_entropy is not None:
                                    runtime.memory_field_physics.adjust_entropy(region_name, new_entropy - 0.5)
                        except Exception as e:
                            logger.warning(f"Error in OSH unified model: {e}")
                
                # Observer effects cascade into recursive mechanics
                if event_type == EventType.OBSERVATION:
                    observer_name = event_data.get('observer_name')
                    state_name = event_data.get('observed_state')
                    
                    # Observations might trigger recursive boundary events
                    if (features['observer'] and features['recursive'] and 
                        observer_name is not None and state_name is not None):
                        try:
                            observer = runtime.observer_registry.get_observer(observer_name)
                            if observer:
                                # Get observer awareness level
                                awareness = observer.get('observer_self_awareness', 0.5)
                                
                                # Higher awareness might breach recursive boundaries
                                if awareness > 0.7:
                                    # Check if this state belongs to a recursive system
                                    system_for_state = runtime.recursive_mechanics.get_system_for_entity(state_name)
                                    if system_for_state:
                                        # Emit recursive boundary event
                                        self.emit_recursive_event(
                                            system_for_state,
                                            None,
                                            EventType.RECURSIVE_BOUNDARY,
                                            {
                                                'observer_initiated': True,
                                                'observer_name': observer_name,
                                                'observer_awareness': awareness,
                                                'target_state': state_name
                                            }
                                        )
                        except Exception as e:
                            logger.warning(f"Error in OSH observer-recursion integration: {e}")
            return on_unified_osh_event
        
        # Register standard hooks across all runtimes
        listener_ids.append(self.add_listener(
            EventType.STATE_CREATION, 
            create_state_change_hook(runtime),
            priority=10,
            description="Monitor state changes for coherence/entropy events"
        ))
        
        listener_ids.append(self.add_listener(
            EventType.OBSERVATION, 
            create_observer_interaction_hook(runtime),
            priority=10,
            description="Monitor observer interactions for collapse events"
        ))
        
        listener_ids.append(self.add_listener(
            EventType.MEMORY_STRAIN, 
            create_memory_field_hook(runtime),
            priority=5,
            description="Monitor memory field strain for critical events"
        ))
        
        listener_ids.append(self.add_listener(
            EventType.SIMULATION_TICK, 
            create_simulation_tick_hook(runtime),
            priority=5,
            description="Monitor simulation ticks for emergent phenomena"
        ))
        
        listener_ids.append(self.add_listener(
            EventType.RECURSIVE_BOUNDARY, 
            create_recursive_boundary_hook(runtime),
            priority=5,
            description="Monitor recursive boundaries for layer breaches"
        ))
        
        # Register feature-specific hooks based on runtime capabilities
        
        if features['coherence']:
            listener_ids.append(self.add_listener(
                EventType.COHERENCE_CHANGE, 
                create_coherence_change_hook(runtime),
                priority=3,
                description="Monitor significant coherence changes"
            ))
        
        if features['entanglement']:
            listener_ids.append(self.add_listener(
                EventType.ENTANGLEMENT_CREATION, 
                create_entanglement_creation_hook(runtime),
                priority=3,
                description="Monitor entanglement creation"
            ))
        
        if features['observer']:
            # Register custom event type for observer phase changes
            self.register_event_type("observer_phase_change_event")
            
            listener_ids.append(self.add_listener(
                "observer_phase_change_event", 
                create_observer_phase_change_hook(runtime),
                priority=8,
                description="Monitor observer phase changes"
            ))
        
        if features['recursive']:
            # Register custom event type for recursion depth changes
            self.register_event_type("recursion_depth_change_event")
            
            listener_ids.append(self.add_listener(
                "recursion_depth_change_event", 
                create_recursion_depth_change_hook(runtime),
                priority=7,
                description="Monitor recursion depth changes"
            ))
        
        if features['hardware']:
            listener_ids.append(self.add_listener(
                EventType.HARDWARE_CONNECTION, 
                create_hardware_connection_hook(runtime),
                priority=10,
                description="Monitor hardware connections"
            ))
            
            listener_ids.append(self.add_listener(
                EventType.HARDWARE_DISCONNECTION, 
                create_hardware_disconnection_hook(runtime),
                priority=10,
                description="Monitor hardware disconnections"
            ))
        
        # Register unified OSH model hook when all required components are present
        if all([features['coherence'], features['memory'], features['observer'], features['recursive']]):
            # Register this unified handler for multiple event types
            for event_type in [
                EventType.COHERENCE_CHANGE, 
                EventType.ENTROPY_INCREASE, 
                EventType.ENTROPY_DECREASE, 
                EventType.OBSERVATION,
                EventType.RECURSIVE_BOUNDARY, 
                EventType.CRITICAL_STRAIN
            ]:
                listener_ids.append(self.add_listener(
                    event_type, 
                    create_unified_osh_model_hook(runtime),
                    priority=1,  # Lower priority to run after specific handlers
                    description="Unified OSH model integration"
                ))
        
        return listener_ids
    
    def disable_hooks_for_runtime(self, listener_ids: List[int]) -> int:
        """
        Disable hooks previously registered for a runtime
        
        Args:
            listener_ids: List of listener IDs to disable
            
        Returns:
            int: Number of listeners successfully removed
        """
        removed_count = 0
        for listener_id in listener_ids:
            if self.remove_listener(listener_id):
                removed_count += 1
        
        return removed_count
    
    def validate_event_type(self, event_type: str) -> bool:
        """
        Validate that an event type is registered
        
        Args:
            event_type: Event type to validate
            
        Returns:
            bool: True if the event type is valid
        """
        return event_type in self.event_types_registry
    