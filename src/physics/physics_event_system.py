import time
from typing import Optional, Dict, Any, Set


class PhysicsEventSystem:
    """
    Centralized management of physics event emissions.
    
    Provides consistent, typed event emission for all subsystems to ensure
    proper event propagation through the physics engine.
    """
    
    def __init__(self, event_system=None):
        """Initialize with optional event_system integration.
        
        Args:
            event_system: External event system to emit events through
        """
        self.event_system = event_system
        self.registered_events = set()
        
        # Register standard physics events if event system exists
        if self.event_system:
            self._register_standard_events()
    
    def _register_standard_events(self):
        """Register all standard physics-related events."""
        standard_events = [
            # System lifecycle events
            "physics_initialization_event", "physics_step_event",
            "simulation_run_start", "simulation_run_complete",
            "simulation_paused_event", "simulation_resumed_event", 
            "simulation_stopped_event", "simulation_reset_event",
            
            # State events
            "coherence_change_event", "entropy_increase_event", 
            "entropy_decrease_event", "state_creation_event",
            
            # Observer events
            "observation_event", "observer_consensus_event",
            
            # Field events
            "coherence_wave_event", "memory_resonance_event",
            
            # Quantum events
            "entanglement_creation_event", "entanglement_breaking_event",
            "teleportation_event", "collapse_event", "measurement_event",
            
            # Memory events
            "memory_strain_threshold_event", "emergency_defragmentation_event",
            "defragmentation_event",
            
            # Recursion events
            "recursive_boundary_event", "time_dilation_event",
            
            # Specialized events
            "resonance_event", "field_collapse_event",
            "quantum_annealing_event", "state_alignment_event",
            "strain_collapse_event"
        ]
        
        for event_name in standard_events:
            self.register_event(event_name)
    
    def register_event(self, event_name):
        """Register an event type with the system.
        
        Args:
            event_name: Name of event to register
        """
        if self.event_system and hasattr(self.event_system, "register_event_type"):
            self.event_system.register_event_type(event_name)
        self.registered_events.add(event_name)
    
    def emit(self, event_name, data, source=None):
        """Emit an event through the event system if available.
        
        Args:
            event_name: Name of the event to emit
            data: Event data dictionary
            source: Optional source identifier
            
        Returns:
            bool: True if event was emitted, False otherwise
        """
        if not self.event_system:
            return False
            
        # Register event if not already registered
        if event_name not in self.registered_events:
            self.register_event(event_name)
        
        # Add timestamp if not present
        if "time" not in data:
            data["time"] = time.time()
            
        # Add source if provided
        if source:
            data["source"] = source
            
        # Emit through event system
        self.event_system.emit(event_name, data)
        return True