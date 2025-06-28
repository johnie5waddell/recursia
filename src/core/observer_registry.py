from typing import Dict, List, Optional, Any, Set, Tuple
import logging
import time
from datetime import datetime

import numpy as np

from src.core.data_classes import ObserverDefinition
from src.core.observer_morph_factory import ObserverMorphFactory
from src.physics.coherence import CoherenceManager

logger = logging.getLogger(__name__)

class ObserverRegistry:
    """
    Registry for observers in the Recursia runtime.
    
    Manages the creation, tracking, and interaction of observer entities
    within the quantum simulation framework. Observers represent entities
    that can observe quantum states, potentially causing wave function collapse
    or other quantum effects based on the simulation parameters.
    """
    
    def __init__(self, coherence_manager=None):
        """Initialize the observer registry
        
        Args:
            coherence_manager (CoherenceManager, optional): Coherence manager for
                quantum coherence calculations
        """
        self.observers = {}  # Maps observer name to observer data
        self.observer_types = {}  # Maps observer type name to type definition
        self.observer_properties = {}  # Maps observer name to properties
        self.observers_by_target = {}  # Maps target state name to list of observer names
        self.observer_phases = {}  # Maps observer name to current phase
        self.observer_histories = {}  # Maps observer name to history of observations
        self.observer_relationships = {}  # Maps (observer1, observer2) to relationship data
        self.observer_creation_times = {}  # Maps observer name to creation timestamp
        self.observer_last_activity = {}  # Maps observer name to last activity timestamp
        
        # Connect to coherence manager if provided, or create a new one
        self.coherence_manager = coherence_manager or CoherenceManager()
        
        # Observer phase transitions
        self.phase_transitions = {
            "passive": ["active", "learning", "measuring"],
            "active": ["passive", "measuring", "analyzing", "entangled"],
            "measuring": ["active", "analyzing", "collapsed"],
            "analyzing": ["active", "passive", "learning"],
            "learning": ["active", "passive", "measuring"],
            "entangled": ["active", "collapsed", "measuring"],
            "collapsed": ["passive", "reset"]
        }
        
        # Register standard observer types
        self._register_standard_types()
        self.morph_factory = ObserverMorphFactory(self)
        
    def _update_observer_property(self, observer_name: str, property_name: str, value: Any) -> bool:
        """
        Internal method to update a single observer property with proper timestamp tracking
        and special handling for focus updates.
        
        Args:
            observer_name (str): Observer name
            property_name (str): Property name
            value: Property value
            
        Returns:
            bool: True if property was updated successfully
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer not found: {observer_name}")
            return False
        
        # Update the property
        self.observer_properties.setdefault(observer_name, {})[property_name] = value
        
        # Update observer version
        self.observers[observer_name]['version'] += 1
        
        # Update last activity time
        self.observer_last_activity[observer_name] = time.time()
        
        # Special handling for observer_focus property
        if property_name == 'observer_focus' and value:
            self._update_focus(observer_name, value)
        
        logger.debug(f"Updated property {property_name}={value} for observer {observer_name}")
        return True

    def _register_standard_types(self):
        """Register the standard observer types"""
        standard_types = [
            ("standard_observer", {
                "collapse_threshold": 0.5,
                "measurement_bias": 0.0,
                "observer_recursion_depth": 1
            }),
            ("quantum_observer", {
                "collapse_threshold": 0.8,
                "measurement_bias": 0.0,
                "entanglement_sensitivity": 0.7,
                "observer_recursion_depth": 2
            }),
            ("recursive_observer", {
                "collapse_threshold": 0.6,
                "observer_recursion_depth": 3,
                "self_observation_factor": 0.9
            }),
            ("collective_observer", {
                "collapse_threshold": 0.4,
                "observer_count": 1,
                "consensus_factor": 1.2,
                "observer_recursion_depth": 1
            }),
            ("conscious_observer", {
                "collapse_threshold": 0.9,
                "self_awareness": 0.8,
                "observer_recursion_depth": 3,
                "measurement_bias": 0.2
            }),
            ("subconscious_observer", {
                "collapse_threshold": 0.3,
                "self_awareness": 0.2,
                "observer_recursion_depth": 2,
                "measurement_bias": -0.1
            }),
            ("meta_observer", {
                "collapse_threshold": 0.7,
                "meta_level": 1,
                "observer_recursion_depth": 4,
                "self_awareness": 0.9
            }),
            ("nested_observer", {
                "collapse_threshold": 0.5,
                "nesting_depth": 2,
                "observer_recursion_depth": 3
            }),
            ("distributed_observer", {
                "collapse_threshold": 0.4,
                "node_count": 3,
                "synchronization_factor": 0.6,
                "observer_recursion_depth": 2
            })
        ]
        
        for name, properties in standard_types:
            self.register_observer_type(name, properties)
    
    def create_observer(self, name: str, observer_type: str = 'standard_observer', 
                       initial_properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new observer
        
        Args:
            name (str): Observer name
            observer_type (str): Observer type
            initial_properties (dict, optional): Initial observer properties
            
        Returns:
            bool: True if observer was created successfully
        """
        logger.info(f"=== CREATE OBSERVER - REGISTRY ===")
        logger.info(f"Creating observer: name='{name}', type='{observer_type}'")
        logger.info(f"Initial properties: {initial_properties}")
        
        if name in self.observers:
            logger.warning(f"Observer already exists: {name}")
            return False
        
        # Check if the observer type exists
        if observer_type not in self.observer_types:
            logger.warning(f"Unknown observer type: {observer_type}, defaulting to standard_observer")
            observer_type = 'standard_observer'
        
        # Get the timestamp
        timestamp = time.time()
        
        # Create observer
        self.observers[name] = {
            'type': observer_type,
            'created_at': timestamp,
            'version': 1
        }
        
        # Initialize properties with type defaults
        self.observer_properties[name] = self.observer_types[observer_type].copy()
        
        # Override with initial properties if provided, ensuring proper types
        if initial_properties:
            for prop_name, prop_value in initial_properties.items():
                # Special handling for focus to ensure it's numeric
                if prop_name == 'observer_focus' and prop_value is not None:
                    try:
                        # Convert to float if possible
                        self.observer_properties[name][prop_name] = float(prop_value)
                    except (ValueError, TypeError):
                        # If it's a string, store it as target_state and set numeric focus
                        if isinstance(prop_value, str):
                            self.observer_properties[name]['target_state'] = prop_value
                            self.observer_properties[name][prop_name] = 0.8  # High focus when targeting a state
                            logger.info(f"Observer {name} targeting state: {prop_value}")
                        else:
                            logger.warning(f"Invalid focus value for {name}: {prop_value}, using default 0.5")
                            self.observer_properties[name][prop_name] = 0.5
                # Handle phase separately - it should be a string
                elif prop_name == 'observer_phase':
                    # Strip any extra quotes from phase value
                    phase_value = str(prop_value)
                    if phase_value.startswith('"') and phase_value.endswith('"'):
                        phase_value = phase_value[1:-1]
                    self.observer_phases[name] = phase_value
                else:
                    self.observer_properties[name][prop_name] = prop_value
        
        # Ensure observer_focus exists and is numeric
        if 'observer_focus' not in self.observer_properties[name]:
            self.observer_properties[name]['observer_focus'] = 0.5
            
        # Set initial phase if not already set
        if name not in self.observer_phases:
            self.observer_phases[name] = "passive"
        
        # Initialize history
        self.observer_histories[name] = []
        
        # Set creation time
        self.observer_creation_times[name] = timestamp
        self.observer_last_activity[name] = timestamp
        
        logger.info(f"Created observer {name} of type {observer_type}")
        logger.info(f"Current observer count: {len(self.observers)}")
        logger.info(f"Observer properties: {self.observer_properties[name]}")
        return True
    
    def get_observer(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get an observer by name
        
        Args:
            name (str): Observer name
            
        Returns:
            dict: Observer information, or None if not found
        """
        if name not in self.observers:
            return None
            
        # Combine observer data and properties
        observer_data = self.observers[name].copy()
        observer_data['properties'] = self.observer_properties.get(name, {})
        observer_data['phase'] = self.observer_phases.get(name, "passive")
        observer_data['focus'] = self.get_property(name, 'observer_focus')
        
        return observer_data
    
    def get_observer_definition(self, name: str) -> Optional[ObserverDefinition]:
        """
        Get an observer as an ObserverDefinition
        
        Args:
            name (str): Observer name
            
        Returns:
            ObserverDefinition: Observer definition, or None if not found
        """
        if name not in self.observers:
            return None
            
        # Get the basic data
        observer_data = self.observers[name]
        properties = self.observer_properties.get(name, {})
        
        # Create the observer definition
        return ObserverDefinition(
            name=name,
            observer_type=observer_data['type'],
            focus=properties.get('observer_focus'),
            collapse_threshold=properties.get('collapse_threshold'),
            properties=properties,
            location=None  # Location not available in runtime
        )
    
    def set_property(self, observer_name: str, property_name: str, value: Any) -> bool:
        return self._update_observer_property(observer_name, property_name, value)
    
    def bulk_set_properties(self, observer_name: str, properties: Dict[str, Any]) -> bool:
        """
        Set multiple properties for an observer at once
        
        Args:
            observer_name (str): Observer name
            properties (dict): Dictionary of property name/value pairs
            
        Returns:
            bool: True if properties were set successfully
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer not found: {observer_name}")
            return False
        
        success = True
        for prop_name, prop_value in properties.items():
            if not self._update_observer_property(observer_name, prop_name, prop_value):
                success = False
        
        return success

    def _update_focus(self, observer_name: str, target: str) -> None:
        """
        Update observer focus tracking
        
        Args:
            observer_name (str): Observer name
            target (str): Target state name
        """
        # Get previous focus
        previous_focus = self.get_property(observer_name, 'observer_focus')
        
        # If different, update the tracking
        if previous_focus != target:
            # Remove from previous target
            if previous_focus in self.observers_by_target:
                if observer_name in self.observers_by_target[previous_focus]:
                    self.observers_by_target[previous_focus].remove(observer_name)
                    
                # Clean up empty lists
                if not self.observers_by_target[previous_focus]:
                    del self.observers_by_target[previous_focus]
            
            # Add to new target
            self.observers_by_target.setdefault(target, []).append(observer_name)
    
    def get_property(self, observer_name: str, property_name: str, default: Any = None) -> Any:
        """
        Get a property value for an observer
        
        Args:
            observer_name (str): Observer name
            property_name (str): Property name
            default: Default value if property not found
            
        Returns:
            Property value, or default if not found
        """
        if observer_name not in self.observers:
            return default
        
        return self.observer_properties.get(observer_name, {}).get(property_name, default)
    
    def _resolve_merged_property(self, prop_name: str, value1: Any, value2: Any, 
                                merge_type: str = 'average') -> Any:
        """
        Resolve a property value during observer merging based on merge strategy
        
        Args:
            prop_name (str): Property name
            value1: First property value 
            value2: Second property value
            merge_type (str): Merge strategy ('average', 'dominant', 'coherent')
            
        Returns:
            Resolved property value
        """
        # For focus properties, use special handling
        if prop_name == 'observer_focus':
            if value1 == value2 and value1 is not None:
                # Both focused on same target
                return value1
            elif value1 is not None and value2 is not None:
                # Different focuses - choose based on strategy
                return self._resolve_focus_conflict(value1, value2, merge_type)
            return value1 if value1 is not None else value2
            
        # Handle numeric properties
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            if merge_type == 'dominant':
                # Use maximum for collapse thresholds and recursion depths
                if 'threshold' in prop_name or 'recursion_depth' in prop_name or 'awareness' in prop_name:
                    return max(value1, value2)
                return value1  # Default to first value for dominant strategy
            elif merge_type == 'coherent':
                # For coherent merges, optimize specific properties differently
                if prop_name == 'collapse_threshold':
                    return min(value1, value2)  # Lower threshold = less collapse
                elif 'awareness' in prop_name or 'recursion_depth' in prop_name:
                    return max(value1, value2)  # Higher = more coherent
                else:
                    # Average other properties
                    return (value1 + value2) / 2
            else:  # 'average' or default
                return (value1 + value2) / 2
        
        # For non-numeric, prefer non-None value
        if value1 is not None:
            return value1
        return value2

    def _resolve_focus_conflict(self, focus1: str, focus2: str, merge_type: str) -> str:
        """
        Resolve conflicting focus targets based on merge strategy
        
        Args:
            focus1 (str): First focus target
            focus2 (str): Second focus target
            merge_type (str): Merge strategy
            
        Returns:
            str: Resolved focus target
        """
        # Get focus strengths
        focus_strength1 = self.get_property(focus1, 'focus_strength', 0.5)
        focus_strength2 = self.get_property(focus2, 'focus_strength', 0.5)
        
        if merge_type == 'dominant':
            return focus1 if focus_strength1 >= focus_strength2 else focus2
        elif merge_type == 'coherent':
            # Choose focus with highest coherence potential (more observers)
            if len(self.observers_by_target.get(focus1, [])) >= len(self.observers_by_target.get(focus2, [])):
                return focus1
            else:
                return focus2
        else:  # 'average' or default
            # Choose randomly weighted by focus strength
            total_strength = focus_strength1 + focus_strength2
            if total_strength == 0:
                return focus1  # Default to first if both have zero strength
            if np.random.random() < focus_strength1 / total_strength:
                return focus1
            else:
                return focus2
            
    def delete_observer(self, name: str) -> bool:
        """
        Delete an observer
        
        Args:
            name (str): Observer name
            
        Returns:
            bool: True if observer was deleted successfully
        """
        if name not in self.observers:
            logger.warning(f"Cannot delete non-existent observer: {name}")
            return False
        
        # Remove from focus tracking
        focus = self.get_property(name, 'observer_focus')
        if focus and focus in self.observers_by_target:
            if name in self.observers_by_target[focus]:
                self.observers_by_target[focus].remove(name)
                
                # Clean up empty lists
                if not self.observers_by_target[focus]:
                    del self.observers_by_target[focus]
        
        # Remove observer data
        del self.observers[name]
        
        # Remove properties
        if name in self.observer_properties:
            del self.observer_properties[name]
        
        # Remove phase
        if name in self.observer_phases:
            del self.observer_phases[name]
        
        # Remove history
        if name in self.observer_histories:
            del self.observer_histories[name]
        
        # Remove timestamps
        if name in self.observer_creation_times:
            del self.observer_creation_times[name]
        if name in self.observer_last_activity:
            del self.observer_last_activity[name]
        
        # Remove from relationships
        self._clean_relationships(name)
        
        logger.info(f"Deleted observer {name}")
        return True
    
    def _clean_relationships(self, observer_name: str) -> None:
        """
        Clean up relationships for a deleted observer
        
        Args:
            observer_name (str): Observer name being deleted
        """
        # Find all relationships involving this observer
        keys_to_remove = []
        for key in self.observer_relationships:
            if observer_name in key:
                keys_to_remove.append(key)
        
        # Remove relationships
        for key in keys_to_remove:
            del self.observer_relationships[key]
    
    def register_observer_type(self, name: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register an observer type
        
        Args:
            name (str): Type name
            properties (dict, optional): Type properties
            
        Returns:
            bool: True if type was registered successfully
        """
        if name in self.observer_types:
            logger.warning(f"Observer type already exists: {name}")
            return False
        
        self.observer_types[name] = properties or {}
        logger.info(f"Registered observer type: {name}")
        return True
    
    def get_observers_by_type(self, observer_type: str) -> List[str]:
        """
        Get all observers of a specific type
        
        Args:
            observer_type (str): Observer type
            
        Returns:
            list: List of observer names
        """
        return [name for name, observer in self.observers.items() 
                if observer.get('type') == observer_type]
    
    def get_observers_for_target(self, target_name: str) -> List[str]:
        """
        Get all observers for a specific target
        
        Args:
            target_name (str): Target name
            
        Returns:
            list: List of observer names
        """
        return self.observers_by_target.get(target_name, [])
    
    def set_observer_phase(self, observer_name: str, phase: str) -> bool:
        return self._transition_observer_phase(observer_name, phase)
    
    
    
    def get_observer_phase(self, observer_name: str) -> Optional[str]:
        """
        Get the current phase of an observer
        
        Args:
            observer_name (str): Observer name
            
        Returns:
            str: Current phase, or None if observer not found
        """
        if observer_name not in self.observers:
            return None
        
        return self.observer_phases.get(observer_name, "passive")
    
    def record_observation(self, observer_name: str, target_name: str, 
                          observation_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Record an observation by an observer
        
        Args:
            observer_name (str): Observer name
            target_name (str): Target state name
            observation_data (dict, optional): Additional observation data
            
        Returns:
            bool: True if observation was recorded successfully
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer not found: {observer_name}")
            return False
        
        # Record in history
        timestamp = time.time()
        observation_record = {
            'type': 'observation',
            'target': target_name,
            'timestamp': timestamp,
            'data': observation_data or {}
        }
        
        self.observer_histories.setdefault(observer_name, []).append(observation_record)
        
        # Update last activity time
        self.observer_last_activity[observer_name] = timestamp
        
        # Set focus to the observed target
        self.set_property(observer_name, 'observer_focus', target_name)
        
        # Move to measuring phase if in passive phase
        if self.observer_phases.get(observer_name) == "passive":
            self.set_observer_phase(observer_name, "measuring")
        
        logger.debug(f"Observer {observer_name} observed {target_name}")
        return True
    
    def get_observation_history(self, observer_name: str, 
                               limit: Optional[int] = None,
                               target_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get observation history for an observer
        
        Args:
            observer_name (str): Observer name
            limit (int, optional): Maximum number of history entries to return
            target_filter (str, optional): Filter to a specific target
            
        Returns:
            list: List of observation records
        """
        if observer_name not in self.observers:
            return []
        
        history = self.observer_histories.get(observer_name, [])
        
        # Apply target filter if specified
        if target_filter:
            history = [entry for entry in history 
                      if entry.get('type') == 'observation' and entry.get('target') == target_filter]
        
        # Apply limit if specified
        if limit is not None and limit > 0:
            history = history[-limit:]
        
        return history
    
    def establish_relationship(self, observer1: str, observer2: str,
                             relationship_type: str, strength: float = 1.0,
                             properties: Optional[Dict[str, Any]] = None) -> bool:
        return self._update_relationship(observer1, observer2, relationship_type, strength, properties)
    

    def get_relationship(self, observer1: str, observer2: str) -> Optional[Dict[str, Any]]:
        """
        Get the relationship between two observers
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            
        Returns:
            dict: Relationship data, or None if no relationship exists
        """
        # Sort observers for consistent key
        key = tuple(sorted([observer1, observer2]))
        
        return self.observer_relationships.get(key)
    
    def get_related_observers(self, observer_name: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Get all observers related to the specified observer
        
        Args:
            observer_name (str): Observer name
            
        Returns:
            list: List of (related_observer_name, relationship_data) tuples
        """
        related = []
        
        for key, relationship in self.observer_relationships.items():
            if observer_name in key:
                # Get the other observer
                other_observer = key[0] if key[1] == observer_name else key[1]
                related.append((other_observer, relationship))
        
        return related
    
    def _calculate_activity_level(self, observation_count: int, phase_changes: int, idle_time: float) -> float:
        """
        Calculate observer activity level
        
        Args:
            observation_count (int): Number of observations
            phase_changes (int): Number of phase changes
            idle_time (float): Time since last activity in seconds
            
        Returns:
            float: Activity level from 0.0 to 1.0
        """
        # Base activity from observations and phase changes
        base_activity = 0.2 + 0.4 * min(1.0, observation_count / 10) + 0.4 * min(1.0, phase_changes / 5)
        
        # Apply idle time decay
        idle_factor = np.exp(-idle_time / 3600)  # Exponential decay based on hours of inactivity
        
        # Combine factors
        activity_level = base_activity * idle_factor
        
        return min(1.0, max(0.0, activity_level))
    
    def get_all_observers(self) -> List[Dict[str, Any]]:
        """
        Get all registered observers with their properties
        
        Returns:
            list: List of observer dictionaries with properties
        """
        all_observers = []
        
        for name, observer_data in self.observers.items():
            # Build complete observer info
            focus_value = self.get_property(name, 'observer_focus', 0.5)
            # Ensure focus is numeric
            if not isinstance(focus_value, (int, float)):
                focus_value = 0.5
            observer_info = {
                'name': name,
                'type': observer_data.get('type', 'standard_observer'),
                'focus': focus_value,  # Ensure numeric focus
                'target_state': self.get_property(name, 'target_state', None),  # Include target state if present
                'phase': self.observer_phases.get(name, 'passive'),
                'collapse_threshold': self.get_property(name, 'observer_collapse_threshold', 0.5),
                'self_awareness': self.get_property(name, 'observer_self_awareness', 0.0),
                'version': observer_data.get('version', 0),
                'created_at': self.observer_creation_times.get(name),
                'last_activity': self.observer_last_activity.get(name),
                'properties': self.observer_properties.get(name, {})
            }
            all_observers.append(observer_info)
        
        return all_observers


    def find_observers(self, criteria: Dict[str, Any]) -> List[str]:
        """
        Find observers matching specific criteria
        
        Args:
            criteria (dict): Search criteria (property name/value pairs)
            
        Returns:
            list: List of matching observer names
        """
        matching = []
        
        for name, observer in self.observers.items():
            # Check if all criteria match
            matches = True
            
            for key, value in criteria.items():
                # Check base observer properties
                if key in observer:
                    if observer[key] != value:
                        matches = False
                        break
                # Check observer-specific properties
                elif key in self.observer_properties.get(name, {}):
                    if self.observer_properties[name][key] != value:
                        matches = False
                        break
                # Check phase
                elif key == 'phase':
                    if self.observer_phases.get(name) != value:
                        matches = False
                        break
                else:
                    # Property not found
                    matches = False
                    break
            
            if matches:
                matching.append(name)
        
        return matching
    
    def update_observer_network(self, time_step: float = 1.0) -> None:
        """
        Update the entire observer network for a time step
        
        Args:
            time_step (float): Time step size
        """
        # Process relationships and influences between observers
        self._update_relationships(time_step)
        
        # Process automatic phase transitions
        self._update_phases(time_step)
        
        # Clean up stale observations
        self._clean_observations()
        
        logger.debug(f"Updated observer network for time step {time_step}")
    
    def _calculate_interaction_strength(self, observer1: str, observer2: str, 
                                    base_strength: float, interaction_type: str = 'standard') -> float:
        """
        Calculate the effective interaction strength between two observers
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            base_strength (float): Base interaction strength value
            interaction_type (str): Type of interaction
            
        Returns:
            float: Effective interaction strength
        """
        # Get observer types
        type1 = self.observers[observer1]['type']
        type2 = self.observers[observer2]['type']
        
        # Calculate interaction strength based on observer types
        type_factor = 1.0
        if type1 == type2:
            # Same type observers interact more strongly
            type_factor = 1.2
        elif 'quantum' in type1 and 'quantum' in type2:
            # Quantum observers interact more strongly
            type_factor = 1.3
        elif 'recursive' in type1 and 'recursive' in type2:
            # Recursive observers interact more strongly
            type_factor = 1.4
        elif 'meta' in type1 or 'meta' in type2:
            # Meta observers have stronger influence
            type_factor = 1.5
        
        # Get phases for phase-based modifiers
        phase1 = self.observer_phases.get(observer1, "passive")
        phase2 = self.observer_phases.get(observer2, "passive")
        
        # Calculate phase factor
        phase_factor = 1.0
        phase_weights = {
            "passive": 0.7,
            "active": 1.2,
            "measuring": 1.5, 
            "analyzing": 1.3,
            "learning": 1.1,
            "entangled": 1.6,
            "collapsed": 0.5
        }
        phase_factor = phase_weights.get(phase1, 1.0) * phase_weights.get(phase2, 1.0)
        phase_factor = phase_factor ** 0.5  # Square root to moderate the effect
        
        # Adjust based on interaction type
        interaction_factor = 1.0
        if interaction_type == 'quantum':
            interaction_factor = 1.5
            # Quantum interactions affected by self-awareness and entanglement
            props1 = self.observer_properties.get(observer1, {})
            props2 = self.observer_properties.get(observer2, {})
            self_awareness1 = props1.get('self_awareness', 0.5)
            self_awareness2 = props2.get('self_awareness', 0.5)
            entangle_sens1 = props1.get('entanglement_sensitivity', 0.5)
            entangle_sens2 = props2.get('entanglement_sensitivity', 0.5)
            
            quantum_factor = (self_awareness1 + self_awareness2 + entangle_sens1 + entangle_sens2) / 4
            interaction_factor *= (0.5 + quantum_factor)
        elif interaction_type == 'recursive':
            interaction_factor = 1.3
            # Recursive interactions affected by recursion depth
            props1 = self.observer_properties.get(observer1, {})
            props2 = self.observer_properties.get(observer2, {})
            depth1 = props1.get('observer_recursion_depth', 1)
            depth2 = props2.get('observer_recursion_depth', 1)
            
            recursive_factor = min(3, max(depth1, depth2)) / 3  # Normalize to [0, 1]
            interaction_factor *= (0.7 + recursive_factor)
        elif interaction_type == 'entangled':
            interaction_factor = 1.8  # Entangled interactions are strongest
        
        # Calculate final interaction strength, ensuring it stays within bounds
        strength = base_strength * type_factor * phase_factor * interaction_factor
        return max(0.0, min(1.0, strength))

    def _transition_observer_phase(self, observer_name: str, new_phase: str) -> bool:
        """
        Handle observer phase transitions with proper validation and history
        
        Args:
            observer_name (str): Observer name
            new_phase (str): Target phase to transition to
            
        Returns:
            bool: True if transition was successful
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer not found: {observer_name}")
            return False
        
        current_phase = self.observer_phases.get(observer_name, "passive")
        
        # No change needed
        if current_phase == new_phase:
            return True
            
        # Check if transition is valid
        if new_phase not in self.phase_transitions.get(current_phase, []):
            logger.warning(f"Invalid phase transition for observer {observer_name}: {current_phase} -> {new_phase}")
            return False
        
        # Set the new phase
        self.observer_phases[observer_name] = new_phase
        
        # Update last activity time
        self.observer_last_activity[observer_name] = time.time()
        
        # Record in history
        timestamp = time.time()
        self.observer_histories.setdefault(observer_name, []).append({
            'type': 'phase_change',
            'from': current_phase,
            'to': new_phase,
            'timestamp': timestamp
        })
        
        logger.debug(f"Observer {observer_name} phase changed: {current_phase} -> {new_phase}")
        return True

    def _update_relationship(self, observer1: str, observer2: str, 
                            rel_type: str = None, rel_strength: float = None, 
                            rel_props: Dict[str, Any] = None) -> bool:
        """
        Update or create a relationship between two observers
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            rel_type (str, optional): Relationship type (only set if creating or updating)
            rel_strength (float, optional): Relationship strength (only set if creating or updating)
            rel_props (Dict, optional): Relationship properties (only set if creating or updating)
            
        Returns:
            bool: True if relationship was updated successfully
        """
        if observer1 not in self.observers or observer2 not in self.observers:
            logger.warning(f"Cannot update relationship: observer not found")
            return False
        
        if observer1 == observer2:
            logger.warning(f"Cannot establish relationship with self: {observer1}")
            return False
        
        # Sort observers for consistent key
        key = tuple(sorted([observer1, observer2]))
        
        # Update existing or create new relationship
        if key in self.observer_relationships:
            relationship = self.observer_relationships[key]
            if rel_type is not None:
                relationship['type'] = rel_type
            if rel_strength is not None:
                relationship['strength'] = max(0.0, min(1.0, rel_strength))
            if rel_props is not None:
                relationship['properties'].update(rel_props)
        else:
            # Create new relationship
            self.observer_relationships[key] = {
                'type': rel_type or 'default',
                'strength': max(0.0, min(1.0, rel_strength or 0.5)),
                'established_at': time.time(),
                'properties': rel_props or {}
            }
        
        # Update last activity time for both observers
        timestamp = time.time()
        self.observer_last_activity[observer1] = timestamp
        self.observer_last_activity[observer2] = timestamp
        
        return True

    def _apply_observer_interaction(self, observer1: str, observer2: str, 
                                  strength: float) -> None:
        """
        Apply interaction effects between two observers
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            strength (float): Interaction strength
        """
        # Skip if interaction is negligible
        if strength < 0.001:
            return
        
        # Get phases
        phase1 = self.observer_phases.get(observer1, "passive")
        phase2 = self.observer_phases.get(observer2, "passive")
        
        # Get focuses
        focus1 = self.get_property(observer1, 'observer_focus')
        focus2 = self.get_property(observer2, 'observer_focus')
        
        # If both observers are focused on the same target, reinforce focus
        if focus1 and focus1 == focus2:
            # Increase focus strength
            focus_strength1 = self.get_property(observer1, 'focus_strength', 0.5)
            focus_strength2 = self.get_property(observer2, 'focus_strength', 0.5)
            
            new_strength1 = min(1.0, focus_strength1 + strength)
            new_strength2 = min(1.0, focus_strength2 + strength)
            
            self.set_property(observer1, 'focus_strength', new_strength1)
            self.set_property(observer2, 'focus_strength', new_strength2)
        
        # Phase interaction effects
        if phase1 == "active" and phase2 == "passive":
            # Active can activate passive
            if np.random.random() < strength * 0.5:
                self.set_observer_phase(observer2, "active")
        elif phase1 == "measuring" and phase2 == "measuring" and focus1 == focus2:
            # Measuring the same target can lead to entanglement
            if np.random.random() < strength * 0.3:
                self.set_observer_phase(observer1, "entangled")
                self.set_observer_phase(observer2, "entangled")
        
        # Influence observer properties
        self._influence_observer_properties(observer1, observer2, strength)
    
    def _influence_observer_properties(self, observer1: str, observer2: str, 
                                     strength: float) -> None:
        """
        Have one observer influence another's properties
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            strength (float): Influence strength
        """
        # Properties that can be influenced
        influence_properties = [
            'collapse_threshold',
            'measurement_bias',
            'entanglement_sensitivity'
        ]
        
        # Get properties for both observers
        props1 = self.observer_properties.get(observer1, {})
        props2 = self.observer_properties.get(observer2, {})
        
        # Apply influences
        for prop in influence_properties:
            if prop in props1 and prop in props2:
                val1 = props1[prop]
                val2 = props2[prop]
                
                # Calculate influence (move slightly toward other observer's value)
                delta1 = (val2 - val1) * strength * 0.1
                delta2 = (val1 - val2) * strength * 0.1
                
                # Apply influence
                new_val1 = val1 + delta1
                new_val2 = val2 + delta2
                
                # Update properties
                self.set_property(observer1, prop, new_val1)
                self.set_property(observer2, prop, new_val2)
    
    
    def _update_phases(self, time_step: float) -> None:
        """
        Update observer phases based on current states and time
        
        Args:
            time_step (float): Time step size
        """
        for name, phase in list(self.observer_phases.items()):
            # Skip if observer no longer exists
            if name not in self.observers:
                continue
                
            # Skip if observer was recently active
            last_activity = self.observer_last_activity.get(name, 0)
            if time.time() - last_activity < 5.0:
                continue
            
            # Get type-specific properties
            observer_type = self.observers[name]['type']
            properties = self.observer_properties.get(name, {})
            
            # Calculate phase transition probabilities based on current phase
            if phase == "active":
                # Active observers gradually become passive
                passive_prob = 0.1 * time_step
                if np.random.random() < passive_prob:
                    self.set_observer_phase(name, "passive")
            
            elif phase == "measuring":
                # Measuring observers become analyzing
                analyzing_prob = 0.2 * time_step
                if np.random.random() < analyzing_prob:
                    self.set_observer_phase(name, "analyzing")
            
            elif phase == "analyzing":
                # Analyzing observers can become active or learning
                if np.random.random() < 0.5:
                    self.set_observer_phase(name, "active")
                else:
                    self.set_observer_phase(name, "learning")
            
            elif phase == "entangled":
                # Entangled observers gradually return to active
                active_prob = 0.05 * time_step
                if np.random.random() < active_prob:
                    self.set_observer_phase(name, "active")
    
    def _clean_observations(self) -> None:
        """Clean up stale observations and history"""
        max_history_size = 100  # Maximum history size per observer
        
        for name, history in self.observer_histories.items():
            # Skip if observer no longer exists
            if name not in self.observers:
                continue
                
            # Trim history if too large
            if len(history) > max_history_size:
                self.observer_histories[name] = history[-max_history_size:]
    
    def merge_observers(self, observer1: str, observer2: str, new_name: str,
                       merge_type: str = 'average') -> bool:
        """
        Merge two observers into a new one
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            new_name (str): Name for the merged observer
            merge_type (str): Merge strategy ('average', 'dominant', 'coherent')
            
        Returns:
            bool: True if merge was successful
        """
        if observer1 not in self.observers or observer2 not in self.observers:
            logger.warning(f"Cannot merge: observer not found")
            return False
        
        if new_name in self.observers:
            logger.warning(f"Merge target observer already exists: {new_name}")
            return False
        
        # Get observer data
        data1 = self.observers[observer1]
        data2 = self.observers[observer2]
        props1 = self.observer_properties.get(observer1, {})
        props2 = self.observer_properties.get(observer2, {})
        
        # Determine merged type based on strategy
        if merge_type == 'dominant':
            # Use type from observer with higher complexity or focus strength
            complexity1 = props1.get('observer_recursion_depth', 1) * props1.get('self_awareness', 0.5)
            complexity2 = props2.get('observer_recursion_depth', 1) * props2.get('self_awareness', 0.5)
            
            merged_type = data1['type'] if complexity1 >= complexity2 else data2['type']
        elif merge_type == 'coherent':
            # Use type that maintains higher coherence
            if 'quantum' in data1['type'] or 'recursive' in data1['type']:
                merged_type = data1['type']
            elif 'quantum' in data2['type'] or 'recursive' in data2['type']:
                merged_type = data2['type']
            else:
                # Default to more complex type
                merged_type = data1['type'] if 'meta' in data1['type'] or 'conscious' in data1['type'] else data2['type']
        else:  # 'average' or default
            # If types are same, keep it; otherwise create a hybrid type
            if data1['type'] == data2['type']:
                merged_type = data1['type']
            elif 'quantum' in data1['type'] and 'recursive' in data2['type']:
                merged_type = 'quantum_recursive_observer'
            elif 'recursive' in data1['type'] and 'quantum' in data2['type']:
                merged_type = 'quantum_recursive_observer'
            elif 'meta' in data1['type'] or 'meta' in data2['type']:
                merged_type = 'meta_observer'
            else:
                # Default to the more specialized type
                standard_types = {'standard_observer', 'basic_observer'}
                merged_type = data2['type'] if data1['type'] in standard_types else data1['type']
        
        # Register the new merged observer type if it doesn't exist
        if merged_type not in self.observer_types and merged_type not in {'quantum_recursive_observer'}:
            # Register as a new type with merged properties
            base_props = {}
            if merged_type == 'quantum_recursive_observer':
                base_props = {
                    'collapse_threshold': 0.7,
                    'observer_recursion_depth': 3,
                    'self_awareness': 0.7,
                    'entanglement_sensitivity': 0.8
                }
            self.register_observer_type(merged_type, base_props)
        
        # Merge properties
        merged_properties = {}
        
        # Combine all properties from both observers
        all_prop_keys = set(props1.keys()) | set(props2.keys())
        
        for key in all_prop_keys:
            if key in props1 and key in props2:
                # Both have this property - merge based on strategy
                if merge_type == 'dominant':
                    # Use property from the more complex observer
                    complexity1 = props1.get('observer_recursion_depth', 1) * props1.get('self_awareness', 0.5)
                    complexity2 = props2.get('observer_recursion_depth', 1) * props2.get('self_awareness', 0.5)
                    merged_properties[key] = props1[key] if complexity1 >= complexity2 else props2[key]
                elif merge_type == 'coherent':
                    # Use property that maintains higher coherence
                    if key == 'collapse_threshold':
                        merged_properties[key] = min(props1[key], props2[key])  # Lower threshold = less collapse
                    elif key == 'self_awareness' or key == 'observer_recursion_depth':
                        merged_properties[key] = max(props1[key], props2[key])  # Higher = more coherent
                    else:
                        # Average other properties
                        merged_properties[key] = (props1[key] + props2[key]) / 2
                else:  # 'average' or default
                    # Average numeric properties, use first for non-numeric
                    if isinstance(props1[key], (int, float)) and isinstance(props2[key], (int, float)):
                        merged_properties[key] = (props1[key] + props2[key]) / 2
                    else:
                        # For non-numeric, prefer non-None value
                        if props1[key] is not None:
                            merged_properties[key] = props1[key]
                        else:
                            merged_properties[key] = props2[key]
            elif key in props1:
                # Only in props1
                merged_properties[key] = props1[key]
            else:
                # Only in props2
                merged_properties[key] = props2[key]
        
        # Special handling for focus
        focus1 = props1.get('observer_focus')
        focus2 = props2.get('observer_focus')
        
        if focus1 == focus2 and focus1 is not None:
            # Both focused on same target
            merged_properties['observer_focus'] = focus1
        elif focus1 is not None and focus2 is not None:
            # Different focuses - choose based on focus strength or strategy
            focus_strength1 = props1.get('focus_strength', 0.5)
            focus_strength2 = props2.get('focus_strength', 0.5)
            
            if merge_type == 'dominant':
                merged_properties['observer_focus'] = focus1 if focus_strength1 >= focus_strength2 else focus2
            elif merge_type == 'coherent':
                # Choose focus with highest coherence potential
                # For simplicity, use the focus with more observers
                if len(self.observers_by_target.get(focus1, [])) >= len(self.observers_by_target.get(focus2, [])):
                    merged_properties['observer_focus'] = focus1
                else:
                    merged_properties['observer_focus'] = focus2
            else:  # 'average' or default
                # Choose randomly weighted by focus strength
                total_strength = focus_strength1 + focus_strength2
                if np.random.random() < focus_strength1 / total_strength:
                    merged_properties['observer_focus'] = focus1
                else:
                    merged_properties['observer_focus'] = focus2
        elif focus1 is not None:
            merged_properties['observer_focus'] = focus1
        elif focus2 is not None:
            merged_properties['observer_focus'] = focus2
        
        # Create the merged observer
        self.create_observer(new_name, merged_type, merged_properties)
        
        # Set phase based on source observers
        phase1 = self.observer_phases.get(observer1, "passive")
        phase2 = self.observer_phases.get(observer2, "passive")
        
        if phase1 == phase2:
            merged_phase = phase1
        elif 'active' in (phase1, phase2):
            merged_phase = 'active'
        elif 'measuring' in (phase1, phase2):
            merged_phase = 'measuring'
        else:
            # Default to most active phase
            phase_priority = {
                "entangled": 5,
                "measuring": 4,
                "analyzing": 3,
                "learning": 2,
                "active": 1,
                "passive": 0
            }
            merged_phase = phase1 if phase_priority.get(phase1, 0) >= phase_priority.get(phase2, 0) else phase2
        
        self.set_observer_phase(new_name, merged_phase)
        
        # Merge relationships - transfer all relationships from both observers to the merged observer
        for key, relationship in list(self.observer_relationships.items()):
            if observer1 in key or observer2 in key:
                # Get the other observer
                other_observer = None
                if key[0] == observer1 or key[0] == observer2:
                    other_observer = key[1]
                else:
                    other_observer = key[0]
                
                # Skip if the other observer is the one we're merging with
                if other_observer == observer1 or other_observer == observer2:
                    continue
                
                # Skip if the other observer doesn't exist
                if other_observer not in self.observers:
                    continue
                
                # Create relationship with the merged observer
                rel_type = relationship['type']
                rel_strength = relationship['strength']
                rel_props = relationship['properties']
                
                self.establish_relationship(new_name, other_observer, rel_type, rel_strength, rel_props)
        
        # Copy observer history
        history1 = self.observer_histories.get(observer1, [])
        history2 = self.observer_histories.get(observer2, [])
        
        # Merge and sort history by timestamp
        merged_history = history1 + history2
        merged_history.sort(key=lambda x: x.get('timestamp', 0))
        
        # Add merge event to history
        merger_record = {
            'type': 'merge',
            'merged_from': [observer1, observer2],
            'timestamp': time.time()
        }
        merged_history.append(merger_record)
        
        # Set merged history
        self.observer_histories[new_name] = merged_history
        
        # Delete the original observers
        self.delete_observer(observer1)
        self.delete_observer(observer2)
        
        logger.info(f"Merged observers {observer1} and {observer2} into {new_name}")
        return True
    
    def split_observer(self, observer_name: str, new_name1: str, new_name2: str,
                     split_ratio: float = 0.5, split_method: str = 'divide') -> bool:
        return self.morph_factory.split(observer_name, new_name1, new_name2, split_ratio, split_method)
    
    def get_observer_stats(self, observer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about an observer or the entire registry
        
        Args:
            observer_name (str, optional): Observer name, or None for global stats
            
        Returns:
            dict: Observer or registry statistics
        """
        if observer_name is not None:
            # Get stats for a specific observer
            if observer_name not in self.observers:
                return {}
            
            observer_data = self.observers[observer_name]
            properties = self.observer_properties.get(observer_name, {})
            phase = self.observer_phases.get(observer_name, "passive")
            history = self.observer_histories.get(observer_name, [])
            
            # Calculate activity metrics
            created_at = self.observer_creation_times.get(observer_name, 0)
            last_activity = self.observer_last_activity.get(observer_name, 0)
            age = time.time() - created_at
            
            # Count observation types
            observation_count = 0
            phase_changes = 0
            targets = set()
            
            for entry in history:
                entry_type = entry.get('type')
                if entry_type == 'observation':
                    observation_count += 1
                    target = entry.get('target')
                    if target:
                        targets.add(target)
                elif entry_type == 'phase_change':
                    phase_changes += 1
            
            # Get relationships
            relationships = []
            for key, rel in self.observer_relationships.items():
                if observer_name in key:
                    other = key[0] if key[1] == observer_name else key[1]
                    relationships.append({
                        'observer': other,
                        'type': rel['type'],
                        'strength': rel['strength']
                    })
            
            return {
                'name': observer_name,
                'type': observer_data['type'],
                'phase': phase,
                'age': age,
                'created_at': created_at,
                'last_activity': last_activity,
                'idle_time': time.time() - last_activity,
                'version': observer_data.get('version', 1),
                'properties': properties,
                'focus': properties.get('observer_focus'),
                'history_entries': len(history),
                'observation_count': observation_count,
                'phase_changes': phase_changes,
                'observed_targets': list(targets),
                'relationships': relationships,
                'activity_level': self._calculate_activity_level(observation_count, phase_changes, time.time() - last_activity)
            }
        else:
            # Get global registry stats
            observer_count = len(self.observers)
            type_counts = {}
            phase_counts = {}
            focus_counts = {}
            
            # Count by type, phase, and focus
            for name, data in self.observers.items():
                # Count types
                obs_type = data['type']
                type_counts[obs_type] = type_counts.get(obs_type, 0) + 1
                
                # Count phases
                phase = self.observer_phases.get(name, "passive")
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
                
                # Count focuses
                focus = self.get_property(name, 'observer_focus')
                if focus:
                    focus_counts[focus] = focus_counts.get(focus, 0) + 1
            
            # Count relationships
            relationship_count = len(self.observer_relationships)
            relationship_types = {}
            
            for rel in self.observer_relationships.values():
                rel_type = rel['type']
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            return {
                'observer_count': observer_count,
                'observer_types': type_counts,
                'observer_phases': phase_counts,
                'focus_targets': focus_counts,
                'relationship_count': relationship_count,
                'relationship_types': relationship_types,
                'active_observations': len(self.observers_by_target),
                'registered_observer_types': len(self.observer_types),
                'most_active_observers': self._get_most_active_observers(5)
            }
    
    def _get_most_active_observers(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get the most active observers
        
        Args:
            limit (int): Maximum number of observers to return
            
        Returns:
            list: List of observer activity stats
        """
        observer_activity = []
        
        for name in self.observers:
            history = self.observer_histories.get(name, [])
            last_activity = self.observer_last_activity.get(name, 0)
            
            # Calculate activity score based on history and recency
            activity_score = len(history) * 0.5
            
            # Recent activity is weighted more
            time_factor = np.exp(-0.1 * (time.time() - last_activity) / 3600)  # Decay factor
            activity_score *= (0.2 + 0.8 * time_factor)  # Base score plus recency bonus
            
            # Add phase factor - active phases get bonus
            phase = self.observer_phases.get(name, "passive")
            phase_bonus = {
                "active": 1.2,
                "measuring": 1.5,
                "analyzing": 1.3,
                "entangled": 1.4,
                "learning": 1.1,
                "passive": 0.8,
                "collapsed": 0.7
            }
            
            activity_score *= phase_bonus.get(phase, 1.0)
            
            observer_activity.append({
                'name': name,
                'score': activity_score,
                'history_size': len(history),
                'last_activity': last_activity,
                'phase': phase
            })
        
        # Sort by activity score
        observer_activity.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top N
        return observer_activity[:limit]
    
    def register_observer(self, name: str, observer_type: Any = 'standard_observer', 
                        initial_properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Register an observer with flexible parameter handling.
        
        Args:
            name (str): Observer name
            observer_type: Either a string type name or an observer object
            initial_properties (dict, optional): Initial observer properties
            
        Returns:
            bool: True if observer was registered successfully
        """
        # Handle case where observer_type is actually an Observer object
        if hasattr(observer_type, 'focus') and hasattr(observer_type, 'phase'):
            # It's an Observer object, extract properties
            logger.info(f"Registering observer '{name}' from Observer object")
            properties = {
                'observer_focus': observer_type.focus,
                'observer_phase': observer_type.phase,
                'observer_collapse_threshold': observer_type.collapse_threshold
            }
            # Use default type
            return self.create_observer(name, 'standard_observer', properties)
        else:
            # Regular string type name
            return self.create_observer(name, observer_type, initial_properties)

    def simulate_observer_interaction(self, observer1: str, observer2: str, 
                                    interaction_time: float = 1.0,
                                    interaction_type: str = 'standard') -> Dict[str, Any]:
        # Use the interaction strength calculator
        base_strength = 0.5 * interaction_time
        interaction_strength = self._calculate_interaction_strength(
            observer1, observer2, base_strength, interaction_type
        )

    def reset_observer(self, observer_name: str) -> bool:
        """
        Reset an observer to its initial state
        
        Args:
            observer_name (str): Observer name
            
        Returns:
            bool: True if reset was successful
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer not found: {observer_name}")
            return False
        
        # Get observer type
        observer_type = self.observers[observer_name]['type']
        
        # Reset properties to type defaults
        default_properties = self.observer_types.get(observer_type, {}).copy()
        self.observer_properties[observer_name] = default_properties
        
        # Reset phase
        self.observer_phases[observer_name] = "passive"
        
        # Add reset event to history
        reset_record = {
            'type': 'reset',
            'timestamp': time.time()
        }
        
        self.observer_histories.setdefault(observer_name, []).append(reset_record)
        
        # Update last activity time
        self.observer_last_activity[observer_name] = time.time()
        
        logger.info(f"Reset observer {observer_name}")
        return True
    
    def clone_observer(self, source_name: str, new_name: str, mutation_factor: float = 0.1) -> bool:
        return self.morph_factory.clone(source_name, new_name, mutation_factor)
    
    def export_observer(self, observer_name: str) -> Dict[str, Any]:
        """
        Export an observer as a serializable dictionary
        
        Args:
            observer_name (str): Observer name
            
        Returns:
            dict: Serialized observer data
        """
        if observer_name not in self.observers:
            logger.warning(f"Observer not found: {observer_name}")
            return {}
        
        # Get observer data
        observer_data = self.observers[observer_name].copy()
        properties = self.observer_properties.get(observer_name, {}).copy()
        phase = self.observer_phases.get(observer_name, "passive")
        history = self.observer_histories.get(observer_name, [])
        
        # Get relationships
        relationships = []
        for key, rel in self.observer_relationships.items():
            if observer_name in key:
                other = key[0] if key[1] == observer_name else key[1]
                relationships.append({
                    'observer': other,
                    'type': rel['type'],
                    'strength': rel['strength'],
                    'properties': rel.get('properties', {})
                })
        
        # Create export data
        export_data = {
            'name': observer_name,
            'type': observer_data['type'],
            'properties': properties,
            'phase': phase,
            'history': history,
            'relationships': relationships,
            'created_at': self.observer_creation_times.get(observer_name, 0),
            'last_activity': self.observer_last_activity.get(observer_name, 0),
            'export_time': time.time()
        }
        
        return export_data
    
    def import_observer(self, data: Dict[str, Any], new_name: Optional[str] = None) -> bool:
        """
        Import an observer from serialized data
        
        Args:
            data (dict): Serialized observer data
            new_name (str, optional): New name for the observer, or None to use original
            
        Returns:
            bool: True if import was successful
        """
        if not isinstance(data, dict) or 'type' not in data or 'properties' not in data:
            logger.warning(f"Invalid observer data format")
            return False
        
        # Get observer name
        observer_name = new_name or data.get('name')
        if not observer_name:
            logger.warning(f"No observer name specified")
            return False
        
        # Check if observer already exists
        if observer_name in self.observers:
            logger.warning(f"Observer already exists: {observer_name}")
            return False
        
        # Check if observer type is valid or needs to be registered
        observer_type = data.get('type')
        if observer_type not in self.observer_types:
            # Register the type with default properties
            self.register_observer_type(observer_type, {})
        
        # Create observer
        self.create_observer(observer_name, observer_type, data.get('properties', {}))
        
        # Set phase
        if 'phase' in data:
            self.set_observer_phase(observer_name, data['phase'])
        
        # Import history if present
        if 'history' in data and isinstance(data['history'], list):
            self.observer_histories[observer_name] = data['history'].copy()
        
        # Set timestamps
        if 'created_at' in data:
            self.observer_creation_times[observer_name] = data['created_at']
        if 'last_activity' in data:
            self.observer_last_activity[observer_name] = data['last_activity']
        
        # Import relationships if present
        if 'relationships' in data and isinstance(data['relationships'], list):
            for rel_data in data['relationships']:
                other_observer = rel_data.get('observer')
                if other_observer and other_observer in self.observers:
                    # Establish relationship
                    self.establish_relationship(
                        observer_name,
                        other_observer,
                        rel_data.get('type', 'imported'),
                        rel_data.get('strength', 0.5),
                        rel_data.get('properties', {})
                    )
        
        # Add import event to history
        import_record = {
            'type': 'import',
            'original_name': data.get('name'),
            'timestamp': time.time()
        }
        
        self.observer_histories.setdefault(observer_name, []).append(import_record)
        
        logger.info(f"Imported observer as {observer_name}")
        return True
    
    def export_all_observers(self) -> Dict[str, Dict[str, Any]]:
        """
        Export all observers
        
        Returns:
            dict: Dictionary mapping observer names to serialized data
        """
        exported = {}
        
        for name in self.observers:
            exported[name] = self.export_observer(name)
        
        return exported
    
    def export_registry_metadata(self) -> Dict[str, Any]:
        """
        Export registry metadata
        
        Returns:
            dict: Registry metadata
        """
        return {
            'observer_count': len(self.observers),
            'observer_types': list(self.observer_types.keys()),
            'relationship_count': len(self.observer_relationships),
            'export_time': time.time(),
            'registry_stats': self.get_observer_stats()
        }
    
    def reset(self) -> None:
        """
        Reset the registry, clearing all observers
        """
        # Clear all data structures
        self.observers = {}
        self.observer_properties = {}
        self.observers_by_target = {}
        self.observer_phases = {}
        self.observer_histories = {}
        self.observer_relationships = {}
        self.observer_creation_times = {}
        self.observer_last_activity = {}
        
        logger.info("Observer registry reset")

