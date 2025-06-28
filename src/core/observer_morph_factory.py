import time
from typing import Any, Dict, Tuple

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


class ObserverMorphFactory:
    """
    Factory class for observer transformations like cloning, merging, and splitting.
    """
    
    def __init__(self, registry):
        """Initialize with reference to parent registry"""
        self.registry = registry
    
    def clone(self, source_name: str, new_name: str, mutation_factor: float = 0.1) -> bool:
        """
        Clone an observer with optional mutation
        
        Args:
            source_name (str): Source observer name
            new_name (str): Name for the clone
            mutation_factor (float): How much to mutate properties (0.0-1.0)
            
        Returns:
            bool: True if clone was successful
        """
        if source_name not in self.registry.observers:
            self.logger.warning(f"Source observer not found: {source_name}")
            return False
        
        if new_name in self.registry.observers:
            self.logger.warning(f"Target observer already exists: {new_name}")
            return False
        
        # Get source observer data
        source_type = self.registry.observers[source_name]['type']
        source_properties = self.registry.observer_properties.get(source_name, {}).copy()
        
        # Apply mutations
        if mutation_factor > 0:
            mutated_properties = self._mutate_properties(source_properties, mutation_factor)
            source_properties = mutated_properties
        
        # Create clone
        self.registry.create_observer(new_name, source_type, source_properties)
        
        # Copy phase
        source_phase = self.registry.observer_phases.get(source_name, "passive")
        self.registry._transition_observer_phase(new_name, source_phase)
        
        # Add clone event to history
        clone_record = {
            'type': 'clone',
            'source_observer': source_name,
            'mutation_factor': mutation_factor,
            'timestamp': time.time()
        }
        
        self.registry.observer_histories.setdefault(new_name, []).append(clone_record)
        
        # Create relationship with source
        self.registry._update_relationship(
            source_name, new_name, 
            rel_type="clone_origin", 
            rel_strength=0.9,
            rel_props={
                'clone_time': time.time(),
                'mutation_factor': mutation_factor
            }
        )
        
        self.logger.info(f"Cloned observer {source_name} to {new_name} with mutation {mutation_factor}")
        return True
    
    def _mutate_properties(self, properties: Dict[str, Any], mutation_factor: float) -> Dict[str, Any]:
        """
        Apply random mutations to observer properties
        
        Args:
            properties (dict): Original properties 
            mutation_factor (float): Mutation strength factor
            
        Returns:
            dict: Mutated properties
        """
        mutated = properties.copy()
        
        for key, value in mutated.items():
            if isinstance(value, (int, float)):
                # Apply random mutation based on factor
                mutation = np.random.normal(0, mutation_factor * 0.3)
                mutated[key] = value * (1.0 + mutation)
                
                # Ensure values stay in reasonable range
                if 'threshold' in key or 'factor' in key or key in ['focus_strength', 'self_awareness']:
                    mutated[key] = max(0.0, min(1.0, mutated[key]))
        
        return mutated
    
    def split(self, observer_name: str, new_name1: str, new_name2: str,
              split_ratio: float = 0.5, split_method: str = 'divide') -> bool:
        """
        Split an observer into two new observers
        
        Args:
            observer_name (str): Name of observer to split
            new_name1 (str): Name for first new observer
            new_name2 (str): Name for second new observer
            split_ratio (float): Ratio for property division (0.0-1.0)
            split_method (str): Split method ('divide', 'replicate', 'specialize')
            
        Returns:
            bool: True if split was successful
        """
        if observer_name not in self.registry.observers:
            self.logger.warning(f"Observer not found: {observer_name}")
            return False
        
        if new_name1 in self.registry.observers or new_name2 in self.registry.observers:
            self.logger.warning(f"Target observer already exists")
            return False
        
        # Ensure split ratio is in valid range
        split_ratio = max(0.1, min(0.9, split_ratio))
        
        # Get original observer data
        original_data = self.registry.observers[observer_name]
        original_properties = self.registry.observer_properties.get(observer_name, {}).copy()
        original_phase = self.registry.observer_phases.get(observer_name, "passive")
        
        # Determine type and properties for split observers based on method
        type1, type2, properties1, properties2 = self._resolve_split_types_and_properties(
            original_data['type'], original_properties, split_ratio, split_method
        )
        
        # Create the new observers
        self.registry.create_observer(new_name1, type1, properties1)
        self.registry.create_observer(new_name2, type2, properties2)
        
        # Set phases based on split method
        self._set_split_phases(new_name1, new_name2, original_phase, split_method)
        
        # Handle relationships
        self._transfer_relationships(observer_name, new_name1, new_name2, split_ratio)
        
        # Establish relationship between the split observers
        self.registry._update_relationship(
            new_name1, new_name2, 
            rel_type="split_origin", 
            rel_strength=0.8,
            rel_props={
                'original_observer': observer_name,
                'split_time': time.time(),
                'split_method': split_method,
                'split_ratio': split_ratio
            }
        )
        
        # Split observer history
        self._split_history(observer_name, new_name1, new_name2, split_ratio, split_method)
        
        # Delete the original observer
        self.registry.delete_observer(observer_name)
        
        self.logger.info(f"Split observer {observer_name} into {new_name1} and {new_name2}")
        return True
    
    def _resolve_split_types_and_properties(self, original_type: str, original_properties: Dict[str, Any],
                                           split_ratio: float, split_method: str) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
        """
        Determine types and properties for split observers
        
        Args:
            original_type (str): Original observer type
            original_properties (dict): Original observer properties
            split_ratio (float): Split ratio
            split_method (str): Split method
            
        Returns:
            Tuple containing: type1, type2, properties1, properties2
        """
        if split_method == 'replicate':
            # Both new observers have same type and properties as original
            type1 = original_type
            type2 = original_type
            
            # Copy properties
            properties1 = original_properties.copy()
            properties2 = original_properties.copy()
            
            # Slightly perturb properties to make them unique
            for key, value in properties1.items():
                if isinstance(value, (int, float)):
                    properties1[key] = value * (1.0 + np.random.normal(0, 0.05))
                    properties2[key] = value * (1.0 + np.random.normal(0, 0.05))
        
        elif split_method == 'specialize':
            # Specialize each observer for different aspects
            
            # Determine specializations based on original type
            if 'quantum' in original_type:
                type1 = 'quantum_observer'
                type2 = 'meta_observer'
            elif 'recursive' in original_type:
                type1 = 'recursive_observer'
                type2 = 'quantum_observer'
            elif 'meta' in original_type:
                type1 = 'meta_observer'
                type2 = 'conscious_observer'
            elif 'conscious' in original_type:
                type1 = 'conscious_observer'
                type2 = 'subconscious_observer'
            else:
                # Default specialization
                type1 = 'standard_observer'
                type2 = 'quantum_observer'
            
            # Get default properties for types
            properties1 = self.registry.observer_types.get(type1, {}).copy()
            properties2 = self.registry.observer_types.get(type2, {}).copy()
            
            # Copy relevant properties from original
            common_props = ['observer_focus', 'focus_strength']
            for prop in common_props:
                if prop in original_properties:
                    properties1[prop] = original_properties[prop]
                    properties2[prop] = original_properties[prop]
            
            # Adjust strengths for specialization
            if 'collapse_threshold' in original_properties:
                properties1['collapse_threshold'] = original_properties['collapse_threshold'] * 1.1
                properties2['collapse_threshold'] = original_properties['collapse_threshold'] * 0.9
        
        else:  # 'divide' or default
            # Divide observer properties based on split ratio
            type1 = original_type
            type2 = original_type
            
            properties1 = {}
            properties2 = {}
            
            for key, value in original_properties.items():
                if isinstance(value, (int, float)):
                    # Divide numeric properties
                    properties1[key] = value * split_ratio
                    properties2[key] = value * (1 - split_ratio)
                else:
                    # Copy non-numeric properties
                    properties1[key] = value
                    properties2[key] = value
        
        return type1, type2, properties1, properties2
    
    def _set_split_phases(self, new_name1: str, new_name2: str, 
                         original_phase: str, split_method: str) -> None:
        """
        Set phases for the split observers based on method
        
        Args:
            new_name1 (str): First new observer
            new_name2 (str): Second new observer
            original_phase (str): Original observer phase
            split_method (str): Split method 
        """
        if split_method == 'replicate':
            # Both get the original phase
            self.registry._transition_observer_phase(new_name1, original_phase)
            self.registry._transition_observer_phase(new_name2, original_phase)
        else:
            # Different phases based on division
            if original_phase in ["measuring", "analyzing", "entangled"]:
                self.registry._transition_observer_phase(new_name1, original_phase)
                self.registry._transition_observer_phase(new_name2, "active")
            else:
                self.registry._transition_observer_phase(new_name1, "active")
                self.registry._transition_observer_phase(new_name2, "passive")
    
    def _transfer_relationships(self, original_name: str, new_name1: str, 
                               new_name2: str, split_ratio: float) -> None:
        """
        Transfer relationships from original observer to split observers
        
        Args:
            original_name (str): Original observer name
            new_name1 (str): First new observer name
            new_name2 (str): Second new observer name
            split_ratio (float): Split ratio for relationship strength
        """
        for key, relationship in list(self.registry.observer_relationships.items()):
            if original_name in key:
                # Get the other observer
                other_observer = key[0] if key[1] == original_name else key[1]
                
                # Skip if the other observer doesn't exist
                if other_observer not in self.registry.observers:
                    continue
                
                # Get relationship data
                rel_type = relationship['type']
                rel_strength = relationship['strength']
                rel_props = relationship['properties'].copy()
                
                # Establish relationship with both new observers, but with reduced strength
                strength1 = rel_strength * split_ratio
                strength2 = rel_strength * (1 - split_ratio)
                
                self.registry._update_relationship(new_name1, other_observer, rel_type, strength1, rel_props)
                self.registry._update_relationship(new_name2, other_observer, rel_type, strength2, rel_props)
    
    def _split_history(self, original_name: str, new_name1: str, new_name2: str, 
                      split_ratio: float, split_method: str) -> None:
        """
        Split and assign observer history
        
        Args:
            original_name (str): Original observer name
            new_name1 (str): First new observer name
            new_name2 (str): Second new observer name
            split_ratio (float): Split ratio
            split_method (str): Split method
        """
        original_history = self.registry.observer_histories.get(original_name, [])
        
        # Split history based on method
        if split_method == 'replicate':
            # Both get full history
            history1 = original_history.copy()
            history2 = original_history.copy()
        else:
            # Divide history based on timestamps
            history_sorted = sorted(original_history, key=lambda x: x.get('timestamp', 0))
            split_index = int(len(history_sorted) * split_ratio)
            
            # Make sure each gets at least one entry if there's any history
            if split_index == 0 and len(history_sorted) > 0:
                split_index = 1
            elif split_index == len(history_sorted) and len(history_sorted) > 0:
                split_index = len(history_sorted) - 1
            
            history1 = history_sorted[:split_index]
            history2 = history_sorted[split_index:]
        
        # Add split event to history
        split_record = {
            'type': 'split',
            'original_observer': original_name,
            'split_ratio': split_ratio,
            'split_method': split_method,
            'timestamp': time.time()
        }
        
        history1.append(split_record)
        history2.append(split_record)
        
        # Set history for new observers
        self.registry.observer_histories[new_name1] = history1
        self.registry.observer_histories[new_name2] = history2
        
    def merge(self, observer1: str, observer2: str, new_name: str,
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
        if observer1 not in self.registry.observers or observer2 not in self.registry.observers:
            self.logger.warning(f"Cannot merge: observer not found")
            return False
        
        if new_name in self.registry.observers:
            self.logger.warning(f"Merge target observer already exists: {new_name}")
            return False
        
        # Get observer data
        data1 = self.registry.observers[observer1]
        data2 = self.registry.observers[observer2]
        props1 = self.registry.observer_properties.get(observer1, {})
        props2 = self.registry.observer_properties.get(observer2, {})
        
        # Determine merged type and properties
        merged_type = self._determine_merged_type(data1['type'], data2['type'], merge_type)
        
        # Register the new merged observer type if it doesn't exist
        if merged_type not in self.registry.observer_types and merged_type not in {'quantum_recursive_observer'}:
            # Register as a new type with merged properties
            base_props = {}
            if merged_type == 'quantum_recursive_observer':
                base_props = {
                    'collapse_threshold': 0.7,
                    'observer_recursion_depth': 3,
                    'self_awareness': 0.7,
                    'entanglement_sensitivity': 0.8
                }
            self.registry.register_observer_type(merged_type, base_props)
        
        # Merge properties
        merged_properties = self._merge_properties(props1, props2, merge_type)
        
        # Create the merged observer
        self.registry.create_observer(new_name, merged_type, merged_properties)
        
        # Determine merged phase
        merged_phase = self._determine_merged_phase(observer1, observer2)
        self.registry._transition_observer_phase(new_name, merged_phase)
        
        # Merge relationships
        self._merge_relationships(observer1, observer2, new_name)
        
        # Merge and set history
        self._merge_history(observer1, observer2, new_name)
        
        # Delete the original observers
        self.registry.delete_observer(observer1)
        self.registry.delete_observer(observer2)
        
        self.logger.info(f"Merged observers {observer1} and {observer2} into {new_name}")
        return True
    
    def _determine_merged_type(self, type1: str, type2: str, merge_type: str) -> str:
        """
        Determine the type for a merged observer
        
        Args:
            type1 (str): First observer type
            type2 (str): Second observer type
            merge_type (str): Merge strategy
            
        Returns:
            str: Merged observer type
        """
        if merge_type == 'dominant':
            # Use type from observer with higher complexity descriptor
            if 'meta' in type1 or 'recursive' in type1:
                return type1
            elif 'meta' in type2 or 'recursive' in type2:
                return type2
            elif 'quantum' in type1:
                return type1
            elif 'quantum' in type2:
                return type2
            else:
                return type1  # Default to first
        elif merge_type == 'coherent':
            # Use type that maintains higher coherence
            if 'quantum' in type1 or 'recursive' in type1:
                return type1
            elif 'quantum' in type2 or 'recursive' in type2:
                return type2
            else:
                # Default to more complex type
                return type1 if 'meta' in type1 or 'conscious' in type1 else type2
        else:  # 'average' or default
            # If types are same, keep it; otherwise create a hybrid type
            if type1 == type2:
                return type1
            elif 'quantum' in type1 and 'recursive' in type2:
                return 'quantum_recursive_observer'
            elif 'recursive' in type1 and 'quantum' in type2:
                return 'quantum_recursive_observer'
            elif 'meta' in type1 or 'meta' in type2:
                return 'meta_observer'
            else:
                # Default to the more specialized type
                standard_types = {'standard_observer', 'basic_observer'}
                return type2 if type1 in standard_types else type1
    
    def _merge_properties(self, props1: Dict[str, Any], props2: Dict[str, Any], 
                         merge_type: str) -> Dict[str, Any]:
        """
        Merge properties of two observers
        
        Args:
            props1 (dict): First observer properties
            props2 (dict): Second observer properties
            merge_type (str): Merge strategy
            
        Returns:
            dict: Merged properties
        """
        merged_properties = {}
        
        # Combine all properties from both observers
        all_prop_keys = set(props1.keys()) | set(props2.keys())
        
        for key in all_prop_keys:
            if key in props1 and key in props2:
                # Both have this property - resolve using strategy
                merged_properties[key] = self.registry._resolve_merged_property(
                    key, props1[key], props2[key], merge_type
                )
            elif key in props1:
                # Only in props1
                merged_properties[key] = props1[key]
            else:
                # Only in props2
                merged_properties[key] = props2[key]
        
        return merged_properties
    
    def _determine_merged_phase(self, observer1: str, observer2: str) -> str:
        """
        Determine the phase for a merged observer
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            
        Returns:
            str: Merged phase
        """
        phase1 = self.registry.observer_phases.get(observer1, "passive")
        phase2 = self.registry.observer_phases.get(observer2, "passive")
        
        if phase1 == phase2:
            return phase1
        elif 'active' in (phase1, phase2):
            return 'active'
        elif 'measuring' in (phase1, phase2):
            return 'measuring'
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
            return phase1 if phase_priority.get(phase1, 0) >= phase_priority.get(phase2, 0) else phase2
    
    def _merge_relationships(self, observer1: str, observer2: str, new_name: str) -> None:
        """
        Transfer relationships from original observers to new merged observer
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            new_name (str): New merged observer name
        """
        for key, relationship in list(self.registry.observer_relationships.items()):
            if observer1 in key or observer2 in key:
                # Get the other observer
                other_observer = None
                if key[0] == observer1 or key[0] == observer2:
                    other_observer = key[1]
                else:
                    other_observer = key[0]
                
                # Skip if the other observer is one we're merging
                if other_observer == observer1 or other_observer == observer2:
                    continue
                
                # Skip if the other observer doesn't exist
                if other_observer not in self.registry.observers:
                    continue
                
                # Create relationship with the merged observer
                rel_type = relationship['type']
                rel_strength = relationship['strength']
                rel_props = relationship['properties']
                
                self.registry._update_relationship(new_name, other_observer, rel_type, rel_strength, rel_props)
    
    def _merge_history(self, observer1: str, observer2: str, new_name: str) -> None:
        """
        Merge and set history for the new observer
        
        Args:
            observer1 (str): First observer name
            observer2 (str): Second observer name
            new_name (str): New merged observer name
        """
        history1 = self.registry.observer_histories.get(observer1, [])
        history2 = self.registry.observer_histories.get(observer2, [])
        
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
        self.registry.observer_histories[new_name] = merged_history