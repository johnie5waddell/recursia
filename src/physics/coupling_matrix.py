from typing import Dict, Optional

class CouplingMatrix:
    """
    Manages the coupling relationships between different physics subsystems.
    
    Provides centralized control over coupling strengths, directions, and delays 
    between quantum, observer, memory, and recursive subsystems in the OSH framework.
    """
    
    def __init__(self, config=None):
        """Initialize coupling matrix with default or configured values.
        
        Args:
            config: Optional configuration overrides
        """
        self.strengths = {}
        self.delays = {}
        self.directionality = {}
        
        # Load defaults
        self._initialize_defaults()
        
        # Apply config overrides if provided
        if config:
            self._apply_config(config)
    
    def _initialize_defaults(self):
        """Initialize default coupling relationships."""
        # Default coupling strengths (0.0-1.0)
        self.strengths = {
            ("observer", "quantum"): 0.8,
            ("memory", "coherence"): 0.7,
            ("recursion", "memory"): 0.6,
            ("field", "quantum"): 0.5,
            ("observer", "memory"): 0.4,
            ("entanglement", "coherence"): 0.9,
        }
        
        # Coupling delays (time steps)
        self.delays = {
            ("observer", "quantum"): 0,
            ("memory", "coherence"): 0,
            ("recursion", "memory"): 1,  # Recursion effects delayed by one time step
            ("field", "quantum"): 0,
            ("observer", "memory"): 0,
            ("entanglement", "coherence"): 0,
        }
        
        # Coupling directionality
        self.directionality = {
            ("observer", "quantum"): "bidirectional",
            ("memory", "coherence"): "bidirectional",
            ("field", "quantum"): "bidirectional",
            ("observer", "memory"): "bidirectional",
            ("entanglement", "coherence"): "bidirectional",
            ("recursion", "memory"): "asymmetric",
        }
    
    def _apply_config(self, config):
        """Apply configuration overrides to coupling settings.
        
        Args:
            config: Dict with coupling overrides
        """
        # Apply strength overrides
        for coupling, value in config.get("coupling_strengths", {}).items():
            if coupling in self.strengths:
                self.strengths[coupling] = value
        
        # Apply delay overrides
        for coupling, value in config.get("coupling_delays", {}).items():
            if coupling in self.delays:
                self.delays[coupling] = value
        
        # Apply directionality overrides
        for coupling, value in config.get("coupling_directionality", {}).items():
            if coupling in self.directionality:
                self.directionality[coupling] = value
    
    def get_strength(self, source, target):
        """Get coupling strength between two subsystems.
        
        Args:
            source: Source subsystem name
            target: Target subsystem name
            
        Returns:
            float: Coupling strength in [0.0-1.0] range, 0.0 if not found
        """
        forward_key = (source, target)
        reverse_key = (target, source)
        
        if forward_key in self.strengths:
            return self.strengths[forward_key]
        elif reverse_key in self.strengths and self.directionality.get(reverse_key) == "bidirectional":
            return self.strengths[reverse_key]
        else:
            return 0.0
    
    def get_delay(self, source, target):
        """Get delay between two subsystems in time steps.
        
        Args:
            source: Source subsystem name
            target: Target subsystem name
            
        Returns:
            int: Delay in time steps, 0 if not found
        """
        forward_key = (source, target)
        reverse_key = (target, source)
        
        if forward_key in self.delays:
            return self.delays[forward_key]
        elif reverse_key in self.delays and self.directionality.get(reverse_key) == "bidirectional":
            return self.delays[reverse_key]
        else:
            return 0
    
    def get_directionality(self, source, target):
        """Get directionality between two subsystems.
        
        Args:
            source: Source subsystem name
            target: Target subsystem name
            
        Returns:
            str: 'bidirectional', 'asymmetric', or None if not found
        """
        forward_key = (source, target)
        reverse_key = (target, source)
        
        if forward_key in self.directionality:
            return self.directionality[forward_key]
        elif reverse_key in self.directionality:
            return self.directionality[reverse_key]
        else:
            return None
    
    def create_interaction(self, source, target, effect, execution_time):
        """Create a delayed interaction object.
        
        Args:
            source: Source subsystem name
            target: Target subsystem name
            effect: Effect data dict
            execution_time: When to execute the interaction
            
        Returns:
            dict: Interaction specification
        """
        return {
            "source": source,
            "target": target,
            "effect": effect,
            "execution_time": execution_time,
            "directionality": self.get_directionality(source, target)
        }