from pyparsing import deque


class TimeStepController:
    """
    Manages adaptive time stepping for physics simulations.
    
    Adjusts time step based on stability criteria, state changes, and coupling
    dynamics to ensure accurate and efficient simulation.
    """
    
    def __init__(self, base_time_step=0.01, min_factor=0.1, max_factor=2.0):
        """Initialize with time step parameters.
        
        Args:
            base_time_step: Base time step size
            min_factor: Minimum scaling factor
            max_factor: Maximum scaling factor
        """
        self.base_time_step = base_time_step
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.current_factor = 1.0
        self.last_coherence_values = None
        self.strain_history = deque(maxlen=5)
        self.coherence_change_rate = 0.0
        
        # Hysteresis parameters
        self.hysteresis = 0.1  # Required change to trigger adjustment
        self.smoothing = 0.7   # Smoothing factor for changes
        
    def calculate_time_step(self, coherence_values=None, strain_values=None, 
                           active_observer_count=0):
        """Calculate adaptive time step based on current state.
        
        Args:
            coherence_values: Current coherence values by state
            strain_values: Current strain values by region
            active_observer_count: Number of active observers
            
        Returns:
            float: Recommended time step
        """
        # Start with current factor
        new_factor = self.current_factor
        
        # Adjust based on coherence changes
        if coherence_values and self.last_coherence_values:
            # Calculate maximum change rate in coherence
            max_delta = 0.0
            
            for state, value in coherence_values.items():
                if state in self.last_coherence_values:
                    delta = abs(value - self.last_coherence_values[state])
                    max_delta = max(max_delta, delta)
            
            # Smooth the change rate
            self.coherence_change_rate = (self.coherence_change_rate * self.smoothing + 
                                        max_delta * (1.0 - self.smoothing))
            
            # Adjust factor based on change rate
            if self.coherence_change_rate > 0.1:
                # High change rate -> smaller steps
                new_factor *= 0.8
            elif self.coherence_change_rate < 0.01:
                # Low change rate -> larger steps
                new_factor *= 1.1
        
        # Store current values for next comparison
        self.last_coherence_values = coherence_values.copy() if coherence_values else None
        
        # Adjust based on strain
        if strain_values:
            # Calculate average strain
            avg_strain = sum(strain_values.values()) / max(1, len(strain_values))
            
            # Store for history
            self.strain_history.append(avg_strain)
            
            # If strain is high or increasing rapidly, reduce time step
            if avg_strain > 0.7:
                new_factor *= 0.7
            elif len(self.strain_history) >= 3:
                # Check for rapidly increasing strain
                increasing = all(self.strain_history[i] < self.strain_history[i+1] 
                               for i in range(len(self.strain_history)-3, len(self.strain_history)-1))
                
                if increasing and self.strain_history[-1] - self.strain_history[-3] > 0.1:
                    new_factor *= 0.8
        
        # Adjust based on observer count (more observers = smaller steps)
        if active_observer_count > 3:
            observer_factor = max(0.5, 1.0 - (active_observer_count - 3) * 0.1)
            new_factor *= observer_factor
        
        # Apply hysteresis to avoid oscillations
        if abs(new_factor - self.current_factor) / self.current_factor > self.hysteresis:
            # Apply smoothing
            self.current_factor = (self.current_factor * self.smoothing + 
                                 new_factor * (1.0 - self.smoothing))
        
        # Ensure factor stays within bounds
        self.current_factor = max(self.min_factor, min(self.max_factor, self.current_factor))
        
        # Calculate final time step
        return self.base_time_step * self.current_factor