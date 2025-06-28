import logging

from src.core.utils import PerformanceProfiler


class PhysicsProfiler:
    """
    Centralized profiling and logging system for the physics engine.
    
    Provides consistent timing, tracing, and logging of physics operations
    with minimal code duplication.
    """
    
    def __init__(self, logger=None, profiler=None):
        """Initialize with optional logger and profiler.
        
        Args:
            logger: Optional logging.Logger instance
            profiler: Optional PerformanceProfiler instance
        """
        self.logger = logger or logging.getLogger("recursia.physics_engine")
        self.profiler = profiler or PerformanceProfiler()
        self.last_step_timings = {}
        self.subsystem_timings = {}
        self.subsystem_counts = {}
        
    def timed_step(self, step_name):
        """Context manager for timing a step with profiling and logging.
        
        Args:
            step_name: Name of the step being timed
            
        Returns:
            Context manager that times the step
        """
        return self.TimedStep(self, step_name)
        
    class TimedStep:
        """Context manager for timing a step with profiling and logging."""
        
        def __init__(self, profiler, step_name):
            """Initialize with parent profiler and step name.
            
            Args:
                profiler: Parent PhysicsProfiler
                step_name: Name of the step being timed
            """
            self.profiler = profiler
            self.step_name = step_name
            self.step_time = None
            
        def __enter__(self):
            """Start timing the step."""
            if self.profiler.profiler:
                self.profiler.profiler.start_timer(self.step_name)
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            """Stop timing and log results."""
            if self.profiler.profiler:
                self.step_time = self.profiler.profiler.stop_timer(self.step_name)
                
                # Update subsystem timings
                subsystem = self.step_name.split('_')[0] if '_' in self.step_name else self.step_name
                
                if subsystem not in self.profiler.subsystem_timings:
                    self.profiler.subsystem_timings[subsystem] = 0
                    self.profiler.subsystem_counts[subsystem] = 0
                    
                self.profiler.subsystem_timings[subsystem] += self.step_time
                self.profiler.subsystem_counts[subsystem] += 1
                
                # Store for last step timings
                self.profiler.last_step_timings[self.step_name] = self.step_time
                
                # Log if slow execution
                if self.step_time > 0.1:  # More than 100ms
                    self.profiler.logger.debug(
                        f"Slow execution in {self.step_name}: {self.step_time:.4f}s"
                    )
                
            # Log any exceptions that occurred
            if exc_type is not None:
                self.profiler.logger.error(
                    f"Exception in {self.step_name}: {exc_type.__name__}: {exc_val}"
                )
                
            return False  # Don't suppress exceptions
            
    def log_step(self, step_name, level="debug", **kwargs):
        """Log a step with consistent formatting.
        
        Args:
            step_name: Name of the step to log
            level: Log level (debug, info, warning, error)
            **kwargs: Additional context data to include in log
        """
        log_method = getattr(self.logger, level, self.logger.debug)
        
        # Format message with step name and context
        context_str = ", ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
        message = f"{step_name}"
        
        if context_str:
            message += f" ({context_str})"
            
        log_method(message)
        
    def get_timing_summary(self):
        """Get summary of timing information.
        
        Returns:
            Dict[str, Any]: Timing summary by subsystem
        """
        summary = {
            "last_step_timings": self.last_step_timings.copy(),
            "subsystem_timings": {
                subsystem: {
                    "total_time": time,
                    "call_count": self.subsystem_counts.get(subsystem, 0),
                    "avg_time": time / max(1, self.subsystem_counts.get(subsystem, 0))
                }
                for subsystem, time in self.subsystem_timings.items()
            }
        }
        
        # Add overall timings
        summary["total_time"] = sum(self.last_step_timings.values())
        summary["subsystem_count"] = len(self.subsystem_timings)
        
        return summary
        
    def reset_timings(self):
        """Reset all timing information."""
        self.last_step_timings.clear()
        self.subsystem_timings.clear()
        self.subsystem_counts.clear()
        
        if hasattr(self.profiler, "reset"):
            self.profiler.reset()
from typing import Any, Dict, Optional