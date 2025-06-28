"""
Minimal interpreter module for bytecode system.

This module provides the CompilationResult class and basic execution result structures
for the bytecode-based system. All AST-based interpretation has been removed.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class CompilationResult:
    """Container for compilation results."""
    success: bool
    bytecode_module: Optional[Any] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __bool__(self) -> bool:
        return self.success


@dataclass 
class ExecutionResult:
    """Container for execution results."""
    success: bool
    output: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    
    def __bool__(self) -> bool:
        return self.success