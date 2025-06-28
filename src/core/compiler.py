"""
Minimal compiler wrapper for bytecode system.

This module provides a simple interface for compiling Recursia code using the DirectParser
and bytecode system. All AST-based compilation has been removed.
"""

from typing import Dict, Optional, Any
from dataclasses import dataclass

from src.core.direct_parser import DirectParser
from src.core.bytecode_vm import RecursiaVM
from src.core.interpreter import CompilationResult
# Runtime import removed to avoid circular dependency


class RecursiaCompiler:
    """
    Simple compiler wrapper that uses DirectParser for bytecode generation.
    This replaces the complex AST-based compiler with a streamlined bytecode approach.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the compiler.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.parser = DirectParser()
    
    def compile(self, code: str, target: str = 'bytecode') -> CompilationResult:
        """
        Compile Recursia code to bytecode.
        
        Args:
            code: The Recursia source code
            target: Target platform (always 'bytecode' now)
            
        Returns:
            CompilationResult with success status and bytecode module
        """
        try:
            # Parse code to bytecode
            bytecode_module = self.parser.parse(code)
            
            if bytecode_module:
                return CompilationResult(
                    success=True,
                    bytecode_module=bytecode_module,
                    errors=[],
                    warnings=[]
                )
            else:
                return CompilationResult(
                    success=False,
                    bytecode_module=None,
                    errors=["Parser returned None"],
                    warnings=[]
                )
                
        except Exception as e:
            return CompilationResult(
                success=False,
                bytecode_module=None,
                errors=[str(e)],
                warnings=[]
            )
    
    def compile_file(self, filename: str, target: str = 'bytecode') -> Optional[Any]:
        """
        Compile a file to bytecode.
        
        Args:
            filename: Path to the file to compile
            target: Target platform (always 'bytecode' now)
            
        Returns:
            Bytecode module or None on error
        """
        try:
            with open(filename, 'r') as f:
                code = f.read()
            
            result = self.compile(code, target)
            if result.success:
                return result.bytecode_module
            else:
                for error in result.errors:
                    print(f"Error: {error}")
                return None
                
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
            return None