"""
Core module for the Recursia bytecode system.

This module provides the foundational components for the bytecode-based
execution system. All AST-related imports have been removed.
"""

# Version information
__version__ = '2.0.0'

# Import bytecode system components
from src.core.direct_parser import DirectParser
from src.core.bytecode import OpCode, Instruction, BytecodeModule
from src.core.bytecode_vm import RecursiaVM
from src.core.compiler import RecursiaCompiler
from src.core.runtime import RecursiaRuntime, create_optimized_runtime, get_global_runtime
from src.core.execution_context import ExecutionContext
from src.core.interpreter import CompilationResult, ExecutionResult
from src.core.data_classes import (
    Token, LexerError, ParserError, SemanticError, CompilerError
)
from src.core.types import TokenType

# Core subsystem imports
from src.core.state_registry import StateRegistry
from src.core.symbol_table import SymbolTable
from src.core.scope import Scope

__all__ = [
    # Parser and compiler
    'DirectParser',
    'RecursiaCompiler',
    
    # Bytecode system
    'OpCode', 'Instruction', 'BytecodeModule',
    'RecursiaVM',
    
    # Runtime
    'RecursiaRuntime', 'create_optimized_runtime', 'get_global_runtime',
    'ExecutionContext',
    
    # Results
    'CompilationResult', 'ExecutionResult',
    
    # Errors and tokens
    'Token', 'TokenType', 'LexerError', 'ParserError', 
    'SemanticError', 'CompilerError',
    
    # Core subsystems
    'StateRegistry', 'SymbolTable', 'Scope',
]