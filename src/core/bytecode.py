"""
Recursia Bytecode Instruction Set
=================================

This module defines a simple, efficient bytecode format for Recursia programs
that eliminates the need for AST and enables direct execution.

Key Benefits:
- Direct execution without AST traversal
- Efficient memory usage
- Simple type system
- Fast serialization/deserialization
- Clear instruction semantics
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import struct
import logging

logger = logging.getLogger(__name__)


class OpCode(Enum):
    """Bytecode operation codes for Recursia VM."""
    
    # Stack Operations
    LOAD_CONST = auto()      # Load constant onto stack
    LOAD_VAR = auto()        # Load variable value
    STORE_VAR = auto()       # Store top of stack to variable
    DUP = auto()             # Duplicate top of stack
    POP = auto()             # Remove top of stack
    SWAP = auto()            # Swap top two stack elements
    
    # Universe Operations
    START_UNIVERSE = auto()  # Start universe declaration
    END_UNIVERSE = auto()    # End universe declaration
    
    # Arithmetic Operations
    ADD = auto()             # Pop 2, push sum
    SUB = auto()             # Pop 2, push difference
    MUL = auto()             # Pop 2, push product
    DIV = auto()             # Pop 2, push quotient
    MOD = auto()             # Pop 2, push remainder
    POW = auto()             # Pop 2, push power
    NEG = auto()             # Pop 1, push negation
    ABS = auto()             # Pop 1, push absolute value
    EXP = auto()             # Pop 1, push e^x
    LOG = auto()             # Pop 1, push ln(x)
    
    # Comparison Operations
    EQ = auto()              # Pop 2, push equality
    NE = auto()              # Pop 2, push inequality
    LT = auto()              # Pop 2, push less than
    LE = auto()              # Pop 2, push less or equal
    GT = auto()              # Pop 2, push greater than
    GE = auto()              # Pop 2, push greater or equal
    
    # Logical Operations
    AND = auto()             # Pop 2, push logical and
    OR = auto()              # Pop 2, push logical or
    NOT = auto()             # Pop 1, push logical not
    
    # Control Flow
    JUMP = auto()            # Unconditional jump
    JUMP_IF = auto()         # Jump if top of stack is true
    JUMP_IF_FALSE = auto()   # Jump if top of stack is false
    CALL = auto()            # Call function
    RETURN = auto()          # Return from function
    
    # Quantum Operations
    CREATE_STATE = auto()    # Create quantum state
    CREATE_OBSERVER = auto() # Create observer
    APPLY_GATE = auto()      # Apply quantum gate
    MEASURE = auto()         # Measure quantum state
    MEASURE_QUBIT = auto()   # Measure specific qubit
    ENTANGLE = auto()        # Entangle states
    TELEPORT = auto()        # Teleport quantum state
    COHERE = auto()          # Set coherence level
    RECURSE = auto()         # Apply recursive simulation for RSP
    
    # Field Operations
    CREATE_FIELD = auto()    # Create field
    EVOLVE = auto()          # Evolve field/state
    
    # I/O Operations
    PRINT = auto()           # Print value
    
    # Special Operations
    BUILD_LIST = auto()      # Build list from stack items
    BUILD_DICT = auto()      # Build dict from stack items
    GET_ATTR = auto()        # Get attribute
    SET_ATTR = auto()        # Set attribute
    GET_ITEM = auto()        # Get item by index/key
    SET_ITEM = auto()        # Set item by index/key
    
    # OSH Measurement Operations
    MEASURE_II = auto()      # Measure integrated information
    MEASURE_KC = auto()      # Measure Kolmogorov complexity
    MEASURE_ENTROPY = auto() # Measure entropy
    MEASURE_COHERENCE = auto() # Measure coherence
    MEASURE_COLLAPSE = auto() # Measure collapse probability
    
    # Loop Operations  
    FOR_SETUP = auto()       # Setup for loop
    FOR_ITER = auto()        # For loop iteration
    BREAK = auto()           # Break from loop
    CONTINUE = auto()        # Continue to next iteration
    
    # End marker
    HALT = auto()            # End program execution


@dataclass
class Instruction:
    """Single bytecode instruction with opcode and arguments."""
    opcode: OpCode
    args: List[Any] = None
    
    def __post_init__(self):
        if self.args is None:
            self.args = []
    
    def __repr__(self):
        if self.args:
            return f"{self.opcode.name}({', '.join(map(repr, self.args))})"
        return self.opcode.name


@dataclass
class BytecodeModule:
    """Container for bytecode instructions and metadata."""
    instructions: List[Instruction]
    constants: List[Any]
    names: List[str]  # Variable/function names
    metadata: Dict[str, Any]
    
    def __init__(self):
        self.instructions = []
        self.constants = []
        self.names = []
        self.metadata = {}
        self.functions = {}  # name -> (start_pc, params)
        self._const_map = {}  # Cache for constant indices
        self._name_map = {}   # Cache for name indices
    
    def add_instruction(self, opcode: OpCode, *args) -> None:
        """Add an instruction to the module."""
        self.instructions.append(Instruction(opcode, list(args)))
    
    def add_constant(self, value: Any) -> int:
        """Add a constant and return its index."""
        # Use constant pooling to avoid duplicates
        key = (type(value), value) if not isinstance(value, (list, dict)) else id(value)
        if key in self._const_map:
            return self._const_map[key]
        
        index = len(self.constants)
        self.constants.append(value)
        self._const_map[key] = index
        return index
    
    def add_name(self, name: str) -> int:
        """Add a name and return its index."""
        if name in self._name_map:
            return self._name_map[name]
        
        index = len(self.names)
        self.names.append(name)
        self._name_map[name] = index
        return index
    
    def get_constant(self, index: int) -> Any:
        """Get constant by index."""
        if index < 0 or index >= len(self.constants):
            # Log error for debugging Windows issues
            import platform
            if platform.system() == 'Windows':
                print(f"[Windows Debug] Constant index {index} out of range. Constants length: {len(self.constants)}")
            # Return a safe default instead of crashing
            return 0
        return self.constants[index]
    
    def get_name(self, index: int) -> str:
        """Get name by index."""
        if index < 0 or index >= len(self.names):
            # Log error for debugging Windows issues
            import platform
            if platform.system() == 'Windows':
                print(f"[Windows Debug] Name index {index} out of range. Names length: {len(self.names)}")
            # Return a safe default instead of crashing
            return "default_var"
        return self.names[index]
    
    def add_function(self, name: str, start_pc: int, params: List[str]) -> int:
        """Add a function definition."""
        self.functions[name] = (start_pc, params)
        return self.add_name(name)
    
    def get_function(self, name: str) -> Optional[Tuple[int, List[str]]]:
        """Get function info by name."""
        return self.functions.get(name)
    
    def serialize(self) -> bytes:
        """Serialize bytecode to binary format."""
        # Simple serialization format:
        # [magic][version][const_count][constants][name_count][names][inst_count][instructions]
        buffer = bytearray()
        
        # Magic number and version
        buffer.extend(b'RCRS')  # Magic: Recursia
        buffer.extend(struct.pack('<H', 1))  # Version 1
        
        # Constants
        buffer.extend(struct.pack('<I', len(self.constants)))
        for const in self.constants:
            self._serialize_value(buffer, const)
        
        # Names
        buffer.extend(struct.pack('<I', len(self.names)))
        for name in self.names:
            name_bytes = name.encode('utf-8')
            buffer.extend(struct.pack('<I', len(name_bytes)))
            buffer.extend(name_bytes)
        
        # Instructions
        buffer.extend(struct.pack('<I', len(self.instructions)))
        for inst in self.instructions:
            buffer.append(inst.opcode.value)
            buffer.append(len(inst.args))
            for arg in inst.args:
                if isinstance(arg, int):
                    buffer.extend(struct.pack('<i', arg))
                else:
                    self._serialize_value(buffer, arg)
        
        return bytes(buffer)
    
    def _serialize_value(self, buffer: bytearray, value: Any) -> None:
        """Serialize a single value."""
        if value is None:
            buffer.append(0)  # Type: None
        elif isinstance(value, bool):
            buffer.append(1)  # Type: bool
            buffer.append(1 if value else 0)
        elif isinstance(value, int):
            buffer.append(2)  # Type: int
            buffer.extend(struct.pack('<q', value))
        elif isinstance(value, float):
            buffer.append(3)  # Type: float
            buffer.extend(struct.pack('<d', value))
        elif isinstance(value, str):
            buffer.append(4)  # Type: str
            str_bytes = value.encode('utf-8')
            buffer.extend(struct.pack('<I', len(str_bytes)))
            buffer.extend(str_bytes)
        elif isinstance(value, list):
            buffer.append(5)  # Type: list
            buffer.extend(struct.pack('<I', len(value)))
            for item in value:
                self._serialize_value(buffer, item)
        elif isinstance(value, dict):
            buffer.append(6)  # Type: dict
            buffer.extend(struct.pack('<I', len(value)))
            for k, v in value.items():
                self._serialize_value(buffer, k)
                self._serialize_value(buffer, v)
        else:
            raise ValueError(f"Cannot serialize type: {type(value)}")
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'BytecodeModule':
        """Deserialize bytecode from binary format."""
        # Implementation for loading bytecode
        # (Not fully implemented for brevity, but would reverse serialize())
        raise NotImplementedError("Bytecode deserialization not yet implemented")
    
    def disassemble(self) -> str:
        """Generate human-readable disassembly."""
        lines = []
        lines.append("=== Recursia Bytecode Disassembly ===")
        lines.append(f"Constants: {len(self.constants)}")
        for i, const in enumerate(self.constants):
            lines.append(f"  [{i}] {repr(const)}")
        
        lines.append(f"\nNames: {len(self.names)}")
        for i, name in enumerate(self.names):
            lines.append(f"  [{i}] {name}")
        
        lines.append(f"\nInstructions: {len(self.instructions)}")
        for i, inst in enumerate(self.instructions):
            lines.append(f"  {i:04d}: {inst}")
        
        return '\n'.join(lines)


class BytecodeCompiler:
    """Compiles tokens directly to bytecode, bypassing AST."""
    
    def __init__(self):
        self.module = BytecodeModule()
        self.stack_size = 0
        self.max_stack = 0
        self.loops = []  # Stack of loop contexts
        self.functions = {}  # Function definitions
    
    def compile_tokens(self, tokens: List[Any]) -> BytecodeModule:
        """Compile tokens directly to bytecode."""
        # This would be implemented to parse tokens and emit bytecode
        # For now, providing structure for the new architecture
        raise NotImplementedError("Direct token compilation not yet implemented")
    
    def emit(self, opcode: OpCode, *args) -> None:
        """Emit a bytecode instruction."""
        self.module.add_instruction(opcode, *args)
        
        # Track stack usage for optimization
        stack_effect = self._get_stack_effect(opcode)
        self.stack_size += stack_effect
        self.max_stack = max(self.max_stack, self.stack_size)
    
    def _get_stack_effect(self, opcode: OpCode) -> int:
        """Calculate stack effect of an opcode."""
        effects = {
            OpCode.LOAD_CONST: 1,
            OpCode.LOAD_VAR: 1,
            OpCode.STORE_VAR: -1,
            OpCode.DUP: 1,
            OpCode.POP: -1,
            OpCode.ADD: -1,
            OpCode.SUB: -1,
            OpCode.MUL: -1,
            OpCode.DIV: -1,
            OpCode.MOD: -1,
            OpCode.POW: -1,
            OpCode.NEG: 0,
            OpCode.EQ: -1,
            OpCode.NE: -1,
            OpCode.LT: -1,
            OpCode.LE: -1,
            OpCode.GT: -1,
            OpCode.GE: -1,
            OpCode.AND: -1,
            OpCode.OR: -1,
            OpCode.NOT: 0,
            OpCode.PRINT: -1,
            OpCode.JUMP: 0,
            OpCode.JUMP_IF: -1,
            OpCode.JUMP_IF_FALSE: -1,
            OpCode.CREATE_STATE: 0,
            OpCode.APPLY_GATE: -1,
            OpCode.MEASURE: 0,
            OpCode.ENTANGLE: -2,
        }
        return effects.get(opcode, 0)