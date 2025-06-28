"""
Direct Bytecode Parser for Recursia
===================================

Parses Recursia source code directly to bytecode, eliminating the AST layer.
This provides significant performance improvements and simpler architecture.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import logging

from src.core.lexer import Token, TokenType, RecursiaLexer
from src.core.bytecode import BytecodeModule, OpCode

logger = logging.getLogger(__name__)


@dataclass 
class ParseContext:
    """Context for parsing operations."""
    module: BytecodeModule
    loop_stack: List[Tuple[int, int]]  # (loop_start, break_target)
    function_stack: List[str]
    current_function: Optional[str] = None


class DirectParser:
    """
    Parser that compiles Recursia source directly to bytecode.
    
    This parser eliminates the AST intermediate representation,
    directly generating bytecode instructions during parsing.
    """
    
    def __init__(self):
        self.tokens: List[Token] = []
        self.current = 0
        self.context: Optional[ParseContext] = None
    
    def parse(self, source: str) -> BytecodeModule:
        """Parse source code and return bytecode module."""
        logger.debug("Starting parse")
        
        # Tokenize source
        lexer = RecursiaLexer(source)
        self.tokens = list(lexer.tokenize())
        self.current = 0
        
        logger.debug(f"Tokenized {len(self.tokens)} tokens")
        
        # Initialize parse context
        module = BytecodeModule()
        self.context = ParseContext(
            module=module,
            loop_stack=[],
            function_stack=[]
        )
        
        # Parse program with safety counter
        statement_count = 0
        max_iterations = len(self.tokens) * 10  # Safety limit
        iterations = 0
        last_position = -1
        
        while not self.is_at_end():
            iterations += 1
            
            # Safety check for infinite loops
            if iterations > max_iterations:
                logger.error(f"Parser exceeded maximum iterations ({max_iterations}). Possible infinite loop.")
                logger.error(f"Current position: {self.current}, Token: {self.peek()}")
                logger.error(f"Parsed {statement_count} statements so far")
                raise RuntimeError("Parser infinite loop detected")
            
            # Check if we're making progress
            if self.current == last_position:
                logger.warning(f"Parser not advancing! Position: {self.current}, Token: {self.peek()}")
                # Force advance to prevent infinite loop
                if not self.is_at_end():
                    logger.warning(f"Force advancing from token: {self.peek()}")
                    self.advance()
            
            last_position = self.current
            
            # Debug logging every iteration for troubleshooting
            if iterations % 100 == 0 or statement_count < 5:
                logger.debug(f"Iteration {iterations}: pos={self.current}/{len(self.tokens)}, token={self.peek()}")
            
            # Skip comments
            comment_count = 0
            while self.check(TokenType.COMMENT):
                self.advance()
                comment_count += 1
                if comment_count > 100:
                    logger.error("Too many consecutive comments, breaking")
                    break
            
            if not self.is_at_end():
                statement_count += 1
                logger.debug(f"Parsing statement {statement_count} at position {self.current}, token: {self.peek()}")
                
                try:
                    self.parse_statement()
                except Exception as e:
                    logger.error(f"Error parsing statement at position {self.current}: {e}")
                    logger.error(f"Current token: {self.peek()}")
                    logger.error(f"Previous tokens: {self.tokens[max(0, self.current-3):self.current]}")
                    raise
        
        logger.debug(f"Parsed {statement_count} statements total in {iterations} iterations")
        
        # Add halt instruction
        self.emit(OpCode.HALT)
        
        # Resolve jump targets
        self._resolve_jumps()
        
        logger.debug("Parse complete")
        return module
    
    def emit(self, opcode: OpCode, *args) -> int:
        """Emit bytecode instruction and return its position."""
        pos = len(self.context.module.instructions)
        self.context.module.add_instruction(opcode, *args)
        return pos
    
    def parse_statement(self) -> None:
        """Parse a single statement."""
        # Skip comments with safety counter
        comment_count = 0
        while self.check(TokenType.COMMENT):
            self.advance()
            comment_count += 1
            if comment_count > 100:
                logger.error("Too many consecutive comments in parse_statement")
                break
            
        if self.is_at_end():
            return
        
        # Check for empty statement (lone semicolon)
        if self.match(TokenType.SEMICOLON):
            return
        
        # Record starting position for infinite loop detection
        start_pos = self.current
            
        if self.match(TokenType.KEYWORD):
            keyword = self.previous().value
            
            if keyword == "universe":
                self.parse_universe_declaration()
            elif keyword == "state":
                self.parse_state_declaration()
            elif keyword == "observer":
                self.parse_observer_declaration()
            elif keyword == "apply":
                self.parse_apply_statement()
            elif keyword == "measure":
                self.parse_measure_statement()
            elif keyword == "entangle":
                self.parse_entangle_statement()
            elif keyword == "recurse":
                self.parse_recurse_statement()
            elif keyword == "cohere":
                self.parse_cohere_statement()
            elif keyword == "evolve":
                self.parse_evolve_statement()
            elif keyword == "print":
                self.parse_print_statement()
            elif keyword == "let":
                self.parse_let_statement()
            elif keyword == "const":
                self.parse_const_statement()
            elif keyword == "if":
                self.parse_if_statement()
            elif keyword == "while":
                self.parse_while_statement()
            elif keyword == "for":
                self.parse_for_statement()
            elif keyword == "function":
                self.parse_function_declaration()
            elif keyword == "return":
                self.parse_return_statement()
            elif keyword == "evolve":
                self.parse_evolve_statement()
            elif keyword == "break":
                self.emit(OpCode.BREAK)
                self.consume(TokenType.SEMICOLON, "Expected ';' after break")
            elif keyword == "continue":
                self.emit(OpCode.CONTINUE)
                self.consume(TokenType.SEMICOLON, "Expected ';' after continue")
            else:
                self.error(f"Unknown keyword: {keyword}")
        else:
            # Expression statement or assignment
            # Check if this is an identifier that might be assigned to
            if self.check(TokenType.IDENTIFIER):
                checkpoint = self.current
                var_name = self.advance().value
                
                # Check for assignment operators
                if self.match(TokenType.ASSIGN):
                    # Simple assignment: var = expr
                    self.parse_expression()
                    var_idx = self.context.module.add_name(var_name)
                    self.emit(OpCode.STORE_VAR, var_idx)
                elif self.check(TokenType.PLUS) and self.peek(1) and self.peek(1).type == TokenType.ASSIGN:
                    # Compound assignment: var += expr
                    self.advance()  # consume +
                    self.advance()  # consume =
                    
                    # Load current value
                    var_idx = self.context.module.add_name(var_name)
                    self.emit(OpCode.LOAD_VAR, var_idx)
                    
                    # Parse right side
                    self.parse_expression()
                    
                    # Add and store
                    self.emit(OpCode.ADD)
                    self.emit(OpCode.STORE_VAR, var_idx)
                elif self.check(TokenType.MINUS) and self.peek(1) and self.peek(1).type == TokenType.ASSIGN:
                    # Compound assignment: var -= expr
                    self.advance()  # consume -
                    self.advance()  # consume =
                    
                    # Load current value
                    var_idx = self.context.module.add_name(var_name)
                    self.emit(OpCode.LOAD_VAR, var_idx)
                    
                    # Parse right side
                    self.parse_expression()
                    
                    # Subtract and store
                    self.emit(OpCode.SUB)
                    self.emit(OpCode.STORE_VAR, var_idx)
                elif self.check(TokenType.ASTERISK) and self.peek(1) and self.peek(1).type == TokenType.ASSIGN:
                    # Compound assignment: var *= expr
                    self.advance()  # consume *
                    self.advance()  # consume =
                    
                    # Load current value
                    var_idx = self.context.module.add_name(var_name)
                    self.emit(OpCode.LOAD_VAR, var_idx)
                    
                    # Parse right side
                    self.parse_expression()
                    
                    # Multiply and store
                    self.emit(OpCode.MUL)
                    self.emit(OpCode.STORE_VAR, var_idx)
                elif self.check(TokenType.SLASH) and self.peek(1) and self.peek(1).type == TokenType.ASSIGN:
                    # Compound assignment: var /= expr
                    self.advance()  # consume /
                    self.advance()  # consume =
                    
                    # Load current value
                    var_idx = self.context.module.add_name(var_name)
                    self.emit(OpCode.LOAD_VAR, var_idx)
                    
                    # Parse right side
                    self.parse_expression()
                    
                    # Divide and store
                    self.emit(OpCode.DIV)
                    self.emit(OpCode.STORE_VAR, var_idx)
                else:
                    # Not an assignment, backtrack and parse as expression
                    self.current = checkpoint
                    self.parse_expression()
                    self.emit(OpCode.POP)  # Discard result
            else:
                # Regular expression
                self.parse_expression()
                self.emit(OpCode.POP)  # Discard result
            
            self.consume(TokenType.SEMICOLON, "Expected ';' after expression")
        
        # Ensure we've made progress
        if self.current == start_pos and not self.is_at_end():
            logger.error(f"parse_statement did not advance! Token: {self.peek()}")
            # Force advance to prevent infinite loop
            self.advance()
    
    def parse_universe_declaration(self) -> None:
        """Parse universe declaration."""
        # universe name { statements }
        name = self.consume(TokenType.IDENTIFIER, "Expected universe name").value
        
        # Emit universe start instruction
        name_const = self.context.module.add_constant(name)
        self.emit(OpCode.LOAD_CONST, name_const)
        self.emit(OpCode.START_UNIVERSE)
        
        # Parse universe body
        self.consume(TokenType.LBRACE, "Expected '{' after universe name")
        
        statement_count = 0
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            statement_count += 1
            if statement_count > 1000:
                logger.error("Too many statements in universe declaration")
                break
                
            # Skip comments
            while self.check(TokenType.COMMENT):
                self.advance()
                
            if not self.check(TokenType.RBRACE):
                self.parse_statement()
        
        self.consume(TokenType.RBRACE, "Expected '}' after universe body")
        
        # Emit universe end instruction
        self.emit(OpCode.END_UNIVERSE)
    
    def parse_state_declaration(self) -> None:
        """Parse quantum state declaration."""
        # state name : type { properties } OR state name { properties }
        name = self.consume(TokenType.IDENTIFIER, "Expected state name").value
        
        # Check if we have a type declaration
        if self.match(TokenType.COLON):
            # Parse type (e.g., quantum_type)
            state_type = self.consume(TokenType.IDENTIFIER, "Expected state type").value
        else:
            # Default type
            state_type = "quantum_type"
        
        # Parse properties
        properties = {}
        if self.match(TokenType.LBRACE):
            prop_iteration = 0
            while not self.check(TokenType.RBRACE) and not self.is_at_end():
                # Skip comments
                while self.check(TokenType.COMMENT):
                    self.advance()
                
                # Check again after skipping comments
                if self.check(TokenType.RBRACE):
                    break
                    
                prop_iteration += 1
                if prop_iteration > 100:
                    logger.error(f"Too many properties in state declaration at position {self.current}")
                    break
                    
                # Property name - allow keywords too
                if self.check(TokenType.IDENTIFIER):
                    prop_name = self.advance().value
                elif self.check(TokenType.KEYWORD):
                    prop_name = self.advance().value
                else:
                    self.error("Expected property name")
                    prop_name = "unknown"
                
                self.consume(TokenType.COLON, "Expected ':' after property name")
                
                # Property value - parse as expression and evaluate immediately
                if self.check(TokenType.NUMBER_LITERAL):
                    prop_value = self.advance().value
                    try:
                        prop_value = float(prop_value)
                    except:
                        pass
                elif self.check(TokenType.STRING_LITERAL):
                    prop_value = self.advance().value.strip('"\'')
                elif self.check(TokenType.PIPE):
                    # Quantum state notation like |0000>
                    self.advance()  # consume |
                    state_str = ""
                    
                    # Collect all tokens until we find >
                    while not self.check(TokenType.GREATER_THAN) and not self.is_at_end():
                        if self.check(TokenType.NUMBER_LITERAL):
                            state_str += str(self.advance().value)
                        elif self.check(TokenType.IDENTIFIER):
                            state_str += self.advance().value
                        elif self.check(TokenType.PLUS) or self.check(TokenType.MINUS):
                            # Handle superposition states like |+> or |->
                            state_str += self.advance().value
                        else:
                            # Any other character in the state
                            state_str += str(self.advance().value)
                    
                    if not state_str:
                        self.error("Expected state value after |")
                        state_str = "0"
                    
                    self.consume(TokenType.GREATER_THAN, "Expected '>' after state")
                    prop_value = f"|{state_str}>"
                elif self.check(TokenType.IDENTIFIER):
                    # Variable reference as property value
                    var_name = self.advance().value
                    # Look up variable value if it's a constant
                    if var_name in ["true", "True"]:
                        prop_value = True
                    elif var_name in ["false", "False"]:
                        prop_value = False
                    else:
                        # Use the variable name as a placeholder for now
                        prop_value = var_name
                else:
                    # Unknown property value type - use default
                    prop_value = 0
                    # Skip to next comma or closing brace
                    while not self.check(TokenType.COMMA) and not self.check(TokenType.RBRACE) and not self.is_at_end():
                        self.advance()
                
                properties[prop_name] = prop_value
                
                # Check if we're at the end of properties
                if self.check(TokenType.RBRACE):
                    break
                    
                # Otherwise expect a comma
                if not self.match(TokenType.COMMA):
                    # Allow missing comma before closing brace
                    if not self.check(TokenType.RBRACE):
                        self.error("Expected ',' or '}' after property")
                    break
                
                # Skip comments after comma
                while self.check(TokenType.COMMENT):
                    self.advance()
            
            self.consume(TokenType.RBRACE, "Expected '}' after properties")
        
        # Emit bytecode to create state
        name_const = self.context.module.add_constant(name)
        props_const = self.context.module.add_constant(properties)
        
        self.emit(OpCode.LOAD_CONST, name_const)
        self.emit(OpCode.LOAD_CONST, props_const)
        self.emit(OpCode.CREATE_STATE)
        
        # Consume semicolon
        self.consume(TokenType.SEMICOLON, "Expected ';' after state declaration")
    
    def parse_observer_declaration(self) -> None:
        """Parse observer declaration."""
        # observer name : type { properties }
        # OR: observer name { properties }  (legacy syntax)
        name = self.consume(TokenType.IDENTIFIER, "Expected observer name").value
        
        # Check for optional type specification
        observer_type = "consciousness"  # default type
        if self.match(TokenType.COLON):
            # New syntax with explicit type
            observer_type = self.consume(TokenType.IDENTIFIER, "Expected observer type").value
        
        # Parse properties
        properties = {'observer_type': observer_type}
        if self.match(TokenType.LBRACE):
            prop_iteration = 0
            while not self.check(TokenType.RBRACE) and not self.is_at_end():
                prop_iteration += 1
                if prop_iteration > 100:
                    logger.error(f"Too many properties in observer declaration at position {self.current}")
                    break
                    
                # Allow keywords or identifiers for property names
                if self.check(TokenType.IDENTIFIER):
                    prop_name = self.advance().value
                elif self.check(TokenType.KEYWORD):
                    prop_name = self.advance().value
                else:
                    self.error("Expected property name")
                    prop_name = "unknown"
                self.consume(TokenType.COLON, "Expected ':' after property name")
                
                # Property value
                if self.check(TokenType.NUMBER_LITERAL):
                    prop_value = self.advance().value
                    try:
                        prop_value = float(prop_value)
                    except:
                        pass
                elif self.check(TokenType.STRING_LITERAL):
                    prop_value = self.advance().value.strip('"\'')
                elif self.check(TokenType.IDENTIFIER):
                    # Variable reference as property value
                    var_name = self.advance().value
                    if var_name in ["true", "True"]:
                        prop_value = True
                    elif var_name in ["false", "False"]:
                        prop_value = False
                    else:
                        # Use the variable name as a placeholder
                        prop_value = var_name
                else:
                    # Unknown property value type - use default and skip tokens
                    prop_value = 0
                    logger.warning(f"Unknown property value type at position {self.current}, token: {self.peek()}")
                    # Skip to next comma or closing brace to avoid infinite loop
                    while not self.check(TokenType.COMMA) and not self.check(TokenType.RBRACE) and not self.is_at_end():
                        logger.debug(f"Skipping token: {self.peek()}")
                        self.advance()
                
                properties[prop_name] = prop_value
                
                # Check if we're at the end of properties
                if self.check(TokenType.RBRACE):
                    break
                    
                # Otherwise expect a comma
                if not self.match(TokenType.COMMA):
                    # Allow missing comma before closing brace
                    if not self.check(TokenType.RBRACE):
                        self.error("Expected ',' or '}' after property")
                    break
                
                # Skip comments after comma
                while self.check(TokenType.COMMENT):
                    self.advance()
            
            self.consume(TokenType.RBRACE, "Expected '}' after properties")
        
        # Emit bytecode
        name_const = self.context.module.add_constant(name)
        props_const = self.context.module.add_constant(properties)
        
        self.emit(OpCode.LOAD_CONST, name_const)
        self.emit(OpCode.LOAD_CONST, props_const)
        self.emit(OpCode.CREATE_OBSERVER)
        
        # Consume semicolon
        self.consume(TokenType.SEMICOLON, "Expected ';' after observer declaration")
    
    def parse_apply_statement(self) -> None:
        """Parse apply gate statement."""
        # apply GATE to state [, state2] [on qubits]
        # OR: apply observe observer_name to state
        
        logger.debug(f"[PARSER] parse_apply_statement at position {self.current}")
        
        # Check if next token is "observe"
        if self.check_keyword("observe"):
            self.advance()  # consume "observe"
            # Parse observer application
            observer_name = self.consume(TokenType.IDENTIFIER, "Expected observer name").value
            self.consume(TokenType.KEYWORD, "Expected 'to' after observer name")
            state_name = self.consume(TokenType.IDENTIFIER, "Expected state name").value
            
            # For now, treat observer application like a measurement
            state_const = self.context.module.add_constant(state_name)
            self.emit(OpCode.LOAD_CONST, state_const)
            self.emit(OpCode.MEASURE)
            
            # Apply statements do NOT have semicolons according to the grammar
            return
        
        gate_name = self.consume(TokenType.IDENTIFIER, "Expected gate name").value
        
        # Handle gate parameters if present
        params = None
        if self.match(TokenType.LPAREN):
            params = []
            while not self.check(TokenType.RPAREN):
                self.parse_expression()
                params.append("param")  # Placeholder
                if not self.match(TokenType.COMMA):
                    break
            self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        
        self.consume(TokenType.KEYWORD, "Expected 'to' after gate name")
        
        # Parse target state(s) and qubits
        targets = []
        target_qubits = []
        
        # Parse first target
        target_name = self.consume(TokenType.IDENTIFIER, "Expected target state").value
        targets.append(target_name)
        
        # Check if qubit is specified for this target
        qubit_list = []
        logger.debug(f"[PARSER] Checking for qubit specification, current token: {self.peek()}")
        if self.check_keyword("qubit") or self.check_keyword("qubits"):
            qubit_keyword = self.advance()  # consume "qubit" or "qubits"
            logger.debug(f"[PARSER] Found qubit keyword: {qubit_keyword.value}")
            # Parse qubit expression  
            if self.check(TokenType.NUMBER_LITERAL):
                qubit_idx = int(self.advance().value)
                qubit_list.append(qubit_idx)
                logger.debug(f"[PARSER] Parsed qubit index: {qubit_idx}")
            elif self.check(TokenType.LBRACKET):
                # Parse qubit list [0, 1, 2]
                logger.debug(f"[PARSER] Parsing qubit list")
                self.advance()  # consume '['
                qubit_list = []
                while not self.check(TokenType.RBRACKET):
                    if self.check(TokenType.NUMBER_LITERAL):
                        qubit_list.append(int(self.advance().value))
                    else:
                        # Parse expression for computed indices
                        self.parse_expression()
                        qubit_list.append(0)  # Placeholder - will be replaced by stack value
                    if not self.match(TokenType.COMMA):
                        break
                self.consume(TokenType.RBRACKET, "Expected ']' after qubit list")
                logger.debug(f"[PARSER] Parsed qubit list: {qubit_list}")
            else:
                logger.debug(f"[PARSER] Parsing qubit expression, current token: {self.peek()}")
                self.parse_expression()
                qubit_list.append(0)  # Placeholder - expression result will be used
        else:
            logger.debug(f"[PARSER] No qubit specification found, using default qubit 0")
            qubit_list.append(0)  # Default qubit 0
        
        # Check for multiple qubits on same state (for CNOT, etc)
        while self.match(TokenType.COMMA):
            # Peek ahead to see if this is the same state or a different one
            if self.check(TokenType.IDENTIFIER):
                next_token = self.peek()
                # Check if next identifier matches current state name
                if next_token and hasattr(next_token, 'value') and next_token.value == target_name:
                    # Same state, consume it and parse next qubit
                    self.advance()  # consume state name
                    if self.check_keyword("qubit"):
                        self.advance()  # consume "qubit"
                        if self.check(TokenType.NUMBER_LITERAL):
                            qubit_idx = int(self.advance().value)
                            qubit_list.append(qubit_idx)
                        else:
                            self.parse_expression()
                            qubit_list.append(0)  # Placeholder
                else:
                    # Different state - parse as separate target
                    target_name = self.advance().value
                    targets.append(target_name)
                    target_qubits.append(qubit_list)
                    qubit_list = []
                    
                    # Check if qubit is specified for this new target
                    if self.check_keyword("qubit"):
                        self.advance()  # consume "qubit"
                        if self.check(TokenType.NUMBER_LITERAL):
                            qubit_idx = int(self.advance().value)
                            qubit_list.append(qubit_idx)
                        else:
                            self.parse_expression()
                            qubit_list.append(0)  # Placeholder
                    else:
                        qubit_list.append(0)  # Default qubit 0
            else:
                break
        
        # Add the last qubit list
        target_qubits.append(qubit_list)
        
        # Check for "qubit N" syntax
        qubits = [0]  # Default to qubit 0
        control_qubits = []  # For controlled gates
        
        if self.check_keyword("qubit"):
            self.advance()  # consume "qubit"
            # Parse qubit index
            if self.check(TokenType.NUMBER_LITERAL):
                qubit_idx = int(self.advance().value)
                qubits = [qubit_idx]
            else:
                logger.warning("Expected qubit index after 'qubit' keyword")
            
            # Check for "control N" syntax for controlled gates
            if self.check_keyword("control"):
                self.advance()  # consume "control"
                if self.check(TokenType.NUMBER_LITERAL):
                    control_idx = int(self.advance().value)
                    control_qubits = [control_idx]
                    # For controlled gates, we need both control and target qubits
                    qubits = control_qubits + qubits
        
        # Also check for "on" keyword (alternative syntax)
        elif self.check_keyword("on"):
            self.advance()  # consume "on"
            # Parse qubit specification
            if self.check_keyword("qubits") or self.check_keyword("qubit"):
                self.advance()  # consume "qubits" or "qubit"
            # Parse qubit list/index
            if self.check(TokenType.NUMBER_LITERAL):
                qubit_idx = int(self.advance().value)
                qubits = [qubit_idx]
            elif self.check(TokenType.LBRACKET):
                # Parse qubit list [0, 1, 2]
                self.advance()  # consume '['
                qubits = []
                while not self.check(TokenType.RBRACKET):
                    if self.check(TokenType.NUMBER_LITERAL):
                        qubits.append(int(self.advance().value))
                    if not self.match(TokenType.COMMA):
                        break
                self.consume(TokenType.RBRACKET, "Expected ']' after qubit list")
        
        # Emit bytecode
        if len(targets) == 1:
            # Single target gate
            state_const = self.context.module.add_constant(targets[0])
            gate_const = self.context.module.add_constant(gate_name.upper())
            
            # Use the collected qubits from parsing
            if target_qubits and target_qubits[0]:
                qubits = target_qubits[0]
            
            qubits_const = self.context.module.add_constant(qubits)
            
            self.emit(OpCode.LOAD_CONST, state_const)
            self.emit(OpCode.LOAD_CONST, gate_const)
            self.emit(OpCode.LOAD_CONST, qubits_const)
            
            if params:
                params_const = self.context.module.add_constant(params)
                self.emit(OpCode.LOAD_CONST, params_const)
                self.emit(OpCode.APPLY_GATE, 3)  # 3 args + params
            else:
                self.emit(OpCode.APPLY_GATE, 2)  # 2 args
        else:
            # Multi-target gate (CNOT, TOFFOLI)
            # Handle as special case
            if gate_name.upper() in ['CNOT_GATE', 'CNOT', 'CX']:
                # CNOT: first is control, second is target
                if len(targets) == 2:
                    control_state = targets[0]
                    target_state = targets[1]
                    
                    # For CNOT between different states, we need to handle it specially
                    # The runtime expects: state_name, gate_name, qubits, params
                    # For now, apply CNOT to the target state with control from first state
                    
                    # First ensure states are entangled
                    control_const = self.context.module.add_constant(control_state)
                    target_const = self.context.module.add_constant(target_state)
                    
                    self.emit(OpCode.LOAD_CONST, control_const)
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.ENTANGLE)
                    
                    # Apply controlled operation on target state
                    # This is a simplification - proper CNOT needs cross-state operations
                    gate_const = self.context.module.add_constant('CX')
                    qubits_const = self.context.module.add_constant([0, 1])  # Control on 0, target on 1
                    
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.LOAD_CONST, gate_const)
                    self.emit(OpCode.LOAD_CONST, qubits_const)
                    self.emit(OpCode.APPLY_GATE, 2)
            elif gate_name.upper() in ['CZ_GATE', 'CZ']:
                # Controlled-Z gate
                if len(targets) == 2:
                    control_state = targets[0]
                    target_state = targets[1]
                    
                    # Ensure entanglement
                    control_const = self.context.module.add_constant(control_state)
                    target_const = self.context.module.add_constant(target_state)
                    
                    self.emit(OpCode.LOAD_CONST, control_const)
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.ENTANGLE)
                    
                    # Apply CZ gate
                    gate_const = self.context.module.add_constant('CZ')
                    qubits_const = self.context.module.add_constant([0, 1])
                    
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.LOAD_CONST, gate_const)
                    self.emit(OpCode.LOAD_CONST, qubits_const)
                    self.emit(OpCode.APPLY_GATE, 2)
                    
            elif gate_name.upper() in ['RZZ_GATE', 'RZZ', 'RXX_GATE', 'RXX', 'RYY_GATE', 'RYY']:
                # Two-qubit rotation gates
                if len(targets) == 2 and params:
                    control_state = targets[0]
                    target_state = targets[1]
                    
                    # Ensure entanglement
                    control_const = self.context.module.add_constant(control_state)
                    target_const = self.context.module.add_constant(target_state)
                    
                    self.emit(OpCode.LOAD_CONST, control_const)
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.ENTANGLE)
                    
                    # Apply rotation gate
                    gate_const = self.context.module.add_constant(gate_name.upper().replace('_GATE', ''))
                    qubits_const = self.context.module.add_constant([0, 1])
                    params_const = self.context.module.add_constant(params)
                    
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.LOAD_CONST, gate_const)
                    self.emit(OpCode.LOAD_CONST, qubits_const)
                    self.emit(OpCode.LOAD_CONST, params_const)
                    self.emit(OpCode.APPLY_GATE, 3)
                    
            elif gate_name.upper() in ['TOFFOLI_GATE', 'TOFFOLI', 'CCX']:
                # TOFFOLI: first two are controls, third is target
                if len(targets) == 3:
                    # For now, simplify as multiple CNOTs
                    control1_const = self.context.module.add_constant(targets[0])
                    control2_const = self.context.module.add_constant(targets[1])
                    target_const = self.context.module.add_constant(targets[2])
                    
                    # Entangle all three
                    self.emit(OpCode.LOAD_CONST, control1_const)
                    self.emit(OpCode.LOAD_CONST, control2_const)
                    self.emit(OpCode.ENTANGLE)
                    
                    self.emit(OpCode.LOAD_CONST, control2_const)
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.ENTANGLE)
                    
                    # Apply as CNOT for now
                    gate_const = self.context.module.add_constant('CNOT')
                    qubits_const = self.context.module.add_constant([0, 0])
                    
                    self.emit(OpCode.LOAD_CONST, target_const)
                    self.emit(OpCode.LOAD_CONST, gate_const)
                    self.emit(OpCode.LOAD_CONST, qubits_const)
                    self.emit(OpCode.APPLY_GATE, 2)
        
        # Apply statements do NOT have semicolons according to the grammar
    
    def parse_measure_statement(self) -> None:
        """Parse measure statement."""
        # measure state [qubit N] [by measurement_type] [into variable]
        # OR: measure qubit state_name  (alternative syntax)
        
        # Check if next token is 'qubit' for alternative syntax
        if self.check(TokenType.KEYWORD) and self.peek().value == "qubit":
            self.advance()  # consume 'qubit'
            state_name = self.consume(TokenType.IDENTIFIER, "Expected state name after 'qubit'").value
            qubit = 0  # Default to qubit 0 for this syntax
        else:
            # Standard syntax: measure state_name
            state_name = self.consume(TokenType.IDENTIFIER, "Expected state name").value
            
            # Check for qubit specification
            qubit = None
            if self.check(TokenType.KEYWORD) and self.peek().value == "qubit":
                self.advance()  # consume 'qubit'
                if self.check(TokenType.NUMBER_LITERAL):
                    qubit = int(self.advance().value)
                else:
                    self.error("Expected qubit index after 'qubit'")
        
        measurement_type = "standard"
        result_var = None
        
        # Check for 'by' clause
        if self.check_keyword("by"):
            self.advance()
            # Allow keywords or identifiers for measurement types
            if self.check(TokenType.IDENTIFIER):
                measurement_type = self.advance().value
            elif self.check(TokenType.KEYWORD):
                measurement_type = self.advance().value
            else:
                self.error("Expected measurement type")
                measurement_type = "standard"
        
        # Check for 'into' clause
        if self.check_keyword("into"):
            self.advance()
            # Allow keywords or identifiers for result variable
            if self.check(TokenType.IDENTIFIER):
                result_var = self.advance().value
            elif self.check(TokenType.KEYWORD):
                result_var = self.advance().value
            else:
                self.error("Expected variable name")
                result_var = "result"
        
        # Emit bytecode
        state_const = self.context.module.add_constant(state_name)
        self.emit(OpCode.LOAD_CONST, state_const)
        
        # Handle different measurement types
        if measurement_type in ["integrated_information", "ii"]:
            self.emit(OpCode.MEASURE_II, self.context.module.add_name(result_var) if result_var else -1)
        elif measurement_type in ["kolmogorov_complexity", "kc", "complexity"]:
            self.emit(OpCode.MEASURE_KC, self.context.module.add_name(result_var) if result_var else -1)
        elif measurement_type in ["entropy", "von_neumann_entropy"]:
            self.emit(OpCode.MEASURE_ENTROPY, self.context.module.add_name(result_var) if result_var else -1)
        elif measurement_type == "coherence":
            self.emit(OpCode.MEASURE_COHERENCE, self.context.module.add_name(result_var) if result_var else -1)
        elif measurement_type == "collapse_probability":
            self.emit(OpCode.MEASURE_COLLAPSE, self.context.module.add_name(result_var) if result_var else -1)
        elif measurement_type == "energy":
            # Energy measurement - use entropy as proxy for now
            self.emit(OpCode.MEASURE_ENTROPY, self.context.module.add_name(result_var) if result_var else -1)
        else:
            # For all other measurement types (including phi, rsp, etc), push the measurement type
            # and use generic MEASURE with type argument
            if measurement_type != "standard":
                # Push measurement type as constant
                type_const = self.context.module.add_constant(measurement_type)
                self.emit(OpCode.LOAD_CONST, type_const)
                # MEASURE will use 2 arguments (state, type)
                # Pass result_var index if specified, otherwise -1
                var_idx = self.context.module.add_name(result_var) if result_var else -1
                self.emit(OpCode.MEASURE, 2, var_idx)
            else:
                # Standard measurement
                if qubit is not None:
                    # Push qubit index for specific qubit measurement
                    qubit_const = self.context.module.add_constant(qubit)
                    self.emit(OpCode.LOAD_CONST, qubit_const)
                    self.emit(OpCode.MEASURE_QUBIT)
                else:
                    self.emit(OpCode.MEASURE)
            if result_var:
                var_idx = self.context.module.add_name(result_var)
                self.emit(OpCode.STORE_VAR, var_idx)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after measure statement")
    
    def parse_entangle_statement(self) -> None:
        """Parse entangle statement."""
        # entangle state1 with state2 or entangle state1, state2
        state1 = self.consume(TokenType.IDENTIFIER, "Expected first state").value
        
        # Check for 'with' keyword or comma
        if self.check(TokenType.KEYWORD) and self.peek().value == "with":
            self.advance()  # consume 'with'
        else:
            self.consume(TokenType.COMMA, "Expected ',' or 'with' between states")
        
        state2 = self.consume(TokenType.IDENTIFIER, "Expected second state").value
        
        # Emit bytecode
        state1_const = self.context.module.add_constant(state1)
        state2_const = self.context.module.add_constant(state2)
        
        self.emit(OpCode.LOAD_CONST, state1_const)
        self.emit(OpCode.LOAD_CONST, state2_const)
        self.emit(OpCode.ENTANGLE)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after entangle statement")
    
    def parse_recurse_statement(self) -> None:
        """Parse recurse statement for recursive simulation."""
        # recurse state_name depth N
        state_name = self.consume(TokenType.IDENTIFIER, "Expected state name").value
        
        # Check for 'depth' keyword
        if self.check(TokenType.KEYWORD) and self.peek().value == "depth":
            self.advance()  # consume 'depth'
        else:
            self.consume(TokenType.KEYWORD, "Expected 'depth' keyword")
        
        # Parse depth value
        if self.check(TokenType.NUMBER_LITERAL):
            depth = int(self.advance().value)
        else:
            # Default depth of 1
            depth = 1
        
        # Emit bytecode for recurse operation
        state_const = self.context.module.add_constant(state_name)
        depth_const = self.context.module.add_constant(depth)
        
        self.emit(OpCode.LOAD_CONST, state_const)
        self.emit(OpCode.LOAD_CONST, depth_const)
        self.emit(OpCode.RECURSE)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after recurse statement")
    
    def parse_cohere_statement(self) -> None:
        """Parse cohere statement."""
        # cohere StateName to level value
        state_name = self.consume(TokenType.IDENTIFIER, "Expected state name").value
        self.consume(TokenType.KEYWORD, "Expected 'to' keyword")
        
        # Optional 'level' keyword
        if self.check(TokenType.KEYWORD) and self.peek().value == "level":
            self.advance()  # consume 'level'
        
        # Parse the coherence level expression
        self.parse_expression()  # Level value is on stack
        
        # Emit bytecode
        state_const = self.context.module.add_constant(state_name)
        self.emit(OpCode.LOAD_CONST, state_const)
        self.emit(OpCode.SWAP)  # Put state name below level value
        self.emit(OpCode.COHERE)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after cohere statement")
    
    def parse_print_statement(self) -> None:
        """Parse print statement."""
        # print expression
        self.parse_expression()
        self.emit(OpCode.PRINT)
        self.consume(TokenType.SEMICOLON, "Expected ';' after print statement")
    
    def parse_evolve_statement(self) -> None:
        """Parse evolve statement."""
        # Support both syntaxes:
        # 1. evolve for N steps
        # 2. evolve target by time_step
        
        if self.check(TokenType.KEYWORD) and self.peek().value == "for":
            # evolve for N steps
            self.advance()  # consume 'for'
            
            # Parse number of steps
            self.parse_expression()  # steps on stack
            
            # Consume 'steps' or 'step' (can be either IDENTIFIER or KEYWORD)
            if self.check(TokenType.IDENTIFIER) or self.check(TokenType.KEYWORD):
                unit = self.peek().value
                if unit in ["steps", "step", "ticks", "tick", "cycles", "cycle"]:
                    self.advance()
            
            # System-wide evolution
            system_const = self.context.module.add_constant("__system__")
            self.emit(OpCode.LOAD_CONST, system_const)
            self.emit(OpCode.SWAP)  # Put system name below step count
            self.emit(OpCode.EVOLVE)
        else:
            # evolve target by time_step
            target = self.consume(TokenType.IDENTIFIER, "Expected target name").value
            
            # Optional 'by' clause
            time_step = 1.0  # Default time step
            if self.check_keyword("by"):
                self.advance()
                # Parse time step expression
                self.parse_expression()
            else:
                # Push default time step
                const_idx = self.context.module.add_constant(time_step)
                self.emit(OpCode.LOAD_CONST, const_idx)
            
            # Push target name
            target_const = self.context.module.add_constant(target)
            self.emit(OpCode.LOAD_CONST, target_const)
            
            # Emit evolve instruction
            self.emit(OpCode.EVOLVE)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after evolve statement")
    
    def parse_let_statement(self) -> None:
        """Parse let statement."""
        # let variable = expression
        # Allow keywords as variable names (like 'entropy')
        if self.check(TokenType.IDENTIFIER):
            var_name = self.advance().value
        elif self.check(TokenType.KEYWORD):
            # Allow certain keywords as variable names
            var_name = self.advance().value
        else:
            self.error("Expected variable name")
            var_name = "unknown"
        
        self.consume(TokenType.ASSIGN, "Expected '=' after variable name")
        
        # Parse expression
        self.parse_expression()
        
        # Store result
        var_idx = self.context.module.add_name(var_name)
        self.emit(OpCode.STORE_VAR, var_idx)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after let statement")
    
    def parse_const_statement(self) -> None:
        """Parse const statement (treated like let for now)."""
        # const variable = expression
        # Allow keywords as variable names
        if self.check(TokenType.IDENTIFIER):
            var_name = self.advance().value
        elif self.check(TokenType.KEYWORD):
            var_name = self.advance().value
        else:
            self.error("Expected constant name")
            var_name = "unknown"
        
        self.consume(TokenType.ASSIGN, "Expected '=' after constant name")
        
        # Parse expression  
        self.parse_expression()
        
        # Store result (same as let for now)
        var_idx = self.context.module.add_name(var_name)
        self.emit(OpCode.STORE_VAR, var_idx)
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after const statement")
    
    def parse_if_statement(self) -> None:
        """Parse if statement."""
        # if (condition) { body } [else { else_body }]
        self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
        self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")
        
        # Emit conditional jump
        jump_to_else = self.emit(OpCode.JUMP_IF_FALSE, 0)  # Placeholder
        
        # Parse then block
        self.consume(TokenType.LBRACE, "Expected '{' before if body")
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            # Skip comments before checking for closing brace
            while self.check(TokenType.COMMENT):
                self.advance()
            
            # Check again after skipping comments
            if self.check(TokenType.RBRACE):
                break
                
            self.parse_statement()
        self.consume(TokenType.RBRACE, "Expected '}' after if body")
        
        # Jump over else block
        jump_to_end = self.emit(OpCode.JUMP, 0)  # Placeholder
        
        # Patch jump to else
        self.context.module.instructions[jump_to_else].args[0] = len(self.context.module.instructions)
        
        # Parse else block if present
        if self.match_keyword("else"):
            # Check for 'else if'
            if self.check_keyword("if"):
                # Handle else if - advance past 'if' and continue as if statement
                self.advance()  # consume 'if'
                
                # Parse condition
                self.consume(TokenType.LPAREN, "Expected '(' after 'if'")
                self.parse_expression()
                self.consume(TokenType.RPAREN, "Expected ')' after condition")
                
                # Emit conditional jump for else if
                else_if_jump = self.emit(OpCode.JUMP_IF_FALSE, 0)
                
                # Parse else if body
                self.consume(TokenType.LBRACE, "Expected '{' before else if body")
                while not self.check(TokenType.RBRACE) and not self.is_at_end():
                    # Skip comments before checking for closing brace
                    while self.check(TokenType.COMMENT):
                        self.advance()
                    
                    # Check again after skipping comments
                    if self.check(TokenType.RBRACE):
                        break
                        
                    self.parse_statement()
                self.consume(TokenType.RBRACE, "Expected '}' after else if body")
                
                # Jump to end after else if body
                else_if_end = self.emit(OpCode.JUMP, 0)
                
                # Patch else if jump
                self.context.module.instructions[else_if_jump].args[0] = len(self.context.module.instructions)
                
                # Handle nested else/else if
                if self.match_keyword("else"):
                    if self.check_keyword("if"):
                        # Recursive else if
                        self.parse_statement()  # Will handle the if
                    else:
                        # Final else
                        self.consume(TokenType.LBRACE, "Expected '{' before else body")
                        while not self.check(TokenType.RBRACE) and not self.is_at_end():
                            # Skip comments before checking for closing brace
                            while self.check(TokenType.COMMENT):
                                self.advance()
                            
                            # Check again after skipping comments
                            if self.check(TokenType.RBRACE):
                                break
                                
                            self.parse_statement()
                        self.consume(TokenType.RBRACE, "Expected '}' after else body")
                
                # Patch else if end jump
                self.context.module.instructions[else_if_end].args[0] = len(self.context.module.instructions)
            else:
                # Regular else block
                self.consume(TokenType.LBRACE, "Expected '{' before else body")
                while not self.check(TokenType.RBRACE) and not self.is_at_end():
                    # Skip comments before checking for closing brace
                    while self.check(TokenType.COMMENT):
                        self.advance()
                    
                    # Check again after skipping comments
                    if self.check(TokenType.RBRACE):
                        break
                        
                    self.parse_statement()
                self.consume(TokenType.RBRACE, "Expected '}' after else body")
        
        # Patch jump to end
        self.context.module.instructions[jump_to_end].args[0] = len(self.context.module.instructions)
    
    def parse_while_statement(self) -> None:
        """Parse while statement."""
        # while (condition) { body }
        loop_start = len(self.context.module.instructions)
        
        self.consume(TokenType.LPAREN, "Expected '(' after 'while'")
        self.parse_expression()
        self.consume(TokenType.RPAREN, "Expected ')' after condition")
        
        # Conditional jump to end
        jump_to_end = self.emit(OpCode.JUMP_IF_FALSE, 0)  # Placeholder
        
        # Track loop for break/continue
        loop_end = 0  # Will be set later
        self.context.loop_stack.append((loop_start, loop_end))
        
        # Parse body
        self.consume(TokenType.LBRACE, "Expected '{' before while body")
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            # Skip comments before checking for closing brace
            while self.check(TokenType.COMMENT):
                self.advance()
            
            # Check again after skipping comments
            if self.check(TokenType.RBRACE):
                break
                
            self.parse_statement()
        self.consume(TokenType.RBRACE, "Expected '}' after while body")
        
        # Jump back to condition
        self.emit(OpCode.JUMP, loop_start)
        
        # Patch jump to end
        loop_end = len(self.context.module.instructions)
        self.context.module.instructions[jump_to_end].args[0] = loop_end
        
        # Update loop stack
        self.context.loop_stack.pop()
    
    def parse_for_statement(self) -> None:
        """Parse for statement."""
        # for variable from start to end [step value] { body }
        var_name = self.consume(TokenType.IDENTIFIER, "Expected loop variable").value
        
        self.consume_keyword("from")
        
        # Parse start value
        start_val = self.parse_simple_expression()
        
        self.consume_keyword("to")
        
        # Parse end value
        end_val = self.parse_simple_expression()
        
        # Check for optional step
        step_val = 1  # Default step
        if self.match_keyword("step"):
            step_val = self.parse_simple_expression()
        
        # Emit for loop setup
        start_const = self.context.module.add_constant(start_val)
        end_const = self.context.module.add_constant(end_val)
        step_const = self.context.module.add_constant(step_val)
        
        self.emit(OpCode.LOAD_CONST, start_const)
        self.emit(OpCode.LOAD_CONST, end_const)
        self.emit(OpCode.LOAD_CONST, step_const)  # Add step to stack
        self.emit(OpCode.FOR_SETUP)
        
        # Loop start
        loop_start = len(self.context.module.instructions)
        var_idx = self.context.module.add_name(var_name)
        
        # For iteration check
        loop_end = 0  # Placeholder
        for_iter_position = len(self.context.module.instructions)  # Capture position of FOR_ITER
        self.emit(OpCode.FOR_ITER, loop_end, var_idx)
        
        # Parse body
        self.consume(TokenType.LBRACE, "Expected '{' before for body")
        while not self.check(TokenType.RBRACE) and not self.is_at_end():
            # Skip comments before checking for closing brace
            while self.check(TokenType.COMMENT):
                self.advance()
            
            # Check again after skipping comments
            if self.check(TokenType.RBRACE):
                break
                
            self.parse_statement()
        self.consume(TokenType.RBRACE, "Expected '}' after for body")
        
        # Jump back to loop start
        self.emit(OpCode.JUMP, loop_start)
        
        # Patch loop end
        loop_end = len(self.context.module.instructions)
        self.context.module.instructions[for_iter_position].args[0] = loop_end
    
    def parse_simple_expression(self) -> Any:
        """Parse simple expression and return constant value."""
        if self.check(TokenType.NUMBER_LITERAL):
            # Convert to float for numeric operations
            value = self.advance().value
            try:
                return float(value)
            except:
                return value
        elif self.check(TokenType.STRING_LITERAL):
            return self.advance().value.strip('"\'')
        elif self.check(TokenType.IDENTIFIER):
            # For now, return a default value for identifiers
            self.advance()
            return 0
        else:
            self.error("Expected simple expression")
            return 0
    
    def parse_expression(self) -> None:
        """Parse expression and leave result on stack."""
        self.parse_or()
    
    def parse_or(self) -> None:
        """Parse logical OR expression."""
        self.parse_and()
        
        while self.match_operator("or"):
            self.parse_and()
            self.emit(OpCode.OR)
    
    def parse_and(self) -> None:
        """Parse logical AND expression."""
        self.parse_equality()
        
        while self.match_operator("and"):
            self.parse_equality()
            self.emit(OpCode.AND)
    
    def parse_equality(self) -> None:
        """Parse equality expression."""
        self.parse_comparison()
        
        while True:
            if self.match_operator("=="):
                self.parse_comparison()
                self.emit(OpCode.EQ)
            elif self.match_operator("!="):
                self.parse_comparison()
                self.emit(OpCode.NE)
            else:
                break
    
    def parse_comparison(self) -> None:
        """Parse comparison expression."""
        self.parse_term()
        
        while True:
            if self.match_operator("<"):
                self.parse_term()
                self.emit(OpCode.LT)
            elif self.match_operator("<="):
                self.parse_term()
                self.emit(OpCode.LE)
            elif self.match_operator(">"):
                self.parse_term()
                self.emit(OpCode.GT)
            elif self.match_operator(">="):
                self.parse_term()
                self.emit(OpCode.GE)
            else:
                break
    
    def parse_term(self) -> None:
        """Parse addition/subtraction expression."""
        self.parse_factor()
        
        while True:
            if self.match_operator("+"):
                self.parse_factor()
                self.emit(OpCode.ADD)
            elif self.match_operator("-"):
                self.parse_factor()
                self.emit(OpCode.SUB)
            else:
                break
    
    def parse_factor(self) -> None:
        """Parse multiplication/division expression."""
        self.parse_unary()
        
        while True:
            if self.match_operator("*"):
                self.parse_unary()
                self.emit(OpCode.MUL)
            elif self.match_operator("/"):
                self.parse_unary()
                self.emit(OpCode.DIV)
            elif self.match_operator("%"):
                self.parse_unary()
                self.emit(OpCode.MOD)
            else:
                break
    
    def parse_unary(self) -> None:
        """Parse unary expression."""
        if self.match_operator("-"):
            self.parse_unary()
            self.emit(OpCode.NEG)
        elif self.match_operator("not") or self.match_operator("!"):
            self.parse_unary()
            self.emit(OpCode.NOT)
        else:
            self.parse_primary()
    
    def parse_primary(self) -> None:
        """Parse primary expression."""
        if self.check(TokenType.NUMBER_LITERAL):
            value = self.advance().value
            # Convert to numeric type
            try:
                # Handle scientific notation
                if 'e' in value.lower():
                    value = float(value)
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except:
                # Fallback to float
                try:
                    value = float(value)
                except:
                    pass
            const_idx = self.context.module.add_constant(value)
            self.emit(OpCode.LOAD_CONST, const_idx)
        
        elif self.check(TokenType.STRING_LITERAL):
            value = self.advance().value.strip('"\'')
            const_idx = self.context.module.add_constant(value)
            self.emit(OpCode.LOAD_CONST, const_idx)
        
        elif self.check(TokenType.IDENTIFIER):
            name = self.advance().value
            
            # Check for function call or array indexing
            if self.match(TokenType.LPAREN):
                # Function call
                self.parse_function_call(name)
            elif self.match(TokenType.LBRACKET):
                # Array indexing
                var_idx = self.context.module.add_name(name)
                self.emit(OpCode.LOAD_VAR, var_idx)
                
                # Parse index expression
                self.parse_expression()
                self.consume(TokenType.RBRACKET, "Expected ']' after array index")
                
                # Get indexed item
                self.emit(OpCode.GET_ITEM)
            else:
                # Variable reference
                var_idx = self.context.module.add_name(name)
                self.emit(OpCode.LOAD_VAR, var_idx)
        
        elif self.check(TokenType.KEYWORD):
            # Some keywords can be function names or variables
            keyword = self.peek().value
            if keyword in ["log", "exp", "sqrt", "sin", "cos", "abs", "pow"]:
                # Treat as function call
                self.advance()
                if self.match(TokenType.LPAREN):
                    self.parse_function_call(keyword)
                else:
                    self.error(f"Expected '(' after function '{keyword}'")
            elif keyword == "measure":
                # Parse measure expression
                self.advance()  # consume 'measure'
                self.parse_measure_expression()
            else:
                # Allow keywords as variable references
                self.advance()
                var_idx = self.context.module.add_name(keyword)
                self.emit(OpCode.LOAD_VAR, var_idx)
        
        elif self.match(TokenType.LPAREN):
            # Parenthesized expression
            self.parse_expression()
            self.consume(TokenType.RPAREN, "Expected ')' after expression")
        
        elif self.match(TokenType.LBRACKET):
            # Array literal
            count = 0
            while not self.check(TokenType.RBRACKET) and not self.is_at_end():
                if count > 0:
                    self.consume(TokenType.COMMA, "Expected ',' between array elements")
                self.parse_expression()
                count += 1
            self.consume(TokenType.RBRACKET, "Expected ']' after array elements")
            
            # Build array from stack elements
            self.emit(OpCode.BUILD_LIST, count)
        
        else:
            self.error(f"Unexpected token in expression: {self.peek()}")
    
    def parse_measure_expression(self) -> None:
        """Parse measure as an expression (returns a value)."""
        # measure state [qubit N] by measurement_type
        state_name = self.consume(TokenType.IDENTIFIER, "Expected state name").value
        
        # Check for qubit specification
        qubit = None
        if self.check_keyword("qubit"):
            self.advance()  # consume 'qubit'
            if self.check(TokenType.NUMBER_LITERAL):
                qubit = int(self.advance().value)
            else:
                self.error("Expected qubit index after 'qubit'")
        
        measurement_type = "standard"
        
        # Check for 'by' clause
        if self.check_keyword("by"):
            self.advance()
            # Allow keywords or identifiers for measurement types
            if self.check(TokenType.IDENTIFIER):
                measurement_type = self.advance().value
            elif self.check(TokenType.KEYWORD):
                measurement_type = self.advance().value
            else:
                self.error("Expected measurement type")
                measurement_type = "standard"
        
        # Push state name
        state_idx = self.context.module.add_constant(state_name)
        self.emit(OpCode.LOAD_CONST, state_idx)
        
        # Push qubit index if specified
        if qubit is not None:
            qubit_idx = self.context.module.add_constant(qubit)
            self.emit(OpCode.LOAD_CONST, qubit_idx)
        else:
            # Push None for no specific qubit
            none_idx = self.context.module.add_constant(None)
            self.emit(OpCode.LOAD_CONST, none_idx)
        
        # Push measurement type
        type_idx = self.context.module.add_constant(measurement_type)
        self.emit(OpCode.LOAD_CONST, type_idx)
        
        # Emit measure operation (leaves result on stack)
        self.emit(OpCode.MEASURE)
    
    def parse_function_call(self, name: str) -> None:
        """Parse function call."""
        args = []
        
        while not self.check(TokenType.RPAREN):
            self.parse_expression()
            args.append(None)  # Placeholder
            if not self.match(TokenType.COMMA):
                break
        
        self.consume(TokenType.RPAREN, "Expected ')' after arguments")
        
        # Handle built-in functions
        if name == "sqrt":
            # Square root - use power of 0.5
            const_idx = self.context.module.add_constant(0.5)
            self.emit(OpCode.LOAD_CONST, const_idx)
            self.emit(OpCode.POW)
        elif name == "log":
            # Natural log - emit LOG opcode
            self.emit(OpCode.LOG)
        elif name == "abs":
            # Absolute value - emit ABS opcode
            self.emit(OpCode.ABS)
        elif name == "exp":
            # Exponential - for now return e^x approximation
            self.emit(OpCode.EXP)
        elif name == "pow":
            # Power function - x^y (y is on top of stack, x below)
            self.emit(OpCode.POW)
        elif name == "sin":
            # Sine function - for now push 0
            self.emit(OpCode.POP)  # Remove argument
            const_idx = self.context.module.add_constant(0.0)
            self.emit(OpCode.LOAD_CONST, const_idx)
        elif name == "cos":
            # Cosine function - for now push 1
            self.emit(OpCode.POP)  # Remove argument
            const_idx = self.context.module.add_constant(1.0)
            self.emit(OpCode.LOAD_CONST, const_idx)
        else:
            # User-defined function
            self.emit(OpCode.CALL, name, len(args))
    
    def parse_function_declaration(self) -> None:
        """Parse function declaration and compile to bytecode."""
        name = self.consume(TokenType.IDENTIFIER, "Expected function name").value
        self.consume(TokenType.LPAREN, "Expected '(' after function name")
        
        # Parse parameters
        params = []
        param_count = 0
        safety_counter = 0
        max_params = 100  # Reasonable limit for parameters
        
        while not self.check(TokenType.RPAREN) and not self.is_at_end():
            safety_counter += 1
            if safety_counter > max_params:
                logger.error(f"Too many parameters in function {name} (>{max_params})")
                # Skip to closing paren
                while not self.check(TokenType.RPAREN) and not self.is_at_end():
                    self.advance()
                break
                
            if param_count > 0:
                self.consume(TokenType.COMMA, "Expected ',' between parameters")
            
            # Skip any comments or whitespace tokens
            comment_count = 0
            while self.check(TokenType.COMMENT):
                self.advance()
                comment_count += 1
                if comment_count > 50:
                    break
                
            if self.check(TokenType.IDENTIFIER):
                param_name = self.advance().value
                params.append(param_name)
                param_count += 1
            else:
                # If we don't see an identifier, we might have hit a parsing issue
                # Skip to the next comma or closing paren
                skip_count = 0
                while not self.check(TokenType.COMMA) and not self.check(TokenType.RPAREN) and not self.is_at_end():
                    self.advance()
                    skip_count += 1
                    if skip_count > 50:
                        logger.warning(f"Skipping too many tokens in function {name} parameters")
                        break
        
        self.consume(TokenType.RPAREN, "Expected ')' after parameters")
        self.consume(TokenType.LBRACE, "Expected '{' before function body")
        
        # Store function start position
        func_start = len(self.context.module.instructions)
        
        # Add function to module
        func_idx = self.context.module.add_function(name, func_start, params)
        
        # Enter function scope
        self.context.function_stack.append(name)
        self.context.current_function = name
        
        # Parse function body
        brace_count = 1
        body_tokens = 0
        max_body_tokens = 10000  # Reasonable limit for function body
        
        while brace_count > 0 and not self.is_at_end():
            body_tokens += 1
            if body_tokens > max_body_tokens:
                logger.error(f"Function {name} body too large (>{max_body_tokens} tokens)")
                break
                
            # Parse statements in function body
            if self.check(TokenType.RBRACE):
                token = self.advance()
                brace_count -= 1
                if brace_count <= 0:
                    break
            elif self.check(TokenType.LBRACE):
                self.advance()
                brace_count += 1
            else:
                # Parse a statement
                try:
                    self.parse_statement()
                except Exception as e:
                    logger.error(f"Error parsing function {name} body: {e}")
                    # Skip to next semicolon or brace
                    while not self.check(TokenType.SEMICOLON) and not self.check(TokenType.RBRACE) and not self.is_at_end():
                        self.advance()
        
        # Add implicit return if needed
        if (not self.context.module.instructions or 
            self.context.module.instructions[-1].opcode != OpCode.RETURN):
            self.emit(OpCode.LOAD_CONST, self.context.module.add_constant(None))
            self.emit(OpCode.RETURN)
        
        # Exit function scope
        self.context.function_stack.pop()
        self.context.current_function = self.context.function_stack[-1] if self.context.function_stack else None
    
    def parse_return_statement(self) -> None:
        """Parse return statement."""
        # Check if we're in a function
        if self.context.current_function:
            # Parse return value expression
            if not self.check(TokenType.SEMICOLON):
                self.parse_expression()
            else:
                # No return value - push None
                const_idx = self.context.module.add_constant(None)
                self.emit(OpCode.LOAD_CONST, const_idx)
            
            # Emit return instruction
            self.emit(OpCode.RETURN)
        else:
            # Not in a function - skip the statement
            if not self.check(TokenType.SEMICOLON):
                # Skip the expression
                depth = 0
                while not self.check(TokenType.SEMICOLON) and not self.is_at_end():
                    if self.check(TokenType.LPAREN):
                        depth += 1
                    elif self.check(TokenType.RPAREN):
                        depth -= 1
                        if depth < 0:
                            break
                    self.advance()
        
        self.consume(TokenType.SEMICOLON, "Expected ';' after return")
    
    # Helper methods
    
    def match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types."""
        for token_type in types:
            if self.check(token_type):
                self.advance()
                return True
        return False
    
    def match_keyword(self, keyword: str) -> bool:
        """Check if current token is a specific keyword."""
        if self.check(TokenType.KEYWORD) and self.peek().value == keyword:
            self.advance()
            return True
        return False
    
    def match_operator(self, op: str) -> bool:
        """Check if current token is a specific operator."""
        # Map operator string to token type
        op_map = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.ASTERISK,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "==": TokenType.EQUALS,
            "!=": TokenType.NOT_EQUALS,
            "<": TokenType.LESS_THAN,
            "<=": TokenType.LESS_EQUALS,
            ">": TokenType.GREATER_THAN,
            ">=": TokenType.GREATER_EQUALS,
            "=": TokenType.ASSIGN,
            "&&": TokenType.LOGICAL_AND,
            "||": TokenType.LOGICAL_OR,
            "!": TokenType.LOGICAL_NOT
        }
        
        token_type = op_map.get(op)
        if token_type and self.check(token_type):
            self.advance()
            return True
        
        # Also check for keyword operators
        if op in ["and", "or", "not"] and self.check_keyword(op):
            self.advance()
            return True
            
        return False
    
    def check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type."""
        if self.is_at_end():
            return False
        return self.peek().type == token_type
    
    def check_keyword(self, keyword: str) -> bool:
        """Check if current token is a specific keyword."""
        return self.check(TokenType.KEYWORD) and self.peek().value == keyword
    
    def advance(self) -> Token:
        """Consume and return current token."""
        if not self.is_at_end():
            self.current += 1
        return self.previous()
    
    def is_at_end(self) -> bool:
        """Check if we're at end of tokens."""
        return self.current >= len(self.tokens) or self.peek().type == TokenType.EOF
    
    def peek(self, offset: int = 0) -> Token:
        """Return token at current position + offset without advancing."""
        pos = self.current + offset
        if 0 <= pos < len(self.tokens):
            return self.tokens[pos]
        return Token(TokenType.EOF, "", 0, 0, 0)
    
    def previous(self) -> Token:
        """Return previous token."""
        return self.tokens[self.current - 1]
    
    def consume(self, token_type: TokenType, message: str) -> Token:
        """Consume token of given type or raise error."""
        if self.check(token_type):
            return self.advance()
        
        self.error(message)
    
    def consume_keyword(self, keyword: str) -> Token:
        """Consume specific keyword or raise error."""
        if self.check_keyword(keyword):
            return self.advance()
        
        self.error(f"Expected keyword '{keyword}'")
        # Return dummy token to allow parsing to continue
        return Token(TokenType.KEYWORD, keyword, 0, 0, 0)
    
    def error(self, message: str) -> None:
        """Raise parse error."""
        token = self.peek()
        raise SyntaxError(f"{message} at line {token.line}, column {token.column}")
    
    def _resolve_jumps(self) -> None:
        """Resolve forward jump references."""
        # In a real implementation, we'd track forward jumps and patch them
        # For now, jumps are resolved during parsing
        pass