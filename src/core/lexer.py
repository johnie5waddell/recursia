from typing import List, Dict, Set, Optional, Tuple
from src.core.data_classes import LexerError, Token
from src.core.types import TokenType


class RecursiaLexer:
    """
    Complete lexer for the Recursia language based on the grammar specification.
    Converts source code into a stream of tokens for parsing.
    """
    
    # Reserved keywords in the language - stored as a set for O(1) lookups
    KEYWORDS: Set[str] = {
        # Core language keywords
        "universe", "state", "observer", "pattern", "apply", "render", "cohere", 
        "if", "when", "while", "for", "function", "return", "import", 
        "export", "let", "const", "measure", "entangle", "teleport",
        "hook", "visualize", "simulate", "align", "defragment", "print",
        "evolve",
        "log", "reset", "control", "params", "basis", "observe", "pow",
        "as", "with", "to", "from", "using", "in", "by", "until",
        "all", "any", "each", "group", "self", "null",
        "true", "false", "undefined", "Infinity", "NaN", "default",
        "public", "private", "protected", "internal", "at", "and", "or", 
        "xor", "implies", "iff", "not", "complex", "vec", "mat", "tensor",
        "statevec", "density", "anticontrol", "protocol", "into", 
        "steps", "ticks", "cycles", "basis", "formatted", "else", "elseif",
        "break", "continue", "scope", "focus", "target", "epoch", "level",
        "remove", "exists", "qubit", "qubits", "depth", "recurse",
        
        # Visualization keywords - CRITICAL additions
        "of", "entanglement", "network", "field", "evolution",
        "probability", "distribution", "wavefunction", "matrix", "bloch", 
        "sphere", "quantum", "trajectory", "circuit", "correlation", "between",
        
        # Basis types
        "standard_basis", "z_basis", "x_basis", "y_basis", "bell_basis", 
        "ghz_basis", "w_basis", "magic_basis", "computational_basis", 
        "hadamard_basis", "pauli_basis", "circular_basis",
        
        # Protocol types
        "standard_protocol", "dense_coding_protocol", "superdense_protocol",
        "entanglement_swapping_protocol", "quantum_repeater_protocol",
        "teleportation_circuit_protocol", "remote_state_preparation_protocol",
        "direct_protocol", "cnot_protocol", "hadamard_protocol", "epr_protocol",
        "ghz_protocol", "w_protocol", "cluster_protocol", "graph_state_protocol",
        "aklt_protocol", "kitaev_honeycomb_protocol", "tensor_network_protocol",
        
        # Simulation algorithms
        "quantum_trajectory_algorithm", "monte_carlo_algorithm", 
        "path_integral_algorithm", "tensor_network_algorithm",
        "density_matrix_algorithm", "quantum_walk_algorithm",
        "stochastic_process_algorithm", "quantum_cellular_automaton_algorithm",
        "wave_function_collapse_algorithm", "quantum_bayesian_algorithm",
        "neural_quantum_state_algorithm", "quantum_boltzmann_machine_algorithm",
        "quantum_metropolis_algorithm", "quantum_langevin_algorithm", 
        "lindblad_algorithm", "many_worlds_algorithm", 
        "consistent_histories_algorithm", "decoherent_histories_algorithm",
        
        # Coherence algorithms
        "quantum_annealing_algorithm", "stochastic_descent_algorithm",
        "gradient_descent_algorithm", "simulated_annealing_algorithm",
        "tensor_compression_algorithm", "density_matrix_evolution_algorithm",
        "variational_algorithm", "recursive_compression_algorithm",
        "holographic_projection_algorithm", "renormalization_group_algorithm",
        
        # Event types
        "state_creation_event", "state_destruction_event", "coherence_change_event",
        "entropy_increase_event", "entropy_decrease_event", "observation_event",
        "entanglement_creation_event", "entanglement_breaking_event",
        "teleportation_event", "measurement_event", "decoherence_event",
        "quantum_error_event", "stability_threshold_event", "collapse_event",
        "convergence_event", "divergence_event", "resonance_event", 
        "interference_event",
        
        # Log levels
        "debug_level", "info_level", "warning_level", "error_level", "critical_level",
        
        # Meta references
        "self_ref", "observer_ref", "system_ref",
        
        # Quantum expressions
        "bra", "ket", "braket", "expectation", "tensor_product", "trace",
        "partial_trace", "fidelity", "entropy", "purity", "schmidt_decomposition",
        "eigenvalues", "eigenvectors"
    }
    
    # Gate types as defined in the grammar
    GATE_TYPES: Set[str] = {
        "H_gate", "X_gate", "Y_gate", "Z_gate", "S_gate", "T_gate", "P_gate", "R_gate", 
        "RX_gate", "RY_gate", "RZ_gate", "U_gate", "U1_gate", "U2_gate", "U3_gate",
        "CNOT_gate", "CX_gate", "CZ_gate", "SWAP_gate", "CSWAP_gate", "TOFFOLI_gate",
        "CCNOT_gate", "Hadamard_gate", "PauliX_gate", "PauliY_gate", "PauliZ_gate",
        "PhaseShift_gate", "ControlledPhaseShift_gate", "ControlledZ_gate",
        "SqrtX_gate", "SqrtY_gate", "SqrtZ_gate", "SqrtW_gate", "SqrtNOT_gate",
        "AdjacentControlledPhaseShift_gate", "ControlledSWAP_gate",
        "QFT_gate", "InverseQFT_gate", "Oracle_gate", "Grover_gate", "Shor_gate",
        "VQE_gate", "QAOA_gate", "Trotter_gate", "RandomUnitary_gate",
        "Ising_gate", "Heisenberg_gate", "FermiHubbard_gate", "I_gate"
    }
    
    # Operator mappings
    OPERATORS: Dict[str, TokenType] = {
        '+': TokenType.PLUS,
        '-': TokenType.MINUS,
        '*': TokenType.ASTERISK,
        '/': TokenType.SLASH,
        '%': TokenType.PERCENT,
        '&': TokenType.AMPERSAND,
        '|': TokenType.PIPE,
        '^': TokenType.CARET,
        '!': TokenType.LOGICAL_NOT,
        '~': TokenType.LOGICAL_NOT,
        '<': TokenType.LESS_THAN,
        '>': TokenType.GREATER_THAN,
        '=': TokenType.ASSIGN,
        '?': TokenType.QUESTION,
        ':': TokenType.COLON,
        '.': TokenType.DOT,
        ',': TokenType.COMMA,
        ';': TokenType.SEMICOLON,
        '(': TokenType.LPAREN,
        ')': TokenType.RPAREN,
        '{': TokenType.LBRACE,
        '}': TokenType.RBRACE,
        '[': TokenType.LBRACKET,
        ']': TokenType.RBRACKET,
        '@': TokenType.AT,
        '#': TokenType.HASH,
    }
    
    # Multi-character operators - stored from longest to shortest to ensure we match the longest variant first
    MULTI_CHAR_OPERATORS: Dict[str, TokenType] = {
        '===': TokenType.STRICT_EQUALS,
        '!==': TokenType.STRICT_NOT_EQUALS,
        '**=': TokenType.POWER_ASSIGN,
        '<<=': TokenType.LSHIFT_ASSIGN,
        '>>=': TokenType.RSHIFT_ASSIGN,
        '>>>': TokenType.URSHIFT,
        '...': TokenType.TRIPLE_DOT,
        '++': TokenType.INCREMENT,
        '--': TokenType.DECREMENT,
        '**': TokenType.POWER,
        '==': TokenType.EQUALS,
        '!=': TokenType.NOT_EQUALS,
        '<=': TokenType.LESS_EQUALS,
        '>=': TokenType.GREATER_EQUALS,
        '&&': TokenType.LOGICAL_AND,
        '||': TokenType.LOGICAL_OR,
        '<<': TokenType.LSHIFT,
        '>>': TokenType.RSHIFT,
        '+=': TokenType.PLUS_ASSIGN,
        '-=': TokenType.MINUS_ASSIGN,
        '*=': TokenType.MULTIPLY_ASSIGN,
        '/=': TokenType.DIVIDE_ASSIGN,
        '%=': TokenType.MODULO_ASSIGN,
        '&=': TokenType.AND_ASSIGN,
        '|=': TokenType.OR_ASSIGN,
        '^=': TokenType.XOR_ASSIGN,
        '->': TokenType.ARROW,
        '..': TokenType.DOUBLE_DOT,
    }
    
    # Literal type mappings for quick lookup
    LITERAL_KEYWORDS: Dict[str, TokenType] = {
        "true": TokenType.BOOLEAN_LITERAL,
        "false": TokenType.BOOLEAN_LITERAL,
        "null": TokenType.NULL_LITERAL,
        "undefined": TokenType.UNDEFINED_LITERAL,
        "Infinity": TokenType.INFINITY_LITERAL,
        "+Infinity": TokenType.INFINITY_LITERAL,
        "-Infinity": TokenType.INFINITY_LITERAL,
        "NaN": TokenType.NAN_LITERAL
    }
    
    # Context markers for scanner
    CONTEXT_BOUNDARIES: Set[str] = {';', '{', '}', '\n'}
    
    def __init__(self, source: str):
        """
        Initialize the lexer with source code.
        
        Args:
            source: The source code to tokenize
        """
        self.source: str = source
        self.length: int = len(source)
        self.position: int = 0
        self.line: int = 1
        self.column: int = 1
        self.tokens: List[Token] = []
        self.current_token_start: int = 0
        self.current_token_start_line: int = 1
        self.current_token_start_column: int = 1
        self.error_recovery_mode: bool = False
    
    def is_at_end(self) -> bool:
        """Check if we've reached the end of the source"""
        return self.position >= self.length
    
    def peek(self, offset: int = 0) -> str:
        """
        Get the character at current position + offset without advancing.
        
        Args:
            offset: Offset from current position (default 0)
            
        Returns:
            Character at position + offset or empty string if out of bounds
        """
        if self.position + offset >= self.length or self.position + offset < 0:
            return ''
        return self.source[self.position + offset]
    
    def advance(self) -> str:
        """
        Consume the current character, update line/column, and return it.
        
        Returns:
            The consumed character
        """
        char = self.peek()
        self.position += 1
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return char
    
    def match(self, expected: str) -> bool:
        """
        Check if the current character matches the expected character and advance if it does.
        
        Args:
            expected: The character to match
            
        Returns:
            True if matched and advanced, False otherwise
        """
        if self.is_at_end() or self.peek() != expected:
            return False
        self.advance()
        return True
    
    def match_sequence(self, sequence: str) -> bool:
        """
        Check if the upcoming characters match the sequence and advance if they do.
        
        Args:
            sequence: The character sequence to match
            
        Returns:
            True if matched and advanced, False otherwise
        """
        if self.position + len(sequence) > self.length:
            return False
        
        # Check if the upcoming characters match the sequence
        for i, char in enumerate(sequence):
            if self.peek(i) != char:
                return False
        
        # If matched, advance by the length of the sequence
        for _ in sequence:
            self.advance()
        
        return True
    
    def is_digit(self, char: str) -> bool:
        """
        Check if a character is a digit (0-9).
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is a digit, False otherwise
        """
        return '0' <= char <= '9'
    
    def is_hex_digit(self, char: str) -> bool:
        """
        Check if a character is a hexadecimal digit (0-9, a-f, A-F).
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is a hex digit, False otherwise
        """
        return self.is_digit(char) or ('a' <= char.lower() <= 'f')
    
    def is_binary_digit(self, char: str) -> bool:
        """
        Check if a character is a binary digit (0 or 1).
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is a binary digit, False otherwise
        """
        return char == '0' or char == '1'
    
    def is_octal_digit(self, char: str) -> bool:
        """
        Check if a character is an octal digit (0-7).
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is an octal digit, False otherwise
        """
        return '0' <= char <= '7'
    
    def is_alpha(self, char: str) -> bool:
        """
        Check if a character is a letter.
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is a letter, False otherwise
        """
        return ('a' <= char.lower() <= 'z')
    
    def is_alphanumeric(self, char: str) -> bool:
        """
        Check if a character is a letter or digit.
        
        Args:
            char: The character to check
            
        Returns:
            True if the character is alphanumeric, False otherwise
        """
        return self.is_alpha(char) or self.is_digit(char)
    
    def is_identifier_start(self, char: str) -> bool:
        """
        Check if a character can start an identifier.
        
        Args:
            char: The character to check
            
        Returns:
            True if the character can start an identifier, False otherwise
        """
        return self.is_alpha(char) or char == '_' or char == '@'
    
    def is_identifier_part(self, char: str) -> bool:
        """
        Check if a character can be part of an identifier.
        
        Args:
            char: The character to check
            
        Returns:
            True if the character can be part of an identifier, False otherwise
        """
        return self.is_alphanumeric(char) or char == '_'
    
    def mark_token_start(self) -> None:
        """
        Mark the beginning of a new token.
        Updates internal state to track the start position of the current token.
        """
        self.current_token_start = self.position
        self.current_token_start_line = self.line
        self.current_token_start_column = self.column
    
    def add_token(self, type: TokenType, value: Optional[str] = None) -> None:
        """
        Add a token to the token list.
        
        Args:
            type: The token type
            value: The token value (Optional, will be extracted from source if None)
        """
        if value is None:
            # Get the lexeme from the source
            start = self.current_token_start
            end = self.position
            value = self.source[start:end]
        
        token = Token(
            type=type,
            value=value,
            line=self.current_token_start_line,
            column=self.current_token_start_column,
            length=len(value)
        )
        self.tokens.append(token)
    
    def scan_string(self) -> None:
        """
        Scan a string literal, handling escape sequences and template expressions.
        Supports single quotes, double quotes, and backtick template literals.
        """
        # Remember the quote character (' or " or `)
        quote = self.source[self.position - 1]
        
        # For template literals, we handle interpolation
        is_template = quote == '`'
        in_template_expr = False
        template_brace_count = 0
        
        # Buffer for building the string content
        string_content = quote  # Start with the opening quote
        
        while not self.is_at_end():
            char = self.peek()
            
            # Handle template expressions ${...}
            if is_template and char == '$' and self.peek(1) == '{' and not in_template_expr:
                string_content += self.advance()  # Add $
                string_content += self.advance()  # Add {
                in_template_expr = True
                template_brace_count = 1
                continue
            
            # Track braces inside template expressions
            if in_template_expr:
                if char == '{':
                    template_brace_count += 1
                elif char == '}':
                    template_brace_count -= 1
                    if template_brace_count == 0:
                        in_template_expr = False
                
                string_content += self.advance()
                continue
            
            # Handle string termination
            if char == quote and not in_template_expr:
                string_content += self.advance()  # Add the closing quote
                break
            
            # Handle escape sequences
            if char == '\\':
                string_content += self.advance()  # Add the backslash
                
                if not self.is_at_end():
                    string_content += self.advance()  # Add the escaped character
                
                continue
            
            # Handle newlines in strings
            if char == '\n':
                if quote != '`':  # Regular strings can't span lines
                    raise LexerError(
                        "Unterminated string literal", 
                        self.line, 
                        self.column, 
                        self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
                    )
            
            # Add the regular character
            string_content += self.advance()
        
        # Check if we ended because of EOF instead of finding the closing quote
        if self.is_at_end() and (not string_content or string_content[-1] != quote):
            raise LexerError(
                "Unterminated string literal", 
                self.line, 
                self.column,
                self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
            )
        
        # Add the token with the EXACT string content, including quotes
        token_type = TokenType.STRING_LITERAL
        self.add_token(token_type, string_content)
    
    def scan_number(self) -> None:
        """
        Scan a number literal (integer, decimal, hex, binary, or octal).
        Handles various numeric formats including scientific notation.
        """
        # Initial state tracking
        is_decimal = False
        has_exponent = False
        number_format = "decimal"
        
        # Check for hex, binary, or octal literals
        if self.peek(-1) == '0':
            if self.match('x') or self.match('X'):
                # Hexadecimal
                number_format = "hex"
                while not self.is_at_end() and self.is_hex_digit(self.peek()):
                    self.advance()
                    
                # Ensure we have at least one hex digit
                if self.peek(-1) in ('x', 'X'):
                    raise LexerError(
                        "Invalid hexadecimal number - missing digits", 
                        self.line, self.column,
                        self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
                    )
                
                self.add_token(TokenType.NUMBER_LITERAL)
                return
            elif self.match('b') or self.match('B'):
                # Binary
                number_format = "binary"
                digit_count = 0
                while not self.is_at_end() and self.is_binary_digit(self.peek()):
                    self.advance()
                    digit_count += 1
                
                # Ensure we have at least one binary digit
                if digit_count == 0:
                    raise LexerError(
                        "Invalid binary number - missing digits", 
                        self.line, self.column,
                        self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
                    )
                
                self.add_token(TokenType.NUMBER_LITERAL)
                return
            elif self.match('o') or self.match('O'):
                # Octal
                number_format = "octal"
                digit_count = 0
                while not self.is_at_end() and self.is_octal_digit(self.peek()):
                    self.advance()
                    digit_count += 1
                
                # Ensure we have at least one octal digit
                if digit_count == 0:
                    raise LexerError(
                        "Invalid octal number - missing digits", 
                        self.line, self.column,
                        self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
                    )
                
                self.add_token(TokenType.NUMBER_LITERAL)
                return
        
        # Regular decimal number
        while not self.is_at_end() and self.is_digit(self.peek()):
            self.advance()
        
        # Handle decimal point
        if not self.is_at_end() and self.peek() == '.':
            # Only treat as decimal point if followed by a digit
            if not self.is_at_end() and self.is_digit(self.peek(1)):
                self.advance()  # Consume the '.'
                is_decimal = True
                
                # Consume all digits after the decimal point
                while not self.is_at_end() and self.is_digit(self.peek()):
                    self.advance()
        
        # Handle exponent
        if not self.is_at_end() and (self.peek() == 'e' or self.peek() == 'E'):
            # Check if we have digits after potential exponent marker
            if self.is_at_end() or (self.peek(1) != '+' and self.peek(1) != '-' and not self.is_digit(self.peek(1))):
                # End of number, not a valid exponent
                self.add_token(TokenType.NUMBER_LITERAL)
                return
                
            # Valid exponent format
            self.advance()  # Consume the 'e' or 'E'
            has_exponent = True
            
            # Handle optional sign
            if self.match('+') or self.match('-'):
                pass  # Just consume the sign character
            
            # Make sure we have at least one digit in the exponent
            if self.is_at_end() or not self.is_digit(self.peek()):
                raise LexerError(
                    "Expected digit in exponent", 
                    self.line, 
                    self.column,
                    self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
                )
            
            # Consume all digits in the exponent
            while not self.is_at_end() and self.is_digit(self.peek()):
                self.advance()
        
        self.add_token(TokenType.NUMBER_LITERAL)
    
    def scan_identifier_or_keyword(self) -> None:
        """
        Scan an identifier or keyword.
        Handles normal identifiers, escaped identifiers, keywords, and literals.
        """
        # Handle @ escaping for identifiers
        is_escaped = self.peek(-1) == '@'
        
        # Consume all valid identifier characters
        while not self.is_at_end() and self.is_identifier_part(self.peek()):
            self.advance()
        
        # Get the identifier text
        identifier = self.source[self.current_token_start:self.position]
        
        # If it's an escaped identifier (with @), always treat it as an identifier
        if is_escaped:
            self.add_token(TokenType.IDENTIFIER, identifier)
            return
        
        # Check for literals
        if identifier in self.LITERAL_KEYWORDS:
            self.add_token(self.LITERAL_KEYWORDS[identifier], identifier)
            return
        
        # Check if it's a reserved keyword
        if identifier in self.KEYWORDS:
            self.add_token(TokenType.KEYWORD, identifier)
            return
            
        # Check if it's a gate type (treated as a special kind of identifier)
        if identifier in self.GATE_TYPES:
            self.add_token(TokenType.IDENTIFIER, identifier)  # Could use a special GateType token if needed
            return
            
        # Regular identifier
        self.add_token(TokenType.IDENTIFIER, identifier)
    
    def scan_comment(self) -> None:
        """
        Scan a comment.
        Handles single-line comments starting with '//'
        """
        # Consume characters until the end of line or end of file
        while not self.is_at_end() and self.peek() != '\n':
            self.advance()
        
        # Add the comment token
        self.add_token(TokenType.COMMENT)
    
    def scan_directive(self) -> None:
        """
        Scan a directive (starts with #).
        Handles Recursia compiler directives like #target, #optimize, etc.
        """
        # Consume the identifier part of the directive
        while not self.is_at_end() and self.is_identifier_part(self.peek()):
            self.advance()
        
        # Skip whitespace before potential parameters
        while not self.is_at_end() and self.peek().isspace() and self.peek() != '\n':
            self.advance()
        
        # Check for directive parameters
        if self.match('('):
            # Consume everything until the matching closing parenthesis
            open_parens = 1
            while not self.is_at_end() and open_parens > 0:
                if self.peek() == '(':
                    open_parens += 1
                elif self.peek() == ')':
                    open_parens -= 1
                
                # Error on newline in directive parameters
                if self.peek() == '\n' and open_parens > 0:
                    raise LexerError(
                        "Unterminated directive parameters - unclosed parenthesis", 
                        self.line, 
                        self.column,
                        self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
                    )
                
                self.advance()
        
        self.add_token(TokenType.DIRECTIVE)
    
    def scan_operator(self) -> bool:
        """
        Attempt to scan a multi-character operator.
        
        Returns:
            True if a multi-character operator was successfully scanned, False otherwise
        """
        # Try to match operators from longest to shortest
        max_op_length = min(5, self.length - self.position + 1)
        
        for length in range(max_op_length, 1, -1):
            if self.position - 1 + length <= self.length:
                potential_op = self.source[self.position-1:self.position-1+length]
                if potential_op in self.MULTI_CHAR_OPERATORS:
                    # Advance to consume the rest of the operator
                    for _ in range(length - 1):  # -1 because we've already consumed one char
                        self.advance()
                    
                    # Use RANGE token if it's '..' and we're in a context where ranges are expected
                    if potential_op == '..' and self._is_in_range_context():
                        self.add_token(TokenType.RANGE)
                    else:
                        self.add_token(self.MULTI_CHAR_OPERATORS[potential_op])
                    return True
        
        return False
    
    def scan_token(self) -> None:
        """
        Scan a single token.
        Main dispatch method that calls specific scanners based on the current character.
        """
        # Mark the start of this token before advancing
        self.mark_token_start()
        
        char = self.advance()
        
        # Handle whitespace (ignore)
        if char.isspace():
            return
        
        # Handle comments
        if char == '/' and self.peek() == '/':
            self.scan_comment()
            return
        
        # Handle strings
        if char == '"' or char == "'" or char == '`':
            self.scan_string()
            return
        
        # Handle numbers
        if self.is_digit(char):
            self.scan_number()
            return
        
        # Handle identifiers and keywords
        if self.is_identifier_start(char):
            self.scan_identifier_or_keyword()
            return
        
        # Handle directives
        if char == '#':
            self.scan_directive()
            return
        
        if char == '~':
            self.add_token(TokenType.BIT_NOT)
            return

        # Try to match multi-character operators
        if self.scan_operator():
            return
        
        # Handle single-character operators and symbols
        if char in self.OPERATORS:
            self.add_token(self.OPERATORS[char])
            return
        
        # If we get here, the character wasn't recognized
        raise LexerError(
            f"Unexpected character: '{char}'", 
            self.line, 
            self.column, 
            self.source.splitlines()[self.line - 1] if self.line - 1 < len(self.source.splitlines()) else ""
        )

    def _is_in_range_context(self) -> bool:
        """
        Determine if the '..' operator should be treated as a range operator based on context.
        Looks at surrounding tokens/chars to determine if we're in a range expression.
        
        Returns:
            True if in a range context, False otherwise
        """
        # Check for digit or identifier before
        has_valid_left = False
        left_pos = self.current_token_start - 1
        
        # Skip whitespace looking backward
        while left_pos >= 0 and self.source[left_pos].isspace():
            left_pos -= 1
        
        # Check what's before
        if left_pos >= 0:
            left_char = self.source[left_pos]
            if self.is_digit(left_char) or self.is_identifier_part(left_char) or left_char == ')' or left_char == ']':
                has_valid_left = True
        
        if not has_valid_left:
            return False
        
        # Check for digit, identifier, or expression start after
        has_valid_right = False
        right_pos = self.position
        
        # Skip whitespace looking forward
        while right_pos < self.length and self.source[right_pos].isspace():
            right_pos += 1
        
        # Check what's after
        if right_pos < self.length:
            right_char = self.source[right_pos]
            if self.is_digit(right_char) or self.is_identifier_part(right_char) or right_char == '(' or right_char == '-':
                has_valid_right = True
        
        return has_valid_left and has_valid_right

    def _get_token_type_at(self, position: int) -> Optional[TokenType]:
        """
        Get an approximate token type at a specific position in the source.
        Used for context-sensitive lexing decisions.
        
        Args:
            position: The source position to check
            
        Returns:
            The estimated TokenType at that position, or None if no match
        """
        if position < 0 or position >= self.length:
            return None
        
        char = self.source[position]
        
        # Check if we're at the end of a token
        if char.isspace():
            # Look back to find a non-whitespace character
            temp_pos = position - 1
            while temp_pos >= 0 and self.source[temp_pos].isspace():
                temp_pos -= 1
            
            if temp_pos < 0:
                return None
            
            char = self.source[temp_pos]
        
        # Attempt token type identification
        if self.is_digit(char):
            return TokenType.NUMBER_LITERAL
        elif char in ('"', "'", '`'):
            return TokenType.STRING_LITERAL
        elif self.is_alpha(char) or char == '_' or char == '@':
            # Check if it's part of a keyword
            word_start = position
            while word_start > 0 and self.is_identifier_part(self.source[word_start-1]):
                word_start -= 1
            
            # Find the word end
            word_end = position
            while word_end < self.length-1 and self.is_identifier_part(self.source[word_end+1]):
                word_end += 1
            
            word = self.source[word_start:word_end+1]
            
            if word in self.KEYWORDS:
                return TokenType.KEYWORD
            elif word in self.GATE_TYPES:
                return TokenType.IDENTIFIER  # Or specialized GateType if implemented
            else:
                return TokenType.IDENTIFIER
        elif char in self.OPERATORS:
            return self.OPERATORS[char]
        elif char == '#':
            return TokenType.DIRECTIVE
        elif char == ';':
            return TokenType.SEMICOLON
        
        # Multi-character operators check
        for length in range(min(3, self.length - position), 0, -1):
            if position + length <= self.length:
                potential_op = self.source[position:position+length]
                if potential_op in self.MULTI_CHAR_OPERATORS:
                    return self.MULTI_CHAR_OPERATORS[potential_op]
        
        return None
    
    def recover_from_error(self) -> None:
        """
        Attempt to recover from a lexical error by advancing to a recognizable
        boundary like a semicolon, newline, or block delimiter.
        """
        self.error_recovery_mode = True
        
        # Advance until we reach something that looks like a token boundary
        while not self.is_at_end():
            if any(self.peek() == boundary for boundary in self.CONTEXT_BOUNDARIES):
                # Consume the boundary character
                self.advance()
                break
            
            # Also stop at operators that might indicate statement boundaries
            if self.peek() in self.OPERATORS:
                break
                
            self.advance()
            
        self.error_recovery_mode = False
    
    def tokenize(self) -> List[Token]:
        """
        Convert the source code into a list of tokens.
        Handles errors gracefully, adding error tokens and continuing.
        
        Returns:
            List[Token]: The complete list of tokens
        """
        while not self.is_at_end():
            try:
                # Mark the beginning of this token
                self.mark_token_start()
                self.scan_token()
            except LexerError as e:
                # Create an error token with the error message
                self.add_token(TokenType.ERROR, str(e))
                
                # Try to recover from the error
                self.recover_from_error()
        
        # Add EOF token at the end
        self.mark_token_start()
        self.add_token(TokenType.EOF, "")
        
        return self.tokens
    
    def debug_token_stream(self) -> str:
        """
        Returns a formatted string representation of the token stream.
        Useful for debugging lexer output.
        
        Returns:
            str: A formatted token stream
        """
        result = []
        current_line = -1
        line_tokens = []
        
        for token in self.tokens:
            if token.line != current_line:
                if line_tokens:
                    result.append(f"Line {current_line}: {' '.join(line_tokens)}")
                    line_tokens = []
                current_line = token.line
            
            if token.type == TokenType.EOF:
                token_repr = "EOF"
            else:
                token_repr = f"{token.type.name}('{token.value}')"
            
            line_tokens.append(token_repr)
        
        if line_tokens:
            result.append(f"Line {current_line}: {' '.join(line_tokens)}")
        
        return "\n".join(result)

    @staticmethod
    def highlight_syntax(code: str) -> str:
        """
        Highlight syntax for the given code string.
        A utility method for visualizing the lexer results with ANSI colors.
        
        Args:
            code: The code string to highlight
            
        Returns:
            str: The highlighted code with ANSI color codes
        """
        lexer = RecursiaLexer(code)
        tokens = lexer.tokenize()
        
        # Create a list of (start, end, type) tuples for each token
        token_ranges = []
        for token in tokens:
            if token.type != TokenType.EOF:
                start_pos = lexer.source.rfind('\n', 0, token.column) + 1 if token.column > 0 else 0
                start_pos = start_pos + token.column - 1
                end_pos = start_pos + len(token.value)
                token_ranges.append((start_pos, end_pos, token.type))
        
        # Sort by start position
        token_ranges.sort()
        
        # ANSI color codes
        COLORS = {
            TokenType.KEYWORD: "\033[1;36m",      # Cyan
            TokenType.IDENTIFIER: "\033[0;37m",   # White
            TokenType.STRING_LITERAL: "\033[0;32m",  # Green
            TokenType.NUMBER_LITERAL: "\033[0;33m",  # Yellow
            TokenType.BOOLEAN_LITERAL: "\033[0;35m", # Magenta
            TokenType.NULL_LITERAL: "\033[0;35m",    # Magenta
            TokenType.COMMENT: "\033[0;37;2m",    # Dim white
            TokenType.DIRECTIVE: "\033[0;34m",    # Blue
            TokenType.ERROR: "\033[0;31m",        # Red
        }
        DEFAULT_COLOR = "\033[0m"  # Reset
        
        # Build the highlighted string
        result = []
        current_pos = 0
        
        for start, end, token_type in token_ranges:
            # Add text before this token
            if start > current_pos:
                result.append(code[current_pos:start])
            
            # Add the highlighted token
            token_text = code[start:end]
            color = COLORS.get(token_type, DEFAULT_COLOR)
            result.append(f"{color}{token_text}{DEFAULT_COLOR}")
            
            current_pos = end
        
        # Add any remaining text
        if current_pos < len(code):
            result.append(code[current_pos:])
        
        return "".join(result)