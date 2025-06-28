from enum import Enum, auto

class TokenType(Enum):
    """Token types for the Recursia language lexer"""
    # Special tokens
    EOF = auto()
    ERROR = auto()
    
    # Structural elements
    IDENTIFIER = auto()
    KEYWORD = auto()
    DIRECTIVE = auto()
    
    # Literals
    STRING_LITERAL = auto()
    NUMBER_LITERAL = auto()
    BOOLEAN_LITERAL = auto()
    NULL_LITERAL = auto()
    UNDEFINED_LITERAL = auto()
    INFINITY_LITERAL = auto()
    NAN_LITERAL = auto()
    
    # Operators
    PLUS = auto()           # +
    MINUS = auto()          # -
    ASTERISK = auto()       # *
    SLASH = auto()          # /
    PERCENT = auto()        # %
    POWER = auto()          # **
    AMPERSAND = auto()      # &
    PIPE = auto()           # |
    CARET = auto()          # ^
    BIT_NOT = auto()        # ~
    LSHIFT = auto()         # 
    RSHIFT = auto()         # >>
    URSHIFT = auto()        # >>>
    
    # Comparison operators
    EQUALS = auto()         # ==
    NOT_EQUALS = auto()     # !=
    STRICT_EQUALS = auto()  # ===
    STRICT_NOT_EQUALS = auto() # !==
    LESS_THAN = auto()      # 
    LESS_EQUALS = auto()    # <=
    GREATER_THAN = auto()   # >
    GREATER_EQUALS = auto() # >=
    
    # Assignment operators
    ASSIGN = auto()         # =
    PLUS_ASSIGN = auto()    # +=
    MINUS_ASSIGN = auto()   # -=
    MULTIPLY_ASSIGN = auto() # *=
    DIVIDE_ASSIGN = auto()  # /=
    MODULO_ASSIGN = auto()  # %=
    POWER_ASSIGN = auto()   # **=
    LSHIFT_ASSIGN = auto()  # <<=
    RSHIFT_ASSIGN = auto()  # >>=
    AND_ASSIGN = auto()     # &=
    OR_ASSIGN = auto()      # |=
    XOR_ASSIGN = auto()     # ^=
    
    # Logical operators
    LOGICAL_AND = auto()    # &&
    LOGICAL_OR = auto()     # ||
    LOGICAL_NOT = auto()    # !
    
    # Increment/decrement
    INCREMENT = auto()      # ++
    DECREMENT = auto()      # --
    
    # Delimiters
    LPAREN = auto()         # (
    RPAREN = auto()         # )
    LBRACE = auto()         # {
    RBRACE = auto()         # }
    LBRACKET = auto()       # [
    RBRACKET = auto()       # ]
    COMMA = auto()          # ,
    DOT = auto()            # .
    SEMICOLON = auto()      # ;
    COLON = auto()          # :
    QUESTION = auto()       # ?
    AT = auto()             # @
    HASH = auto()           # #
    
    # Quantum specific
    ARROW = auto()          # ->
    DOUBLE_DOT = auto()     # ..
    TRIPLE_DOT = auto()     # ...
    RANGE = auto()  

    # Comments
    COMMENT = auto()
    
    # Language keywords
    STATE = auto()
    OBSERVER = auto()
    PATTERN = auto()
    APPLY = auto()
    RENDER = auto()
    COHERE = auto()
    IF = auto()
    WHEN = auto()
    WHILE = auto()
    FOR = auto()
    FUNCTION = auto()
    RETURN = auto()
    IMPORT = auto()
    EXPORT = auto()
    LET = auto()
    CONST = auto()
    MEASURE = auto()
    ENTANGLE = auto()
    TELEPORT = auto()
    HOOK = auto()
    VISUALIZE = auto()
    SIMULATE = auto()
    ALIGN = auto()
    DEFRAGMENT = auto()
    PRINT = auto()
    LOG = auto()
    RESET = auto()
    QUBIT = auto()
    QUBITS = auto()
    CONTROL = auto()
    PARAMS = auto()
    BASIS = auto()
    AS = auto()
    WITH = auto()
    TO = auto()
    FROM = auto()
    USING = auto()
    IN = auto()
    BY = auto()
    UNTIL = auto()
    ALL = auto()
    ANY = auto()
    EACH = auto()
    GROUP = auto()
    SELF = auto()
    SYSTEM = auto()
    NULL = auto()
    TRUE = auto()
    FALSE = auto()
    UNDEFINED = auto()
    INFINITY = auto()
    NAN = auto()
    DEFAULT = auto()
    PUBLIC = auto()
    PRIVATE = auto()
    PROTECTED = auto()
    INTERNAL = auto()
    
    # Visualization keywords
    ENTANGLEMENT = auto()
    NETWORK = auto()
    COHERENCE = auto()
    FIELD = auto()
    EVOLUTION = auto()
    PROBABILITY = auto()
    DISTRIBUTION = auto()
    WAVEFUNCTION = auto()
    DENSITY = auto()
    MATRIX = auto()
    QUANTUM = auto()
    CIRCUIT = auto()
    TRAJECTORY = auto()
    BLOCH = auto()
    SPHERE = auto()
    CORRELATION = auto()
    BETWEEN = auto()
    AND = auto()
    OF = auto()
    
    # Quantum gate types
    H_GATE = auto()
    X_GATE = auto()
    Y_GATE = auto()
    Z_GATE = auto()
    S_GATE = auto()
    T_GATE = auto()
    P_GATE = auto()
    R_GATE = auto()
    RX_GATE = auto()
    RY_GATE = auto()
    RZ_GATE = auto()
    U_GATE = auto()
    U1_GATE = auto()
    U2_GATE = auto()
    U3_GATE = auto()
    CNOT_GATE = auto()
    CX_GATE = auto()
    CY_GATE = auto()
    CZ_GATE = auto()
    SWAP_GATE = auto()
    CSWAP_GATE = auto()
    TOFFOLI_GATE = auto()
    CCNOT_GATE = auto()
    HADAMARD_GATE = auto()
    PAULIX_GATE = auto()
    PAULIY_GATE = auto()
    PAULIZ_GATE = auto()
    PHASESHIFT_GATE = auto()
    CONTROLLEDPHASESHIFT_GATE = auto()
    CONTROLLEDZ_GATE = auto()
    SQRTX_GATE = auto()
    SQRTY_GATE = auto()
    SQRTZ_GATE = auto()
    SQRTW_GATE = auto()
    SQRTNOT_GATE = auto()
    ADJACENTCONTROLLEDPHASESHIFT_GATE = auto()
    CONTROLLEDSWAP_GATE = auto()
    QFT_GATE = auto()
    INVERSEQFT_GATE = auto()
    ORACLE_GATE = auto()
    GROVER_GATE = auto()
    SHOR_GATE = auto()
    VQE_GATE = auto()
    QAOA_GATE = auto()
    TROTTER_GATE = auto()
    RANDOMUNITARY_GATE = auto()
    ISING_GATE = auto()
    HEISENBERG_GATE = auto()
    FERMIHUBBARD_GATE = auto()
    
    # Logical operators (text form)
    AND_OP = auto()         # and
    OR_OP = auto()          # or
    XOR_OP = auto()         # xor
    NOT_OP = auto()         # not
    IMPLIES = auto()        # implies
    IFF = auto()            # iff
    
    # Control flow
    ELSE = auto()
    ELSEIF = auto()
    BREAK = auto()
    CONTINUE = auto()
    
    # Scope and visibility
    SCOPE = auto()
    PHASE = auto()
    FOCUS = auto()
    TARGET = auto()
    
    # Measurement and basis types
    STANDARD_BASIS = auto()
    Z_BASIS = auto()
    X_BASIS = auto()
    Y_BASIS = auto()
    BELL_BASIS = auto()
    GHZ_BASIS = auto()
    W_BASIS = auto()
    MAGIC_BASIS = auto()
    COMPUTATIONAL_BASIS = auto()
    HADAMARD_BASIS = auto()
    PAULI_BASIS = auto()
    CIRCULAR_BASIS = auto()
    
    # Protocol types
    STANDARD_PROTOCOL = auto()
    DENSE_CODING_PROTOCOL = auto()
    SUPERDENSE_PROTOCOL = auto()
    ENTANGLEMENT_SWAPPING_PROTOCOL = auto()
    QUANTUM_REPEATER_PROTOCOL = auto()
    TELEPORTATION_CIRCUIT_PROTOCOL = auto()
    REMOTE_STATE_PREPARATION_PROTOCOL = auto()
    DIRECT_PROTOCOL = auto()
    CNOT_PROTOCOL = auto()
    HADAMARD_PROTOCOL = auto()
    EPR_PROTOCOL = auto()
    GHZ_PROTOCOL = auto()
    W_PROTOCOL = auto()
    CLUSTER_PROTOCOL = auto()
    GRAPH_STATE_PROTOCOL = auto()
    AKLT_PROTOCOL = auto()
    KITAEV_HONEYCOMB_PROTOCOL = auto()
    TENSOR_NETWORK_PROTOCOL = auto()
    
    # Simulation algorithms
    QUANTUM_TRAJECTORY_ALGORITHM = auto()
    MONTE_CARLO_ALGORITHM = auto()
    PATH_INTEGRAL_ALGORITHM = auto()
    TENSOR_NETWORK_ALGORITHM = auto()
    DENSITY_MATRIX_ALGORITHM = auto()
    QUANTUM_WALK_ALGORITHM = auto()
    STOCHASTIC_PROCESS_ALGORITHM = auto()
    QUANTUM_CELLULAR_AUTOMATON_ALGORITHM = auto()
    WAVE_FUNCTION_COLLAPSE_ALGORITHM = auto()
    QUANTUM_BAYESIAN_ALGORITHM = auto()
    NEURAL_QUANTUM_STATE_ALGORITHM = auto()
    QUANTUM_BOLTZMANN_MACHINE_ALGORITHM = auto()
    QUANTUM_METROPOLIS_ALGORITHM = auto()
    QUANTUM_LANGEVIN_ALGORITHM = auto()
    LINDBLAD_ALGORITHM = auto()
    MANY_WORLDS_ALGORITHM = auto()
    CONSISTENT_HISTORIES_ALGORITHM = auto()
    DECOHERENT_HISTORIES_ALGORITHM = auto()
    
    # Coherence algorithms
    QUANTUM_ANNEALING_ALGORITHM = auto()
    STOCHASTIC_DESCENT_ALGORITHM = auto()
    GRADIENT_DESCENT_ALGORITHM = auto()
    SIMULATED_ANNEALING_ALGORITHM = auto()
    TENSOR_COMPRESSION_ALGORITHM = auto()
    DENSITY_MATRIX_EVOLUTION_ALGORITHM = auto()
    PATH_INTEGRAL_ALGORITHM_COHERENCE = auto()
    VARIATIONAL_ALGORITHM = auto()
    RECURSIVE_COMPRESSION_ALGORITHM = auto()
    HOLOGRAPHIC_PROJECTION_ALGORITHM = auto()
    RENORMALIZATION_GROUP_ALGORITHM = auto()
    
    # Event types
    STATE_CREATION_EVENT = auto()
    STATE_DESTRUCTION_EVENT = auto()
    COHERENCE_CHANGE_EVENT = auto()
    ENTROPY_INCREASE_EVENT = auto()
    ENTROPY_DECREASE_EVENT = auto()
    OBSERVATION_EVENT = auto()
    ENTANGLEMENT_CREATION_EVENT = auto()
    ENTANGLEMENT_BREAKING_EVENT = auto()
    TELEPORTATION_EVENT = auto()
    MEASUREMENT_EVENT = auto()
    DECOHERENCE_EVENT = auto()
    QUANTUM_ERROR_EVENT = auto()
    STABILITY_THRESHOLD_EVENT = auto()
    COLLAPSE_EVENT = auto()
    CONVERGENCE_EVENT = auto()
    DIVERGENCE_EVENT = auto()
    RESONANCE_EVENT = auto()
    INTERFERENCE_EVENT = auto()
    
    # Log levels
    DEBUG_LEVEL = auto()
    INFO_LEVEL = auto()
    WARNING_LEVEL = auto()
    ERROR_LEVEL = auto()
    CRITICAL_LEVEL = auto()
    
    # Special values and constants
    STEPS = auto()
    TICKS = auto()
    CYCLES = auto()
    EPOCH = auto()
    LEVEL = auto()
    ANTICONTROL = auto()
    INTO = auto()
    REMOVE = auto()
    EXISTS = auto()
    FORMATTED = auto()
    
    # Meta references
    SELF_REF = auto()
    OBSERVER_REF = auto()
    SYSTEM_REF = auto()
    
    # Complex and special literals
    COMPLEX = auto()
    VEC = auto()
    MAT = auto()
    TENSOR = auto()
    STATEVEC = auto()
    
    # Quantum expressions
    BRA = auto()
    KET = auto()
    BRAKET = auto()
    EXPECTATION = auto()
    TENSOR_PRODUCT = auto()
    TRACE = auto()
    PARTIAL_TRACE = auto()
    FIDELITY = auto()
    ENTROPY_FUNC = auto()
    PURITY = auto()
    SCHMIDT_DECOMPOSITION = auto()
    EIGENVALUES = auto()
    EIGENVECTORS = auto()


class CodeGenerationTarget(Enum):
    """Available targets for code generation"""
    QUANTUM_SIMULATOR = auto()
    CLASSICAL_SIMULATOR = auto()
    IBM_QUANTUM = auto()
    RIGETTI_QUANTUM = auto()
    GOOGLE_QUANTUM = auto()
    IONQ_QUANTUM = auto()


class TypeRegistry:
    """
    Registry for type information in the Recursia language
    """
    
    def __init__(self):
        """Initialize the type registry"""
        self.primitive_types = set()
        self.state_types = set()
        self.observer_types = {}
        self.custom_types = {}
        self.alias_types = {}
    
    def register_primitive_type(self, name):
        """Register a primitive type"""
        self.primitive_types.add(name)
    
    def register_state_type(self, name):
        """Register a state type"""
        self.state_types.add(name)
    
    def register_observer_type(self, name, properties=None):
        """Register an observer type with optional properties"""
        self.observer_types[name] = properties or {}
    
    def register_custom_type(self, name, definition):
        """Register a custom type"""
        self.custom_types[name] = definition
    
    def register_alias(self, alias, target_type):
        """Register a type alias"""
        self.alias_types[alias] = target_type
    
    def is_primitive(self, name):
        """Check if a type is primitive"""
        return name in self.primitive_types
    
    def is_state_type(self, name):
        """Check if a type is a state type"""
        return name in self.state_types
    
    def is_observer_type(self, name):
        """Check if a type is an observer type"""
        return name in self.observer_types
    
    def resolve_type(self, name):
        """Resolve a type name to its definition, following aliases"""
        if name in self.alias_types:
            return self.resolve_type(self.alias_types[name])
        
        if name in self.primitive_types:
            return {'kind': 'primitive', 'name': name}
        
        if name in self.state_types:
            return {'kind': 'state', 'name': name}
        
        if name in self.observer_types:
            return {'kind': 'observer', 'name': name, 'properties': self.observer_types[name]}
        
        if name in self.custom_types:
            return self.custom_types[name]
        
        return None