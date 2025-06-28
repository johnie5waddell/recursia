from typing import List, Dict, Optional

from src.core.data_classes import FunctionDefinition, ObserverDefinition, PatternDeclaration, QuantumStateDefinition, VariableDefinition
from src.core.scope import Scope


class SymbolTable:
    """Symbol table for tracking identifiers and their types"""
    
    def __init__(self):
        self.global_scope = Scope("global", None)
        self.current_scope = self.global_scope
        self.states: Dict[str, QuantumStateDefinition] = {}
        self.observers: Dict[str, ObserverDefinition] = {}
        self.functions: Dict[str, FunctionDefinition] = {}
        self.variables: Dict[str, Dict[str, VariableDefinition]] = {"global": {}}
        self.patterns: Dict[str, PatternDeclaration] = {}  # Adding patterns registry
        
        # Register built-in patterns
        self._register_builtin_patterns()
        
        # Register built-in functions
        self._register_builtin_functions()

    def _register_builtin_patterns(self):
        """Register built-in patterns like 'observe'"""
        # Create a built-in observe pattern
        from src.core.data_classes import PatternDeclaration
        
        # Built-in patterns
        builtin_patterns = [
            "observe", "memory_field", "coherence_field", "entropy_field",
            "information_field", "curvature_field", "temporal_field"
        ]
        
        for pattern_name in builtin_patterns:
            pattern = PatternDeclaration(
                name=pattern_name,
                location=(0, 0, 0),  # Built-in location
                pattern_type="builtin",
                statements=[],  # Built-in patterns don't have user-defined statements
                generic_params=[]
            )
            self.patterns[pattern_name] = pattern
    
    def _register_builtin_functions(self):
        """Register built-in mathematical functions"""
        from src.core.data_classes import FunctionDefinition, VariableDefinition
        
        # Built-in math functions
        builtin_funcs = [
            ("abs", ["value"], "primitive"),  # Absolute value
            ("exp", ["value"], "primitive"),  # Exponential
            ("log", ["value"], "primitive"),  # Natural logarithm
            ("sqrt", ["value"], "primitive"), # Square root
            ("sin", ["value"], "primitive"),  # Sine
            ("cos", ["value"], "primitive"),  # Cosine
            ("tan", ["value"], "primitive"),  # Tangent
            ("pow", ["base", "exponent"], "primitive"), # Power
            ("min", ["a", "b"], "primitive"), # Minimum
            ("max", ["a", "b"], "primitive"), # Maximum
            ("floor", ["value"], "primitive"), # Floor
            ("ceil", ["value"], "primitive"),  # Ceiling
        ]
        
        for func_name, params, return_type in builtin_funcs:
            func_def = FunctionDefinition(
                name=func_name,
                params=[(p, None) for p in params],  # List of (name, type) tuples
                return_type=None,
                body=[],  # Built-in functions don't have user-defined bodies
                location=(0, 0, 0)  # Built-in location
            )
            self.functions[func_name] = func_def
    
    def has_state(self, name) -> bool:
        """Check if a state with the given name exists"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
        return name in self.states

    def enter_scope(self, name: str) -> 'Scope':
        """Enter a new scope"""
        scope = Scope(name, self.current_scope)
        self.current_scope = scope
        self.variables[name] = {}
        return scope
    
    def exit_scope(self) -> 'Scope':
        """Exit the current scope"""
        if self.current_scope.parent:
            self.current_scope = self.current_scope.parent
        return self.current_scope
    
    def add_state(self, state: QuantumStateDefinition) -> bool:
        """Add a state definition to the symbol table"""
        if state.name in self.states:
            return False
        self.states[state.name] = state
        return True
    
    def add_observer(self, observer: ObserverDefinition) -> bool:
        """Add an observer definition to the symbol table"""
        if observer.name in self.observers:
            return False
        self.observers[observer.name] = observer
        return True
    
    def add_function(self, function: FunctionDefinition) -> bool:
        """Add a function definition to the symbol table"""
        if function.name in self.functions:
            return False
        self.functions[function.name] = function
        return True
    
    def add_pattern(self, pattern: PatternDeclaration) -> bool:
        """Add a pattern definition to the symbol table"""
        if pattern.name in self.patterns:
            return False
        self.patterns[pattern.name] = pattern
        return True
    
    def add_variable(self, variable: VariableDefinition, scope_name: Optional[str] = None) -> bool:
        """Add a variable definition to the current or specified scope"""
        scope = scope_name or self.current_scope.name
        if scope not in self.variables:
            self.variables[scope] = {}
        
        if variable.name in self.variables[scope]:
            return False
        self.variables[scope][variable.name] = variable
        return True
    
    def get_state(self, name) -> Optional[QuantumStateDefinition]:
        """Get a state definition"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
        return self.states.get(name)
    
    def get_observer(self, name) -> Optional[ObserverDefinition]:
        """Get an observer definition"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
        return self.observers.get(name)
    
    def get_function(self, name) -> Optional[FunctionDefinition]:
        """Get a function definition"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
        return self.functions.get(name)
    
    def get_pattern(self, name) -> Optional[PatternDeclaration]:
        """Get a pattern definition"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
        return self.patterns.get(name)
    
    def get_variable(self, name) -> Optional[VariableDefinition]:
        """Get a variable definition from the current scope chain"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            # Convert other types to string as fallback
            name = str(name)
            
        scope = self.current_scope
        while scope:
            if scope.name in self.variables and name in self.variables[scope.name]:
                return self.variables[scope.name][name]
            scope = scope.parent
            
        # Check global scope if not found in scope chain
        if name in self.variables["global"]:
            return self.variables["global"][name]
        
        return None
    
    def is_defined(self, name) -> bool:
        """Check if a name is defined in any category"""
        # Handle IdentifierExpression objects by extracting the name
        if hasattr(name, 'name'):
            name = name.name
        elif not isinstance(name, str):
            name = str(name)
            
        return (name in self.states or 
                name in self.observers or 
                name in self.functions or 
                name in self.patterns or
                self.get_variable(name) is not None)
