from typing import Dict, List, Optional, Set, Union, Any

class Scope:
    """
    Represents a single scope in the symbol table
    
    Scopes form a hierarchical structure for managing variable, function,
    and other definitions across different program contexts. Each scope can
    have a parent scope, creating a chain for symbol resolution.
    """
    
    def __init__(self, name: str, parent: Optional['Scope'] = None, filename: Optional[str] = None):
        """
        Initialize a scope
        
        Args:
            name (str): Scope name (e.g., 'global', 'function_name', 'for_loop_1')
            parent (Scope, optional): Parent scope for symbol resolution chain
            filename (str, optional): Source filename associated with this scope
        """
        self.name = name
        self.parent = parent
        self.symbols: Dict[str, Dict[str, Any]] = {}
        self.filename = filename
        self.children: List['Scope'] = []
        self.has_return = False
        self.loop_depth = 0  # For handling break/continue
        self.visibility = "local"  # Options: "local", "global", "recursive", "module"
        self.function_symbol = None  # Direct reference to function symbol if this is a function scope
        
        # Track symbols by category
        self.variables: Set[str] = set()
        self.functions: Set[str] = set()
        self.states: Set[str] = set()
        self.observers: Set[str] = set()
        self.patterns: Set[str] = set()
        
        # Add this scope as a child of the parent
        if parent:
            parent.children.append(self)
            # Inherit loop depth from parent
            self.loop_depth = parent.loop_depth
    
    def create_child(self, name: str) -> 'Scope':
        """
        Create a new child scope without making it the current scope
        
        Args:
            name (str): Scope name
            
        Returns:
            Scope: The new child scope
        """
        return Scope(name, self, self.filename)

    def _update_symbol_categories(self, name: str, symbol: Dict[str, Any], is_removal: bool = False) -> None:
        """
        Update the symbol category sets based on the symbol's kind
        
        Args:
            name (str): Symbol name
            symbol (dict): Symbol information including type, properties, etc.
            is_removal (bool): If True, remove from category instead of adding
        """
        symbol_kind = symbol.get('kind', '')
        
        # Get the right category set
        category_set = None
        if symbol_kind == 'variable':
            category_set = self.variables
        elif symbol_kind == 'function':
            category_set = self.functions
        elif symbol_kind == 'state':
            category_set = self.states
        elif symbol_kind == 'observer':
            category_set = self.observers
        elif symbol_kind == 'pattern':
            category_set = self.patterns
        
        if category_set is not None:
            if is_removal:
                category_set.discard(name)
            else:
                category_set.add(name)
    
    def define(self, name: str, symbol: Dict[str, Any]) -> bool:
        """
        Define a symbol in this scope
        
        Args:
            name (str): Symbol name
            symbol (dict): Symbol information including type, properties, etc.
            
        Returns:
            bool: True if defined, False if already exists
        """
        if name in self.symbols:
            return False
        
        self.symbols[name] = symbol
        self._update_symbol_categories(name, symbol)
        
        # If this is a function scope and we're defining the function itself,
        # store a direct reference for faster access
        if self.is_function_scope() and symbol.get('kind') == 'function' and name == self.name:
            self.function_symbol = symbol
        
        return True
    
    def redefine(self, name: str, symbol: Dict[str, Any]) -> bool:
        """
        Redefine an existing symbol in this scope
        
        Args:
            name (str): Symbol name
            symbol (dict): Updated symbol information
            
        Returns:
            bool: True if redefined, False if not found
        """
        if name not in self.symbols:
            return False
        
        old_symbol = self.symbols[name]
        
        # If category is changing, update category sets
        if old_symbol.get('kind', '') != symbol.get('kind', ''):
            self._update_symbol_categories(name, old_symbol, is_removal=True)
            self._update_symbol_categories(name, symbol)
        
        # Update the symbol
        self.symbols[name] = symbol
        
        # Update function_symbol reference if needed
        if self.is_function_scope() and name == self.name and symbol.get('kind') == 'function':
            self.function_symbol = symbol
            
        return True
    
    def remove(self, name: str) -> bool:
        """
        Remove a symbol from this scope
        
        Args:
            name (str): Symbol name to remove
            
        Returns:
            bool: True if removed, False if not found
        """
        if name not in self.symbols:
            return False
        
        # Remove from category sets
        symbol = self.symbols[name]
        self._update_symbol_categories(name, symbol, is_removal=True)
        
        # Remove from symbols dictionary
        del self.symbols[name]
        
        # Clear function_symbol reference if needed
        if self.is_function_scope() and name == self.name:
            self.function_symbol = None
            
        return True
    
    def lookup(self, name: str, stop_at_function: bool = False) -> Optional[Dict[str, Any]]:
        """
        Look up a symbol in this scope and parent scopes
        
        Args:
            name (str): Symbol name
            stop_at_function (bool): Whether to stop at function scope boundaries
        
        Returns:
            dict or None: Symbol information if found, None otherwise
        """
        # Check this scope first
        symbol = self.symbols.get(name)
        if symbol is not None:
            return symbol
        
        # Check if we should stop at this boundary
        if stop_at_function and self.is_function_scope():
            return None
        
        # Check parent scopes
        if self.parent:
            return self.parent.lookup(name, stop_at_function)
        
        return None
    
    def lookup_in_current(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Look up a symbol in this scope only
        
        Args:
            name (str): Symbol name
        
        Returns:
            dict or None: Symbol information if found, None otherwise
        """
        return self.symbols.get(name)
    
    def get_visible_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all symbols visible from this scope
        
        Returns:
            dict: Dictionary of visible symbols
        """
        visible = {}
        
        # Get symbols from parent scopes first (to be overridden by local definitions)
        if self.parent:
            visible.update(self.parent.get_visible_symbols())
        
        # Add local symbols (overriding parent definitions if names clash)
        visible.update(self.symbols)
        
        return visible
    
    def is_function_scope(self) -> bool:
        """
        Check if this is a function scope
        
        Returns:
            bool: True if this is a function scope
        """
        return self.name != "global" and not self.name.startswith("for_") and not self.name.startswith("iteration") and not self.name.startswith("if_") and not self.name.startswith("while_")
    
    def is_loop_scope(self) -> bool:
        """
        Check if this is a loop scope
        
        Returns:
            bool: True if this is a loop scope (for, while)
        """
        return self.name.startswith("for_") or self.name.startswith("while_")
    
    def is_global_scope(self) -> bool:
        """
        Check if this is the global scope
        
        Returns:
            bool: True if this is the global scope
        """
        return self.name == "global"
    
    def is_conditional_scope(self) -> bool:
        """
        Check if this is a conditional scope (if/else)
        
        Returns:
            bool: True if this is a conditional scope
        """
        return self.name.startswith("if_")
    
    def get_function_symbol(self) -> Optional[Dict[str, Any]]:
        """
        Get the function symbol for this scope
        
        Returns:
            dict or None: Function symbol if this is a function scope, None otherwise
        """
        # Use direct reference if available
        if self.function_symbol is not None:
            return self.function_symbol
            
        if not self.is_function_scope() or not self.parent:
            return None
        
        # Fallback to lookup if direct reference not set
        function_symbol = self.parent.lookup(self.name)
        if function_symbol and function_symbol.get('kind') == 'function':
            # Cache for future use
            self.function_symbol = function_symbol
            return function_symbol
            
        return None
    
    def get_scope_chain(self) -> List[str]:
        """
        Get the chain of scope names from global to this scope
        
        Returns:
            list: List of scope names
        """
        if not self.parent:
            return [self.name]
        
        chain = self.parent.get_scope_chain()
        chain.append(self.name)
        return chain
    
    def add_loop_level(self) -> None:
        """
        Increment the loop depth when entering a loop
        """
        self.loop_depth += 1
    
    def remove_loop_level(self) -> None:
        """
        Decrement the loop depth when exiting a loop
        
        Raises:
            ValueError: If attempting to decrement below zero
        """
        if self.loop_depth > 0:
            self.loop_depth -= 1
        else:
            raise ValueError(f"Cannot remove loop level: already at zero in scope '{self.name}'")
    
    def can_use_break(self) -> bool:
        """
        Check if a break statement is valid in this scope
        
        Returns:
            bool: True if break can be used
        """
        return self.loop_depth > 0
    
    def can_use_continue(self) -> bool:
        """
        Check if a continue statement is valid in this scope
        
        Returns:
            bool: True if continue can be used
        """
        return self.loop_depth > 0
    
    def mark_has_return(self) -> None:
        """
        Mark this scope as having a return statement
        """
        self.has_return = True
    
    def get_all_symbols_recursively(self) -> Dict[str, Dict[str, Any]]:
        """
        Get all symbols defined in this scope and all child scopes
        
        Returns:
            dict: Dictionary of all symbols
        """
        all_symbols = self.symbols.copy()
        
        # Add symbols from all child scopes
        for child in self.children:
            child_symbols = child.get_all_symbols_recursively()
            
            # Add with scope prefix to avoid name collisions
            for name, symbol in child_symbols.items():
                prefixed_name = f"{child.name}.{name}"
                all_symbols[prefixed_name] = symbol
        
        return all_symbols
    
    def set_visibility(self, visibility: str) -> None:
        """
        Set the visibility mode for this scope
        
        Args:
            visibility (str): One of "local", "global", "recursive", or "module"
            
        Raises:
            ValueError: If visibility is not one of the valid options
        """
        valid_visibilities = {"local", "global", "recursive", "module"}
        if visibility not in valid_visibilities:
            raise ValueError(f"Invalid visibility: {visibility}. Must be one of: {', '.join(valid_visibilities)}")
        
        self.visibility = visibility
    
    def get_visibility(self) -> str:
        """
        Get the visibility mode for this scope
        
        Returns:
            str: Visibility mode
        """
        return self.visibility
    
    def find_symbol_scope(self, name: str) -> Optional['Scope']:
        """
        Find which scope contains the definition of a symbol
        
        Args:
            name (str): Symbol name
            
        Returns:
            Scope or None: The scope containing the symbol, or None if not found
        """
        if name in self.symbols:
            return self
            
        if self.parent:
            return self.parent.find_symbol_scope(name)
            
        return None
    
    def __str__(self) -> str:
        """String representation of the scope"""
        parent_name = self.parent.name if self.parent else "None"
        return f"Scope({self.name}, parent={parent_name}, symbols={len(self.symbols)})"
    
    def __repr__(self) -> str:
        """String representation of the scope"""
        return self.__str__()