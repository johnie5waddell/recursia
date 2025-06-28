from src.core.data_classes import (
    BinaryExpression, BooleanLiteral, ComplexLiteral, DensityMatrixLiteral, 
    Expression, FunctionCallExpression, IdentifierExpression, MatrixLiteral, MetaReference, 
    NumberLiteral, QuantumExpression, QubitLiteral, SemanticError, 
    StateVectorLiteral, StringLiteral, TensorLiteral, TypeAnnotation, 
    UnaryExpression, VectorLiteral
)
from src.core.symbol_table import SymbolTable
from src.core.types import TokenType, TypeRegistry  # Import just what we need from types.py

class TypeChecker:
    """Type checker for Recursia programs"""
    
    def __init__(self, symbol_table: SymbolTable):
        self.symbol_table = symbol_table
        self.primitive_types = {
            "string_type", "number_type", "boolean_type", "complex_type", 
            "qubit_type", "null_type", "void_type"
        }
        self.quantum_types = {
            "quantum_type", "superposition_type", "entangled_type"
        }
        self.errors: list[SemanticError] = []
        self.filename = "current_file"  # Default filename
    
    def check_type_compatibility(self, expected: TypeAnnotation, actual: TypeAnnotation, 
                             location: tuple[int, int, int], context: str = "") -> bool:
        """Check if two types are compatible"""
        if not expected or not actual:
            return True  # We're lenient if types are unknown
            
        # Exact match
        if expected.name == actual.name:
            return True
            
        # Special cases for numeric types
        if expected.name == "number_type" and actual.name in ("integer_type", "float_type"):
            return True
            
        # Special case for null assignment
        if actual.name == "null_type":
            return True
            
        # If expected is any_type, it's compatible with everything
        if expected.name == "any_type":
            return True
            
        # If expected is a quantum type, any other quantum type is compatible
        quantum_types = {"quantum_type", "superposition_type", "entangled_type"}
        if expected.name in quantum_types and actual.name in quantum_types:
            return True
            
        # Types are not compatible
        line, column, _ = location
        self.errors.append(SemanticError(
            f"Type mismatch{f' in {context}' if context else ''}: expected {expected.name}, got {actual.name}",
            self.filename, line, column, "type"
        ))
        return False

    def get_expression_type(self, expr: Expression) -> TypeAnnotation | None:
        """Determine the type of an expression"""
        if isinstance(expr, StringLiteral):
            return TypeAnnotation(expr.location, "primitive", "string_type")
        elif isinstance(expr, NumberLiteral):
            return TypeAnnotation(expr.location, "primitive", "number_type")
        elif isinstance(expr, BooleanLiteral):
            return TypeAnnotation(expr.location, "primitive", "boolean_type")
        elif isinstance(expr, ComplexLiteral):
            return TypeAnnotation(expr.location, "primitive", "complex_type")
        elif isinstance(expr, QubitLiteral):
            return TypeAnnotation(expr.location, "primitive", "qubit_type")
        elif isinstance(expr, StateVectorLiteral):
            return TypeAnnotation(expr.location, "quantum", "state_vector_type")
        elif isinstance(expr, DensityMatrixLiteral):
            return TypeAnnotation(expr.location, "quantum", "density_matrix_type")
        elif isinstance(expr, VectorLiteral):
            return TypeAnnotation(expr.location, "primitive", "vector_type")
        elif isinstance(expr, MatrixLiteral):
            return TypeAnnotation(expr.location, "primitive", "matrix_type")
        elif isinstance(expr, TensorLiteral):
            return TypeAnnotation(expr.location, "primitive", "tensor_type")

        elif isinstance(expr, MetaReference):
            # Check if it's a variable reference
            var = self.symbol_table.get_variable(expr.name)
            if var:
                return var.var_type
            
            # Check if it's a state reference
            state = self.symbol_table.get_state(expr.name)
            if state:
                return TypeAnnotation(expr.location, "state", state.state_type)
            
            # Check if it's an observer reference
            observer = self.symbol_table.get_observer(expr.name)
            if observer:
                return TypeAnnotation(expr.location, "observer", observer.observer_type)
                
            # Unknown reference
            line, column, _ = expr.location
            self.errors.append(SemanticError(
                f"Undefined identifier: {expr.name}",
                self.filename, line, column, "semantic"
            ))
            return None
            
        elif isinstance(expr, IdentifierExpression):
            # Check if it's a variable reference
            var = self.symbol_table.get_variable(expr.name)
            if var and var.var_type:
                return var.var_type
            
            # Check if it's a state reference
            state = self.symbol_table.get_state(expr.name)
            if state:
                return TypeAnnotation(expr.location, "state", state.state_type)
            
            # Check if it's an observer reference
            observer = self.symbol_table.get_observer(expr.name)
            if observer:
                return TypeAnnotation(expr.location, "observer", observer.observer_type)
                
            # Unknown reference - return unknown type instead of error
            # This might be a loop variable that hasn't been typed yet
            return None
            
        elif isinstance(expr, BinaryExpression):
            left_type = self.get_expression_type(expr.left)
            right_type = self.get_expression_type(expr.right)
            
            # Handle different operators
            if expr.operator in ("+", "-", "*", "/", "%", "**"):
                if left_type and right_type:
                    if expr.operator == "+":
                        # String concatenation - allow string + any type
                        if left_type.name == "string_type" or right_type.name == "string_type":
                            return TypeAnnotation(expr.location, "primitive", "string_type")
                    
                    if left_type.name == "number_type" and right_type.name == "number_type":
                        return left_type  # Numeric operation
                    elif left_type.name == "complex_type" or right_type.name == "complex_type":
                        return TypeAnnotation(expr.location, "primitive", "complex_type")  # Complex result
                elif (left_type and left_type.name == "number_type") or (right_type and right_type.name == "number_type"):
                    # If at least one side is a number and the other is unknown (likely a loop variable),
                    # assume it will be a number at runtime
                    return TypeAnnotation(expr.location, "primitive", "number_type")
                
                # Type error only if both types are known and incompatible
                if left_type and right_type:
                    line, column, _ = expr.location
                    self.errors.append(SemanticError(
                        f"Invalid operands for {expr.operator}: {left_type.name if left_type else 'unknown'} and {right_type.name if right_type else 'unknown'}",
                        self.filename, line, column, "type"
                    ))
                    return None
                
                # If we can't determine the type, return None without error
                return None
            elif expr.operator in ("==", "!=", "<", "<=", ">", ">="):
                # Comparison operators return boolean
                return TypeAnnotation(expr.location, "primitive", "boolean_type")
            elif expr.operator in ("&&", "||", "and", "or", "xor"):
                # Logical operators return boolean
                if left_type and left_type.name != "boolean_type":
                    line, column, _ = expr.left.location
                    self.errors.append(SemanticError(
                        f"Logical operator {expr.operator} requires boolean operands, got {left_type.name}",
                        self.filename, line, column, "type"
                    ))
                if right_type and right_type.name != "boolean_type":
                    line, column, _ = expr.right.location
                    self.errors.append(SemanticError(
                        f"Logical operator {expr.operator} requires boolean operands, got {right_type.name}",
                        self.filename, line, column, "type"
                    ))
                return TypeAnnotation(expr.location, "primitive", "boolean_type")
            
            # Default case
            return None
            
        elif isinstance(expr, UnaryExpression):
            operand_type = self.get_expression_type(expr.operand)
            
            if expr.operator in ("+", "-"):
                if operand_type and operand_type.name == "number_type":
                    return operand_type
                elif operand_type and operand_type.name == "complex_type":
                    return operand_type
                else:
                    line, column, _ = expr.location
                    self.errors.append(SemanticError(
                        f"Unary {expr.operator} requires numeric operand, got {operand_type.name if operand_type else 'unknown'}",
                        self.filename, line, column, "type"
                    ))
            elif expr.operator in ("!", "not"):
                if operand_type and operand_type.name != "boolean_type":
                    line, column, _ = expr.location
                    self.errors.append(SemanticError(
                        f"Unary {expr.operator} requires boolean operand, got {operand_type.name if operand_type else 'unknown'}",
                        self.filename, line, column, "type"
                    ))
                return TypeAnnotation(expr.location, "primitive", "boolean_type")
            
            return operand_type
            
        elif isinstance(expr, FunctionCallExpression):
            func = self.symbol_table.get_function(expr.function)
            if not func:
                line, column, _ = expr.location
                self.errors.append(SemanticError(
                    f"Undefined function: {expr.function}",
                    self.filename, line, column, "semantic"
                ))
                return None
                
            # Check argument count
            if len(expr.arguments) != len(func.params):
                line, column, _ = expr.location
                self.errors.append(SemanticError(
                    f"Function {expr.function} expects {len(func.params)} arguments, got {len(expr.arguments)}",
                    self.filename, line, column, "semantic"
                ))
            
            # Check argument types
            for i, (arg, (param_name, param_type)) in enumerate(zip(expr.arguments, func.params)):
                arg_type = self.get_expression_type(arg)
                if arg_type and param_type:
                    self.check_type_compatibility(
                        param_type, arg_type, arg.location, 
                        f"argument {i+1} ({param_name})"
                    )
            
            return func.return_type
            
        elif isinstance(expr, QuantumExpression):
            # Check arguments
            for arg in expr.args:
                self.get_expression_type(arg)
                
            # Determine return type based on expression kind
            if expr.kind in ("bra", "ket", "statevec"):
                return TypeAnnotation(expr.location, "quantum", "state_vector_type")
            elif expr.kind == "braket":
                return TypeAnnotation(expr.location, "primitive", "complex_type")
            elif expr.kind in ("fidelity", "entropy", "purity"):
                return TypeAnnotation(expr.location, "primitive", "number_type")
            elif expr.kind in ("eigenvalues", "eigenvectors"):
                return TypeAnnotation(expr.location, "primitive", "vector_type")
            
            return None
        
        # Default case for other expressions
        return None