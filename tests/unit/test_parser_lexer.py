"""
Unit tests for parser and lexer modules.
Tests tokenization, AST generation, and syntax validation.
"""

import pytest
from unittest.mock import Mock, patch

from src.core.lexer import Lexer, Token, TokenType
from src.core.parser import Parser, ASTNode, ParseError
from src.core.types import RecursiaInt, RecursiaFloat


class TestLexer:
    """Test lexical analysis and tokenization."""
    
    def test_basic_tokenization(self):
        """Test tokenization of basic constructs."""
        lexer = Lexer()
        
        code = """
        quantum_register qreg[4];
        coherence_field field = 0.95;
        """
        
        tokens = list(lexer.tokenize(code))
        
        # Check token types
        assert tokens[0].type == TokenType.QUANTUM_REGISTER
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "qreg"
        assert tokens[2].type == TokenType.LBRACKET
        assert tokens[3].type == TokenType.INTEGER
        assert tokens[3].value == 4
        assert tokens[4].type == TokenType.RBRACKET
        assert tokens[5].type == TokenType.SEMICOLON
        
    def test_number_tokenization(self):
        """Test tokenization of different number formats."""
        lexer = Lexer()
        
        # Integers
        tokens = list(lexer.tokenize("42 -17 0"))
        assert all(t.type == TokenType.INTEGER for t in tokens)
        assert [t.value for t in tokens] == [42, -17, 0]
        
        # Floats
        tokens = list(lexer.tokenize("3.14 -0.5 1e-3 2.5e10"))
        assert all(t.type == TokenType.FLOAT for t in tokens)
        assert tokens[0].value == 3.14
        assert tokens[2].value == 0.001
        
        # Complex numbers
        tokens = list(lexer.tokenize("3+4i -2i 0.5-0.5i"))
        assert all(t.type == TokenType.COMPLEX for t in tokens)
        
    def test_string_tokenization(self):
        """Test string literal tokenization."""
        lexer = Lexer()
        
        # Simple strings
        tokens = list(lexer.tokenize('"hello world" "recursia"'))
        assert all(t.type == TokenType.STRING for t in tokens)
        assert tokens[0].value == "hello world"
        assert tokens[1].value == "recursia"
        
        # Escaped strings
        tokens = list(lexer.tokenize(r'"line1\nline2" "tab\there"'))
        assert tokens[0].value == "line1\nline2"
        assert tokens[1].value == "tab\there"
        
    def test_operator_tokenization(self):
        """Test operator tokenization."""
        lexer = Lexer()
        
        operators = "+ - * / % ** = += -= *= /= == != < > <= >= && || !"
        tokens = list(lexer.tokenize(operators))
        
        expected_types = [
            TokenType.PLUS, TokenType.MINUS, TokenType.MULTIPLY,
            TokenType.DIVIDE, TokenType.MODULO, TokenType.POWER,
            TokenType.ASSIGN, TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.MULTIPLY_ASSIGN, TokenType.DIVIDE_ASSIGN,
            TokenType.EQUAL, TokenType.NOT_EQUAL, TokenType.LESS_THAN,
            TokenType.GREATER_THAN, TokenType.LESS_EQUAL, TokenType.GREATER_EQUAL,
            TokenType.AND, TokenType.OR, TokenType.NOT
        ]
        
        for token, expected in zip(tokens, expected_types):
            assert token.type == expected
            
    def test_keyword_recognition(self):
        """Test keyword vs identifier recognition."""
        lexer = Lexer()
        
        code = """
        quantum observer measure if else while for
        quantum_register my_observer measurement
        """
        
        tokens = list(lexer.tokenize(code))
        
        # Keywords
        assert tokens[0].type == TokenType.QUANTUM
        assert tokens[1].type == TokenType.OBSERVER
        assert tokens[2].type == TokenType.MEASURE
        assert tokens[3].type == TokenType.IF
        assert tokens[4].type == TokenType.ELSE
        assert tokens[5].type == TokenType.WHILE
        assert tokens[6].type == TokenType.FOR
        
        # Identifiers (not keywords)
        assert tokens[7].type == TokenType.QUANTUM_REGISTER
        assert tokens[8].type == TokenType.IDENTIFIER
        assert tokens[9].type == TokenType.IDENTIFIER
        
    def test_comment_handling(self):
        """Test single and multi-line comments."""
        lexer = Lexer()
        
        code = """
        // This is a comment
        quantum x = 1; // inline comment
        /* Multi-line
           comment */
        observer obs;
        """
        
        tokens = list(lexer.tokenize(code))
        
        # Comments should be skipped
        assert tokens[0].type == TokenType.QUANTUM
        assert tokens[1].type == TokenType.IDENTIFIER
        assert tokens[1].value == "x"
        # No comment tokens
        
    def test_probability_syntax(self):
        """Test probability-specific syntax."""
        lexer = Lexer()
        
        code = "P(0.7) P(0.3|coherence > 0.5)"
        tokens = list(lexer.tokenize(code))
        
        assert tokens[0].type == TokenType.PROBABILITY
        assert tokens[1].type == TokenType.LPAREN
        assert tokens[2].type == TokenType.FLOAT
        assert tokens[2].value == 0.7
        
    def test_special_constructs(self):
        """Test special Recursia constructs."""
        lexer = Lexer()
        
        code = """
        |0⟩ |1⟩ |+⟩ |-⟩
        ⊗ ⊕
        @recursive @parallel
        """
        
        tokens = list(lexer.tokenize(code))
        
        # Quantum states
        assert tokens[0].type == TokenType.KET_ZERO
        assert tokens[1].type == TokenType.KET_ONE
        assert tokens[2].type == TokenType.KET_PLUS
        assert tokens[3].type == TokenType.KET_MINUS
        
        # Operators
        assert tokens[4].type == TokenType.TENSOR_PRODUCT
        assert tokens[5].type == TokenType.XOR
        
        # Decorators
        assert tokens[6].type == TokenType.DECORATOR
        assert tokens[6].value == "recursive"
        
    def test_error_handling(self):
        """Test lexer error handling."""
        lexer = Lexer()
        
        # Unterminated string
        with pytest.raises(LexError):
            list(lexer.tokenize('"unterminated'))
            
        # Invalid character
        with pytest.raises(LexError):
            list(lexer.tokenize("valid § invalid"))
            
        # Invalid number format
        with pytest.raises(LexError):
            list(lexer.tokenize("3.14.159"))


class TestParser:
    """Test syntax analysis and AST generation."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return Parser()
        
    def test_variable_declaration(self, parser):
        """Test parsing variable declarations."""
        # Simple declaration
        ast = parser.parse("quantum x = 0.5;")
        assert ast.type == "program"
        assert ast.children[0].type == "declaration"
        assert ast.children[0].var_type == "quantum"
        assert ast.children[0].identifier == "x"
        assert ast.children[0].initializer.value == 0.5
        
        # Array declaration
        ast = parser.parse("quantum_register qreg[8];")
        assert ast.children[0].type == "declaration"
        assert ast.children[0].var_type == "quantum_register"
        assert ast.children[0].array_size == 8
        
    def test_expression_parsing(self, parser):
        """Test expression parsing with precedence."""
        # Arithmetic
        ast = parser.parse("x = 2 + 3 * 4;")
        assign = ast.children[0]
        assert assign.type == "assignment"
        expr = assign.expression
        assert expr.type == "binary_op"
        assert expr.operator == "+"
        assert expr.right.type == "binary_op"
        assert expr.right.operator == "*"
        
        # Comparison and logical
        ast = parser.parse("result = (a > b) && (c <= d);")
        expr = ast.children[0].expression
        assert expr.type == "binary_op"
        assert expr.operator == "&&"
        
    def test_control_flow_parsing(self, parser):
        """Test parsing of control flow structures."""
        # If-else
        code = """
        if (coherence > 0.5) {
            measure(qreg);
        } else {
            evolve(qreg, 0.1);
        }
        """
        ast = parser.parse(code)
        if_stmt = ast.children[0]
        assert if_stmt.type == "if_statement"
        assert if_stmt.condition.type == "binary_op"
        assert len(if_stmt.then_block.children) == 1
        assert len(if_stmt.else_block.children) == 1
        
        # While loop
        code = """
        while (strain < 0.8) {
            allocate_memory(10);
            strain = calculate_strain();
        }
        """
        ast = parser.parse(code)
        while_stmt = ast.children[0]
        assert while_stmt.type == "while_loop"
        assert while_stmt.condition.type == "binary_op"
        assert len(while_stmt.body.children) == 2
        
        # For loop
        code = """
        for (i = 0; i < 10; i++) {
            qubits[i] = |0⟩;
        }
        """
        ast = parser.parse(code)
        for_stmt = ast.children[0]
        assert for_stmt.type == "for_loop"
        assert for_stmt.init.type == "assignment"
        assert for_stmt.condition.type == "binary_op"
        assert for_stmt.update.type == "assignment"
        
    def test_function_parsing(self, parser):
        """Test function declaration and call parsing."""
        # Function declaration
        code = """
        function entangle(q1: quantum, q2: quantum) -> quantum {
            gate CNOT(q1, q2);
            return measure(q1) ⊗ measure(q2);
        }
        """
        ast = parser.parse(code)
        func = ast.children[0]
        assert func.type == "function_declaration"
        assert func.name == "entangle"
        assert len(func.parameters) == 2
        assert func.return_type == "quantum"
        
        # Function call
        ast = parser.parse("result = entangle(qbit1, qbit2);")
        call = ast.children[0].expression
        assert call.type == "function_call"
        assert call.name == "entangle"
        assert len(call.arguments) == 2
        
    def test_quantum_operations(self, parser):
        """Test parsing of quantum-specific operations."""
        # Gate application
        ast = parser.parse("gate H(qubit);")
        gate_op = ast.children[0]
        assert gate_op.type == "gate_operation"
        assert gate_op.gate_name == "H"
        assert gate_op.target == "qubit"
        
        # Measurement
        ast = parser.parse("result = measure(qreg, basis='Z');")
        measure = ast.children[0].expression
        assert measure.type == "measurement"
        assert measure.target == "qreg"
        assert measure.basis == "Z"
        
        # Observer declaration
        code = """
        observer obs {
            focus: 0.8,
            phase: pi/4,
            threshold: 0.6
        };
        """
        ast = parser.parse(code)
        observer = ast.children[0]
        assert observer.type == "observer_declaration"
        assert observer.properties["focus"] == 0.8
        assert observer.properties["threshold"] == 0.6
        
    def test_probability_syntax_parsing(self, parser):
        """Test parsing of probability constructs."""
        code = """
        P(0.7) {
            state = |1⟩;
        } P(0.3) {
            state = |0⟩;
        }
        """
        ast = parser.parse(code)
        prob_block = ast.children[0]
        assert prob_block.type == "probability_block"
        assert prob_block.probability == 0.7
        assert len(prob_block.branches) == 2
        
    def test_simulation_block_parsing(self, parser):
        """Test parsing of simulation blocks."""
        code = """
        simulate(steps=100, dt=0.01) {
            quantum_register qreg[4];
            observer obs;
            
            initialize {
                qreg = |0000⟩;
                obs.focus = 0.5;
            }
            
            evolution {
                gate H(qreg[0]);
                if (obs.detect(qreg)) {
                    collapse(qreg);
                }
            }
            
            measurement {
                return measure(qreg);
            }
        }
        """
        ast = parser.parse(code)
        sim = ast.children[0]
        assert sim.type == "simulation_block"
        assert sim.parameters["steps"] == 100
        assert sim.parameters["dt"] == 0.01
        assert "initialize" in sim.sections
        assert "evolution" in sim.sections
        assert "measurement" in sim.sections
        
    def test_recursive_constructs(self, parser):
        """Test parsing of recursive language features."""
        code = """
        @recursive(depth=3)
        function fractal_measure(state) {
            if (recursion_depth() == 0) {
                return measure(state);
            }
            
            substates = split(state, 2);
            results = [];
            for (s in substates) {
                results.append(fractal_measure(s));
            }
            return combine(results);
        }
        """
        ast = parser.parse(code)
        func = ast.children[0]
        assert func.type == "function_declaration"
        assert func.decorators[0].name == "recursive"
        assert func.decorators[0].parameters["depth"] == 3
        
    def test_error_recovery(self, parser):
        """Test parser error handling and recovery."""
        # Missing semicolon
        with pytest.raises(ParseError) as exc:
            parser.parse("quantum x = 0.5")
        assert "semicolon" in str(exc.value).lower()
        
        # Mismatched brackets
        with pytest.raises(ParseError) as exc:
            parser.parse("quantum_register qreg[8;")
        assert "bracket" in str(exc.value).lower()
        
        # Invalid syntax
        with pytest.raises(ParseError) as exc:
            parser.parse("quantum quantum x;")
        assert "syntax" in str(exc.value).lower()
        
    def test_type_annotations(self, parser):
        """Test parsing of type annotations."""
        code = """
        quantum<float> coherence = 0.95;
        observer<recursive> obs;
        field<memory, 3> mem_field;
        """
        ast = parser.parse(code)
        
        # Check type parameters
        assert ast.children[0].type_params == ["float"]
        assert ast.children[1].type_params == ["recursive"]
        assert ast.children[2].type_params == ["memory", "3"]


class TestASTNode:
    """Test AST node functionality."""
    
    def test_node_creation(self):
        """Test AST node creation and properties."""
        node = ASTNode(
            type="binary_op",
            operator="+",
            position=(10, 5)
        )
        
        assert node.type == "binary_op"
        assert node.operator == "+"
        assert node.position == (10, 5)
        assert node.children == []
        
    def test_node_traversal(self):
        """Test AST traversal methods."""
        # Build simple tree
        root = ASTNode("program")
        decl = ASTNode("declaration", identifier="x")
        expr = ASTNode("binary_op", operator="+")
        left = ASTNode("identifier", value="a")
        right = ASTNode("literal", value=5)
        
        root.add_child(decl)
        decl.add_child(expr)
        expr.add_child(left)
        expr.add_child(right)
        
        # Test traversal
        nodes = list(root.traverse_preorder())
        assert len(nodes) == 5
        assert nodes[0] == root
        assert nodes[-1] == right
        
        # Test search
        found = root.find_nodes("identifier")
        assert len(found) == 1
        assert found[0].value == "a"
        
    def test_node_modification(self):
        """Test AST node modification."""
        node = ASTNode("literal", value=10)
        
        # Update value
        node.value = 20
        assert node.value == 20
        
        # Add metadata
        node.add_metadata("type_info", "int")
        assert node.metadata["type_info"] == "int"
        
    def test_node_validation(self):
        """Test AST node validation."""
        # Valid node
        valid = ASTNode("declaration", identifier="x", var_type="quantum")
        assert valid.validate()
        
        # Invalid node (missing required field)
        invalid = ASTNode("declaration", identifier="x")
        assert not invalid.validate()


# Integration tests
class TestParserLexerIntegration:
    """Test integration between lexer and parser."""
    
    def test_full_program_parsing(self):
        """Test parsing a complete Recursia program."""
        code = """
        // Quantum teleportation example
        quantum_register alice[2];
        quantum_register bob[1];
        
        // Create entangled pair
        gate H(alice[0]);
        gate CNOT(alice[0], bob[0]);
        
        // Alice's operations
        quantum message = |1⟩;
        gate CNOT(message, alice[0]);
        gate H(message);
        
        // Measure Alice's qubits
        bit c1 = measure(message);
        bit c2 = measure(alice[0]);
        
        // Bob's corrections
        if (c2 == 1) {
            gate X(bob[0]);
        }
        if (c1 == 1) {
            gate Z(bob[0]);
        }
        
        // Verify teleportation
        quantum result = bob[0];
        """
        
        lexer = Lexer()
        parser = Parser()
        
        tokens = list(lexer.tokenize(code))
        assert len(tokens) > 50  # Substantial program
        
        ast = parser.parse(code)
        assert ast.type == "program"
        assert len(ast.children) > 10  # Multiple statements
        
        # Verify key structures parsed correctly
        declarations = ast.find_nodes("declaration")
        assert any(d.identifier == "alice" for d in declarations)
        assert any(d.identifier == "bob" for d in declarations)
        
        gates = ast.find_nodes("gate_operation")
        assert any(g.gate_name == "H" for g in gates)
        assert any(g.gate_name == "CNOT" for g in gates)