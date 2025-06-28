#!/usr/bin/env python3
"""
Comprehensive Quantum Programs Test Suite
Tests all quantum programs in the library to ensure they run without error to completion.
Provides detailed diagnostics and fixes for any issues found.
"""

import os
import sys
import json
import subprocess
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import time
import signal
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
import threading

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.lexer import RecursiaLexer
from src.core.parser import RecursiaParser
from src.core.semantic_analyzer import SemanticAnalyzer
from src.core.compiler import RecursiaCompiler
from src.core.runtime import Runtime
from src.core.data_classes import ParserError


@dataclass
class TestResult:
    """Result of testing a quantum program"""
    program_path: str
    program_name: str
    category: str
    status: str  # 'success', 'syntax_error', 'semantic_error', 'runtime_error', 'timeout'
    execution_time: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    error_line: Optional[int] = None
    error_details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    completion_percentage: float = 0.0
    memory_usage_mb: float = 0.0
    state_count: int = 0
    observer_count: int = 0
    measurement_count: int = 0
    gate_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class QuantumProgramValidator:
    """Validates and tests all quantum programs in the library"""
    
    def __init__(self, timeout: float = 30.0, verbose: bool = True):
        self.timeout = timeout
        self.verbose = verbose
        self.results: List[TestResult] = []
        self.compiler = RecursiaCompiler()
        
        # Program categories to test
        self.categories = [
            'basic',
            'intermediate', 
            'advanced',
            'consciousness',
            'experimental',
            'osh_calculations',
            'osh_predictions',
            'osh_testing',
            'theoretical',
            'enterprise',
            'optimization',
            'quantum_computing'
        ]
        
        # Track execution statistics
        self.total_programs = 0
        self.successful_programs = 0
        self.failed_programs = 0
        self.syntax_errors = 0
        self.semantic_errors = 0
        self.runtime_errors = 0
        self.timeouts = 0
        
    def find_all_programs(self) -> List[Tuple[str, str, str]]:
        """Find all .recursia programs in the quantum_programs directory"""
        programs = []
        base_path = Path('quantum_programs')
        
        for category in self.categories:
            category_path = base_path / category
            if category_path.exists():
                for file_path in category_path.glob('*.recursia'):
                    programs.append((
                        str(file_path),
                        file_path.stem,
                        category
                    ))
        
        # Also check root quantum_programs directory
        for file_path in base_path.glob('*.recursia'):
            if file_path.is_file():
                programs.append((
                    str(file_path),
                    file_path.stem,
                    'root'
                ))
                
        return sorted(programs)
    
    def test_program(self, program_path: str, program_name: str, category: str) -> TestResult:
        """Test a single quantum program"""
        start_time = time.time()
        
        try:
            # Read the program
            with open(program_path, 'r') as f:
                source_code = f.read()
            
            if not source_code.strip():
                return TestResult(
                    program_path=program_path,
                    program_name=program_name,
                    category=category,
                    status='syntax_error',
                    execution_time=time.time() - start_time,
                    error_message="Empty program file",
                    error_type="EmptyFile"
                )
            
            # Track metrics
            metrics = {
                'source_lines': len(source_code.splitlines()),
                'source_size_bytes': len(source_code),
                'state_declarations': source_code.count('state '),
                'observer_declarations': source_code.count('observer '),
                'measurements': source_code.count('measure '),
                'gate_applications': source_code.count('apply '),
                'simulations': source_code.count('simulate'),
                'visualizations': source_code.count('visualize')
            }
            
            # Lexical analysis
            if self.verbose:
                print(f"  Lexing {program_name}...")
            lexer = RecursiaLexer(source_code)
            tokens = lexer.tokenize()
            
            # Parsing
            if self.verbose:
                print(f"  Parsing {program_name}...")
            parser = RecursiaParser(tokens)
            ast = parser.parse()
            
            if parser.errors:
                error = parser.errors[0]
                return TestResult(
                    program_path=program_path,
                    program_name=program_name,
                    category=category,
                    status='syntax_error',
                    execution_time=time.time() - start_time,
                    error_message=str(error),
                    error_type="ParserError",
                    error_line=getattr(error, 'line', None),
                    metrics=metrics
                )
            
            # Semantic analysis
            if self.verbose:
                print(f"  Analyzing {program_name}...")
            analyzer = SemanticAnalyzer()
            semantic_errors = analyzer.analyze(ast, program_path)
            
            if semantic_errors:
                error = semantic_errors[0]
                return TestResult(
                    program_path=program_path,
                    program_name=program_name,
                    category=category,
                    status='semantic_error',
                    execution_time=time.time() - start_time,
                    error_message=str(error),
                    error_type="SemanticError",
                    error_line=getattr(error, 'line', None),
                    metrics=metrics
                )
            
            # Compilation
            if self.verbose:
                print(f"  Compiling {program_name}...")
            compilation_result = self.compiler.compile(source_code, target='quantum_simulator')
            
            if not compilation_result.success:
                return TestResult(
                    program_path=program_path,
                    program_name=program_name,
                    category=category,
                    status='semantic_error',
                    execution_time=time.time() - start_time,
                    error_message=compilation_result.errors[0] if compilation_result.errors else "Compilation failed",
                    error_type="CompilationError",
                    metrics=metrics
                )
            
            # Runtime execution with timeout
            if self.verbose:
                print(f"  Executing {program_name}...")
            
            # Use subprocess for better timeout control
            result = self._execute_with_timeout(program_path, metrics)
            result.execution_time = time.time() - start_time
            
            return result
            
        except Exception as e:
            return TestResult(
                program_path=program_path,
                program_name=program_name,
                category=category,
                status='runtime_error',
                execution_time=time.time() - start_time,
                error_message=str(e),
                error_type=type(e).__name__,
                error_details={'traceback': traceback.format_exc()},
                metrics=metrics if 'metrics' in locals() else None
            )
    
    def _execute_with_timeout(self, program_path: str, metrics: Dict[str, Any]) -> TestResult:
        """Execute a program with timeout using subprocess"""
        program_name = Path(program_path).stem
        category = Path(program_path).parent.name
        
        # Create a simple execution script
        exec_script = f"""
import sys
sys.path.insert(0, '.')
from src.recursia import main
import json
import time
import traceback
import psutil
import os

start_time = time.time()
start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

try:
    # Run the program
    result = main(['run', '{program_path}', '--no-interactive'])
    
    # Calculate metrics
    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # Success result
    print(json.dumps({{
        'status': 'success',
        'execution_time': end_time - start_time,
        'memory_usage_mb': end_memory - start_memory,
        'completion_percentage': 100.0
    }}))
    
except Exception as e:
    print(json.dumps({{
        'status': 'runtime_error',
        'error_message': str(e),
        'error_type': type(e).__name__,
        'traceback': traceback.format_exc()
    }}))
"""
        
        try:
            # Run the execution script with timeout
            process = subprocess.Popen(
                [sys.executable, '-c', exec_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            stdout, stderr = process.communicate(timeout=self.timeout)
            
            # Parse the result
            if stdout:
                try:
                    result_data = json.loads(stdout.strip().split('\n')[-1])
                    
                    return TestResult(
                        program_path=program_path,
                        program_name=program_name,
                        category=category,
                        status=result_data.get('status', 'runtime_error'),
                        execution_time=result_data.get('execution_time', 0.0),
                        error_message=result_data.get('error_message'),
                        error_type=result_data.get('error_type'),
                        error_details={'traceback': result_data.get('traceback')},
                        metrics=metrics,
                        memory_usage_mb=result_data.get('memory_usage_mb', 0.0),
                        completion_percentage=result_data.get('completion_percentage', 0.0)
                    )
                except json.JSONDecodeError:
                    # Fallback for non-JSON output
                    return TestResult(
                        program_path=program_path,
                        program_name=program_name,
                        category=category,
                        status='success' if process.returncode == 0 else 'runtime_error',
                        execution_time=0.0,
                        error_message=stderr if stderr else None,
                        metrics=metrics
                    )
            else:
                return TestResult(
                    program_path=program_path,
                    program_name=program_name,
                    category=category,
                    status='runtime_error',
                    execution_time=0.0,
                    error_message=stderr if stderr else "No output from program",
                    metrics=metrics
                )
                
        except subprocess.TimeoutExpired:
            process.kill()
            return TestResult(
                program_path=program_path,
                program_name=program_name,
                category=category,
                status='timeout',
                execution_time=self.timeout,
                error_message=f"Program execution exceeded timeout of {self.timeout} seconds",
                error_type="TimeoutError",
                metrics=metrics
            )
    
    def test_all_programs(self) -> None:
        """Test all quantum programs in the library"""
        print("\n" + "="*80)
        print("QUANTUM PROGRAMS COMPREHENSIVE TEST SUITE")
        print("="*80)
        print(f"Timeout per program: {self.timeout} seconds")
        print(f"Verbose mode: {'ON' if self.verbose else 'OFF'}")
        
        # Find all programs
        programs = self.find_all_programs()
        self.total_programs = len(programs)
        
        print(f"\nFound {self.total_programs} quantum programs to test")
        print("-"*80)
        
        # Test each program
        for i, (program_path, program_name, category) in enumerate(programs, 1):
            print(f"\n[{i}/{self.total_programs}] Testing: {program_path}")
            
            result = self.test_program(program_path, program_name, category)
            self.results.append(result)
            
            # Update statistics
            if result.status == 'success':
                self.successful_programs += 1
                print(f"  ✅ SUCCESS (execution time: {result.execution_time:.2f}s)")
            elif result.status == 'syntax_error':
                self.syntax_errors += 1
                self.failed_programs += 1
                print(f"  ❌ SYNTAX ERROR: {result.error_message}")
            elif result.status == 'semantic_error':
                self.semantic_errors += 1
                self.failed_programs += 1
                print(f"  ❌ SEMANTIC ERROR: {result.error_message}")
            elif result.status == 'runtime_error':
                self.runtime_errors += 1
                self.failed_programs += 1
                print(f"  ❌ RUNTIME ERROR: {result.error_message}")
            elif result.status == 'timeout':
                self.timeouts += 1
                self.failed_programs += 1
                print(f"  ⏱️  TIMEOUT: {result.error_message}")
            
            # Print metrics if available
            if result.metrics and self.verbose:
                print(f"  📊 Metrics: {result.metrics}")
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive test report"""
        # Group results by category
        by_category = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = {
                    'total': 0,
                    'success': 0,
                    'syntax_errors': 0,
                    'semantic_errors': 0,
                    'runtime_errors': 0,
                    'timeouts': 0,
                    'programs': []
                }
            
            cat = by_category[result.category]
            cat['total'] += 1
            cat['programs'].append(result.to_dict())
            
            if result.status == 'success':
                cat['success'] += 1
            elif result.status == 'syntax_error':
                cat['syntax_errors'] += 1
            elif result.status == 'semantic_error':
                cat['semantic_errors'] += 1
            elif result.status == 'runtime_error':
                cat['runtime_errors'] += 1
            elif result.status == 'timeout':
                cat['timeouts'] += 1
        
        # Calculate overall statistics
        total_execution_time = sum(r.execution_time for r in self.results)
        avg_execution_time = total_execution_time / len(self.results) if self.results else 0
        
        report = {
            'test_date': datetime.now().isoformat(),
            'summary': {
                'total_programs': self.total_programs,
                'successful_programs': self.successful_programs,
                'failed_programs': self.failed_programs,
                'success_rate': (self.successful_programs / self.total_programs * 100) if self.total_programs > 0 else 0,
                'syntax_errors': self.syntax_errors,
                'semantic_errors': self.semantic_errors,
                'runtime_errors': self.runtime_errors,
                'timeouts': self.timeouts,
                'total_execution_time': total_execution_time,
                'average_execution_time': avg_execution_time
            },
            'by_category': by_category,
            'failed_programs': [r.to_dict() for r in self.results if r.status != 'success'],
            'all_results': [r.to_dict() for r in self.results]
        }
        
        return report
    
    def print_summary(self) -> None:
        """Print a summary of test results"""
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\nTotal Programs Tested: {self.total_programs}")
        print(f"Successful: {self.successful_programs} ({self.successful_programs/self.total_programs*100:.1f}%)")
        print(f"Failed: {self.failed_programs} ({self.failed_programs/self.total_programs*100:.1f}%)")
        
        print(f"\nError Breakdown:")
        print(f"  Syntax Errors: {self.syntax_errors}")
        print(f"  Semantic Errors: {self.semantic_errors}")
        print(f"  Runtime Errors: {self.runtime_errors}")
        print(f"  Timeouts: {self.timeouts}")
        
        # Print failed programs
        if self.failed_programs > 0:
            print(f"\nFailed Programs:")
            for result in self.results:
                if result.status != 'success':
                    print(f"  - {result.program_path}")
                    print(f"    Status: {result.status}")
                    print(f"    Error: {result.error_message}")
                    if result.error_line:
                        print(f"    Line: {result.error_line}")
    
    def save_report(self, filename: str = None) -> str:
        """Save the test report to a JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"quantum_programs_test_report_{timestamp}.json"
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {filename}")
        return filename


def main():
    """Main entry point for the test suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test all quantum programs in the library')
    parser.add_argument('--timeout', type=float, default=30.0,
                        help='Timeout for each program execution (seconds)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for test report')
    parser.add_argument('--category', type=str, default=None,
                        help='Test only programs in a specific category')
    parser.add_argument('--program', type=str, default=None,
                        help='Test only a specific program')
    
    args = parser.parse_args()
    
    # Create validator
    validator = QuantumProgramValidator(timeout=args.timeout, verbose=args.verbose)
    
    # Override categories if specific category requested
    if args.category:
        validator.categories = [args.category]
    
    # Test programs
    if args.program:
        # Test single program
        program_path = args.program
        program_name = Path(program_path).stem
        category = Path(program_path).parent.name
        
        print(f"Testing single program: {program_path}")
        result = validator.test_program(program_path, program_name, category)
        validator.results = [result]
        
        if result.status == 'success':
            validator.successful_programs = 1
        else:
            validator.failed_programs = 1
            
        validator.total_programs = 1
    else:
        # Test all programs
        validator.test_all_programs()
    
    # Print summary
    validator.print_summary()
    
    # Save report
    validator.save_report(args.output)
    
    # Exit with appropriate code
    sys.exit(0 if validator.failed_programs == 0 else 1)


if __name__ == '__main__':
    main()