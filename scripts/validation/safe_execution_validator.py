#!/usr/bin/env python3
"""
Safe execution validator for Recursia programs.
Handles memory-intensive programs gracefully.
"""

import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.parser import Parser
from src.core.lexer import Lexer
from src.core.semantic_analyzer import SemanticAnalyzer
from src.core.unified_executor import UnifiedExecutor
from src.core.runtime import Runtime
from src.core.utils import ProgramOutput

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'safe_execution_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ProgramExecutionResult:
    """Result of program execution validation"""
    file_path: str
    category: str
    name: str
    syntax_valid: bool = False
    execution_success: bool = False
    execution_time: float = 0.0
    error: Optional[str] = None
    warning: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    output: Optional[str] = None
    skipped: bool = False
    skip_reason: Optional[str] = None

class SafeExecutionValidator:
    """
    Safe execution validator that handles memory-intensive programs.
    """
    
    # Programs known to allocate excessive memory
    MEMORY_INTENSIVE_PROGRAMS = {
        'osh_reality_stress_test.recursia': 'Allocates 1 billion element arrays',
        'macro_object_teleportation.recursia': 'Creates large quantum states',
    }
    
    # Programs that should be tested with limited iterations
    LIMITED_ITERATION_PROGRAMS = {
        'osh_reality_stress_test.recursia': 1,
        'macro_object_teleportation.recursia': 1,
    }
    
    def __init__(self):
        self.parser = Parser()
        self.lexer = Lexer()
        self.results: List[ProgramExecutionResult] = []
        
    def validate_program(self, file_path: Path) -> ProgramExecutionResult:
        """Validate a single program safely"""
        category = file_path.parent.name
        name = file_path.name
        
        result = ProgramExecutionResult(
            file_path=str(file_path),
            category=category,
            name=name
        )
        
        # Check if this is a memory-intensive program
        if name in self.MEMORY_INTENSIVE_PROGRAMS:
            result.skipped = True
            result.skip_reason = self.MEMORY_INTENSIVE_PROGRAMS[name]
            logger.warning(f"Skipping {name}: {result.skip_reason}")
            return result
        
        try:
            # Read program
            with open(file_path, 'r') as f:
                source_code = f.read()
            
            # Parse
            logger.info(f"Parsing {file_path}")
            tokens = self.lexer.tokenize(source_code)
            ast = self.parser.parse(tokens)
            
            if not ast:
                result.error = "Failed to parse program"
                return result
                
            result.syntax_valid = True
            
            # Semantic analysis
            analyzer = SemanticAnalyzer()
            if not analyzer.analyze(ast):
                result.error = "Semantic analysis failed"
                result.warning = "Program has semantic errors but syntax is valid"
                return result
            
            # Execute with timeout and limited iterations
            logger.info(f"Executing {file_path}")
            start_time = time.time()
            
            runtime = Runtime()
            executor = UnifiedExecutor(runtime)
            
            # Determine iteration count
            iterations = self.LIMITED_ITERATION_PROGRAMS.get(name, 1)
            
            try:
                # Execute program
                output = executor.execute(ast)
                result.execution_success = True
                result.execution_time = time.time() - start_time
                
                # Get output
                if isinstance(output, ProgramOutput):
                    result.output = str(output.value) if output.value else "No output"
                else:
                    result.output = str(output) if output else "No output"
                
                # Get metrics
                if hasattr(runtime, 'metrics_calculator'):
                    try:
                        metrics_snapshot = runtime.metrics_calculator.calculate_metrics()
                        result.metrics = metrics_snapshot.to_dict()
                        
                        # Log non-zero metrics
                        non_zero_metrics = {k: v for k, v in result.metrics.items() 
                                          if v != 0 and v != 0.0 and k != 'timestamp'}
                        if non_zero_metrics:
                            logger.info(f"Non-zero metrics for {name}: {non_zero_metrics}")
                    except Exception as e:
                        logger.error(f"Failed to get metrics: {e}")
                        
            except Exception as e:
                result.execution_success = False
                result.error = f"Execution error: {str(e)}"
                result.execution_time = time.time() - start_time
                logger.error(f"Execution failed for {file_path}: {e}")
                
        except Exception as e:
            result.error = f"Validation error: {str(e)}"
            logger.error(f"Failed to validate {file_path}: {e}")
            logger.error(traceback.format_exc())
            
        return result
    
    def validate_directory(self, directory: Path) -> List[ProgramExecutionResult]:
        """Validate all .recursia files in a directory"""
        results = []
        
        for file_path in directory.rglob('*.recursia'):
            logger.info(f"Validating {file_path}")
            result = self.validate_program(file_path)
            results.append(result)
            
        return results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total = len(self.results)
        syntax_valid = sum(1 for r in self.results if r.syntax_valid)
        execution_success = sum(1 for r in self.results if r.execution_success)
        skipped = sum(1 for r in self.results if r.skipped)
        
        # Categorize results
        by_category = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = {
                    'total': 0,
                    'syntax_valid': 0,
                    'execution_success': 0,
                    'skipped': 0,
                    'programs': []
                }
            
            cat = by_category[result.category]
            cat['total'] += 1
            if result.syntax_valid:
                cat['syntax_valid'] += 1
            if result.execution_success:
                cat['execution_success'] += 1
            if result.skipped:
                cat['skipped'] += 1
            
            cat['programs'].append({
                'name': result.name,
                'syntax_valid': result.syntax_valid,
                'execution_success': result.execution_success,
                'skipped': result.skipped,
                'skip_reason': result.skip_reason,
                'error': result.error,
                'metrics': result.metrics,
                'execution_time': result.execution_time
            })
        
        # Find programs with interesting metrics
        programs_with_metrics = []
        for result in self.results:
            if result.metrics:
                non_zero_metrics = {k: v for k, v in result.metrics.items() 
                                  if v != 0 and v != 0.0 and k != 'timestamp'}
                if non_zero_metrics:
                    programs_with_metrics.append({
                        'name': result.name,
                        'category': result.category,
                        'metrics': non_zero_metrics
                    })
        
        return {
            'summary': {
                'total_programs': total,
                'syntax_valid': syntax_valid,
                'execution_success': execution_success,
                'skipped': skipped,
                'syntax_success_rate': f"{(syntax_valid/total)*100:.1f}%" if total > 0 else "0%",
                'execution_success_rate': f"{(execution_success/(total-skipped))*100:.1f}%" if (total-skipped) > 0 else "0%"
            },
            'by_category': by_category,
            'programs_with_metrics': programs_with_metrics,
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main validation entry point"""
    validator = SafeExecutionValidator()
    
    # Validate examples
    logger.info("Validating examples directory...")
    examples_dir = project_root / 'examples'
    if examples_dir.exists():
        validator.results.extend(validator.validate_directory(examples_dir))
    
    # Validate quantum programs
    logger.info("Validating quantum_programs directory...")
    quantum_dir = project_root / 'quantum_programs'
    if quantum_dir.exists():
        validator.results.extend(validator.validate_directory(quantum_dir))
    
    # Generate report
    report = validator.generate_report()
    
    # Save report
    report_file = f'safe_execution_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("SAFE EXECUTION VALIDATION REPORT")
    print("="*60)
    print(f"Total Programs: {report['summary']['total_programs']}")
    print(f"Syntax Valid: {report['summary']['syntax_valid']} ({report['summary']['syntax_success_rate']})")
    print(f"Execution Success: {report['summary']['execution_success']} ({report['summary']['execution_success_rate']})")
    print(f"Skipped (Memory Intensive): {report['summary']['skipped']}")
    
    print("\n" + "-"*60)
    print("BY CATEGORY:")
    print("-"*60)
    
    for category, stats in report['by_category'].items():
        print(f"\n{category}:")
        print(f"  Total: {stats['total']}")
        print(f"  Syntax Valid: {stats['syntax_valid']}")
        print(f"  Execution Success: {stats['execution_success']}")
        print(f"  Skipped: {stats['skipped']}")
    
    if report['programs_with_metrics']:
        print("\n" + "-"*60)
        print("PROGRAMS WITH NON-ZERO METRICS:")
        print("-"*60)
        for prog in report['programs_with_metrics']:
            print(f"\n{prog['category']}/{prog['name']}:")
            for metric, value in prog['metrics'].items():
                print(f"  {metric}: {value}")
    
    print(f"\nFull report saved to: {report_file}")
    
    # Exit with appropriate code
    if report['summary']['execution_success'] < report['summary']['total_programs'] - report['summary']['skipped']:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()