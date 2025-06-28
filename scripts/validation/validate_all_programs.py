#!/usr/bin/env python3
"""
Production-Ready Recursia Program Validator and Fixer
Ensures all programs are grammatically correct and executable for release
"""

import os
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.lexer import RecursiaLexer
from src.core.parser import RecursiaParser
from src.core.semantic_analyzer import SemanticAnalyzer
from src.core.runtime import RecursiaRuntime
from src.core.unified_executor import UnifiedExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProgramValidationResult:
    """Result of validating a single program"""
    file_path: str
    category: str
    name: str
    valid: bool = False
    executable: bool = False
    parse_errors: List[str] = field(default_factory=list)
    semantic_errors: List[str] = field(default_factory=list)
    execution_errors: List[str] = field(default_factory=list)
    fixes_applied: List[str] = field(default_factory=list)
    description: str = ""
    features: List[str] = field(default_factory=list)
    metrics_generated: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


class ProductionProgramValidator:
    """
    Enterprise-grade validator ensuring all Recursia programs are production-ready
    """
    
    def __init__(self):
        self.results: List[ProgramValidationResult] = []
        self.program_metadata: Dict[str, Dict[str, Any]] = {}
        self._load_program_metadata()
        
    def _load_program_metadata(self):
        """Load metadata about programs for categorization"""
        self.program_metadata = {
            # Basic examples
            "hello_world.recursia": {
                "description": "Introduction to Recursia quantum programming",
                "features": ["quantum states", "gates", "visualization"],
                "difficulty": "beginner"
            },
            
            # Complete demonstrations
            "coherence_collapse_demo.recursia": {
                "description": "Demonstrates coherence field collapse dynamics",
                "features": ["coherence", "field dynamics", "observers"],
                "difficulty": "intermediate"
            },
            "consciousness_emergence_demo.recursia": {
                "description": "Shows consciousness emergence from quantum substrates",
                "features": ["consciousness", "emergence", "meta-observers"],
                "difficulty": "advanced"
            },
            "entanglement_dynamics_demo.recursia": {
                "description": "Explores entanglement evolution and dynamics",
                "features": ["entanglement", "evolution", "visualization"],
                "difficulty": "intermediate"
            },
            "memory_strain_gravity_demo.recursia": {
                "description": "Memory field strain and gravitational effects",
                "features": ["memory fields", "gravity", "OSH physics"],
                "difficulty": "advanced"
            },
            "observer_decoherence_demo.recursia": {
                "description": "Observer-induced decoherence patterns",
                "features": ["observers", "decoherence", "measurement"],
                "difficulty": "intermediate"
            },
            "quantum_teleportation_consciousness.recursia": {
                "description": "Consciousness-aware quantum teleportation",
                "features": ["teleportation", "consciousness", "entanglement"],
                "difficulty": "advanced"
            },
            
            # OSH demonstrations
            "gravitational_wave_echoes.recursia": {
                "description": "Searches for OSH signatures in gravitational waves",
                "features": ["gravitational waves", "OSH predictions", "analysis"],
                "difficulty": "expert"
            },
            "information_curvature_gravity.recursia": {
                "description": "Information geometry and gravitational emergence",
                "features": ["information geometry", "gravity", "curvature"],
                "difficulty": "expert"
            },
            "observer_collapse_coherence.recursia": {
                "description": "Observer-driven coherence collapse mechanics",
                "features": ["observers", "coherence", "collapse"],
                "difficulty": "advanced"
            },
            "recursive_memory_time.recursia": {
                "description": "Recursive memory fields and time emergence",
                "features": ["recursion", "memory", "time"],
                "difficulty": "expert"
            },
            
            # Quantum programs - Basic
            "bell_state_creation.recursia": {
                "description": "Creates and analyzes Bell states",
                "features": ["Bell states", "entanglement", "measurement"],
                "difficulty": "beginner"
            },
            "quantum_superposition.recursia": {
                "description": "Basic quantum superposition demonstration",
                "features": ["superposition", "measurement", "visualization"],
                "difficulty": "beginner"
            },
            
            # Advanced quantum programs
            "consciousness_enhanced_teleportation.recursia": {
                "description": "Teleportation with consciousness field coupling",
                "features": ["teleportation", "consciousness", "advanced"],
                "difficulty": "advanced"
            },
            "quantum_error_correction_consciousness.recursia": {
                "description": "Error correction using consciousness principles",
                "features": ["error correction", "consciousness", "reliability"],
                "difficulty": "advanced"
            },
            
            # OSH calculations and predictions
            "compression_principle_optimizer.recursia": {
                "description": "Optimizes compression in OSH framework",
                "features": ["compression", "optimization", "OSH"],
                "difficulty": "expert"
            },
            "rsp_dimensional_analysis.recursia": {
                "description": "Analyzes RSP across dimensions",
                "features": ["RSP", "dimensions", "analysis"],
                "difficulty": "expert"
            },
            "cmb_complexity_analysis.recursia": {
                "description": "Analyzes CMB for OSH signatures",
                "features": ["CMB", "complexity", "cosmology"],
                "difficulty": "expert"
            },
            "eeg_cosmic_resonance.recursia": {
                "description": "Searches for EEG-cosmic correlations",
                "features": ["EEG", "cosmic", "consciousness"],
                "difficulty": "expert"
            }
        }
    
    def validate_all_programs(self) -> Dict[str, Any]:
        """Validate all programs and return comprehensive report"""
        logger.info("Starting production validation of all Recursia programs")
        
        # Find all programs
        program_files = self._find_all_programs()
        logger.info(f"Found {len(program_files)} programs to validate")
        
        # Validate each program
        for file_path in program_files:
            result = self._validate_program(file_path)
            self.results.append(result)
            
            # Log progress
            status = "✅" if result.valid and result.executable else "❌"
            logger.info(f"{status} {result.name}: Valid={result.valid}, Executable={result.executable}")
        
        # Generate comprehensive report
        report = self._generate_report()
        
        # Save report
        self._save_report(report)
        
        return report
    
    def _find_all_programs(self) -> List[Path]:
        """Find all .recursia programs in the project"""
        programs = []
        
        for directory in ['examples', 'quantum_programs']:
            dir_path = project_root / directory
            if dir_path.exists():
                programs.extend(dir_path.rglob('*.recursia'))
        
        return sorted(programs)
    
    def _validate_program(self, file_path: Path) -> ProgramValidationResult:
        """Validate a single program"""
        # Determine category and name
        relative_path = file_path.relative_to(project_root)
        category = str(relative_path.parent)
        name = file_path.name
        
        result = ProgramValidationResult(
            file_path=str(file_path),
            category=category,
            name=name
        )
        
        # Get metadata
        metadata = self.program_metadata.get(name, {})
        result.description = metadata.get('description', '')
        result.features = metadata.get('features', [])
        
        try:
            # Read program content
            content = file_path.read_text()
            
            # Apply any necessary fixes
            fixed_content, fixes = self._apply_syntax_fixes(content)
            if fixes:
                result.fixes_applied = fixes
                # Write fixed content back
                file_path.write_text(fixed_content)
                content = fixed_content
            
            # Parse program
            try:
                lexer = RecursiaLexer(content)
                tokens = lexer.tokenize()
                parser = RecursiaParser(tokens)
                ast = parser.parse()
                result.valid = True
            except Exception as e:
                result.parse_errors.append(str(e))
                return result
            
            # Semantic analysis
            try:
                analyzer = SemanticAnalyzer()
                analyzer.analyze(ast)
            except Exception as e:
                result.semantic_errors.append(str(e))
                # Continue to test execution anyway
            
            # Test execution
            try:
                runtime = RecursiaRuntime()
                executor = UnifiedExecutor(runtime)
                
                start_time = time.time()
                exec_result = executor.execute(ast)
                result.execution_time = time.time() - start_time
                
                if exec_result['success']:
                    result.executable = True
                    # Extract metrics
                    if 'metrics' in exec_result:
                        result.metrics_generated = self._extract_key_metrics(exec_result['metrics'])
                else:
                    result.execution_errors = exec_result.get('errors', ['Unknown execution error'])
                    
            except Exception as e:
                result.execution_errors.append(str(e))
        
        except Exception as e:
            result.parse_errors.append(f"Failed to read file: {e}")
        
        return result
    
    def _apply_syntax_fixes(self, content: str) -> Tuple[str, List[str]]:
        """Apply production-ready syntax fixes"""
        import re
        
        fixes_applied = []
        fixed_content = content
        
        # Comprehensive syntax fixes based on updated grammar
        fixes = {
            # Field normalizations (already applied by previous script)
            r'\bqubits\s*:': 'state_qubits:',
            r'\bcoherence\s*:': 'state_coherence:',
            r'\bentropy\s*:': 'state_entropy:',
            r'\bfocus\s*:': 'observer_focus:',
            r'\bphase\s*:': 'observer_phase:',
            
            # Observer type normalization
            r'"QuantumObserver"': '"quantum_observer"',
            r'"ConsciousObserver"': '"conscious_observer"',
            r'"RecursiveObserver"': '"recursive_observer"',
            
            # Ensure proper statement endings
            r'(apply\s+\w+_gate\s+to\s+\w+(?:\s+qubit\s+\d+)?)\s*(?!;)$': r'\1;',
            r'(measure\s+\w+(?:\s+qubit\s+\d+)?)\s*(?!;)$': r'\1;',
            r'(print\s+"[^"]+?")\s*(?!;)$': r'\1;',
        }
        
        for pattern, replacement in fixes.items():
            if re.search(pattern, fixed_content, re.MULTILINE):
                fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.MULTILINE)
                fixes_applied.append(f"{pattern} -> {replacement}")
        
        return fixed_content, fixes_applied
    
    def _extract_key_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics for reporting"""
        return {
            'state_count': metrics.get('state_count', 0),
            'observer_count': metrics.get('observer_count', 0),
            'rsp': metrics.get('rsp', 0),
            'coherence': metrics.get('coherence', 0),
            'information': metrics.get('information', 0),
            'complexity': metrics.get('complexity', 0)
        }
    
    def _generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        total = len(self.results)
        valid = sum(1 for r in self.results if r.valid)
        executable = sum(1 for r in self.results if r.executable)
        
        # Group by category
        by_category = {}
        for result in self.results:
            category = result.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(result)
        
        # Generate summary
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_programs': total,
                'valid_programs': valid,
                'executable_programs': executable,
                'success_rate': f"{(executable/total)*100:.1f}%" if total > 0 else "0%"
            },
            'by_category': {},
            'issues': [],
            'all_results': []
        }
        
        # Add category summaries
        for category, results in by_category.items():
            cat_total = len(results)
            cat_valid = sum(1 for r in results if r.valid)
            cat_exec = sum(1 for r in results if r.executable)
            
            report['by_category'][category] = {
                'total': cat_total,
                'valid': cat_valid,
                'executable': cat_exec,
                'programs': [
                    {
                        'name': r.name,
                        'description': r.description,
                        'features': r.features,
                        'valid': r.valid,
                        'executable': r.executable,
                        'errors': r.parse_errors + r.semantic_errors + r.execution_errors,
                        'fixes_applied': r.fixes_applied
                    }
                    for r in results
                ]
            }
        
        # Collect all issues
        for result in self.results:
            if not result.valid or not result.executable:
                issue = {
                    'program': result.name,
                    'category': result.category,
                    'parse_errors': result.parse_errors,
                    'semantic_errors': result.semantic_errors,
                    'execution_errors': result.execution_errors
                }
                report['issues'].append(issue)
        
        # Add all detailed results
        report['all_results'] = [
            {
                'file': r.file_path,
                'name': r.name,
                'category': r.category,
                'valid': r.valid,
                'executable': r.executable,
                'description': r.description,
                'features': r.features,
                'execution_time': r.execution_time,
                'metrics': r.metrics_generated,
                'fixes': r.fixes_applied
            }
            for r in self.results
        ]
        
        return report
    
    def _save_report(self, report: Dict[str, Any]):
        """Save validation report"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = project_root / f'validation_report_{timestamp}.json'
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {report_path}")
        
        # Also save a human-readable summary
        summary_path = project_root / f'validation_summary_{timestamp}.txt'
        with open(summary_path, 'w') as f:
            f.write("RECURSIA PROGRAM VALIDATION SUMMARY\n")
            f.write("="*60 + "\n\n")
            f.write(f"Generated: {report['timestamp']}\n\n")
            
            f.write("OVERALL SUMMARY:\n")
            f.write(f"  Total Programs: {report['summary']['total_programs']}\n")
            f.write(f"  Valid Programs: {report['summary']['valid_programs']}\n")
            f.write(f"  Executable Programs: {report['summary']['executable_programs']}\n")
            f.write(f"  Success Rate: {report['summary']['success_rate']}\n\n")
            
            f.write("BY CATEGORY:\n")
            for category, data in report['by_category'].items():
                f.write(f"\n  {category}:\n")
                f.write(f"    Total: {data['total']}\n")
                f.write(f"    Valid: {data['valid']}\n")
                f.write(f"    Executable: {data['executable']}\n")
                
                for prog in data['programs']:
                    status = "✅" if prog['executable'] else "❌"
                    f.write(f"    {status} {prog['name']}")
                    if prog['description']:
                        f.write(f" - {prog['description']}")
                    f.write("\n")
                    if prog['errors']:
                        for error in prog['errors']:
                            f.write(f"        ERROR: {error}\n")
            
            if report['issues']:
                f.write("\n\nISSUES REQUIRING ATTENTION:\n")
                for issue in report['issues']:
                    f.write(f"\n  {issue['program']} ({issue['category']}):\n")
                    all_errors = issue['parse_errors'] + issue['semantic_errors'] + issue['execution_errors']
                    for error in all_errors:
                        f.write(f"    - {error}\n")
        
        logger.info(f"Human-readable summary saved to {summary_path}")


def main():
    """Main entry point for production validation"""
    print("🚀 RECURSIA PRODUCTION VALIDATION")
    print("="*60)
    print("Validating all programs for production release...\n")
    
    validator = ProductionProgramValidator()
    report = validator.validate_all_programs()
    
    print("\n" + "="*60)
    print("VALIDATION COMPLETE")
    print("="*60)
    print(f"Total Programs: {report['summary']['total_programs']}")
    print(f"Valid Programs: {report['summary']['valid_programs']}")
    print(f"Executable Programs: {report['summary']['executable_programs']}")
    print(f"Success Rate: {report['summary']['success_rate']}")
    
    if report['issues']:
        print(f"\n⚠️  {len(report['issues'])} programs require attention")
    else:
        print("\n✅ All programs are production-ready!")
    
    print("\nReports saved - check validation_report_*.json and validation_summary_*.txt")


if __name__ == '__main__':
    main()