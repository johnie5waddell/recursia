#!/usr/bin/env python3
"""
Comprehensive Quantum Programs Validation Script
Tests all quantum programs in the library to ensure they compile and run correctly
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Result of testing a quantum program"""
    program_path: str
    category: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    output_preview: Optional[str] = None


class QuantumProgramValidator:
    """Validates all quantum programs in the library"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.quantum_programs_dir = project_root / "quantum_programs"
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def find_all_programs(self) -> List[Path]:
        """Find all .recursia files in quantum_programs directory"""
        programs = []
        for root, dirs, files in os.walk(self.quantum_programs_dir):
            for file in files:
                if file.endswith('.recursia'):
                    programs.append(Path(root) / file)
        return sorted(programs)
    
    def get_category(self, program_path: Path) -> str:
        """Extract category from program path"""
        relative_path = program_path.relative_to(self.quantum_programs_dir)
        parts = relative_path.parts
        return parts[0] if len(parts) > 1 else "uncategorized"
    
    def test_program(self, program_path: Path) -> TestResult:
        """Test a single quantum program"""
        start_time = time.time()
        category = self.get_category(program_path)
        
        try:
            # Run the program using recursia CLI
            cmd = [
                "recursia",
                "run",
                str(program_path)
            ]
            
            # Set up environment
            env = os.environ.copy()
            env['PYTHONPATH'] = str(self.project_root)
            
            # Run with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120,  # 2 minute timeout
                cwd=str(self.project_root),
                env=env
            )
            
            execution_time = time.time() - start_time
            
            # Check for success
            success = result.returncode == 0 and "✓ Execution completed successfully" in result.stdout
            
            # Extract error if failed
            error_message = None
            if not success:
                if result.stderr:
                    error_message = result.stderr.strip()
                elif "Error" in result.stdout or "Exception" in result.stdout:
                    # Extract error from stdout
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if "Error" in line or "Exception" in line:
                            error_message = '\n'.join(lines[i:i+5])
                            break
            
            # Get output preview
            output_lines = result.stdout.split('\n')
            output_preview = '\n'.join(output_lines[-20:]) if output_lines else ""
            
            return TestResult(
                program_path=str(program_path.relative_to(self.project_root)),
                category=category,
                success=success,
                execution_time=execution_time,
                error_message=error_message,
                output_preview=output_preview
            )
            
        except subprocess.TimeoutExpired:
            return TestResult(
                program_path=str(program_path.relative_to(self.project_root)),
                category=category,
                success=False,
                execution_time=120.0,
                error_message="Timeout: Program took longer than 2 minutes to execute"
            )
        except Exception as e:
            return TestResult(
                program_path=str(program_path.relative_to(self.project_root)),
                category=category,
                success=False,
                execution_time=time.time() - start_time,
                error_message=f"Unexpected error: {str(e)}"
            )
    
    def run_validation(self) -> Dict[str, any]:
        """Run validation on all quantum programs"""
        print("=" * 80)
        print("QUANTUM PROGRAMS VALIDATION")
        print("=" * 80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Find all programs
        programs = self.find_all_programs()
        print(f"Found {len(programs)} quantum programs to test")
        print()
        
        # Test each program
        for i, program in enumerate(programs, 1):
            print(f"[{i}/{len(programs)}] Testing: {program.relative_to(self.project_root)}")
            result = self.test_program(program)
            self.results.append(result)
            
            if result.success:
                print(f"  ✓ SUCCESS ({result.execution_time:.2f}s)")
            else:
                print(f"  ✗ FAILED: {result.error_message}")
            print()
        
        # Generate summary
        total_time = time.time() - self.start_time
        return self.generate_summary(total_time)
    
    def generate_summary(self, total_time: float) -> Dict[str, any]:
        """Generate validation summary"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        # Group by category
        by_category = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = {'success': 0, 'failed': 0, 'programs': []}
            
            if result.success:
                by_category[result.category]['success'] += 1
            else:
                by_category[result.category]['failed'] += 1
            by_category[result.category]['programs'].append(result)
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_programs': len(self.results),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.results) * 100 if self.results else 0,
            'total_time': total_time,
            'average_time': sum(r.execution_time for r in self.results) / len(self.results) if self.results else 0,
            'by_category': by_category,
            'failed_programs': [
                {
                    'path': r.program_path,
                    'category': r.category,
                    'error': r.error_message
                }
                for r in failed
            ]
        }
        
        return summary
    
    def print_summary(self, summary: Dict[str, any]):
        """Print validation summary"""
        print("=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total Programs: {summary['total_programs']}")
        print(f"Successful: {summary['successful']} ({summary['success_rate']:.1f}%)")
        print(f"Failed: {summary['failed']}")
        print(f"Total Time: {summary['total_time']:.1f}s")
        print(f"Average Time: {summary['average_time']:.1f}s per program")
        print()
        
        print("Results by Category:")
        print("-" * 40)
        for category, stats in summary['by_category'].items():
            total = stats['success'] + stats['failed']
            success_rate = stats['success'] / total * 100 if total > 0 else 0
            print(f"{category:20} {stats['success']:3}/{total:3} ({success_rate:5.1f}%)")
        print()
        
        if summary['failed'] > 0:
            print("Failed Programs:")
            print("-" * 40)
            for failed in summary['failed_programs']:
                print(f"• {failed['path']}")
                print(f"  Error: {failed['error']}")
                print()
    
    def save_report(self, summary: Dict[str, any], output_path: Optional[Path] = None):
        """Save detailed validation report"""
        if output_path is None:
            output_path = self.project_root / f"quantum_programs_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert TestResult objects to dictionaries
        detailed_results = []
        for r in self.results:
            detailed_results.append({
                'program_path': r.program_path,
                'category': r.category,
                'success': r.success,
                'execution_time': r.execution_time,
                'error_message': r.error_message,
                'output_preview': r.output_preview
            })
        
        summary['detailed_results'] = detailed_results
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Detailed report saved to: {output_path}")


def main():
    """Main validation entry point"""
    # Set up project root
    project_root = Path(__file__).parent.parent.parent
    
    # Create validator
    validator = QuantumProgramValidator(project_root)
    
    # Run validation
    summary = validator.run_validation()
    
    # Print summary
    validator.print_summary(summary)
    
    # Save detailed report
    validator.save_report(summary)
    
    # Exit with appropriate code
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == "__main__":
    main()