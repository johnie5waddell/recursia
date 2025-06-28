#!/usr/bin/env python3
"""
Quick syntax validation for all Recursia programs
Checks parsing and reports issues without full execution
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.lexer import RecursiaLexer
from src.core.parser import RecursiaParser
from src.core.semantic_analyzer import SemanticAnalyzer


def validate_program_syntax(file_path: Path) -> Tuple[bool, List[str]]:
    """Validate program syntax only"""
    errors = []
    
    try:
        content = file_path.read_text()
        
        # Parse
        lexer = RecursiaLexer(content)
        tokens = lexer.tokenize()
        parser = RecursiaParser(tokens)
        ast = parser.parse()
        
        # Semantic analysis
        analyzer = SemanticAnalyzer()
        analyzer.analyze(ast)
        
        return True, []
        
    except Exception as e:
        errors.append(str(e))
        return False, errors


def main():
    """Quick validation of all programs"""
    print("🔍 RECURSIA QUICK SYNTAX VALIDATION")
    print("="*60)
    
    # Find all programs
    programs = []
    for directory in ['examples', 'quantum_programs']:
        dir_path = project_root / directory
        if dir_path.exists():
            programs.extend(dir_path.rglob('*.recursia'))
    
    programs = sorted(programs)
    print(f"Found {len(programs)} programs to validate\n")
    
    # Validate each
    results = {}
    valid_count = 0
    
    for program in programs:
        relative_path = program.relative_to(project_root)
        valid, errors = validate_program_syntax(program)
        
        results[str(relative_path)] = {
            'valid': valid,
            'errors': errors
        }
        
        if valid:
            valid_count += 1
            print(f"✅ {relative_path}")
        else:
            print(f"❌ {relative_path}")
            for error in errors:
                print(f"   ERROR: {error}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {valid_count}/{len(programs)} programs are syntactically valid")
    print(f"Success Rate: {(valid_count/len(programs)*100):.1f}%")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = project_root / f'syntax_validation_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_programs': len(programs),
            'valid_programs': valid_count,
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # List programs needing attention
    issues = [p for p, r in results.items() if not r['valid']]
    if issues:
        print(f"\n⚠️  Programs needing attention ({len(issues)}):")
        for program in issues:
            print(f"  - {program}")


if __name__ == '__main__':
    main()