#!/usr/bin/env python3
"""
Quick Recursia Program Syntax Fixer
Applies syntax fixes to all .recursia programs without full execution
"""

import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Common syntax fixes based on grammar alignment
SYNTAX_FIXES = {
    # Fix simulate statement syntax
    r'simulate\s+(\w+)\s+using\s+(\w+)': r'simulate using \2 for \1 steps',
    r'simulate\s+for\s+(\d+)\s+using\s+(\w+)': r'simulate using \2 for \1 steps',
    
    # Fix field names to normalized versions
    r'\bqubits\s*:': 'state_qubits:',
    r'\bcoherence\s*:': 'state_coherence:',
    r'\bentropy\s*:': 'state_entropy:',
    r'\btype\s*:': 'observer_type:',
    r'\bfocus\s*:': 'observer_focus:',
    r'\bphase\s*:': 'observer_phase:',
    r'\bcollapse_threshold\s*:': 'observer_collapse_threshold:',
    r'\bself_awareness\s*:': 'observer_self_awareness:',
    
    # Fix observer type values to lowercase with underscores
    r'"QuantumObserver"': '"quantum_observer"',
    r'"StandardObserver"': '"standard_observer"',
    r'"RecursiveObserver"': '"recursive_observer"',
    r'"ConsciousObserver"': '"conscious_observer"',
    r'"CollectiveObserver"': '"collective_observer"',
    r'"MetaObserver"': '"meta_observer"',
    r'"NestedObserver"': '"nested_observer"',
    r'"DistributedObserver"': '"distributed_observer"',
    r'"SubconsciousObserver"': '"subconscious_observer"',
    
    # Fix print statements (ensure semicolon)
    r'print\s+"([^"]+)"\s*$': r'print "\1";',
    r'print\s+\'([^\']+)\'\s*$': r'print "\1";',
    
    # Fix visualization statements
    r'visualize\s+(\w+)\s+as\s+(\w+)': r'visualize \1 mode \2',
    r'visualize\s+state\s+(\w+)': r'visualize \1 mode state_evolution',
    
    # Ensure semicolons on statements
    r'(apply\s+\w+_gate\s+to\s+\w+(?:\s+qubit\s+\d+)?(?:\s+control\s+\d+)?)\s*$': r'\1;',
    r'(measure\s+\w+(?:\s+qubit\s+\d+)?\s+by\s+\w+)\s*$': r'\1;',
    r'(entangle\s+\w+\s+qubit\s+\d+\s*,\s*\w+\s+qubit\s+\d+)\s*$': r'\1;',
    r'(teleport\s+\w+\s+qubit\s+\d+\s*->\s*\w+\s+qubit\s+\d+)\s*$': r'\1;',
    r'(cohere\s+\w+\s+to\s+level\s+[\d.]+)\s*$': r'\1;',
    r'(render\s+\w+)\s*$': r'\1;',
}

def find_all_programs() -> List[Path]:
    """Find all .recursia files in examples and quantum_programs folders"""
    programs = []
    
    # Search directories
    search_dirs = [
        project_root / 'examples',
        project_root / 'quantum_programs',
    ]
    
    for dir_path in search_dirs:
        if dir_path.exists():
            programs.extend(dir_path.rglob('*.recursia'))
    
    # Also check validation_programs folder
    validation_dir = project_root / 'validation_programs'
    if validation_dir.exists():
        programs.extend(validation_dir.rglob('*.recursia'))
    
    return sorted(set(programs))

def apply_syntax_fixes(content: str) -> Tuple[str, List[str]]:
    """Apply syntax fixes to program content"""
    fixed_content = content
    fixes_applied = []
    
    # Apply each fix pattern
    for pattern, replacement in SYNTAX_FIXES.items():
        if re.search(pattern, fixed_content, re.MULTILINE):
            fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.MULTILINE)
            fixes_applied.append(f"Applied: {pattern} -> {replacement}")
    
    return fixed_content, fixes_applied

def fix_program(file_path: Path) -> bool:
    """Fix a single program file"""
    try:
        # Read original content
        original_content = file_path.read_text()
        
        # Apply fixes
        fixed_content, fixes = apply_syntax_fixes(original_content)
        
        # Only write if changes were made
        if fixes:
            # Create backup
            backup_path = file_path.with_suffix('.recursia.bak')
            backup_path.write_text(original_content)
            
            # Write fixed content
            file_path.write_text(fixed_content)
            
            print(f"✅ Fixed {file_path.name} ({len(fixes)} fixes)")
            for fix in fixes[:3]:  # Show first 3 fixes
                print(f"   - {fix}")
            if len(fixes) > 3:
                print(f"   ... and {len(fixes) - 3} more")
            return True
        else:
            print(f"✓ {file_path.name} (no fixes needed)")
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path.name}: {e}")
        return False

def main():
    """Main entry point"""
    print("Quick Recursia Program Syntax Fixer")
    print("=" * 50)
    
    # Find all programs
    programs = find_all_programs()
    print(f"Found {len(programs)} .recursia programs")
    print()
    
    # Fix each program
    fixed_count = 0
    for program in programs:
        if fix_program(program):
            fixed_count += 1
    
    print()
    print(f"Summary: Fixed {fixed_count} out of {len(programs)} programs")
    
    # Now test a few key programs
    print()
    print("Testing key programs...")
    print("=" * 50)
    
    test_programs = [
        project_root / 'examples' / 'basic' / 'hello_world.recursia',
        project_root / 'quantum_programs' / 'basic' / 'quantum_superposition.recursia',
        project_root / 'quantum_programs' / 'basic' / 'bell_state_creation.recursia',
    ]
    
    from src.core.lexer import RecursiaLexer
    from src.core.parser import RecursiaParser
    
    for program_path in test_programs:
        if program_path.exists():
            print(f"\nTesting {program_path.name}...")
            try:
                content = program_path.read_text()
                lexer = RecursiaLexer(content)
                tokens = lexer.tokenize()
                parser = RecursiaParser(tokens)
                ast = parser.parse()
                print(f"  ✅ Parses successfully!")
            except Exception as e:
                print(f"  ❌ Parse error: {e}")

if __name__ == '__main__':
    main()