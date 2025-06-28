#!/usr/bin/env python3
"""Extract and test the Information-Curvature Coupling Analyzer"""

import re
from src.core.compiler import RecursiaCompiler

# Read the TypeScript file
with open('frontend/src/data/oshQuantumPrograms.ts', 'r') as f:
    content = f.read()

# Find the Information-Curvature Coupling Analyzer
pattern = r"id: 'osh_information_curvature'.*?code: `(.*?)`,"
match = re.search(pattern, content, re.DOTALL)

if match:
    code = match.group(1)
    
    # Save to file
    with open('test_full_analyzer.recursia', 'w') as f:
        f.write(code)
    
    print("✓ Code extracted successfully")
    print(f"  Length: {len(code)} characters")
    
    # Try to compile
    compiler = RecursiaCompiler()
    result = compiler.compile(code)
    
    if result.success:
        print("✓ Compilation SUCCESSFUL!")
        print(f"  Program has {len(result.ast.statements)} statements")
    else:
        print("✗ Compilation FAILED!")
        print("Errors:")
        for err in result.errors:
            if hasattr(err, 'line'):
                print(f"  - Line {err['line']}: {err['message']}")
            else:
                print(f"  - {err}")
else:
    print("Could not find the analyzer code!")