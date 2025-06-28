#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursia CLI Wrapper

This module provides a simplified command-line interface for Recursia,
wrapping the main entry point functionality with more ergonomic commands
and simplified arguments. It serves as a user-friendly alternative to
directly invoking recursia.py.

Usage:
    rcli run <file> [options]
    rcli compile <file> [--target=<target>] [options]
    rcli repl [options]
    rcli viz <file> [options]
    rcli check <file> [options]
    rcli analyze <file> [options]
    rcli init [<directory>] [options]
    rcli hw [status|connect|disconnect|list] [options]
    rcli -h | --help
    rcli --version

Options:
    -h --help               Show this help message
    --version               Show version
    -v --verbose            Verbose output
    -d --debug              Debug mode
    -t --target=<target>    Compilation target [default: quantum_simulator]
    --hw=<provider>         Hardware provider (ibm, rigetti, google, ionq)
    --device=<device>       Specific quantum device
    --shots=<num>           Number of shots for quantum execution [default: 1024]
    --no-viz                Disable visualization
    --opt=<level>           Optimization level (0-3) [default: 1]
    --config=<file>         Configuration file
"""

import os
import sys
import subprocess
from docopt import docopt
from pathlib import Path

# Import the main module from this package 
from . import __version__
from .recursia import main as recursia_main

def translate_args(args):
    """
    Translate simplified CLI arguments to full recursia.py arguments
    
    Args:
        args: Dictionary of CLI arguments from docopt
        
    Returns:
        List of arguments for recursia.py
    """
    # Start with basic arguments
    recursia_args = []
    
    # Add verbose and debug flags
    if args.get('--verbose'):
        recursia_args.append('--verbose')
    if args.get('--debug'):
        recursia_args.append('--debug')
    
    # Add configuration if specified
    if args.get('--config'):
        recursia_args.extend(['--config', args['--config']])
    
    # Map commands from simplified to full
    cmd_map = {
        'run': 'run',
        'compile': 'compile',
        'repl': 'repl',
        'viz': 'visualize',
        'check': 'check',
        'analyze': 'analyze',
        'init': 'initialize',
        'hw': 'hardware'
    }
    
    # Find the active command
    for cmd in cmd_map:
        if args.get(cmd):
            recursia_args.append(cmd_map[cmd])
            break
    
    # Add file argument if present
    if args.get('<file>'):
        recursia_args.append(args['<file>'])
    
    # Handle initialization directory
    if args.get('init') and args.get('<directory>'):
        recursia_args.append(args['<directory>'])
    
    # Handle hardware commands
    if args.get('hw'):
        # Check for specific hardware commands
        if args.get('status'):
            recursia_args.append('status')
        elif args.get('connect'):
            recursia_args.append('connect')
        elif args.get('disconnect'):
            recursia_args.append('disconnect')
        elif args.get('list'):
            recursia_args.append('list')
        else:
            # Default to status
            recursia_args.append('status')
    
    # Add target if specified
    if args.get('--target'):
        recursia_args.extend(['--target', args['--target']])
    
    # Add hardware provider if specified
    if args.get('--hw'):
        recursia_args.extend(['--hardware', args['--hw']])
    
    # Add device if specified
    if args.get('--device'):
        recursia_args.extend(['--device', args['--device']])
    
    # Add shots if specified
    if args.get('--shots'):
        recursia_args.extend(['--shots', args['--shots']])
    
    # Add visualization flag if disabled
    if args.get('--no-viz'):
        recursia_args.append('--no-visualization')
    
    # Add optimization level if specified
    if args.get('--opt'):
        recursia_args.extend(['--optimization', args['--opt']])
    
    return recursia_args

def run_direct():
    """
    Run recursia.py directly using the Python interpreter
    """
    # Parse arguments using docopt
    args = docopt(__doc__, version=f"Recursia CLI v{__version__}")
    
    # Translate to recursia.py arguments
    recursia_args = translate_args(args)
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:
        # Set up sys.argv for the main function
        sys.argv = ['recursia'] + recursia_args
        
        # Call the main function directly
        result = recursia_main()
        
        # Ensure we have a valid result
        if result is None:
            result = 0  # Success by default
            
        return result
    
    except Exception as e:
        print(f"Error executing command: {e}")
        return 1
    
    finally:
        # Restore original sys.argv
        sys.argv = original_argv

def run_subprocess():
    """
    Run recursia.py as a subprocess
    """
    # Parse arguments using docopt
    args = docopt(__doc__, version=f"Recursia CLI v{__version__}")
    
    # Translate to recursia.py arguments
    recursia_args = translate_args(args)
    
    # Get the path to recursia.py
    # Find the path to recursia.py within the package
    try:
        from importlib.resources import files
        recursia_path = files('src') / 'recursia.py'
    except ImportError:
        print(f"Error finding path: {e}")
    # Build the command
    cmd = [sys.executable, str(recursia_path)] + recursia_args
    
    # Run the subprocess
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        print(f"Error running command: {e}")
        return 1

def display_examples():
    """
    Display example usage
    """
    print("""
Example Usage:
--------------
  rcli run simulation.recursia          # Run a Recursia program
  rcli viz experiment.recursia          # Visualize a program
  rcli compile quantum.recursia -t ibm  # Compile for IBM hardware
  rcli repl                             # Start interactive REPL
  rcli check code.recursia              # Validate syntax and semantics
  rcli analyze model.recursia           # Analyze OSH metrics
  rcli init my_project                  # Create new project
  rcli hw list                          # List quantum hardware
  rcli hw connect --hw=ibm              # Connect to IBM quantum
    """)

def main():
    """
    Main entry point for the CLI wrapper
    """
    # Check if "--examples" is passed
    if len(sys.argv) == 2 and sys.argv[1] == "--examples":
        display_examples()
        return 0
    
    # Use subprocess approach for better isolation and error handling
    try:
        return run_subprocess()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"Critical error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())