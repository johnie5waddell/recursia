#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main Recursia package initialization

This file initializes the Recursia package and exposes the main components
that users will need to import.
"""

# Version information
__version__ = "0.2.5"
__author__ = "Johnie Waddell"
__license__ = "MIT"
__copyright__ = "Copyright 2025"

# Import core components for direct access
from .core.repl import RecursiaREPL
from .core.compiler import RecursiaCompiler

# Make key modules available at package level
from . import core
from . import physics
from . import quantum
from . import visualization
from . import simulator

# Import main entry points
try:
    from .recursia import main as recursia_main
    from .cli import main as cli_main
except ImportError as e:
    # Allow core modules to be imported without visualization dependencies
    import logging
    logging.getLogger(__name__).warning(f"Visualization components unavailable: {e}")
    recursia_main = None
    cli_main = None