#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursia Entry Point

This module provides the main functionality for the Recursia command-line
interface when invoked through `python -m recursia`.
"""

import sys
from src.recursia import main

if __name__ == "__main__":
    sys.exit(main())