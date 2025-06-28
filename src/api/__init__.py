"""
API module for Recursia backend services.

This module provides unified API endpoints for:
- Gravitational wave echo simulation and analysis
- OSH (Organic Simulation Hypothesis) calculations
- Unified backend services

Usage:
    from src.api import unified_api_server
    from src.api import gravitational_wave_api
    from src.api import osh_calculations_api
"""

# Import main API components
from .unified_api_server import UnifiedAPIServer
from .gravitational_wave_api import router as gw_router
from .osh_calculations_api import CalculationType

__version__ = "1.0.0"

__all__ = [
    'UnifiedAPIServer',
    'gw_router', 
    'CalculationType'
]