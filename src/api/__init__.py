"""
IPAI API Module

This module provides the FastAPI implementation for the IPAI system,
including RESTful endpoints for coherence calculations, LLM interactions,
and blockchain integration.
"""

from .main import create_app
from .dependencies import get_current_user, get_db, get_gct_calculator
from .middleware.auth import AuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.cors import setup_cors

__all__ = [
    'create_app',
    'get_current_user',
    'get_db', 
    'get_gct_calculator',
    'AuthMiddleware',
    'RateLimitMiddleware',
    'setup_cors'
]