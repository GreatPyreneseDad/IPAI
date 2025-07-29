"""
IPAI API v1 Routes

This module contains all v1 API routes for the IPAI system.
"""

from . import coherence, llm, identity, assessment, analytics

__all__ = [
    'coherence',
    'llm', 
    'identity',
    'assessment',
    'analytics'
]