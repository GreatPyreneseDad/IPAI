"""
IPAI Core Module

This module contains core functionality for the IPAI system,
including configuration, security, database, and utilities.
"""

try:
    from .config import Settings
except ImportError:
    Settings = None

try:
    from .security import SecurityManager
except ImportError:
    SecurityManager = None

try:
    from .database import Database
except ImportError:
    Database = None

try:
    from .performance import PerformanceOptimizer, CacheManager
except ImportError:
    PerformanceOptimizer = None
    CacheManager = None

__all__ = [
    'Settings',
    'SecurityManager',
    'Database',
    'PerformanceOptimizer',
    'CacheManager'
]
