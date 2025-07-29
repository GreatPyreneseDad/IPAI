"""
IPAI Test Suite

Comprehensive testing framework for the IPAI system with
defensive security practices and thorough coverage.
"""

import pytest
import asyncio
import logging
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
from pathlib import Path

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test configuration
TEST_CONFIG = {
    'database_url': 'postgresql://localhost/ipai_test',
    'redis_url': 'redis://localhost:6379/1',
    'test_data_dir': Path(__file__).parent / 'data',
    'mock_llm': True,
    'mock_blockchain': True,
    'enable_security_tests': True,
    'test_timeout': 30
}

# Test fixtures and utilities will be imported from conftest.py