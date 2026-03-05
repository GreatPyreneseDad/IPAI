"""
Phase 1 Test Configuration and Fixtures
"""

import pytest
from src.core.rose_glass_v2 import RoseGlassEngine


@pytest.fixture
def engine():
    """Rose Glass v2 engine instance."""
    return RoseGlassEngine()


@pytest.fixture
def api_client():
    """FastAPI test client."""
    from fastapi.testclient import TestClient
    from main import app
    with TestClient(app) as client:
        yield client
