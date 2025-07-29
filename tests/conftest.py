"""
Test Configuration and Fixtures

This module provides pytest fixtures and configuration for testing
the IPAI system with security and performance testing capabilities.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, AsyncGenerator
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

# Test imports
from src.models.coherence_profile import CoherenceProfile, GCTComponents, IndividualParameters
from src.models.user import User, UserPreferences
from src.models.assessment import Assessment, AssessmentResult
from src.coherence.gct_calculator import GCTCalculator
from src.coherence.triadic_processor import TriadicProcessor
from src.core.security import SecurityManager
from src.core.config import Settings
from src.core.database import Database
from src.core.performance import PerformanceOptimizer, CacheManager


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        'database_url': 'postgresql://localhost/ipai_test',
        'redis_url': 'redis://localhost:6379/1',
        'secret_key': 'test-secret-key-for-testing-only',
        'jwt_secret': 'test-jwt-secret',
        'encryption_key': 'test-encryption-key-32-bytes-long',
        'enable_security_tests': True,
        'mock_external_services': True,
        'test_timeout': 30
    }


@pytest.fixture
def security_manager(test_config):
    """Security manager for testing"""
    return SecurityManager(test_config)


@pytest.fixture
def performance_optimizer():
    """Performance optimizer for testing"""
    config = {
        'slow_query_threshold': 0.1,  # Lower threshold for tests
        'max_threads': 2,
        'enable_profiling': True
    }
    return PerformanceOptimizer(config)


@pytest.fixture
def cache_manager():
    """Cache manager for testing"""
    config = {
        'max_size': 100,
        'default_ttl': 60,
        'cleanup_interval': 10
    }
    return CacheManager(config)


@pytest.fixture
async def database(test_config):
    """Database instance for testing"""
    db = Database(test_config)
    await db.connect()
    yield db
    await db.disconnect()


@pytest.fixture
def gct_calculator():
    """GCT calculator for testing"""
    return GCTCalculator()


@pytest.fixture
def triadic_processor():
    """Triadic processor for testing"""
    return TriadicProcessor()


@pytest.fixture
def sample_user():
    """Sample user for testing"""
    return User(
        id="test-user-123",
        email="test@example.com",
        password_hash="hashed_password",
        preferences=UserPreferences(
            language="en",
            timezone="UTC",
            notifications_enabled=True
        )
    )


@pytest.fixture
def sample_gct_components():
    """Sample GCT components for testing"""
    return GCTComponents(
        psi=0.7,    # Internal consistency
        rho=0.8,    # Accumulated wisdom
        q=0.6,      # Moral activation
        f=0.9       # Social belonging
    )


@pytest.fixture
def sample_individual_parameters():
    """Sample individual parameters for testing"""
    return IndividualParameters(
        k_m=0.5,    # Moral sensitivity
        k_i=2.0     # Inhibition strength
    )


@pytest.fixture
def sample_coherence_profile(sample_gct_components, sample_individual_parameters):
    """Sample coherence profile for testing"""
    return CoherenceProfile(
        user_id="test-user-123",
        components=sample_gct_components,
        parameters=sample_individual_parameters,
        timestamp=datetime.utcnow()
    )


@pytest.fixture
def sample_assessment():
    """Sample assessment for testing"""
    return Assessment(
        id="test-assessment-123",
        user_id="test-user-123",
        assessment_type="moral_reasoning",
        questions=[
            {
                "id": "q1",
                "text": "What would you do in this situation?",
                "type": "multiple_choice",
                "options": ["A", "B", "C", "D"]
            }
        ],
        created_at=datetime.utcnow()
    )


@pytest.fixture
def sample_assessment_result():
    """Sample assessment result for testing"""
    return AssessmentResult(
        assessment_id="test-assessment-123",
        user_id="test-user-123",
        answers={"q1": "A"},
        scores={
            "moral_reasoning": 0.8,
            "consistency": 0.7,
            "wisdom": 0.9
        },
        parameter_adjustments={
            "k_m": 0.1,
            "k_i": -0.05
        },
        completed_at=datetime.utcnow()
    )


@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "text": "This is a test response from the LLM",
        "coherence_analysis": {
            "consistency_score": 0.8,
            "wisdom_indicators": ["thoughtful", "balanced"],
            "moral_markers": ["ethical", "principled"],
            "social_context": "collaborative"
        },
        "triadic_processing": {
            "generate_phase": "Initial response generated",
            "analyze_phase": "Response analyzed for coherence",
            "ground_phase": "Response grounded in user context"
        }
    }


@pytest.fixture
def mock_blockchain_response():
    """Mock blockchain response for testing"""
    return {
        "transaction_hash": "0x123456789abcdef",
        "block_number": 123456,
        "gas_used": 50000,
        "status": "success",
        "coherence_stored": True
    }


@pytest.fixture
def temp_directory():
    """Temporary directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_data_directory():
    """Test data directory"""
    return Path(__file__).parent / "data"


@pytest.fixture(autouse=True)
def mock_external_services():
    """Mock external services for testing"""
    with patch('src.llm.interface.LLMInterface') as mock_llm, \
         patch('src.blockchain.interface.BlockchainInterface') as mock_blockchain:
        
        # Configure LLM mock
        mock_llm_instance = AsyncMock()
        mock_llm.return_value = mock_llm_instance
        mock_llm_instance.generate.return_value = {
            "text": "Test LLM response",
            "coherence_score": 0.8
        }
        
        # Configure blockchain mock
        mock_blockchain_instance = AsyncMock()
        mock_blockchain.return_value = mock_blockchain_instance
        mock_blockchain_instance.store_coherence.return_value = {
            "transaction_hash": "0xtest123",
            "success": True
        }
        
        yield {
            'llm': mock_llm_instance,
            'blockchain': mock_blockchain_instance
        }


@pytest.fixture
def security_test_data():
    """Security test data for vulnerability testing"""
    return {
        'sql_injection_payloads': [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--"
        ],
        'xss_payloads': [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>"
        ],
        'path_traversal_payloads': [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
        ],
        'command_injection_payloads': [
            "; ls -la",
            "| whoami",
            "`id`",
            "$(whoami)"
        ],
        'weak_passwords': [
            "123456",
            "password",
            "qwerty",
            "admin",
            "user"
        ],
        'malformed_tokens': [
            "invalid.jwt.token",
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.invalid",
            "",
            "bearer token"
        ]
    }


@pytest.fixture
def performance_test_data():
    """Performance test data"""
    return {
        'large_coherence_data': {
            'user_id': 'perf-test-user',
            'components': {
                'psi': 0.8,
                'rho': 0.7,
                'q': 0.9,
                'f': 0.6
            },
            'parameters': {
                'k_m': 0.4,
                'k_i': 1.8
            },
            'history': [
                {'timestamp': datetime.utcnow() - timedelta(days=i), 'score': 0.7 + (i * 0.01)}
                for i in range(100)
            ]
        },
        'stress_test_requests': 1000,
        'concurrent_users': 50,
        'expected_response_time': 0.5  # seconds
    }


@pytest.fixture
def api_test_client():
    """Test client for API testing"""
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def authenticated_user(api_test_client, sample_user):
    """Authenticated user for API testing"""
    # Register user
    register_response = api_test_client.post(
        "/api/v1/auth/register",
        json={
            "email": sample_user.email,
            "password": "TestPassword123!",
            "confirm_password": "TestPassword123!"
        }
    )
    
    # Login user
    login_response = api_test_client.post(
        "/api/v1/auth/login",
        json={
            "email": sample_user.email,
            "password": "TestPassword123!"
        }
    )
    
    token = login_response.json()["access_token"]
    
    return {
        "user": sample_user,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"}
    }


# Utility functions for testing

def assert_valid_coherence_profile(profile: CoherenceProfile):
    """Assert that a coherence profile is valid"""
    assert profile.user_id is not None
    assert 0 <= profile.components.psi <= 1
    assert 0 <= profile.components.rho <= 1
    assert 0 <= profile.components.q <= 1
    assert 0 <= profile.components.f <= 1
    assert profile.parameters.k_m > 0
    assert profile.parameters.k_i > 0
    assert profile.timestamp is not None


def assert_valid_assessment(assessment: Assessment):
    """Assert that an assessment is valid"""
    assert assessment.id is not None
    assert assessment.user_id is not None
    assert assessment.assessment_type is not None
    assert len(assessment.questions) > 0
    assert assessment.created_at is not None


def assert_security_response(response, expected_status=403):
    """Assert that a security response is correct"""
    assert response.status_code == expected_status
    response_data = response.json()
    assert "error" in response_data
    assert "security" in response_data["error"].lower() or "blocked" in response_data["error"].lower()


def assert_performance_metrics(metrics: Dict[str, Any], max_response_time: float = 1.0):
    """Assert that performance metrics are within acceptable ranges"""
    assert "avg_response_time" in metrics
    assert metrics["avg_response_time"] <= max_response_time
    assert "error_rate" in metrics
    assert metrics["error_rate"] <= 0.05  # 5% error rate maximum


def create_test_coherence_data(user_id: str, num_profiles: int = 10) -> List[Dict]:
    """Create test coherence data"""
    profiles = []
    base_time = datetime.utcnow()
    
    for i in range(num_profiles):
        profile = {
            "user_id": user_id,
            "components": {
                "psi": 0.5 + (i * 0.05),
                "rho": 0.6 + (i * 0.04),
                "q": 0.7 + (i * 0.03),
                "f": 0.8 + (i * 0.02)
            },
            "parameters": {
                "k_m": 0.3 + (i * 0.02),
                "k_i": 1.5 + (i * 0.1)
            },
            "timestamp": base_time - timedelta(days=i)
        }
        profiles.append(profile)
    
    return profiles


# Performance testing utilities

async def measure_async_performance(func, *args, **kwargs):
    """Measure async function performance"""
    import time
    start_time = time.time()
    result = await func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


def measure_sync_performance(func, *args, **kwargs):
    """Measure sync function performance"""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


# Test data generators

def generate_test_user_data(count: int = 10) -> List[Dict]:
    """Generate test user data"""
    users = []
    for i in range(count):
        user = {
            "id": f"test-user-{i}",
            "email": f"test{i}@example.com",
            "password_hash": f"hashed_password_{i}",
            "created_at": datetime.utcnow() - timedelta(days=i)
        }
        users.append(user)
    return users


def generate_test_assessment_data(user_id: str, count: int = 5) -> List[Dict]:
    """Generate test assessment data"""
    assessments = []
    assessment_types = ["moral_reasoning", "consistency_check", "wisdom_assessment", "social_context"]
    
    for i in range(count):
        assessment = {
            "id": f"test-assessment-{user_id}-{i}",
            "user_id": user_id,
            "assessment_type": assessment_types[i % len(assessment_types)],
            "questions": [
                {
                    "id": f"q{j}",
                    "text": f"Test question {j}",
                    "type": "multiple_choice",
                    "options": ["A", "B", "C", "D"]
                }
                for j in range(5)
            ],
            "created_at": datetime.utcnow() - timedelta(hours=i)
        }
        assessments.append(assessment)
    
    return assessments


# Cleanup utilities

@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Cleanup test data after each test"""
    yield
    # Cleanup would be implemented here
    pass