"""
Security Tests

Comprehensive security testing for the IPAI system including
vulnerability testing, authentication, and defensive measures.
"""

import pytest
import asyncio
import time
from typing import Dict, List
from unittest.mock import Mock, patch
import json

from src.core.security import SecurityManager
from src.api.middleware.security import SecurityMiddleware
from src.api.middleware.rate_limiter import RateLimitMiddleware, RateLimitConfig
from tests.conftest import assert_security_response


class TestSecurityManager:
    """Test security manager functionality"""
    
    def test_password_hashing(self, security_manager):
        """Test password hashing and verification"""
        password = "TestPassword123!"
        
        # Hash password
        hashed = security_manager.hash_password(password)
        assert hashed != password
        assert len(hashed) > 50  # bcrypt hashes are long
        
        # Verify correct password
        assert security_manager.verify_password(password, hashed, "test-user")
        
        # Verify incorrect password
        assert not security_manager.verify_password("wrong-password", hashed, "test-user")
    
    def test_password_strength_validation(self, security_manager):
        """Test password strength validation"""
        # Valid password
        strong_password = "StrongPassword123!"
        assert security_manager._validate_password_strength(strong_password)
        
        # Test weak passwords
        weak_passwords = [
            "short",                    # Too short
            "nouppercase123!",         # No uppercase
            "NOLOWERCASE123!",         # No lowercase
            "NoDigitsHere!",           # No digits
            "NoSpecialChars123",       # No special chars
            "password123!"             # Common weak pattern
        ]
        
        for weak_password in weak_passwords:
            with pytest.raises(ValueError):
                security_manager._validate_password_strength(weak_password)
    
    def test_secure_password_generation(self, security_manager):
        """Test secure password generation"""
        password = security_manager.generate_secure_password(16)
        
        assert len(password) == 16
        assert any(c.isupper() for c in password)
        assert any(c.islower() for c in password)
        assert any(c.isdigit() for c in password)
        assert any(c in "!@#$%^&*()_+-=" for c in password)
    
    def test_jwt_token_creation_and_verification(self, security_manager):
        """Test JWT token creation and verification"""
        user_id = "test-user-123"
        
        # Create access token
        access_token = security_manager.create_access_token(user_id)
        assert access_token is not None
        assert len(access_token) > 50
        
        # Verify token
        payload = security_manager.verify_token(access_token, 'access')
        assert payload is not None
        assert payload['sub'] == user_id
        assert payload['type'] == 'access'
        
        # Create refresh token
        refresh_token = security_manager.create_refresh_token(user_id)
        assert refresh_token is not None
        
        # Verify refresh token
        refresh_payload = security_manager.verify_token(refresh_token, 'refresh')
        assert refresh_payload is not None
        assert refresh_payload['sub'] == user_id
        assert refresh_payload['type'] == 'refresh'
    
    def test_token_revocation(self, security_manager):
        """Test token revocation"""
        user_id = "test-user-123"
        token = security_manager.create_access_token(user_id)
        
        # Token should be valid initially
        payload = security_manager.verify_token(token, 'access')
        assert payload is not None
        
        # Revoke token
        revoked = security_manager.revoke_token(token)
        assert revoked
    
    def test_data_encryption_and_decryption(self, security_manager):
        """Test sensitive data encryption"""
        sensitive_data = "This is sensitive information"
        
        # Encrypt data
        encrypted = security_manager.encrypt_sensitive_data(sensitive_data)
        assert encrypted != sensitive_data
        assert len(encrypted) > len(sensitive_data)
        
        # Decrypt data
        decrypted = security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == sensitive_data
    
    def test_encryption_integrity_check(self, security_manager):
        """Test encryption integrity verification"""
        sensitive_data = "Test data for integrity check"
        encrypted = security_manager.encrypt_sensitive_data(sensitive_data)
        
        # Tampering with encrypted data should fail integrity check
        tampered = encrypted[:-5] + "XXXXX"
        
        with pytest.raises(ValueError, match="Failed to decrypt data"):
            security_manager.decrypt_sensitive_data(tampered)
    
    def test_account_lockout_mechanism(self, security_manager):
        """Test account lockout after failed attempts"""
        user_id = "test-user-lockout"
        correct_password = "CorrectPassword123!"
        wrong_password = "WrongPassword123!"
        
        # Hash correct password
        hashed_password = security_manager.hash_password(correct_password)
        
        # Make failed attempts
        for i in range(5):
            result = security_manager.verify_password(wrong_password, hashed_password, user_id)
            assert not result
        
        # Account should be locked now
        assert security_manager._is_account_locked(user_id)
        
        # Even correct password should fail when locked
        result = security_manager.verify_password(correct_password, hashed_password, user_id)
        assert not result
    
    def test_csrf_token_generation_and_validation(self, security_manager):
        """Test CSRF token functionality"""
        session_id = "test-session-123"
        
        # Generate CSRF token
        csrf_token = security_manager.generate_csrf_token(session_id)
        assert csrf_token is not None
        assert len(csrf_token) > 20
        
        # Validate token
        is_valid = security_manager.validate_csrf_token(csrf_token, session_id)
        assert is_valid
        
        # Invalid session ID should fail
        is_valid = security_manager.validate_csrf_token(csrf_token, "wrong-session")
        assert not is_valid
        
        # Expired token should fail
        is_valid = security_manager.validate_csrf_token(csrf_token, session_id, max_age=0)
        assert not is_valid
    
    def test_api_key_hashing_and_verification(self, security_manager):
        """Test API key hashing and verification"""
        api_key = "test-api-key-12345"
        
        # Hash API key
        hashed_key = security_manager.hash_api_key(api_key)
        assert hashed_key != api_key
        assert len(hashed_key) > 50
        
        # Verify API key
        is_valid = security_manager.verify_api_key(api_key, hashed_key)
        assert is_valid
        
        # Wrong API key should fail
        is_valid = security_manager.verify_api_key("wrong-api-key", hashed_key)
        assert not is_valid
    
    def test_security_event_logging(self, security_manager):
        """Test security event logging"""
        # Clear existing events
        security_manager.security_events.clear()
        
        # Trigger some security events
        security_manager.hash_password("test123!")
        security_manager.create_access_token("test-user")
        
        # Check events were logged
        events = security_manager.get_security_events()
        assert len(events) >= 2
        
        # Check event structure
        for event in events:
            assert 'timestamp' in event
            assert 'event_type' in event
            assert 'details' in event
            assert 'source' in event
    
    def test_security_configuration_summary(self, security_manager):
        """Test security configuration summary"""
        summary = security_manager.get_security_summary()
        
        assert 'password_requirements' in summary
        assert 'authentication' in summary
        assert 'security_features' in summary
        assert 'recent_events_count' in summary
        assert 'locked_accounts_count' in summary
        
        # Check password requirements
        pwd_reqs = summary['password_requirements']
        assert pwd_reqs['min_length'] >= 12
        assert pwd_reqs['require_uppercase']
        assert pwd_reqs['require_lowercase']
        assert pwd_reqs['require_digits']
        assert pwd_reqs['require_special']


class TestSecurityMiddleware:
    """Test security middleware functionality"""
    
    @pytest.fixture
    def security_middleware(self):
        """Create security middleware for testing"""
        from fastapi import FastAPI
        app = FastAPI()
        return SecurityMiddleware(app)
    
    def test_sql_injection_detection(self, security_middleware):
        """Test SQL injection pattern detection"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
            "1; DELETE FROM users WHERE 1=1--"
        ]
        
        for payload in sql_payloads:
            assert security_middleware._contains_attack_patterns(payload)
    
    def test_xss_detection(self, security_middleware):
        """Test XSS pattern detection"""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "<iframe src='javascript:alert(1)'></iframe>"
        ]
        
        for payload in xss_payloads:
            assert security_middleware._contains_attack_patterns(payload)
    
    def test_path_traversal_detection(self, security_middleware):
        """Test path traversal pattern detection"""
        traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd"
        ]
        
        for payload in traversal_payloads:
            assert security_middleware._contains_attack_patterns(payload)
    
    def test_command_injection_detection(self, security_middleware):
        """Test command injection pattern detection"""
        command_payloads = [
            "; ls -la",
            "| whoami",
            "`id`",
            "$(whoami)",
            "&& cat /etc/passwd"
        ]
        
        for payload in command_payloads:
            assert security_middleware._contains_attack_patterns(payload)
    
    def test_request_size_validation(self, security_middleware):
        """Test request size validation"""
        from fastapi import Request
        
        # Mock request with large content
        mock_request = Mock(spec=Request)
        mock_request.headers = {"content-length": "50000000"}  # 50MB
        
        result = asyncio.run(security_middleware._validate_request_size(mock_request))
        assert not result  # Should reject large requests
        
        # Normal sized request
        mock_request.headers = {"content-length": "1024"}  # 1KB
        result = asyncio.run(security_middleware._validate_request_size(mock_request))
        assert result
    
    def test_header_validation(self, security_middleware):
        """Test request header validation"""
        from fastapi import Request
        
        # Normal headers
        mock_request = Mock(spec=Request)
        mock_request.headers = {
            "content-type": "application/json",
            "user-agent": "Mozilla/5.0"
        }
        mock_request.headers.items = lambda: [("content-type", "application/json"), ("user-agent", "Mozilla/5.0")]
        
        result = security_middleware._validate_headers(mock_request)
        assert result
        
        # Malicious headers
        mock_request.headers = {
            "x-forwarded-host": "evil.com'; DROP TABLE users; --"
        }
        mock_request.headers.items = lambda: [("x-forwarded-host", "evil.com'; DROP TABLE users; --")]
        
        result = security_middleware._validate_headers(mock_request)
        assert not result
    
    def test_user_agent_blocking(self, security_middleware):
        """Test user agent blocking"""
        from fastapi import Request
        
        blocked_agents = [
            "curl/7.68.0",
            "wget/1.20.3",
            "python-requests/2.25.1",
            "Googlebot/2.1",
            "bot-scanner-1.0"
        ]
        
        for agent in blocked_agents:
            mock_request = Mock(spec=Request)
            mock_request.headers = {"user-agent": agent}
            mock_request.headers.get = lambda key, default="": agent if key == "user-agent" else default
            
            result = security_middleware._validate_user_agent(mock_request)
            assert not result
    
    def test_ip_blocking_after_attacks(self, security_middleware):
        """Test IP blocking after malicious activity"""
        test_ip = "192.168.1.100"
        
        # Mark IP as suspicious
        security_middleware._mark_suspicious_ip(test_ip)
        
        # Check if IP is in suspicious list
        assert test_ip in security_middleware._suspicious_ips


class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_config_creation(self):
        """Test rate limit configuration creation"""
        config = RateLimitConfig(
            requests_per_minute=100,
            requests_per_hour=1000,
            enabled=True
        )
        
        assert config.requests_per_minute == 100
        assert config.requests_per_hour == 1000
        assert config.enabled
    
    def test_in_memory_rate_limiter(self):
        """Test in-memory rate limiter"""
        from src.api.middleware.rate_limiter import InMemoryRateLimiter
        
        config = RateLimitConfig(requests_per_minute=5, requests_per_hour=50)
        limiter = InMemoryRateLimiter(config)
        
        identifier = "test-user-123"
        
        # First 5 requests should be allowed
        for i in range(5):
            allowed, info = limiter.check_rate_limit(identifier)
            assert allowed
            assert info['remaining'] >= 0
        
        # 6th request should be blocked
        allowed, info = limiter.check_rate_limit(identifier)
        assert not allowed
        assert info['retry_after'] > 0
    
    def test_sliding_window_counter(self):
        """Test sliding window counter"""
        from src.api.middleware.rate_limiter import SlidingWindowCounter
        
        counter = SlidingWindowCounter(window_size=60, max_requests=5)
        
        # First 5 requests should be allowed
        for i in range(5):
            allowed, remaining = counter.is_allowed()
            assert allowed
            assert remaining == 4 - i
        
        # 6th request should be blocked
        allowed, remaining = counter.is_allowed()
        assert not allowed
        assert remaining == 0
    
    def test_token_bucket(self):
        """Test token bucket algorithm"""
        from src.api.middleware.rate_limiter import TokenBucket
        
        bucket = TokenBucket(capacity=5, refill_rate=1.0)  # 1 token per second
        
        # Should be able to consume initial tokens
        for i in range(5):
            assert bucket.consume(1)
        
        # Should not be able to consume more
        assert not bucket.consume(1)
        
        # Wait for refill and try again
        time.sleep(1.1)
        assert bucket.consume(1)
    
    @pytest.mark.asyncio
    async def test_rate_limit_middleware_integration(self):
        """Test rate limit middleware integration"""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        
        config = RateLimitConfig(requests_per_minute=3, requests_per_hour=10)
        
        app = FastAPI()
        middleware = RateLimitMiddleware(app, config)
        
        @app.get("/test")
        async def test_endpoint():
            return {"message": "success"}
        
        # Mock request
        mock_request = Mock(spec=Request)
        mock_request.url.path = "/test"
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {}
        mock_request.headers.get = lambda key, default=None: None
        
        # Mock call_next
        async def mock_call_next(request):
            return JSONResponse({"message": "success"})
        
        # First few requests should succeed
        for i in range(3):
            response = await middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 200
        
        # Next request should be rate limited
        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 429


class TestVulnerabilityScanning:
    """Test vulnerability scanning and protection"""
    
    @pytest.mark.parametrize("payload", [
        "'; DROP TABLE users; --",
        "<script>alert('XSS')</script>",
        "../../../etc/passwd",
        "; ls -la"
    ])
    def test_malicious_payload_detection(self, security_middleware, payload):
        """Test detection of various malicious payloads"""
        assert security_middleware._contains_attack_patterns(payload)
    
    def test_password_policy_enforcement(self, security_manager):
        """Test password policy enforcement"""
        weak_passwords = [
            "123456",
            "password",
            "qwerty",
            "admin",
            "Password1",  # Common pattern
            "short"       # Too short
        ]
        
        for password in weak_passwords:
            with pytest.raises(ValueError):
                security_manager._validate_password_strength(password)
    
    def test_session_security(self, security_manager):
        """Test session security measures"""
        user_id = "test-user-session"
        
        # Create token
        token = security_manager.create_access_token(user_id)
        
        # Verify token contains security claims
        import jwt
        payload = jwt.decode(token, options={"verify_signature": False})
        
        assert 'jti' in payload  # JWT ID for revocation
        assert 'aud' in payload  # Audience
        assert 'iss' in payload  # Issuer
        assert 'iat' in payload  # Issued at
        assert 'exp' in payload  # Expiration
    
    def test_timing_attack_resistance(self, security_manager):
        """Test resistance to timing attacks"""
        correct_password = "CorrectPassword123!"
        wrong_password = "WrongPassword123!"
        
        hashed = security_manager.hash_password(correct_password)
        
        # Measure verification times
        import time
        
        times_correct = []
        times_wrong = []
        
        for _ in range(10):
            start = time.time()
            security_manager.verify_password(correct_password, hashed)
            times_correct.append(time.time() - start)
            
            start = time.time()
            security_manager.verify_password(wrong_password, hashed)
            times_wrong.append(time.time() - start)
        
        # Times should be similar (within reasonable variance)
        avg_correct = sum(times_correct) / len(times_correct)
        avg_wrong = sum(times_wrong) / len(times_wrong)
        
        # Allow for some variance, but times shouldn't differ significantly
        assert abs(avg_correct - avg_wrong) < 0.01  # Less than 10ms difference


class TestSecurityCompliance:
    """Test security compliance and best practices"""
    
    def test_encryption_standards(self, security_manager):
        """Test encryption meets security standards"""
        # Test that encryption uses strong algorithms
        test_data = "sensitive information"
        encrypted = security_manager.encrypt_sensitive_data(test_data)
        
        # Encrypted data should be significantly different
        assert len(encrypted) > len(test_data)
        assert encrypted != test_data
        
        # Should be deterministically decryptable
        decrypted = security_manager.decrypt_sensitive_data(encrypted)
        assert decrypted == test_data
    
    def test_jwt_security_claims(self, security_manager):
        """Test JWT tokens contain required security claims"""
        user_id = "test-user-jwt"
        token = security_manager.create_access_token(user_id)
        
        # Decode without verification to check claims
        import jwt
        payload = jwt.decode(token, options={"verify_signature": False})
        
        required_claims = ['sub', 'iat', 'exp', 'jti', 'aud', 'iss', 'type']
        for claim in required_claims:
            assert claim in payload, f"Missing required claim: {claim}"
    
    def test_security_headers_presence(self, security_middleware):
        """Test that security headers are properly set"""
        from fastapi import Response
        
        response = Response()
        security_middleware._add_security_headers(response)
        
        expected_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Content-Security-Policy',
            'Strict-Transport-Security',
            'Referrer-Policy',
            'Permissions-Policy'
        ]
        
        for header in expected_headers:
            assert header in response.headers
    
    def test_input_sanitization(self, security_middleware):
        """Test input sanitization capabilities"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "$(rm -rf /)"
        ]
        
        for malicious_input in malicious_inputs:
            # Should detect as malicious
            assert security_middleware._contains_attack_patterns(malicious_input)
    
    def test_error_message_security(self, security_manager):
        """Test that error messages don't leak sensitive information"""
        # Test with invalid token
        payload = security_manager.verify_token("invalid.jwt.token")
        assert payload is None
        
        # Check that security events are logged but don't contain sensitive data
        events = security_manager.get_security_events(event_type='token_invalid')
        for event in events:
            # Should not contain the actual token or sensitive details
            assert 'token' not in str(event.get('details', {})).lower()
            assert 'password' not in str(event.get('details', {})).lower()