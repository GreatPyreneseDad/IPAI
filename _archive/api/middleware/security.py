"""
Security Middleware

This module provides security middleware for the IPAI API,
including security headers, request validation, and threat detection.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import re
import logging
from typing import Dict, List, Set
from urllib.parse import unquote

logger = logging.getLogger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for API protection"""
    
    def __init__(self, app, config: Dict = None):
        super().__init__(app)
        self.config = config or self._default_config()
        self._suspicious_ips: Set[str] = set()
        self._attack_patterns = self._compile_attack_patterns()
        
    def _default_config(self) -> Dict:
        """Default security configuration"""
        return {
            'max_request_size': 10 * 1024 * 1024,  # 10MB
            'max_header_size': 8192,  # 8KB
            'max_query_params': 50,
            'allowed_methods': ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
            'blocked_user_agents': [
                'curl', 'wget', 'python-requests', 'bot', 'crawler', 'scanner'
            ],
            'enable_csrf_protection': True,
            'enable_xss_protection': True,
            'enable_sql_injection_protection': True,
            'enable_path_traversal_protection': True,
            'rate_limit_suspicious_requests': True
        }
    
    def _compile_attack_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for attack detection"""
        return {
            'sql_injection': [
                re.compile(r'(\bUNION\b.*\bSELECT\b)', re.IGNORECASE),
                re.compile(r'(\bSELECT\b.*\bFROM\b.*\bWHERE\b)', re.IGNORECASE),
                re.compile(r'(\bINSERT\b.*\bINTO\b)', re.IGNORECASE),
                re.compile(r'(\bDELETE\b.*\bFROM\b)', re.IGNORECASE),
                re.compile(r'(\bDROP\b.*\bTABLE\b)', re.IGNORECASE),
                re.compile(r'(\bUPDATE\b.*\bSET\b)', re.IGNORECASE),
                re.compile(r"('|\"|;|--|\b(OR|AND)\b.*=)", re.IGNORECASE),
            ],
            'xss': [
                re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
                re.compile(r'javascript:', re.IGNORECASE),
                re.compile(r'on\w+\s*=', re.IGNORECASE),
                re.compile(r'<iframe[^>]*>', re.IGNORECASE),
                re.compile(r'<object[^>]*>', re.IGNORECASE),
                re.compile(r'<embed[^>]*>', re.IGNORECASE),
            ],
            'path_traversal': [
                re.compile(r'\.\./', re.IGNORECASE),
                re.compile(r'\.\.\\', re.IGNORECASE),
                re.compile(r'%2e%2e%2f', re.IGNORECASE),
                re.compile(r'%2e%2e%5c', re.IGNORECASE),
                re.compile(r'/etc/passwd', re.IGNORECASE),
                re.compile(r'/proc/version', re.IGNORECASE),
            ],
            'command_injection': [
                re.compile(r';\s*(cat|ls|pwd|whoami|id|uname)', re.IGNORECASE),
                re.compile(r'\|\s*(cat|ls|pwd|whoami|id|uname)', re.IGNORECASE),
                re.compile(r'`.*`', re.IGNORECASE),
                re.compile(r'\$\(.*\)', re.IGNORECASE),
            ]
        }
    
    async def dispatch(self, request: Request, call_next):
        """Process security checks for incoming requests"""
        
        start_time = time.time()
        
        try:
            # Get client IP
            client_ip = self._get_client_ip(request)
            
            # Check if IP is suspicious
            if client_ip in self._suspicious_ips:
                logger.warning(f"Blocked request from suspicious IP: {client_ip}")
                return self._create_security_response("Blocked: Suspicious activity detected")
            
            # Validate request method
            if not self._validate_method(request):
                logger.warning(f"Invalid method {request.method} from {client_ip}")
                return self._create_security_response("Method not allowed")
            
            # Validate request size
            if not await self._validate_request_size(request):
                logger.warning(f"Request too large from {client_ip}")
                return self._create_security_response("Request too large")
            
            # Validate headers
            if not self._validate_headers(request):
                logger.warning(f"Invalid headers from {client_ip}")
                return self._create_security_response("Invalid headers")
            
            # Check user agent
            if not self._validate_user_agent(request):
                logger.warning(f"Blocked user agent from {client_ip}")
                return self._create_security_response("Blocked: Automated request detected")
            
            # Validate URL and parameters
            if not self._validate_url(request):
                logger.warning(f"Malicious URL pattern from {client_ip}")
                self._mark_suspicious_ip(client_ip)
                return self._create_security_response("Blocked: Malicious request detected")
            
            # Validate query parameters
            if not self._validate_query_params(request):
                logger.warning(f"Malicious query parameters from {client_ip}")
                self._mark_suspicious_ip(client_ip)
                return self._create_security_response("Blocked: Malicious request detected")
            
            # Check for attack patterns in body (if applicable)
            body_valid = await self._validate_request_body(request)
            if not body_valid:
                logger.warning(f"Malicious request body from {client_ip}")
                self._mark_suspicious_ip(client_ip)
                return self._create_security_response("Blocked: Malicious content detected")
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Add timing information
            security_time = time.time() - start_time
            response.headers["X-Security-Time"] = str(security_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            return self._create_security_response("Security processing failed", status_code=500)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check forwarded headers first
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        # Fall back to direct connection
        return request.client.host if request.client else "unknown"
    
    def _validate_method(self, request: Request) -> bool:
        """Validate HTTP method"""
        return request.method in self.config['allowed_methods']
    
    async def _validate_request_size(self, request: Request) -> bool:
        """Validate request size"""
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                return size <= self.config['max_request_size']
            except ValueError:
                return False
        return True
    
    def _validate_headers(self, request: Request) -> bool:
        """Validate request headers"""
        
        # Check header size
        total_header_size = sum(len(name) + len(value) for name, value in request.headers.items())
        if total_header_size > self.config['max_header_size']:
            return False
        
        # Check for suspicious headers
        suspicious_headers = [
            'x-forwarded-host', 'x-original-url', 'x-rewrite-url'
        ]
        
        for header in suspicious_headers:
            if header in request.headers:
                value = request.headers[header]
                if self._contains_attack_patterns(value):
                    return False
        
        return True
    
    def _validate_user_agent(self, request: Request) -> bool:
        """Validate user agent"""
        user_agent = request.headers.get("user-agent", "").lower()
        
        # Allow empty user agent for now (mobile apps might not set it)
        if not user_agent:
            return True
        
        # Check against blocked user agents
        for blocked in self.config['blocked_user_agents']:
            if blocked.lower() in user_agent:
                return False
        
        return True
    
    def _validate_url(self, request: Request) -> bool:
        """Validate URL for malicious patterns"""
        
        url_path = unquote(str(request.url.path))
        
        # Check for path traversal
        if self.config['enable_path_traversal_protection']:
            for pattern in self._attack_patterns['path_traversal']:
                if pattern.search(url_path):
                    return False
        
        # Check for other malicious patterns
        if self._contains_attack_patterns(url_path):
            return False
        
        return True
    
    def _validate_query_params(self, request: Request) -> bool:
        """Validate query parameters"""
        
        query_params = dict(request.query_params)
        
        # Check number of parameters
        if len(query_params) > self.config['max_query_params']:
            return False
        
        # Check each parameter
        for key, value in query_params.items():
            # Check parameter name and value for attacks
            if self._contains_attack_patterns(key) or self._contains_attack_patterns(value):
                return False
        
        return True
    
    async def _validate_request_body(self, request: Request) -> bool:
        """Validate request body for malicious content"""
        
        # Only check certain content types
        content_type = request.headers.get("content-type", "")
        
        if "application/json" in content_type or "text/" in content_type:
            try:
                # Get body without consuming it (this is tricky with FastAPI)
                # For now, we'll skip body validation to avoid issues
                # In production, this would need careful implementation
                return True
                
            except Exception as e:
                logger.error(f"Error validating request body: {e}")
                return True  # Allow request if we can't validate
        
        return True
    
    def _contains_attack_patterns(self, text: str) -> bool:
        """Check if text contains attack patterns"""
        
        if not text:
            return False
        
        text = unquote(text)  # Decode URL encoding
        
        # Check SQL injection patterns
        if self.config['enable_sql_injection_protection']:
            for pattern in self._attack_patterns['sql_injection']:
                if pattern.search(text):
                    return True
        
        # Check XSS patterns
        if self.config['enable_xss_protection']:
            for pattern in self._attack_patterns['xss']:
                if pattern.search(text):
                    return True
        
        # Check command injection patterns
        for pattern in self._attack_patterns['command_injection']:
            if pattern.search(text):
                return True
        
        return False
    
    def _mark_suspicious_ip(self, ip: str):
        """Mark IP as suspicious"""
        self._suspicious_ips.add(ip)
        logger.warning(f"Marked IP as suspicious: {ip}")
        
        # Clean up old suspicious IPs periodically
        if len(self._suspicious_ips) > 1000:
            # Remove oldest half (simple cleanup)
            ips_list = list(self._suspicious_ips)
            self._suspicious_ips = set(ips_list[500:])
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        
        # Prevent XSS
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Content Security Policy
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https:; "
            "font-src 'self'; "
            "object-src 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Content-Security-Policy"] = csp
        
        # HSTS (if HTTPS)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        # Remove server info
        response.headers["Server"] = "IPAI"
    
    def _create_security_response(self, message: str, status_code: int = 403) -> JSONResponse:
        """Create security response"""
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "Security violation",
                "message": message,
                "timestamp": time.time()
            },
            headers={"X-Security-Block": "true"}
        )


class CSRFProtection:
    """CSRF protection utility"""
    
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token"""
        import hmac
        import hashlib
        import base64
        
        timestamp = str(int(time.time()))
        data = f"{session_id}:{timestamp}"
        signature = hmac.new(
            self.secret_key.encode(),
            data.encode(),
            hashlib.sha256
        ).digest()
        
        token = base64.b64encode(f"{data}:{signature.hex()}".encode()).decode()
        return token
    
    def validate_token(self, token: str, session_id: str, max_age: int = 3600) -> bool:
        """Validate CSRF token"""
        try:
            import hmac
            import hashlib
            import base64
            
            decoded = base64.b64decode(token.encode()).decode()
            parts = decoded.split(':')
            
            if len(parts) != 3:
                return False
            
            token_session_id, timestamp, signature = parts
            
            # Check session ID
            if token_session_id != session_id:
                return False
            
            # Check timestamp
            if int(time.time()) - int(timestamp) > max_age:
                return False
            
            # Verify signature
            data = f"{token_session_id}:{timestamp}"
            expected_signature = hmac.new(
                self.secret_key.encode(),
                data.encode(),
                hashlib.sha256
            ).digest().hex()
            
            return hmac.compare_digest(signature, expected_signature)
            
        except Exception:
            return False