"""
CORS Middleware Configuration

This module provides CORS (Cross-Origin Resource Sharing) configuration
for the IPAI API.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from typing import List


def setup_cors(
    app: FastAPI,
    allowed_origins: List[str] = None,
    allowed_methods: List[str] = None,
    allowed_headers: List[str] = None,
    allow_credentials: bool = True,
    expose_headers: List[str] = None
):
    """Setup CORS middleware for the FastAPI application"""
    
    # Default allowed origins for development
    if allowed_origins is None:
        allowed_origins = [
            "http://localhost:3000",  # React development server
            "http://localhost:3001",  # Alternative React port
            "http://localhost:8080",  # Vue.js development server
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8080",
        ]
    
    # Default allowed methods
    if allowed_methods is None:
        allowed_methods = [
            "GET",
            "POST", 
            "PUT",
            "DELETE",
            "PATCH",
            "OPTIONS",
            "HEAD"
        ]
    
    # Default allowed headers
    if allowed_headers is None:
        allowed_headers = [
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Content-Type",
            "Authorization",
            "X-Requested-With",
            "X-CSRF-Token",
            "X-Client-Version",
            "User-Agent"
        ]
    
    # Default exposed headers
    if expose_headers is None:
        expose_headers = [
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset", 
            "X-Rate-Limit-Limit",
            "X-Process-Time",
            "X-Auth-Time",
            "X-Security-Time"
        ]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=allow_credentials,
        allow_methods=allowed_methods,
        allow_headers=allowed_headers,
        expose_headers=expose_headers,
        max_age=86400  # 24 hours
    )


def get_production_cors_config() -> dict:
    """Get CORS configuration for production environment"""
    return {
        "allowed_origins": [
            "https://ipai.app",
            "https://www.ipai.app",
            "https://app.ipai.com",
            "https://api.ipai.com"
        ],
        "allowed_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
        "allowed_headers": [
            "Accept",
            "Content-Type", 
            "Authorization",
            "X-Requested-With"
        ],
        "allow_credentials": True,
        "expose_headers": [
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset"
        ]
    }


def get_development_cors_config() -> dict:
    """Get CORS configuration for development environment"""
    return {
        "allowed_origins": [
            "http://localhost:3000",
            "http://localhost:3001", 
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
            "http://127.0.0.1:8080",
            "http://localhost:5173",  # Vite default port
            "http://127.0.0.1:5173"
        ],
        "allowed_methods": ["*"],  # Allow all methods in development
        "allowed_headers": ["*"],  # Allow all headers in development
        "allow_credentials": True,
        "expose_headers": [
            "X-Rate-Limit-Remaining",
            "X-Rate-Limit-Reset",
            "X-Rate-Limit-Limit", 
            "X-Process-Time",
            "X-Auth-Time",
            "X-Security-Time"
        ]
    }


def validate_origin(origin: str, allowed_origins: List[str]) -> bool:
    """Validate if origin is allowed"""
    
    if not origin:
        return False
    
    # Exact match
    if origin in allowed_origins:
        return True
    
    # Wildcard support
    for allowed in allowed_origins:
        if allowed == "*":
            return True
        
        # Simple subdomain matching
        if allowed.startswith("*."):
            domain = allowed[2:]
            if origin.endswith(f".{domain}") or origin == domain:
                return True
    
    return False


class CustomCORSMiddleware:
    """Custom CORS middleware with additional security features"""
    
    def __init__(
        self,
        app,
        allowed_origins: List[str],
        allowed_methods: List[str] = None,
        allowed_headers: List[str] = None,
        allow_credentials: bool = True,
        max_age: int = 86400,
        vary_header: bool = True
    ):
        self.app = app
        self.allowed_origins = allowed_origins
        self.allowed_methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]
        self.allowed_headers = allowed_headers or ["*"]
        self.allow_credentials = allow_credentials
        self.max_age = max_age
        self.vary_header = vary_header
    
    async def __call__(self, scope, receive, send):
        """ASGI middleware implementation"""
        
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Get request headers
        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode()
        
        # Handle preflight requests
        if scope["method"] == "OPTIONS":
            await self._handle_preflight(scope, receive, send, origin)
            return
        
        # Add CORS headers to response
        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                self._add_cors_headers(message, origin)
            await send(message)
        
        await self.app(scope, receive, send_with_cors)
    
    async def _handle_preflight(self, scope, receive, send, origin):
        """Handle CORS preflight requests"""
        
        # Validate origin
        if not validate_origin(origin, self.allowed_origins):
            # Reject preflight
            await send({
                "type": "http.response.start",
                "status": 403,
                "headers": [[b"content-type", b"application/json"]]
            })
            await send({
                "type": "http.response.body",
                "body": b'{"error": "CORS: Origin not allowed"}'
            })
            return
        
        # Build preflight response
        headers = [
            [b"access-control-allow-origin", origin.encode()],
            [b"access-control-allow-methods", ", ".join(self.allowed_methods).encode()],
            [b"access-control-allow-headers", ", ".join(self.allowed_headers).encode()],
            [b"access-control-max-age", str(self.max_age).encode()]
        ]
        
        if self.allow_credentials:
            headers.append([b"access-control-allow-credentials", b"true"])
        
        if self.vary_header:
            headers.append([b"vary", b"Origin"])
        
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": headers
        })
        await send({"type": "http.response.body"})
    
    def _add_cors_headers(self, message, origin):
        """Add CORS headers to response"""
        
        if not validate_origin(origin, self.allowed_origins):
            return  # Don't add CORS headers for invalid origins
        
        cors_headers = [
            [b"access-control-allow-origin", origin.encode()],
            [b"access-control-expose-headers", b"X-Rate-Limit-Remaining, X-Rate-Limit-Reset"]
        ]
        
        if self.allow_credentials:
            cors_headers.append([b"access-control-allow-credentials", b"true"])
        
        if self.vary_header:
            cors_headers.append([b"vary", b"Origin"])
        
        # Add to existing headers
        message["headers"].extend(cors_headers)