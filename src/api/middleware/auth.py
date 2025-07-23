"""
Authentication Middleware

This module provides authentication middleware for the IPAI API.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import logging

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """Authentication middleware for API requests"""
    
    def __init__(self, app, excluded_paths: list = None):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/docs",
            "/redoc", 
            "/openapi.json",
            "/api/v1/identity/register",
            "/api/v1/identity/login",
            "/api/v1/identity/reset-password",
            "/api/v1/identity/check-username"
        ]
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for incoming requests"""
        
        # Skip authentication for excluded paths
        if self._is_excluded_path(request.url.path):
            return await call_next(request)
        
        # Skip for OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        start_time = time.time()
        
        try:
            # Check for authorization header
            auth_header = request.headers.get("authorization")
            
            if not auth_header:
                # Let the dependency injection handle this
                # This middleware just logs and monitors
                pass
            else:
                # Log successful authentication attempt
                logger.debug(f"Auth header present for {request.url.path}")
            
            # Process request
            response = await call_next(request)
            
            # Add authentication timing
            auth_time = time.time() - start_time
            response.headers["X-Auth-Time"] = str(auth_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {e}")
            return JSONResponse(
                status_code=500,
                content={"error": "Authentication processing failed"}
            )
    
    def _is_excluded_path(self, path: str) -> bool:
        """Check if path is excluded from authentication"""
        for excluded in self.excluded_paths:
            if path.startswith(excluded):
                return True
        return False