"""
Rate Limiting Middleware

This module provides rate limiting middleware for the IPAI API.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import asyncio
from collections import defaultdict
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using token bucket algorithm"""
    
    def __init__(
        self, 
        app,
        calls_per_minute: int = 60,
        calls_per_hour: int = 1000,
        burst_size: int = 10
    ):
        super().__init__(app)
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
        self.burst_size = burst_size
        
        # Storage for rate limit buckets
        self.minute_buckets: Dict[str, List[float]] = defaultdict(list)
        self.hour_buckets: Dict[str, List[float]] = defaultdict(list)
        self.burst_buckets: Dict[str, int] = defaultdict(int)
        self.last_refill: Dict[str, float] = defaultdict(float)
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup():
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                await self._cleanup_old_entries()
        
        self._cleanup_task = asyncio.create_task(cleanup())
    
    async def dispatch(self, request: Request, call_next):
        """Process rate limiting for incoming requests"""
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        allowed, reason = await self._check_rate_limits(client_id, request)
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_id}: {reason}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "message": reason,
                    "retry_after": self._get_retry_after(client_id)
                },
                headers={
                    "Retry-After": str(self._get_retry_after(client_id)),
                    "X-RateLimit-Limit": str(self.calls_per_minute),
                    "X-RateLimit-Remaining": str(self._get_remaining_calls(client_id)),
                    "X-RateLimit-Reset": str(int(time.time() + 60))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.calls_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(self._get_remaining_calls(client_id))
        response.headers["X-RateLimit-Reset"] = str(int(time.time() + 60))
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        
        # Try to get user ID from request if authenticated
        user_id = getattr(request.state, 'user_id', None)
        if user_id:
            return f"user:{user_id}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for forwarded IP headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            client_ip = real_ip
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limits(self, client_id: str, request: Request) -> tuple[bool, str]:
        """Check if request is within rate limits"""
        
        async with self._lock:
            current_time = time.time()
            
            # Refill burst bucket
            self._refill_burst_bucket(client_id, current_time)
            
            # Check burst limit first
            if self.burst_buckets[client_id] <= 0:
                return False, "Burst limit exceeded"
            
            # Check minute limit
            minute_ago = current_time - 60
            self.minute_buckets[client_id] = [
                t for t in self.minute_buckets[client_id] if t > minute_ago
            ]
            
            if len(self.minute_buckets[client_id]) >= self.calls_per_minute:
                return False, f"Minute limit exceeded ({self.calls_per_minute} calls/minute)"
            
            # Check hour limit
            hour_ago = current_time - 3600
            self.hour_buckets[client_id] = [
                t for t in self.hour_buckets[client_id] if t > hour_ago
            ]
            
            if len(self.hour_buckets[client_id]) >= self.calls_per_hour:
                return False, f"Hour limit exceeded ({self.calls_per_hour} calls/hour)"
            
            # Special limits for specific endpoints
            if not self._check_endpoint_limits(client_id, request):
                return False, "Endpoint-specific limit exceeded"
            
            # Record the request
            self.minute_buckets[client_id].append(current_time)
            self.hour_buckets[client_id].append(current_time)
            self.burst_buckets[client_id] -= 1
            
            return True, "OK"
    
    def _refill_burst_bucket(self, client_id: str, current_time: float):
        """Refill burst bucket based on time elapsed"""
        
        last_refill = self.last_refill[client_id]
        if last_refill == 0:
            last_refill = current_time
            self.last_refill[client_id] = current_time
            self.burst_buckets[client_id] = self.burst_size
            return
        
        # Refill at rate of 1 token per second
        time_elapsed = current_time - last_refill
        tokens_to_add = int(time_elapsed)
        
        if tokens_to_add > 0:
            self.burst_buckets[client_id] = min(
                self.burst_size,
                self.burst_buckets[client_id] + tokens_to_add
            )
            self.last_refill[client_id] = current_time
    
    def _check_endpoint_limits(self, client_id: str, request: Request) -> bool:
        """Check endpoint-specific rate limits"""
        
        path = request.url.path
        
        # Stricter limits for sensitive endpoints
        if "/llm/" in path:
            # LLM endpoints: max 10/minute
            minute_ago = time.time() - 60
            llm_calls = [
                t for t in self.minute_buckets[client_id] if t > minute_ago
            ]
            return len(llm_calls) < 10
        
        elif "/assessment/" in path:
            # Assessment endpoints: max 5/minute
            minute_ago = time.time() - 60
            assessment_calls = [
                t for t in self.minute_buckets[client_id] if t > minute_ago
            ]
            return len(assessment_calls) < 5
        
        elif "/identity/login" in path or "/identity/register" in path:
            # Auth endpoints: max 5/minute
            minute_ago = time.time() - 60
            auth_calls = [
                t for t in self.minute_buckets[client_id] if t > minute_ago
            ]
            return len(auth_calls) < 5
        
        return True
    
    def _get_remaining_calls(self, client_id: str) -> int:
        """Get remaining calls for the minute"""
        current_time = time.time()
        minute_ago = current_time - 60
        
        recent_calls = [
            t for t in self.minute_buckets[client_id] if t > minute_ago
        ]
        
        return max(0, self.calls_per_minute - len(recent_calls))
    
    def _get_retry_after(self, client_id: str) -> int:
        """Get retry after time in seconds"""
        
        if not self.minute_buckets[client_id]:
            return 1
        
        # Find oldest call in current minute
        current_time = time.time()
        minute_ago = current_time - 60
        
        calls_in_minute = [
            t for t in self.minute_buckets[client_id] if t > minute_ago
        ]
        
        if calls_in_minute:
            oldest_call = min(calls_in_minute)
            return max(1, int(oldest_call + 60 - current_time))
        
        return 1
    
    async def _cleanup_old_entries(self):
        """Clean up old rate limit entries"""
        
        async with self._lock:
            current_time = time.time()
            hour_ago = current_time - 3600
            
            # Clean up old entries
            for client_id in list(self.minute_buckets.keys()):
                # Clean minute buckets
                self.minute_buckets[client_id] = [
                    t for t in self.minute_buckets[client_id] if t > current_time - 60
                ]
                
                # Clean hour buckets
                self.hour_buckets[client_id] = [
                    t for t in self.hour_buckets[client_id] if t > hour_ago
                ]
                
                # Remove empty buckets
                if not self.minute_buckets[client_id] and not self.hour_buckets[client_id]:
                    del self.minute_buckets[client_id]
                    del self.hour_buckets[client_id]
                    if client_id in self.burst_buckets:
                        del self.burst_buckets[client_id]
                    if client_id in self.last_refill:
                        del self.last_refill[client_id]
    
    def __del__(self):
        """Cleanup when middleware is destroyed"""
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()


class EndpointRateLimit:
    """Decorator for endpoint-specific rate limiting"""
    
    def __init__(self, calls_per_minute: int, calls_per_hour: int = None):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour or calls_per_minute * 60
    
    def __call__(self, func):
        """Apply rate limiting to endpoint"""
        
        async def wrapper(*args, **kwargs):
            # This would integrate with the main rate limiting middleware
            # For now, just pass through
            return await func(*args, **kwargs)
        
        wrapper._rate_limit = {
            'calls_per_minute': self.calls_per_minute,
            'calls_per_hour': self.calls_per_hour
        }
        
        return wrapper


# Convenience decorators for common rate limits
llm_rate_limit = EndpointRateLimit(calls_per_minute=10)
assessment_rate_limit = EndpointRateLimit(calls_per_minute=5)
auth_rate_limit = EndpointRateLimit(calls_per_minute=5)