"""
Rate Limiting Middleware

This module provides rate limiting functionality for the IPAI API
with multiple strategies and security features.
"""

import time
import asyncio
import logging
from typing import Dict, Optional, Tuple, List
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import hashlib
import redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10
    window_size: int = 60  # seconds
    enabled: bool = True
    
    # Per-endpoint limits
    endpoint_limits: Dict[str, Dict[str, int]] = None
    
    # User-specific limits
    user_limits: Dict[str, Dict[str, int]] = None
    
    # IP-specific limits
    ip_limits: Dict[str, Dict[str, int]] = None


class TokenBucket:
    """Token bucket algorithm implementation"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate
        self.last_refill = time.time()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        now = time.time()
        
        # Refill tokens
        time_passed = now - self.last_refill
        tokens_to_add = time_passed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill = now
        
        # Check if we can consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        
        return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait before tokens are available"""
        if self.tokens >= tokens:
            return 0.0
        
        needed_tokens = tokens - self.tokens
        return needed_tokens / self.refill_rate


class SlidingWindowCounter:
    """Sliding window counter for rate limiting"""
    
    def __init__(self, window_size: int, max_requests: int):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()
    
    def is_allowed(self) -> Tuple[bool, int]:
        """Check if request is allowed"""
        now = time.time()
        
        # Remove old requests
        while self.requests and self.requests[0] <= now - self.window_size:
            self.requests.popleft()
        
        # Check if we can allow this request
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True, self.max_requests - len(self.requests)
        
        return False, 0
    
    def get_reset_time(self) -> float:
        """Get time when window resets"""
        if not self.requests:
            return 0.0
        
        return self.requests[0] + self.window_size


class InMemoryRateLimiter:
    """In-memory rate limiter with multiple strategies"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.token_buckets = {}
        self.sliding_windows = {}
        self.request_counts = defaultdict(lambda: defaultdict(int))
        self.window_starts = defaultdict(lambda: defaultdict(float))
        self._cleanup_interval = 300  # 5 minutes
        self._last_cleanup = time.time()
    
    def _cleanup(self):
        """Clean up old entries"""
        now = time.time()
        
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        # Clean up sliding windows
        keys_to_remove = []
        for key, window in self.sliding_windows.items():
            if not window.requests or window.requests[-1] < now - window.window_size * 2:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.sliding_windows[key]
        
        # Clean up token buckets (keep recently used ones)
        keys_to_remove = []
        for key, bucket in self.token_buckets.items():
            if bucket.last_refill < now - 3600:  # 1 hour
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            del self.token_buckets[key]
        
        self._last_cleanup = now
    
    def _get_bucket_key(self, identifier: str, limit_type: str) -> str:
        """Generate bucket key"""
        return f"{identifier}:{limit_type}"
    
    def check_rate_limit(self, identifier: str, endpoint: str = None, user_id: str = None) -> Tuple[bool, Dict]:
        """Check if request is within rate limits"""
        self._cleanup()
        
        now = time.time()
        limit_info = {
            'allowed': True,
            'remaining': 0,
            'reset_time': 0,
            'retry_after': 0
        }
        
        # Check minute limit
        minute_key = self._get_bucket_key(identifier, 'minute')
        if minute_key not in self.sliding_windows:
            self.sliding_windows[minute_key] = SlidingWindowCounter(60, self.config.requests_per_minute)
        
        allowed, remaining = self.sliding_windows[minute_key].is_allowed()
        if not allowed:
            limit_info['allowed'] = False
            limit_info['remaining'] = 0
            limit_info['reset_time'] = self.sliding_windows[minute_key].get_reset_time()
            limit_info['retry_after'] = limit_info['reset_time'] - now
            return False, limit_info
        
        limit_info['remaining'] = remaining
        limit_info['reset_time'] = self.sliding_windows[minute_key].get_reset_time()
        
        # Check hour limit
        hour_key = self._get_bucket_key(identifier, 'hour')
        if hour_key not in self.sliding_windows:
            self.sliding_windows[hour_key] = SlidingWindowCounter(3600, self.config.requests_per_hour)
        
        allowed, remaining_hour = self.sliding_windows[hour_key].is_allowed()
        if not allowed:
            limit_info['allowed'] = False
            limit_info['remaining'] = 0
            limit_info['reset_time'] = self.sliding_windows[hour_key].get_reset_time()
            limit_info['retry_after'] = limit_info['reset_time'] - now
            return False, limit_info
        
        # Use the lower remaining count
        limit_info['remaining'] = min(remaining, remaining_hour)
        
        # Check endpoint-specific limits
        if endpoint and self.config.endpoint_limits and endpoint in self.config.endpoint_limits:
            endpoint_limits = self.config.endpoint_limits[endpoint]
            endpoint_key = self._get_bucket_key(f"{identifier}:{endpoint}", 'endpoint')
            
            if endpoint_key not in self.sliding_windows:
                requests_per_minute = endpoint_limits.get('requests_per_minute', self.config.requests_per_minute)
                self.sliding_windows[endpoint_key] = SlidingWindowCounter(60, requests_per_minute)
            
            allowed, remaining_endpoint = self.sliding_windows[endpoint_key].is_allowed()
            if not allowed:
                limit_info['allowed'] = False
                limit_info['remaining'] = 0
                limit_info['reset_time'] = self.sliding_windows[endpoint_key].get_reset_time()
                limit_info['retry_after'] = limit_info['reset_time'] - now
                return False, limit_info
            
            limit_info['remaining'] = min(limit_info['remaining'], remaining_endpoint)
        
        # Check user-specific limits
        if user_id and self.config.user_limits and user_id in self.config.user_limits:
            user_limits = self.config.user_limits[user_id]
            user_key = self._get_bucket_key(f"user:{user_id}", 'user')
            
            if user_key not in self.sliding_windows:
                requests_per_minute = user_limits.get('requests_per_minute', self.config.requests_per_minute)
                self.sliding_windows[user_key] = SlidingWindowCounter(60, requests_per_minute)
            
            allowed, remaining_user = self.sliding_windows[user_key].is_allowed()
            if not allowed:
                limit_info['allowed'] = False
                limit_info['remaining'] = 0
                limit_info['reset_time'] = self.sliding_windows[user_key].get_reset_time()
                limit_info['retry_after'] = limit_info['reset_time'] - now
                return False, limit_info
            
            limit_info['remaining'] = min(limit_info['remaining'], remaining_user)
        
        return True, limit_info


class RedisRateLimiter:
    """Redis-based rate limiter for distributed systems"""
    
    def __init__(self, redis_client, config: RateLimitConfig):
        self.redis = redis_client
        self.config = config
    
    async def check_rate_limit(self, identifier: str, endpoint: str = None, user_id: str = None) -> Tuple[bool, Dict]:
        """Check rate limit using Redis"""
        try:
            # Use Lua script for atomic operations
            lua_script = """
            local key = KEYS[1]
            local window = tonumber(ARGV[1])
            local limit = tonumber(ARGV[2])
            local current_time = tonumber(ARGV[3])
            
            -- Remove old entries
            redis.call('zremrangebyscore', key, 0, current_time - window)
            
            -- Count current requests
            local current_count = redis.call('zcard', key)
            
            if current_count < limit then
                -- Add current request
                redis.call('zadd', key, current_time, current_time)
                redis.call('expire', key, window)
                return {1, limit - current_count - 1}
            else
                return {0, 0}
            end
            """
            
            now = time.time()
            
            # Check minute limit
            minute_key = f"rate_limit:{identifier}:minute"
            result = await self.redis.eval(lua_script, 1, minute_key, 60, self.config.requests_per_minute, now)
            
            if result[0] == 0:
                return False, {
                    'allowed': False,
                    'remaining': 0,
                    'reset_time': now + 60,
                    'retry_after': 60
                }
            
            # Check hour limit
            hour_key = f"rate_limit:{identifier}:hour"
            result_hour = await self.redis.eval(lua_script, 1, hour_key, 3600, self.config.requests_per_hour, now)
            
            if result_hour[0] == 0:
                return False, {
                    'allowed': False,
                    'remaining': 0,
                    'reset_time': now + 3600,
                    'retry_after': 3600
                }
            
            return True, {
                'allowed': True,
                'remaining': min(result[1], result_hour[1]),
                'reset_time': now + 60,
                'retry_after': 0
            }
            
        except Exception as e:
            logger.error(f"Redis rate limiter error: {e}")
            # Fallback to allowing request if Redis fails
            return True, {
                'allowed': True,
                'remaining': 100,
                'reset_time': now + 60,
                'retry_after': 0
            }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware for FastAPI"""
    
    def __init__(self, app, config: RateLimitConfig, redis_client=None):
        super().__init__(app)
        self.config = config
        
        if redis_client:
            self.limiter = RedisRateLimiter(redis_client, config)
        else:
            self.limiter = InMemoryRateLimiter(config)
        
        # Track suspicious activity
        self.suspicious_ips = set()
        self.repeated_violations = defaultdict(int)
    
    async def dispatch(self, request: Request, call_next):
        """Process rate limiting for requests"""
        
        if not self.config.enabled:
            return await call_next(request)
        
        # Get client identifier
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        user_id = self._get_user_id(request)
        
        # Check if IP is suspicious
        if client_ip in self.suspicious_ips:
            return self._create_rate_limit_response(
                "IP blocked due to suspicious activity",
                status_code=429
            )
        
        # Check rate limits
        try:
            allowed, limit_info = await self._check_limits(client_ip, endpoint, user_id)
            
            if not allowed:
                # Track repeated violations
                self.repeated_violations[client_ip] += 1
                
                # Mark IP as suspicious after too many violations
                if self.repeated_violations[client_ip] > 10:
                    self.suspicious_ips.add(client_ip)
                    logger.warning(f"IP {client_ip} marked as suspicious due to repeated rate limit violations")
                
                return self._create_rate_limit_response(
                    "Rate limit exceeded",
                    limit_info=limit_info
                )
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, limit_info)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limit middleware error: {e}")
            # Allow request if rate limiting fails
            return await call_next(request)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address"""
        # Check forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _get_user_id(self, request: Request) -> Optional[str]:
        """Extract user ID from request"""
        # Try to get from JWT token
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                import jwt
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                return payload.get("sub")
            except:
                pass
        
        # Try to get from session
        if hasattr(request.state, 'user_id'):
            return request.state.user_id
        
        return None
    
    async def _check_limits(self, client_ip: str, endpoint: str, user_id: str) -> Tuple[bool, Dict]:
        """Check all applicable rate limits"""
        
        # Primary identifier is IP
        identifier = client_ip
        
        # If user is authenticated, use user ID as primary identifier
        if user_id:
            identifier = f"user:{user_id}"
        
        if hasattr(self.limiter, 'check_rate_limit'):
            if asyncio.iscoroutinefunction(self.limiter.check_rate_limit):
                return await self.limiter.check_rate_limit(identifier, endpoint, user_id)
            else:
                return self.limiter.check_rate_limit(identifier, endpoint, user_id)
        
        # Fallback
        return True, {'allowed': True, 'remaining': 100, 'reset_time': time.time() + 60}
    
    def _add_rate_limit_headers(self, response: Response, limit_info: Dict):
        """Add rate limit headers to response"""
        response.headers["X-RateLimit-Limit"] = str(self.config.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(limit_info.get('remaining', 0))
        response.headers["X-RateLimit-Reset"] = str(int(limit_info.get('reset_time', 0)))
    
    def _create_rate_limit_response(self, message: str, limit_info: Dict = None, status_code: int = 429):
        """Create rate limit exceeded response"""
        content = {
            "error": "Rate limit exceeded",
            "message": message,
            "timestamp": time.time()
        }
        
        if limit_info:
            content["retry_after"] = limit_info.get('retry_after', 60)
            content["reset_time"] = limit_info.get('reset_time', time.time() + 60)
        
        headers = {"Retry-After": str(int(limit_info.get('retry_after', 60))) if limit_info else "60"}
        
        return JSONResponse(
            status_code=status_code,
            content=content,
            headers=headers
        )


def create_rate_limit_config(
    requests_per_minute: int = 60,
    requests_per_hour: int = 1000,
    endpoint_limits: Dict[str, Dict] = None,
    user_limits: Dict[str, Dict] = None
) -> RateLimitConfig:
    """Create rate limit configuration"""
    
    # Default endpoint limits
    if endpoint_limits is None:
        endpoint_limits = {
            '/api/v1/coherence/calculate': {'requests_per_minute': 30},
            '/api/v1/llm/generate': {'requests_per_minute': 20},
            '/api/v1/assessment/start': {'requests_per_minute': 10},
            '/api/v1/auth/login': {'requests_per_minute': 5},
            '/api/v1/auth/register': {'requests_per_minute': 3}
        }
    
    return RateLimitConfig(
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        endpoint_limits=endpoint_limits,
        user_limits=user_limits
    )


# Utility functions
def get_client_identifier(request: Request) -> str:
    """Get unique client identifier for rate limiting"""
    # Try to get authenticated user ID
    auth_header = request.headers.get("authorization")
    if auth_header and auth_header.startswith("Bearer "):
        try:
            import jwt
            token = auth_header.split(" ")[1]
            payload = jwt.decode(token, options={"verify_signature": False})
            user_id = payload.get("sub")
            if user_id:
                return f"user:{user_id}"
        except:
            pass
    
    # Fall back to IP address
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return f"ip:{forwarded_for.split(',')[0].strip()}"
    
    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


def create_rate_limit_decorator(limiter, requests_per_minute: int = 60):
    """Create a decorator for rate limiting specific functions"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # This would need to be implemented based on context
            # For now, just call the function
            return await func(*args, **kwargs)
        return wrapper
    return decorator