"""
IPAI FastAPI Main Application

Comprehensive FastAPI application with advanced middleware integration,
security, performance optimization, and GCT functionality.
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

# Core imports
from ..core.config import get_settings, Settings
from ..core.database import database
from ..core.security import SecurityManager
from ..core.performance import performance_optimizer, cache_manager

# Middleware imports
from .middleware.cors import setup_cors
from .middleware.security import SecurityMiddleware
from .middleware.rate_limiter import RateLimitMiddleware, create_rate_limit_config

# Exception handlers
from .exceptions import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler,
    EXCEPTION_HANDLERS
)

# Router imports (will be created)
try:
    from .v1 import coherence, llm, identity, assessment, analytics, safety, config
except ImportError:
    # Temporary placeholder for development
    logger.warning("Some router modules not found - using placeholder routers")
    coherence = llm = identity = assessment = analytics = safety = config = type('Router', (), {'router': None})()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("Starting IPAI application...")
    
    # Initialize database
    try:
        await database.connect()
        await database.create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise
    
    # Initialize performance monitoring
    logger.info("Performance monitoring initialized")
    
    # Cache warmup
    cache_manager.set("app_status", "running", ttl=3600)
    logger.info("Cache initialized")
    
    logger.info("IPAI application started successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down IPAI application...")
    
    try:
        await database.disconnect()
        performance_optimizer.shutdown()
        logger.info("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")


def create_app(settings: Settings = None) -> FastAPI:
    """Create and configure the FastAPI application"""
    
    if settings is None:
        settings = get_settings()
    
    app = FastAPI(
        title="IPAI - Integrated Personal AI",
        description="Advanced AI system with Grounded Coherence Theory integration",
        version="1.0.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
        lifespan=lifespan
    )
    
    # Configure middleware
    setup_middleware(app, settings)
    
    # Configure exception handlers
    setup_exception_handlers(app)
    
    # Include routers
    setup_routes(app)
    
    # Configure static files
    setup_static_files(app, settings)
    
    return app


def setup_middleware(app: FastAPI, settings: Settings):
    """Setup application middleware"""
    
    # Trusted hosts (security) - production only
    if settings.is_production:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["ipai.app", "www.ipai.app", "api.ipai.com"]
        )
    
    # Compression
    if settings.ENABLE_COMPRESSION:
        app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # Rate limiting
    rate_limit_config = create_rate_limit_config(
        requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
        requests_per_hour=settings.RATE_LIMIT_PER_HOUR
    )
    app.add_middleware(RateLimitMiddleware, config=rate_limit_config)
    
    # Security middleware
    security_config = {
        'enable_csrf_protection': True,
        'enable_xss_protection': True,
        'enable_sql_injection_protection': True,
        'enable_path_traversal_protection': True,
        'max_request_size': 10 * 1024 * 1024,  # 10MB
        'blocked_user_agents': ['curl', 'wget', 'python-requests'] if settings.is_production else []
    }
    app.add_middleware(SecurityMiddleware, config=security_config)
    
    # CORS
    setup_cors(
        app,
        allowed_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True
    )
    
    # Performance monitoring middleware
    app.add_middleware(PerformanceMiddleware)
    
    # Request ID middleware
    app.add_middleware(RequestIDMiddleware)
    
    # Logging middleware
    app.add_middleware(LoggingMiddleware)
    
    # Coherence tracking middleware
    app.add_middleware(CoherenceTrackingMiddleware)


def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers"""
    
    # Register all exception handlers
    for exception_type, handler in EXCEPTION_HANDLERS.items():
        app.add_exception_handler(exception_type, handler)


def setup_routes(app: FastAPI):
    """Setup API routes"""
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Advanced health check endpoint"""
        try:
            # Check database
            db_health = await database.health_check()
            
            # Check cache
            cache_status = cache_manager.get_stats()
            
            # Check performance
            perf_metrics = performance_optimizer.get_metrics()
            
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "version": "1.0.0",
                "database": db_health,
                "cache": cache_status,
                "performance": perf_metrics.get('summary', {})
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Service unavailable")
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "name": "IPAI - Integrated Personal AI",
            "version": "1.0.0",
            "description": "Advanced AI system with Grounded Coherence Theory",
            "docs_url": "/docs",
            "health_url": "/health"
        }
    
    # API versioning
    api_prefix = "/api/v1"
    
    # Include routers with updated imports (will be created)
    if hasattr(coherence, 'router') and coherence.router:
        app.include_router(coherence.router, prefix=f"{api_prefix}/coherence", tags=["Coherence"])
    if hasattr(llm, 'router') and llm.router:
        app.include_router(llm.router, prefix=f"{api_prefix}/llm", tags=["LLM"])
    if hasattr(identity, 'router') and identity.router:
        app.include_router(identity.router, prefix=f"{api_prefix}/users", tags=["Users"])
    if hasattr(assessment, 'router') and assessment.router:
        app.include_router(assessment.router, prefix=f"{api_prefix}/assessment", tags=["Assessment"])
    if hasattr(analytics, 'router') and analytics.router:
        app.include_router(analytics.router, prefix=f"{api_prefix}/analytics", tags=["Analytics"])
    if hasattr(safety, 'router') and safety.router:
        app.include_router(safety.router, prefix=f"{api_prefix}/safety", tags=["Safety"])
    if hasattr(config, 'router') and config.router:
        app.include_router(config.router, prefix=f"{api_prefix}/config", tags=["Configuration"])


def setup_static_files(app: FastAPI, settings: Settings):
    """Configure static file serving"""
    
    import os
    
    # Serve uploaded files (in production, use CDN)
    if os.path.exists(settings.UPLOAD_DIR):
        app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
    
    # Serve static assets
    static_dir = "static"
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Middleware Classes
class PerformanceMiddleware(BaseHTTPMiddleware):
    """Performance monitoring middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Record metrics
        operation = f"{request.method} {request.url.path}"
        performance_optimizer.record_performance(
            operation,
            process_time,
            is_error=response.status_code >= 400
        )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        
        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Request ID middleware for tracing"""
    
    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        
        # Add to request state
        request.state.request_id = request_id
        
        # Process request
        response = await call_next(request)
        
        # Add request ID to response
        response.headers["X-Request-ID"] = request_id
        
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Calculate response time
        process_time = time.time() - start_time
        
        # Log response
        logger.info(
            f"Request completed",
            extra={
                "request_id": getattr(request.state, "request_id", "unknown"),
                "status_code": response.status_code,
                "process_time": process_time
            }
        )
        
        return response


class CoherenceTrackingMiddleware(BaseHTTPMiddleware):
    """Middleware to track coherence-related requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Check if this is a coherence-related request
        is_coherence_request = any([
            "/coherence" in request.url.path,
            "/assessment" in request.url.path,
            "/llm" in request.url.path
        ])
        
        if is_coherence_request:
            # Add coherence tracking
            request.state.track_coherence = True
            start_time = time.time()
        
        response = await call_next(request)
        
        if is_coherence_request:
            # Record coherence interaction
            process_time = time.time() - start_time
            
            # Log coherence usage
            logger.info(
                "Coherence interaction",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "endpoint": request.url.path,
                    "process_time": process_time,
                    "status_code": response.status_code
                }
            )
        
        return response


# Dependency injection
async def get_security_manager() -> SecurityManager:
    """Get security manager instance"""
    settings = get_settings()
    return SecurityManager(settings.security_config)


async def get_current_user(
    request: Request, 
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Get current authenticated user"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    token = auth_header.split(" ")[1]
    payload = security_manager.verify_token(token, "access")
    
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return {
        "user_id": payload["sub"],
        "token_data": payload
    }


async def get_optional_user(
    request: Request, 
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Get current user if authenticated, None otherwise"""
    try:
        return await get_current_user(request, security_manager)
    except HTTPException:
        return None


# Create the application instance
app = create_app()


# Development utilities
if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.RELOAD,
        workers=1 if settings.RELOAD else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )