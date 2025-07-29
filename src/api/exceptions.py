"""
Exception Handlers

Custom exception handlers for the IPAI API with
security-conscious error responses and logging.
"""

import logging
import time
import traceback
from typing import Dict, Any

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle validation errors with secure error messages"""
    
    # Log the full error for debugging
    logger.warning(
        f"Validation error on {request.method} {request.url.path}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "errors": exc.errors(),
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    # Create user-friendly error messages
    error_details = []
    for error in exc.errors():
        loc = " -> ".join(str(x) for x in error["loc"])
        msg = error["msg"]
        error_details.append(f"{loc}: {msg}")
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "message": "The provided data is invalid",
            "details": error_details,
            "status_code": 422,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers={"X-Error-Type": "validation"}
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with consistent formatting"""
    
    # Log error details
    log_level = logging.WARNING if exc.status_code < 500 else logging.ERROR
    logger.log(
        log_level,
        f"HTTP {exc.status_code}: {exc.detail}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "status_code": exc.status_code,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    # Determine error category
    error_category = "client_error" if 400 <= exc.status_code < 500 else "server_error"
    
    # Security: Don't expose sensitive information in error messages
    safe_detail = exc.detail
    if exc.status_code == 500:
        safe_detail = "Internal server error"
    elif exc.status_code == 401:
        safe_detail = "Authentication required"
    elif exc.status_code == 403:
        safe_detail = "Access forbidden"
    elif exc.status_code == 404:
        safe_detail = "Resource not found"
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_category.replace("_", " ").title(),
            "message": safe_detail,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers={
            "X-Error-Type": error_category,
            **getattr(exc, "headers", {})
        }
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle unexpected exceptions with security measures"""
    
    # Generate unique error ID for tracking
    import uuid
    error_id = str(uuid.uuid4())
    
    # Log full error details
    logger.error(
        f"Unhandled exception [{error_id}]: {type(exc).__name__}: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "error_id": error_id,
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    # Security: Never expose internal error details to clients
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown"),
            "error_id": error_id
        },
        headers={"X-Error-Type": "server_error"}
    )


async def coherence_calculation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle coherence calculation specific errors"""
    
    logger.error(
        f"Coherence calculation error: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Coherence Calculation Error",
            "message": "Unable to process coherence calculation with provided parameters",
            "status_code": 422,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers={"X-Error-Type": "coherence_error"}
    )


async def llm_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle LLM processing errors"""
    
    logger.error(
        f"LLM processing error: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "LLM Service Error",
            "message": "Language model service is temporarily unavailable",
            "status_code": 503,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers={"X-Error-Type": "llm_error"}
    )


async def security_violation_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle security violations"""
    
    # Log security incident
    logger.critical(
        f"Security violation detected: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "user_agent": request.headers.get("user-agent", "unknown"),
            "headers": dict(request.headers),
            "security_event": True
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={
            "error": "Security Violation",
            "message": "Request blocked for security reasons",
            "status_code": 403,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers={
            "X-Error-Type": "security_violation",
            "X-Security-Block": "true"
        }
    )


async def rate_limit_exceeded_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle rate limit exceeded errors"""
    
    logger.warning(
        f"Rate limit exceeded",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown"
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "error": "Rate Limit Exceeded",
            "message": "Too many requests. Please slow down.",
            "status_code": 429,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown"),
            "retry_after": getattr(exc, "retry_after", 60)
        },
        headers={
            "X-Error-Type": "rate_limit",
            "Retry-After": str(getattr(exc, "retry_after", 60))
        }
    )


async def database_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle database errors"""
    
    logger.error(
        f"Database error: {str(exc)}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "exception_type": type(exc).__name__
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={
            "error": "Database Error",
            "message": "Database service is temporarily unavailable",
            "status_code": 503,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown")
        },
        headers={"X-Error-Type": "database_error"}
    )


class IPAIException(Exception):
    """Base exception for IPAI-specific errors"""
    
    def __init__(self, message: str, status_code: int = 500, details: Dict[str, Any] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class CoherenceCalculationError(IPAIException):
    """Raised when coherence calculation fails"""
    
    def __init__(self, message: str = "Coherence calculation failed", details: Dict[str, Any] = None):
        super().__init__(message, status_code=422, details=details)


class LLMProcessingError(IPAIException):
    """Raised when LLM processing fails"""
    
    def __init__(self, message: str = "LLM processing failed", details: Dict[str, Any] = None):
        super().__init__(message, status_code=503, details=details)


class SecurityViolationError(IPAIException):
    """Raised when security violation is detected"""
    
    def __init__(self, message: str = "Security violation detected", details: Dict[str, Any] = None):
        super().__init__(message, status_code=403, details=details)


class AuthenticationError(IPAIException):
    """Raised when authentication fails"""
    
    def __init__(self, message: str = "Authentication failed", details: Dict[str, Any] = None):
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(IPAIException):
    """Raised when authorization fails"""
    
    def __init__(self, message: str = "Authorization failed", details: Dict[str, Any] = None):
        super().__init__(message, status_code=403, details=details)


class ResourceNotFoundError(IPAIException):
    """Raised when requested resource is not found"""
    
    def __init__(self, message: str = "Resource not found", details: Dict[str, Any] = None):
        super().__init__(message, status_code=404, details=details)


class ValidationError(IPAIException):
    """Raised when data validation fails"""
    
    def __init__(self, message: str = "Data validation failed", details: Dict[str, Any] = None):
        super().__init__(message, status_code=422, details=details)


# Custom exception handler for IPAI exceptions
async def ipai_exception_handler(request: Request, exc: IPAIException) -> JSONResponse:
    """Handle IPAI-specific exceptions"""
    
    logger.error(
        f"IPAI Exception: {exc.message}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "method": request.method,
            "path": request.url.path,
            "exception_type": type(exc).__name__,
            "details": exc.details
        }
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": type(exc).__name__.replace("Error", "").replace("Exception", ""),
            "message": exc.message,
            "status_code": exc.status_code,
            "timestamp": time.time(),
            "request_id": getattr(request.state, "request_id", "unknown"),
            "details": exc.details
        },
        headers={"X-Error-Type": "ipai_exception"}
    )


# Exception handler registry
EXCEPTION_HANDLERS = {
    RequestValidationError: validation_exception_handler,
    ValidationError: validation_exception_handler,
    HTTPException: http_exception_handler,
    StarletteHTTPException: http_exception_handler,
    IPAIException: ipai_exception_handler,
    CoherenceCalculationError: coherence_calculation_error_handler,
    LLMProcessingError: llm_error_handler,
    SecurityViolationError: security_violation_handler,
    Exception: general_exception_handler
}