"""
FastAPI Dependencies

This module provides dependency injection for the IPAI API,
including authentication, database connections, and service dependencies.
"""

from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Generator
import jwt
from datetime import datetime

from ..models.user import User, UserRole
from ..models.coherence_profile import CoherenceProfile
from ..core.security import SecurityManager
from ..core.database import Database
from ..coherence.gct_calculator import EnhancedGCTCalculator
from ..llm.interface import GCTLLMInterface

# Initialize security manager
security_manager = SecurityManager()
security = HTTPBearer(auto_error=False)


async def get_database(request: Request) -> Database:
    """Get database connection from app state"""
    if not hasattr(request.app.state, 'database'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not available"
        )
    return request.app.state.database


async def get_gct_calculator(request: Request) -> EnhancedGCTCalculator:
    """Get GCT calculator from app state"""
    if not hasattr(request.app.state, 'gct_calculator'):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="GCT Calculator not available"
        )
    return request.app.state.gct_calculator


async def get_llm_interface(request: Request) -> Optional[GCTLLMInterface]:
    """Get LLM interface from app state"""
    return getattr(request.app.state, 'llm_interface', None)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Database = Depends(get_database)
) -> User:
    """Get current authenticated user"""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Verify JWT token
        payload = security_manager.verify_token(credentials.credentials)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = await db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication error"
        )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Database = Depends(get_database)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None"""
    try:
        if credentials:
            return await get_current_user(credentials, db)
        return None
    except HTTPException:
        return None


async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current verified user"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email verification required"
        )
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """Get current admin user"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user


async def get_premium_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """Get current premium user"""
    if current_user.role not in [UserRole.PREMIUM, UserRole.RESEARCHER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Premium subscription required"
        )
    return current_user


async def get_researcher_user(
    current_user: User = Depends(get_current_verified_user)
) -> User:
    """Get current researcher user"""
    if current_user.role not in [UserRole.RESEARCHER, UserRole.ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Researcher privileges required"
        )
    return current_user


async def get_user_coherence_profile(
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
) -> Optional[CoherenceProfile]:
    """Get user's latest coherence profile"""
    try:
        return await db.get_latest_coherence_profile(current_user.id)
    except Exception:
        return None


async def require_user_coherence_profile(
    profile: Optional[CoherenceProfile] = Depends(get_user_coherence_profile)
) -> CoherenceProfile:
    """Require user to have a coherence profile"""
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Coherence profile required. Please complete an assessment first."
        )
    return profile


async def check_feature_access(
    feature: str,
    current_user: User = Depends(get_current_active_user)
) -> bool:
    """Check if user has access to a specific feature"""
    return current_user.can_access_feature(feature)


def require_feature_access(feature: str):
    """Dependency factory to require specific feature access"""
    async def _require_access(
        current_user: User = Depends(get_current_active_user)
    ) -> User:
        if not current_user.can_access_feature(feature):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access to '{feature}' feature required"
            )
        return current_user
    
    return _require_access


class RateLimitDependency:
    """Rate limiting dependency"""
    
    def __init__(self, calls_per_minute: int = 60, calls_per_hour: int = 1000):
        self.calls_per_minute = calls_per_minute
        self.calls_per_hour = calls_per_hour
    
    async def __call__(self, request: Request):
        """Check rate limits"""
        # This would integrate with the rate limiting middleware
        # For now, we'll just pass through
        return True


# Common rate limiters
standard_rate_limit = RateLimitDependency(60, 1000)
llm_rate_limit = RateLimitDependency(10, 100)  # More restrictive for LLM
assessment_rate_limit = RateLimitDependency(5, 50)  # Very restrictive for assessments


class DatabaseTransaction:
    """Database transaction context manager dependency"""
    
    def __init__(self, db: Database = Depends(get_database)):
        self.db = db
        self.transaction = None
    
    async def __aenter__(self):
        """Start transaction"""
        self.transaction = await self.db.begin_transaction()
        return self.transaction
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """End transaction"""
        if exc_type:
            await self.transaction.rollback()
        else:
            await self.transaction.commit()


async def get_db_transaction(
    db: Database = Depends(get_database)
) -> DatabaseTransaction:
    """Get database transaction context manager"""
    return DatabaseTransaction(db)


# Validation dependencies

async def validate_coherence_data(
    psi: float,
    rho: float,
    q: float,
    f: float
) -> bool:
    """Validate coherence component data"""
    from ..models.coherence_profile import validate_gct_components
    
    if not validate_gct_components(psi, rho, q, f):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid coherence component values. All values must be between 0 and 1."
        )
    return True


async def validate_individual_parameters(
    k_m: float,
    k_i: float
) -> bool:
    """Validate individual parameters"""
    from ..models.coherence_profile import validate_individual_parameters
    
    if not validate_individual_parameters(k_m, k_i):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid parameters. k_m must be 0.1-0.5, k_i must be 0.5-2.0."
        )
    return True


# Pagination dependencies

class PaginationParams:
    """Pagination parameters"""
    
    def __init__(self, skip: int = 0, limit: int = 50):
        if skip < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Skip parameter must be >= 0"
            )
        if limit <= 0 or limit > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit parameter must be between 1 and 100"
            )
        
        self.skip = skip
        self.limit = limit


async def get_pagination_params(
    skip: int = 0,
    limit: int = 50
) -> PaginationParams:
    """Get pagination parameters"""
    return PaginationParams(skip, limit)


# Request context dependencies

async def get_request_context(request: Request) -> dict:
    """Get request context information"""
    return {
        'ip_address': request.client.host if request.client else None,
        'user_agent': request.headers.get('user-agent'),
        'timestamp': datetime.utcnow(),
        'method': request.method,
        'url': str(request.url),
        'headers': dict(request.headers)
    }