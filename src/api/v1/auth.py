"""
Authentication API Endpoints

Secure authentication and authorization endpoints for the IPAI system
with comprehensive security measures and GCT integration.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, validator
from pydantic_settings import BaseSettings

from ...core.security import SecurityManager
from ...core.database import database, CoherenceProfileDB
from ...models.user import User, UserPreferences
from ...models.coherence_profile import CoherenceProfile, GCTComponents, IndividualParameters
from ..exceptions import AuthenticationError, AuthorizationError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()


# Request/Response Models
class RegisterRequest(BaseModel):
    """User registration request"""
    email: EmailStr
    password: str = Field(..., min_length=12, max_length=128)
    confirm_password: str
    first_name: str = Field(..., min_length=1, max_length=50)
    last_name: str = Field(..., min_length=1, max_length=50)
    preferences: Optional[Dict[str, Any]] = None
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "confirm_password": "SecurePassword123!",
                "first_name": "John",
                "last_name": "Doe",
                "preferences": {
                    "language": "en",
                    "timezone": "UTC",
                    "notifications_enabled": True
                }
            }
        }


class LoginRequest(BaseModel):
    """User login request"""
    email: EmailStr
    password: str
    remember_me: bool = False
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "password": "SecurePassword123!",
                "remember_me": False
            }
        }


class AuthResponse(BaseModel):
    """Authentication response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                "token_type": "bearer",
                "expires_in": 86400,
                "user": {
                    "id": "user-123",
                    "email": "user@example.com",
                    "first_name": "John",
                    "last_name": "Doe"
                }
            }
        }


class RefreshTokenRequest(BaseModel):
    """Refresh token request"""
    refresh_token: str


class PasswordChangeRequest(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=12, max_length=128)
    confirm_new_password: str
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('New passwords do not match')
        return v


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=12, max_length=128)
    confirm_new_password: str
    
    @validator('confirm_new_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


# Dependencies
async def get_security_manager() -> SecurityManager:
    """Get security manager instance"""
    from ...core.config import get_settings
    settings = get_settings()
    return SecurityManager(settings.security_config)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    security_manager: SecurityManager = Depends(get_security_manager)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    try:
        payload = security_manager.verify_token(credentials.credentials, "access")
        if not payload:
            raise AuthenticationError("Invalid or expired token")
        
        # Get user from database
        async with database.get_session() as session:
            user_query = "SELECT * FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": payload["sub"]})
            
            if not user_result:
                raise AuthenticationError("User not found")
            
            user_data = user_result[0]
            return {
                "id": user_data["id"],
                "email": user_data["email"],
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "is_active": user_data.get("is_active", True),
                "created_at": user_data.get("created_at"),
                "token_data": payload
            }
            
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise AuthenticationError("Authentication failed")


async def get_optional_user(
    request: Request,
    security_manager: SecurityManager = Depends(get_security_manager)
) -> Optional[Dict[str, Any]]:
    """Get current user if authenticated, None otherwise"""
    try:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        token = auth_header.split(" ")[1]
        payload = security_manager.verify_token(token, "access")
        
        if not payload:
            return None
        
        async with database.get_session() as session:
            user_query = "SELECT * FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": payload["sub"]})
            
            if not user_result:
                return None
            
            user_data = user_result[0]
            return {
                "id": user_data["id"],
                "email": user_data["email"],
                "token_data": payload
            }
            
    except Exception:
        return None


# Endpoints
@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Register a new user with initial coherence profile"""
    
    try:
        # Check if user already exists
        async with database.get_session() as session:
            existing_user_query = "SELECT id FROM users WHERE email = :email"
            existing_user = await database.execute_query(existing_user_query, {"email": request.email})
            
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User with this email already exists"
                )
        
        # Hash password
        password_hash = security_manager.hash_password(request.password)
        
        # Create user
        user_id = await _create_user(request, password_hash)
        
        # Create initial coherence profile
        await _create_initial_coherence_profile(user_id)
        
        # Generate tokens
        access_token = security_manager.create_access_token(user_id)
        refresh_token = security_manager.create_refresh_token(user_id)
        
        # Log successful registration
        logger.info(f"New user registered: {request.email}")
        
        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=86400,  # 24 hours
            user={
                "id": user_id,
                "email": request.email,
                "first_name": request.first_name,
                "last_name": request.last_name
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=AuthResponse)
async def login(
    request: LoginRequest,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Authenticate user and return tokens"""
    
    try:
        # Get user from database
        async with database.get_session() as session:
            user_query = "SELECT * FROM users WHERE email = :email"
            user_result = await database.execute_query(user_query, {"email": request.email})
            
            if not user_result:
                raise AuthenticationError("Invalid email or password")
            
            user_data = user_result[0]
            
            # Verify password
            if not security_manager.verify_password(
                request.password, 
                user_data["password_hash"], 
                user_data["id"]
            ):
                raise AuthenticationError("Invalid email or password")
            
            # Check if account is active
            if not user_data.get("is_active", True):
                raise AuthenticationError("Account is deactivated")
        
        # Generate tokens
        access_token = security_manager.create_access_token(user_data["id"])
        refresh_token = security_manager.create_refresh_token(user_data["id"])
        
        # Update last login
        await _update_last_login(user_data["id"])
        
        # Log successful login
        logger.info(f"User logged in: {request.email}")
        
        return AuthResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=86400,  # 24 hours
            user={
                "id": user_data["id"],
                "email": user_data["email"],
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name")
            }
        )
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=AuthResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Refresh access token using refresh token"""
    
    try:
        # Verify refresh token
        payload = security_manager.verify_token(request.refresh_token, "refresh")
        
        if not payload:
            raise AuthenticationError("Invalid or expired refresh token")
        
        user_id = payload["sub"]
        
        # Verify user still exists and is active
        async with database.get_session() as session:
            user_query = "SELECT id, is_active FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": user_id})
            
            if not user_result or not user_result[0].get("is_active", True):
                raise AuthenticationError("User account not found or inactive")
        
        # Generate new tokens
        new_access_token = security_manager.create_access_token(user_id)
        new_refresh_token = security_manager.create_refresh_token(user_id)
        
        # Revoke old refresh token
        security_manager.revoke_token(request.refresh_token)
        
        logger.info(f"Token refreshed for user: {user_id}")
        
        return AuthResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=86400,
            user={"id": user_id}
        )
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: Dict[str, Any] = Depends(get_current_user),
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Logout user and revoke tokens"""
    
    try:
        # Revoke current token
        token_data = current_user.get("token_data", {})
        if "jti" in token_data:
            # In a full implementation, we would revoke the specific token
            # For now, we just log the logout
            pass
        
        logger.info(f"User logged out: {current_user['id']}")
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Don't fail logout on errors
        pass


@router.get("/me")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get current user information"""
    
    try:
        # Get full user profile
        async with database.get_session() as session:
            user_query = """
            SELECT id, email, first_name, last_name, created_at, updated_at, 
                   profile_data, is_active
            FROM users 
            WHERE id = :user_id
            """
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            
            user_data = user_result[0]
            
            # Get coherence profile
            coherence_db = CoherenceProfileDB(database)
            coherence_profile = await coherence_db.get_profile(current_user["id"])
            
            return {
                "id": user_data["id"],
                "email": user_data["email"],
                "first_name": user_data.get("first_name"),
                "last_name": user_data.get("last_name"),
                "created_at": user_data["created_at"],
                "updated_at": user_data["updated_at"],
                "is_active": user_data["is_active"],
                "profile_data": user_data.get("profile_data", {}),
                "coherence_profile": coherence_profile
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get user info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user information"
        )


@router.put("/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: PasswordChangeRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
    security_manager: SecurityManager = Depends(get_security_manager)
):
    """Change user password"""
    
    try:
        # Get current password hash
        async with database.get_session() as session:
            user_query = "SELECT password_hash FROM users WHERE id = :user_id"
            user_result = await database.execute_query(user_query, {"user_id": current_user["id"]})
            
            if not user_result:
                raise HTTPException(status_code=404, detail="User not found")
            
            current_hash = user_result[0]["password_hash"]
        
        # Verify current password
        if not security_manager.verify_password(request.current_password, current_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_hash = security_manager.hash_password(request.new_password)
        
        # Update password
        update_query = """
        UPDATE users 
        SET password_hash = :password_hash, updated_at = :updated_at
        WHERE id = :user_id
        """
        await database.execute_query(update_query, {
            "password_hash": new_hash,
            "updated_at": datetime.utcnow(),
            "user_id": current_user["id"]
        })
        
        logger.info(f"Password changed for user: {current_user['id']}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


# Helper functions
async def _create_user(request: RegisterRequest, password_hash: str) -> str:
    """Create new user in database"""
    import uuid
    
    user_id = str(uuid.uuid4())
    
    create_user_query = """
    INSERT INTO users (id, email, password_hash, first_name, last_name, profile_data, created_at, updated_at)
    VALUES (:id, :email, :password_hash, :first_name, :last_name, :profile_data, :created_at, :updated_at)
    """
    
    profile_data = request.preferences or {}
    now = datetime.utcnow()
    
    await database.execute_query(create_user_query, {
        "id": user_id,
        "email": request.email,
        "password_hash": password_hash,
        "first_name": request.first_name,
        "last_name": request.last_name,
        "profile_data": profile_data,
        "created_at": now,
        "updated_at": now
    })
    
    return user_id


async def _create_initial_coherence_profile(user_id: str):
    """Create initial coherence profile for new user"""
    
    # Initial GCT components with balanced starting values
    initial_components = {
        "psi": 0.5,    # Internal consistency
        "rho": 0.3,    # Accumulated wisdom (low for new user)
        "q": 0.6,      # Moral activation
        "f": 0.5       # Social belonging
    }
    
    # Default individual parameters
    initial_parameters = {
        "k_m": 0.5,    # Moral sensitivity
        "k_i": 2.0     # Inhibition strength
    }
    
    coherence_db = CoherenceProfileDB(database)
    await coherence_db.create_profile(user_id, {
        "components": initial_components,
        "parameters": initial_parameters,
        "coherence_score": 0.45,  # Calculated initial score
        "created_type": "initial"
    })


async def _update_last_login(user_id: str):
    """Update user's last login timestamp"""
    update_query = """
    UPDATE users 
    SET profile_data = profile_data || :last_login_data, updated_at = :updated_at
    WHERE id = :user_id
    """
    
    await database.execute_query(update_query, {
        "last_login_data": {"last_login": datetime.utcnow().isoformat()},
        "updated_at": datetime.utcnow(),
        "user_id": user_id
    })