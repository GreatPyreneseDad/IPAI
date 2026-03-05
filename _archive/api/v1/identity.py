"""
Identity API Endpoints

This module provides REST endpoints for identity management,
including user registration, authentication, and profile management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, EmailStr, validator
from datetime import datetime, timedelta
import re

from ...models.user import User, UserRole
from ...core.security import SecurityManager
from ...core.database import Database
from ..dependencies import (
    get_current_user, get_current_active_user, get_database,
    get_optional_user, standard_rate_limit
)

router = APIRouter(prefix="/identity")
security_manager = SecurityManager()

# Pydantic models

class UserRegistration(BaseModel):
    """User registration request"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(None, max_length=100)
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain letters, numbers, underscores, and hyphens')
        return v
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v


class UserLogin(BaseModel):
    """User login request"""
    username: str
    password: str


class UserResponse(BaseModel):
    """User response model"""
    id: str
    email: str
    username: str
    full_name: Optional[str]
    role: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: Optional[datetime]
    coherence_profiles_count: int


class UserProfileUpdate(BaseModel):
    """User profile update request"""
    full_name: Optional[str] = Field(None, max_length=100)
    bio: Optional[str] = Field(None, max_length=500)
    preferences: Optional[Dict[str, Any]] = None


class PasswordChange(BaseModel):
    """Password change request"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=128)
    
    @validator('new_password')
    def validate_new_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse


class EmailVerificationRequest(BaseModel):
    """Email verification request"""
    token: str


class PasswordResetRequest(BaseModel):
    """Password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8, max_length=128)


# Endpoints

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserRegistration,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database),
    _: bool = Depends(standard_rate_limit)
):
    """Register a new user"""
    
    try:
        # Check if user already exists
        existing_user = await db.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        existing_username = await db.get_user_by_username(user_data.username)
        if existing_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
        
        # Hash password
        password_hash = security_manager.hash_password(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            username=user_data.username,
            password_hash=password_hash,
            full_name=user_data.full_name,
            role=UserRole.STANDARD,
            is_active=True,
            is_verified=False
        )
        
        # Save to database
        await db.create_user(user)
        
        # Send verification email in background
        background_tasks.add_task(send_verification_email, user.email, user.id)
        
        return UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            role=user.role.value,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login=user.last_login,
            coherence_profiles_count=user.coherence_profiles_count
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Database = Depends(get_database),
    _: bool = Depends(standard_rate_limit)
):
    """Login user and return access token"""
    
    try:
        # Get user by username or email
        user = await db.get_user_by_username(form_data.username)
        if not user:
            user = await db.get_user_by_email(form_data.username)
        
        if not user or not security_manager.verify_password(form_data.password, user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is inactive"
            )
        
        # Update last login
        user.update_last_login()
        await db.update_user(user)
        
        # Create access token
        access_token = security_manager.create_access_token(user.id)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=24 * 3600,  # 24 hours
            user=UserResponse(
                id=user.id,
                email=user.email,
                username=user.username,
                full_name=user.full_name,
                role=user.role.value,
                is_active=user.is_active,
                is_verified=user.is_verified,
                created_at=user.created_at,
                last_login=user.last_login,
                coherence_profiles_count=user.coherence_profiles_count
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user)
):
    """Get current user profile"""
    
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        role=current_user.role.value,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified,
        created_at=current_user.created_at,
        last_login=current_user.last_login,
        coherence_profiles_count=current_user.coherence_profiles_count
    )


@router.put("/me", response_model=UserResponse)
async def update_user_profile(
    profile_update: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Update user profile"""
    
    try:
        # Update user fields
        if profile_update.full_name is not None:
            current_user.full_name = profile_update.full_name
        
        if profile_update.bio is not None:
            current_user.bio = profile_update.bio
        
        if profile_update.preferences is not None:
            current_user.preferences.update(profile_update.preferences)
        
        current_user.updated_at = datetime.utcnow()
        
        # Save to database
        await db.update_user(current_user)
        
        return UserResponse(
            id=current_user.id,
            email=current_user.email,
            username=current_user.username,
            full_name=current_user.full_name,
            role=current_user.role.value,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            created_at=current_user.created_at,
            last_login=current_user.last_login,
            coherence_profiles_count=current_user.coherence_profiles_count
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed"
        )


@router.post("/change-password")
async def change_password(
    password_change: PasswordChange,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Change user password"""
    
    try:
        # Verify current password
        if not security_manager.verify_password(password_change.current_password, current_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Hash new password
        new_password_hash = security_manager.hash_password(password_change.new_password)
        
        # Update user
        current_user.password_hash = new_password_hash
        current_user.updated_at = datetime.utcnow()
        
        await db.update_user(current_user)
        
        return {
            "status": "success",
            "message": "Password changed successfully",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


@router.post("/verify-email")
async def verify_email(
    verification: EmailVerificationRequest,
    db: Database = Depends(get_database)
):
    """Verify user email"""
    
    try:
        # Verify token and get user ID
        user_id = await verify_email_token(verification.token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired verification token"
            )
        
        # Get user
        user = await db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Update verification status
        user.is_verified = True
        user.updated_at = datetime.utcnow()
        
        await db.update_user(user)
        
        return {
            "status": "success",
            "message": "Email verified successfully",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email verification failed"
        )


@router.post("/resend-verification")
async def resend_verification_email(
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Resend email verification"""
    
    if current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already verified"
        )
    
    # Send verification email in background
    background_tasks.add_task(send_verification_email, current_user.email, current_user.id)
    
    return {
        "status": "success",
        "message": "Verification email sent",
        "timestamp": datetime.utcnow()
    }


@router.post("/reset-password")
async def request_password_reset(
    reset_request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database),
    _: bool = Depends(standard_rate_limit)
):
    """Request password reset"""
    
    try:
        # Get user by email
        user = await db.get_user_by_email(reset_request.email)
        
        # Always return success to prevent email enumeration
        # but only send email if user exists
        if user:
            background_tasks.add_task(send_password_reset_email, user.email, user.id)
        
        return {
            "status": "success", 
            "message": "If the email exists, a password reset link has been sent",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset request failed"
        )


@router.post("/reset-password/confirm")
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    db: Database = Depends(get_database)
):
    """Confirm password reset"""
    
    try:
        # Verify token and get user ID
        user_id = await verify_password_reset_token(reset_confirm.token)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token"
            )
        
        # Get user
        user = await db.get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Hash new password
        new_password_hash = security_manager.hash_password(reset_confirm.new_password)
        
        # Update user
        user.password_hash = new_password_hash
        user.updated_at = datetime.utcnow()
        
        await db.update_user(user)
        
        return {
            "status": "success",
            "message": "Password reset successfully",
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password reset confirmation failed"
        )


@router.delete("/deactivate")
async def deactivate_account(
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Deactivate user account"""
    
    try:
        # Deactivate user
        current_user.is_active = False
        current_user.updated_at = datetime.utcnow()
        
        await db.update_user(current_user)
        
        return {
            "status": "success",
            "message": "Account deactivated successfully",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Account deactivation failed"
        )


@router.get("/check-username/{username}")
async def check_username_availability(
    username: str,
    db: Database = Depends(get_database),
    current_user: Optional[User] = Depends(get_optional_user)
):
    """Check if username is available"""
    
    try:
        existing_user = await db.get_user_by_username(username)
        
        # If user exists and it's not the current user, username is taken
        is_available = not existing_user or (current_user and existing_user.id == current_user.id)
        
        return {
            "username": username,
            "available": is_available,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Username check failed"
        )


# Background task functions

async def send_verification_email(email: str, user_id: str):
    """Background task to send verification email"""
    try:
        # Generate verification token
        token = generate_email_verification_token(user_id)
        
        # Send email (mock implementation)
        print(f"Sending verification email to {email} with token: {token}")
        
        # In production, integrate with email service like SendGrid, AWS SES, etc.
        
    except Exception as e:
        print(f"Failed to send verification email: {e}")


async def send_password_reset_email(email: str, user_id: str):
    """Background task to send password reset email"""
    try:
        # Generate reset token
        token = generate_password_reset_token(user_id)
        
        # Send email (mock implementation)
        print(f"Sending password reset email to {email} with token: {token}")
        
        # In production, integrate with email service
        
    except Exception as e:
        print(f"Failed to send password reset email: {e}")


# Utility functions

def generate_email_verification_token(user_id: str) -> str:
    """Generate email verification token"""
    # Generate JWT token with expiration
    claims = {
        'sub': user_id,
        'type': 'email_verification',
        'exp': datetime.utcnow() + timedelta(hours=24)
    }
    return security_manager.create_access_token(user_id, claims)


def generate_password_reset_token(user_id: str) -> str:
    """Generate password reset token"""
    # Generate JWT token with shorter expiration
    claims = {
        'sub': user_id,
        'type': 'password_reset',
        'exp': datetime.utcnow() + timedelta(hours=2)
    }
    return security_manager.create_access_token(user_id, claims)


async def verify_email_token(token: str) -> Optional[str]:
    """Verify email verification token"""
    try:
        payload = security_manager.verify_token(token)
        if payload and payload.get('type') == 'email_verification':
            return payload.get('sub')
        return None
    except Exception:
        return None


async def verify_password_reset_token(token: str) -> Optional[str]:
    """Verify password reset token"""
    try:
        payload = security_manager.verify_token(token)
        if payload and payload.get('type') == 'password_reset':
            return payload.get('sub')
        return None
    except Exception:
        return None