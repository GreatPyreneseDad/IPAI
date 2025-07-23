"""
Authentication API Endpoints

This module provides authentication endpoints including login, 
registration, password reset, and token management.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr, Field
import secrets
import hashlib
from jose import jwt, JWTError

from ...core.config import get_settings
from ...core.database import Database
from ...models.user import User
from ...models.database_models import User as UserDB, UserRole
from ..dependencies import get_database
from ...utils.email import send_email
from ...safety.enhanced_coherence_tracker import EnhancedCoherenceTracker
from ...blockchain.personal_chain import PersonalBlockchain

router = APIRouter(prefix="/auth", tags=["authentication"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

settings = get_settings()

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    email: str
    username: str
    full_name: Optional[str]
    is_active: bool
    is_verified: bool
    role: str
    created_at: datetime
    current_coherence_score: float
    coherence_level: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(..., min_length=8)

class PasswordChange(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)

# Utility functions
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return hash_password(plain_password) == hashed_password

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict):
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def create_tokens(user_id: str) -> Token:
    """Create access and refresh tokens"""
    access_token = create_access_token(data={"sub": user_id})
    refresh_token = create_refresh_token(data={"sub": user_id})
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Database = Depends(get_database)
) -> UserDB:
    """Get current user from JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "access":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user = await db.get_user(user_id)
    if user is None:
        raise credentials_exception
        
    return user

# Endpoints
@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Register a new user"""
    
    # Check if user exists
    existing_user = await db.get_user_by_email(user_data.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    existing_user = await db.get_user_by_username(user_data.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already taken"
        )
    
    # Create user
    hashed_password = hash_password(user_data.password)
    
    user = UserDB(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=UserRole.USER,
        is_active=True,
        is_verified=False,
        current_coherence_score=1.0,
        coherence_level="moderate",
        sage_balance=0.0,
        preferences={},
        notification_settings={
            "email": True,
            "push": True,
            "coherence_alerts": True
        }
    )
    
    created_user = await db.create_user(user)
    
    # Initialize user's coherence tracker and blockchain
    coherence_tracker = EnhancedCoherenceTracker()
    personal_chain = PersonalBlockchain(created_user.id, f"ipai_{created_user.id}")
    
    # Send verification email
    verification_token = secrets.token_urlsafe(32)
    await db.save_verification_token(created_user.id, verification_token)
    
    background_tasks.add_task(
        send_email,
        user_data.email,
        "Verify your IPAI account",
        f"Welcome to IPAI! Please verify your email: {settings.FRONTEND_URL}/verify-email?token={verification_token}"
    )
    
    return UserResponse(
        id=created_user.id,
        email=created_user.email,
        username=created_user.username,
        full_name=created_user.full_name,
        is_active=created_user.is_active,
        is_verified=created_user.is_verified,
        role=created_user.role.value,
        created_at=created_user.created_at,
        current_coherence_score=created_user.current_coherence_score,
        coherence_level=created_user.coherence_level.value
    )

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Database = Depends(get_database)
):
    """Login with username/email and password"""
    
    # Try to find user by username or email
    user = await db.get_user_by_username(form_data.username)
    if not user:
        user = await db.get_user_by_email(form_data.username)
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    # Update last login
    await db.update_last_login(user.id)
    
    # Create tokens
    return create_tokens(user.id)

@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str,
    db: Database = Depends(get_database)
):
    """Refresh access token using refresh token"""
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if user_id is None or token_type != "refresh":
            raise credentials_exception
            
    except JWTError:
        raise credentials_exception
    
    user = await db.get_user(user_id)
    if not user or not user.is_active:
        raise credentials_exception
    
    return create_tokens(user_id)

@router.post("/logout")
async def logout(
    current_user: UserDB = Depends(get_current_user),
    db: Database = Depends(get_database)
):
    """Logout current user"""
    
    # In a production app, you might want to blacklist the token
    # For now, we'll just return success
    return {"message": "Successfully logged out"}

@router.post("/reset-password")
async def reset_password(
    request: PasswordResetRequest,
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Request password reset"""
    
    user = await db.get_user_by_email(request.email)
    
    # Always return success to prevent email enumeration
    if user:
        reset_token = secrets.token_urlsafe(32)
        await db.save_password_reset_token(user.id, reset_token)
        
        background_tasks.add_task(
            send_email,
            request.email,
            "Reset your IPAI password",
            f"Reset your password here: {settings.FRONTEND_URL}/reset-password?token={reset_token}"
        )
    
    return {"message": "If the email exists, a reset link has been sent"}

@router.post("/confirm-reset")
async def confirm_password_reset(
    request: PasswordResetConfirm,
    db: Database = Depends(get_database)
):
    """Confirm password reset with token"""
    
    user_id = await db.verify_password_reset_token(request.token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Update password
    hashed_password = hash_password(request.new_password)
    await db.update_user_password(user_id, hashed_password)
    
    # Invalidate token
    await db.invalidate_password_reset_token(request.token)
    
    return {"message": "Password successfully reset"}

@router.post("/change-password")
async def change_password(
    request: PasswordChange,
    current_user: UserDB = Depends(get_current_user),
    db: Database = Depends(get_database)
):
    """Change password for authenticated user"""
    
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect"
        )
    
    hashed_password = hash_password(request.new_password)
    await db.update_user_password(current_user.id, hashed_password)
    
    return {"message": "Password successfully changed"}

@router.post("/verify-email")
async def verify_email(
    token: str,
    db: Database = Depends(get_database)
):
    """Verify email with token"""
    
    user_id = await db.verify_email_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    await db.mark_user_verified(user_id)
    await db.invalidate_email_token(token)
    
    return {"message": "Email successfully verified"}

@router.post("/resend-verification")
async def resend_verification(
    current_user: UserDB = Depends(get_current_user),
    background_tasks: BackgroundTasks,
    db: Database = Depends(get_database)
):
    """Resend email verification"""
    
    if current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )
    
    verification_token = secrets.token_urlsafe(32)
    await db.save_verification_token(current_user.id, verification_token)
    
    background_tasks.add_task(
        send_email,
        current_user.email,
        "Verify your IPAI account",
        f"Please verify your email: {settings.FRONTEND_URL}/verify-email?token={verification_token}"
    )
    
    return {"message": "Verification email sent"}