"""
User Model

This module defines the user data structure for the IPAI system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum
import uuid


class UserRole(Enum):
    """User role classifications"""
    STANDARD = "standard"
    PREMIUM = "premium"
    RESEARCHER = "researcher"
    ADMIN = "admin"


@dataclass
class User:
    """User profile and authentication data"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    email: str = ""
    username: str = ""
    password_hash: str = ""
    role: UserRole = UserRole.STANDARD
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    is_verified: bool = False
    
    # Profile information
    full_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    
    # Preferences
    preferences: Dict = field(default_factory=dict)
    
    # Coherence tracking
    coherence_profiles_count: int = 0
    last_coherence_update: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate user data"""
        if not self.email:
            raise ValueError("Email is required")
        if not self.username:
            raise ValueError("Username is required")
        if "@" not in self.email:
            raise ValueError("Invalid email format")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'email': self.email,
            'username': self.username,
            'role': self.role.value,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'full_name': self.full_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'preferences': self.preferences,
            'coherence_profiles_count': self.coherence_profiles_count,
            'last_coherence_update': self.last_coherence_update.isoformat() if self.last_coherence_update else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        """Create from dictionary"""
        user = cls(
            id=data['id'],
            email=data['email'],
            username=data['username'],
            password_hash=data.get('password_hash', ''),
            role=UserRole(data['role']),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            is_active=data['is_active'],
            is_verified=data['is_verified'],
            full_name=data.get('full_name'),
            bio=data.get('bio'),
            avatar_url=data.get('avatar_url'),
            preferences=data.get('preferences', {}),
            coherence_profiles_count=data.get('coherence_profiles_count', 0)
        )
        
        if data.get('last_login'):
            user.last_login = datetime.fromisoformat(data['last_login'])
        if data.get('last_coherence_update'):
            user.last_coherence_update = datetime.fromisoformat(data['last_coherence_update'])
        
        return user
    
    def update_last_login(self):
        """Update last login timestamp"""
        self.last_login = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def update_coherence_tracking(self):
        """Update coherence tracking information"""
        self.coherence_profiles_count += 1
        self.last_coherence_update = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def can_access_feature(self, feature: str) -> bool:
        """Check if user can access a specific feature"""
        feature_permissions = {
            'basic_coherence': [UserRole.STANDARD, UserRole.PREMIUM, UserRole.RESEARCHER, UserRole.ADMIN],
            'advanced_analytics': [UserRole.PREMIUM, UserRole.RESEARCHER, UserRole.ADMIN],
            'blockchain_integration': [UserRole.PREMIUM, UserRole.RESEARCHER, UserRole.ADMIN],
            'research_data': [UserRole.RESEARCHER, UserRole.ADMIN],
            'admin_panel': [UserRole.ADMIN],
            'llm_integration': [UserRole.PREMIUM, UserRole.RESEARCHER, UserRole.ADMIN]
        }
        
        return self.role in feature_permissions.get(feature, [])
    
    def to_public_dict(self) -> Dict:
        """Convert to public dictionary format (no sensitive data)"""
        return {
            'id': self.id,
            'username': self.username,
            'full_name': self.full_name,
            'bio': self.bio,
            'avatar_url': self.avatar_url,
            'created_at': self.created_at.isoformat(),
            'role': self.role.value,
            'coherence_profiles_count': self.coherence_profiles_count,
            'last_coherence_update': self.last_coherence_update.isoformat() if self.last_coherence_update else None
        }