"""
Database Models for IPAI

Complete SQLAlchemy models for all database entities.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, Table, Enum as SQLEnum, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional, List, Dict, Any
import enum
import uuid

Base = declarative_base()


# Enums
class UserRole(enum.Enum):
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    MODERATOR = "moderator"


class CoherenceLevel(enum.Enum):
    CRITICAL = "critical"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    OPTIMAL = "optimal"


class InteractionType(enum.Enum):
    CHAT = "chat"
    ASSESSMENT = "assessment"
    MEDITATION = "meditation"
    JOURNAL = "journal"
    ANALYSIS = "analysis"


class InferenceStatus(enum.Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONSENSUS = "consensus"


# Association Tables
user_achievements = Table(
    'user_achievements',
    Base.metadata,
    Column('user_id', String, ForeignKey('users.id')),
    Column('achievement_id', Integer, ForeignKey('achievements.id')),
    Column('earned_at', DateTime, default=func.now())
)


# Main Models
class User(Base):
    __tablename__ = 'users'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Profile
    full_name = Column(String(100))
    bio = Column(Text)
    avatar_url = Column(String(500))
    
    # Status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    
    # Coherence
    current_coherence_score = Column(Float, default=1.0)
    coherence_level = Column(SQLEnum(CoherenceLevel), default=CoherenceLevel.MODERATE)
    
    # Settings
    preferences = Column(JSON, default={})
    notification_settings = Column(JSON, default={
        "email": True,
        "push": True,
        "coherence_alerts": True
    })
    
    # Blockchain
    wallet_address = Column(String(42), unique=True)
    ipai_identity_token_id = Column(Integer)
    sage_balance = Column(Float, default=0.0)
    
    # Relationships
    coherence_profiles = relationship("CoherenceProfile", back_populates="user", cascade="all, delete-orphan")
    interactions = relationship("UserInteraction", back_populates="user", cascade="all, delete-orphan")
    assessments = relationship("Assessment", back_populates="user", cascade="all, delete-orphan")
    achievements = relationship("Achievement", secondary=user_achievements, back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("Notification", back_populates="user", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_email_active', 'email', 'is_active'),
        Index('idx_user_coherence', 'current_coherence_score', 'coherence_level'),
    )


class CoherenceProfile(Base):
    __tablename__ = 'coherence_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # GCT Components
    psi = Column(Float, nullable=False)  # Internal consistency
    rho = Column(Float, nullable=False)  # Accumulated wisdom
    q = Column(Float, nullable=False)    # Moral activation energy
    f = Column(Float, nullable=False)    # Social belonging
    
    # Calculated values
    coherence_score = Column(Float, nullable=False)
    soul_echo = Column(Float, nullable=False)
    level = Column(SQLEnum(CoherenceLevel), nullable=False)
    
    # Individual parameters
    k_m = Column(Float, default=0.3)  # Activation threshold
    k_i = Column(Float, default=1.0)  # Sustainability threshold
    
    # Metadata
    calculated_at = Column(DateTime, default=func.now())
    calculation_method = Column(String(50), default="enhanced_gct")
    confidence_score = Column(Float, default=0.85)
    
    # Analysis
    risk_factors = Column(JSON, default={})
    growth_potential = Column(Float)
    stability_index = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="coherence_profiles")
    
    # Indexes
    __table_args__ = (
        Index('idx_coherence_user_time', 'user_id', 'calculated_at'),
        Index('idx_coherence_score', 'coherence_score', 'level'),
    )


class UserInteraction(Base):
    __tablename__ = 'user_interactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Interaction details
    interaction_type = Column(SQLEnum(InteractionType), nullable=False)
    input_text = Column(Text)
    output_text = Column(Text)
    
    # Coherence impact
    coherence_before = Column(Float)
    coherence_after = Column(Float)
    coherence_delta = Column(Float)
    
    # Safety metrics
    safety_score = Column(Float, default=1.0)
    howlround_risk = Column(Float, default=0.0)
    pressure_score = Column(Float, default=0.0)
    intervention_triggered = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    duration_seconds = Column(Integer)
    llm_provider = Column(String(50))
    llm_model = Column(String(100))
    
    # Blockchain
    block_hash = Column(String(66))
    transaction_hash = Column(String(66))
    
    # Relationships
    user = relationship("User", back_populates="interactions")
    inferences = relationship("Inference", back_populates="interaction", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_interaction_user_time', 'user_id', 'created_at'),
        Index('idx_interaction_type', 'interaction_type'),
    )


class Assessment(Base):
    __tablename__ = 'assessments'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Assessment info
    assessment_type = Column(String(50), nullable=False)
    version = Column(String(10), default="1.0")
    
    # Questions and responses
    questions = Column(JSON, nullable=False)
    responses = Column(JSON, nullable=False)
    
    # Scores
    raw_scores = Column(JSON, nullable=False)
    normalized_scores = Column(JSON, nullable=False)
    
    # Results
    psi_score = Column(Float)
    rho_score = Column(Float)
    q_score = Column(Float)
    f_score = Column(Float)
    
    # Calibration
    suggested_k_m = Column(Float)
    suggested_k_i = Column(Float)
    
    # Metadata
    completed_at = Column(DateTime, default=func.now())
    time_taken_seconds = Column(Integer)
    completion_rate = Column(Float)
    
    # Relationships
    user = relationship("User", back_populates="assessments")
    
    # Indexes
    __table_args__ = (
        Index('idx_assessment_user_time', 'user_id', 'completed_at'),
    )


class Inference(Base):
    __tablename__ = 'inferences'
    
    id = Column(Integer, primary_key=True)
    interaction_id = Column(Integer, ForeignKey('user_interactions.id'), nullable=False)
    
    # Inference details
    inference_type = Column(String(50), nullable=False)
    content = Column(JSON, nullable=False)
    confidence_score = Column(Float, nullable=False)
    
    # Coherence context
    coherence_context = Column(JSON, nullable=False)
    
    # Verification
    verification_hash = Column(String(66), unique=True, nullable=False)
    status = Column(SQLEnum(InferenceStatus), default=InferenceStatus.PENDING)
    verified_at = Column(DateTime)
    verified_by = Column(String(100))  # Could be another IPAI or consensus
    
    # Blockchain
    public_ledger_hash = Column(String(66))
    sage_reward = Column(Float, default=0.0)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    # Relationships
    interaction = relationship("UserInteraction", back_populates="inferences")
    
    # Indexes
    __table_args__ = (
        Index('idx_inference_status', 'status'),
        Index('idx_inference_hash', 'verification_hash'),
    )


class Achievement(Base):
    __tablename__ = 'achievements'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    icon_url = Column(String(500))
    
    # Requirements
    requirement_type = Column(String(50))  # coherence_level, interaction_count, etc.
    requirement_value = Column(JSON)
    
    # Rewards
    sage_reward = Column(Float, default=0.0)
    badge_color = Column(String(7))  # Hex color
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    # Relationships
    users = relationship("User", secondary=user_achievements, back_populates="achievements")


class APIKey(Base):
    __tablename__ = 'api_keys'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    key_hash = Column(String(255), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    
    # Permissions
    scopes = Column(JSON, default=["read"])
    rate_limit = Column(Integer, default=1000)  # requests per hour
    
    # Status
    is_active = Column(Boolean, default=True)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_key_hash', 'key_hash'),
        Index('idx_api_key_user', 'user_id', 'is_active'),
    )


class Notification(Base):
    __tablename__ = 'notifications'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String, ForeignKey('users.id'), nullable=False)
    
    # Notification details
    type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    data = Column(JSON, default={})
    
    # Status
    is_read = Column(Boolean, default=False)
    read_at = Column(DateTime)
    
    # Delivery
    channels = Column(JSON, default=["in_app"])  # in_app, email, push
    delivered_channels = Column(JSON, default=[])
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    expires_at = Column(DateTime)
    
    # Relationships
    user = relationship("User", back_populates="notifications")
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_user_unread', 'user_id', 'is_read'),
        Index('idx_notification_created', 'created_at'),
    )


class SystemMetrics(Base):
    __tablename__ = 'system_metrics'
    
    id = Column(Integer, primary_key=True)
    
    # Metrics
    metric_type = Column(String(50), nullable=False)
    metric_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(20))
    
    # Context
    tags = Column(JSON, default={})
    
    # Timestamp
    recorded_at = Column(DateTime, default=func.now())
    
    # Indexes
    __table_args__ = (
        Index('idx_metric_type_time', 'metric_type', 'recorded_at'),
        Index('idx_metric_name', 'metric_name'),
    )


class BlockchainContract(Base):
    __tablename__ = 'blockchain_contracts'
    
    id = Column(Integer, primary_key=True)
    
    # Contract info
    name = Column(String(50), nullable=False)
    network = Column(String(50), nullable=False)
    address = Column(String(42), nullable=False)
    
    # ABI
    abi = Column(JSON, nullable=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    deployed_at = Column(DateTime)
    deployed_by = Column(String(42))
    
    # Metadata
    created_at = Column(DateTime, default=func.now())
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('name', 'network', name='_name_network_uc'),
        Index('idx_contract_network', 'network', 'is_active'),
    )