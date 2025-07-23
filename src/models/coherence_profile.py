"""
GCT Coherence Profile Models

This module defines the core data structures for Grounded Coherence Theory
implementation in the IPAI system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
from enum import Enum
import json


class CoherenceLevel(Enum):
    """GCT Coherence level classifications"""
    HIGH = "high"        # > 0.7
    MEDIUM = "medium"    # 0.4 - 0.7
    LOW = "low"          # < 0.4
    CRITICAL = "critical" # < 0.2


@dataclass(frozen=True)
class GCTComponents:
    """Immutable GCT coherence components"""
    psi: float  # Internal consistency [0, 1]
    rho: float  # Accumulated wisdom [0, 1]
    q: float    # Moral activation energy [0, 1]
    f: float    # Social belonging [0, 1]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate component ranges"""
        for name, value in [
            ('psi', self.psi), ('rho', self.rho), 
            ('q', self.q), ('f', self.f)
        ]:
            if not 0 <= value <= 1:
                raise ValueError(f"{name}={value} out of range [0, 1]")
    
    @property
    def soul_echo(self) -> float:
        """Calculate soul echo metric"""
        return self.psi * self.rho * self.q * self.f
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'psi': self.psi,
            'rho': self.rho,
            'q': self.q,
            'f': self.f,
            'soul_echo': self.soul_echo,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class IndividualParameters:
    """Individual GCT optimization parameters"""
    k_m: float  # Activation threshold [0.1, 0.5]
    k_i: float  # Sustainability threshold [0.5, 2.0]
    user_id: str
    calibrated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        """Validate parameter ranges"""
        if not 0.1 <= self.k_m <= 0.5:
            raise ValueError(f"k_m={self.k_m} out of range [0.1, 0.5]")
        if not 0.5 <= self.k_i <= 2.0:
            raise ValueError(f"k_i={self.k_i} out of range [0.5, 2.0]")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'k_m': self.k_m,
            'k_i': self.k_i,
            'user_id': self.user_id,
            'calibrated_at': self.calibrated_at.isoformat()
        }


@dataclass
class CoherenceProfile:
    """Complete user coherence profile"""
    user_id: str
    components: GCTComponents
    parameters: IndividualParameters
    coherence_score: float
    level: CoherenceLevel
    derivatives: Optional[Dict[str, float]] = None
    network_position: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate coherence score"""
        if not 0 <= self.coherence_score <= 4.0:
            raise ValueError(f"coherence_score={self.coherence_score} out of theoretical range [0, 4]")
    
    def to_blockchain_format(self) -> Dict:
        """Convert to blockchain storage format"""
        return {
            'userId': self.user_id,
            'coherenceScore': int(self.coherence_score * 10000),  # Store as basis points
            'components': {
                'psi': int(self.components.psi * 10000),
                'rho': int(self.components.rho * 10000),
                'q': int(self.components.q * 10000),
                'f': int(self.components.f * 10000)
            },
            'timestamp': int(self.components.timestamp.timestamp()),
            'level': self.level.value
        }
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'user_id': self.user_id,
            'components': self.components.to_dict(),
            'parameters': self.parameters.to_dict(),
            'coherence_score': self.coherence_score,
            'level': self.level.value,
            'derivatives': self.derivatives,
            'network_position': self.network_position
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CoherenceProfile':
        """Create from dictionary"""
        components = GCTComponents(
            psi=data['components']['psi'],
            rho=data['components']['rho'],
            q=data['components']['q'],
            f=data['components']['f'],
            timestamp=datetime.fromisoformat(data['components']['timestamp'])
        )
        
        parameters = IndividualParameters(
            k_m=data['parameters']['k_m'],
            k_i=data['parameters']['k_i'],
            user_id=data['parameters']['user_id'],
            calibrated_at=datetime.fromisoformat(data['parameters']['calibrated_at'])
        )
        
        return cls(
            user_id=data['user_id'],
            components=components,
            parameters=parameters,
            coherence_score=data['coherence_score'],
            level=CoherenceLevel(data['level']),
            derivatives=data.get('derivatives'),
            network_position=data.get('network_position')
        )


@dataclass
class CoherenceTrajectory:
    """Time series of coherence profiles"""
    user_id: str
    profiles: List[CoherenceProfile]
    start_time: datetime
    end_time: datetime
    
    @property
    def duration_days(self) -> float:
        """Calculate duration in days"""
        return (self.end_time - self.start_time).total_seconds() / 86400
    
    @property
    def average_coherence(self) -> float:
        """Calculate average coherence over trajectory"""
        if not self.profiles:
            return 0.0
        return sum(p.coherence_score for p in self.profiles) / len(self.profiles)
    
    @property
    def coherence_trend(self) -> float:
        """Calculate coherence trend (slope)"""
        if len(self.profiles) < 2:
            return 0.0
        
        # Simple linear regression
        n = len(self.profiles)
        x_mean = (n - 1) / 2
        y_mean = self.average_coherence
        
        numerator = sum((i - x_mean) * (p.coherence_score - y_mean) 
                       for i, p in enumerate(self.profiles))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'user_id': self.user_id,
            'profiles': [p.to_dict() for p in self.profiles],
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_days': self.duration_days,
            'average_coherence': self.average_coherence,
            'coherence_trend': self.coherence_trend
        }


# Utility functions for coherence level determination
def get_coherence_level(score: float) -> CoherenceLevel:
    """Determine coherence level from score"""
    if score >= 0.7:
        return CoherenceLevel.HIGH
    elif score >= 0.4:
        return CoherenceLevel.MEDIUM
    elif score >= 0.2:
        return CoherenceLevel.LOW
    else:
        return CoherenceLevel.CRITICAL


def validate_gct_components(psi: float, rho: float, q: float, f: float) -> bool:
    """Validate GCT component values"""
    try:
        GCTComponents(psi=psi, rho=rho, q=q, f=f)
        return True
    except ValueError:
        return False


def validate_individual_parameters(k_m: float, k_i: float) -> bool:
    """Validate individual parameter values"""
    return 0.1 <= k_m <= 0.5 and 0.5 <= k_i <= 2.0