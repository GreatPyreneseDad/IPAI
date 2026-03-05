"""
Assessment Model

This module defines the assessment data structure for parameter calibration
and coherence evaluation in the IPAI system.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime
from enum import Enum
import json


class AssessmentType(Enum):
    """Types of assessments"""
    INITIAL_CALIBRATION = "initial_calibration"
    PERIODIC_RECALIBRATION = "periodic_recalibration"
    CRISIS_INTERVENTION = "crisis_intervention"
    RESEARCH_EVALUATION = "research_evaluation"


class AssessmentStatus(Enum):
    """Assessment completion status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class AssessmentQuestion:
    """Individual assessment question"""
    id: str
    question: str
    question_type: str  # "likert", "multiple_choice", "text", "slider"
    options: Optional[List[str]] = None
    scale_min: Optional[int] = None
    scale_max: Optional[int] = None
    required: bool = True
    category: Optional[str] = None  # "psi", "rho", "q", "f", "general"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'question': self.question,
            'question_type': self.question_type,
            'options': self.options,
            'scale_min': self.scale_min,
            'scale_max': self.scale_max,
            'required': self.required,
            'category': self.category
        }


@dataclass
class AssessmentResponse:
    """Individual assessment response"""
    question_id: str
    response: Any  # Can be str, int, float, list depending on question type
    response_time_ms: Optional[int] = None
    confidence: Optional[float] = None  # [0, 1] self-reported confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'question_id': self.question_id,
            'response': self.response,
            'response_time_ms': self.response_time_ms,
            'confidence': self.confidence
        }


@dataclass
class Assessment:
    """Complete assessment session"""
    id: str
    user_id: str
    assessment_type: AssessmentType
    status: AssessmentStatus
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    
    # Assessment content
    questions: List[AssessmentQuestion] = field(default_factory=list)
    responses: List[AssessmentResponse] = field(default_factory=list)
    
    # Calculated results
    calculated_k_m: Optional[float] = None
    calculated_k_i: Optional[float] = None
    confidence_score: Optional[float] = None
    
    # Metadata
    version: str = "1.0"
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate assessment data"""
        if not self.user_id:
            raise ValueError("User ID is required")
        if not self.id:
            raise ValueError("Assessment ID is required")
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        if not self.questions:
            return 0.0
        required_questions = [q for q in self.questions if q.required]
        answered_required = len([r for r in self.responses 
                               if r.question_id in [q.id for q in required_questions]])
        return (answered_required / len(required_questions)) * 100 if required_questions else 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if assessment is complete"""
        return self.completion_percentage >= 100.0
    
    @property
    def duration_minutes(self) -> Optional[float]:
        """Calculate assessment duration in minutes"""
        if not self.started_at or not self.completed_at:
            return None
        return (self.completed_at - self.started_at).total_seconds() / 60
    
    def start_assessment(self):
        """Mark assessment as started"""
        self.started_at = datetime.utcnow()
        self.status = AssessmentStatus.IN_PROGRESS
    
    def complete_assessment(self):
        """Mark assessment as completed"""
        if not self.is_complete:
            raise ValueError("Assessment is not complete")
        self.completed_at = datetime.utcnow()
        self.status = AssessmentStatus.COMPLETED
    
    def add_response(self, response: AssessmentResponse):
        """Add a response to the assessment"""
        # Remove existing response for same question
        self.responses = [r for r in self.responses if r.question_id != response.question_id]
        # Add new response
        self.responses.append(response)
    
    def get_response(self, question_id: str) -> Optional[AssessmentResponse]:
        """Get response for a specific question"""
        for response in self.responses:
            if response.question_id == question_id:
                return response
        return None
    
    def get_responses_by_category(self, category: str) -> List[AssessmentResponse]:
        """Get all responses for a specific category"""
        category_question_ids = [q.id for q in self.questions if q.category == category]
        return [r for r in self.responses if r.question_id in category_question_ids]
    
    def calculate_parameters(self) -> tuple[float, float]:
        """Calculate K_m and K_i from assessment responses"""
        # This is a simplified calculation - in practice, this would use
        # sophisticated psychometric analysis
        
        # Get responses by category
        psi_responses = self.get_responses_by_category("psi")
        rho_responses = self.get_responses_by_category("rho")
        q_responses = self.get_responses_by_category("q")
        f_responses = self.get_responses_by_category("f")
        
        # Calculate average scores for each component
        def avg_score(responses: List[AssessmentResponse]) -> float:
            if not responses:
                return 0.5  # Default middle value
            scores = [float(r.response) for r in responses if isinstance(r.response, (int, float))]
            return sum(scores) / len(scores) if scores else 0.5
        
        psi_avg = avg_score(psi_responses)
        rho_avg = avg_score(rho_responses)
        q_avg = avg_score(q_responses)
        f_avg = avg_score(f_responses)
        
        # Calculate K_m (activation threshold) - lower for more active individuals
        k_m = 0.5 - (q_avg * 0.4)  # Range [0.1, 0.5]
        k_m = max(0.1, min(0.5, k_m))
        
        # Calculate K_i (sustainability threshold) - higher for more stable individuals
        k_i = 0.5 + (psi_avg * rho_avg * 1.5)  # Range [0.5, 2.0]
        k_i = max(0.5, min(2.0, k_i))
        
        self.calculated_k_m = k_m
        self.calculated_k_i = k_i
        
        return k_m, k_i
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'assessment_type': self.assessment_type.value,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'questions': [q.to_dict() for q in self.questions],
            'responses': [r.to_dict() for r in self.responses],
            'calculated_k_m': self.calculated_k_m,
            'calculated_k_i': self.calculated_k_i,
            'confidence_score': self.confidence_score,
            'version': self.version,
            'metadata': self.metadata,
            'completion_percentage': self.completion_percentage,
            'duration_minutes': self.duration_minutes
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Assessment':
        """Create from dictionary"""
        assessment = cls(
            id=data['id'],
            user_id=data['user_id'],
            assessment_type=AssessmentType(data['assessment_type']),
            status=AssessmentStatus(data['status']),
            created_at=datetime.fromisoformat(data['created_at']),
            version=data.get('version', '1.0'),
            metadata=data.get('metadata', {}),
            calculated_k_m=data.get('calculated_k_m'),
            calculated_k_i=data.get('calculated_k_i'),
            confidence_score=data.get('confidence_score')
        )
        
        # Set optional datetime fields
        if data.get('started_at'):
            assessment.started_at = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            assessment.completed_at = datetime.fromisoformat(data['completed_at'])
        if data.get('expires_at'):
            assessment.expires_at = datetime.fromisoformat(data['expires_at'])
        
        # Load questions
        for q_data in data.get('questions', []):
            question = AssessmentQuestion(**q_data)
            assessment.questions.append(question)
        
        # Load responses
        for r_data in data.get('responses', []):
            response = AssessmentResponse(**r_data)
            assessment.responses.append(response)
        
        return assessment


# Predefined assessment templates
def create_initial_calibration_assessment(user_id: str) -> Assessment:
    """Create initial calibration assessment template"""
    questions = [
        # Psi (Internal Consistency) questions
        AssessmentQuestion(
            id="psi_1",
            question="How consistent are your thoughts and actions on a daily basis?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="psi"
        ),
        AssessmentQuestion(
            id="psi_2",
            question="How often do you experience internal conflicts about your decisions?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="psi"
        ),
        
        # Rho (Accumulated Wisdom) questions
        AssessmentQuestion(
            id="rho_1",
            question="How well do you learn from your past experiences?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="rho"
        ),
        AssessmentQuestion(
            id="rho_2",
            question="How often do you reflect on your life experiences to gain insights?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="rho"
        ),
        
        # Q (Moral Activation) questions
        AssessmentQuestion(
            id="q_1",
            question="How important is it to you to act according to your moral principles?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="q"
        ),
        AssessmentQuestion(
            id="q_2",
            question="How often do you take action when you see something wrong?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="q"
        ),
        
        # F (Social Belonging) questions
        AssessmentQuestion(
            id="f_1",
            question="How connected do you feel to your community?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="f"
        ),
        AssessmentQuestion(
            id="f_2",
            question="How supported do you feel by the people around you?",
            question_type="likert",
            scale_min=1,
            scale_max=7,
            category="f"
        )
    ]
    
    import uuid
    assessment = Assessment(
        id=str(uuid.uuid4()),
        user_id=user_id,
        assessment_type=AssessmentType.INITIAL_CALIBRATION,
        status=AssessmentStatus.PENDING,
        questions=questions,
        expires_at=datetime.utcnow().replace(hour=23, minute=59, second=59) + 
                   datetime.timedelta(days=7)  # Expires in 7 days
    )
    
    return assessment