"""
Assessment API Endpoints

This module provides REST endpoints for assessment management,
parameter calibration, and assessment completion.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import uuid

from ...models.user import User
from ...models.assessment import (
    Assessment, AssessmentType, AssessmentStatus, AssessmentQuestion,
    AssessmentResponse, create_initial_calibration_assessment
)
from ...models.coherence_profile import IndividualParameters
from ...core.database import Database
from ..dependencies import (
    get_current_active_user, get_database, assessment_rate_limit,
    get_pagination_params, PaginationParams
)

router = APIRouter(prefix="/assessment")

# Pydantic models

class AssessmentQuestionResponse(BaseModel):
    """Response model for assessment questions"""
    id: str
    question: str
    question_type: str
    options: Optional[List[str]] = None
    scale_min: Optional[int] = None
    scale_max: Optional[int] = None
    required: bool
    category: Optional[str] = None


class AssessmentResponseRequest(BaseModel):
    """Request model for assessment responses"""
    question_id: str
    response: Any = Field(..., description="Response value (type depends on question)")
    response_time_ms: Optional[int] = None
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Confidence level 0-1")


class AssessmentDetailsResponse(BaseModel):
    """Response model for assessment details"""
    id: str
    assessment_type: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    questions: List[AssessmentQuestionResponse]
    completion_percentage: float
    calculated_k_m: Optional[float] = None
    calculated_k_i: Optional[float] = None


class AssessmentSummaryResponse(BaseModel):
    """Response model for assessment summary"""
    id: str
    assessment_type: str
    status: str
    created_at: datetime
    completion_percentage: float
    expires_at: Optional[datetime] = None


class AssessmentResultResponse(BaseModel):
    """Response model for completed assessment results"""
    id: str
    assessment_type: str
    completion_percentage: float
    calculated_k_m: Optional[float] = None
    calculated_k_i: Optional[float] = None
    confidence_score: Optional[float] = None
    duration_minutes: Optional[float] = None
    parameter_explanation: Dict[str, str]


class ParameterCalibrationRequest(BaseModel):
    """Request model for manual parameter setting"""
    assessment_id: str
    override_k_m: Optional[float] = Field(None, ge=0.1, le=0.5)
    override_k_i: Optional[float] = Field(None, ge=0.5, le=2.0)


# Endpoints

@router.post("/create", response_model=AssessmentDetailsResponse)
async def create_assessment(
    assessment_type: AssessmentType = AssessmentType.INITIAL_CALIBRATION,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database),
    _: bool = Depends(assessment_rate_limit)
):
    """Create a new assessment"""
    
    try:
        # Check if user has pending assessments
        pending_assessment = await db.get_pending_assessment(current_user.id)
        if pending_assessment:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You have a pending assessment. Please complete it before creating a new one."
            )
        
        # Create assessment based on type
        if assessment_type == AssessmentType.INITIAL_CALIBRATION:
            assessment = create_initial_calibration_assessment(current_user.id)
        else:
            # For other types, create basic assessment structure
            assessment = Assessment(
                id=str(uuid.uuid4()),
                user_id=current_user.id,
                assessment_type=assessment_type,
                status=AssessmentStatus.PENDING,
                expires_at=datetime.utcnow() + timedelta(days=7)
            )
        
        # Save to database
        await db.save_assessment(assessment)
        
        return format_assessment_details(assessment)
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create assessment"
        )


@router.get("/current", response_model=Optional[AssessmentDetailsResponse])
async def get_current_assessment(
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Get user's current pending assessment"""
    
    try:
        assessment = await db.get_pending_assessment(current_user.id)
        if not assessment:
            return None
        
        return format_assessment_details(assessment)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve current assessment"
        )


@router.get("/{assessment_id}", response_model=AssessmentDetailsResponse)
async def get_assessment(
    assessment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Get assessment by ID"""
    
    try:
        assessment = await db.get_assessment_by_id(assessment_id)
        if not assessment:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        if assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return format_assessment_details(assessment)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assessment"
        )


@router.post("/{assessment_id}/start")
async def start_assessment(
    assessment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Start an assessment"""
    
    try:
        assessment = await db.get_assessment_by_id(assessment_id)
        if not assessment or assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        if assessment.status != AssessmentStatus.PENDING:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment has already been started or completed"
            )
        
        if assessment.expires_at and datetime.utcnow() > assessment.expires_at:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment has expired"
            )
        
        # Start assessment
        assessment.start_assessment()
        await db.update_assessment(assessment)
        
        return {
            "status": "started",
            "assessment_id": assessment_id,
            "started_at": assessment.started_at,
            "message": "Assessment started successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start assessment"
        )


@router.post("/{assessment_id}/respond")
async def submit_response(
    assessment_id: str,
    response_data: AssessmentResponseRequest,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Submit response to assessment question"""
    
    try:
        assessment = await db.get_assessment_by_id(assessment_id)
        if not assessment or assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        if assessment.status != AssessmentStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment is not in progress"
            )
        
        # Validate question exists
        question = next((q for q in assessment.questions if q.id == response_data.question_id), None)
        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question not found in assessment"
            )
        
        # Validate response based on question type
        validate_response(question, response_data.response)
        
        # Create response object
        response = AssessmentResponse(
            question_id=response_data.question_id,
            response=response_data.response,
            response_time_ms=response_data.response_time_ms,
            confidence=response_data.confidence
        )
        
        # Add response to assessment
        assessment.add_response(response)
        
        # Save to database
        await db.update_assessment(assessment)
        
        return {
            "status": "recorded",
            "question_id": response_data.question_id,
            "completion_percentage": assessment.completion_percentage,
            "is_complete": assessment.is_complete
        }
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit response"
        )


@router.post("/{assessment_id}/complete", response_model=AssessmentResultResponse)
async def complete_assessment(
    assessment_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Complete assessment and calculate parameters"""
    
    try:
        assessment = await db.get_assessment_by_id(assessment_id)
        if not assessment or assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        if assessment.status != AssessmentStatus.IN_PROGRESS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment is not in progress"
            )
        
        if not assessment.is_complete:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Assessment is only {assessment.completion_percentage:.1f}% complete"
            )
        
        # Calculate parameters
        k_m, k_i = assessment.calculate_parameters()
        
        # Complete assessment
        assessment.complete_assessment()
        
        # Save to database
        await db.update_assessment(assessment)
        
        # Create individual parameters
        parameters = IndividualParameters(
            k_m=k_m,
            k_i=k_i,
            user_id=current_user.id
        )
        
        # Save parameters in background
        background_tasks.add_task(save_parameters_from_assessment, db, parameters, assessment)
        
        # Generate explanation
        explanation = generate_parameter_explanation(k_m, k_i, assessment)
        
        return AssessmentResultResponse(
            id=assessment.id,
            assessment_type=assessment.assessment_type.value,
            completion_percentage=assessment.completion_percentage,
            calculated_k_m=k_m,
            calculated_k_i=k_i,
            confidence_score=assessment.confidence_score,
            duration_minutes=assessment.duration_minutes,
            parameter_explanation=explanation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete assessment"
        )


@router.get("/history", response_model=List[AssessmentSummaryResponse])
async def get_assessment_history(
    pagination: PaginationParams = Depends(get_pagination_params),
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Get user's assessment history"""
    
    try:
        assessments = await db.get_user_assessments(
            current_user.id,
            skip=pagination.skip,
            limit=pagination.limit
        )
        
        return [
            AssessmentSummaryResponse(
                id=a.id,
                assessment_type=a.assessment_type.value,
                status=a.status.value,
                created_at=a.created_at,
                completion_percentage=a.completion_percentage,
                expires_at=a.expires_at
            )
            for a in assessments
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assessment history"
        )


@router.post("/{assessment_id}/calibrate-parameters")
async def manual_parameter_calibration(
    assessment_id: str,
    calibration: ParameterCalibrationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Manually calibrate parameters from completed assessment"""
    
    try:
        assessment = await db.get_assessment_by_id(assessment_id)
        if not assessment or assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        if assessment.status != AssessmentStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Assessment must be completed first"
            )
        
        # Use overrides if provided, otherwise use calculated values
        k_m = calibration.override_k_m if calibration.override_k_m is not None else assessment.calculated_k_m
        k_i = calibration.override_k_i if calibration.override_k_i is not None else assessment.calculated_k_i
        
        if k_m is None or k_i is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Parameters not available. Complete assessment calculation first."
            )
        
        # Create parameters
        parameters = IndividualParameters(
            k_m=k_m,
            k_i=k_i,
            user_id=current_user.id
        )
        
        # Save parameters
        background_tasks.add_task(save_parameters_from_assessment, db, parameters, assessment)
        
        return {
            "status": "calibrated",
            "k_m": k_m,
            "k_i": k_i,
            "assessment_id": assessment_id,
            "manual_override": calibration.override_k_m is not None or calibration.override_k_i is not None,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calibrate parameters"
        )


@router.delete("/{assessment_id}")
async def cancel_assessment(
    assessment_id: str,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Cancel pending or in-progress assessment"""
    
    try:
        assessment = await db.get_assessment_by_id(assessment_id)
        if not assessment or assessment.user_id != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Assessment not found"
            )
        
        if assessment.status == AssessmentStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot cancel completed assessment"
            )
        
        # Cancel assessment
        assessment.status = AssessmentStatus.CANCELLED
        await db.update_assessment(assessment)
        
        return {
            "status": "cancelled",
            "assessment_id": assessment_id,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel assessment"
        )


# Background task functions

async def save_parameters_from_assessment(
    db: Database, 
    parameters: IndividualParameters, 
    assessment: Assessment
):
    """Background task to save parameters from assessment"""
    try:
        await db.save_user_parameters(parameters)
        print(f"Parameters saved for user {parameters.user_id}: k_m={parameters.k_m}, k_i={parameters.k_i}")
    except Exception as e:
        print(f"Failed to save parameters: {e}")


# Utility functions

def format_assessment_details(assessment: Assessment) -> AssessmentDetailsResponse:
    """Format assessment for API response"""
    
    questions = [
        AssessmentQuestionResponse(
            id=q.id,
            question=q.question,
            question_type=q.question_type,
            options=q.options,
            scale_min=q.scale_min,
            scale_max=q.scale_max,
            required=q.required,
            category=q.category
        )
        for q in assessment.questions
    ]
    
    return AssessmentDetailsResponse(
        id=assessment.id,
        assessment_type=assessment.assessment_type.value,
        status=assessment.status.value,
        created_at=assessment.created_at,
        started_at=assessment.started_at,
        completed_at=assessment.completed_at,
        expires_at=assessment.expires_at,
        questions=questions,
        completion_percentage=assessment.completion_percentage,
        calculated_k_m=assessment.calculated_k_m,
        calculated_k_i=assessment.calculated_k_i
    )


def validate_response(question: AssessmentQuestion, response: Any):
    """Validate response based on question type"""
    
    if question.question_type == "likert":
        if not isinstance(response, (int, float)):
            raise ValueError("Likert response must be a number")
        if question.scale_min is not None and response < question.scale_min:
            raise ValueError(f"Response below minimum scale value {question.scale_min}")
        if question.scale_max is not None and response > question.scale_max:
            raise ValueError(f"Response above maximum scale value {question.scale_max}")
    
    elif question.question_type == "multiple_choice":
        if not question.options:
            raise ValueError("Multiple choice question has no options")
        if response not in question.options:
            raise ValueError("Response not in available options")
    
    elif question.question_type == "text":
        if not isinstance(response, str):
            raise ValueError("Text response must be a string")
        if len(response.strip()) == 0:
            raise ValueError("Text response cannot be empty")
    
    elif question.question_type == "slider":
        if not isinstance(response, (int, float)):
            raise ValueError("Slider response must be a number")
        if question.scale_min is not None and response < question.scale_min:
            raise ValueError(f"Response below minimum value {question.scale_min}")
        if question.scale_max is not None and response > question.scale_max:
            raise ValueError(f"Response above maximum value {question.scale_max}")


def generate_parameter_explanation(k_m: float, k_i: float, assessment: Assessment) -> Dict[str, str]:
    """Generate explanation for calculated parameters"""
    
    explanation = {}
    
    # K_m explanation
    if k_m < 0.2:
        explanation["k_m"] = "Low activation threshold - you tend to engage quickly with moral situations and take action readily."
    elif k_m > 0.4:
        explanation["k_m"] = "High activation threshold - you prefer to carefully consider situations before taking moral action."
    else:
        explanation["k_m"] = "Moderate activation threshold - you balance reflection with timely moral action."
    
    # K_i explanation  
    if k_i < 0.8:
        explanation["k_i"] = "Lower sustainability - you may benefit from shorter, more frequent coherence-building activities."
    elif k_i > 1.5:
        explanation["k_i"] = "Higher sustainability - you can maintain coherence through longer, deeper practices."
    else:
        explanation["k_i"] = "Moderate sustainability - standard coherence practices should work well for you."
    
    # Overall explanation
    explanation["overall"] = f"Your parameters suggest a {get_profile_type(k_m, k_i)} approach to coherence development."
    
    return explanation


def get_profile_type(k_m: float, k_i: float) -> str:
    """Get profile type description"""
    
    if k_m < 0.25 and k_i > 1.2:
        return "quick-engaging, sustainable"
    elif k_m > 0.35 and k_i > 1.2:
        return "thoughtful, deep-practice"
    elif k_m < 0.25 and k_i < 0.8:
        return "action-oriented, flexible"
    elif k_m > 0.35 and k_i < 0.8:
        return "careful, adaptive"
    else:
        return "balanced"