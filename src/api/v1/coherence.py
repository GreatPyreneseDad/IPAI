"""
Coherence API Endpoints

This module provides REST endpoints for coherence calculations,
trajectory analysis, and parameter calibration.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel, Field, validator
import asyncio

from ...models.user import User
from ...models.coherence_profile import (
    CoherenceProfile, GCTComponents, IndividualParameters, 
    CoherenceLevel, get_coherence_level
)
from ...coherence.gct_calculator import EnhancedGCTCalculator
from ...core.database import Database
from ..dependencies import (
    get_current_active_user, get_database, get_gct_calculator,
    require_feature_access, validate_coherence_data, validate_individual_parameters,
    get_pagination_params, PaginationParams
)

router = APIRouter(prefix="/coherence")

# Pydantic models for request/response

class CoherenceComponentsRequest(BaseModel):
    """Request model for coherence components"""
    psi: float = Field(..., ge=0, le=1, description="Internal consistency")
    rho: float = Field(..., ge=0, le=1, description="Accumulated wisdom") 
    q: float = Field(..., ge=0, le=1, description="Moral activation energy")
    f: float = Field(..., ge=0, le=1, description="Social belonging")
    
    @validator('psi', 'rho', 'q', 'f')
    def validate_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Value must be between 0 and 1')
        return v


class ParameterCalibrationRequest(BaseModel):
    """Request model for parameter calibration"""
    k_m: float = Field(..., ge=0.1, le=0.5, description="Activation threshold")
    k_i: float = Field(..., ge=0.5, le=2.0, description="Sustainability threshold")


class CoherenceResponse(BaseModel):
    """Response model for coherence calculation"""
    coherence_score: float
    level: str
    components: Dict[str, float]
    metrics: Dict[str, float]
    timestamp: datetime
    user_id: str


class CoherenceTrajectoryResponse(BaseModel):
    """Response model for coherence trajectory"""
    trajectory: List[CoherenceResponse]
    derivatives: Dict[str, float]
    statistics: Dict[str, float]
    predictions: Optional[Dict[str, List[float]]] = None
    trend_analysis: Dict[str, Any]


class InterventionRecommendations(BaseModel):
    """Response model for intervention recommendations"""
    priority: str
    interventions: List[Dict[str, Any]]
    focus_areas: List[str]
    timeline: str
    personalized_plan: Dict[str, Any]


# Endpoints

@router.post("/calculate", response_model=CoherenceResponse)
async def calculate_coherence(
    components: CoherenceComponentsRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database),
    calculator: EnhancedGCTCalculator = Depends(get_gct_calculator),
    _: bool = Depends(validate_coherence_data)
):
    """Calculate coherence score from components"""
    
    try:
        # Get user parameters
        user_params = await db.get_user_parameters(current_user.id)
        if not user_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User parameters not calibrated. Please complete calibration first."
            )
        
        # Create GCT components
        gct_components = GCTComponents(
            psi=components.psi,
            rho=components.rho,
            q=components.q,
            f=components.f
        )
        
        # Calculate coherence
        coherence_score, metrics = calculator.calculate_coherence(
            gct_components, user_params
        )
        
        # Determine coherence level
        coherence_level = get_coherence_level(coherence_score)
        
        # Create profile
        profile = CoherenceProfile(
            user_id=current_user.id,
            components=gct_components,
            parameters=user_params,
            coherence_score=coherence_score,
            level=coherence_level
        )
        
        # Save to database
        background_tasks.add_task(save_coherence_profile, db, profile)
        
        # Update user tracking
        background_tasks.add_task(update_user_coherence_tracking, db, current_user.id)
        
        return CoherenceResponse(
            coherence_score=coherence_score,
            level=coherence_level.value,
            components={
                "psi": components.psi,
                "rho": components.rho,
                "q": components.q,
                "f": components.f,
                "soul_echo": gct_components.soul_echo
            },
            metrics=metrics,
            timestamp=datetime.utcnow(),
            user_id=current_user.id
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate coherence"
        )


@router.get("/trajectory", response_model=CoherenceTrajectoryResponse)
async def get_coherence_trajectory(
    days: int = Field(30, ge=1, le=365, description="Number of days to analyze"),
    include_predictions: bool = Field(True, description="Include predictive analysis"),
    current_user: User = Depends(require_feature_access("basic_coherence")),
    db: Database = Depends(get_database),
    calculator: EnhancedGCTCalculator = Depends(get_gct_calculator)
):
    """Get coherence trajectory with analysis and predictions"""
    
    try:
        # Get historical profiles
        since = datetime.utcnow() - timedelta(days=days)
        profiles = await db.get_coherence_history(current_user.id, since)
        
        if len(profiles) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient history for trajectory analysis. Need at least 2 data points."
            )
        
        # Calculate derivatives and statistics
        derivatives = calculator.calculate_derivatives(profiles)
        statistics = calculate_trajectory_statistics(profiles)
        
        # Generate predictions if requested and user has access
        predictions = None
        if include_predictions and current_user.can_access_feature("advanced_analytics"):
            if len(profiles) >= 5:
                latest_profile = profiles[-1]
                predictions = calculator.predict_trajectory(latest_profile, horizon_days=30)
        
        # Perform trend analysis
        trend_analysis = analyze_coherence_trends(profiles, derivatives)
        
        # Format trajectory response
        trajectory = [
            CoherenceResponse(
                coherence_score=p.coherence_score,
                level=p.level.value,
                components={
                    'psi': p.components.psi,
                    'rho': p.components.rho,
                    'q': p.components.q,
                    'f': p.components.f,
                    'soul_echo': p.components.soul_echo
                },
                metrics={'timestamp': p.components.timestamp.isoformat()},
                timestamp=p.components.timestamp,
                user_id=p.user_id
            )
            for p in profiles
        ]
        
        return CoherenceTrajectoryResponse(
            trajectory=trajectory,
            derivatives=derivatives,
            statistics=statistics,
            predictions=predictions,
            trend_analysis=trend_analysis
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze trajectory"
        )


@router.post("/calibrate-parameters")
async def calibrate_parameters(
    params: ParameterCalibrationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database),
    _: bool = Depends(validate_individual_parameters)
):
    """Calibrate individual K_m and K_i parameters"""
    
    try:
        # Create parameters object
        parameters = IndividualParameters(
            k_m=params.k_m,
            k_i=params.k_i,
            user_id=current_user.id
        )
        
        # Save to database
        await db.save_user_parameters(parameters)
        
        # Update blockchain in background
        background_tasks.add_task(update_blockchain_parameters, parameters)
        
        return {
            "status": "success",
            "message": "Parameters calibrated successfully",
            "k_m": params.k_m,
            "k_i": params.k_i,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calibrate parameters"
        )


@router.post("/auto-calibrate")
async def auto_calibrate_parameters(
    assessment_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Auto-calibrate parameters from assessment data"""
    
    try:
        # Run calibration algorithm
        k_m, k_i = await run_parameter_calibration(assessment_data, current_user.id)
        
        # Create parameters object
        parameters = IndividualParameters(
            k_m=k_m,
            k_i=k_i,
            user_id=current_user.id
        )
        
        # Save to database
        await db.save_user_parameters(parameters)
        
        # Update blockchain in background
        background_tasks.add_task(update_blockchain_parameters, parameters)
        
        return {
            "status": "success",
            "message": "Parameters auto-calibrated successfully",
            "k_m": k_m,
            "k_i": k_i,
            "confidence": 0.85,  # Would be calculated from assessment quality
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to auto-calibrate parameters"
        )


@router.get("/interventions", response_model=InterventionRecommendations)
async def get_intervention_recommendations(
    current_user: User = Depends(require_feature_access("advanced_analytics")),
    db: Database = Depends(get_database),
    calculator: EnhancedGCTCalculator = Depends(get_gct_calculator)
):
    """Get personalized intervention recommendations"""
    
    try:
        # Get latest profile
        latest_profile = await db.get_latest_coherence_profile(current_user.id)
        if not latest_profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No coherence profile found. Please complete an assessment first."
            )
        
        # Get recent trajectory for context
        since = datetime.utcnow() - timedelta(days=30)
        recent_profiles = await db.get_coherence_history(current_user.id, since)
        
        # Calculate intervention recommendations
        recommendations = calculator.calculate_intervention_recommendations(
            latest_profile, 
            {'recent_profiles': recent_profiles}
        )
        
        # Generate personalized plan
        personalized_plan = generate_personalized_intervention_plan(
            latest_profile, recommendations, recent_profiles
        )
        
        return InterventionRecommendations(
            priority=recommendations['priority'],
            interventions=recommendations['interventions'],
            focus_areas=recommendations['focus_areas'],
            timeline=recommendations['timeline'],
            personalized_plan=personalized_plan
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate intervention recommendations"
        )


@router.get("/current", response_model=Optional[CoherenceResponse])
async def get_current_coherence(
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Get user's current coherence profile"""
    
    try:
        profile = await db.get_latest_coherence_profile(current_user.id)
        if not profile:
            return None
        
        return CoherenceResponse(
            coherence_score=profile.coherence_score,
            level=profile.level.value,
            components={
                'psi': profile.components.psi,
                'rho': profile.components.rho,
                'q': profile.components.q,
                'f': profile.components.f,
                'soul_echo': profile.components.soul_echo
            },
            metrics={'last_update': profile.components.timestamp.isoformat()},
            timestamp=profile.components.timestamp,
            user_id=profile.user_id
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve current coherence"
        )


@router.get("/history", response_model=List[CoherenceResponse])
async def get_coherence_history(
    pagination: PaginationParams = Depends(get_pagination_params),
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Get user's coherence history with pagination"""
    
    try:
        profiles = await db.get_coherence_history_paginated(
            current_user.id, 
            skip=pagination.skip, 
            limit=pagination.limit
        )
        
        return [
            CoherenceResponse(
                coherence_score=p.coherence_score,
                level=p.level.value,
                components={
                    'psi': p.components.psi,
                    'rho': p.components.rho,
                    'q': p.components.q,
                    'f': p.components.f,
                    'soul_echo': p.components.soul_echo
                },
                metrics={},
                timestamp=p.components.timestamp,
                user_id=p.user_id
            )
            for p in profiles
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve coherence history"
        )


# Background task functions

async def save_coherence_profile(db: Database, profile: CoherenceProfile):
    """Background task to save coherence profile"""
    try:
        await db.save_coherence_profile(profile)
    except Exception as e:
        # Log error but don't fail the main request
        print(f"Failed to save coherence profile: {e}")


async def update_user_coherence_tracking(db: Database, user_id: str):
    """Background task to update user coherence tracking"""
    try:
        await db.update_user_coherence_count(user_id)
    except Exception as e:
        print(f"Failed to update user tracking: {e}")


async def update_blockchain_parameters(parameters: IndividualParameters):
    """Background task to update parameters on blockchain"""
    try:
        # This would integrate with blockchain interface
        pass
    except Exception as e:
        print(f"Failed to update blockchain parameters: {e}")


# Utility functions

async def run_parameter_calibration(
    assessment_data: Dict[str, Any], 
    user_id: str
) -> tuple[float, float]:
    """Run parameter calibration algorithm"""
    
    # This would implement sophisticated calibration logic
    # For now, return reasonable defaults based on assessment
    
    # Extract key metrics from assessment
    consistency_score = assessment_data.get('consistency_score', 0.5)
    wisdom_score = assessment_data.get('wisdom_score', 0.5)
    moral_score = assessment_data.get('moral_score', 0.5)
    social_score = assessment_data.get('social_score', 0.5)
    
    # Calculate K_m (activation threshold) - lower for more reactive individuals
    k_m = 0.5 - (moral_score * 0.4)
    k_m = max(0.1, min(0.5, k_m))
    
    # Calculate K_i (sustainability threshold) - higher for more stable individuals
    k_i = 0.5 + (consistency_score * wisdom_score * 1.5)
    k_i = max(0.5, min(2.0, k_i))
    
    return k_m, k_i


def calculate_trajectory_statistics(profiles: List[CoherenceProfile]) -> Dict[str, float]:
    """Calculate trajectory statistics"""
    
    coherence_scores = [p.coherence_score for p in profiles]
    
    import numpy as np
    
    return {
        'mean': float(np.mean(coherence_scores)),
        'std': float(np.std(coherence_scores)),
        'min': float(np.min(coherence_scores)),
        'max': float(np.max(coherence_scores)),
        'range': float(np.max(coherence_scores) - np.min(coherence_scores)),
        'latest': coherence_scores[-1] if coherence_scores else 0.0,
        'count': len(profiles)
    }


def analyze_coherence_trends(
    profiles: List[CoherenceProfile], 
    derivatives: Dict[str, float]
) -> Dict[str, Any]:
    """Analyze coherence trends"""
    
    trend_direction = "stable"
    trend_strength = "weak"
    
    dC_dt = derivatives.get('dC_dt', 0)
    volatility = derivatives.get('volatility', 0)
    
    if dC_dt > 0.02:
        trend_direction = "improving"
    elif dC_dt < -0.02:
        trend_direction = "declining"
    
    if abs(dC_dt) > 0.05:
        trend_strength = "strong"
    elif abs(dC_dt) > 0.02:
        trend_strength = "moderate"
    
    stability = "stable" if volatility < 0.1 else "volatile"
    
    return {
        'direction': trend_direction,
        'strength': trend_strength,
        'stability': stability,
        'velocity': dC_dt,
        'volatility': volatility,
        'recommendation': generate_trend_recommendation(trend_direction, trend_strength, stability)
    }


def generate_trend_recommendation(direction: str, strength: str, stability: str) -> str:
    """Generate recommendation based on trend analysis"""
    
    if direction == "declining" and strength in ["strong", "moderate"]:
        return "Consider immediate intervention and support"
    elif direction == "improving" and strength == "strong":
        return "Continue current practices and maintain momentum"
    elif stability == "volatile":
        return "Focus on consistency and stabilization"
    else:
        return "Maintain current approach with regular monitoring"


def generate_personalized_intervention_plan(
    profile: CoherenceProfile,
    recommendations: Dict[str, Any],
    recent_profiles: List[CoherenceProfile]
) -> Dict[str, Any]:
    """Generate personalized intervention plan"""
    
    plan = {
        'immediate_actions': [],
        'weekly_goals': [],
        'monthly_objectives': [],
        'resources': [],
        'monitoring_schedule': 'daily' if profile.level == CoherenceLevel.CRITICAL else 'weekly'
    }
    
    # Add specific actions based on coherence level and weak areas
    weak_components = [
        component for component, value in [
            ('psi', profile.components.psi),
            ('rho', profile.components.rho),
            ('q', profile.components.q),
            ('f', profile.components.f)
        ] if value < 0.4
    ]
    
    for component in weak_components:
        if component == 'psi':
            plan['weekly_goals'].append("Practice daily consistency checks between thoughts and actions")
        elif component == 'rho':
            plan['weekly_goals'].append("Engage in daily reflection and experience integration")
        elif component == 'q':
            plan['weekly_goals'].append("Identify and commit to 3 core values with daily alignment check")
        elif component == 'f':
            plan['weekly_goals'].append("Strengthen social connections through meaningful interactions")
    
    return plan