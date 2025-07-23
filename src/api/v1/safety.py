"""
Safety API Endpoints

This module provides REST endpoints for safety monitoring,
howlround detection, and personal blockchain management.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

from ...models.user import User
from ...safety.enhanced_coherence_tracker import EnhancedCoherenceTracker, SafetyMetrics
from ...safety.howlround_detector import HowlroundDetector
from ...blockchain.personal_chain import PersonalBlockchain, InferenceStatus
from ...core.database import Database
from ..dependencies import get_current_active_user, get_database

router = APIRouter(prefix="/safety")

# Global trackers (in production, use proper state management)
user_trackers: Dict[str, EnhancedCoherenceTracker] = {}
user_chains: Dict[str, PersonalBlockchain] = {}


# Pydantic models

class InteractionRequest(BaseModel):
    """Request model for recording interactions"""
    user_input: str = Field(..., min_length=1, max_length=5000)
    ai_response: str = Field(..., min_length=1, max_length=10000)
    event_type: str = Field("conversation", description="Type of interaction")
    ambiguity_level: float = Field(0.0, ge=0, le=1, description="Ambiguity level")
    inference_data: Optional[Dict[str, Any]] = None


class SafetyStatusResponse(BaseModel):
    """Response model for safety status"""
    user_id: str
    coherence_state: str
    safety_score: float
    howlround_risk: float
    pressure_score: float
    intervention_needed: bool
    relationship_quality: float
    active_risks: Dict[str, Any]
    recommendations: List[str]


class BlockchainStatusResponse(BaseModel):
    """Response model for personal blockchain status"""
    user_id: str
    chain_length: int
    average_coherence: float
    verified_inferences: int
    pending_inferences: int
    chain_valid: bool
    latest_block: Dict[str, Any]


class InferenceSubmission(BaseModel):
    """Request model for inference submission"""
    inference_type: str = Field(..., description="Type of inference")
    content: Dict[str, Any] = Field(..., description="Inference content")
    confidence_score: float = Field(..., ge=0, le=1)


# Helper functions

def get_or_create_tracker(user_id: str, ipai_id: str) -> EnhancedCoherenceTracker:
    """Get or create coherence tracker for user"""
    if user_id not in user_trackers:
        user_trackers[user_id] = EnhancedCoherenceTracker(
            initial_psi=1.0,
            personal_chain_id=f"{user_id}_{ipai_id}"
        )
    return user_trackers[user_id]


def get_or_create_chain(user_id: str, ipai_id: str) -> PersonalBlockchain:
    """Get or create personal blockchain for user"""
    if user_id not in user_chains:
        user_chains[user_id] = PersonalBlockchain(user_id, ipai_id)
    return user_chains[user_id]


# Endpoints

@router.post("/interaction")
async def record_interaction(
    interaction: InteractionRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Record an interaction with safety monitoring"""
    
    try:
        # Get or create trackers
        tracker = get_or_create_tracker(current_user.id, current_user.id)
        chain = get_or_create_chain(current_user.id, current_user.id)
        
        # Calculate delta_psi based on interaction quality
        # This would be more sophisticated in production
        base_delta = 0.01  # Small positive change for successful interaction
        
        # Adjust based on ambiguity
        if interaction.ambiguity_level > 0.7:
            base_delta *= -2  # High ambiguity is negative
        elif interaction.ambiguity_level < 0.3:
            base_delta *= 2  # Low ambiguity is positive
        
        # Update coherence with safety tracking
        snapshot = await tracker.update_coherence_with_safety(
            delta_psi=base_delta,
            user_input=interaction.user_input,
            ai_response=interaction.ai_response,
            event_type=interaction.event_type,
            trigger="api_interaction",
            ambiguity_level=interaction.ambiguity_level
        )
        
        # Add to personal blockchain
        block = await chain.add_interaction(
            user_input=interaction.user_input,
            ai_response=interaction.ai_response,
            coherence_score=snapshot.psi_value,
            soul_echo=snapshot.psi_value,  # Simplified for now
            safety_score=snapshot.safety_metrics.safety_score,
            inference_data=interaction.inference_data
        )
        
        # Save to database in background
        background_tasks.add_task(
            save_interaction_to_db,
            db, current_user.id, snapshot, block
        )
        
        return {
            "status": "recorded",
            "coherence": snapshot.psi_value,
            "safety_score": snapshot.safety_metrics.safety_score,
            "intervention_needed": snapshot.safety_metrics.intervention_needed,
            "block_hash": block.hash,
            "recommendations": snapshot.intervention_log
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record interaction: {str(e)}"
        )


@router.get("/status", response_model=SafetyStatusResponse)
async def get_safety_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get current safety status and relationship assessment"""
    
    try:
        tracker = get_or_create_tracker(current_user.id, current_user.id)
        assessment = tracker.get_relationship_assessment()
        
        # Get latest snapshot
        if not tracker.snapshots:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No safety data available. Record an interaction first."
            )
        
        latest = tracker.snapshots[-1]
        
        return SafetyStatusResponse(
            user_id=current_user.id,
            coherence_state=latest.state.label,
            safety_score=latest.safety_metrics.safety_score,
            howlround_risk=latest.safety_metrics.howlround_risk,
            pressure_score=latest.pressure_metrics.pressure_score,
            intervention_needed=latest.safety_metrics.intervention_needed,
            relationship_quality=latest.relationship_quality,
            active_risks=assessment['current_risks'],
            recommendations=assessment['recommendations']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve safety status"
        )


@router.get("/blockchain/status", response_model=BlockchainStatusResponse)
async def get_blockchain_status(
    current_user: User = Depends(get_current_active_user)
):
    """Get personal blockchain status"""
    
    try:
        chain = get_or_create_chain(current_user.id, current_user.id)
        chain_data = chain.export_chain_data()
        
        latest_block = chain.get_latest_block()
        
        return BlockchainStatusResponse(
            user_id=current_user.id,
            chain_length=chain_data['chain_length'],
            average_coherence=chain_data['average_coherence'],
            verified_inferences=len(chain.verified_inferences),
            pending_inferences=len(chain.pending_inferences),
            chain_valid=chain_data['chain_valid'],
            latest_block={
                'index': latest_block.index,
                'hash': latest_block.hash,
                'timestamp': latest_block.timestamp,
                'coherence_score': latest_block.coherence_score
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve blockchain status"
        )


@router.post("/inference/submit")
async def submit_inference(
    inference: InferenceSubmission,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """Submit an inference for verification"""
    
    try:
        chain = get_or_create_chain(current_user.id, current_user.id)
        
        # Get latest block index
        latest_block = chain.get_latest_block()
        
        # Record inference
        inference_record = await chain.record_inference(
            block_index=latest_block.index,
            inference_type=inference.inference_type,
            inference_content=inference.content,
            confidence_score=inference.confidence_score
        )
        
        # Try to verify in background
        background_tasks.add_task(
            verify_inference_task,
            chain, inference_record
        )
        
        return {
            "status": "submitted",
            "verification_hash": inference_record.verification_hash,
            "inference_status": inference_record.status.value,
            "block_index": inference_record.block_index
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit inference"
        )


@router.get("/inference/pending")
async def get_pending_inferences(
    current_user: User = Depends(get_current_active_user)
):
    """Get pending inferences awaiting verification"""
    
    try:
        chain = get_or_create_chain(current_user.id, current_user.id)
        
        pending = [
            {
                'verification_hash': inf.verification_hash,
                'type': inf.inference_type,
                'confidence': inf.confidence_score,
                'timestamp': inf.timestamp,
                'coherence_context': inf.coherence_context
            }
            for inf in chain.pending_inferences
        ]
        
        return {
            "count": len(pending),
            "inferences": pending
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pending inferences"
        )


@router.post("/inference/verify/{verification_hash}")
async def verify_inference(
    verification_hash: str,
    current_user: User = Depends(get_current_active_user)
):
    """Manually verify a pending inference"""
    
    try:
        chain = get_or_create_chain(current_user.id, current_user.id)
        
        # Find inference
        inference = next(
            (i for i in chain.pending_inferences if i.verification_hash == verification_hash),
            None
        )
        
        if not inference:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Inference not found"
            )
        
        # Verify
        verified = await chain.verify_inference(inference)
        
        return {
            "status": "verified" if verified else "rejected",
            "verification_hash": verification_hash,
            "inference_type": inference.inference_type
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify inference"
        )


@router.post("/intervention/callback")
async def register_intervention_callback(
    callback_url: str,
    current_user: User = Depends(get_current_active_user)
):
    """Register a callback URL for intervention events"""
    
    try:
        tracker = get_or_create_tracker(current_user.id, current_user.id)
        
        # In production, validate the callback URL
        # For now, just register a simple callback
        async def url_callback(snapshot):
            # Would make HTTP request to callback_url
            print(f"Would notify {callback_url} about intervention")
        
        tracker.register_intervention_callback(url_callback)
        
        return {
            "status": "registered",
            "callback_url": callback_url
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register callback"
        )


@router.get("/howlround/analysis")
async def get_howlround_analysis(
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed howlround analysis"""
    
    try:
        tracker = get_or_create_tracker(current_user.id, current_user.id)
        
        # Get howlround detector report
        report = tracker.howlround_detector.get_safety_report()
        
        return {
            "user_id": current_user.id,
            "howlround_status": report['status'],
            "total_events": report['total_events'],
            "current_risk": report['current_risk'],
            "dominant_pattern": report.get('dominant_pattern', 'none'),
            "recommendations": report['recommendations']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get howlround analysis"
        )


# Background task functions

async def save_interaction_to_db(
    db: Database,
    user_id: str,
    snapshot: Any,
    block: Any
):
    """Save interaction data to database"""
    try:
        # This would save to the database
        # For now, just log
        print(f"Saving interaction for user {user_id}")
    except Exception as e:
        print(f"Failed to save interaction: {e}")


async def verify_inference_task(chain: PersonalBlockchain, inference: Any):
    """Background task to verify inference"""
    try:
        await chain.verify_inference(inference)
    except Exception as e:
        print(f"Failed to verify inference: {e}")