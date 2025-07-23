"""
LLM API Endpoints

This module provides REST endpoints for LLM interactions
with GCT-aware processing and coherence analysis.
"""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import asyncio
import json

from ...models.user import User
from ...models.coherence_profile import CoherenceProfile
from ...llm.interface import GCTLLMInterface, LLMResponse
from ...llm.coherence_analyzer import CoherenceAnalysis
from ...core.database import Database
from ..dependencies import (
    get_current_active_user, get_database, get_llm_interface,
    require_feature_access, get_user_coherence_profile, require_user_coherence_profile,
    llm_rate_limit, get_request_context
)

router = APIRouter(prefix="/llm")

# Pydantic models

class ChatRequest(BaseModel):
    """Request model for chat interactions"""
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    stream: Optional[bool] = Field(True, description="Enable streaming response")
    include_analysis: Optional[bool] = Field(False, description="Include coherence analysis")


class CoherenceCheckRequest(BaseModel):
    """Request model for coherence checking"""
    message: str = Field(..., min_length=1, max_length=2000, description="Message to analyze")
    detailed: Optional[bool] = Field(False, description="Include detailed analysis")


class InterventionRequest(BaseModel):
    """Request model for intervention recommendations"""
    focus_area: Optional[str] = Field(None, description="Specific focus area")
    urgency: Optional[str] = Field("normal", description="Urgency level")


class ChatResponse(BaseModel):
    """Response model for chat interactions"""
    response: str
    coherence_analysis: Optional[Dict[str, Any]] = None
    processing_metadata: Dict[str, Any]
    timestamp: datetime
    user_id: str


class CoherenceAnalysisResponse(BaseModel):
    """Response model for coherence analysis"""
    psi_score: float
    rho_score: float
    q_score: float
    f_score: float
    overall_coherence: float
    red_flags: List[str]
    positive_indicators: List[str]
    needs_grounding: bool
    confidence: float
    recommendations: List[str]


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    conversations: List[Dict[str, Any]]
    total_count: int
    page_info: Dict[str, Any]


# Endpoints

@router.post("/chat", response_model=ChatResponse)
async def chat_with_llm(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(require_feature_access("llm_integration")),
    profile: CoherenceProfile = Depends(require_user_coherence_profile),
    llm_interface: Optional[GCTLLMInterface] = Depends(get_llm_interface),
    db: Database = Depends(get_database),
    context: Dict = Depends(get_request_context),
    _: bool = Depends(llm_rate_limit)
):
    """Chat with GCT-aware LLM"""
    
    if not llm_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )
    
    try:
        # Merge request context with user context
        full_context = {**request.context, **context}
        
        # Generate response
        if request.stream:
            # Handle streaming in separate endpoint
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Use /chat/stream endpoint for streaming responses"
            )
        else:
            llm_response = await llm_interface.generate_response(
                request.message,
                profile,
                full_context,
                stream=False
            )
        
        # Save conversation to database
        background_tasks.add_task(
            save_conversation, 
            db, 
            current_user.id, 
            request.message, 
            llm_response
        )
        
        # Prepare response
        response = ChatResponse(
            response=llm_response.processed_text,
            coherence_analysis=format_coherence_analysis(llm_response) if request.include_analysis else None,
            processing_metadata=llm_response.processing_metadata,
            timestamp=llm_response.timestamp,
            user_id=current_user.id
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate response"
        )


@router.post("/chat/stream")
async def stream_chat_with_llm(
    request: ChatRequest,
    current_user: User = Depends(require_feature_access("llm_integration")),
    profile: CoherenceProfile = Depends(require_user_coherence_profile),
    llm_interface: Optional[GCTLLMInterface] = Depends(get_llm_interface),
    context: Dict = Depends(get_request_context),
    _: bool = Depends(llm_rate_limit)
):
    """Stream chat response from GCT-aware LLM"""
    
    if not llm_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )
    
    try:
        # Merge contexts
        full_context = {**request.context, **context}
        
        # Generate streaming response
        async def generate_stream():
            try:
                async for chunk in await llm_interface.generate_response(
                    request.message,
                    profile, 
                    full_context,
                    stream=True
                ):
                    # Format as Server-Sent Events
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk})}\n\n"
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                
            except Exception as e:
                # Send error signal
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8"
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start streaming response"
        )


@router.post("/coherence-check", response_model=CoherenceAnalysisResponse)
async def check_message_coherence(
    request: CoherenceCheckRequest,
    current_user: User = Depends(get_current_active_user),
    llm_interface: Optional[GCTLLMInterface] = Depends(get_llm_interface),
    _: bool = Depends(llm_rate_limit)
):
    """Analyze message coherence"""
    
    if not llm_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )
    
    try:
        # Analyze message coherence
        analysis = await llm_interface.check_message_coherence(request.message)
        
        # Generate recommendations based on analysis
        recommendations = generate_coherence_recommendations(analysis)
        
        return CoherenceAnalysisResponse(
            psi_score=analysis.psi_score,
            rho_score=analysis.rho_score,
            q_score=analysis.q_score,
            f_score=analysis.f_score,
            overall_coherence=analysis.overall_coherence,
            red_flags=analysis.red_flags,
            positive_indicators=analysis.positive_indicators,
            needs_grounding=analysis.needs_grounding,
            confidence=analysis.confidence,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze message coherence"
        )


@router.post("/intervention")
async def get_llm_intervention(
    request: InterventionRequest,
    current_user: User = Depends(require_feature_access("advanced_analytics")),
    profile: CoherenceProfile = Depends(require_user_coherence_profile),
    llm_interface: Optional[GCTLLMInterface] = Depends(get_llm_interface)
):
    """Get LLM-generated intervention recommendations"""
    
    if not llm_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )
    
    try:
        # Generate intervention recommendations
        intervention_prompt = llm_interface.generate_intervention_recommendations(profile)
        
        # Get LLM response for intervention
        context = {
            'intervention_request': True,
            'focus_area': request.focus_area,
            'urgency': request.urgency
        }
        
        response = await llm_interface.generate_response(
            f"Please provide intervention recommendations based on my coherence profile. Focus area: {request.focus_area or 'general'}",
            profile,
            context,
            stream=False
        )
        
        return {
            "intervention_text": response.processed_text,
            "focus_area": request.focus_area or "general",
            "urgency": request.urgency,
            "coherence_level": profile.level.value,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate intervention recommendations"
        )


@router.get("/conversation-history", response_model=ConversationHistoryResponse)
async def get_conversation_history(
    limit: int = Field(20, ge=1, le=100),
    offset: int = Field(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Get conversation history"""
    
    try:
        conversations, total_count = await db.get_conversation_history(
            current_user.id,
            limit=limit,
            offset=offset
        )
        
        return ConversationHistoryResponse(
            conversations=conversations,
            total_count=total_count,
            page_info={
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation history"
        )


@router.delete("/conversation-history")
async def clear_conversation_history(
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Clear conversation history"""
    
    try:
        await db.clear_conversation_history(current_user.id)
        
        return {
            "status": "success",
            "message": "Conversation history cleared",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear conversation history"
        )


@router.get("/performance-metrics")
async def get_llm_performance_metrics(
    current_user: User = Depends(require_feature_access("advanced_analytics")),
    llm_interface: Optional[GCTLLMInterface] = Depends(get_llm_interface)
):
    """Get LLM performance metrics"""
    
    if not llm_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )
    
    try:
        metrics = llm_interface.get_performance_metrics()
        
        return {
            "performance_metrics": metrics,
            "timestamp": datetime.utcnow(),
            "user_id": current_user.id
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.post("/reflection-exercise")
async def generate_reflection_exercise(
    focus_area: str = Field(..., description="Area for reflection"),
    current_user: User = Depends(get_current_active_user),
    profile: CoherenceProfile = Depends(require_user_coherence_profile),
    llm_interface: Optional[GCTLLMInterface] = Depends(get_llm_interface)
):
    """Generate personalized reflection exercise"""
    
    if not llm_interface:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is currently unavailable"
        )
    
    try:
        # Generate reflection prompt
        reflection_prompt = llm_interface.prompt_generator.generate_reflection_prompt(
            focus_area, profile
        )
        
        # Generate reflection exercise
        context = {
            'reflection_exercise': True,
            'focus_area': focus_area
        }
        
        response = await llm_interface.generate_response(
            reflection_prompt,
            profile,
            context,
            stream=False
        )
        
        return {
            "exercise": response.processed_text,
            "focus_area": focus_area,
            "coherence_level": profile.level.value,
            "estimated_duration": "10-15 minutes",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate reflection exercise"
        )


# Background task functions

async def save_conversation(
    db: Database,
    user_id: str,
    user_message: str,
    llm_response: LLMResponse
):
    """Background task to save conversation"""
    try:
        await db.save_conversation(
            user_id=user_id,
            user_message=user_message,
            assistant_response=llm_response.processed_text,
            coherence_analysis=llm_response.response_analysis.coherence_analysis if llm_response.response_analysis else None,
            processing_metadata=llm_response.processing_metadata,
            timestamp=llm_response.timestamp
        )
    except Exception as e:
        print(f"Failed to save conversation: {e}")


# Utility functions

def format_coherence_analysis(llm_response: LLMResponse) -> Optional[Dict[str, Any]]:
    """Format coherence analysis for API response"""
    
    if not llm_response.response_analysis or not llm_response.response_analysis.coherence_analysis:
        return None
    
    analysis = llm_response.response_analysis.coherence_analysis
    
    return {
        "psi_score": analysis.psi_score,
        "rho_score": analysis.rho_score,
        "q_score": analysis.q_score,
        "f_score": analysis.f_score,
        "overall_coherence": analysis.overall_coherence,
        "red_flags": analysis.red_flags,
        "positive_indicators": analysis.positive_indicators,
        "needs_grounding": analysis.needs_grounding,
        "confidence": analysis.confidence
    }


def generate_coherence_recommendations(analysis: CoherenceAnalysis) -> List[str]:
    """Generate recommendations based on coherence analysis"""
    
    recommendations = []
    
    # Component-specific recommendations
    if analysis.psi_score < 0.4:
        recommendations.append("Focus on aligning your thoughts and actions for greater internal consistency")
    
    if analysis.rho_score < 0.4:
        recommendations.append("Practice regular reflection to integrate experiences and build wisdom")
    
    if analysis.q_score < 0.4:
        recommendations.append("Clarify your values and commit to action aligned with your principles")
    
    if analysis.f_score < 0.4:
        recommendations.append("Strengthen social connections and community engagement")
    
    # Overall coherence recommendations
    if analysis.overall_coherence < 0.3:
        recommendations.append("Consider speaking with a counselor or coach for additional support")
    
    # Red flag recommendations
    if analysis.red_flags:
        if any("CRISIS" in flag for flag in analysis.red_flags):
            recommendations.append("URGENT: Please reach out to a mental health professional or crisis helpline immediately")
        elif "absolutist thinking" in str(analysis.red_flags).lower():
            recommendations.append("Practice seeing situations in shades of gray rather than black and white")
        elif "circular reasoning" in str(analysis.red_flags).lower():
            recommendations.append("Try to ground your thoughts in concrete evidence and examples")
    
    # Positive reinforcements
    if analysis.positive_indicators:
        if "reflective thinking" in analysis.positive_indicators:
            recommendations.append("Continue your excellent practice of reflective thinking")
        if "growth mindset" in analysis.positive_indicators:
            recommendations.append("Your growth mindset is a strong foundation for continued development")
    
    # Default recommendations if none specific
    if not recommendations:
        recommendations.append("Continue monitoring your coherence and practicing mindful self-awareness")
    
    return recommendations