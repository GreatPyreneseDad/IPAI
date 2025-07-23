"""
LLM Integration API Endpoints

This module provides REST endpoints for LLM interactions,
chat functionality, and AI-powered features.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime
from pydantic import BaseModel, Field
import asyncio
import json

from ...models.user import User
from ...models.database_models import UserInteraction, InteractionType
from ...integrations.llm_providers import LLMManager
from ...safety.enhanced_coherence_tracker import EnhancedCoherenceTracker
from ...blockchain.personal_chain import PersonalBlockchain
from ...llm.coherence_analyzer import CoherenceAnalyzer
from ...llm.triadic_handler import TriadicResponseHandler
from ...core.database import Database
from ..dependencies import get_current_active_user, get_database

router = APIRouter(prefix="/llm")

# Global managers
llm_manager = LLMManager()
user_sessions: Dict[str, Dict[str, Any]] = {}


# Pydantic models

class ChatMessage(BaseModel):
    """Chat message model"""
    content: str = Field(..., min_length=1, max_length=10000)
    role: str = Field("user", regex="^(user|assistant|system)$")
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat request model"""
    message: str = Field(..., min_length=1, max_length=5000)
    context: Optional[List[ChatMessage]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(2000, ge=1, le=10000)
    stream: bool = False


class ChatResponse(BaseModel):
    """Chat response model"""
    message: str
    coherence_impact: float
    safety_score: float
    metadata: Dict[str, Any]


class AnalysisRequest(BaseModel):
    """Analysis request model"""
    text: str = Field(..., min_length=1, max_length=10000)
    analysis_type: str = Field(..., regex="^(coherence|sentiment|themes|summary)$")
    depth: str = Field("standard", regex="^(quick|standard|deep)$")


class CompletionRequest(BaseModel):
    """Completion request model"""
    prompt: str = Field(..., min_length=1, max_length=5000)
    system_prompt: Optional[str] = None
    examples: Optional[List[Dict[str, str]]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1000, ge=1, le=10000)


# Helper functions

def get_user_session(user_id: str) -> Dict[str, Any]:
    """Get or create user session"""
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "conversation_history": [],
            "coherence_tracker": EnhancedCoherenceTracker(),
            "personal_chain": PersonalBlockchain(user_id, f"ipai_{user_id}"),
            "triadic_handler": TriadicResponseHandler(),
            "last_interaction": datetime.utcnow()
        }
    return user_sessions[user_id]


async def process_chat_message(
    user_id: str,
    message: str,
    session: Dict[str, Any],
    db: Database,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Dict[str, Any]:
    """Process a chat message with coherence tracking"""
    
    # Get LLM response
    llm_response = await llm_manager.complete(
        prompt=message,
        provider=provider,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    # Apply triadic processing for coherence
    processed_response = session["triadic_handler"].process_response(
        user_input=message,
        llm_response=llm_response,
        coherence_level=session["coherence_tracker"].current_psi
    )
    
    # Update coherence tracking
    coherence_snapshot = await session["coherence_tracker"].update_coherence_with_safety(
        delta_psi=processed_response.get("coherence_delta", 0.01),
        user_input=message,
        ai_response=processed_response["response"],
        event_type="chat",
        trigger="llm_interaction",
        ambiguity_level=processed_response.get("ambiguity_level", 0.3)
    )
    
    # Add to personal blockchain
    block = await session["personal_chain"].add_interaction(
        user_input=message,
        ai_response=processed_response["response"],
        coherence_score=coherence_snapshot.psi_value,
        soul_echo=coherence_snapshot.psi_value,
        safety_score=coherence_snapshot.safety_metrics.safety_score,
        inference_data={
            "type": "chat",
            "triadic_processing": processed_response.get("triadic_components", {}),
            "provider": provider or llm_manager.active_provider,
            "model": model
        }
    )
    
    # Save interaction to database
    interaction = UserInteraction(
        user_id=user_id,
        interaction_type=InteractionType.CHAT,
        input_text=message,
        output_text=processed_response["response"],
        coherence_before=coherence_snapshot.psi_value - processed_response.get("coherence_delta", 0),
        coherence_after=coherence_snapshot.psi_value,
        coherence_delta=processed_response.get("coherence_delta", 0),
        safety_score=coherence_snapshot.safety_metrics.safety_score,
        howlround_risk=coherence_snapshot.safety_metrics.howlround_risk,
        pressure_score=coherence_snapshot.pressure_metrics.pressure_score,
        intervention_triggered=coherence_snapshot.safety_metrics.intervention_needed,
        llm_provider=provider or llm_manager.active_provider,
        llm_model=model,
        block_hash=block.hash
    )
    
    await db.add_interaction(interaction)
    
    # Update conversation history
    session["conversation_history"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.utcnow()
    })
    session["conversation_history"].append({
        "role": "assistant",
        "content": processed_response["response"],
        "timestamp": datetime.utcnow()
    })
    
    # Trim history if too long
    if len(session["conversation_history"]) > 50:
        session["conversation_history"] = session["conversation_history"][-40:]
    
    return {
        "response": processed_response["response"],
        "coherence_impact": processed_response.get("coherence_delta", 0),
        "safety_score": coherence_snapshot.safety_metrics.safety_score,
        "intervention_log": coherence_snapshot.intervention_log,
        "block_hash": block.hash,
        "metadata": {
            "triadic_components": processed_response.get("triadic_components", {}),
            "current_coherence": coherence_snapshot.psi_value,
            "coherence_state": coherence_snapshot.state.label,
            "relationship_quality": coherence_snapshot.relationship_quality
        }
    }


# Endpoints

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: Database = Depends(get_database)
):
    """Chat with LLM with coherence tracking"""
    
    try:
        session = get_user_session(current_user.id)
        
        # Add context to conversation if provided
        if request.context:
            for msg in request.context:
                session["conversation_history"].append({
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": datetime.utcnow()
                })
        
        # Build full prompt with history
        full_prompt = build_conversation_prompt(
            message=request.message,
            history=session["conversation_history"][-10:]  # Last 10 messages
        )
        
        # Process message
        result = await process_chat_message(
            user_id=current_user.id,
            message=full_prompt,
            session=session,
            db=db,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Update user coherence in background
        background_tasks.add_task(
            update_user_coherence,
            db,
            current_user.id,
            result["metadata"]["current_coherence"]
        )
        
        return ChatResponse(
            message=result["response"],
            coherence_impact=result["coherence_impact"],
            safety_score=result["safety_score"],
            metadata=result["metadata"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat processing failed: {str(e)}"
        )


@router.websocket("/chat/stream")
async def chat_stream(
    websocket: WebSocket,
    token: str
):
    """WebSocket endpoint for streaming chat"""
    
    await websocket.accept()
    
    try:
        # Authenticate user
        user = await authenticate_websocket(token)
        if not user:
            await websocket.close(code=4001, reason="Unauthorized")
            return
        
        session = get_user_session(user.id)
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            message = data.get("message", "")
            if not message:
                continue
            
            # Stream response
            full_response = ""
            coherence_delta = 0
            
            try:
                async for chunk in llm_manager.stream_complete(
                    prompt=message,
                    provider=data.get("provider"),
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens", 2000)
                ):
                    full_response += chunk
                    await websocket.send_json({
                        "type": "chunk",
                        "content": chunk
                    })
                
                # Process complete response for coherence
                processed = session["triadic_handler"].process_response(
                    user_input=message,
                    llm_response=full_response,
                    coherence_level=session["coherence_tracker"].current_psi
                )
                
                # Update coherence
                snapshot = await session["coherence_tracker"].update_coherence_with_safety(
                    delta_psi=processed.get("coherence_delta", 0.01),
                    user_input=message,
                    ai_response=full_response,
                    event_type="chat",
                    trigger="stream_interaction",
                    ambiguity_level=processed.get("ambiguity_level", 0.3)
                )
                
                # Send completion message
                await websocket.send_json({
                    "type": "complete",
                    "coherence_impact": processed.get("coherence_delta", 0),
                    "safety_score": snapshot.safety_metrics.safety_score,
                    "metadata": {
                        "current_coherence": snapshot.psi_value,
                        "coherence_state": snapshot.state.label
                    }
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.close(code=4000, reason=str(e))


@router.post("/analyze")
async def analyze_text(
    request: AnalysisRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Analyze text for various insights"""
    
    try:
        analyzer = CoherenceAnalyzer()
        
        if request.analysis_type == "coherence":
            result = await analyzer.analyze_coherence(
                request.text,
                depth=request.depth
            )
        elif request.analysis_type == "sentiment":
            result = await analyzer.analyze_sentiment(request.text)
        elif request.analysis_type == "themes":
            result = await analyzer.extract_themes(request.text)
        elif request.analysis_type == "summary":
            result = await analyzer.generate_summary(
                request.text,
                style=request.depth
            )
        else:
            raise ValueError(f"Unknown analysis type: {request.analysis_type}")
        
        return {
            "analysis_type": request.analysis_type,
            "result": result,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@router.post("/complete")
async def complete_prompt(
    request: CompletionRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Complete a prompt with optional examples"""
    
    try:
        # Build full prompt with system and examples
        full_prompt = build_completion_prompt(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            examples=request.examples
        )
        
        # Get completion
        response = await llm_manager.complete(
            prompt=full_prompt,
            provider=request.provider,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        return {
            "completion": response,
            "prompt_tokens": estimate_tokens(full_prompt),
            "completion_tokens": estimate_tokens(response),
            "model": request.model or "default",
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Completion failed: {str(e)}"
        )


@router.get("/models")
async def list_available_models(
    current_user: User = Depends(get_current_active_user)
):
    """List available LLM models"""
    
    providers = llm_manager.list_providers()
    models_by_provider = {}
    
    for provider in providers:
        provider_type = provider["provider"]
        models = llm_manager.get_provider_models(provider_type)
        models_by_provider[provider["name"]] = {
            "provider_type": provider_type,
            "models": models,
            "active": provider["active"]
        }
    
    return {
        "providers": models_by_provider,
        "active_provider": llm_manager.active_provider
    }


@router.post("/set-provider")
async def set_active_provider(
    provider_name: str,
    current_user: User = Depends(get_current_active_user)
):
    """Set the active LLM provider"""
    
    try:
        llm_manager.set_active_provider(provider_name)
        
        return {
            "status": "success",
            "active_provider": provider_name
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("/conversation-history")
async def get_conversation_history(
    limit: int = 50,
    current_user: User = Depends(get_current_active_user)
):
    """Get user's conversation history"""
    
    session = get_user_session(current_user.id)
    history = session["conversation_history"][-limit:]
    
    return {
        "history": history,
        "total_messages": len(session["conversation_history"]),
        "session_start": session.get("session_start", datetime.utcnow())
    }


@router.delete("/conversation-history")
async def clear_conversation_history(
    current_user: User = Depends(get_current_active_user)
):
    """Clear user's conversation history"""
    
    session = get_user_session(current_user.id)
    session["conversation_history"] = []
    session["session_start"] = datetime.utcnow()
    
    return {
        "status": "success",
        "message": "Conversation history cleared"
    }


# Helper functions

def build_conversation_prompt(message: str, history: List[Dict]) -> str:
    """Build a conversation prompt with history"""
    prompt_parts = []
    
    # Add relevant history
    for msg in history[-5:]:  # Last 5 messages
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            prompt_parts.append(f"User: {content}")
        elif role == "assistant":
            prompt_parts.append(f"Assistant: {content}")
    
    # Add current message
    prompt_parts.append(f"User: {message}")
    prompt_parts.append("Assistant:")
    
    return "\n\n".join(prompt_parts)


def build_completion_prompt(
    prompt: str,
    system_prompt: Optional[str] = None,
    examples: Optional[List[Dict[str, str]]] = None
) -> str:
    """Build a completion prompt with system message and examples"""
    parts = []
    
    if system_prompt:
        parts.append(f"System: {system_prompt}")
    
    if examples:
        parts.append("\nExamples:")
        for example in examples:
            parts.append(f"Input: {example.get('input', '')}")
            parts.append(f"Output: {example.get('output', '')}")
    
    parts.append(f"\nInput: {prompt}")
    parts.append("Output:")
    
    return "\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count"""
    # Approximate: 1 token ≈ 4 characters
    return len(text) // 4


async def update_user_coherence(db: Database, user_id: str, coherence_score: float):
    """Background task to update user's coherence score"""
    try:
        await db.update_user_coherence(user_id, coherence_score)
    except Exception as e:
        print(f"Failed to update user coherence: {e}")


async def authenticate_websocket(token: str) -> Optional[User]:
    """Authenticate WebSocket connection"""
    # This would validate the JWT token
    # For now, return None to indicate not implemented
    return None