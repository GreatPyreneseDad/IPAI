#!/usr/bin/env python3
"""
IPAI Dashboard Server

Serves the IPAI dashboard and provides API endpoints for the helical blockchain.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

# Simple data models
class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str = "1.0.0"

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    timestamp: datetime

class BlockchainLogRequest(BaseModel):
    interaction_type: str
    data: Dict[str, Any]
    user_id: str = "anonymous"
    coherence_profile: Optional[Dict[str, Any]] = None

# Create FastAPI app
app = FastAPI(
    title="IPAI Dashboard - Helical Blockchain",
    description="AI system with Grounded Coherence Theory and Web3 integration",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize blockchain
try:
    from blockchain.helical_chain import helical_blockchain
    print("✅ Helical blockchain initialized")
except Exception as e:
    print(f"⚠️  Could not initialize blockchain: {e}")
    helical_blockchain = None

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirect to dashboard"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IPAI Dashboard</title>
        <meta http-equiv="refresh" content="0; url=/dashboard">
        <style>
            body { font-family: Arial, sans-serif; text-align: center; padding: 50px; background: #0f0f23; color: white; }
            .logo { font-size: 2em; margin-bottom: 20px; }
            .loading { animation: pulse 2s infinite; }
            @keyframes pulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
        </style>
    </head>
    <body>
        <div class="logo">🧬 IPAI Dashboard</div>
        <div class="loading">Loading Helical Blockchain...</div>
        <p>If you are not redirected, <a href="/dashboard" style="color: #6366f1;">click here</a></p>
    </body>
    </html>
    """

@app.get("/dashboard")
async def dashboard():
    """Serve the main dashboard"""
    dashboard_path = Path(__file__).parent / "dashboard" / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        raise HTTPException(status_code=404, detail="Dashboard not found")

# Mount static files for dashboard assets
dashboard_dir = Path(__file__).parent / "dashboard"
if dashboard_dir.exists():
    app.mount("/dashboard", StaticFiles(directory=str(dashboard_dir), html=True), name="dashboard")
    print("✅ Dashboard mounted at /dashboard")
else:
    print("⚠️  Dashboard directory not found")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="1.0.0"
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Simple chat endpoint using Ollama"""
    try:
        # Import Ollama client
        from simple_ollama_test import OllamaClient
        
        # Create client
        client = OllamaClient()
        
        # Check if model is available
        model_available = await client.check_model()
        if not model_available:
            raise HTTPException(
                status_code=503, 
                detail="Ollama model not available. Please run: ollama pull llama3.2"
            )
        
        # Generate response with GCT context
        gct_prompt = f"""You are a coherence-focused AI assistant that helps users build internal consistency, accumulated wisdom, moral activation, and social belonging.

Consider the user's message and provide a thoughtful, supportive response that:
1. Promotes internal consistency (alignment between thoughts and actions)
2. Draws on wisdom and experience
3. Encourages values-aligned action
4. Fosters connection and belonging

User: {request.message}"""
        
        # Generate response
        response_text = await client.generate(gct_prompt, max_tokens=200)
        
        # Close client
        await client.close()
        
        return ChatResponse(
            response=response_text,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/api/v1/status")
async def system_status():
    """Get system status including Ollama connectivity"""
    try:
        from simple_ollama_test import OllamaClient
        
        client = OllamaClient()
        model_available = await client.check_model()
        await client.close()
        
        return {
            "system": "IPAI",
            "status": "operational",
            "ollama_connected": model_available,
            "llm_model": "llama3.2:latest" if model_available else "unavailable",
            "blockchain_active": helical_blockchain is not None,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "system": "IPAI",
            "status": "degraded",
            "ollama_connected": False,
            "blockchain_active": False,
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

# Blockchain endpoints
@app.post("/api/v1/blockchain/log")
async def log_to_blockchain(request: BlockchainLogRequest, background_tasks: BackgroundTasks):
    """Log interaction to helical blockchain"""
    if not helical_blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not available")
    
    try:
        # Add interaction to blockchain
        transaction_id = helical_blockchain.add_interaction(
            user_id=request.user_id,
            interaction_type=request.interaction_type,
            data=request.data,
            coherence_profile=None  # Simplified for now
        )
        
        # Background mining
        background_tasks.add_task(mine_pending_transactions)
        
        return {
            "status": "success",
            "transaction_id": transaction_id,
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain error: {str(e)}")

@app.get("/api/v1/blockchain/state")
async def get_blockchain_state():
    """Get current blockchain state"""
    if not helical_blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not available")
    
    try:
        return helical_blockchain.get_blockchain_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain error: {str(e)}")

@app.get("/api/v1/blockchain/visualization")
async def get_blockchain_visualization():
    """Get blockchain visualization data"""
    if not helical_blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not available")
    
    try:
        return helical_blockchain.get_helical_visualization_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain error: {str(e)}")

@app.get("/api/v1/blockchain/export")
async def export_blockchain():
    """Export blockchain data"""
    if not helical_blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not available")
    
    try:
        state = helical_blockchain.get_blockchain_state()
        viz_data = helical_blockchain.get_helical_visualization_data()
        
        return {
            "export_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "network": "IPAI Helical Blockchain"
            },
            "blockchain_state": state,
            "visualization_data": viz_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@app.get("/api/v1/blockchain/latest")
async def get_latest_block():
    """Get latest block from blockchain"""
    if not helical_blockchain:
        raise HTTPException(status_code=503, detail="Blockchain not available")
    
    try:
        latest_block = None
        latest_timestamp = None
        
        for strand, blocks in helical_blockchain.strands.items():
            if blocks:
                block = blocks[-1]
                if latest_timestamp is None or block.timestamp > latest_timestamp:
                    latest_block = block
                    latest_timestamp = block.timestamp
        
        if latest_block:
            return {
                "hash": latest_block.block_hash,
                "index": latest_block.index,
                "strand": latest_block.strand.value,
                "timestamp": latest_block.timestamp.isoformat(),
                "coherence_score": latest_block.coherence_score
            }
        else:
            raise HTTPException(status_code=404, detail="No blocks found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blockchain error: {str(e)}")

async def mine_pending_transactions():
    """Background task to mine pending transactions"""
    try:
        if helical_blockchain and helical_blockchain.pending_transactions:
            await helical_blockchain._mine_pending_transactions()
    except Exception as e:
        print(f"Mining error: {e}")

if __name__ == "__main__":
    print("🚀 Starting IPAI Dashboard with Helical Blockchain...")
    print("🎛️  Dashboard: http://localhost:8000/dashboard")
    print("🌐 Server: http://localhost:8000")
    print("📊 API Docs: http://localhost:8000/docs")
    print("💬 Chat: POST /api/v1/chat")
    print("📈 Status: GET /api/v1/status")
    print("⛓️  Blockchain: GET /api/v1/blockchain/state")
    print("🧬 Visualization: GET /api/v1/blockchain/visualization")
    print("⏹️  Press Ctrl+C to stop\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )