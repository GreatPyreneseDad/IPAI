"""
Helical Blockchain API Endpoints

REST endpoints for interacting with the helical blockchain system,
logging interactions, and retrieving blockchain state data.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from datetime import datetime
import json

from ...blockchain.helical_chain import helical_blockchain, StrandType, HelicalBlockchain
from ...models.coherence_profile import CoherenceProfile, GCTComponents, IndividualParameters

router = APIRouter(prefix="/blockchain")

# Pydantic models for API

class BlockchainLogRequest(BaseModel):
    """Request model for logging interactions to blockchain"""
    interaction_type: str = Field(..., description="Type of interaction")
    data: Dict[str, Any] = Field(..., description="Interaction data")
    user_id: str = Field("anonymous", description="User identifier")
    coherence_profile: Optional[Dict[str, Any]] = Field(None, description="User coherence profile")

class BlockchainStateResponse(BaseModel):
    """Response model for blockchain state"""
    network_info: Dict[str, Any]
    strands: Dict[str, Any]
    recent_cross_links: List[Dict[str, Any]]
    helical_parameters: Dict[str, Any]

class StrandBlocksResponse(BaseModel):
    """Response model for strand blocks"""
    strand: str
    blocks: List[Dict[str, Any]]
    total_count: int

class VisualizationDataResponse(BaseModel):
    """Response model for visualization data"""
    strands: Dict[str, List[Dict[str, Any]]]
    cross_links: List[Dict[str, Any]]
    parameters: Dict[str, Any]

# Endpoints

@router.post("/log")
async def log_interaction(
    request: BlockchainLogRequest,
    background_tasks: BackgroundTasks
):
    """Log an interaction to the helical blockchain"""
    
    try:
        # Create coherence profile if provided
        coherence_profile = None
        if request.coherence_profile:
            try:
                profile_data = request.coherence_profile
                coherence_profile = CoherenceProfile(
                    user_id=request.user_id,
                    components=GCTComponents(
                        psi=profile_data.get('psi', 0.5),
                        rho=profile_data.get('rho', 0.5),
                        q=profile_data.get('q', 0.5),
                        f=profile_data.get('f', 0.5)
                    ),
                    parameters=IndividualParameters(
                        k_m=profile_data.get('k_m', 0.5),
                        k_i=profile_data.get('k_i', 2.0)
                    ),
                    timestamp=datetime.utcnow()
                )
            except Exception as e:
                print(f"Failed to create coherence profile: {e}")
        
        # Add interaction to blockchain
        transaction_id = helical_blockchain.add_interaction(
            user_id=request.user_id,
            interaction_type=request.interaction_type,
            data=request.data,
            coherence_profile=coherence_profile
        )
        
        # Background task to mine pending transactions
        background_tasks.add_task(mine_pending_transactions_task)
        
        return {
            "status": "success",
            "transaction_id": transaction_id,
            "message": "Interaction logged to helical blockchain",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to log interaction: {str(e)}"
        )

@router.get("/state", response_model=BlockchainStateResponse)
async def get_blockchain_state():
    """Get current state of the helical blockchain"""
    
    try:
        state = helical_blockchain.get_blockchain_state()
        return state
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get blockchain state: {str(e)}"
        )

@router.get("/strand/{strand_name}")
async def get_strand_blocks(
    strand_name: str,
    limit: int = 10
):
    """Get blocks from a specific strand"""
    
    try:
        # Validate strand name
        try:
            strand = StrandType(strand_name)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strand name: {strand_name}"
            )
        
        blocks = helical_blockchain.get_strand_blocks(strand, limit)
        
        return {
            "strand": strand_name,
            "blocks": blocks,
            "total_count": len(helical_blockchain.strands.get(strand, []))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get strand blocks: {str(e)}"
        )

@router.get("/visualization", response_model=VisualizationDataResponse)
async def get_visualization_data():
    """Get data for 3D helical visualization"""
    
    try:
        data = helical_blockchain.get_helical_visualization_data()
        return data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get visualization data: {str(e)}"
        )

@router.get("/export")
async def export_blockchain_data():
    """Export complete blockchain data"""
    
    try:
        # Get complete blockchain state
        state = helical_blockchain.get_blockchain_state()
        
        # Get all strand data
        all_strands = {}
        for strand in StrandType:
            strand_blocks = helical_blockchain.get_strand_blocks(strand, limit=1000)
            all_strands[strand.value] = strand_blocks
        
        # Get visualization data
        viz_data = helical_blockchain.get_helical_visualization_data()
        
        # Create comprehensive export
        export_data = {
            "export_info": {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0",
                "network": "IPAI Helical Blockchain",
                "total_blocks": state["network_info"]["total_blocks"],
                "total_strands": state["network_info"]["total_strands"],
                "current_height": state["network_info"]["current_height"]
            },
            "network_state": state,
            "strand_data": all_strands,
            "visualization_data": viz_data,
            "helical_parameters": {
                "radius": helical_blockchain.helix_radius,
                "pitch": helical_blockchain.helix_pitch,
                "num_strands": helical_blockchain.num_strands,
                "current_height": helical_blockchain.current_height,
                "difficulty": helical_blockchain.current_difficulty
            },
            "cross_links": [
                {
                    "source_block": link.source_block,
                    "target_block": link.target_block,
                    "source_strand": link.source_strand.value,
                    "target_strand": link.target_strand.value,
                    "link_strength": link.link_strength,
                    "coherence_delta": link.coherence_delta,
                    "timestamp": link.timestamp.isoformat()
                }
                for link in helical_blockchain.cross_links
            ]
        }
        
        return export_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to export blockchain data: {str(e)}"
        )

@router.get("/latest")
async def get_latest_block():
    """Get the latest block from any strand"""
    
    try:
        latest_block = None
        latest_timestamp = None
        
        # Find the most recent block across all strands
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
                "coherence_score": latest_block.coherence_score,
                "position": {
                    "theta": latest_block.position.theta,
                    "z": latest_block.position.z,
                    "radius": latest_block.position.radius
                }
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="No blocks found in blockchain"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get latest block: {str(e)}"
        )

@router.get("/metrics")
async def get_blockchain_metrics():
    """Get blockchain performance and coherence metrics"""
    
    try:
        # Calculate metrics
        total_blocks = sum(len(blocks) for blocks in helical_blockchain.strands.values())
        
        # Average coherence by strand
        strand_coherence = {}
        for strand, blocks in helical_blockchain.strands.items():
            if blocks:
                avg_coherence = sum(block.coherence_score for block in blocks) / len(blocks)
                strand_coherence[strand.value] = round(avg_coherence, 3)
            else:
                strand_coherence[strand.value] = 0.0
        
        # Cross-link analysis
        total_cross_links = len(helical_blockchain.cross_links)
        if helical_blockchain.cross_links:
            avg_link_strength = sum(link.link_strength for link in helical_blockchain.cross_links) / total_cross_links
        else:
            avg_link_strength = 0.0
        
        # Helical geometry metrics
        current_height = helical_blockchain.current_height
        helix_volume = 3.14159 * (helical_blockchain.helix_radius ** 2) * current_height
        
        return {
            "blockchain_metrics": {
                "total_blocks": total_blocks,
                "total_strands": len(helical_blockchain.strands),
                "total_cross_links": total_cross_links,
                "average_link_strength": round(avg_link_strength, 3),
                "current_height": round(current_height, 2),
                "helix_volume": round(helix_volume, 2),
                "difficulty": helical_blockchain.current_difficulty
            },
            "strand_coherence": strand_coherence,
            "network_health": {
                "strands_active": sum(1 for blocks in helical_blockchain.strands.values() if blocks),
                "cross_link_density": round(total_cross_links / max(1, total_blocks), 3),
                "height_efficiency": round(current_height / max(1, total_blocks), 3)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get blockchain metrics: {str(e)}"
        )

@router.post("/mine")
async def force_mine_blocks():
    """Force mining of pending transactions (for testing)"""
    
    try:
        if not helical_blockchain.pending_transactions:
            return {
                "status": "no_pending_transactions",
                "message": "No pending transactions to mine"
            }
        
        # Mine pending transactions
        await helical_blockchain._mine_pending_transactions()
        
        return {
            "status": "success",
            "message": "Pending transactions mined successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mine blocks: {str(e)}"
        )

@router.get("/search/{query}")
async def search_blockchain(query: str):
    """Search blockchain for blocks containing specific data"""
    
    try:
        results = []
        
        # Search through all blocks in all strands
        for strand, blocks in helical_blockchain.strands.items():
            for block in blocks:
                # Search in block data
                block_json = json.dumps(block.data).lower()
                if query.lower() in block_json or query.lower() in block.block_hash.lower():
                    results.append({
                        "block_hash": block.block_hash,
                        "strand": strand.value,
                        "index": block.index,
                        "timestamp": block.timestamp.isoformat(),
                        "coherence_score": block.coherence_score,
                        "data_summary": helical_blockchain._summarize_block_data(block.data),
                        "match_type": "data" if query.lower() in block_json else "hash"
                    })
        
        return {
            "query": query,
            "results": results,
            "total_matches": len(results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to search blockchain: {str(e)}"
        )

# Background tasks

async def mine_pending_transactions_task():
    """Background task to mine pending transactions"""
    try:
        if helical_blockchain.pending_transactions:
            await helical_blockchain._mine_pending_transactions()
    except Exception as e:
        print(f"Failed to mine pending transactions: {e}")

# Health check for blockchain

@router.get("/health")
async def blockchain_health():
    """Check blockchain health status"""
    
    try:
        total_blocks = sum(len(blocks) for blocks in helical_blockchain.strands.values())
        
        health_status = {
            "status": "healthy",
            "total_blocks": total_blocks,
            "pending_transactions": len(helical_blockchain.pending_transactions),
            "strands_active": sum(1 for blocks in helical_blockchain.strands.values() if blocks),
            "cross_links": len(helical_blockchain.cross_links),
            "current_height": helical_blockchain.current_height,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Determine health status
        if total_blocks == 0:
            health_status["status"] = "initializing"
        elif len(helical_blockchain.pending_transactions) > 10:
            health_status["status"] = "congested"
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }