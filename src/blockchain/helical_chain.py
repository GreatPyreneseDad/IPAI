"""
Helical Blockchain Implementation

A novel blockchain architecture that uses helical (spiral) geometry to organize blocks
across multiple intertwined strands, with GCT coherence determining positioning.

The helical structure provides:
- Multi-dimensional block organization
- Cross-strand validation
- Coherence-based positioning
- Enhanced security through geometric complexity
"""

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np

from ..models.coherence_profile import CoherenceProfile, GCTComponents


class StrandType(Enum):
    """Types of helical strands"""
    COHERENCE = "coherence"  # GCT coherence data
    INTERACTION = "interaction"  # User interactions
    WISDOM = "wisdom"  # Learning and insights
    MORAL = "moral"  # Moral/ethical decisions
    SOCIAL = "social"  # Social connections


@dataclass
class HelicalPosition:
    """3D position in helical space"""
    theta: float  # Angular position (0-2π)
    z: float      # Height along helix axis
    radius: float # Distance from central axis
    strand: StrandType
    coherence_influence: float = 0.0


@dataclass
class HelicalBlock:
    """Block in helical blockchain"""
    index: int
    timestamp: datetime
    data: Dict[str, Any]
    previous_hash: str
    merkle_root: str
    nonce: int
    difficulty: int
    
    # Helical-specific fields
    position: HelicalPosition
    strand: StrandType
    coherence_score: float
    cross_strand_refs: List[str]  # References to blocks in other strands
    helix_signature: str
    
    # Computed fields
    block_hash: str = ""
    validation_score: float = 0.0


@dataclass
class CrossStrandLink:
    """Link between blocks in different strands"""
    source_block: str
    target_block: str
    source_strand: StrandType
    target_strand: StrandType
    link_strength: float
    coherence_delta: float
    timestamp: datetime


class HelicalBlockchain:
    """Helical blockchain implementation with GCT integration"""
    
    def __init__(self, 
                 helix_radius: float = 10.0,
                 helix_pitch: float = 5.0,
                 num_strands: int = 5):
        
        self.helix_radius = helix_radius
        self.helix_pitch = helix_pitch
        self.num_strands = num_strands
        
        # Blockchain state
        self.strands: Dict[StrandType, List[HelicalBlock]] = {
            strand: [] for strand in StrandType
        }
        self.cross_links: List[CrossStrandLink] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        
        # Helical parameters
        self.current_height = 0.0
        self.angular_offset = 2 * math.pi / num_strands
        
        # Mining parameters
        self.target_block_time = 10  # seconds
        self.difficulty_adjustment_interval = 10
        self.current_difficulty = 4
        
        # Initialize genesis blocks
        self._create_genesis_blocks()
    
    def _create_genesis_blocks(self):
        """Create genesis blocks for each strand"""
        genesis_time = datetime.utcnow()
        
        for i, strand in enumerate(StrandType):
            # Calculate initial position
            theta = i * self.angular_offset
            position = HelicalPosition(
                theta=theta,
                z=0.0,
                radius=self.helix_radius,
                strand=strand,
                coherence_influence=0.5
            )
            
            # Create genesis block
            genesis_data = {
                "type": "genesis",
                "strand": strand.value,
                "message": f"Genesis block for {strand.value} strand",
                "network_id": "ipai_helical_testnet"
            }
            
            genesis_block = HelicalBlock(
                index=0,
                timestamp=genesis_time,
                data=genesis_data,
                previous_hash="0",
                merkle_root=self._calculate_merkle_root([genesis_data]),
                nonce=0,
                difficulty=1,
                position=position,
                strand=strand,
                coherence_score=0.5,
                cross_strand_refs=[],
                helix_signature="genesis"
            )
            
            # Calculate hash
            genesis_block.block_hash = self._calculate_block_hash(genesis_block)
            self.strands[strand].append(genesis_block)
    
    def add_interaction(self, 
                       user_id: str,
                       interaction_type: str,
                       data: Dict[str, Any],
                       coherence_profile: Optional[CoherenceProfile] = None) -> str:
        """Add an interaction to the blockchain"""
        
        # Determine strand based on interaction type
        strand = self._determine_strand(interaction_type, data)
        
        # Calculate coherence influence
        coherence_score = 0.5
        if coherence_profile:
            coherence_score = coherence_profile.coherence_score
        
        # Create transaction
        transaction = {
            "type": interaction_type,
            "user_id": user_id,
            "data": data,
            "coherence_score": coherence_score,
            "timestamp": datetime.utcnow().isoformat(),
            "strand": strand.value
        }
        
        self.pending_transactions.append(transaction)
        
        # Auto-mine if enough transactions
        if len(self.pending_transactions) >= 3:
            asyncio.create_task(self._mine_pending_transactions())
        
        return f"tx_{int(time.time())}_{hash(str(transaction))}"
    
    def _determine_strand(self, interaction_type: str, data: Dict[str, Any]) -> StrandType:
        """Determine which strand an interaction belongs to"""
        
        if interaction_type in ["coherence_assessment", "gct_calculation"]:
            return StrandType.COHERENCE
        elif interaction_type in ["chat", "query", "response"]:
            return StrandType.INTERACTION
        elif interaction_type in ["learning", "insight", "reflection"]:
            return StrandType.WISDOM
        elif interaction_type in ["moral_decision", "ethical_choice", "values_alignment"]:
            return StrandType.MORAL
        elif interaction_type in ["social_connection", "community", "belonging"]:
            return StrandType.SOCIAL
        else:
            return StrandType.INTERACTION  # Default
    
    async def _mine_pending_transactions(self):
        """Mine pending transactions into new blocks"""
        
        if not self.pending_transactions:
            return
        
        # Group transactions by strand
        strand_transactions = {}
        for tx in self.pending_transactions:
            strand = StrandType(tx["strand"])
            if strand not in strand_transactions:
                strand_transactions[strand] = []
            strand_transactions[strand].append(tx)
        
        # Mine blocks for each strand with transactions
        for strand, transactions in strand_transactions.items():
            await self._mine_block(strand, transactions)
        
        # Clear pending transactions
        self.pending_transactions.clear()
        
        # Create cross-strand links
        await self._create_cross_strand_links()
    
    async def _mine_block(self, strand: StrandType, transactions: List[Dict[str, Any]]):
        """Mine a new block for a specific strand"""
        
        # Get previous block
        previous_blocks = self.strands[strand]
        previous_block = previous_blocks[-1] if previous_blocks else None
        previous_hash = previous_block.block_hash if previous_block else "0"
        
        # Calculate new helical position
        position = self._calculate_next_position(strand, transactions)
        
        # Calculate coherence score for block
        coherence_scores = [tx.get("coherence_score", 0.5) for tx in transactions]
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        
        # Create new block
        new_block = HelicalBlock(
            index=len(previous_blocks),
            timestamp=datetime.utcnow(),
            data={"transactions": transactions},
            previous_hash=previous_hash,
            merkle_root=self._calculate_merkle_root(transactions),
            nonce=0,
            difficulty=self.current_difficulty,
            position=position,
            strand=strand,
            coherence_score=avg_coherence,
            cross_strand_refs=[],
            helix_signature=""
        )
        
        # Mine the block (find valid nonce)
        await self._proof_of_coherence(new_block)
        
        # Add to blockchain
        self.strands[strand].append(new_block)
        
        print(f"⛏️  Mined block {new_block.index} on {strand.value} strand at θ={new_block.position.theta:.2f}, z={new_block.position.z:.2f}")
    
    def _calculate_next_position(self, strand: StrandType, transactions: List[Dict[str, Any]]) -> HelicalPosition:
        """Calculate helical position for next block"""
        
        # Base angular position for this strand
        strand_index = list(StrandType).index(strand)
        base_theta = strand_index * self.angular_offset
        
        # Calculate height progression
        self.current_height += self.helix_pitch / len(StrandType)
        
        # Coherence influence on position
        coherence_scores = [tx.get("coherence_score", 0.5) for tx in transactions]
        avg_coherence = sum(coherence_scores) / len(coherence_scores)
        
        # Theta varies based on coherence (higher coherence = tighter spiral)
        coherence_influence = (avg_coherence - 0.5) * 0.3  # ±0.15 radians max
        theta = (base_theta + coherence_influence + self.current_height * 0.1) % (2 * math.pi)
        
        # Radius varies with coherence (higher coherence = closer to center)
        radius = self.helix_radius * (1.2 - avg_coherence * 0.4)
        
        return HelicalPosition(
            theta=theta,
            z=self.current_height,
            radius=radius,
            strand=strand,
            coherence_influence=coherence_influence
        )
    
    async def _proof_of_coherence(self, block: HelicalBlock):
        """Proof of Coherence mining algorithm"""
        
        target = "0" * block.difficulty
        start_time = time.time()
        
        while True:
            # Calculate hash
            block.block_hash = self._calculate_block_hash(block)
            
            # Check if hash meets difficulty target
            if block.block_hash.startswith(target):
                # Additional coherence check
                coherence_validation = self._validate_coherence_proof(block)
                if coherence_validation > 0.7:  # Coherence threshold
                    block.validation_score = coherence_validation
                    break
            
            block.nonce += 1
            
            # Prevent infinite loops in testing
            if time.time() - start_time > 2:  # 2 second timeout for demo
                block.validation_score = 0.6
                break
            
            # Yield control occasionally
            if block.nonce % 1000 == 0:
                await asyncio.sleep(0.001)
        
        # Generate helix signature
        block.helix_signature = self._generate_helix_signature(block)
    
    def _validate_coherence_proof(self, block: HelicalBlock) -> float:
        """Validate coherence aspects of the proof"""
        
        # Base validation from coherence score
        coherence_factor = block.coherence_score
        
        # Position consistency factor
        expected_theta = self._calculate_expected_theta(block.strand, block.index)
        theta_diff = abs(block.position.theta - expected_theta)
        position_factor = max(0, 1 - theta_diff / math.pi)
        
        # Geometric harmony factor (golden ratio influence)
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio
        harmony_factor = abs(math.sin(block.position.theta * phi)) * 0.3 + 0.7
        
        return (coherence_factor * 0.5 + position_factor * 0.3 + harmony_factor * 0.2)
    
    def _calculate_expected_theta(self, strand: StrandType, index: int) -> float:
        """Calculate expected theta for a block"""
        strand_index = list(StrandType).index(strand)
        base_theta = strand_index * self.angular_offset
        return (base_theta + index * 0.1) % (2 * math.pi)
    
    async def _create_cross_strand_links(self):
        """Create links between related blocks in different strands"""
        
        # Find recent blocks from each strand
        recent_blocks = {}
        for strand, blocks in self.strands.items():
            if blocks:
                recent_blocks[strand] = blocks[-1]
        
        # Create links based on coherence similarity
        strands = list(recent_blocks.keys())
        for i in range(len(strands)):
            for j in range(i + 1, len(strands)):
                strand1, strand2 = strands[i], strands[j]
                block1, block2 = recent_blocks[strand1], recent_blocks[strand2]
                
                # Calculate link strength based on coherence similarity
                coherence_delta = abs(block1.coherence_score - block2.coherence_score)
                link_strength = max(0, 1 - coherence_delta)
                
                if link_strength > 0.5:  # Only create strong links
                    link = CrossStrandLink(
                        source_block=block1.block_hash,
                        target_block=block2.block_hash,
                        source_strand=strand1,
                        target_strand=strand2,
                        link_strength=link_strength,
                        coherence_delta=coherence_delta,
                        timestamp=datetime.utcnow()
                    )
                    
                    self.cross_links.append(link)
                    
                    # Add cross-references to blocks
                    block1.cross_strand_refs.append(block2.block_hash)
                    block2.cross_strand_refs.append(block1.block_hash)
    
    def _calculate_block_hash(self, block: HelicalBlock) -> str:
        """Calculate hash for a block"""
        
        # Create hashable content
        content = {
            "index": block.index,
            "timestamp": block.timestamp.isoformat(),
            "data": json.dumps(block.data, sort_keys=True),
            "previous_hash": block.previous_hash,
            "merkle_root": block.merkle_root,
            "nonce": block.nonce,
            "position": {
                "theta": round(block.position.theta, 6),
                "z": round(block.position.z, 6),
                "radius": round(block.position.radius, 6)
            },
            "strand": block.strand.value,
            "coherence_score": round(block.coherence_score, 6)
        }
        
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _calculate_merkle_root(self, transactions: List[Dict[str, Any]]) -> str:
        """Calculate Merkle root of transactions"""
        
        if not transactions:
            return hashlib.sha256(b"empty").hexdigest()
        
        # Hash each transaction
        tx_hashes = []
        for tx in transactions:
            tx_str = json.dumps(tx, sort_keys=True)
            tx_hash = hashlib.sha256(tx_str.encode()).hexdigest()
            tx_hashes.append(tx_hash)
        
        # Build Merkle tree
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 == 1:
                tx_hashes.append(tx_hashes[-1])  # Duplicate last hash if odd
            
            new_level = []
            for i in range(0, len(tx_hashes), 2):
                combined = tx_hashes[i] + tx_hashes[i + 1]
                new_hash = hashlib.sha256(combined.encode()).hexdigest()
                new_level.append(new_hash)
            
            tx_hashes = new_level
        
        return tx_hashes[0]
    
    def _generate_helix_signature(self, block: HelicalBlock) -> str:
        """Generate geometric signature based on helical position"""
        
        pos = block.position
        
        # Create signature from geometric properties
        signature_data = {
            "spiral_phase": math.sin(pos.theta) * math.cos(pos.z / self.helix_pitch),
            "radial_component": pos.radius / self.helix_radius,
            "height_ratio": pos.z / (self.helix_pitch * 2),
            "coherence_influence": pos.coherence_influence,
            "golden_ratio_phase": math.sin(pos.theta * ((1 + math.sqrt(5)) / 2))
        }
        
        sig_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(sig_str.encode()).hexdigest()[:16]
    
    def get_blockchain_state(self) -> Dict[str, Any]:
        """Get current state of the helical blockchain"""
        
        state = {
            "network_info": {
                "name": "IPAI Helical Blockchain",
                "total_blocks": sum(len(blocks) for blocks in self.strands.values()),
                "total_strands": len(self.strands),
                "current_height": self.current_height,
                "current_difficulty": self.current_difficulty,
                "cross_links": len(self.cross_links)
            },
            "strands": {},
            "recent_cross_links": self.cross_links[-5:] if self.cross_links else [],
            "helical_parameters": {
                "radius": self.helix_radius,
                "pitch": self.helix_pitch,
                "angular_offset": self.angular_offset
            }
        }
        
        # Add strand information
        for strand, blocks in self.strands.items():
            latest_block = blocks[-1] if blocks else None
            state["strands"][strand.value] = {
                "block_count": len(blocks),
                "latest_block": {
                    "index": latest_block.index,
                    "hash": latest_block.block_hash[:16] + "...",
                    "coherence_score": latest_block.coherence_score,
                    "position": asdict(latest_block.position),
                    "timestamp": latest_block.timestamp.isoformat()
                } if latest_block else None
            }
        
        return state
    
    def get_strand_blocks(self, strand: StrandType, limit: int = 10) -> List[Dict[str, Any]]:
        """Get blocks from a specific strand"""
        
        blocks = self.strands.get(strand, [])
        recent_blocks = blocks[-limit:] if blocks else []
        
        return [
            {
                "index": block.index,
                "hash": block.block_hash,
                "timestamp": block.timestamp.isoformat(),
                "coherence_score": block.coherence_score,
                "position": asdict(block.position),
                "data_summary": self._summarize_block_data(block.data),
                "cross_strand_refs": len(block.cross_strand_refs),
                "validation_score": block.validation_score
            }
            for block in recent_blocks
        ]
    
    def _summarize_block_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of block data for API response"""
        
        if "transactions" in data:
            txs = data["transactions"]
            return {
                "transaction_count": len(txs),
                "types": list(set(tx.get("type", "unknown") for tx in txs)),
                "avg_coherence": sum(tx.get("coherence_score", 0) for tx in txs) / len(txs) if txs else 0
            }
        else:
            return {"type": data.get("type", "unknown")}
    
    def get_helical_visualization_data(self) -> Dict[str, Any]:
        """Get data for 3D helical visualization"""
        
        visualization_data = {
            "strands": {},
            "cross_links": [],
            "parameters": {
                "radius": self.helix_radius,
                "pitch": self.helix_pitch,
                "height": self.current_height
            }
        }
        
        # Add block positions for each strand
        for strand, blocks in self.strands.items():
            strand_data = []
            for block in blocks:
                pos = block.position
                strand_data.append({
                    "index": block.index,
                    "x": pos.radius * math.cos(pos.theta),
                    "y": pos.radius * math.sin(pos.theta),
                    "z": pos.z,
                    "theta": pos.theta,
                    "radius": pos.radius,
                    "coherence": block.coherence_score,
                    "hash": block.block_hash[:8],
                    "timestamp": block.timestamp.isoformat()
                })
            
            visualization_data["strands"][strand.value] = strand_data
        
        # Add cross-strand links
        for link in self.cross_links[-20:]:  # Last 20 links
            visualization_data["cross_links"].append({
                "source": link.source_block[:8],
                "target": link.target_block[:8],
                "source_strand": link.source_strand.value,
                "target_strand": link.target_strand.value,
                "strength": link.link_strength,
                "coherence_delta": link.coherence_delta
            })
        
        return visualization_data


# Global helical blockchain instance
helical_blockchain = HelicalBlockchain()