#!/usr/bin/env python3
"""
Personal Blockchain for IPAI - Individual user chains for coherence tracking
Each user maintains their own blockchain that syncs with the public ledger
"""
import hashlib
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import asyncio
from web3 import Web3
import sqlite3
import os


class InferenceStatus(Enum):
    """Status of inference verification"""
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONSENSUS = "consensus"


@dataclass
class PersonalBlock:
    """Block structure for personal IPAI blockchain"""
    index: int
    timestamp: float
    user_id: str
    ipai_id: str
    
    # Coherence data
    coherence_score: float
    soul_echo: float
    safety_score: float
    
    # Interaction data
    interaction_hash: str  # Hash of user input + AI response
    inference_data: Dict[str, Any]
    
    # Chain data
    previous_hash: str
    nonce: int = 0
    hash: str = ""
    
    # Verification data
    local_signature: str = ""
    ipai_signature: str = ""
    verification_status: InferenceStatus = InferenceStatus.PENDING
    
    def calculate_hash(self) -> str:
        """Calculate block hash"""
        block_data = {
            'index': self.index,
            'timestamp': self.timestamp,
            'user_id': self.user_id,
            'ipai_id': self.ipai_id,
            'coherence_score': self.coherence_score,
            'soul_echo': self.soul_echo,
            'safety_score': self.safety_score,
            'interaction_hash': self.interaction_hash,
            'inference_data': json.dumps(self.inference_data, sort_keys=True),
            'previous_hash': self.previous_hash,
            'nonce': self.nonce
        }
        
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty: int = 4):
        """Mine block with proof of coherence"""
        target = '0' * difficulty
        
        while True:
            self.hash = self.calculate_hash()
            
            # Proof of Coherence: hash must start with zeros AND coherence must be above threshold
            if self.hash[:difficulty] == target and self.coherence_score > 0.3:
                break
                
            self.nonce += 1
            
            # Coherence bonus: reduce difficulty for high coherence scores
            if self.coherence_score > 0.8 and difficulty > 2:
                difficulty = 2


@dataclass 
class InferenceRecord:
    """Record of an inference for verification"""
    timestamp: float
    block_index: int
    inference_type: str
    inference_content: Dict[str, Any]
    confidence_score: float
    coherence_context: Dict[str, float]
    verification_hash: str
    status: InferenceStatus = InferenceStatus.PENDING
    
    def calculate_verification_hash(self) -> str:
        """Calculate hash for inference verification"""
        data = {
            'timestamp': self.timestamp,
            'inference_type': self.inference_type,
            'inference_content': json.dumps(self.inference_content, sort_keys=True),
            'confidence_score': self.confidence_score,
            'coherence_context': json.dumps(self.coherence_context, sort_keys=True)
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class PersonalBlockchain:
    """
    Personal blockchain for individual IPAI instance.
    Maintains local chain and syncs verified inferences to public ledger.
    """
    
    def __init__(self, user_id: str, ipai_id: str, db_path: Optional[str] = None):
        self.user_id = user_id
        self.ipai_id = ipai_id
        self.chain: List[PersonalBlock] = []
        self.pending_inferences: List[InferenceRecord] = []
        self.verified_inferences: List[InferenceRecord] = []
        
        # Database for persistence
        self.db_path = db_path or f"personal_chain_{user_id}.db"
        self._init_database()
        
        # Mining parameters
        self.base_difficulty = 4
        self.coherence_threshold = 0.3
        
        # Public ledger connection (to be initialized)
        self.public_ledger_contract = None
        self.web3 = None
        
        # Create genesis block if chain is empty
        if not self.load_chain():
            self.create_genesis_block()
    
    def _init_database(self):
        """Initialize SQLite database for chain persistence"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Blocks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blocks (
                index INTEGER PRIMARY KEY,
                timestamp REAL,
                user_id TEXT,
                ipai_id TEXT,
                coherence_score REAL,
                soul_echo REAL,
                safety_score REAL,
                interaction_hash TEXT,
                inference_data TEXT,
                previous_hash TEXT,
                nonce INTEGER,
                hash TEXT,
                local_signature TEXT,
                ipai_signature TEXT,
                verification_status TEXT
            )
        ''')
        
        # Inferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                block_index INTEGER,
                inference_type TEXT,
                inference_content TEXT,
                confidence_score REAL,
                coherence_context TEXT,
                verification_hash TEXT,
                status TEXT,
                FOREIGN KEY (block_index) REFERENCES blocks(index)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_genesis_block(self):
        """Create the genesis block for personal chain"""
        genesis = PersonalBlock(
            index=0,
            timestamp=time.time(),
            user_id=self.user_id,
            ipai_id=self.ipai_id,
            coherence_score=1.0,
            soul_echo=1.0,
            safety_score=1.0,
            interaction_hash="genesis",
            inference_data={"type": "genesis", "message": "Personal IPAI chain initialized"},
            previous_hash="0"
        )
        
        genesis.mine_block(self.base_difficulty)
        self.chain.append(genesis)
        self._save_block(genesis)
        
        return genesis
    
    async def add_interaction(
        self,
        user_input: str,
        ai_response: str,
        coherence_score: float,
        soul_echo: float,
        safety_score: float,
        inference_data: Optional[Dict[str, Any]] = None
    ) -> PersonalBlock:
        """
        Add new interaction to personal blockchain.
        
        Args:
            user_input: User's input text
            ai_response: AI's response
            coherence_score: Current coherence score
            soul_echo: Soul echo metric
            safety_score: Safety assessment score
            inference_data: Optional inference data to record
            
        Returns:
            New mined block
        """
        # Create interaction hash
        interaction_content = f"{user_input}|{ai_response}|{time.time()}"
        interaction_hash = hashlib.sha256(interaction_content.encode()).hexdigest()
        
        # Prepare inference data
        if inference_data is None:
            inference_data = {
                "type": "interaction",
                "user_input_length": len(user_input),
                "response_length": len(ai_response),
                "timestamp": time.time()
            }
        
        # Create new block
        previous_block = self.get_latest_block()
        new_block = PersonalBlock(
            index=len(self.chain),
            timestamp=time.time(),
            user_id=self.user_id,
            ipai_id=self.ipai_id,
            coherence_score=coherence_score,
            soul_echo=soul_echo,
            safety_score=safety_score,
            interaction_hash=interaction_hash,
            inference_data=inference_data,
            previous_hash=previous_block.hash
        )
        
        # Adjust difficulty based on coherence
        difficulty = self._calculate_difficulty(coherence_score)
        new_block.mine_block(difficulty)
        
        # Sign block
        new_block.local_signature = self._sign_block(new_block)
        new_block.ipai_signature = await self._get_ipai_signature(new_block)
        
        # Add to chain
        self.chain.append(new_block)
        self._save_block(new_block)
        
        # Process any inferences
        if "inference" in inference_data:
            await self.record_inference(
                block_index=new_block.index,
                inference_type=inference_data["inference"]["type"],
                inference_content=inference_data["inference"]["content"],
                confidence_score=inference_data["inference"].get("confidence", 0.8)
            )
        
        return new_block
    
    async def record_inference(
        self,
        block_index: int,
        inference_type: str,
        inference_content: Dict[str, Any],
        confidence_score: float
    ) -> InferenceRecord:
        """
        Record an inference for later verification.
        
        Args:
            block_index: Index of block containing this inference
            inference_type: Type of inference (prediction, analysis, recommendation)
            inference_content: The actual inference data
            confidence_score: Confidence level of inference
            
        Returns:
            InferenceRecord object
        """
        # Get coherence context from block
        block = self.chain[block_index]
        coherence_context = {
            "coherence_score": block.coherence_score,
            "soul_echo": block.soul_echo,
            "safety_score": block.safety_score
        }
        
        # Create inference record
        inference = InferenceRecord(
            timestamp=time.time(),
            block_index=block_index,
            inference_type=inference_type,
            inference_content=inference_content,
            confidence_score=confidence_score,
            coherence_context=coherence_context,
            verification_hash=""
        )
        
        inference.verification_hash = inference.calculate_verification_hash()
        
        # Add to pending
        self.pending_inferences.append(inference)
        self._save_inference(inference)
        
        # Auto-verify if confidence and coherence are high
        if confidence_score > 0.9 and block.coherence_score > 0.8:
            await self.verify_inference(inference)
        
        return inference
    
    async def verify_inference(self, inference: InferenceRecord) -> bool:
        """
        Verify an inference with IPAI and prepare for public ledger.
        
        Args:
            inference: Inference to verify
            
        Returns:
            True if verified successfully
        """
        # Check verification criteria
        block = self.chain[inference.block_index]
        
        # Verification requires:
        # 1. High coherence at time of inference
        # 2. Consistent safety score
        # 3. Confidence above threshold
        
        verified = (
            inference.coherence_context["coherence_score"] > 0.6 and
            inference.coherence_context["safety_score"] > 0.7 and
            inference.confidence_score > 0.7
        )
        
        if verified:
            inference.status = InferenceStatus.VERIFIED
            self.verified_inferences.append(inference)
            self.pending_inferences.remove(inference)
            
            # Update in database
            self._update_inference_status(inference)
            
            # Prepare for public ledger submission
            if self.web3 and self.public_ledger_contract:
                await self._submit_to_public_ledger(inference)
            
            return True
        else:
            inference.status = InferenceStatus.REJECTED
            self._update_inference_status(inference)
            return False
    
    async def collaborate_verification(
        self, 
        other_ipai_id: str,
        inference_hash: str
    ) -> bool:
        """
        Collaborate with another IPAI for inference verification.
        
        Args:
            other_ipai_id: ID of collaborating IPAI
            inference_hash: Hash of inference to verify
            
        Returns:
            True if collaborative verification successful
        """
        # Find inference by hash
        inference = next(
            (i for i in self.pending_inferences if i.verification_hash == inference_hash),
            None
        )
        
        if not inference:
            return False
        
        # In production, this would involve P2P communication
        # For now, simulate collaborative verification
        
        # Check if both IPAIs have high coherence
        my_coherence = self.chain[inference.block_index].coherence_score
        
        # Simulated: assume other IPAI has similar coherence
        other_coherence = my_coherence * 0.95  # Slight variation
        
        if my_coherence > 0.7 and other_coherence > 0.7:
            inference.status = InferenceStatus.CONSENSUS
            self.verified_inferences.append(inference)
            self.pending_inferences.remove(inference)
            self._update_inference_status(inference)
            
            return True
        
        return False
    
    def _calculate_difficulty(self, coherence_score: float) -> int:
        """Calculate mining difficulty based on coherence"""
        # Higher coherence = easier mining (reward good coherence)
        if coherence_score > 0.9:
            return max(1, self.base_difficulty - 2)
        elif coherence_score > 0.7:
            return max(2, self.base_difficulty - 1)
        elif coherence_score < 0.3:
            return self.base_difficulty + 2
        else:
            return self.base_difficulty
    
    def _sign_block(self, block: PersonalBlock) -> str:
        """Generate local signature for block"""
        # In production, use actual cryptographic signing
        sign_data = f"{block.hash}|{self.user_id}|{self.ipai_id}"
        return hashlib.sha256(sign_data.encode()).hexdigest()
    
    async def _get_ipai_signature(self, block: PersonalBlock) -> str:
        """Get IPAI signature for block"""
        # In production, IPAI would sign with its key
        sign_data = f"{block.hash}|{self.ipai_id}|{block.coherence_score}"
        return hashlib.sha256(sign_data.encode()).hexdigest()
    
    async def _submit_to_public_ledger(self, inference: InferenceRecord):
        """Submit verified inference to public ledger"""
        # This would interact with the main blockchain
        # For now, we'll just log it
        print(f"Submitting inference {inference.verification_hash} to public ledger")
    
    def get_latest_block(self) -> PersonalBlock:
        """Get the most recent block"""
        return self.chain[-1]
    
    def validate_chain(self) -> bool:
        """Validate the entire chain"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Check hash
            if current.hash != current.calculate_hash():
                return False
            
            # Check previous hash link
            if current.previous_hash != previous.hash:
                return False
            
            # Check proof of work
            if not current.hash.startswith('0' * self._calculate_difficulty(current.coherence_score)):
                return False
        
        return True
    
    def get_coherence_history(self) -> List[Tuple[float, float]]:
        """Get coherence score history (timestamp, score)"""
        return [(block.timestamp, block.coherence_score) for block in self.chain]
    
    def get_inference_summary(self) -> Dict[str, Any]:
        """Get summary of inference activity"""
        return {
            "total_inferences": len(self.pending_inferences) + len(self.verified_inferences),
            "pending": len(self.pending_inferences),
            "verified": len(self.verified_inferences),
            "verification_rate": len(self.verified_inferences) / max(1, len(self.pending_inferences) + len(self.verified_inferences)),
            "recent_inferences": [
                {
                    "type": inf.inference_type,
                    "confidence": inf.confidence_score,
                    "status": inf.status.value,
                    "timestamp": inf.timestamp
                }
                for inf in (self.verified_inferences + self.pending_inferences)[-10:]
            ]
        }
    
    # Database operations
    def _save_block(self, block: PersonalBlock):
        """Save block to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO blocks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            block.index, block.timestamp, block.user_id, block.ipai_id,
            block.coherence_score, block.soul_echo, block.safety_score,
            block.interaction_hash, json.dumps(block.inference_data),
            block.previous_hash, block.nonce, block.hash,
            block.local_signature, block.ipai_signature, block.verification_status.value
        ))
        
        conn.commit()
        conn.close()
    
    def _save_inference(self, inference: InferenceRecord):
        """Save inference to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO inferences (timestamp, block_index, inference_type, 
                                  inference_content, confidence_score, coherence_context,
                                  verification_hash, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            inference.timestamp, inference.block_index, inference.inference_type,
            json.dumps(inference.inference_content), inference.confidence_score,
            json.dumps(inference.coherence_context), inference.verification_hash,
            inference.status.value
        ))
        
        conn.commit()
        conn.close()
    
    def _update_inference_status(self, inference: InferenceRecord):
        """Update inference status in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE inferences SET status = ? WHERE verification_hash = ?
        ''', (inference.status.value, inference.verification_hash))
        
        conn.commit()
        conn.close()
    
    def load_chain(self) -> bool:
        """Load chain from database"""
        if not os.path.exists(self.db_path):
            return False
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Load blocks
        cursor.execute('SELECT * FROM blocks ORDER BY index')
        rows = cursor.fetchall()
        
        if not rows:
            conn.close()
            return False
        
        self.chain = []
        for row in rows:
            block = PersonalBlock(
                index=row[0], timestamp=row[1], user_id=row[2], ipai_id=row[3],
                coherence_score=row[4], soul_echo=row[5], safety_score=row[6],
                interaction_hash=row[7], inference_data=json.loads(row[8]),
                previous_hash=row[9], nonce=row[10], hash=row[11],
                local_signature=row[12], ipai_signature=row[13],
                verification_status=InferenceStatus(row[14])
            )
            self.chain.append(block)
        
        # Load inferences
        cursor.execute('SELECT * FROM inferences')
        inference_rows = cursor.fetchall()
        
        for row in inference_rows:
            inference = InferenceRecord(
                timestamp=row[1], block_index=row[2], inference_type=row[3],
                inference_content=json.loads(row[4]), confidence_score=row[5],
                coherence_context=json.loads(row[6]), verification_hash=row[7],
                status=InferenceStatus(row[8])
            )
            
            if inference.status == InferenceStatus.VERIFIED:
                self.verified_inferences.append(inference)
            else:
                self.pending_inferences.append(inference)
        
        conn.close()
        return True
    
    def export_chain_data(self) -> Dict[str, Any]:
        """Export chain data for analysis"""
        return {
            "user_id": self.user_id,
            "ipai_id": self.ipai_id,
            "chain_length": len(self.chain),
            "total_interactions": len(self.chain) - 1,  # Minus genesis
            "average_coherence": np.mean([b.coherence_score for b in self.chain[1:]]),
            "average_safety": np.mean([b.safety_score for b in self.chain[1:]]),
            "inference_summary": self.get_inference_summary(),
            "chain_valid": self.validate_chain()
        }