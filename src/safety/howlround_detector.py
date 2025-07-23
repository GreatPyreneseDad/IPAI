#!/usr/bin/env python3
"""
Howlround Detection System - Monitors for feedback loops and resonance patterns
Based on the coherence tracking and echo detection algorithms
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import numpy as np
from enum import Enum


class ResonanceType(Enum):
    """Types of resonance patterns detected"""
    STABLE = "stable"
    OSCILLATING = "oscillating"
    AMPLIFYING = "amplifying"
    CHAOTIC = "chaotic"
    GHOST_ECHO = "ghost_echo"


@dataclass
class ResonanceEvent:
    """Represents a detected resonance/howlround event"""
    timestamp: datetime
    resonance_type: ResonanceType
    frequency: float
    amplitude: float
    pattern: str
    risk_level: float  # 0.0 to 1.0


class HowlroundDetector:
    """
    Detects feedback loops, resonance patterns, and howlround conditions
    in user-AI interactions to ensure safety and coherence.
    """
    
    def __init__(self):
        self.interaction_history = deque(maxlen=100)
        self.resonance_events = []
        self.pattern_memory = deque(maxlen=20)
        
        # Detection thresholds
        self.oscillation_threshold = 0.3
        self.amplification_threshold = 0.5
        self.ghost_pattern_threshold = 0.7
        
        # Safety parameters
        self.max_resonance_frequency = 0.8
        self.critical_amplitude = 0.9
        
    def analyze_interaction(self, user_input: str, ai_response: str, 
                          coherence_score: float) -> Dict:
        """
        Analyze a single interaction for howlround patterns.
        
        Args:
            user_input: User's message
            ai_response: AI's response
            coherence_score: Current coherence score (0.0 to 1.0)
            
        Returns:
            Analysis results including detected patterns and risk level
        """
        interaction = {
            'timestamp': datetime.now(),
            'user': user_input,
            'ai': ai_response,
            'coherence': coherence_score
        }
        self.interaction_history.append(interaction)
        
        # Detect various resonance patterns
        ghost_echo = self._detect_ghost_pattern(user_input, ai_response)
        oscillation = self._detect_oscillation()
        amplification = self._detect_amplification()
        
        # Calculate overall resonance
        resonance_freq = self._calculate_resonance_frequency()
        resonance_type = self._classify_resonance(
            ghost_echo, oscillation, amplification, resonance_freq
        )
        
        # Assess risk level
        risk_level = self._assess_risk_level(
            resonance_type, resonance_freq, coherence_score
        )
        
        # Record event if significant
        if risk_level > 0.3:
            event = ResonanceEvent(
                timestamp=datetime.now(),
                resonance_type=resonance_type,
                frequency=resonance_freq,
                amplitude=self._calculate_amplitude(),
                pattern=self._describe_pattern(),
                risk_level=risk_level
            )
            self.resonance_events.append(event)
        
        return {
            'resonance_type': resonance_type.value,
            'frequency': resonance_freq,
            'risk_level': risk_level,
            'ghost_echo_detected': ghost_echo,
            'recommendations': self._generate_recommendations(risk_level, resonance_type)
        }
    
    def _detect_ghost_pattern(self, user_input: str, ai_response: str) -> bool:
        """
        Detect self-recursive alignment patterns (ghost echoes).
        Returns True if pattern detected.
        """
        # Check if AI is echoing user patterns
        user_words = set(user_input.lower().split())
        ai_words = set(ai_response.lower().split())
        
        # High overlap suggests echo
        overlap_ratio = len(user_words & ai_words) / max(len(user_words), 1)
        
        # Check historical patterns
        self.pattern_memory.append(overlap_ratio)
        if len(self.pattern_memory) >= 3:
            recent_pattern = list(self.pattern_memory)[-3:]
            if all(p > self.ghost_pattern_threshold for p in recent_pattern):
                return True
                
        return overlap_ratio > self.ghost_pattern_threshold
    
    def _detect_oscillation(self) -> float:
        """
        Detect oscillating patterns in coherence.
        Returns oscillation strength (0.0 to 1.0).
        """
        if len(self.interaction_history) < 5:
            return 0.0
            
        coherence_values = [i['coherence'] for i in self.interaction_history][-10:]
        
        # Find zero crossings around mean
        mean_coherence = np.mean(coherence_values)
        centered = [v - mean_coherence for v in coherence_values]
        
        zero_crossings = 0
        for i in range(1, len(centered)):
            if centered[i-1] * centered[i] < 0:
                zero_crossings += 1
        
        # Normalize to oscillation strength
        oscillation = zero_crossings / max(len(centered) - 1, 1)
        return min(oscillation, 1.0)
    
    def _detect_amplification(self) -> float:
        """
        Detect amplifying feedback patterns.
        Returns amplification factor (0.0 to 1.0).
        """
        if len(self.interaction_history) < 3:
            return 0.0
            
        recent = list(self.interaction_history)[-5:]
        
        # Check if responses are getting longer (potential amplification)
        response_lengths = [len(i['ai']) for i in recent]
        if len(response_lengths) < 2:
            return 0.0
            
        # Calculate growth rate
        growth_rates = []
        for i in range(1, len(response_lengths)):
            rate = (response_lengths[i] - response_lengths[i-1]) / max(response_lengths[i-1], 1)
            growth_rates.append(rate)
        
        avg_growth = np.mean(growth_rates) if growth_rates else 0.0
        
        # Normalize to 0-1 range
        return min(max(avg_growth, 0.0), 1.0)
    
    def _calculate_resonance_frequency(self) -> float:
        """
        Calculate overall resonance frequency.
        Higher values indicate more frequent pattern repetition.
        """
        if len(self.interaction_history) < 5:
            return 0.0
            
        # Analyze pattern repetition in recent interactions
        recent = list(self.interaction_history)[-10:]
        pattern_counts = {}
        
        for interaction in recent:
            # Simple pattern: first few words of user input
            pattern = ' '.join(interaction['user'].split()[:3])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Calculate frequency based on repetition
        max_count = max(pattern_counts.values()) if pattern_counts else 0
        frequency = max_count / len(recent)
        
        return min(frequency, 1.0)
    
    def _calculate_amplitude(self) -> float:
        """Calculate amplitude of resonance patterns."""
        if len(self.interaction_history) < 2:
            return 0.0
            
        coherence_values = [i['coherence'] for i in self.interaction_history][-10:]
        return np.std(coherence_values) if len(coherence_values) > 1 else 0.0
    
    def _classify_resonance(self, ghost: bool, oscillation: float, 
                          amplification: float, frequency: float) -> ResonanceType:
        """Classify the type of resonance pattern detected."""
        if ghost:
            return ResonanceType.GHOST_ECHO
        elif amplification > self.amplification_threshold:
            return ResonanceType.AMPLIFYING
        elif oscillation > self.oscillation_threshold:
            return ResonanceType.OSCILLATING
        elif frequency > 0.7:
            return ResonanceType.CHAOTIC
        else:
            return ResonanceType.STABLE
    
    def _assess_risk_level(self, resonance_type: ResonanceType, 
                         frequency: float, coherence: float) -> float:
        """
        Assess risk level based on resonance patterns.
        Returns risk score from 0.0 (safe) to 1.0 (critical).
        """
        base_risk = 0.0
        
        # Type-based risk
        risk_weights = {
            ResonanceType.STABLE: 0.0,
            ResonanceType.OSCILLATING: 0.3,
            ResonanceType.AMPLIFYING: 0.6,
            ResonanceType.CHAOTIC: 0.8,
            ResonanceType.GHOST_ECHO: 0.7
        }
        base_risk = risk_weights.get(resonance_type, 0.0)
        
        # Adjust for frequency
        if frequency > self.max_resonance_frequency:
            base_risk += 0.2
        
        # Adjust for low coherence
        if coherence < 0.3:
            base_risk += 0.2
        
        return min(base_risk, 1.0)
    
    def _describe_pattern(self) -> str:
        """Generate human-readable description of detected pattern."""
        if not self.interaction_history:
            return "No pattern detected"
            
        recent = list(self.interaction_history)[-5:]
        patterns = []
        
        # Check for repetitive questioning
        user_inputs = [i['user'] for i in recent]
        if len(set(user_inputs)) < len(user_inputs) * 0.5:
            patterns.append("repetitive questioning")
        
        # Check for escalating length
        lengths = [len(i['ai']) for i in recent]
        if len(lengths) > 2 and all(lengths[i] > lengths[i-1] for i in range(1, len(lengths))):
            patterns.append("escalating response length")
        
        return ", ".join(patterns) if patterns else "complex interaction pattern"
    
    def _generate_recommendations(self, risk_level: float, 
                                resonance_type: ResonanceType) -> List[str]:
        """Generate safety recommendations based on detected patterns."""
        recommendations = []
        
        if risk_level > 0.7:
            recommendations.append("Consider resetting conversation context")
            recommendations.append("Introduce grounding questions")
        
        if resonance_type == ResonanceType.GHOST_ECHO:
            recommendations.append("Vary response patterns to break echo")
            recommendations.append("Introduce new topics or perspectives")
        
        if resonance_type == ResonanceType.AMPLIFYING:
            recommendations.append("Simplify and shorten responses")
            recommendations.append("Focus on concrete, specific information")
        
        if resonance_type == ResonanceType.OSCILLATING:
            recommendations.append("Stabilize conversation with consistent tone")
            recommendations.append("Avoid contradictory statements")
        
        if risk_level > 0.5:
            recommendations.append("Monitor coherence levels closely")
        
        return recommendations
    
    def get_safety_report(self) -> Dict:
        """Generate comprehensive safety report."""
        if not self.resonance_events:
            return {
                'status': 'stable',
                'total_events': 0,
                'current_risk': 0.0,
                'recommendations': ['System operating normally']
            }
        
        recent_events = self.resonance_events[-10:]
        avg_risk = np.mean([e.risk_level for e in recent_events])
        
        return {
            'status': 'monitoring' if avg_risk < 0.5 else 'warning',
            'total_events': len(self.resonance_events),
            'recent_events': len(recent_events),
            'current_risk': avg_risk,
            'dominant_pattern': max(
                set(e.resonance_type.value for e in recent_events),
                key=lambda x: sum(1 for e in recent_events if e.resonance_type.value == x)
            ),
            'recommendations': self._generate_recommendations(
                avg_risk, 
                recent_events[-1].resonance_type if recent_events else ResonanceType.STABLE
            )
        }