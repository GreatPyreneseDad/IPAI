"""
Message Coherence Analyzer

This module analyzes text messages for coherence indicators
using both rule-based and ML-enhanced approaches.
"""

import re
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import nltk
from textstat import flesch_reading_ease, flesch_kincaid_grade

from ..models.coherence_profile import GCTComponents


@dataclass
class CoherenceAnalysis:
    """Results of coherence analysis"""
    psi_score: float  # Internal consistency
    rho_score: float  # Wisdom indicators
    q_score: float    # Moral activation
    f_score: float    # Social connection
    overall_coherence: float
    red_flags: List[str]
    positive_indicators: List[str]
    needs_grounding: bool
    confidence: float
    analysis_details: Dict[str, Any]


class MessageCoherenceAnalyzer:
    """Analyze text messages for coherence indicators"""
    
    def __init__(self):
        self._initialize_patterns()
        self._initialize_word_lists()
        
    def _initialize_patterns(self):
        """Initialize regex patterns for analysis"""
        self.patterns = {
            # Toxic patterns that reduce coherence
            'absolutist': re.compile(r'\b(always|never|must|can\'t|impossible|perfect|completely|totally|absolutely)\b', re.IGNORECASE),
            'circular_reasoning': re.compile(r'\b(obviously|clearly|self-evident|proves|because it is|it just is)\b', re.IGNORECASE),
            'catastrophizing': re.compile(r'\b(disaster|catastrophe|terrible|awful|horrible|worst|end of world)\b', re.IGNORECASE),
            'black_white': re.compile(r'\b(all or nothing|completely|totally|either.*or|always.*never)\b', re.IGNORECASE),
            
            # Positive coherence patterns
            'reflection': re.compile(r'\b(learned|realized|understand|reflection|insight|growth|experience taught)\b', re.IGNORECASE),
            'nuanced': re.compile(r'\b(sometimes|often|usually|might|could|perhaps|seems|appears|complex|nuanced)\b', re.IGNORECASE),
            'values_based': re.compile(r'\b(important|values?|principles?|believe|care about|matters?|meaningful)\b', re.IGNORECASE),
            'social_connection': re.compile(r'\b(we|us|together|community|friends?|family|support|help|share)\b', re.IGNORECASE),
            'growth_mindset': re.compile(r'\b(learning|growing|improving|developing|progress|better|challenge)\b', re.IGNORECASE),
            'emotional_awareness': re.compile(r'\b(feel|feeling|emotion|aware|notice|sense|experience)\b', re.IGNORECASE),
        }
        
        # Question patterns indicate reflection
        self.question_patterns = re.compile(r'\?', re.IGNORECASE)
        
        # Future/action patterns indicate agency
        self.action_patterns = re.compile(r'\b(will|going to|plan|intend|commit|action|steps?|do|make)\b', re.IGNORECASE)
    
    def _initialize_word_lists(self):
        """Initialize word lists for analysis"""
        self.wisdom_words = {
            'high': ['wisdom', 'insight', 'understanding', 'perspective', 'experience', 'learned', 'growth'],
            'medium': ['think', 'believe', 'consider', 'reflect', 'ponder', 'contemplate'],
            'low': ['guess', 'assume', 'suppose', 'probably', 'maybe']
        }
        
        self.moral_words = {
            'high': ['values', 'principles', 'ethics', 'moral', 'right', 'wrong', 'should', 'ought'],
            'medium': ['important', 'matters', 'care', 'concern', 'responsibility'],
            'low': ['whatever', 'doesn\'t matter', 'don\'t care']
        }
        
        self.social_words = {
            'high': ['we', 'us', 'together', 'community', 'team', 'family', 'friends'],
            'medium': ['people', 'others', 'someone', 'they', 'relationship'],
            'low': ['I', 'me', 'myself', 'alone', 'isolated']
        }
        
        self.consistency_words = {
            'high': ['consistent', 'aligned', 'coherent', 'integrated', 'unified'],
            'medium': ['connected', 'related', 'linked', 'follows'],
            'low': ['confused', 'conflicted', 'contradictory', 'inconsistent']
        }
    
    def analyze_message(self, message: str, context: Optional[Dict] = None) -> CoherenceAnalysis:
        """Analyze a message for coherence indicators"""
        context = context or {}
        
        # Basic text processing
        text_stats = self._get_text_statistics(message)
        
        # Analyze each GCT component
        psi_analysis = self._analyze_internal_consistency(message, text_stats)
        rho_analysis = self._analyze_wisdom_indicators(message, text_stats)
        q_analysis = self._analyze_moral_activation(message, text_stats)
        f_analysis = self._analyze_social_connection(message, text_stats)
        
        # Calculate overall coherence
        overall_coherence = self._calculate_overall_coherence(
            psi_analysis, rho_analysis, q_analysis, f_analysis
        )
        
        # Identify red flags and positive indicators
        red_flags = self._identify_red_flags(message, text_stats)
        positive_indicators = self._identify_positive_indicators(message, text_stats)
        
        # Determine if grounding is needed
        needs_grounding = self._needs_grounding(
            overall_coherence, red_flags, [psi_analysis, rho_analysis, q_analysis, f_analysis]
        )
        
        # Calculate confidence in analysis
        confidence = self._calculate_confidence(message, text_stats)
        
        return CoherenceAnalysis(
            psi_score=psi_analysis['score'],
            rho_score=rho_analysis['score'],
            q_score=q_analysis['score'],
            f_score=f_analysis['score'],
            overall_coherence=overall_coherence,
            red_flags=red_flags,
            positive_indicators=positive_indicators,
            needs_grounding=needs_grounding,
            confidence=confidence,
            analysis_details={
                'text_stats': text_stats,
                'psi_analysis': psi_analysis,
                'rho_analysis': rho_analysis,
                'q_analysis': q_analysis,
                'f_analysis': f_analysis
            }
        )
    
    def _get_text_statistics(self, message: str) -> Dict[str, Any]:
        """Get basic text statistics"""
        words = message.split()
        sentences = message.split('.')
        
        stats = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'question_count': len(self.question_patterns.findall(message)),
            'exclamation_count': message.count('!'),
            'uppercase_ratio': sum(1 for c in message if c.isupper()) / len(message) if message else 0,
        }
        
        # Add readability scores
        try:
            stats['flesch_score'] = flesch_reading_ease(message)
            stats['flesch_kincaid'] = flesch_kincaid_grade(message)
        except:
            stats['flesch_score'] = 50  # Average
            stats['flesch_kincaid'] = 8   # 8th grade level
        
        return stats
    
    def _analyze_internal_consistency(self, message: str, text_stats: Dict) -> Dict[str, Any]:
        """Analyze internal consistency (Psi)"""
        score = 0.5  # Base score
        factors = []
        
        # Check for contradictory statements
        contradictions = self._find_contradictions(message)
        if contradictions:
            score -= 0.3
            factors.append(f"Contradictions found: {contradictions}")
        
        # Check for logical flow
        logical_connectors = len(re.findall(r'\b(because|therefore|thus|however|although|since|so)\b', message, re.IGNORECASE))
        if logical_connectors > 0:
            score += min(0.2, logical_connectors * 0.05)
            factors.append(f"Logical connectors: {logical_connectors}")
        
        # Check for absolutist thinking (reduces consistency)
        absolutist_matches = len(self.patterns['absolutist'].findall(message))
        if absolutist_matches > 0:
            score -= min(0.2, absolutist_matches * 0.05)
            factors.append(f"Absolutist language: {absolutist_matches}")
        
        # Check for nuanced thinking (increases consistency)
        nuanced_matches = len(self.patterns['nuanced'].findall(message))
        if nuanced_matches > 0:
            score += min(0.2, nuanced_matches * 0.03)
            factors.append(f"Nuanced language: {nuanced_matches}")
        
        # Consistency word analysis
        for level, words in self.consistency_words.items():
            word_count = sum(1 for word in words if word.lower() in message.lower())
            if word_count > 0:
                if level == 'high':
                    score += min(0.15, word_count * 0.05)
                elif level == 'low':
                    score -= min(0.15, word_count * 0.05)
                factors.append(f"{level} consistency words: {word_count}")
        
        return {
            'score': max(0.0, min(1.0, score)),
            'factors': factors,
            'contradictions': contradictions
        }
    
    def _analyze_wisdom_indicators(self, message: str, text_stats: Dict) -> Dict[str, Any]:
        """Analyze wisdom indicators (Rho)"""
        score = 0.4  # Base score
        factors = []
        
        # Check for reflection patterns
        reflection_matches = len(self.patterns['reflection'].findall(message))
        if reflection_matches > 0:
            score += min(0.3, reflection_matches * 0.1)
            factors.append(f"Reflection indicators: {reflection_matches}")
        
        # Check for experience-based language
        experience_patterns = re.findall(r'\b(learned|experienced|discovered|realized|found out)\b', message, re.IGNORECASE)
        if experience_patterns:
            score += min(0.2, len(experience_patterns) * 0.05)
            factors.append(f"Experience-based language: {len(experience_patterns)}")
        
        # Wisdom word analysis
        for level, words in self.wisdom_words.items():
            word_count = sum(1 for word in words if word.lower() in message.lower())
            if word_count > 0:
                if level == 'high':
                    score += min(0.2, word_count * 0.06)
                elif level == 'medium':
                    score += min(0.1, word_count * 0.03)
                elif level == 'low':
                    score -= min(0.1, word_count * 0.03)
                factors.append(f"{level} wisdom words: {word_count}")
        
        # Questions indicate reflection
        if text_stats['question_count'] > 0:
            score += min(0.15, text_stats['question_count'] * 0.05)
            factors.append(f"Reflective questions: {text_stats['question_count']}")
        
        return {
            'score': max(0.0, min(1.0, score)),
            'factors': factors
        }
    
    def _analyze_moral_activation(self, message: str, text_stats: Dict) -> Dict[str, Any]:
        """Analyze moral activation (Q)"""
        score = 0.4  # Base score
        factors = []
        
        # Check for values-based language
        values_matches = len(self.patterns['values_based'].findall(message))
        if values_matches > 0:
            score += min(0.3, values_matches * 0.08)
            factors.append(f"Values-based language: {values_matches}")
        
        # Check for action orientation
        action_matches = len(self.action_patterns.findall(message))
        if action_matches > 0:
            score += min(0.2, action_matches * 0.05)
            factors.append(f"Action-oriented language: {action_matches}")
        
        # Moral word analysis
        for level, words in self.moral_words.items():
            word_count = sum(1 for word in words if word.lower() in message.lower())
            if word_count > 0:
                if level == 'high':
                    score += min(0.25, word_count * 0.08)
                elif level == 'medium':
                    score += min(0.1, word_count * 0.04)
                elif level == 'low':
                    score -= min(0.15, word_count * 0.05)
                factors.append(f"{level} moral words: {word_count}")
        
        # Check for commitment language
        commitment_patterns = re.findall(r'\b(commit|dedicated|determined|resolved|will)\b', message, re.IGNORECASE)
        if commitment_patterns:
            score += min(0.15, len(commitment_patterns) * 0.05)
            factors.append(f"Commitment language: {len(commitment_patterns)}")
        
        return {
            'score': max(0.0, min(1.0, score)),
            'factors': factors
        }
    
    def _analyze_social_connection(self, message: str, text_stats: Dict) -> Dict[str, Any]:
        """Analyze social connection (F)"""
        score = 0.4  # Base score
        factors = []
        
        # Check for social connection patterns
        social_matches = len(self.patterns['social_connection'].findall(message))
        if social_matches > 0:
            score += min(0.3, social_matches * 0.06)
            factors.append(f"Social connection language: {social_matches}")
        
        # Social word analysis
        for level, words in self.social_words.items():
            word_count = sum(1 for word in words if word.lower() in message.lower())
            if word_count > 0:
                if level == 'high':
                    score += min(0.2, word_count * 0.04)
                elif level == 'medium':
                    score += min(0.1, word_count * 0.02)
                elif level == 'low':
                    score -= min(0.1, word_count * 0.03)
                factors.append(f"{level} social words: {word_count}")
        
        # Check for empathy indicators
        empathy_patterns = re.findall(r'\b(understand|empathy|compassion|feel for|relate to)\b', message, re.IGNORECASE)
        if empathy_patterns:
            score += min(0.15, len(empathy_patterns) * 0.05)
            factors.append(f"Empathy indicators: {len(empathy_patterns)}")
        
        # Check for isolation indicators
        isolation_patterns = re.findall(r'\b(alone|lonely|isolated|nobody|no one cares)\b', message, re.IGNORECASE)
        if isolation_patterns:
            score -= min(0.2, len(isolation_patterns) * 0.1)
            factors.append(f"Isolation indicators: {len(isolation_patterns)}")
        
        return {
            'score': max(0.0, min(1.0, score)),
            'factors': factors
        }
    
    def _find_contradictions(self, message: str) -> List[str]:
        """Find potential contradictions in the message"""
        contradictions = []
        
        # Simple contradiction patterns
        contradiction_patterns = [
            (r'\bbut\b.*\bhowever\b', "Multiple contradictory connectors"),
            (r'\balways\b.*\bnever\b', "Absolute contradictions"),
            (r'\blove\b.*\bhate\b', "Emotional contradictions"),
            (r'\byes\b.*\bno\b', "Direct contradictions")
        ]
        
        for pattern, description in contradiction_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                contradictions.append(description)
        
        return contradictions
    
    def _calculate_overall_coherence(self, psi: Dict, rho: Dict, q: Dict, f: Dict) -> float:
        """Calculate overall coherence from component scores"""
        # Weighted average with wisdom enhancement
        psi_score = psi['score']
        rho_score = rho['score']
        q_score = q['score']
        f_score = f['score']
        
        # Base coherence
        base = psi_score + (rho_score * psi_score) + q_score + (f_score * psi_score)
        
        # Wisdom-activation coupling
        coupling = 0.15 * rho_score * q_score
        
        total = base + coupling
        
        # Normalize to 0-1 range
        return min(1.0, total / 4.0)
    
    def _identify_red_flags(self, message: str, text_stats: Dict) -> List[str]:
        """Identify concerning patterns in the message"""
        red_flags = []
        
        # Check for various concerning patterns
        if len(self.patterns['absolutist'].findall(message)) > 3:
            red_flags.append("Excessive absolutist thinking")
        
        if len(self.patterns['circular_reasoning'].findall(message)) > 0:
            red_flags.append("Circular reasoning patterns")
        
        if len(self.patterns['catastrophizing'].findall(message)) > 0:
            red_flags.append("Catastrophizing language")
        
        if text_stats['uppercase_ratio'] > 0.3:
            red_flags.append("Excessive capitalization (possible agitation)")
        
        if text_stats['exclamation_count'] > 5:
            red_flags.append("Excessive exclamation marks")
        
        # Check for crisis indicators
        crisis_patterns = re.findall(r'\b(suicide|kill myself|end it all|can\'t go on|hopeless)\b', message, re.IGNORECASE)
        if crisis_patterns:
            red_flags.append("CRISIS: Suicidal ideation detected")
        
        return red_flags
    
    def _identify_positive_indicators(self, message: str, text_stats: Dict) -> List[str]:
        """Identify positive patterns in the message"""
        positive_indicators = []
        
        if len(self.patterns['reflection'].findall(message)) > 0:
            positive_indicators.append("Reflective thinking")
        
        if len(self.patterns['nuanced'].findall(message)) > 2:
            positive_indicators.append("Nuanced perspective")
        
        if len(self.patterns['growth_mindset'].findall(message)) > 0:
            positive_indicators.append("Growth mindset")
        
        if len(self.patterns['values_based'].findall(message)) > 0:
            positive_indicators.append("Values-based thinking")
        
        if len(self.patterns['social_connection'].findall(message)) > 1:
            positive_indicators.append("Social connection awareness")
        
        if text_stats['question_count'] > 0:
            positive_indicators.append("Self-inquiry and reflection")
        
        return positive_indicators
    
    def _needs_grounding(self, overall_coherence: float, red_flags: List[str], component_analyses: List[Dict]) -> bool:
        """Determine if the message needs grounding"""
        # Crisis indicators always need grounding
        if any("CRISIS" in flag for flag in red_flags):
            return True
        
        # Low overall coherence needs grounding
        if overall_coherence < 0.3:
            return True
        
        # Multiple red flags need grounding
        if len(red_flags) >= 3:
            return True
        
        # Any component score very low
        if any(analysis['score'] < 0.2 for analysis in component_analyses):
            return True
        
        return False
    
    def _calculate_confidence(self, message: str, text_stats: Dict) -> float:
        """Calculate confidence in the analysis"""
        confidence = 0.5  # Base confidence
        
        # More text generally means higher confidence
        if text_stats['word_count'] > 50:
            confidence += 0.2
        elif text_stats['word_count'] < 10:
            confidence -= 0.2
        
        # Clear structure increases confidence
        if text_stats['sentence_count'] > 1:
            confidence += 0.1
        
        # Reasonable readability increases confidence
        if 30 <= text_stats.get('flesch_score', 50) <= 70:
            confidence += 0.1
        
        # Not too much noise
        if text_stats['uppercase_ratio'] < 0.1:
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def batch_analyze(self, messages: List[str]) -> List[CoherenceAnalysis]:
        """Analyze multiple messages in batch"""
        return [self.analyze_message(msg) for msg in messages]
    
    def get_analysis_summary(self, analysis: CoherenceAnalysis) -> str:
        """Get a human-readable summary of the analysis"""
        summary = f"""
Coherence Analysis Summary:
- Overall Coherence: {analysis.overall_coherence:.2f}
- Internal Consistency (Ψ): {analysis.psi_score:.2f}
- Wisdom Indicators (ρ): {analysis.rho_score:.2f}
- Moral Activation (q): {analysis.q_score:.2f}
- Social Connection (f): {analysis.f_score:.2f}

Positive Indicators: {', '.join(analysis.positive_indicators) if analysis.positive_indicators else 'None'}
Red Flags: {', '.join(analysis.red_flags) if analysis.red_flags else 'None'}
Needs Grounding: {'Yes' if analysis.needs_grounding else 'No'}
Analysis Confidence: {analysis.confidence:.2f}
"""
        return summary