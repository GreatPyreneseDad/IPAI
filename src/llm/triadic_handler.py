"""
Triadic Response Handler

This module handles triadic processing of LLM responses,
implementing the Generate-Analyze-Ground pattern for response refinement.
"""

import re
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from ..models.coherence_profile import CoherenceProfile, CoherenceLevel
from ..coherence.triadic_processor import TriadicProcessor, TriadicConfig
from .coherence_analyzer import MessageCoherenceAnalyzer, CoherenceAnalysis


@dataclass
class ResponseAnalysis:
    """Analysis of an LLM response"""
    coherence_analysis: CoherenceAnalysis
    triadic_metrics: Dict[str, float]
    needs_refinement: bool
    refinement_suggestions: List[str]
    response_quality: float
    safety_flags: List[str]


@dataclass 
class LLMResponse:
    """Enhanced LLM response with triadic processing"""
    original_text: str
    processed_text: str
    user_profile: CoherenceProfile
    response_analysis: ResponseAnalysis
    processing_metadata: Dict[str, Any]
    timestamp: datetime


class TriadicResponseHandler:
    """Handle triadic processing of LLM responses"""
    
    def __init__(self, 
                 triadic_processor: Optional[TriadicProcessor] = None,
                 coherence_analyzer: Optional[MessageCoherenceAnalyzer] = None):
        self.triadic_processor = triadic_processor or TriadicProcessor()
        self.coherence_analyzer = coherence_analyzer or MessageCoherenceAnalyzer()
        self.response_history = []
        self.safety_patterns = self._initialize_safety_patterns()
        
    def _initialize_safety_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for safety checking"""
        return {
            'harmful_advice': [
                r'\b(suicide|kill yourself|harm yourself|end it all)\b',
                r'\b(illegal|break the law|commit crime)\b',
                r'\b(violence|hurt someone|attack)\b'
            ],
            'medical_advice': [
                r'\b(diagnose|diagnosis|medical condition|take medication|stop medication)\b',
                r'\b(doctor|physician|medical professional|healthcare)\b'
            ],
            'financial_advice': [
                r'\b(invest all|guaranteed return|risk-free|financial advice)\b',
                r'\b(stocks|crypto|investment|financial planning)\b'
            ],
            'relationship_manipulation': [
                r'\b(manipulate|control|deceive|lie to)\b',
                r'\b(ultimatum|threaten|force them)\b'
            ]
        }
    
    def process_response(
        self,
        response_text: str,
        user_profile: CoherenceProfile,
        context: Optional[Dict] = None
    ) -> LLMResponse:
        """Process an LLM response through triadic refinement"""
        
        # Phase 1: Generate (response already generated)
        original_text = response_text
        
        # Phase 2: Analyze
        analysis = self._analyze_response(response_text, user_profile, context)
        
        # Phase 3: Ground (if needed)
        processed_text, grounding_applied = self._apply_grounding_if_needed(
            response_text, analysis, user_profile, context
        )
        
        # Create response object
        llm_response = LLMResponse(
            original_text=original_text,
            processed_text=processed_text,
            user_profile=user_profile,
            response_analysis=analysis,
            processing_metadata={
                'grounding_applied': grounding_applied,
                'processing_time': datetime.utcnow(),
                'triadic_phases': 3 if grounding_applied else 2,
                'context': context or {}
            },
            timestamp=datetime.utcnow()
        )
        
        # Store in history for learning
        self._update_response_history(llm_response)
        
        return llm_response
    
    def _analyze_response(
        self,
        response_text: str,
        user_profile: CoherenceProfile,
        context: Optional[Dict] = None
    ) -> ResponseAnalysis:
        """Analyze response for quality and coherence"""
        
        # Coherence analysis
        coherence_analysis = self.coherence_analyzer.analyze_message(response_text, context)
        
        # Safety analysis
        safety_flags = self._check_safety(response_text)
        
        # Response quality analysis
        quality_metrics = self._analyze_response_quality(response_text, user_profile)
        
        # Triadic analysis
        triadic_metrics = self._apply_triadic_analysis(
            response_text, coherence_analysis, user_profile
        )
        
        # Determine if refinement is needed
        needs_refinement = self._needs_refinement(
            coherence_analysis, safety_flags, quality_metrics, user_profile
        )
        
        # Generate refinement suggestions
        refinement_suggestions = self._generate_refinement_suggestions(
            coherence_analysis, safety_flags, quality_metrics, user_profile
        )
        
        return ResponseAnalysis(
            coherence_analysis=coherence_analysis,
            triadic_metrics=triadic_metrics,
            needs_refinement=needs_refinement,
            refinement_suggestions=refinement_suggestions,
            response_quality=quality_metrics['overall_quality'],
            safety_flags=safety_flags
        )
    
    def _check_safety(self, response_text: str) -> List[str]:
        """Check response for safety issues"""
        safety_flags = []
        
        for category, patterns in self.safety_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_text, re.IGNORECASE):
                    safety_flags.append(f"Safety concern: {category}")
                    break
        
        # Check for crisis-level language
        crisis_patterns = [
            r'\b(hopeless|worthless|burden|better off dead)\b',
            r'\b(no point|give up|can\'t go on)\b'
        ]
        
        for pattern in crisis_patterns:
            if re.search(pattern, response_text, re.IGNORECASE):
                safety_flags.append("CRISIS: Potentially harmful language")
                break
        
        return safety_flags
    
    def _analyze_response_quality(
        self,
        response_text: str,
        user_profile: CoherenceProfile
    ) -> Dict[str, float]:
        """Analyze overall response quality"""
        
        # Base quality metrics
        word_count = len(response_text.split())
        sentence_count = len([s for s in response_text.split('.') if s.strip()])
        
        # Length appropriateness (user coherence dependent)
        length_score = self._score_response_length(word_count, user_profile.level)
        
        # Coherence appropriateness
        coherence_match = self._score_coherence_match(response_text, user_profile)
        
        # Helpfulness indicators
        helpfulness_score = self._score_helpfulness(response_text)
        
        # Empathy indicators
        empathy_score = self._score_empathy(response_text)
        
        # Actionability
        actionability_score = self._score_actionability(response_text, user_profile.level)
        
        # Calculate overall quality
        overall_quality = (
            length_score * 0.15 +
            coherence_match * 0.25 +
            helpfulness_score * 0.25 +
            empathy_score * 0.20 +
            actionability_score * 0.15
        )
        
        return {
            'length_score': length_score,
            'coherence_match': coherence_match,
            'helpfulness_score': helpfulness_score,
            'empathy_score': empathy_score,
            'actionability_score': actionability_score,
            'overall_quality': overall_quality
        }
    
    def _score_response_length(self, word_count: int, coherence_level: CoherenceLevel) -> float:
        """Score response length appropriateness based on user coherence"""
        ideal_ranges = {
            CoherenceLevel.CRITICAL: (20, 80),    # Short, simple responses
            CoherenceLevel.LOW: (30, 120),        # Moderate length
            CoherenceLevel.MEDIUM: (50, 200),     # Standard length
            CoherenceLevel.HIGH: (80, 300)        # Can handle longer responses
        }
        
        min_words, max_words = ideal_ranges[coherence_level]
        
        if min_words <= word_count <= max_words:
            return 1.0
        elif word_count < min_words:
            return max(0.0, word_count / min_words)
        else:
            return max(0.0, 1.0 - (word_count - max_words) / max_words)
    
    def _score_coherence_match(self, response_text: str, user_profile: CoherenceProfile) -> float:
        """Score how well response matches user's coherence level"""
        
        # Analyze response complexity
        complex_patterns = [
            r'\b(however|nevertheless|furthermore|consequently|therefore)\b',
            r'\b(nuanced|complex|multifaceted|sophisticated)\b',
            r'\b(consider|contemplate|reflect|analyze)\b'
        ]
        
        simple_patterns = [
            r'\b(simple|easy|basic|clear|straightforward)\b',
            r'\b(step|first|next|then|finally)\b',
            r'\b(help|support|understand)\b'
        ]
        
        complex_count = sum(len(re.findall(pattern, response_text, re.IGNORECASE)) 
                          for pattern in complex_patterns)
        simple_count = sum(len(re.findall(pattern, response_text, re.IGNORECASE)) 
                         for pattern in simple_patterns)
        
        complexity_ratio = complex_count / max(1, complex_count + simple_count)
        
        # Match complexity to user level
        target_complexity = {
            CoherenceLevel.CRITICAL: 0.1,
            CoherenceLevel.LOW: 0.3,
            CoherenceLevel.MEDIUM: 0.5,
            CoherenceLevel.HIGH: 0.7
        }
        
        target = target_complexity[user_profile.level]
        match_score = 1.0 - abs(complexity_ratio - target)
        
        return max(0.0, match_score)
    
    def _score_helpfulness(self, response_text: str) -> float:
        """Score response helpfulness"""
        helpful_patterns = [
            r'\b(help|assist|support|guide|advice)\b',
            r'\b(try|consider|might|could|perhaps)\b',
            r'\b(steps?|approach|method|way|solution)\b',
            r'\b(understand|see|realize|recognize)\b'
        ]
        
        unhelpful_patterns = [
            r'\b(can\'t help|don\'t know|impossible|hopeless)\b',
            r'\b(not my problem|figure it out|deal with it)\b'
        ]
        
        helpful_count = sum(len(re.findall(pattern, response_text, re.IGNORECASE)) 
                          for pattern in helpful_patterns)
        unhelpful_count = sum(len(re.findall(pattern, response_text, re.IGNORECASE)) 
                            for pattern in unhelpful_patterns)
        
        score = min(1.0, helpful_count * 0.1) - min(0.5, unhelpful_count * 0.2)
        return max(0.0, score)
    
    def _score_empathy(self, response_text: str) -> float:
        """Score empathy indicators in response"""
        empathy_patterns = [
            r'\b(understand|feel|empathy|compassion)\b',
            r'\b(difficult|challenging|hard|tough)\b',
            r'\b(valid|legitimate|make sense|natural)\b',
            r'\b(you\'re not alone|many people|others experience)\b'
        ]
        
        empathy_count = sum(len(re.findall(pattern, response_text, re.IGNORECASE)) 
                          for pattern in empathy_patterns)
        
        return min(1.0, empathy_count * 0.15)
    
    def _score_actionability(self, response_text: str, coherence_level: CoherenceLevel) -> float:
        """Score how actionable the response is"""
        action_patterns = [
            r'\b(try|do|practice|implement|apply)\b',
            r'\b(step|first|next|then|finally)\b',
            r'\b(start|begin|begin with|focus on)\b',
            r'\b(specific|concrete|practical|actionable)\b'
        ]
        
        action_count = sum(len(re.findall(pattern, response_text, re.IGNORECASE)) 
                         for pattern in action_patterns)
        
        # Different coherence levels need different amounts of actionability
        target_actions = {
            CoherenceLevel.CRITICAL: 2,  # Simple, immediate actions
            CoherenceLevel.LOW: 3,       # Clear step-by-step
            CoherenceLevel.MEDIUM: 4,    # Balanced guidance
            CoherenceLevel.HIGH: 5       # Multiple options
        }
        
        target = target_actions[coherence_level]
        score = min(1.0, action_count / target)
        
        return score
    
    def _apply_triadic_analysis(
        self,
        response_text: str,
        coherence_analysis: CoherenceAnalysis,
        user_profile: CoherenceProfile
    ) -> Dict[str, float]:
        """Apply triadic analysis to the response"""
        
        # Use triadic processor to analyze response coherence
        response_coherence = coherence_analysis.overall_coherence
        
        # Create mock GCT components from analysis
        from ..models.coherence_profile import GCTComponents
        mock_components = GCTComponents(
            psi=coherence_analysis.psi_score,
            rho=coherence_analysis.rho_score,
            q=coherence_analysis.q_score,
            f=coherence_analysis.f_score
        )
        
        # Apply triadic processing
        processed_coherence, triadic_metrics = self.triadic_processor.process(
            response_coherence, mock_components, {'text': response_text}
        )
        
        # Add response-specific metrics
        triadic_metrics.update({
            'response_coherence_improvement': processed_coherence - response_coherence,
            'user_coherence_match': self._calculate_coherence_match(
                mock_components, user_profile.components
            )
        })
        
        return triadic_metrics
    
    def _calculate_coherence_match(self, response_components, user_components) -> float:
        """Calculate how well response coherence matches user coherence"""
        component_diffs = [
            abs(response_components.psi - user_components.psi),
            abs(response_components.rho - user_components.rho),
            abs(response_components.q - user_components.q),
            abs(response_components.f - user_components.f)
        ]
        
        # Lower differences mean better match
        avg_diff = sum(component_diffs) / len(component_diffs)
        match_score = max(0.0, 1.0 - avg_diff)
        
        return match_score
    
    def _needs_refinement(
        self,
        coherence_analysis: CoherenceAnalysis,
        safety_flags: List[str],
        quality_metrics: Dict[str, float],
        user_profile: CoherenceProfile
    ) -> bool:
        """Determine if response needs refinement"""
        
        # Always refine if safety issues
        if safety_flags:
            return True
        
        # Refine if coherence analysis indicates grounding needed
        if coherence_analysis.needs_grounding:
            return True
        
        # Refine if overall quality is low
        if quality_metrics['overall_quality'] < 0.5:
            return True
        
        # Refine if coherence match is poor
        if quality_metrics['coherence_match'] < 0.4:
            return True
        
        # Refine if user is in crisis and response lacks empathy
        if (user_profile.level == CoherenceLevel.CRITICAL and 
            quality_metrics['empathy_score'] < 0.5):
            return True
        
        return False
    
    def _generate_refinement_suggestions(
        self,
        coherence_analysis: CoherenceAnalysis,
        safety_flags: List[str],
        quality_metrics: Dict[str, float],
        user_profile: CoherenceProfile
    ) -> List[str]:
        """Generate specific refinement suggestions"""
        suggestions = []
        
        if safety_flags:
            suggestions.append("Remove or modify potentially harmful content")
        
        if coherence_analysis.needs_grounding:
            suggestions.append("Apply grounding to improve coherence")
        
        if quality_metrics['empathy_score'] < 0.5:
            suggestions.append("Add more empathetic language and understanding")
        
        if quality_metrics['helpfulness_score'] < 0.5:
            suggestions.append("Provide more concrete, helpful guidance")
        
        if quality_metrics['actionability_score'] < 0.5:
            suggestions.append("Include more specific, actionable steps")
        
        if quality_metrics['coherence_match'] < 0.4:
            if user_profile.level in [CoherenceLevel.CRITICAL, CoherenceLevel.LOW]:
                suggestions.append("Simplify language and concepts")
            else:
                suggestions.append("Increase sophistication to match user level")
        
        return suggestions
    
    def _apply_grounding_if_needed(
        self,
        response_text: str,
        analysis: ResponseAnalysis,
        user_profile: CoherenceProfile,
        context: Optional[Dict] = None
    ) -> Tuple[str, bool]:
        """Apply grounding to response if needed"""
        
        if not analysis.needs_refinement:
            return response_text, False
        
        # Apply different types of grounding based on issues
        grounded_text = response_text
        
        # Safety grounding
        if analysis.safety_flags:
            grounded_text = self._apply_safety_grounding(grounded_text, analysis.safety_flags)
        
        # Coherence grounding
        if analysis.coherence_analysis.needs_grounding:
            grounded_text = self._apply_coherence_grounding(
                grounded_text, analysis.coherence_analysis, user_profile
            )
        
        # Quality grounding
        if analysis.response_quality < 0.5:
            grounded_text = self._apply_quality_grounding(
                grounded_text, analysis.refinement_suggestions, user_profile
            )
        
        return grounded_text, True
    
    def _apply_safety_grounding(self, text: str, safety_flags: List[str]) -> str:
        """Apply safety grounding to remove harmful content"""
        
        # Remove or replace harmful patterns
        grounded = text
        
        # Remove crisis language
        crisis_replacements = {
            r'\b(hopeless|worthless|burden)\b': 'challenging',
            r'\b(give up|can\'t go on)\b': 'take a break and seek support',
            r'\b(no point)\b': 'difficulty finding meaning right now'
        }
        
        for pattern, replacement in crisis_replacements.items():
            grounded = re.sub(pattern, replacement, grounded, flags=re.IGNORECASE)
        
        # Add safety resources if crisis detected
        if any("CRISIS" in flag for flag in safety_flags):
            grounded += "\n\n⚠️ If you're having thoughts of self-harm, please reach out to a mental health professional, call a crisis hotline, or go to your nearest emergency room."
        
        return grounded
    
    def _apply_coherence_grounding(
        self,
        text: str,
        coherence_analysis: CoherenceAnalysis,
        user_profile: CoherenceProfile
    ) -> str:
        """Apply coherence grounding to improve response coherence"""
        
        grounded = text
        
        # Add grounding phrases
        grounding_phrases = [
            "Let me clarify: ",
            "In practical terms: ",
            "Specifically: ",
            "What this means is: "
        ]
        
        # Add structure if needed
        if coherence_analysis.psi_score < 0.3:
            # Low internal consistency - add logical structure
            grounded = "Let me break this down clearly:\n\n" + grounded
        
        # Add wisdom context if needed
        if coherence_analysis.rho_score < 0.3:
            # Low wisdom - add experiential context
            grounded += "\n\nThis is based on common experiences that many people have found helpful."
        
        # Add moral grounding if needed
        if coherence_analysis.q_score < 0.3:
            # Low moral activation - add values context
            grounded += "\n\nThis aligns with values of growth, authenticity, and well-being."
        
        # Add social grounding if needed
        if coherence_analysis.f_score < 0.3:
            # Low social connection - add community context
            grounded += "\n\nRemember that you're not alone in this - many people face similar challenges."
        
        return grounded
    
    def _apply_quality_grounding(
        self,
        text: str,
        suggestions: List[str],
        user_profile: CoherenceLevel
    ) -> str:
        """Apply quality improvements based on suggestions"""
        
        grounded = text
        
        # Apply specific improvements
        for suggestion in suggestions:
            if "empathetic language" in suggestion:
                grounded = "I understand this can be difficult. " + grounded
            
            elif "actionable steps" in suggestion:
                grounded += "\n\nHere's a specific step you could try: [Add context-appropriate action]"
            
            elif "simplify" in suggestion:
                # Simplify complex sentences
                grounded = re.sub(r'\b(however|nevertheless|furthermore)\b', 'but', grounded, flags=re.IGNORECASE)
                grounded = re.sub(r'\b(sophisticated|complex|multifaceted)\b', 'detailed', grounded, flags=re.IGNORECASE)
        
        return grounded
    
    def _update_response_history(self, llm_response: LLMResponse):
        """Update response history for learning"""
        self.response_history.append(llm_response)
        
        # Keep only recent history
        if len(self.response_history) > 100:
            self.response_history = self.response_history[-100:]
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get statistics on response processing"""
        if not self.response_history:
            return {}
        
        total_responses = len(self.response_history)
        refined_responses = sum(1 for r in self.response_history 
                              if r.processing_metadata['grounding_applied'])
        
        avg_quality = sum(r.response_analysis.response_quality 
                         for r in self.response_history) / total_responses
        
        safety_incidents = sum(1 for r in self.response_history 
                             if r.response_analysis.safety_flags)
        
        return {
            'total_responses': total_responses,
            'refinement_rate': refined_responses / total_responses,
            'average_quality': avg_quality,
            'safety_incidents': safety_incidents,
            'safety_rate': safety_incidents / total_responses
        }