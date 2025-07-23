"""
Triadic Logic Processor

This module implements triadic logic processing for coherence refinement,
following the Generate-Analyze-Ground pattern.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import re
from enum import Enum

from ..models.coherence_profile import GCTComponents


class TriadicPhase(Enum):
    """Triadic processing phases"""
    GENERATE = "generate"
    ANALYZE = "analyze"
    GROUND = "ground"


@dataclass
class TriadicConfig:
    """Configuration for triadic processing"""
    coherence_threshold: float = 0.7
    collapse_threshold: float = 0.5
    ground_threshold: float = 0.3
    grounding_boost_factor: float = 0.2
    max_grounding_iterations: int = 3
    enable_neural_processing: bool = False
    
    # Pattern recognition
    toxic_patterns: List[str] = None
    recursive_patterns: List[str] = None
    grounding_phrases: List[str] = None
    
    def __post_init__(self):
        if self.toxic_patterns is None:
            self.toxic_patterns = ['must', 'always', 'never', 'only', 'impossible', 'perfect']
        if self.recursive_patterns is None:
            self.recursive_patterns = ['proves', 'recursive', 'self-evident', 'obviously', 'clearly']
        if self.grounding_phrases is None:
            self.grounding_phrases = ['Let me clarify: ', 'In practical terms: ', 'Specifically: ']


class TriadicProcessor:
    """Triadic logic processor for coherence refinement"""
    
    def __init__(self, config: Optional[TriadicConfig] = None):
        self.config = config or TriadicConfig()
        self.processing_history = []
        self.pattern_cache = {}
        
    def process(
        self, 
        coherence: float, 
        components: GCTComponents,
        context: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply triadic logic to refine coherence
        
        Args:
            coherence: Initial coherence score
            components: GCT components
            context: Optional context for processing
            
        Returns:
            Tuple of (adjusted_coherence, triadic_metrics)
        """
        context = context or {}
        
        # Phase 1: Generate (coherence already generated)
        generation_metrics = self._record_generation_phase(coherence, components)
        
        # Phase 2: Analyze
        analysis_metrics = self._analyze_coherence(coherence, components, context)
        
        # Phase 3: Ground (if needed)
        grounding_metrics, grounded_coherence = self._apply_grounding_if_needed(
            coherence, components, analysis_metrics, context
        )
        
        # Combine all metrics
        triadic_metrics = {
            **generation_metrics,
            **analysis_metrics,
            **grounding_metrics,
            'triadic_phases_applied': self._get_phases_applied(analysis_metrics, grounding_metrics)
        }
        
        # Record processing for learning
        self._record_processing(coherence, grounded_coherence, triadic_metrics)
        
        return grounded_coherence, triadic_metrics
    
    def _record_generation_phase(self, coherence: float, components: GCTComponents) -> Dict[str, float]:
        """Record metrics from generation phase"""
        return {
            'generation_completeness': 1.0,  # Always complete for numerical input
            'generation_quality': min(1.0, coherence / 0.8),  # Quality based on coherence level
            'component_balance': self._calculate_component_balance(components)
        }
    
    def _analyze_coherence(
        self, 
        coherence: float, 
        components: GCTComponents,
        context: Dict
    ) -> Dict[str, float]:
        """Analyze coherence for issues requiring grounding"""
        psi, rho, q, f = components.psi, components.rho, components.q, components.f
        
        # Calculate various risk metrics
        collapse_risk = self._calculate_collapse_risk(psi, q, rho)
        drift_risk = self._calculate_drift_risk(coherence, rho)
        isolation_risk = self._calculate_isolation_risk(f)
        inconsistency_risk = self._calculate_inconsistency_risk(psi, components)
        
        # Analyze textual context if provided
        context_risks = self._analyze_context_patterns(context)
        
        # Calculate overall instability
        instability = max(collapse_risk, drift_risk, isolation_risk, inconsistency_risk)
        
        # Determine if grounding is needed
        needs_grounding = (
            collapse_risk > self.config.collapse_threshold or
            coherence < self.config.ground_threshold or
            instability > 0.6
        )
        
        return {
            'collapse_risk': collapse_risk,
            'drift_risk': drift_risk,
            'isolation_risk': isolation_risk,
            'inconsistency_risk': inconsistency_risk,
            'instability_index': instability,
            'needs_grounding': float(needs_grounding),
            'analysis_confidence': self._calculate_analysis_confidence(components),
            **context_risks
        }
    
    def _apply_grounding_if_needed(
        self,
        coherence: float,
        components: GCTComponents,
        analysis_metrics: Dict[str, float],
        context: Dict
    ) -> Tuple[Dict[str, float], float]:
        """Apply grounding transformation if needed"""
        needs_grounding = analysis_metrics['needs_grounding'] > 0.5
        
        if not needs_grounding:
            return {
                'grounding_applied': 0.0,
                'grounding_strength': 0.0,
                'grounding_iterations': 0,
                'coherence_adjustment': 0.0
            }, coherence
        
        # Apply grounding transformation
        grounding_strength = self._calculate_grounding_strength(analysis_metrics)
        grounded_coherence = self._apply_grounding_transformation(
            coherence, components, grounding_strength, analysis_metrics
        )
        
        # Iterative grounding if needed
        iterations = 1
        while (grounded_coherence < self.config.ground_threshold and 
               iterations < self.config.max_grounding_iterations):
            grounding_strength *= 1.2  # Increase grounding strength
            grounded_coherence = self._apply_grounding_transformation(
                grounded_coherence, components, grounding_strength, analysis_metrics
            )
            iterations += 1
        
        grounding_metrics = {
            'grounding_applied': 1.0,
            'grounding_strength': grounding_strength,
            'grounding_iterations': iterations,
            'coherence_adjustment': grounded_coherence - coherence,
            'grounding_effectiveness': min(1.0, grounded_coherence / max(coherence, 0.1))
        }
        
        return grounding_metrics, grounded_coherence
    
    def _calculate_collapse_risk(self, psi: float, q: float, rho: float) -> float:
        """Calculate identity collapse risk"""
        # Basic multiplicative risk
        basic_risk = max(0, 1.0 - (psi * q * rho))
        
        # Enhanced risk factors
        critical_thresholds = {'psi': 0.2, 'q': 0.15, 'rho': 0.3}
        
        critical_risk = 0.0
        if psi < critical_thresholds['psi']:
            critical_risk += 0.4
        if q < critical_thresholds['q']:
            critical_risk += 0.3
        if rho < critical_thresholds['rho']:
            critical_risk += 0.2
        
        return min(1.0, basic_risk + critical_risk)
    
    def _calculate_drift_risk(self, coherence: float, rho: float) -> float:
        """Calculate coherence drift risk"""
        # Drift occurs when coherence is unstable and wisdom is low
        base_drift = abs(0.5 - coherence) * (1 - rho)
        
        # Additional drift from extreme values
        extreme_drift = 0.0
        if coherence < 0.2 or coherence > 0.9:
            extreme_drift = 0.3
        
        return min(1.0, base_drift + extreme_drift)
    
    def _calculate_isolation_risk(self, f: float) -> float:
        """Calculate social isolation risk"""
        # Higher risk with lower social belonging
        isolation_risk = max(0, 0.5 - f) * 2
        
        # Critical isolation threshold
        if f < 0.2:
            isolation_risk += 0.3
        
        return min(1.0, isolation_risk)
    
    def _calculate_inconsistency_risk(self, psi: float, components: GCTComponents) -> float:
        """Calculate internal inconsistency risk"""
        # Base inconsistency from low psi
        base_inconsistency = max(0, 0.4 - psi) * 2
        
        # Inconsistency from component imbalance
        component_values = [components.psi, components.rho, components.q, components.f]
        std_dev = np.std(component_values)
        imbalance_inconsistency = min(0.4, std_dev * 1.5)
        
        return min(1.0, base_inconsistency + imbalance_inconsistency)
    
    def _analyze_context_patterns(self, context: Dict) -> Dict[str, float]:
        """Analyze contextual patterns for risks"""
        if not context or 'text' not in context:
            return {
                'toxic_pattern_risk': 0.0,
                'recursive_pattern_risk': 0.0,
                'context_coherence': 0.5
            }
        
        text = context['text'].lower()
        
        # Count toxic patterns
        toxic_count = sum(1 for pattern in self.config.toxic_patterns if pattern in text)
        toxic_risk = min(1.0, toxic_count * 0.2)
        
        # Count recursive patterns
        recursive_count = sum(1 for pattern in self.config.recursive_patterns if pattern in text)
        recursive_risk = min(1.0, recursive_count * 0.3)
        
        # Estimate context coherence
        context_coherence = max(0.1, 0.8 - toxic_risk - recursive_risk)
        
        return {
            'toxic_pattern_risk': toxic_risk,
            'recursive_pattern_risk': recursive_risk,
            'context_coherence': context_coherence
        }
    
    def _calculate_component_balance(self, components: GCTComponents) -> float:
        """Calculate how balanced the components are"""
        values = [components.psi, components.rho, components.q, components.f]
        std_dev = np.std(values)
        # Convert to balance score (0 = imbalanced, 1 = perfectly balanced)
        return max(0.0, 1.0 - std_dev * 2)
    
    def _calculate_analysis_confidence(self, components: GCTComponents) -> float:
        """Calculate confidence in analysis"""
        # Higher confidence with higher component values
        min_component = min(components.psi, components.rho, components.q, components.f)
        avg_component = (components.psi + components.rho + components.q + components.f) / 4
        
        confidence = (min_component * 0.3 + avg_component * 0.7)
        return min(1.0, confidence)
    
    def _calculate_grounding_strength(self, analysis_metrics: Dict[str, float]) -> float:
        """Calculate appropriate grounding strength"""
        # Base strength from instability
        base_strength = analysis_metrics['instability_index'] * self.config.grounding_boost_factor
        
        # Additional strength for specific risks
        if analysis_metrics['collapse_risk'] > 0.7:
            base_strength += 0.3
        if analysis_metrics['drift_risk'] > 0.6:
            base_strength += 0.2
        if analysis_metrics['isolation_risk'] > 0.6:
            base_strength += 0.2
        
        return min(0.5, base_strength)  # Cap at 0.5 to avoid over-grounding
    
    def _apply_grounding_transformation(
        self,
        coherence: float,
        components: GCTComponents,
        grounding_strength: float,
        analysis_metrics: Dict[str, float]
    ) -> float:
        """Apply grounding transformation to coherence"""
        # Minimum grounded coherence
        min_grounded = self.config.ground_threshold
        
        # Adaptive grounding based on specific issues
        if analysis_metrics['collapse_risk'] > 0.7:
            # Severe collapse risk - significant boost
            boost = grounding_strength * 1.5
        elif analysis_metrics['drift_risk'] > 0.6:
            # Moderate drift - moderate boost
            boost = grounding_strength * 1.2
        elif analysis_metrics['isolation_risk'] > 0.6:
            # Social isolation - targeted boost
            boost = grounding_strength * 1.1
        else:
            # General instability - standard boost
            boost = grounding_strength
        
        # Apply grounding with wisdom modulation
        wisdom_factor = 1.0 + (components.rho - 0.5) * 0.3  # Wisdom helps grounding
        effective_boost = boost * wisdom_factor
        
        # Calculate grounded coherence
        grounded = coherence + effective_boost
        
        # Ensure minimum grounding
        grounded = max(min_grounded, grounded)
        
        # Cap at reasonable maximum to avoid overgrounding
        grounded = min(grounded, 0.95)
        
        return grounded
    
    def _get_phases_applied(self, analysis_metrics: Dict, grounding_metrics: Dict) -> int:
        """Get number of triadic phases applied"""
        phases = 2  # Always apply Generate and Analyze
        if grounding_metrics.get('grounding_applied', 0) > 0:
            phases += 1
        return phases
    
    def _record_processing(self, original_coherence: float, final_coherence: float, metrics: Dict):
        """Record processing for learning and adaptation"""
        processing_record = {
            'original_coherence': original_coherence,
            'final_coherence': final_coherence,
            'improvement': final_coherence - original_coherence,
            'metrics': metrics,
            'timestamp': np.datetime64('now')
        }
        
        self.processing_history.append(processing_record)
        
        # Keep only recent history
        if len(self.processing_history) > 1000:
            self.processing_history = self.processing_history[-1000:]
    
    def get_processing_statistics(self) -> Dict[str, float]:
        """Get statistics on triadic processing performance"""
        if not self.processing_history:
            return {}
        
        improvements = [record['improvement'] for record in self.processing_history]
        grounding_applications = [record['metrics'].get('grounding_applied', 0) 
                                for record in self.processing_history]
        
        return {
            'total_processed': len(self.processing_history),
            'average_improvement': np.mean(improvements),
            'median_improvement': np.median(improvements),
            'std_improvement': np.std(improvements),
            'grounding_frequency': np.mean(grounding_applications),
            'positive_outcomes': sum(1 for imp in improvements if imp > 0) / len(improvements)
        }
    
    def adapt_thresholds(self, target_performance: float = 0.7):
        """Adapt processing thresholds based on performance"""
        stats = self.get_processing_statistics()
        
        if not stats or stats['total_processed'] < 50:
            return  # Not enough data
        
        # Adjust grounding threshold based on performance
        if stats['positive_outcomes'] < target_performance:
            # Lower threshold to apply grounding more frequently
            self.config.ground_threshold = min(0.4, self.config.ground_threshold * 1.1)
        elif stats['positive_outcomes'] > target_performance + 0.1:
            # Raise threshold to apply grounding less frequently
            self.config.ground_threshold = max(0.2, self.config.ground_threshold * 0.9)
        
        # Adjust grounding strength based on average improvement
        if stats['average_improvement'] < 0.05:
            # Increase grounding strength
            self.config.grounding_boost_factor = min(0.4, self.config.grounding_boost_factor * 1.1)
        elif stats['average_improvement'] > 0.15:
            # Decrease grounding strength
            self.config.grounding_boost_factor = max(0.1, self.config.grounding_boost_factor * 0.9)


class AdvancedTriadicProcessor(TriadicProcessor):
    """Advanced triadic processor with neural network capabilities"""
    
    def __init__(self, config: Optional[TriadicConfig] = None, model_path: Optional[str] = None):
        super().__init__(config)
        self.neural_processor = None
        self.model_path = model_path
        self._load_neural_model()
    
    def _load_neural_model(self):
        """Load neural network for advanced processing"""
        if self.model_path and self.config.enable_neural_processing:
            try:
                # In a real implementation, this would load a PyTorch model
                self.neural_processor = "neural_model_placeholder"
            except Exception as e:
                print(f"Warning: Could not load neural model: {e}")
                self.neural_processor = None
    
    def process_with_neural_enhancement(
        self,
        coherence: float,
        components: GCTComponents,
        context: Optional[Dict] = None
    ) -> Tuple[float, Dict[str, float]]:
        """Process with neural network enhancement"""
        if not self.neural_processor:
            return self.process(coherence, components, context)
        
        # Apply standard triadic processing first
        base_coherence, base_metrics = self.process(coherence, components, context)
        
        # Apply neural enhancement
        neural_metrics = self._apply_neural_processing(base_coherence, components, context)
        
        # Combine results
        enhanced_coherence = base_coherence * neural_metrics['enhancement_factor']
        combined_metrics = {**base_metrics, **neural_metrics}
        
        return enhanced_coherence, combined_metrics
    
    def _apply_neural_processing(
        self,
        coherence: float,
        components: GCTComponents,
        context: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Apply neural network processing (placeholder)"""
        # In a real implementation, this would use a trained neural network
        # For now, we'll use a simple heuristic
        
        # Neural "enhancement" based on component patterns
        pattern_strength = self._detect_neural_patterns(components)
        enhancement_factor = 1.0 + (pattern_strength * 0.1)
        
        return {
            'neural_enhancement_applied': 1.0,
            'pattern_strength': pattern_strength,
            'enhancement_factor': enhancement_factor,
            'neural_confidence': 0.8  # Placeholder confidence
        }
    
    def _detect_neural_patterns(self, components: GCTComponents) -> float:
        """Detect patterns using neural-like processing"""
        # Simple pattern detection based on component relationships
        psi, rho, q, f = components.psi, components.rho, components.q, components.f
        
        # Pattern 1: Wisdom-consistency synergy
        synergy_1 = psi * rho
        
        # Pattern 2: Moral-social alignment
        alignment_2 = q * f
        
        # Pattern 3: Balanced growth
        balance_3 = 1.0 - abs(psi - rho) - abs(q - f)
        
        # Combine patterns
        pattern_strength = (synergy_1 + alignment_2 + balance_3) / 3
        
        return min(1.0, pattern_strength)