#!/usr/bin/env python3
"""
Enhanced Coherence Tracker - Advanced monitoring with safety integration
Incorporates howlround detection, pressure monitoring, and predictive analysis
"""
from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import numpy as np
from enum import Enum
import asyncio
import json

from .howlround_detector import HowlroundDetector, ResonanceType


class CoherenceState(Enum):
    """Enhanced coherence state classifications with safety levels"""
    FRAGMENTED = ("fragmented", 0.1)      # Ψ < 0.3, high risk
    UNSTABLE = ("unstable", 0.3)          # 0.3 ≤ Ψ < 0.5, medium risk  
    EMERGING = ("emerging", 0.5)          # 0.5 ≤ Ψ < 0.7, low risk
    STABLE = ("stable", 0.7)              # 0.7 ≤ Ψ < 0.9, minimal risk
    HARMONIZED = ("harmonized", 0.9)      # Ψ ≥ 0.9, optimal
    
    def __init__(self, label: str, safety_threshold: float):
        self.label = label
        self.safety_threshold = safety_threshold


@dataclass
class PressureMetrics:
    """Tracks ambiguity and unresolved requests pressure"""
    ambiguity_level: float = 0.0
    unresolved_count: int = 0
    pressure_score: float = 0.0
    time_under_pressure: float = 0.0
    last_resolution: Optional[datetime] = None


@dataclass
class SafetyMetrics:
    """Comprehensive safety tracking metrics"""
    howlround_risk: float = 0.0
    ghost_pattern_detected: bool = False
    pressure_critical: bool = False
    coherence_stable: bool = True
    intervention_needed: bool = False
    safety_score: float = 1.0  # 1.0 = safe, 0.0 = critical


@dataclass
class EnhancedCoherenceSnapshot:
    """Extended snapshot with safety and quality metrics"""
    timestamp: datetime
    psi_value: float
    state: CoherenceState
    stability_score: float
    recent_delta: float
    resonance_frequency: float
    safety_metrics: SafetyMetrics
    pressure_metrics: PressureMetrics
    relationship_quality: float = 0.0
    intervention_log: List[str] = field(default_factory=list)


class EnhancedCoherenceTracker:
    """
    Advanced coherence tracking with integrated safety monitoring,
    pressure detection, and relationship quality assessment.
    """
    
    def __init__(self, initial_psi: float = 1.0, personal_chain_id: Optional[str] = None):
        # Core coherence tracking
        self.current_psi: float = initial_psi
        self.history: deque = deque(maxlen=2000)
        self.snapshots: List[EnhancedCoherenceSnapshot] = []
        
        # Safety systems
        self.howlround_detector = HowlroundDetector()
        self.pressure_metrics = PressureMetrics()
        self.safety_metrics = SafetyMetrics()
        
        # Thresholds and parameters
        self.warning_threshold: float = 0.3
        self.critical_threshold: float = 0.15
        self.pressure_threshold: float = 0.6
        self.intervention_threshold: float = 0.4
        
        # Personal blockchain reference
        self.personal_chain_id = personal_chain_id
        
        # Relationship quality tracking
        self.interaction_count: int = 0
        self.positive_interactions: int = 0
        self.coherence_improvements: int = 0
        
        # Callbacks for interventions
        self.intervention_callbacks: List[Callable] = []
        
        # Initialize with first snapshot
        self._take_enhanced_snapshot()
    
    async def update_coherence_with_safety(
        self, 
        delta_psi: float, 
        user_input: str,
        ai_response: str,
        event_type: str,
        trigger: str,
        ambiguity_level: float = 0.0
    ) -> EnhancedCoherenceSnapshot:
        """
        Update coherence with comprehensive safety analysis.
        
        Args:
            delta_psi: Change in coherence
            user_input: User's message for howlround detection
            ai_response: AI's response for pattern analysis
            event_type: Type of interaction event
            trigger: What triggered the update
            ambiguity_level: Level of ambiguity in interaction (0-1)
            
        Returns:
            Enhanced coherence snapshot with safety analysis
        """
        # Update basic coherence
        old_psi = self.current_psi
        self.current_psi = max(0.0, min(2.0, self.current_psi + delta_psi))
        
        # Track interaction quality
        self.interaction_count += 1
        if delta_psi > 0:
            self.positive_interactions += 1
            self.coherence_improvements += 1
        
        # Update pressure metrics
        self._update_pressure_metrics(ambiguity_level, delta_psi)
        
        # Perform howlround detection
        howlround_analysis = self.howlround_detector.analyze_interaction(
            user_input, ai_response, self.current_psi
        )
        
        # Update safety metrics
        self._update_safety_metrics(howlround_analysis, old_psi)
        
        # Record in history
        self.history.append({
            'timestamp': datetime.now(),
            'psi': self.current_psi,
            'delta': delta_psi,
            'event_type': event_type,
            'safety_score': self.safety_metrics.safety_score,
            'pressure': self.pressure_metrics.pressure_score
        })
        
        # Check for interventions
        interventions = await self._check_interventions(old_psi, howlround_analysis)
        
        # Take enhanced snapshot
        snapshot = self._take_enhanced_snapshot(interventions)
        
        # Trigger callbacks if needed
        if self.safety_metrics.intervention_needed:
            await self._trigger_intervention_callbacks(snapshot)
        
        return snapshot
    
    def _update_pressure_metrics(self, ambiguity_level: float, delta_psi: float):
        """Update pressure metrics based on ambiguity and coherence changes."""
        self.pressure_metrics.ambiguity_level = ambiguity_level
        
        # Increase pressure for negative coherence changes with high ambiguity
        if delta_psi < 0 and ambiguity_level > 0.5:
            self.pressure_metrics.pressure_score += ambiguity_level * abs(delta_psi)
            self.pressure_metrics.unresolved_count += 1
        else:
            # Decrease pressure for positive interactions
            self.pressure_metrics.pressure_score *= 0.9
            if delta_psi > 0.1:
                self.pressure_metrics.unresolved_count = max(0, self.pressure_metrics.unresolved_count - 1)
                self.pressure_metrics.last_resolution = datetime.now()
        
        # Track time under pressure
        if self.pressure_metrics.pressure_score > self.pressure_threshold:
            self.pressure_metrics.time_under_pressure += 1
        else:
            self.pressure_metrics.time_under_pressure = 0
        
        # Critical pressure check
        self.pressure_metrics.pressure_score = min(1.0, self.pressure_metrics.pressure_score)
        self.safety_metrics.pressure_critical = (
            self.pressure_metrics.pressure_score > 0.8 or
            self.pressure_metrics.time_under_pressure > 5
        )
    
    def _update_safety_metrics(self, howlround_analysis: Dict, old_psi: float):
        """Update safety metrics based on howlround and coherence analysis."""
        self.safety_metrics.howlround_risk = howlround_analysis['risk_level']
        self.safety_metrics.ghost_pattern_detected = howlround_analysis['ghost_echo_detected']
        
        # Check coherence stability
        self.safety_metrics.coherence_stable = (
            self.current_psi > self.warning_threshold and
            abs(self.current_psi - old_psi) < 0.2
        )
        
        # Calculate overall safety score
        safety_factors = [
            1.0 - self.safety_metrics.howlround_risk,
            1.0 if self.safety_metrics.coherence_stable else 0.5,
            1.0 - self.pressure_metrics.pressure_score,
            0.3 if self.safety_metrics.ghost_pattern_detected else 1.0
        ]
        
        self.safety_metrics.safety_score = np.mean(safety_factors)
        
        # Determine if intervention needed
        self.safety_metrics.intervention_needed = (
            self.safety_metrics.safety_score < self.intervention_threshold or
            self.current_psi < self.critical_threshold or
            self.safety_metrics.pressure_critical
        )
    
    async def _check_interventions(self, old_psi: float, howlround_analysis: Dict) -> List[str]:
        """Check and log necessary interventions."""
        interventions = []
        
        # Rapid coherence drop
        if old_psi - self.current_psi > 0.2:
            interventions.append(f"Rapid coherence drop detected: {old_psi:.2f} → {self.current_psi:.2f}")
        
        # Critical coherence level
        if self.current_psi < self.critical_threshold:
            interventions.append(f"CRITICAL: Coherence at {self.current_psi:.2f} - immediate stabilization needed")
        
        # Howlround intervention
        if howlround_analysis['risk_level'] > 0.7:
            interventions.extend(howlround_analysis['recommendations'])
        
        # Pressure intervention
        if self.safety_metrics.pressure_critical:
            interventions.append("High ambiguity pressure - clarification dialogue recommended")
            interventions.append("Consider simplifying responses and confirming understanding")
        
        # Ghost pattern intervention
        if self.safety_metrics.ghost_pattern_detected:
            interventions.append("Ghost echo pattern detected - vary response patterns")
            interventions.append("Introduce novel perspectives to break recursive loops")
        
        return interventions
    
    def _take_enhanced_snapshot(self, interventions: List[str] = None) -> EnhancedCoherenceSnapshot:
        """Create comprehensive snapshot with all metrics."""
        # Calculate relationship quality
        if self.interaction_count > 0:
            relationship_quality = (
                (self.positive_interactions / self.interaction_count) * 0.4 +
                (self.coherence_improvements / self.interaction_count) * 0.3 +
                self.safety_metrics.safety_score * 0.3
            )
        else:
            relationship_quality = 0.5
        
        snapshot = EnhancedCoherenceSnapshot(
            timestamp=datetime.now(),
            psi_value=self.current_psi,
            state=self._classify_state(self.current_psi),
            stability_score=self._calculate_stability(),
            recent_delta=self._calculate_recent_delta(),
            resonance_frequency=self._calculate_resonance(),
            safety_metrics=SafetyMetrics(
                howlround_risk=self.safety_metrics.howlround_risk,
                ghost_pattern_detected=self.safety_metrics.ghost_pattern_detected,
                pressure_critical=self.safety_metrics.pressure_critical,
                coherence_stable=self.safety_metrics.coherence_stable,
                intervention_needed=self.safety_metrics.intervention_needed,
                safety_score=self.safety_metrics.safety_score
            ),
            pressure_metrics=PressureMetrics(
                ambiguity_level=self.pressure_metrics.ambiguity_level,
                unresolved_count=self.pressure_metrics.unresolved_count,
                pressure_score=self.pressure_metrics.pressure_score,
                time_under_pressure=self.pressure_metrics.time_under_pressure,
                last_resolution=self.pressure_metrics.last_resolution
            ),
            relationship_quality=relationship_quality,
            intervention_log=interventions or []
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def _classify_state(self, psi: float) -> CoherenceState:
        """Classify coherence value into enhanced state category."""
        if psi < 0.3:
            return CoherenceState.FRAGMENTED
        elif psi < 0.5:
            return CoherenceState.UNSTABLE
        elif psi < 0.7:
            return CoherenceState.EMERGING
        elif psi < 0.9:
            return CoherenceState.STABLE
        else:
            return CoherenceState.HARMONIZED
    
    def _calculate_stability(self) -> float:
        """Calculate stability with safety considerations."""
        if len(self.history) < 3:
            return 0.5
            
        recent = list(self.history)[-20:]
        psi_values = [h['psi'] for h in recent]
        safety_values = [h.get('safety_score', 1.0) for h in recent]
        
        # Combine coherence variance and safety scores
        psi_variance = np.var(psi_values)
        avg_safety = np.mean(safety_values)
        
        # Stability is inverse of variance, weighted by safety
        base_stability = 1.0 / (1.0 + psi_variance * 10)
        weighted_stability = base_stability * avg_safety
        
        return weighted_stability
    
    def _calculate_recent_delta(self) -> float:
        """Calculate recent coherence change."""
        if len(self.history) < 2:
            return 0.0
            
        recent = list(self.history)[-10:]
        return sum(h['delta'] for h in recent)
    
    def _calculate_resonance(self) -> float:
        """Enhanced resonance calculation including safety factors."""
        if len(self.history) < 10:
            return 0.0
            
        recent = list(self.history)[-30:]
        psi_values = [h['psi'] for h in recent]
        
        # Standard resonance calculation
        mean_psi = np.mean(psi_values)
        centered = [v - mean_psi for v in psi_values]
        
        zero_crossings = sum(1 for i in range(1, len(centered)) 
                           if centered[i-1] * centered[i] < 0)
        
        base_resonance = zero_crossings / len(centered)
        
        # Adjust for safety concerns
        safety_factor = self.safety_metrics.safety_score
        adjusted_resonance = base_resonance * safety_factor
        
        # Optimal resonance is moderate frequency
        optimal = 0.3
        return 1.0 - abs(adjusted_resonance - optimal) / optimal
    
    async def _trigger_intervention_callbacks(self, snapshot: EnhancedCoherenceSnapshot):
        """Trigger registered intervention callbacks."""
        for callback in self.intervention_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(snapshot)
                else:
                    callback(snapshot)
            except Exception as e:
                print(f"Intervention callback error: {e}")
    
    def register_intervention_callback(self, callback: Callable):
        """Register a callback for intervention events."""
        self.intervention_callbacks.append(callback)
    
    def get_relationship_assessment(self) -> Dict:
        """Generate comprehensive relationship quality assessment."""
        if not self.snapshots:
            return {'status': 'no_data', 'quality': 0.0}
        
        current = self.snapshots[-1]
        
        # Analyze relationship trajectory
        if len(self.snapshots) > 10:
            recent_quality = [s.relationship_quality for s in self.snapshots[-10:]]
            quality_trend = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
        else:
            quality_trend = 0.0
        
        # Determine relationship status
        if current.relationship_quality > 0.8 and current.safety_metrics.safety_score > 0.8:
            status = "excellent"
        elif current.relationship_quality > 0.6 and current.safety_metrics.safety_score > 0.6:
            status = "good"
        elif current.relationship_quality > 0.4 or current.safety_metrics.safety_score < 0.4:
            status = "needs_attention"
        else:
            status = "critical"
        
        return {
            'status': status,
            'quality': current.relationship_quality,
            'trend': 'improving' if quality_trend > 0.01 else 'declining' if quality_trend < -0.01 else 'stable',
            'coherence_state': current.state.label,
            'safety_score': current.safety_metrics.safety_score,
            'interaction_count': self.interaction_count,
            'positive_ratio': self.positive_interactions / max(self.interaction_count, 1),
            'current_risks': {
                'howlround': current.safety_metrics.howlround_risk,
                'pressure': current.pressure_metrics.pressure_score,
                'ghost_patterns': current.safety_metrics.ghost_pattern_detected
            },
            'recommendations': self._generate_relationship_recommendations(current)
        }
    
    def _generate_relationship_recommendations(self, snapshot: EnhancedCoherenceSnapshot) -> List[str]:
        """Generate recommendations for improving user-AI relationship."""
        recommendations = []
        
        if snapshot.relationship_quality < 0.5:
            recommendations.append("Focus on building trust through consistent, helpful interactions")
        
        if snapshot.safety_metrics.howlround_risk > 0.5:
            recommendations.append("Vary interaction patterns to prevent feedback loops")
        
        if snapshot.pressure_metrics.pressure_score > 0.6:
            recommendations.append("Clarify ambiguous requests promptly")
            recommendations.append("Use confirmation loops for complex tasks")
        
        if snapshot.state == CoherenceState.FRAGMENTED:
            recommendations.append("Prioritize coherence restoration through grounding exercises")
        
        if self.positive_interactions / max(self.interaction_count, 1) < 0.5:
            recommendations.append("Increase positive, constructive interactions")
        
        return recommendations
    
    def export_enhanced_data(self) -> Dict:
        """Export comprehensive tracking data including safety metrics."""
        return {
            'current_psi': self.current_psi,
            'relationship_quality': self.snapshots[-1].relationship_quality if self.snapshots else 0.0,
            'safety_status': {
                'overall_score': self.safety_metrics.safety_score,
                'intervention_needed': self.safety_metrics.intervention_needed,
                'active_risks': {
                    'howlround': self.safety_metrics.howlround_risk,
                    'ghost_patterns': self.safety_metrics.ghost_pattern_detected,
                    'pressure_critical': self.safety_metrics.pressure_critical
                }
            },
            'interaction_stats': {
                'total': self.interaction_count,
                'positive': self.positive_interactions,
                'improvements': self.coherence_improvements
            },
            'snapshots': [
                {
                    'timestamp': s.timestamp.isoformat(),
                    'psi': s.psi_value,
                    'state': s.state.label,
                    'safety_score': s.safety_metrics.safety_score,
                    'relationship_quality': s.relationship_quality,
                    'interventions': s.intervention_log
                }
                for s in self.snapshots[-100:]  # Last 100 snapshots
            ],
            'personal_chain_id': self.personal_chain_id
        }