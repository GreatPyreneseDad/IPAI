"""
GCT Calculator

Enhanced GCT coherence calculator with individual parameters and
machine learning enhancements.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import math
from datetime import datetime

from ..models.coherence_profile import (
    GCTComponents, IndividualParameters, CoherenceProfile, CoherenceLevel,
    get_coherence_level
)


class GCTCalculator:
    """Enhanced GCT coherence calculator with individual parameters"""
    
    def __init__(self, triadic_processor: Optional['TriadicProcessor'] = None):
        self.triadic_processor = triadic_processor
        self._cache = {}  # LRU cache for performance
        self._cache_size = 1000
        
    def calculate_coherence(
        self,
        components: GCTComponents,
        parameters: IndividualParameters,
        apply_triadic: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate enhanced coherence with individual parameters
        
        Args:
            components: GCT components (psi, rho, q, f)
            parameters: Individual optimization parameters (k_m, k_i)
            apply_triadic: Whether to apply triadic logic processing
            
        Returns:
            Tuple of (coherence_score, detailed_metrics)
        """
        # Check cache first
        cache_key = self._get_cache_key(components, parameters, apply_triadic)
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Extract values
        psi, rho, q, f = components.psi, components.rho, components.q, components.f
        k_m, k_i = parameters.k_m, parameters.k_i
        
        # Calculate optimal q with individual parameters
        q_optimal = self._calculate_optimal_q(q, k_m, k_i)
        
        # Base coherence calculation using enhanced SoulMath
        base_coherence = self._calculate_base_coherence(psi, rho, q_optimal, f)
        
        # Enhanced coupling terms
        coupling_terms = self._calculate_coupling_terms(psi, rho, q_optimal, f)
        
        # Total coherence
        coherence = base_coherence + coupling_terms['total_coupling']
        
        # Apply triadic processing if enabled
        if apply_triadic and self.triadic_processor:
            coherence, triadic_metrics = self.triadic_processor.process(
                coherence, components
            )
        else:
            triadic_metrics = {}
        
        # Compile detailed metrics
        detailed_metrics = {
            'base_coherence': base_coherence,
            'q_optimal': q_optimal,
            'coupling_terms': coupling_terms,
            'soul_echo': components.soul_echo,
            'collapse_risk': self._calculate_collapse_risk(psi, q, rho),
            'stability_index': self._calculate_stability_index(psi, rho, q_optimal, f),
            'growth_potential': self._calculate_growth_potential(psi, rho, q, f),
            'resonance_factor': self._calculate_resonance_factor(psi, rho, q_optimal, f),
            **triadic_metrics
        }
        
        # Cache result
        result = (coherence, detailed_metrics)
        self._update_cache(cache_key, result)
        
        return result
    
    def _calculate_optimal_q(self, q: float, k_m: float, k_i: float) -> float:
        """Calculate biologically optimized moral activation"""
        q_max = 1.0
        
        # Enhanced biological optimization with sigmoidal activation
        numerator = q_max * q * (1 + 0.1 * np.tanh(5 * (q - 0.5)))
        denominator = k_m + q + (q**2) / k_i
        
        return numerator / denominator
    
    def _calculate_base_coherence(self, psi: float, rho: float, q_optimal: float, f: float) -> float:
        """Calculate base coherence using enhanced SoulMath"""
        # Core coherence components
        internal_consistency = psi
        wisdom_modulated_consistency = rho * psi
        moral_activation = q_optimal
        social_grounding = f * psi
        
        # Base coherence with non-linear enhancements
        base = (internal_consistency + 
                wisdom_modulated_consistency + 
                moral_activation + 
                social_grounding)
        
        # Apply harmonic enhancement for balanced profiles
        balance_factor = 1.0 + 0.1 * (1 - self._calculate_imbalance(psi, rho, q_optimal, f))
        
        return base * balance_factor
    
    def _calculate_coupling_terms(self, psi: float, rho: float, q_optimal: float, f: float) -> Dict[str, float]:
        """Calculate enhanced coupling terms"""
        # Wisdom-activation coupling
        wisdom_activation = 0.15 * rho * q_optimal
        
        # Consistency-belonging coupling
        consistency_belonging = 0.1 * psi * f
        
        # Triple coupling (wisdom-moral-social)
        triple_coupling = 0.05 * rho * q_optimal * f
        
        # Resonance coupling (all four components)
        resonance_coupling = 0.02 * psi * rho * q_optimal * f
        
        total_coupling = (wisdom_activation + 
                         consistency_belonging + 
                         triple_coupling + 
                         resonance_coupling)
        
        return {
            'wisdom_activation': wisdom_activation,
            'consistency_belonging': consistency_belonging,
            'triple_coupling': triple_coupling,
            'resonance_coupling': resonance_coupling,
            'total_coupling': total_coupling
        }
    
    def _calculate_collapse_risk(self, psi: float, q: float, rho: float) -> float:
        """Calculate identity collapse risk"""
        # Basic collapse risk
        basic_risk = max(0, 1.0 - (psi * q * rho))
        
        # Enhanced risk factors
        consistency_risk = max(0, 0.3 - psi) * 2  # High risk if psi < 0.3
        moral_drift_risk = max(0, 0.2 - q) * 3    # High risk if q < 0.2
        wisdom_deficit_risk = max(0, 0.4 - rho) * 1.5  # Risk if rho < 0.4
        
        total_risk = min(1.0, basic_risk + consistency_risk + moral_drift_risk + wisdom_deficit_risk)
        
        return total_risk
    
    def _calculate_stability_index(self, psi: float, rho: float, q_optimal: float, f: float) -> float:
        """Calculate coherence stability index"""
        # Stability comes from wisdom-modulated consistency
        wisdom_stability = psi * rho
        
        # Social stability from belonging
        social_stability = f * 0.5
        
        # Moral stability from optimal activation
        moral_stability = q_optimal * 0.3
        
        # Combined stability with diminishing returns
        total_stability = wisdom_stability + social_stability + moral_stability
        return min(1.0, total_stability)
    
    def _calculate_growth_potential(self, psi: float, rho: float, q: float, f: float) -> float:
        """Calculate potential for coherence growth"""
        # Growth potential is higher when components are unbalanced but not too low
        component_min = min(psi, rho, q, f)
        component_max = max(psi, rho, q, f)
        
        # Avoid very low components (< 0.2) as they indicate crisis
        if component_min < 0.2:
            return 0.1
        
        # Growth potential from imbalance (up to a point)
        imbalance = component_max - component_min
        growth_from_imbalance = min(0.4, imbalance * 0.5)
        
        # Growth potential from room for improvement
        average_component = (psi + rho + q + f) / 4
        growth_from_room = (1.0 - average_component) * 0.6
        
        return min(1.0, growth_from_imbalance + growth_from_room)
    
    def _calculate_resonance_factor(self, psi: float, rho: float, q_optimal: float, f: float) -> float:
        """Calculate resonance factor - how well components work together"""
        # Resonance is high when all components are similar and moderately high
        components = [psi, rho, q_optimal, f]
        mean_component = np.mean(components)
        std_component = np.std(components)
        
        # Penalty for too much variation
        variation_penalty = std_component * 2
        
        # Bonus for moderate-to-high levels
        level_bonus = min(0.3, max(0, mean_component - 0.5))
        
        resonance = max(0, mean_component - variation_penalty + level_bonus)
        return min(1.0, resonance)
    
    def _calculate_imbalance(self, psi: float, rho: float, q: float, f: float) -> float:
        """Calculate component imbalance (0 = balanced, 1 = highly imbalanced)"""
        components = [psi, rho, q, f]
        std_dev = np.std(components)
        # Normalize to [0, 1] range (max std dev for [0,1] range is 0.5)
        return min(1.0, std_dev * 2)
    
    def calculate_derivatives(
        self,
        history: List[CoherenceProfile],
        window_size: int = 5
    ) -> Dict[str, float]:
        """Calculate coherence derivatives from history"""
        if len(history) < 2:
            return {}
        
        # Extract time series
        recent_history = history[-window_size:]
        coherence_values = [p.coherence_score for p in recent_history]
        timestamps = [p.components.timestamp.timestamp() for p in recent_history]
        
        # Calculate derivatives using finite differences
        if len(coherence_values) > 1:
            dt = np.diff(timestamps)
            dc = np.diff(coherence_values)
            
            # Primary derivative (rate of change)
            dC_dt = float(np.mean(dc / dt)) if len(dt) > 0 and np.all(dt > 0) else 0.0
            
            # Volatility (standard deviation of changes)
            volatility = float(np.std(dc / dt)) if len(dt) > 0 and np.all(dt > 0) else 0.0
            
            # Acceleration (second derivative)
            if len(dc) > 1:
                d2c_dt2 = float(np.mean(np.diff(dc / dt[:-1]) / dt[1:])) if len(dt) > 1 else 0.0
            else:
                d2c_dt2 = 0.0
        else:
            dC_dt = 0.0
            volatility = 0.0
            d2c_dt2 = 0.0
        
        derivatives = {
            'dC_dt': dC_dt,
            'volatility': volatility,
            'acceleration': d2c_dt2,
            'trend_strength': abs(dC_dt) / (volatility + 1e-6)  # Avoid division by zero
        }
        
        # Component derivatives
        for component in ['psi', 'rho', 'q', 'f']:
            values = [getattr(p.components, component) for p in recent_history]
            if len(values) > 1:
                dv = np.diff(values)
                dt_comp = np.diff([p.components.timestamp.timestamp() for p in recent_history])
                derivatives[f'd{component}_dt'] = float(np.mean(dv / dt_comp)) if len(dt_comp) > 0 and np.all(dt_comp > 0) else 0.0
        
        return derivatives
    
    def _get_cache_key(self, components: GCTComponents, parameters: IndividualParameters, apply_triadic: bool) -> str:
        """Generate cache key for coherence calculation"""
        return f"{components.psi:.3f}_{components.rho:.3f}_{components.q:.3f}_{components.f:.3f}_{parameters.k_m:.3f}_{parameters.k_i:.3f}_{apply_triadic}"
    
    def _update_cache(self, key: str, value: Tuple[float, Dict[str, float]]):
        """Update cache with LRU eviction"""
        if len(self._cache) >= self._cache_size:
            # Remove oldest entry (simple FIFO for now)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value


class EnhancedGCTCalculator(GCTCalculator):
    """GCT Calculator with ML-enhanced predictions and advanced analytics"""
    
    def __init__(self, model_path: Optional[str] = None):
        super().__init__()
        self.prediction_model = None
        self.model_path = model_path
        self._load_prediction_model()
    
    def _load_prediction_model(self):
        """Load pre-trained coherence prediction model"""
        if self.model_path:
            try:
                # In a real implementation, this would load a PyTorch model
                # For now, we'll use a simple statistical model
                self.prediction_model = "statistical_model"
            except Exception as e:
                print(f"Warning: Could not load prediction model: {e}")
                self.prediction_model = None
    
    def predict_trajectory(
        self,
        current_profile: CoherenceProfile,
        horizon_days: int = 30
    ) -> Dict[str, np.ndarray]:
        """Predict future coherence trajectory"""
        if not self.prediction_model:
            return self._simple_trajectory_prediction(current_profile, horizon_days)
        
        # If we had a real ML model, we would use it here
        return self._simple_trajectory_prediction(current_profile, horizon_days)
    
    def _simple_trajectory_prediction(
        self,
        current_profile: CoherenceProfile,
        horizon_days: int = 30
    ) -> Dict[str, np.ndarray]:
        """Simple statistical trajectory prediction"""
        # Generate time points
        time_points = np.linspace(0, horizon_days, horizon_days)
        
        # Base prediction: slight regression to mean with some noise
        mean_coherence = 0.5  # Population mean
        current_coherence = current_profile.coherence_score
        
        # Exponential decay towards mean
        decay_rate = 0.1  # Adjust based on stability
        predicted_coherence = (
            mean_coherence + 
            (current_coherence - mean_coherence) * np.exp(-decay_rate * time_points / 30)
        )
        
        # Add uncertainty bounds
        uncertainty = 0.1 + 0.05 * time_points / 30  # Increasing uncertainty over time
        upper_bound = predicted_coherence + uncertainty
        lower_bound = predicted_coherence - uncertainty
        
        # Generate component predictions
        components_forecast = {}
        for component in ['psi', 'rho', 'q', 'f']:
            current_value = getattr(current_profile.components, component)
            component_mean = 0.5
            
            # Similar exponential decay for each component
            predicted_component = (
                component_mean + 
                (current_value - component_mean) * np.exp(-decay_rate * time_points / 30)
            )
            components_forecast[component] = predicted_component
        
        return {
            'coherence_forecast': predicted_coherence,
            'coherence_upper': upper_bound,
            'coherence_lower': lower_bound,
            'time_points': time_points,
            'components_forecast': components_forecast,
            'model_confidence': np.maximum(0.3, 1.0 - time_points / 60)  # Decreasing confidence
        }
    
    def calculate_intervention_recommendations(
        self,
        current_profile: CoherenceProfile,
        trajectory: Dict[str, np.ndarray]
    ) -> Dict[str, any]:
        """Calculate intervention recommendations based on profile and trajectory"""
        recommendations = {
            'priority': 'low',
            'interventions': [],
            'focus_areas': [],
            'timeline': 'routine'
        }
        
        # Check current coherence level
        if current_profile.level == CoherenceLevel.CRITICAL:
            recommendations['priority'] = 'critical'
            recommendations['timeline'] = 'immediate'
            recommendations['interventions'].append({
                'type': 'crisis_intervention',
                'description': 'Immediate stabilization and grounding exercises',
                'urgency': 'high'
            })
        elif current_profile.level == CoherenceLevel.LOW:
            recommendations['priority'] = 'high'
            recommendations['timeline'] = 'within_week'
        
        # Analyze component weaknesses
        components = current_profile.components
        if components.psi < 0.3:
            recommendations['focus_areas'].append('internal_consistency')
            recommendations['interventions'].append({
                'type': 'consistency_building',
                'description': 'Exercises to align thoughts and actions',
                'urgency': 'medium'
            })
        
        if components.rho < 0.4:
            recommendations['focus_areas'].append('wisdom_development')
            recommendations['interventions'].append({
                'type': 'reflection_practice',
                'description': 'Structured reflection and learning exercises',
                'urgency': 'medium'
            })
        
        if components.q < 0.3:
            recommendations['focus_areas'].append('moral_activation')
            recommendations['interventions'].append({
                'type': 'values_clarification',
                'description': 'Values identification and commitment exercises',
                'urgency': 'medium'
            })
        
        if components.f < 0.4:
            recommendations['focus_areas'].append('social_connection')
            recommendations['interventions'].append({
                'type': 'relationship_building',
                'description': 'Social connection and community engagement',
                'urgency': 'medium'
            })
        
        # Check trajectory for concerning trends
        if 'coherence_forecast' in trajectory:
            forecast = trajectory['coherence_forecast']
            if len(forecast) > 7:  # Look at week ahead
                week_change = forecast[7] - forecast[0]
                if week_change < -0.1:  # Declining trend
                    recommendations['interventions'].append({
                        'type': 'trend_reversal',
                        'description': 'Interventions to reverse declining coherence',
                        'urgency': 'high'
                    })
        
        return recommendations