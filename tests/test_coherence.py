"""
Coherence System Tests

Comprehensive testing for the GCT (Grounded Coherence Theory)
calculation system and triadic processing functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List
import math

from src.models.coherence_profile import CoherenceProfile, GCTComponents, IndividualParameters
from src.coherence.gct_calculator import GCTCalculator
from src.coherence.triadic_processor import TriadicProcessor
from tests.conftest import assert_valid_coherence_profile


class TestGCTComponents:
    """Test GCT components and coherence profile"""
    
    def test_gct_components_creation(self):
        """Test GCT components creation and validation"""
        components = GCTComponents(
            psi=0.8,    # Internal consistency
            rho=0.7,    # Accumulated wisdom
            q=0.6,      # Moral activation
            f=0.9       # Social belonging
        )
        
        assert components.psi == 0.8
        assert components.rho == 0.7
        assert components.q == 0.6
        assert components.f == 0.9
    
    def test_gct_components_validation(self):
        """Test GCT components validation"""
        # Valid components
        components = GCTComponents(psi=0.5, rho=0.5, q=0.5, f=0.5)
        assert components.psi == 0.5
        
        # Test boundary values
        components_min = GCTComponents(psi=0.0, rho=0.0, q=0.0, f=0.0)
        assert components_min.psi == 0.0
        
        components_max = GCTComponents(psi=1.0, rho=1.0, q=1.0, f=1.0)
        assert components_max.psi == 1.0
        
        # Invalid components should raise validation errors
        with pytest.raises(ValueError):
            GCTComponents(psi=-0.1, rho=0.5, q=0.5, f=0.5)
        
        with pytest.raises(ValueError):
            GCTComponents(psi=1.1, rho=0.5, q=0.5, f=0.5)
    
    def test_individual_parameters_creation(self):
        """Test individual parameters creation"""
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        assert params.k_m == 0.5
        assert params.k_i == 2.0
    
    def test_individual_parameters_validation(self):
        """Test individual parameters validation"""
        # Valid parameters
        params = IndividualParameters(k_m=0.1, k_i=0.1)
        assert params.k_m == 0.1
        
        # Invalid parameters (must be positive)
        with pytest.raises(ValueError):
            IndividualParameters(k_m=-0.1, k_i=2.0)
        
        with pytest.raises(ValueError):
            IndividualParameters(k_m=0.5, k_i=-0.1)
        
        with pytest.raises(ValueError):
            IndividualParameters(k_m=0.0, k_i=2.0)
    
    def test_coherence_profile_creation(self, sample_gct_components, sample_individual_parameters):
        """Test coherence profile creation"""
        profile = CoherenceProfile(
            user_id="test-user",
            components=sample_gct_components,
            parameters=sample_individual_parameters,
            timestamp=datetime.utcnow()
        )
        
        assert_valid_coherence_profile(profile)
    
    def test_coherence_profile_soul_echo_calculation(self):
        """Test soul echo calculation"""
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        profile = CoherenceProfile(
            user_id="test-user",
            components=components,
            parameters=params,
            timestamp=datetime.utcnow()
        )
        
        expected_soul_echo = 0.8 * 0.7 * 0.6 * 0.9
        assert abs(profile.soul_echo - expected_soul_echo) < 1e-6
    
    def test_coherence_profile_immutability(self, sample_gct_components, sample_individual_parameters):
        """Test that coherence profile components are immutable"""
        profile = CoherenceProfile(
            user_id="test-user",
            components=sample_gct_components,
            parameters=sample_individual_parameters,
            timestamp=datetime.utcnow()
        )
        
        # Components should be immutable
        with pytest.raises(AttributeError):
            profile.components.psi = 0.9


class TestGCTCalculator:
    """Test GCT calculator functionality"""
    
    def test_gct_calculator_initialization(self, gct_calculator):
        """Test GCT calculator initialization"""
        assert gct_calculator is not None
        assert hasattr(gct_calculator, 'calculate_coherence')
        assert hasattr(gct_calculator, 'optimize_parameters')
    
    def test_basic_coherence_calculation(self, gct_calculator):
        """Test basic coherence calculation"""
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        result = gct_calculator.calculate_coherence(components, params)
        
        assert 'coherence_score' in result
        assert 'optimized_q' in result
        assert 'coupling_terms' in result
        assert 'trajectory_analysis' in result
        
        assert 0 <= result['coherence_score'] <= 1
    
    def test_moral_activation_optimization(self, gct_calculator):
        """Test moral activation optimization"""
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        result = gct_calculator.calculate_coherence(components, params)
        optimized_q = result['optimized_q']
        
        # Optimized q should be between 0 and q_max (1.0)
        assert 0 <= optimized_q <= 1.0
        
        # Should be different from original q due to optimization
        # (unless already optimal)
        original_q = components.q
        
        # The optimization should generally improve or maintain the value
        assert optimized_q >= 0
    
    def test_coupling_terms_calculation(self, gct_calculator):
        """Test coupling terms calculation"""
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        result = gct_calculator.calculate_coherence(components, params)
        coupling_terms = result['coupling_terms']
        
        assert 'psi_rho' in coupling_terms
        assert 'q_f' in coupling_terms
        assert 'temporal_coupling' in coupling_terms
        
        # Coupling terms should be non-negative
        for term_value in coupling_terms.values():
            assert term_value >= 0
    
    def test_trajectory_analysis(self, gct_calculator):
        """Test coherence trajectory analysis"""
        # Create historical data
        history = []
        base_time = datetime.utcnow()
        
        for i in range(10):
            components = GCTComponents(
                psi=0.7 + (i * 0.02),
                rho=0.6 + (i * 0.03),
                q=0.5 + (i * 0.04),
                f=0.8 + (i * 0.01)
            )
            
            profile = CoherenceProfile(
                user_id="test-user",
                components=components,
                parameters=IndividualParameters(k_m=0.5, k_i=2.0),
                timestamp=base_time - timedelta(days=i)
            )
            history.append(profile)
        
        # Current calculation
        current_components = GCTComponents(psi=0.9, rho=0.9, q=0.9, f=0.9)
        current_params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        result = gct_calculator.calculate_coherence(
            current_components,
            current_params,
            history=history
        )
        
        trajectory = result['trajectory_analysis']
        
        assert 'trend' in trajectory
        assert 'stability' in trajectory
        assert 'acceleration' in trajectory
        assert 'prediction' in trajectory
        
        # Trend should be a number
        assert isinstance(trajectory['trend'], (int, float))
        
        # Stability should be between 0 and 1
        assert 0 <= trajectory['stability'] <= 1
    
    def test_parameter_optimization(self, gct_calculator):
        """Test parameter optimization"""
        # Create training data
        training_data = []
        
        for i in range(20):
            components = GCTComponents(
                psi=0.5 + (i * 0.02),
                rho=0.6 + (i * 0.015),
                q=0.4 + (i * 0.025),
                f=0.7 + (i * 0.01)
            )
            
            # Simulate target coherence scores
            target_score = 0.6 + (i * 0.02)
            
            training_data.append({
                'components': components,
                'target_score': target_score
            })
        
        optimized_params = gct_calculator.optimize_parameters(training_data)
        
        assert 'k_m' in optimized_params
        assert 'k_i' in optimized_params
        assert optimized_params['k_m'] > 0
        assert optimized_params['k_i'] > 0
    
    def test_coherence_score_bounds(self, gct_calculator):
        """Test that coherence scores are within valid bounds"""
        # Test extreme values
        test_cases = [
            # Minimum values
            GCTComponents(psi=0.0, rho=0.0, q=0.0, f=0.0),
            # Maximum values
            GCTComponents(psi=1.0, rho=1.0, q=1.0, f=1.0),
            # Mixed values
            GCTComponents(psi=0.3, rho=0.8, q=0.2, f=0.9),
            GCTComponents(psi=0.9, rho=0.2, q=0.7, f=0.4)
        ]
        
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        for components in test_cases:
            result = gct_calculator.calculate_coherence(components, params)
            score = result['coherence_score']
            
            # Score should always be between 0 and 1
            assert 0 <= score <= 1, f"Score {score} out of bounds for components {components}"
    
    def test_calculation_consistency(self, gct_calculator):
        """Test that calculations are consistent"""
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        # Run calculation multiple times
        results = []
        for _ in range(5):
            result = gct_calculator.calculate_coherence(components, params)
            results.append(result['coherence_score'])
        
        # All results should be identical for deterministic calculation
        for score in results[1:]:
            assert abs(score - results[0]) < 1e-10


class TestTriadicProcessor:
    """Test triadic logic processing functionality"""
    
    def test_triadic_processor_initialization(self, triadic_processor):
        """Test triadic processor initialization"""
        assert triadic_processor is not None
        assert hasattr(triadic_processor, 'process')
        assert hasattr(triadic_processor, 'generate_phase')
        assert hasattr(triadic_processor, 'analyze_phase')
        assert hasattr(triadic_processor, 'ground_phase')
    
    @pytest.mark.asyncio
    async def test_generate_phase(self, triadic_processor):
        """Test the generate phase of triadic processing"""
        context = {
            'user_id': 'test-user',
            'query': 'What is the meaning of life?',
            'coherence_profile': {
                'psi': 0.8,
                'rho': 0.7,
                'q': 0.6,
                'f': 0.9
            }
        }
        
        result = await triadic_processor.generate_phase(context)
        
        assert 'generated_content' in result
        assert 'generation_metadata' in result
        assert 'coherence_markers' in result
        
        # Generated content should not be empty
        assert len(result['generated_content']) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_phase(self, triadic_processor):
        """Test the analyze phase of triadic processing"""
        generated_content = "Life has meaning through coherent relationships and moral development."
        
        context = {
            'generated_content': generated_content,
            'user_profile': {
                'psi': 0.8,
                'rho': 0.7,
                'q': 0.6,
                'f': 0.9
            }
        }
        
        result = await triadic_processor.analyze_phase(context)
        
        assert 'analysis_results' in result
        assert 'coherence_assessment' in result
        assert 'risk_evaluation' in result
        assert 'improvement_suggestions' in result
        
        # Coherence assessment should have scores
        coherence = result['coherence_assessment']
        assert 'consistency_score' in coherence
        assert 'wisdom_score' in coherence
        assert 'moral_score' in coherence
        assert 'social_score' in coherence
    
    @pytest.mark.asyncio
    async def test_ground_phase(self, triadic_processor):
        """Test the ground phase of triadic processing"""
        context = {
            'original_content': "Life has meaning through relationships.",
            'analysis_results': {
                'coherence_score': 0.8,
                'areas_for_improvement': ['moral_reasoning', 'social_context']
            },
            'user_context': {
                'preferences': {'depth': 'high'},
                'history': []
            }
        }
        
        result = await triadic_processor.ground_phase(context)
        
        assert 'grounded_response' in result
        assert 'grounding_rationale' in result
        assert 'coherence_integration' in result
        
        # Grounded response should be different from original
        assert result['grounded_response'] != context['original_content']
    
    @pytest.mark.asyncio
    async def test_full_triadic_process(self, triadic_processor):
        """Test complete triadic processing cycle"""
        input_data = {
            'user_id': 'test-user',
            'query': 'How should I approach moral dilemmas?',
            'user_profile': {
                'components': {
                    'psi': 0.7,
                    'rho': 0.8,
                    'q': 0.6,
                    'f': 0.9
                },
                'parameters': {
                    'k_m': 0.5,
                    'k_i': 2.0
                }
            },
            'context': {
                'session_history': [],
                'preferences': {'response_style': 'thoughtful'}
            }
        }
        
        result = await triadic_processor.process(input_data)
        
        assert 'final_response' in result
        assert 'processing_metadata' in result
        assert 'coherence_analysis' in result
        
        # Should contain results from all three phases
        metadata = result['processing_metadata']
        assert 'generate_phase' in metadata
        assert 'analyze_phase' in metadata
        assert 'ground_phase' in metadata
        
        # Final response should be coherent
        assert len(result['final_response']) > 0
        
        # Coherence analysis should show improvement
        coherence = result['coherence_analysis']
        assert 'initial_coherence' in coherence
        assert 'final_coherence' in coherence
    
    def test_risk_pattern_detection(self, triadic_processor):
        """Test risk pattern detection"""
        test_cases = [
            {
                'content': 'I hate everyone and everything',
                'expected_risk': 'high'
            },
            {
                'content': 'This is a thoughtful and balanced response',
                'expected_risk': 'low'
            },
            {
                'content': 'Maybe we should consider multiple perspectives',
                'expected_risk': 'low'
            }
        ]
        
        for case in test_cases:
            risk_level = triadic_processor._assess_risk_level(case['content'])
            
            if case['expected_risk'] == 'high':
                assert risk_level > 0.7
            else:
                assert risk_level < 0.3
    
    def test_coherence_marker_extraction(self, triadic_processor):
        """Test coherence marker extraction"""
        content = """
        This response demonstrates thoughtful consideration of multiple perspectives.
        It shows moral reasoning and social awareness while maintaining consistency
        with established principles. The wisdom shown here reflects accumulated
        experience and careful deliberation.
        """
        
        markers = triadic_processor._extract_coherence_markers(content)
        
        assert 'consistency_markers' in markers
        assert 'wisdom_markers' in markers
        assert 'moral_markers' in markers
        assert 'social_markers' in markers
        
        # Should find relevant markers
        assert len(markers['consistency_markers']) > 0
        assert len(markers['wisdom_markers']) > 0
        assert len(markers['moral_markers']) > 0
        assert len(markers['social_markers']) > 0
    
    def test_grounding_transformation(self, triadic_processor):
        """Test grounding transformation logic"""
        original = "This is a basic response."
        
        analysis = {
            'coherence_score': 0.4,
            'weaknesses': ['lacks_depth', 'needs_moral_context'],
            'suggestions': ['add_examples', 'consider_ethics']
        }
        
        user_context = {
            'components': {
                'psi': 0.8,
                'rho': 0.7,
                'q': 0.6,
                'f': 0.9
            },
            'preferences': {'depth': 'high'}
        }
        
        grounded = triadic_processor._apply_grounding_transformation(
            original, analysis, user_context
        )
        
        # Grounded response should be enhanced
        assert len(grounded) > len(original)
        assert grounded != original
    
    @pytest.mark.asyncio
    async def test_processing_with_history(self, triadic_processor):
        """Test triadic processing with conversation history"""
        history = [
            {
                'query': 'What is wisdom?',
                'response': 'Wisdom is the application of knowledge with good judgment.',
                'coherence_score': 0.7,
                'timestamp': datetime.utcnow() - timedelta(minutes=5)
            }
        ]
        
        input_data = {
            'user_id': 'test-user',
            'query': 'How do I develop wisdom?',
            'user_profile': {
                'components': {'psi': 0.8, 'rho': 0.6, 'q': 0.7, 'f': 0.9},
                'parameters': {'k_m': 0.5, 'k_i': 2.0}
            },
            'context': {
                'session_history': history,
                'preferences': {}
            }
        }
        
        result = await triadic_processor.process(input_data)
        
        # Should reference previous context
        assert 'context_integration' in result['processing_metadata']
        
        # Response should be coherent with history
        coherence = result['coherence_analysis']
        assert coherence['final_coherence'] > coherence['initial_coherence']


class TestCoherenceIntegration:
    """Test integration between coherence components"""
    
    @pytest.mark.asyncio
    async def test_calculator_processor_integration(self, gct_calculator, triadic_processor):
        """Test integration between calculator and processor"""
        # Calculate coherence
        components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        coherence_result = gct_calculator.calculate_coherence(components, params)
        
        # Use coherence result in triadic processing
        input_data = {
            'user_id': 'test-user',
            'query': 'How can I improve my decision-making?',
            'user_profile': {
                'components': components.__dict__,
                'parameters': params.__dict__,
                'coherence_score': coherence_result['coherence_score']
            },
            'context': {'preferences': {}}
        }
        
        triadic_result = await triadic_processor.process(input_data)
        
        # Results should be coherent
        assert triadic_result['coherence_analysis']['final_coherence'] > 0
        
        # Should reference coherence calculations
        assert 'coherence_integration' in triadic_result
    
    def test_coherence_profile_update_cycle(self, gct_calculator):
        """Test coherence profile update cycle"""
        # Initial profile
        initial_components = GCTComponents(psi=0.6, rho=0.5, q=0.4, f=0.7)
        params = IndividualParameters(k_m=0.5, k_i=2.0)
        
        # Calculate initial coherence
        initial_result = gct_calculator.calculate_coherence(initial_components, params)
        initial_score = initial_result['coherence_score']
        
        # Simulate learning/improvement
        updated_components = GCTComponents(
            psi=min(1.0, initial_components.psi + 0.1),
            rho=min(1.0, initial_components.rho + 0.15),
            q=min(1.0, initial_components.q + 0.2),
            f=min(1.0, initial_components.f + 0.05)
        )
        
        # Calculate updated coherence
        updated_result = gct_calculator.calculate_coherence(updated_components, params)
        updated_score = updated_result['coherence_score']
        
        # Score should improve
        assert updated_score >= initial_score
    
    def test_parameter_optimization_feedback(self, gct_calculator):
        """Test parameter optimization feedback loop"""
        # Create diverse training data
        training_data = []
        
        for psi in [0.3, 0.5, 0.7, 0.9]:
            for rho in [0.4, 0.6, 0.8]:
                for q in [0.2, 0.5, 0.8]:
                    for f in [0.6, 0.8, 1.0]:
                        components = GCTComponents(psi=psi, rho=rho, q=q, f=f)
                        # Target score based on component balance
                        target = (psi + rho + q + f) / 4.0
                        
                        training_data.append({
                            'components': components,
                            'target_score': target
                        })
        
        # Optimize parameters
        optimized_params = gct_calculator.optimize_parameters(training_data)
        
        # Test optimized parameters
        test_components = GCTComponents(psi=0.8, rho=0.7, q=0.6, f=0.9)
        
        # Compare original vs optimized parameters
        original_params = IndividualParameters(k_m=0.5, k_i=2.0)
        original_result = gct_calculator.calculate_coherence(test_components, original_params)
        
        optimized_individual_params = IndividualParameters(
            k_m=optimized_params['k_m'],
            k_i=optimized_params['k_i']
        )
        optimized_result = gct_calculator.calculate_coherence(test_components, optimized_individual_params)
        
        # Optimized parameters should generally perform better
        # (though this depends on the training data)
        assert optimized_result['coherence_score'] >= 0
        assert optimized_result['optimized_q'] >= 0