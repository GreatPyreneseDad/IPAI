"""
GCT-Enhanced LLM Interface

This module provides the main interface for GCT-aware LLM interactions,
integrating coherence analysis, triadic processing, and response refinement.
"""

import asyncio
import json
import time
import aiohttp
from typing import Dict, Optional, AsyncGenerator, List, Any, Union
from dataclasses import dataclass
from datetime import datetime
import logging

from ..models.coherence_profile import CoherenceProfile, CoherenceLevel
from ..coherence.gct_calculator import GCTCalculator
from .gct_prompts import GCTPromptGenerator, ContextualPromptEnhancer
from .coherence_analyzer import MessageCoherenceAnalyzer, CoherenceAnalysis
from .triadic_handler import TriadicResponseHandler, LLMResponse


@dataclass
class LLMConfig:
    """Configuration for LLM interface"""
    # Ollama configuration
    model_name: str = "llama3.2:latest"
    ollama_host: str = "http://localhost:11434"
    context_length: int = 4096
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    enable_streaming: bool = True
    enable_coherence_checking: bool = True
    enable_triadic_processing: bool = True
    response_timeout: float = 30.0
    
    # Safety settings
    enable_safety_filtering: bool = True
    crisis_intervention_enabled: bool = True
    max_response_length: int = 2000
    
    # Backward compatibility
    model_path: str = ""  # Deprecated, kept for compatibility
    n_threads: int = 8  # Not used with Ollama
    n_gpu_layers: int = 35  # Not used with Ollama


class GCTLLMInterface:
    """LLM interface with GCT integration and triadic processing"""
    
    def __init__(self, 
                 config: LLMConfig,
                 gct_calculator: Optional[GCTCalculator] = None):
        self.config = config
        self.gct_calculator = gct_calculator or GCTCalculator()
        
        # Initialize components
        self.prompt_generator = GCTPromptGenerator()
        self.prompt_enhancer = ContextualPromptEnhancer()
        self.coherence_analyzer = MessageCoherenceAnalyzer()
        self.triadic_handler = TriadicResponseHandler()
        
        # Initialize LLM
        self.llm = None
        self._initialize_llm()
        
        # Conversation state
        self.conversation_history = []
        self.user_context = {}
        
        # Performance tracking
        self.response_times = []
        self.error_count = 0
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _initialize_llm(self):
        """Initialize the Ollama LLM connection"""
        try:
            # Initialize Ollama client
            self.llm = OllamaClient(
                host=self.config.ollama_host,
                model=self.config.model_name,
                timeout=self.config.response_timeout
            )
            
            # Test connection
            asyncio.create_task(self._test_ollama_connection())
            
            self.logger.info(f"Ollama LLM initialized successfully: {self.config.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama LLM: {e}")
            self.logger.warning("Falling back to mock LLM")
            self.llm = MockLLM()
    
    async def _test_ollama_connection(self):
        """Test Ollama connection and model availability"""
        try:
            test_response = await self.llm.generate("Hello", max_tokens=5)
            self.logger.info("Ollama connection test successful")
        except Exception as e:
            self.logger.error(f"Ollama connection test failed: {e}")
            self.llm = MockLLM()
    
    async def generate_response(
        self,
        message: str,
        profile: CoherenceProfile,
        context: Optional[Dict] = None,
        stream: bool = None
    ) -> Union[LLMResponse, AsyncGenerator[str, None]]:
        """Generate GCT-aware response with optional streaming"""
        
        start_time = time.time()
        stream = stream if stream is not None else self.config.enable_streaming
        context = context or {}
        
        try:
            # Phase 1: Analyze incoming message
            if self.config.enable_coherence_checking:
                message_analysis = await self._analyze_incoming_message(message, context)
                context['message_analysis'] = message_analysis
            
            # Phase 2: Generate system prompt
            system_prompt = self._generate_system_prompt(profile, context)
            
            # Phase 3: Generate response
            if stream:
                return self._generate_streaming_response(
                    message, system_prompt, profile, context, start_time
                )
            else:
                return await self._generate_complete_response(
                    message, system_prompt, profile, context, start_time
                )
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Error generating response: {e}")
            return await self._generate_error_response(e, profile)
    
    async def _analyze_incoming_message(
        self, 
        message: str, 
        context: Dict
    ) -> CoherenceAnalysis:
        """Analyze incoming message for coherence"""
        
        # Run coherence analysis
        analysis = await asyncio.to_thread(
            self.coherence_analyzer.analyze_message, 
            message, 
            context
        )
        
        # Log concerning patterns
        if analysis.red_flags:
            self.logger.warning(f"Red flags detected: {analysis.red_flags}")
        
        # Handle crisis situations
        if any("CRISIS" in flag for flag in analysis.red_flags):
            self.logger.critical(f"Crisis situation detected in message: {message[:100]}...")
            if self.config.crisis_intervention_enabled:
                context['crisis_mode'] = True
        
        return analysis
    
    def _generate_system_prompt(
        self, 
        profile: CoherenceProfile, 
        context: Dict
    ) -> str:
        """Generate comprehensive system prompt"""
        
        # Base system prompt
        base_prompt = self.prompt_generator.generate_system_prompt(profile)
        
        # Enhance with context
        enhanced_prompt = self.prompt_enhancer.enhance_prompt(
            base_prompt, context, profile
        )
        
        # Add crisis support if needed
        if context.get('crisis_mode'):
            crisis_addition = """
🚨 CRISIS INTERVENTION MODE 🚨
The user may be in emotional crisis. Your response should:
- Prioritize safety and immediate stabilization
- Provide crisis resources and emergency contacts
- Use calm, supportive, non-judgmental language
- Avoid complex advice or overwhelming information
- Encourage professional help immediately
- Be especially careful about any advice given
"""
            enhanced_prompt += crisis_addition
        
        return enhanced_prompt
    
    async def _generate_streaming_response(
        self,
        message: str,
        system_prompt: str,
        profile: CoherenceProfile,
        context: Dict,
        start_time: float
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response with real-time processing"""
        
        # Prepare full prompt
        full_prompt = self._prepare_full_prompt(message, system_prompt, context)
        
        # Start generation
        accumulated_response = ""
        
        # Generate tokens
        async for token in self._stream_tokens(full_prompt):
            accumulated_response += token
            yield token
            
            # Check for early termination conditions
            if len(accumulated_response) > self.config.max_response_length:
                break
        
        # Post-process complete response
        await self._post_process_response(
            accumulated_response, profile, context, start_time
        )
    
    async def _generate_complete_response(
        self,
        message: str,
        system_prompt: str,
        profile: CoherenceProfile,
        context: Dict,
        start_time: float
    ) -> LLMResponse:
        """Generate complete response with full triadic processing"""
        
        # Prepare full prompt
        full_prompt = self._prepare_full_prompt(message, system_prompt, context)
        
        # Generate response
        raw_response = await self._generate_raw_response(full_prompt)
        
        # Apply triadic processing
        if self.config.enable_triadic_processing:
            processed_response = await self._apply_triadic_processing(
                raw_response, profile, context
            )
        else:
            processed_response = LLMResponse(
                original_text=raw_response,
                processed_text=raw_response,
                user_profile=profile,
                response_analysis=None,
                processing_metadata={'triadic_processing': False},
                timestamp=datetime.utcnow()
            )
        
        # Update performance metrics
        response_time = time.time() - start_time
        self.response_times.append(response_time)
        
        # Update conversation history
        self._update_conversation_history(message, processed_response, context)
        
        return processed_response
    
    def _prepare_full_prompt(
        self, 
        message: str, 
        system_prompt: str, 
        context: Dict
    ) -> str:
        """Prepare the complete prompt for the LLM"""
        
        # Add conversation history if available
        history_context = self._get_conversation_context()
        
        # Build full prompt
        full_prompt = f"{system_prompt}\n\n"
        
        if history_context:
            full_prompt += f"Previous conversation context:\n{history_context}\n\n"
        
        full_prompt += f"User: {message}\n\nAssistant:"
        
        return full_prompt
    
    async def _stream_tokens(self, prompt: str) -> AsyncGenerator[str, None]:
        """Stream tokens from Ollama LLM"""
        try:
            # Stream from Ollama
            async for token in self.llm.stream_generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            ):
                if token:
                    yield token
                    
        except Exception as e:
            self.logger.error(f"Error in token streaming: {e}")
            yield "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    async def _generate_raw_response(self, prompt: str) -> str:
        """Generate raw response from Ollama LLM"""
        try:
            # Generate from Ollama
            response = await self.llm.generate(
                prompt,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p
            )
            
            return response.strip()
                
        except Exception as e:
            self.logger.error(f"Error in response generation: {e}")
            return "I apologize, but I'm experiencing technical difficulties. Please try again."
    
    async def _apply_triadic_processing(
        self,
        raw_response: str,
        profile: CoherenceProfile,
        context: Dict
    ) -> LLMResponse:
        """Apply triadic processing to refine response"""
        
        return await asyncio.to_thread(
            self.triadic_handler.process_response,
            raw_response,
            profile,
            context
        )
    
    async def _post_process_response(
        self,
        response: str,
        profile: CoherenceProfile,
        context: Dict,
        start_time: float
    ):
        """Post-process streaming response"""
        
        # Apply triadic processing to complete response
        if self.config.enable_triadic_processing:
            processed = await self._apply_triadic_processing(response, profile, context)
            
            # Log any issues found
            if processed.response_analysis and processed.response_analysis.safety_flags:
                self.logger.warning(f"Safety flags in streaming response: {processed.response_analysis.safety_flags}")
        
        # Update metrics
        response_time = time.time() - start_time
        self.response_times.append(response_time)
    
    def _get_conversation_context(self, max_turns: int = 3) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_turns = self.conversation_history[-max_turns:]
        context_parts = []
        
        for turn in recent_turns:
            context_parts.append(f"User: {turn['user_message']}")
            context_parts.append(f"Assistant: {turn['assistant_response'][:100]}...")
        
        return "\n".join(context_parts)
    
    def _update_conversation_history(
        self, 
        user_message: str, 
        response: LLMResponse, 
        context: Dict
    ):
        """Update conversation history"""
        
        turn = {
            'timestamp': datetime.utcnow(),
            'user_message': user_message,
            'assistant_response': response.processed_text,
            'user_coherence': response.user_profile.coherence_score,
            'response_quality': response.response_analysis.response_quality if response.response_analysis else 0.0,
            'context': context
        }
        
        self.conversation_history.append(turn)
        
        # Keep only recent history
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    async def _generate_error_response(
        self, 
        error: Exception, 
        profile: CoherenceProfile
    ) -> LLMResponse:
        """Generate appropriate error response based on user coherence"""
        
        error_messages = {
            CoherenceLevel.CRITICAL: "I'm having technical difficulties right now. If this is urgent, please contact a crisis helpline or emergency services.",
            CoherenceLevel.LOW: "I'm experiencing some technical issues. Please try again in a moment.",
            CoherenceLevel.MEDIUM: "I apologize, but I'm having technical difficulties. Please try your request again.",
            CoherenceLevel.HIGH: "I'm experiencing a technical error. The system may be temporarily unavailable."
        }
        
        error_text = error_messages.get(profile.level, error_messages[CoherenceLevel.MEDIUM])
        
        return LLMResponse(
            original_text=error_text,
            processed_text=error_text,
            user_profile=profile,
            response_analysis=None,
            processing_metadata={'error': str(error), 'error_response': True},
            timestamp=datetime.utcnow()
        )
    
    # Utility methods
    
    def update_user_context(self, context_updates: Dict):
        """Update persistent user context"""
        self.user_context.update(context_updates)
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.response_times:
            return {}
        
        import numpy as np
        
        return {
            'total_responses': len(self.response_times),
            'average_response_time': np.mean(self.response_times),
            'median_response_time': np.median(self.response_times),
            'error_rate': self.error_count / max(1, len(self.response_times)),
            'triadic_processing_stats': self.triadic_handler.get_processing_statistics()
        }
    
    async def check_message_coherence(self, message: str) -> CoherenceAnalysis:
        """Standalone coherence checking for messages"""
        return await self._analyze_incoming_message(message, {})
    
    def generate_intervention_recommendations(
        self, 
        profile: CoherenceProfile
    ) -> str:
        """Generate intervention recommendations"""
        
        # Analyze profile for intervention needs
        analysis = {
            'weak_components': self._identify_weak_components(profile),
            'strong_components': self._identify_strong_components(profile),
            'risk_factors': self._identify_risk_factors(profile),
            'growth_potential': self._assess_growth_potential(profile)
        }
        
        return self.prompt_generator.generate_intervention_prompt(profile, analysis)
    
    def _identify_weak_components(self, profile: CoherenceProfile) -> List[str]:
        """Identify weak components in profile"""
        weak = []
        threshold = 0.4
        
        if profile.components.psi < threshold:
            weak.append("Internal Consistency (Ψ)")
        if profile.components.rho < threshold:
            weak.append("Accumulated Wisdom (ρ)")
        if profile.components.q < threshold:
            weak.append("Moral Activation (q)")
        if profile.components.f < threshold:
            weak.append("Social Belonging (f)")
        
        return weak
    
    def _identify_strong_components(self, profile: CoherenceProfile) -> List[str]:
        """Identify strong components in profile"""
        strong = []
        threshold = 0.7
        
        if profile.components.psi >= threshold:
            strong.append("Internal Consistency (Ψ)")
        if profile.components.rho >= threshold:
            strong.append("Accumulated Wisdom (ρ)")
        if profile.components.q >= threshold:
            strong.append("Moral Activation (q)")
        if profile.components.f >= threshold:
            strong.append("Social Belonging (f)")
        
        return strong
    
    def _identify_risk_factors(self, profile: CoherenceProfile) -> List[str]:
        """Identify risk factors from profile"""
        risks = []
        
        if profile.level == CoherenceLevel.CRITICAL:
            risks.append("Critical coherence level")
        
        if profile.derivatives:
            if profile.derivatives.get('dC_dt', 0) < -0.05:
                risks.append("Declining coherence trend")
            
            if profile.derivatives.get('volatility', 0) > 0.3:
                risks.append("High coherence volatility")
        
        # Component-specific risks
        if profile.components.psi < 0.2:
            risks.append("Severe inconsistency risk")
        if profile.components.f < 0.2:
            risks.append("Social isolation risk")
        if profile.components.q < 0.2:
            risks.append("Moral disengagement risk")
        
        return risks
    
    def _assess_growth_potential(self, profile: CoherenceProfile) -> str:
        """Assess growth potential"""
        if profile.level == CoherenceLevel.HIGH:
            return "optimization and leadership"
        elif profile.level == CoherenceLevel.MEDIUM:
            return "moderate to high"
        elif profile.level == CoherenceLevel.LOW:
            return "significant foundation-building needed"
        else:
            return "crisis stabilization required first"


class OllamaClient:
    """Ollama API client for Llama 3.2 integration"""
    
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:latest", timeout: float = 30.0):
        self.host = host.rstrip('/')
        self.model = model
        self.timeout = timeout
        self.session = None
        
    async def _get_session(self):
        """Get or create aiohttp session"""
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self.session
    
    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> str:
        """Generate response from Ollama"""
        session = await self._get_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": ["User:", "Human:", "\n\nUser:", "\n\nHuman:"]
            }
        }
        
        try:
            async with session.post(f"{self.host}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "").strip()
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
    
    async def stream_generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> AsyncGenerator[str, None]:
        """Stream response from Ollama"""
        session = await self._get_session()
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": ["User:", "Human:", "\n\nUser:", "\n\nHuman:"]
            }
        }
        
        try:
            async with session.post(f"{self.host}/api/generate", json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if 'response' in data:
                                    token = data['response']
                                    if token:
                                        yield token
                                if data.get('done', False):
                                    break
                            except json.JSONDecodeError:
                                continue
                else:
                    error_text = await response.text()
                    raise Exception(f"Ollama API error {response.status}: {error_text}")
        except aiohttp.ClientError as e:
            raise Exception(f"Ollama connection error: {e}")
    
    async def check_model(self) -> bool:
        """Check if model is available"""
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    return self.model in models
                return False
        except aiohttp.ClientError:
            return False
    
    async def pull_model(self) -> bool:
        """Pull model if not available"""
        session = await self._get_session()
        
        payload = {"name": self.model}
        
        try:
            async with session.post(f"{self.host}/api/pull", json=payload) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                if data.get('status') == 'success':
                                    return True
                            except json.JSONDecodeError:
                                continue
                return False
        except aiohttp.ClientError:
            return False
    
    async def close(self):
        """Close the session"""
        if self.session and not self.session.closed:
            await self.session.close()


class MockLLM:
    """Mock LLM for testing and fallback"""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        # Simple mock responses based on prompt content
        if "crisis" in prompt.lower() or "emergency" in prompt.lower():
            return "I understand you may be going through a difficult time. Please consider reaching out to a mental health professional or crisis helpline for immediate support."
        
        elif "coherence" in prompt.lower():
            return "I can help you work on building coherence through reflection, values alignment, and practical steps toward your goals."
        
        else:
            return "I understand you're looking for guidance. Could you tell me more about what you'd like to explore or work on together?"
    
    async def stream_generate(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        """Mock streaming generation"""
        response = await self.generate(prompt, **kwargs)
        words = response.split()
        for word in words:
            yield word + " "
            await asyncio.sleep(0.05)  # Simulate streaming delay
    
    async def check_model(self) -> bool:
        """Mock model check"""
        return True
    
    async def close(self):
        """Mock close"""
        pass