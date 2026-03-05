"""
LLM Integration Module

This module provides GCT-enhanced LLM interfaces for the IPAI system,
including coherence-aware prompting and response processing.
"""

from .interface import GCTLLMInterface, LLMResponse
from .gct_prompts import GCTPromptGenerator, PromptTemplate
from .triadic_handler import TriadicResponseHandler
from .coherence_analyzer import MessageCoherenceAnalyzer

__all__ = [
    'GCTLLMInterface',
    'LLMResponse',
    'GCTPromptGenerator',
    'PromptTemplate',
    'TriadicResponseHandler',
    'MessageCoherenceAnalyzer'
]