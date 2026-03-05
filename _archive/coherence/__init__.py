"""
GCT Coherence Engine

This module contains the core coherence calculation and processing components
for the IPAI system, implementing Grounded Coherence Theory.
"""

from .gct_calculator import GCTCalculator, EnhancedGCTCalculator
from .triadic_processor import TriadicProcessor, TriadicNeuralProcessor
from .soulmath import SoulMathEngine
from .network_dynamics import NetworkDynamicsEngine

__all__ = [
    'GCTCalculator',
    'EnhancedGCTCalculator',
    'TriadicProcessor',
    'TriadicNeuralProcessor',
    'SoulMathEngine',
    'NetworkDynamicsEngine'
]