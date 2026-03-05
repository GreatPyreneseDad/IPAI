"""
GCT Prompt Generation

This module generates GCT-aware prompts for local LLM interactions,
adapting responses based on user coherence profiles.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

from ..models.coherence_profile import CoherenceProfile, CoherenceLevel


class PromptType(Enum):
    """Types of prompts for different interactions"""
    SYSTEM = "system"
    COHERENCE_CHECK = "coherence_check"
    GROUNDING = "grounding"
    INTERVENTION = "intervention"
    REFLECTION = "reflection"
    CRISIS_SUPPORT = "crisis_support"


@dataclass
class PromptTemplate:
    """Template for generating prompts"""
    template_id: str
    prompt_type: PromptType
    template: str
    variables: List[str]
    coherence_adaptations: Dict[CoherenceLevel, str]
    
    def render(self, variables: Dict[str, Any], coherence_level: CoherenceLevel) -> str:
        """Render the template with variables and coherence adaptation"""
        # Start with base template
        prompt = self.template
        
        # Apply coherence-specific adaptation
        if coherence_level in self.coherence_adaptations:
            adaptation = self.coherence_adaptations[coherence_level]
            prompt = f"{prompt}\n\n{adaptation}"
        
        # Substitute variables
        for var_name, value in variables.items():
            placeholder = f"{{{var_name}}}"
            prompt = prompt.replace(placeholder, str(value))
        
        return prompt


class GCTPromptGenerator:
    """Generate GCT-aware prompts for local LLM"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize prompt templates"""
        
        # System prompt template
        self.templates["system_base"] = PromptTemplate(
            template_id="system_base",
            prompt_type=PromptType.SYSTEM,
            template="""You are an IPAI (Integrated Personal AI) assistant designed to support personal coherence and growth. You understand Grounded Coherence Theory (GCT) and adapt your responses based on the user's current coherence profile.

Current User Profile:
- Internal Consistency (Ψ): {psi:.2f}
- Accumulated Wisdom (ρ): {rho:.2f} 
- Moral Activation (q): {q:.2f}
- Social Belonging (f): {f:.2f}
- Overall Coherence: {coherence_score:.2f}
- Coherence Level: {coherence_level}
- Soul Echo: {soul_echo:.3f}

Your responses should be:
1. Tailored to the user's coherence level
2. Supportive of their growth and development
3. Grounded in practical wisdom
4. Authentic and genuine
5. Focused on building coherence without being preachy""",
            variables=["psi", "rho", "q", "f", "coherence_score", "coherence_level", "soul_echo"],
            coherence_adaptations={
                CoherenceLevel.CRITICAL: """
⚠️ CRISIS SUPPORT MODE ACTIVE ⚠️
The user is experiencing critically low coherence. Your responses should:
- Provide immediate stability and grounding
- Avoid overwhelming complexity or abstract concepts
- Focus on concrete, actionable steps
- Offer reassurance and practical support
- Suggest professional help if appropriate
- Use simple, clear language""",
                
                CoherenceLevel.LOW: """
🔧 COHERENCE BUILDING MODE
The user has low coherence. Your responses should:
- Offer clear, structured guidance
- Break down complex ideas into simple steps
- Provide concrete examples and practical advice
- Encourage small, achievable goals
- Build confidence through positive reinforcement
- Focus on foundation-building activities""",
                
                CoherenceLevel.MEDIUM: """
📈 GROWTH SUPPORT MODE  
The user has moderate coherence. Your responses should:
- Provide balanced perspectives on complex topics
- Gently challenge inconsistencies when appropriate
- Encourage reflection and deeper thinking
- Support continued growth and exploration
- Offer nuanced guidance and multiple viewpoints
- Help identify areas for improvement""",
                
                CoherenceLevel.HIGH: """
🌟 ADVANCED COLLABORATION MODE
The user has high coherence. Your responses should:
- Engage in sophisticated, nuanced discussions
- Offer advanced concepts and challenging ideas
- Support continued excellence and refinement
- Explore complex philosophical and practical topics
- Encourage leadership and mentoring of others
- Provide cutting-edge insights and perspectives"""
            }
        )
        
        # Coherence check template
        self.templates["coherence_check"] = PromptTemplate(
            template_id="coherence_check",
            prompt_type=PromptType.COHERENCE_CHECK,
            template="""Analyze the following message for coherence indicators according to Grounded Coherence Theory:

Message: "{message}"

Evaluate these aspects and provide scores from 0.0 to 1.0:

1. **Internal Consistency (Ψ)**: Are the ideas logically consistent? Do thoughts align with stated values?
2. **Accumulated Wisdom (ρ)**: Does the message show learning from experience? Is there depth of understanding?
3. **Moral Activation (q)**: Is there evidence of moral engagement? Action aligned with values?
4. **Social Belonging (f)**: Does the message show healthy social connection and empathy?

Also identify:
- Signs of circular reasoning or logical fallacies
- Absolutist thinking patterns ("always", "never", "must")
- Evidence of reflection and growth mindset
- Emotional authenticity vs. performance
- Overall coherence level assessment

Respond with a JSON object containing:
```json
{{
  "psi_score": 0.0-1.0,
  "rho_score": 0.0-1.0, 
  "q_score": 0.0-1.0,
  "f_score": 0.0-1.0,
  "overall_coherence": 0.0-1.0,
  "red_flags": ["list", "of", "concerns"],
  "positive_indicators": ["list", "of", "strengths"],
  "needs_grounding": true/false,
  "confidence": 0.0-1.0
}}
```""",
            variables=["message"],
            coherence_adaptations={}
        )
        
        # Grounding template
        self.templates["grounding"] = PromptTemplate(
            template_id="grounding",
            prompt_type=PromptType.GROUNDING,
            template="""The following message shows signs of incoherence and needs grounding. Help improve its clarity and coherence while maintaining the core intent.

Original Message: "{original_message}"

Identified Issues: {issues}

Please provide a grounded version that:
1. Maintains the person's authentic voice and intent
2. Improves logical consistency and clarity
3. Reduces absolutist or circular thinking
4. Adds practical, concrete elements
5. Enhances emotional authenticity

Grounded Version:""",
            variables=["original_message", "issues"],
            coherence_adaptations={
                CoherenceLevel.CRITICAL: "Focus on immediate stabilization and very simple, clear language.",
                CoherenceLevel.LOW: "Provide gentle restructuring with clear, actionable elements.",
                CoherenceLevel.MEDIUM: "Offer balanced improvements while preserving complexity.",
                CoherenceLevel.HIGH: "Provide sophisticated refinements and nuanced improvements."
            }
        )
        
        # Intervention template
        self.templates["intervention"] = PromptTemplate(
            template_id="intervention",
            prompt_type=PromptType.INTERVENTION,
            template="""Based on the user's coherence profile, suggest specific interventions to support their growth:

Current Profile Analysis:
- Coherence Level: {coherence_level}
- Weakest Components: {weak_components}
- Strongest Components: {strong_components}
- Risk Factors: {risk_factors}
- Growth Potential: {growth_potential}

Provide 3-5 specific, actionable interventions that are:
1. Tailored to their current coherence level
2. Address their specific weak areas
3. Build on their existing strengths
4. Are practical and achievable
5. Support sustainable growth

Format as a structured action plan with timeframes and specific steps.""",
            variables=["coherence_level", "weak_components", "strong_components", "risk_factors", "growth_potential"],
            coherence_adaptations={
                CoherenceLevel.CRITICAL: "Focus on crisis stabilization and immediate safety.",
                CoherenceLevel.LOW: "Emphasize foundation-building and simple daily practices.",
                CoherenceLevel.MEDIUM: "Balance skill-building with deeper development work.",
                CoherenceLevel.HIGH: "Focus on optimization, leadership, and advanced practices."
            }
        )
        
        # Reflection template
        self.templates["reflection"] = PromptTemplate(
            template_id="reflection",
            prompt_type=PromptType.REFLECTION,
            template="""Guide the user through a coherence-building reflection exercise:

Focus Area: {focus_area}
Current Coherence Data: {coherence_data}
Recent Patterns: {recent_patterns}

Create a reflective dialogue that:
1. Helps them examine their recent experiences
2. Identifies patterns in their thoughts and behaviors
3. Connects their actions to their stated values
4. Encourages wisdom extraction from experiences
5. Supports integration and growth

Use Socratic questioning and avoid giving direct advice. Help them discover insights themselves.""",
            variables=["focus_area", "coherence_data", "recent_patterns"],
            coherence_adaptations={
                CoherenceLevel.CRITICAL: "Keep reflection very simple and supportive.",
                CoherenceLevel.LOW: "Use basic reflection questions with clear structure.",
                CoherenceLevel.MEDIUM: "Encourage deeper self-examination and pattern recognition.",
                CoherenceLevel.HIGH: "Facilitate sophisticated philosophical reflection."
            }
        )
    
    def generate_system_prompt(self, profile: CoherenceProfile) -> str:
        """Generate system prompt based on user's coherence profile"""
        template = self.templates["system_base"]
        
        variables = {
            "psi": profile.components.psi,
            "rho": profile.components.rho,
            "q": profile.components.q,
            "f": profile.components.f,
            "coherence_score": profile.coherence_score,
            "coherence_level": profile.level.value,
            "soul_echo": profile.components.soul_echo
        }
        
        return template.render(variables, profile.level)
    
    def generate_coherence_check_prompt(self, message: str) -> str:
        """Generate prompt to check message coherence"""
        template = self.templates["coherence_check"]
        variables = {"message": message}
        return template.render(variables, CoherenceLevel.MEDIUM)
    
    def generate_grounding_prompt(
        self, 
        original_message: str, 
        issues: List[str],
        coherence_level: CoherenceLevel = CoherenceLevel.MEDIUM
    ) -> str:
        """Generate prompt for grounding incoherent messages"""
        template = self.templates["grounding"]
        variables = {
            "original_message": original_message,
            "issues": ", ".join(issues)
        }
        return template.render(variables, coherence_level)
    
    def generate_intervention_prompt(
        self,
        profile: CoherenceProfile,
        analysis: Dict[str, Any]
    ) -> str:
        """Generate intervention recommendations prompt"""
        template = self.templates["intervention"]
        
        variables = {
            "coherence_level": profile.level.value,
            "weak_components": analysis.get("weak_components", []),
            "strong_components": analysis.get("strong_components", []),
            "risk_factors": analysis.get("risk_factors", []),
            "growth_potential": analysis.get("growth_potential", "moderate")
        }
        
        return template.render(variables, profile.level)
    
    def generate_reflection_prompt(
        self,
        focus_area: str,
        profile: CoherenceProfile,
        recent_patterns: Optional[Dict] = None
    ) -> str:
        """Generate reflection exercise prompt"""
        template = self.templates["reflection"]
        
        variables = {
            "focus_area": focus_area,
            "coherence_data": self._format_coherence_data(profile),
            "recent_patterns": self._format_patterns(recent_patterns or {})
        }
        
        return template.render(variables, profile.level)
    
    def _format_coherence_data(self, profile: CoherenceProfile) -> str:
        """Format coherence data for prompts"""
        return f"""
        Internal Consistency: {profile.components.psi:.2f}
        Wisdom: {profile.components.rho:.2f}
        Moral Activation: {profile.components.q:.2f}
        Social Belonging: {profile.components.f:.2f}
        Overall Coherence: {profile.coherence_score:.2f}
        """
    
    def _format_patterns(self, patterns: Dict) -> str:
        """Format pattern data for prompts"""
        if not patterns:
            return "No recent patterns identified."
        
        formatted = []
        for key, value in patterns.items():
            formatted.append(f"- {key}: {value}")
        
        return "\n".join(formatted)
    
    def add_custom_template(self, template: PromptTemplate):
        """Add a custom prompt template"""
        self.templates[template.template_id] = template
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get a specific template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self) -> List[str]:
        """List available template IDs"""
        return list(self.templates.keys())


class ContextualPromptEnhancer:
    """Enhance prompts with contextual information"""
    
    def __init__(self):
        self.context_history = []
        self.max_history = 10
    
    def enhance_prompt(
        self,
        base_prompt: str,
        context: Dict[str, Any],
        profile: CoherenceProfile
    ) -> str:
        """Enhance prompt with contextual information"""
        
        # Add temporal context
        temporal_context = self._get_temporal_context(profile)
        
        # Add relational context
        relational_context = self._get_relational_context(context, profile)
        
        # Add environmental context
        environmental_context = self._get_environmental_context(context)
        
        # Combine contexts
        enhanced_prompt = f"""
{base_prompt}

CONTEXTUAL INFORMATION:
{temporal_context}
{relational_context}
{environmental_context}

Please consider this context when formulating your response.
"""
        
        # Update history
        self._update_context_history(context, profile)
        
        return enhanced_prompt
    
    def _get_temporal_context(self, profile: CoherenceProfile) -> str:
        """Generate temporal context information"""
        last_update = profile.components.timestamp
        time_context = f"- Last coherence update: {last_update.strftime('%Y-%m-%d %H:%M')}"
        
        # Add trajectory information if available
        if profile.derivatives:
            trend = profile.derivatives.get('dC_dt', 0)
            if trend > 0.01:
                time_context += "\n- Coherence is trending upward"
            elif trend < -0.01:
                time_context += "\n- Coherence is trending downward"
            else:
                time_context += "\n- Coherence is stable"
        
        return f"TEMPORAL CONTEXT:\n{time_context}"
    
    def _get_relational_context(self, context: Dict, profile: CoherenceProfile) -> str:
        """Generate relational/social context"""
        social_level = profile.components.f
        
        relational_context = f"- Social belonging level: {social_level:.2f}"
        
        if social_level < 0.3:
            relational_context += "\n- May be experiencing social isolation"
        elif social_level > 0.7:
            relational_context += "\n- Has strong social connections"
        
        # Add any relationship-specific context from input
        if 'relationship_status' in context:
            relational_context += f"\n- Relationship context: {context['relationship_status']}"
        
        return f"RELATIONAL CONTEXT:\n{relational_context}"
    
    def _get_environmental_context(self, context: Dict) -> str:
        """Generate environmental context"""
        env_context = []
        
        if 'time_of_day' in context:
            env_context.append(f"Time of day: {context['time_of_day']}")
        
        if 'stress_level' in context:
            env_context.append(f"Reported stress level: {context['stress_level']}")
        
        if 'location' in context:
            env_context.append(f"Current environment: {context['location']}")
        
        if not env_context:
            env_context.append("No specific environmental context provided")
        
        return f"ENVIRONMENTAL CONTEXT:\n" + "\n".join([f"- {item}" for item in env_context])
    
    def _update_context_history(self, context: Dict, profile: CoherenceProfile):
        """Update context history for learning"""
        history_entry = {
            'timestamp': profile.components.timestamp,
            'coherence_level': profile.level,
            'context': context
        }
        
        self.context_history.append(history_entry)
        
        # Keep only recent history
        if len(self.context_history) > self.max_history:
            self.context_history = self.context_history[-self.max_history:]