"""
Rose Glass v2 Engine — IPAI Phase 1 Cornerstone
=================================================

Adapted from the canonical Rose Glass LLM Lens v2 mathematics
(WP-2026-001 revision) into the IPAI API context.

Pure Python stdlib. No numpy. No external dependencies.

Core mathematics:
- Michaelis-Menten biological optimization with substrate inhibition
- Four-dimensional coherence equation: C = Ψ + (ρ×Ψ) + q_opt + (f×Ψ) + coupling
- τ-attenuated resilience decay: λ_eff = λ₀ / (1 + κτ)
- Restoration integral: ∫₀ᵗ μ(s)e^(-λ_eff(t-s)) ds
- Generational cascade modeling
- λ decomposition into dimensional contributors
- Veritas authenticity detection

Author: Christopher MacGregor bin Joseph
ROSE Corp. | MacGregor Holding Company

"Coherence is constructed, not discovered." — Ibn Rushd, adapted
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import requests


# =============================================================================
# DIMENSIONAL ARCHITECTURE
# =============================================================================

class CulturalCalibration(Enum):
    """Cultural calibration presets for the lens."""
    WESTERN_ACADEMIC = "western_academic"
    SPIRITUAL_CONTEMPLATIVE = "spiritual_contemplative"
    INDIGENOUS_ORAL = "indigenous_oral"
    CRISIS_TRANSLATION = "crisis_translation"
    LEGAL_ADVERSARIAL = "legal_adversarial"
    CLINICAL_THERAPEUTIC = "clinical_therapeutic"
    NEURODIVERGENT = "neurodivergent"


@dataclass
class DimensionalCalibration:
    """Parameters that tune the lens for specific cultural contexts."""
    km: float = 0.20              # Michaelis-Menten saturation constant
    ki: float = 0.80              # Substrate inhibition constant
    coupling_strength: float = 0.15
    psi_weight: float = 1.0       # Dimensional weight in λ decomposition
    rho_weight: float = 1.0
    q_weight: float = 1.0
    f_weight: float = 1.0
    tau_sensitivity: float = 0.5  # Temporal depth detection sensitivity
    kappa: float = 0.5            # τ-attenuation coefficient for λ reduction
    mu_baseline: float = 0.1      # Baseline restoration pulse magnitude

    def to_dict(self) -> Dict[str, float]:
        return {k: round(v, 3) for k, v in self.__dict__.items()}


# Calibration presets derived from WP-2026-001
# NOTE ON κ VALUES: These are calibration hypotheses, not empirical constants.
CALIBRATION_PRESETS: Dict[CulturalCalibration, DimensionalCalibration] = {
    CulturalCalibration.WESTERN_ACADEMIC: DimensionalCalibration(
        km=0.20, ki=0.80, coupling_strength=0.15,
        kappa=0.3, mu_baseline=0.08
    ),
    CulturalCalibration.SPIRITUAL_CONTEMPLATIVE: DimensionalCalibration(
        km=0.30, ki=1.20, coupling_strength=0.10,
        rho_weight=1.3, q_weight=0.8, tau_sensitivity=0.9,
        kappa=0.9, mu_baseline=0.15
    ),
    CulturalCalibration.INDIGENOUS_ORAL: DimensionalCalibration(
        km=0.25, ki=1.00, coupling_strength=0.08,
        f_weight=1.4, rho_weight=1.2, tau_sensitivity=0.85,
        kappa=0.85, mu_baseline=0.18
    ),
    CulturalCalibration.CRISIS_TRANSLATION: DimensionalCalibration(
        km=0.20, ki=0.80, coupling_strength=0.15,
        q_weight=1.3, psi_weight=1.2, tau_sensitivity=0.3,
        kappa=0.2, mu_baseline=0.05
    ),
    CulturalCalibration.LEGAL_ADVERSARIAL: DimensionalCalibration(
        km=0.15, ki=0.60, coupling_strength=0.20,
        psi_weight=1.4, rho_weight=1.3, q_weight=0.7,
        kappa=0.15, mu_baseline=0.03
    ),
    CulturalCalibration.CLINICAL_THERAPEUTIC: DimensionalCalibration(
        km=0.25, ki=1.00, coupling_strength=0.12,
        q_weight=1.2, f_weight=1.1, tau_sensitivity=0.7,
        kappa=0.6, mu_baseline=0.12
    ),
    CulturalCalibration.NEURODIVERGENT: DimensionalCalibration(
        km=0.45, ki=3.50, coupling_strength=0.15,
        psi_weight=1.3,
        kappa=0.4, mu_baseline=0.07
    ),
}


# =============================================================================
# CORE MATHEMATICS
# =============================================================================

def biological_optimization(q_raw: float, km: float = 0.20, ki: float = 0.80) -> float:
    """
    Michaelis-Menten with substrate inhibition.
    q_opt = q / (Km + q + q²/Ki)
    Prevents extremism. Mirrors natural regulatory systems.
    """
    if q_raw <= 0:
        return 0.0
    return q_raw / (km + q_raw + (q_raw ** 2 / ki))


def calculate_coherence(
    psi: float, rho: float, q_raw: float, f: float,
    cal: Optional[DimensionalCalibration] = None
) -> Dict[str, float]:
    """
    C = Ψ + (ρ × Ψ) + q_opt + (f × Ψ) + coupling
    Returns the full decomposition.
    """
    if cal is None:
        cal = DimensionalCalibration()

    q_opt = biological_optimization(q_raw, cal.km, cal.ki)

    base = psi
    wisdom_amplification = rho * psi
    social_amplification = f * psi
    coupling = cal.coupling_strength * rho * q_opt

    coherence = base + wisdom_amplification + q_opt + social_amplification + coupling
    coherence = min(coherence, 4.0)

    return {
        "coherence": round(coherence, 4),
        "base_psi": round(base, 4),
        "wisdom_amplification": round(wisdom_amplification, 4),
        "q_optimized": round(q_opt, 4),
        "social_amplification": round(social_amplification, 4),
        "coupling": round(coupling, 4),
        "pattern_intensity": round(coherence / 4.0, 4),
    }


# =============================================================================
# RESILIENCE MODEL v2
# =============================================================================

def tau_attenuated_lambda(
    lambda_base: float, tau: float, kappa: float = 0.5
) -> float:
    """
    λ_eff = λ₀ / (1 + κτ)
    Temporal depth formally attenuates the decay constant.
    """
    if lambda_base <= 0:
        return 0.0
    return lambda_base / (1 + kappa * tau)


def restoration_integral(
    lambda_eff: float, t: float,
    translation_events: List[Tuple[float, float]],
) -> float:
    """
    ∫₀ᵗ μ(s) × e^(-λ_eff(t-s)) ds
    Each accurate translation event at time s contributes a restoration
    pulse μ(s) that decays exponentially from the moment of occurrence.
    """
    if not translation_events:
        return 0.0

    restoration = 0.0
    for s, mu in translation_events:
        if s <= t:
            restoration += mu * math.exp(-lambda_eff * (t - s))

    return restoration


def _instantaneous_dR_dt(
    lambda_eff: float, t: float,
    decay_component: float,
    translation_events: List[Tuple[float, float]],
) -> float:
    """Compute dR/dt at time t from the complete model."""
    rest = restoration_integral(lambda_eff, t, translation_events)
    r_t = decay_component + rest

    decay_rate = -lambda_eff * r_t

    restoration_rate = 0.0
    for s, mu in translation_events:
        if s <= t:
            restoration_rate += mu * math.exp(-lambda_eff * (t - s))

    return decay_rate + restoration_rate


def _classify_trajectory(
    lambda_eff: float, t: float, r_t: float,
    decay_component: float,
    translation_events: List[Tuple[float, float]],
) -> str:
    """Classify trajectory based on sign of dR/dt."""
    dr_dt = _instantaneous_dR_dt(lambda_eff, t, decay_component, translation_events)
    if dr_dt > 0.001:
        return "restoring"
    elif dr_dt < -0.001:
        return "decaying"
    else:
        return "equilibrium"


def resilience_complete(
    n0: float, lambda_base: float, t: float,
    tau: float = 0.0, kappa: float = 0.5,
    translation_events: Optional[List[Tuple[float, float]]] = None,
) -> Dict[str, Any]:
    """
    R(t) = N₀e^(-λ_eff × t) + ∫₀ᵗ μ(s)e^(-λ_eff(t-s)) ds
    where λ_eff = λ₀ / (1 + κτ)

    The complete resilience model from WP-2026-001 v2.
    """
    if translation_events is None:
        translation_events = []

    lambda_eff = tau_attenuated_lambda(lambda_base, tau, kappa)
    decay_component = n0 * math.exp(-lambda_eff * t)
    rest = restoration_integral(lambda_eff, t, translation_events)

    r_t = min(max(decay_component + rest, 0.0), 1.0)

    half_life = math.log(2) / lambda_eff if lambda_eff > 0 else float('inf')
    lambda_reduction = 1.0 - (lambda_eff / lambda_base) if lambda_base > 0 else 0.0

    return {
        "resilience": round(r_t, 4),
        "decay_component": round(decay_component, 4),
        "restoration_component": round(rest, 4),
        "baseline_n0": n0,
        "lambda_base": round(lambda_base, 4),
        "lambda_effective": round(lambda_eff, 4),
        "lambda_reduction_from_tau": f"{lambda_reduction:.1%}",
        "tau": tau,
        "kappa": kappa,
        "half_life_years": round(half_life, 2),
        "time": t,
        "percent_remaining": round((r_t / n0) * 100, 1) if n0 > 0 else 0,
        "translation_events_count": len(translation_events),
        "dR_dt": round(_instantaneous_dR_dt(
            lambda_eff, t, decay_component, translation_events
        ), 6),
        "net_trajectory": _classify_trajectory(
            lambda_eff, t, r_t, decay_component, translation_events
        ),
    }


def generational_cascade(
    n0_g1: float, lambda_g1: float,
    transmission_time: float, lambda_escalation: float = 1.5,
    generations: int = 4,
    tau_per_generation: Optional[List[float]] = None,
    kappa: float = 0.5,
    restoration_per_generation: Optional[List[List[Tuple[float, float]]]] = None,
) -> List[Dict[str, Any]]:
    """
    Models intergenerational compounding from Section 4.
    Includes τ-attenuation and restoration capacity per generation.
    """
    cascade = []
    n0 = n0_g1
    lam = lambda_g1

    if tau_per_generation is None:
        tau_per_generation = [max(0.8 - (0.2 * g), 0.05) for g in range(generations)]

    if restoration_per_generation is None:
        restoration_per_generation = [[] for _ in range(generations)]

    for g in range(generations):
        tau = tau_per_generation[g] if g < len(tau_per_generation) else 0.05
        events = restoration_per_generation[g] if g < len(restoration_per_generation) else []

        result = resilience_complete(
            n0=n0, lambda_base=lam, t=transmission_time,
            tau=tau, kappa=kappa,
            translation_events=events
        )

        cascade.append({
            "generation": g + 1,
            "baseline_n0": round(n0, 4),
            "lambda_base": round(lam, 4),
            "lambda_effective": result["lambda_effective"],
            "tau": tau,
            "half_life_years": result["half_life_years"],
            "resilience_at_transmission": result["resilience"],
            "decay_component": result["decay_component"],
            "restoration_component": result["restoration_component"],
            "net_trajectory": result["net_trajectory"],
        })

        n0 = result["resilience"]
        lam = lam * lambda_escalation

    return cascade


def lambda_decomposition(
    lambda_psi: float, lambda_rho: float,
    lambda_q: float, lambda_f: float,
    cal: Optional[DimensionalCalibration] = None
) -> Dict[str, float]:
    """
    λ = w₁λΨ + w₂λρ + w₃λq + w₄λf
    Decomposes the decay constant into dimensional contributors.
    """
    if cal is None:
        cal = DimensionalCalibration()

    total = (
        cal.psi_weight * lambda_psi +
        cal.rho_weight * lambda_rho +
        cal.q_weight * lambda_q +
        cal.f_weight * lambda_f
    )

    contributions = [
        ("Ψ", cal.psi_weight * lambda_psi),
        ("ρ", cal.rho_weight * lambda_rho),
        ("q", cal.q_weight * lambda_q),
        ("f", cal.f_weight * lambda_f),
    ]

    return {
        "lambda_total": round(total, 4),
        "psi_contribution": round(contributions[0][1], 4),
        "rho_contribution": round(contributions[1][1], 4),
        "q_contribution": round(contributions[2][1], 4),
        "f_contribution": round(contributions[3][1], 4),
        "dominant_attack_vector": max(contributions, key=lambda x: x[1])[0],
    }


def veritas_check(
    psi: float, rho: float, q: float, f: float,
    cal: Optional[DimensionalCalibration] = None
) -> Dict[str, Any]:
    """
    Authenticity detection via dimensional coherence patterns.

    Checks for signatures of performative vs. genuine expression
    by analyzing dimensional balance and biological plausibility.

    High q with low psi suggests performed emotion (activation without consistency).
    High psi with zero rho suggests theoretical without experiential grounding.
    Perfectly balanced dimensions are statistically implausible in genuine expression.
    """
    if cal is None:
        cal = DimensionalCalibration()

    flags = []
    dims = [psi, rho, q, f]

    # Check for performed emotion: high activation without internal consistency
    if q > 0.7 and psi < 0.2:
        flags.append("high_activation_low_consistency")

    # Check for theoretical without experiential grounding
    if psi > 0.8 and rho < 0.1:
        flags.append("consistency_without_experience")

    # Check for suspiciously balanced dimensions (genuine expression is messy)
    if len(set(round(d, 2) for d in dims)) == 1 and all(d > 0.3 for d in dims):
        flags.append("suspiciously_uniform")

    # Check for social performance: high f with minimal other dimensions
    if f > 0.8 and psi < 0.2 and rho < 0.2:
        flags.append("social_performance_without_substance")

    # Dimensional variance — genuine expression has texture
    mean_d = sum(dims) / 4
    variance = sum((d - mean_d) ** 2 for d in dims) / 4
    dimensional_texture = math.sqrt(variance)

    # Biological plausibility via q optimization
    q_opt = biological_optimization(q, cal.km, cal.ki)
    q_ratio = q_opt / q if q > 0 else 1.0

    # Very low ratio means extreme q — biologically implausible as sustained state
    if q > 0.95 and q_ratio < 0.3:
        flags.append("biologically_implausible_sustained_activation")

    authenticity_score = 1.0
    authenticity_score -= 0.15 * len(flags)
    authenticity_score = max(authenticity_score, 0.0)

    return {
        "authenticity_score": round(authenticity_score, 4),
        "flags": flags,
        "dimensional_texture": round(dimensional_texture, 4),
        "q_biological_ratio": round(q_ratio, 4),
        "assessment": "genuine" if not flags else "review_recommended",
    }


# =============================================================================
# ENGINE CLASS
# =============================================================================

@dataclass
class RoseGlassScore:
    """Complete result from a Rose Glass analysis."""
    psi: float
    rho: float
    q_raw: float
    q: float           # biologically optimized
    f: float
    tau: float
    lambda_: float
    coherence: float
    calibration: str
    decomposition: Dict[str, float]
    resilience: Optional[Dict[str, Any]] = None
    veritas: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "psi": self.psi,
            "rho": self.rho,
            "q_raw": self.q_raw,
            "q_optimized": self.q,
            "f": self.f,
            "tau": self.tau,
            "lambda": self.lambda_,
            "coherence": self.coherence,
            "calibration": self.calibration,
            "decomposition": self.decomposition,
        }
        if self.resilience:
            result["resilience"] = self.resilience
        if self.veritas:
            result["veritas"] = self.veritas
        return result


class RoseGlassEngine:
    """
    Phase 1 engine wrapping Rose Glass v2 mathematics.

    Provides dimensional analysis, text heuristic estimation,
    and calibration management.
    """

    def __init__(self):
        self._presets = CALIBRATION_PRESETS

    def get_calibration(self, name: str) -> DimensionalCalibration:
        """Get a calibration preset by string name."""
        try:
            key = CulturalCalibration(name)
        except ValueError:
            valid = [c.value for c in CulturalCalibration]
            raise ValueError(
                f"Unknown calibration '{name}'. Valid: {valid}"
            )
        return self._presets[key]

    def list_calibrations(self) -> List[Dict[str, Any]]:
        """Return all available calibration names and their parameters."""
        return [
            {"name": cal.value, "parameters": preset.to_dict()}
            for cal, preset in self._presets.items()
        ]

    def analyze_dimensions(
        self,
        psi: float,
        rho: float,
        q: float,
        f: float,
        tau: float = 0.0,
        lambda_: float = 0.3,
        calibration: str = "western_academic",
    ) -> RoseGlassScore:
        """
        Full dimensional analysis from known dimension values.

        Args:
            psi: Internal consistency (0-1)
            rho: Accumulated wisdom (0-1)
            q: Moral/emotional activation energy (0-1)
            f: Social belonging architecture (0-1)
            tau: Temporal depth anchoring (0-1)
            lambda_: Decay constant from misperception environment
            calibration: Cultural calibration preset name
        """
        cal = self.get_calibration(calibration)
        q_opt = biological_optimization(q, cal.km, cal.ki)
        decomp = calculate_coherence(psi, rho, q, f, cal)

        # Resilience with tau attenuation
        resil = resilience_complete(
            n0=decomp["coherence"] / 4.0,  # normalize to 0-1 range
            lambda_base=lambda_,
            t=1.0,
            tau=tau,
            kappa=cal.kappa,
        )

        ver = veritas_check(psi, rho, q, f, cal)

        return RoseGlassScore(
            psi=round(psi, 4),
            rho=round(rho, 4),
            q_raw=round(q, 4),
            q=round(q_opt, 4),
            f=round(f, 4),
            tau=round(tau, 4),
            lambda_=round(lambda_, 4),
            coherence=decomp["coherence"],
            calibration=calibration,
            decomposition=decomp,
            resilience=resil,
            veritas=ver,
        )

    # Ollama configuration
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama3.2:latest"

    def _llm_raw_dimensions(self, text: str) -> Optional[Dict[str, float]]:
        """
        Call Ollama to estimate Rose Glass dimensions from text.
        Returns raw LLM dict with keys psi, rho, q, f, tau, lambda_ or None on failure.
        """
        prompt = (
            "Analyze the following text and estimate these psychological/philosophical "
            "dimensions as float values between 0.0 and 1.0. Return ONLY valid JSON "
            "with exactly these keys, no explanation:\n"
            "- psi: internal consistency (how logically coherent and self-consistent the text is)\n"
            "- rho: accumulated wisdom (depth of experiential knowledge and insight)\n"
            "- q: emotional activation (intensity of moral/emotional energy)\n"
            "- f: social belonging (strength of relational and communal language)\n"
            "- tau: temporal depth (references to history, tradition, generational time)\n"
            "- lambda: stress/decay signals (indicators of misperception, distortion, or erosion)\n\n"
            f"Text:\n{text}\n\n"
            "JSON:"
        )
        try:
            resp = requests.post(
                f"{self.OLLAMA_URL}/api/generate",
                json={"model": self.OLLAMA_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            # Extract JSON from response (handle markdown fences)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            data = json.loads(raw)
            dims = {}
            for key, out_key in [
                ("psi", "psi"), ("rho", "rho"), ("q", "q"),
                ("f", "f"), ("tau", "tau"), ("lambda", "lambda_"),
            ]:
                val = float(data[key])
                dims[out_key] = max(0.0, min(val, 1.0))
            return dims
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError):
            return None

    # Hybrid strategy: which source to use per dimension
    # LLM outperforms heuristic on these:
    _LLM_DIMS = {"q", "f", "tau"}
    # Heuristic more reliable on these:
    _HEURISTIC_DIMS = {"rho", "lambda_"}
    # Average both for these:
    _HYBRID_DIMS = {"psi"}

    def _llm_estimate_dimensions(self, text: str) -> Optional[Dict[str, float]]:
        """
        Hybrid LLM + heuristic dimension estimation.

        Uses LLM values for q, f, tau (LLM outperforms heuristic).
        Uses heuristic values for rho, lambda_ (heuristic more reliable).
        Uses average of both for psi (both reasonable).

        Returns blended dict or None if LLM is unavailable (full heuristic fallback).
        """
        llm_dims = self._llm_raw_dimensions(text)
        if llm_dims is None:
            return None

        heuristic_dims = self._heuristic_estimate_dimensions(text)

        blended = {}
        for dim in ("psi", "rho", "q", "f", "tau", "lambda_"):
            if dim in self._LLM_DIMS:
                blended[dim] = llm_dims[dim]
            elif dim in self._HEURISTIC_DIMS:
                blended[dim] = heuristic_dims[dim]
            else:  # hybrid — average
                blended[dim] = (llm_dims[dim] + heuristic_dims[dim]) / 2.0

        return blended

    def _heuristic_estimate_dimensions(self, text: str) -> Dict[str, float]:
        """Keyword-based heuristic fallback for dimension estimation."""
        words = text.split()
        word_count = len(words)
        unique_words = len(set(w.lower() for w in words))

        # Ψ (internal consistency): lexical diversity as rough proxy
        if word_count > 0:
            diversity = unique_words / word_count
            psi = min(diversity * 0.8, 0.95)
        else:
            psi = 0.1

        # ρ (accumulated wisdom): sentence complexity and length
        sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
        avg_sentence_len = word_count / sentences
        rho = min(avg_sentence_len / 25.0, 0.95)

        # q (emotional activation): punctuation intensity + emotional markers
        exclamations = text.count('!')
        questions = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        emotional_words = sum(1 for w in words if w.lower() in {
            'feel', 'love', 'hate', 'fear', 'hope', 'pain', 'joy',
            'anger', 'grief', 'loss', 'carry', 'weight', 'heart',
            'soul', 'believe', 'fight', 'struggle', 'survive',
        })
        q = min(
            (exclamations * 0.1 + questions * 0.05 + caps_ratio * 0.5 +
             emotional_words * 0.08 + 0.1),
            0.95
        )

        # f (social belonging): relational language
        social_words = sum(1 for w in words if w.lower() in {
            'we', 'us', 'our', 'they', 'them', 'family', 'community',
            'together', 'people', 'children', 'brother', 'sister',
            'mother', 'father', 'friend', 'neighbor', 'tribe',
        })
        f = min(social_words * 0.1 + 0.05, 0.95)

        # τ (temporal depth): temporal language
        temporal_words = sum(1 for w in words if w.lower() in {
            'always', 'never', 'generations', 'ancestors', 'history',
            'ancient', 'forever', 'remember', 'tradition', 'heritage',
            'roots', 'legacy', 'time', 'years', 'centuries',
        })
        tau = min(temporal_words * 0.12 + 0.05, 0.95)

        # λ (misperception decay): inverse of clarity
        lambda_ = max(0.3 - (psi * 0.2), 0.05)

        return {"psi": psi, "rho": rho, "q": q, "f": f, "tau": tau, "lambda_": lambda_}

    def analyze_text(
        self,
        text: str,
        calibration: str = "western_academic",
    ) -> RoseGlassScore:
        """
        Estimate Rose Glass dimensions from text using LLM analysis.

        Calls Ollama (llama3.2) for dimensional estimation.
        Falls back to keyword heuristic if Ollama is unreachable.
        """
        dims = self._llm_estimate_dimensions(text)
        if dims is None:
            dims = self._heuristic_estimate_dimensions(text)

        return self.analyze_dimensions(
            psi=dims["psi"], rho=dims["rho"], q=dims["q"], f=dims["f"],
            tau=dims["tau"], lambda_=dims["lambda_"],
            calibration=calibration,
        )
