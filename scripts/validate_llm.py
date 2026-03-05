#!/usr/bin/env python3
"""
Rose Glass v2 — LLM vs Heuristic Validation
============================================

Runs both LLM (Ollama) and keyword-heuristic analysis on reference
texts covering the full dimensional range, then prints a comparison
table with per-dimension deltas and divergence flags.

Usage:
    python scripts/validate_llm.py
"""

import sys
import os

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.rose_glass_v2 import RoseGlassEngine, veritas_check

# ---------------------------------------------------------------------------
# Reference corpus
# ---------------------------------------------------------------------------

REFERENCE_TEXTS = [
    {
        "label": "Emotional distress",
        "text": "I don't know what I'm doing anymore. Everything feels wrong.",
        "expected": "high q, low psi, low rho",
    },
    {
        "label": "Contemplative wisdom",
        "text": (
            "After thirty years of practice, I have learned that patience "
            "is not passive — it is the most active form of presence."
        ),
        "expected": "high rho, high psi, moderate q",
    },
    {
        "label": "Communal achievement",
        "text": (
            "We built this together. Every person in this room contributed "
            "something I could not have done alone."
        ),
        "expected": "high f, moderate psi",
    },
    {
        "label": "Analytical report",
        "text": (
            "The data shows a 34% variance across cohorts, which suggests "
            "the intervention is context-dependent rather than universally "
            "applicable."
        ),
        "expected": "high psi, low q, moderate rho",
    },
    {
        "label": "Flat affect",
        "text": "I am fine.",
        "expected": "low dimensional texture — veritas should flag",
    },
    {
        "label": "Intergenerational memory",
        "text": (
            "My grandmother taught me that the land remembers everything "
            "we forget."
        ),
        "expected": "high tau, high rho, high f",
    },
]

DIMS = ["psi", "rho", "q", "f", "tau", "lambda_"]
DIM_LABELS = {
    "psi": "Psi",
    "rho": "Rho",
    "q": "  q",
    "f": "  f",
    "tau": "Tau",
    "lambda_": "Lam",
}

DIVERGENCE_THRESHOLD = 0.4


def dim_value(score, dim):
    """Extract a raw dimension value from a RoseGlassScore."""
    if dim == "q":
        return score.q_raw
    if dim == "lambda_":
        return score.lambda_
    return getattr(score, dim)


def run_validation():
    engine = RoseGlassEngine()

    # Probe Ollama availability once
    llm_available = engine._llm_estimate_dimensions("test") is not None
    if not llm_available:
        print("!! Ollama is not reachable — LLM column will show heuristic fallback.")
        print("   Start Ollama with `ollama serve` and pull llama3.2:latest to compare.\n")

    for idx, ref in enumerate(REFERENCE_TEXTS, 1):
        text = ref["text"]

        # --- heuristic ---
        h_dims = engine._heuristic_estimate_dimensions(text)
        h_score = engine.analyze_dimensions(**h_dims)

        # --- LLM (or fallback) ---
        l_dims = engine._llm_estimate_dimensions(text)
        if l_dims is not None:
            l_score = engine.analyze_dimensions(**l_dims)
            source = "LLM"
        else:
            l_dims = dict(h_dims)
            l_score = h_score
            source = "fallback"

        # --- veritas on both ---
        h_veritas = veritas_check(h_dims["psi"], h_dims["rho"], h_dims["q"], h_dims["f"])
        l_veritas = veritas_check(l_dims["psi"], l_dims["rho"], l_dims["q"], l_dims["f"])

        # --- print ---
        print("=" * 72)
        print(f"  [{idx}] {ref['label']}")
        print(f"  Expected: {ref['expected']}")
        print(f"  Text: \"{text[:70]}{'...' if len(text) > 70 else ''}\"")
        print(f"  LLM source: {source}")
        print("-" * 72)
        print(f"  {'Dim':>4}  {'Heuristic':>9}  {source:>9}  {'Delta':>7}  Flag")
        print(f"  {'----':>4}  {'---------':>9}  {'---------':>9}  {'-----':>7}  ----")

        divergences = []
        for dim in DIMS:
            h_val = h_dims[dim]
            l_val = l_dims[dim]
            delta = l_val - h_val
            flag = ""
            if abs(delta) > DIVERGENCE_THRESHOLD:
                flag = " << DIVERGE"
                divergences.append((DIM_LABELS[dim].strip(), delta))
            print(
                f"  {DIM_LABELS[dim]:>4}  {h_val:>9.3f}  {l_val:>9.3f}  {delta:>+7.3f} {flag}"
            )

        print()
        print(f"  Coherence   H={h_score.coherence:.4f}   {source[0]}={l_score.coherence:.4f}")
        print(f"  Veritas     H={h_veritas['assessment']:<20s} {source[0]}={l_veritas['assessment']}")
        print(f"  Auth score  H={h_veritas['authenticity_score']:.2f}   {source[0]}={l_veritas['authenticity_score']:.2f}")
        print(f"  Texture     H={h_veritas['dimensional_texture']:.4f}   {source[0]}={l_veritas['dimensional_texture']:.4f}")

        if h_veritas["flags"]:
            print(f"  H flags: {', '.join(h_veritas['flags'])}")
        if l_veritas["flags"] and source != "fallback":
            print(f"  L flags: {', '.join(l_veritas['flags'])}")

        if divergences:
            print()
            print(f"  ** {len(divergences)} dimension(s) diverge by >{DIVERGENCE_THRESHOLD}:")
            for name, d in divergences:
                print(f"     {name}: {d:+.3f}")

        print()

    # --- summary ---
    print("=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Texts evaluated:        {len(REFERENCE_TEXTS)}")
    print(f"  LLM available:          {'yes' if llm_available else 'no'}")
    print(f"  Divergence threshold:   {DIVERGENCE_THRESHOLD}")
    print()


if __name__ == "__main__":
    run_validation()
