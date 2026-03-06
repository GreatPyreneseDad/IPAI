#!/usr/bin/env python3
"""
Rose Glass v2 — News Source Comparison Tool
============================================

Takes a story covered by multiple sources, runs each through the
Rose Glass engine with a culturally appropriate calibration per
source type, and produces a side-by-side dimensional comparison.

No source is ranked better or worse. The tool translates, not judges.

Source type -> calibration mapping:
    mainstream_secular       -> western_academic
    faith_based              -> spiritual_contemplative
    indigenous_press         -> indigenous_oral
    legal_regulatory         -> legal_adversarial
    clinical_public_health   -> clinical_therapeutic
    crisis_breaking          -> crisis_translation
    neurodivergent_community -> neurodivergent

Usage:
    python3 scripts/news_compare.py           # Real story (default)
    python3 scripts/news_compare.py --demo     # Fictional public health demo
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.rose_glass_v2 import RoseGlassEngine, RoseGlassScore

# ---------------------------------------------------------------------------
# Source type -> calibration mapping
# ---------------------------------------------------------------------------

SOURCE_CALIBRATIONS = {
    "mainstream_secular": "western_academic",
    "faith_based": "spiritual_contemplative",
    "indigenous_press": "indigenous_oral",
    "legal_regulatory": "legal_adversarial",
    "clinical_public_health": "clinical_therapeutic",
    "crisis_breaking": "crisis_translation",
    "neurodivergent_community": "neurodivergent",
}

DIMS = ["psi", "rho", "q_raw", "f", "tau", "lambda_", "coherence"]
DIM_LABELS = {
    "psi": "Psi (consistency)",
    "rho": "Rho (wisdom)",
    "q_raw": "q (activation)",
    "f": "f (social)",
    "tau": "Tau (temporal)",
    "lambda_": "Lambda (decay)",
    "coherence": "Coherence",
}


def _get_dim(score: RoseGlassScore, dim: str) -> float:
    return getattr(score, dim)


# ---------------------------------------------------------------------------
# Core comparison logic
# ---------------------------------------------------------------------------

def compare_sources(sources: list[dict], engine: RoseGlassEngine) -> dict:
    """
    Run each source through the engine and produce comparison data.

    Each source dict must have: source_name, source_type, text.
    Returns structured comparison result.
    """
    results = []
    for src in sources:
        source_type = src["source_type"]
        calibration = SOURCE_CALIBRATIONS.get(source_type)
        if calibration is None:
            valid = list(SOURCE_CALIBRATIONS.keys())
            raise ValueError(
                f"Unknown source_type '{source_type}'. Valid: {valid}"
            )

        score = engine.analyze_text(src["text"], calibration=calibration)
        results.append({
            "source_name": src["source_name"],
            "source_type": source_type,
            "calibration": calibration,
            "score": score,
        })

    # Compute per-dimension variance across sources
    divergence = {}
    for dim in DIMS:
        values = [_get_dim(r["score"], dim) for r in results]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        divergence[dim] = {
            "values": values,
            "mean": mean,
            "variance": var,
            "std_dev": math.sqrt(var),
        }

    return {"results": results, "divergence": divergence}


# ---------------------------------------------------------------------------
# Display functions
# ---------------------------------------------------------------------------

def print_calibration_map(results: list[dict]):
    """Print the source type -> calibration mapping used in this run."""
    print("=" * 78)
    print("  SOURCE TYPE -> CALIBRATION MAPPING")
    print("=" * 78)
    print()
    seen = set()
    for r in results:
        key = (r["source_type"], r["calibration"])
        if key not in seen:
            seen.add(key)
            print(f"    {r['source_type']:<28s}  ->  {r['calibration']}")
    print()


def print_per_source(results: list[dict]):
    """Section 1: Per-source dimensional scores."""
    print("=" * 78)
    print("  PER-SOURCE DIMENSIONAL SCORES")
    print("=" * 78)

    for r in results:
        score = r["score"]
        print()
        print(f"  {r['source_name']}")
        print(f"  Type: {r['source_type']}  |  Calibration: {r['calibration']}")
        print(f"  {'─' * 50}")
        for dim in DIMS:
            val = _get_dim(score, dim)
            label = DIM_LABELS[dim]
            bar = "#" * int(val * 30)
            print(f"    {label:<22s}  {val:>6.3f}  {bar}")

        if score.veritas:
            v = score.veritas
            flags = ", ".join(v["flags"]) if v["flags"] else "(none)"
            print(f"    {'Veritas':<22s}  auth={v['authenticity_score']:.2f}  flags={flags}")
    print()


def print_comparison_table(results: list[dict]):
    """Section 2: Side-by-side comparison table."""
    print("=" * 78)
    print("  SIDE-BY-SIDE COMPARISON")
    print("=" * 78)

    names = [r["source_name"] for r in results]
    # Truncate names for table fit
    short_names = [n[:14] for n in names]
    col_w = 14

    # Header
    print()
    header = f"  {'Dimension':<22s}"
    for sn in short_names:
        header += f"  {sn:>{col_w}s}"
    print(header)
    print(f"  {'─' * 22}" + f"  {'─' * col_w}" * len(names))

    for dim in DIMS:
        row = f"  {DIM_LABELS[dim]:<22s}"
        for r in results:
            val = _get_dim(r["score"], dim)
            row += f"  {val:>{col_w}.3f}"
        print(row)
    print()


def print_divergence(divergence: dict):
    """Section 3: Divergence analysis — which dimensions are contested."""
    print("=" * 78)
    print("  DIVERGENCE ANALYSIS (high variance = contested territory)")
    print("=" * 78)
    print()

    ranked = sorted(divergence.items(), key=lambda x: x[1]["variance"], reverse=True)

    for dim, info in ranked:
        label = DIM_LABELS[dim]
        bar = "#" * int(info["std_dev"] * 60)
        print(f"  {label:<22s}  std={info['std_dev']:.3f}  mean={info['mean']:.3f}  {bar}")

    # Identify most and least contested
    most = ranked[0]
    least = ranked[-1]
    print()
    print(f"  Most contested:  {DIM_LABELS[most[0]]} (std={most[1]['std_dev']:.3f})")
    print(f"  Least contested: {DIM_LABELS[least[0]]} (std={least[1]['std_dev']:.3f})")
    print()


def print_summary(results: list[dict], divergence: dict):
    """Section 4: Plain-language summary — what each source emphasizes."""
    print("=" * 78)
    print("  DIMENSIONAL EMPHASIS BY SOURCE")
    print("  (what each source foregrounds that others do not)")
    print("=" * 78)
    print()

    # For each source, find dims where it deviates most from the group mean
    for r in results:
        score = r["score"]
        deviations = []
        for dim in DIMS:
            val = _get_dim(score, dim)
            mean = divergence[dim]["mean"]
            std = divergence[dim]["std_dev"]
            if std > 0.01:
                z = (val - mean) / std
                deviations.append((dim, val, mean, z))

        # Sort by absolute z-score
        deviations.sort(key=lambda x: abs(x[3]), reverse=True)

        print(f"  {r['source_name']} ({r['source_type']})")

        emphasis_parts = []
        for dim, val, mean, z in deviations[:3]:
            label = DIM_LABELS[dim].split("(")[0].strip()
            if abs(z) < 0.5:
                continue
            direction = "higher" if z > 0 else "lower"
            emphasis_parts.append(f"{label} {direction} than group ({val:.2f} vs mean {mean:.2f})")

        if emphasis_parts:
            for part in emphasis_parts:
                print(f"    - {part}")
        else:
            print("    - Dimensions close to group mean across the board")
        print()


# ---------------------------------------------------------------------------
# Real story: US-Israeli military strikes on Iran, March 5 2026
# ---------------------------------------------------------------------------

IRAN_SOURCES = [
    {
        "source_name": "CNN",
        "source_type": "mainstream_secular",
        "text": (
            "Israel said it has begun a broad-scale wave of strikes on regime "
            "infrastructure in Tehran. Defense Secretary Pete Hegseth signaled "
            "that the war will escalate, saying the conflict has only just begun. "
            "President Trump said he has no time limits on how long the war could "
            "go on. The US has hit 200 targets deep inside Iran in the last 72 hours."
        ),
    },
    {
        "source_name": "Al Jazeera",
        "source_type": "indigenous_press",
        "text": (
            "Tehran, Iran — The war might last weeks, so my family and I will "
            "only leave if it gets too bad. For now, life goes on. The booming "
            "sound of explosions has been a daily reality. Iranian authorities "
            "are blocking access to the global internet for a sixth day as the "
            "bombs fall. The internet in Iran is disconnected. We are left without "
            "news while state television says Iran is on the verge of taking over "
            "Tel Aviv and Washington."
        ),
    },
    {
        "source_name": "UK PM Statement",
        "source_type": "legal_regulatory",
        "text": (
            "Iran has now fired drones and missiles at ten countries that did "
            "not attack them. Our number one priority is protecting our people. "
            "My focus is providing calm, level-headed leadership in the national "
            "interest. That means deploying our military and diplomatic strength "
            "to protect our people. The goal is a negotiated settlement with Iran "
            "where they give up their nuclear ambitions."
        ),
    },
    {
        "source_name": "Fox News",
        "source_type": "crisis_breaking",
        "text": (
            "Trump says he does not care if Iran pulls out of the 2026 World Cup. "
            "Republicans rejected a resolution aimed at requiring Trump seek "
            "congressional approval for future military action against Tehran. "
            "The US would start striking progressively deeper into Iran. Israel "
            "vows to kill Iran's next supreme leader."
        ),
    },
]

# ---------------------------------------------------------------------------
# Fictional demo corpus: public health policy announcement
# ---------------------------------------------------------------------------

DEMO_SOURCES = [
    {
        "source_name": "National Herald",
        "source_type": "mainstream_secular",
        "text": (
            "The Department of Health announced today that mandatory air quality "
            "monitoring will be expanded to 200 additional school districts following "
            "a peer-reviewed study linking particulate exposure to developmental delays "
            "in children ages 5 to 12. Officials emphasized the policy is evidence-based "
            "and will be phased in over 18 months. Critics argue the timeline is too slow."
        ),
    },
    {
        "source_name": "The Faithful Observer",
        "source_type": "faith_based",
        "text": (
            "Our children are a sacred trust. The government's decision to monitor air "
            "quality in schools acknowledges what faith communities have long understood: "
            "that caring for the vulnerable is not optional but a moral imperative. We "
            "must ask whether 18 months of waiting honors that imperative, or whether it "
            "asks our children to bear a burden that patience alone cannot justify."
        ),
    },
    {
        "source_name": "First Nations Environmental Watch",
        "source_type": "indigenous_press",
        "text": (
            "For decades, our communities have lived with contaminated air that "
            "researchers are only now measuring in suburban schools. Our elders remember "
            "when the rivers ran clear. This policy is welcome but incomplete — it does "
            "not address the reservations where monitoring has never existed, where our "
            "grandchildren breathe what the cities refuse."
        ),
    },
    {
        "source_name": "Federal Register Digest",
        "source_type": "legal_regulatory",
        "text": (
            "Pursuant to Section 112(d) of the Clean Air Act, the EPA has promulgated "
            "an interim final rule requiring continuous ambient air monitoring in K-12 "
            "facilities within designated nonattainment areas. Compliance deadlines are "
            "staggered by district population. Enforcement provisions include civil "
            "penalties of up to $25,000 per day of violation. Public comment period "
            "closes in 60 days."
        ),
    },
    {
        "source_name": "Public Health Weekly",
        "source_type": "clinical_public_health",
        "text": (
            "The new monitoring mandate responds to a growing body of evidence linking "
            "chronic low-level PM2.5 exposure to neurodevelopmental impairment in "
            "children. Clinicians have reported increased referrals for attention and "
            "processing difficulties in affected districts. The 18-month rollout raises "
            "concerns about continued exposure during the implementation gap. Early "
            "screening protocols may help identify at-risk children before monitors "
            "are installed."
        ),
    },
    {
        "source_name": "Breaking Now Network",
        "source_type": "crisis_breaking",
        "text": (
            "BREAKING: Government orders air monitors in 200 school districts after "
            "study finds toxic air is harming kids. Parents are furious about the "
            "18-month delay. 'My child can't wait a year and a half,' says one mother. "
            "Schools in the worst-hit areas may not see equipment until 2028. "
            "Officials urge calm."
        ),
    },
    {
        "source_name": "The Sensory Inclusive",
        "source_type": "neurodivergent_community",
        "text": (
            "Air quality affects everyone, but for neurodivergent kids the stakes are "
            "compounded. Sensory processing differences mean that environments most "
            "people tolerate can be overwhelming or physically painful. The new "
            "monitoring is a step, but the policy says nothing about accommodations "
            "during the transition — no guidance on ventilation, no recognition that "
            "some children will need immediate environmental changes, not data "
            "collection in 18 months."
        ),
    },
]


def run_comparison(sources: list[dict], title: str):
    """Run the comparison for a given source set."""
    engine = RoseGlassEngine()

    print()
    print("  ROSE GLASS NEWS COMPARISON TOOL")
    print(f"  Story: {title}")
    print(f"  Sources: {len(sources)}")
    print()

    comparison = compare_sources(sources, engine)

    print_calibration_map(comparison["results"])
    print_per_source(comparison["results"])
    print_comparison_table(comparison["results"])
    print_divergence(comparison["divergence"])
    print_summary(comparison["results"], comparison["divergence"])

    print("=" * 78)
    print("  NOTE: No source is ranked better or worse. The Rose Glass lens")
    print("  translates dimensional emphasis — it does not judge.")
    print("=" * 78)
    print()


if __name__ == "__main__":
    if "--demo" in sys.argv:
        run_comparison(
            DEMO_SOURCES,
            "Public Health Air Quality Monitoring Policy (fictional)",
        )
    else:
        run_comparison(
            IRAN_SOURCES,
            "US-Israeli Military Strikes on Iran — Day 5, March 5 2026",
        )
