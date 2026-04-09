"""
run_analysis.py — callable wrapper around GDELT + Rose Glass pipeline
Returns structured dict matching the Supabase schema.

v2: Uses CERATA bridge for real dimensional computation instead of
    keyword heuristics. Falls back to heuristic if bridge unreachable.
"""

import re
import sys
import os
from urllib.parse import urlparse

import requests
from google.cloud import bigquery

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.core.rose_glass_v2 import RoseGlassEngine
from scripts.news_compare import SOURCE_CALIBRATIONS, compare_sources

PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "triad-web-analyzer")

# CERATA v2 bridge — real dimensional computation via spaCy NLP
CERATA_BRIDGE_URL = os.environ.get(
    "CERATA_BRIDGE_URL",
    "https://cerata-nematocysts-production.up.railway.app"
)

DOMAIN_SOURCE_MAP = {
    "cnn.com": "mainstream_secular",
    "nytimes.com": "mainstream_secular",
    "washingtonpost.com": "mainstream_secular",
    "bbc.com": "mainstream_secular",
    "bbc.co.uk": "mainstream_secular",
    "reuters.com": "mainstream_secular",
    "apnews.com": "mainstream_secular",
    "theguardian.com": "mainstream_secular",
    "npr.org": "mainstream_secular",
    "abcnews.go.com": "mainstream_secular",
    "nbcnews.com": "mainstream_secular",
    "cbsnews.com": "mainstream_secular",
    "usatoday.com": "mainstream_secular",
    "foxnews.com": "crisis_breaking",
    "nypost.com": "crisis_breaking",
    "dailymail.co.uk": "crisis_breaking",
    "aljazeera.com": "indigenous_press",
    "aljazeera.net": "indigenous_press",
    "middleeasteye.net": "indigenous_press",
    "indiancountrytoday.com": "indigenous_press",
    "christianpost.com": "faith_based",
    "christianitytoday.com": "faith_based",
    "ncronline.org": "faith_based",
    "gov.uk": "legal_regulatory",
    "whitehouse.gov": "legal_regulatory",
    "state.gov": "legal_regulatory",
    "un.org": "legal_regulatory",
    "congress.gov": "legal_regulatory",
    "who.int": "clinical_public_health",
    "cdc.gov": "clinical_public_health",
    "thelancet.com": "clinical_public_health",
    "bmj.com": "clinical_public_health",
}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def _extract_domain(url: str) -> str:
    hostname = urlparse(url).hostname or ""
    parts = hostname.split(".")
    if len(parts) >= 3 and parts[-1] in ("uk", "au", "nz"):
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def _fetch_text(url: str, timeout: int = 10) -> str:
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        text = _TAG_RE.sub(" ", html)
        text = _WHITESPACE_RE.sub(" ", text).strip()
        return text[:3000]
    except Exception:
        return ""


def _gdelt_query(topic: str, date: str, limit: int) -> list[dict]:
    client = bigquery.Client(project=PROJECT_ID)
    topic_title = topic.title()
    topic_upper = topic.upper()
    query = """
        SELECT
            DocumentIdentifier AS url,
            SourceCommonName AS source,
            V2Tone AS v2tone,
            DATE AS date
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME = TIMESTAMP(@partition_date)
            AND (
                V2Locations LIKE @topic_title
                OR V2Themes LIKE @topic_upper
                OR V2Persons LIKE @topic_title
                OR V2Organizations LIKE @topic_title
            )
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("partition_date", "STRING", date),
            bigquery.ScalarQueryParameter("topic_title", "STRING", f"%{topic_title}%"),
            bigquery.ScalarQueryParameter("topic_upper", "STRING", f"%{topic_upper}%"),
            bigquery.ScalarQueryParameter("limit", "INT64", limit * 4),
        ]
    )
    rows = client.query(query, job_config=job_config).result()
    articles = []
    for row in rows:
        articles.append({
            "url": row.url,
            "source": row.source or _extract_domain(row.url),
            "date": str(row.date),
        })
    return articles


# =============================================================================
# CERATA BRIDGE INTEGRATION
# =============================================================================

def _bridge_perceive(text: str, timeout: int = 15) -> dict | None:
    """
    Call CERATA v2 bridge for real dimensional computation.
    Returns dict with psi, rho, q, f, tau, lambda_ or None on failure.
    
    The bridge uses spaCy NLP for:
    - psi: POS diversity, entity density, sentence consistency
    - rho: emergent topology (temporal grounding, reflective register, aphoristic density)
    - q: Michaelis-Menten optimized emotional activation
    - f: text-derived relational graph (belonging, influence, reach)
    """
    try:
        resp = requests.post(
            f"{CERATA_BRIDGE_URL}/cx",
            json={"text": text[:3000]},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            return None

        zones = data.get("zones", {})
        details = data.get("zone_details", {})

        # Extract computed dimensions from bridge response
        psi_detail = details.get("psi", {})
        rho_detail = details.get("rho", {})
        q_detail = details.get("q", {})
        f_detail = details.get("f", {})

        return {
            "psi": psi_detail.get("psi", zones.get("psi", {}).get("A", 0.5)),
            "rho": rho_detail.get("wisdom_score", zones.get("rho", {}).get("A", 0.3)),
            "q": q_detail.get("raw_q", q_detail.get("optimized_q", zones.get("q", {}).get("A", 0.3))),
            "f": f_detail.get("belonging_score", zones.get("f", {}).get("A", 0.1)),
            "tau": data.get("tau", 0.5),
            "lambda_": data.get("lambda", 0.3),
            "compute_source": "cerata-nematocysts-v2",
            "cx": data.get("Cx", 0.0),
            "veritas_ratio": data.get("veritas_ratio", 1.0),
            "has_dark_spot": data.get("has_dark_spot", False),
            "rho_method": rho_detail.get("method", "unknown"),
            "rho_maturity": rho_detail.get("maturity_level", "unknown"),
            "rho_components": rho_detail.get("component_scores", {}),
            "psi_method": psi_detail.get("method", "unknown"),
            "psi_entities": psi_detail.get("entities", []),
        }
    except Exception as e:
        print(f"[bridge] CERATA bridge error: {e}")
        return None


def compare_sources_with_bridge(raw_sources: list[dict], engine: RoseGlassEngine) -> dict:
    """
    Run each source through CERATA bridge first, fall back to heuristic.
    Uses bridge-computed dimensions fed into engine.analyze_dimensions()
    for coherence/resilience/veritas calculations.
    """
    import math
    results = []
    bridge_hits = 0
    bridge_misses = 0

    for src in raw_sources:
        source_type = src["source_type"]
        calibration = SOURCE_CALIBRATIONS.get(source_type, "western_academic")

        # Try bridge first
        bridge_dims = _bridge_perceive(src["text"])

        if bridge_dims:
            bridge_hits += 1
            score = engine.analyze_dimensions(
                psi=bridge_dims["psi"],
                rho=bridge_dims["rho"],
                q=bridge_dims["q"],
                f=bridge_dims["f"],
                tau=bridge_dims.get("tau", 0.5),
                lambda_=bridge_dims.get("lambda_", 0.3),
                calibration=calibration,
            )
        else:
            bridge_misses += 1
            # Fallback to heuristic
            score = engine.analyze_text(src["text"], calibration=calibration)

        results.append({
            "source_name": src["source_name"],
            "source_type": source_type,
            "calibration": calibration,
            "score": score,
            "compute_source": bridge_dims.get("compute_source", "heuristic-fallback") if bridge_dims else "heuristic-fallback",
            "bridge_meta": bridge_dims if bridge_dims else None,
        })

    # Compute per-dimension divergence
    DIMS = ["psi", "rho", "q_raw", "f", "tau", "lambda_", "coherence"]
    DIM_LABELS = {
        "psi": "Psi (consistency)", "rho": "Rho (wisdom)",
        "q_raw": "q (activation)", "f": "f (social)",
        "tau": "Tau (temporal)", "lambda_": "Lambda (decay)",
        "coherence": "Coherence",
    }
    divergence = {}
    for dim in DIMS:
        values = [getattr(r["score"], dim) for r in results]
        mean = sum(values) / len(values) if values else 0
        var = sum((v - mean) ** 2 for v in values) / len(values) if values else 0
        divergence[dim] = {
            "label": DIM_LABELS.get(dim, dim),
            "values": values,
            "mean": round(mean, 4),
            "variance": round(var, 4),
            "std_dev": round(math.sqrt(var), 4),
        }

    print(f"[bridge] CERATA hits: {bridge_hits}, fallbacks: {bridge_misses}")
    return {"results": results, "divergence": divergence}


def run_analysis(topic: str, date_str: str, limit: int = 10) -> dict:
    """
    Full pipeline: GDELT → article fetch → CERATA bridge scoring.
    Returns dict with keys: sources, divergence.
    Each source has: source_name, source_type, calibration, url,
                     article_text, dimensions, coherence, veritas
    """
    articles = _gdelt_query(topic, date_str, limit)
    if not articles:
        return {"sources": [], "divergence": {}}

    # Deduplicate by domain
    seen = set()
    deduped = []
    for a in articles:
        domain = _extract_domain(a["url"])
        if domain and domain not in seen:
            seen.add(domain)
            deduped.append({**a, "domain": domain})
        if len(deduped) >= limit:
            break

    # Fetch article text
    raw_sources = []
    for a in deduped:
        text = _fetch_text(a["url"])
        if not text:
            continue
        source_type = DOMAIN_SOURCE_MAP.get(a["domain"], "mainstream_secular")
        raw_sources.append({
            "source_name": a["source"] or a["domain"],
            "source_type": source_type,
            "text": text,
            "url": a["url"],
        })

    if len(raw_sources) < 2:
        return {"sources": [], "divergence": {}}

    engine = RoseGlassEngine()

    # Use CERATA bridge with heuristic fallback
    comparison = compare_sources_with_bridge(raw_sources, engine)

    # Shape results to match Supabase schema
    out_sources = []
    for r in comparison["results"]:
        score = r["score"]
        out_sources.append({
            "source_name": r["source_name"],
            "source_type": r["source_type"],
            "calibration": r["calibration"],
            "url": next((s["url"] for s in raw_sources if s["source_name"] == r["source_name"]), None),
            "article_text": next((s["text"] for s in raw_sources if s["source_name"] == r["source_name"]), None),
            "dimensions": {
                "psi": score.psi,
                "rho": score.rho,
                "q": score.q_raw,
                "f": score.f,
                "tau": score.tau,
                "lambda": score.lambda_,
            },
            "coherence": score.coherence,
            "veritas": {
                "authenticity_score": score.veritas.get("authenticity_score") if score.veritas else None,
                "flags": score.veritas.get("flags", []) if score.veritas else [],
            },
            "compute_source": r.get("compute_source", "unknown"),
        })

    return {"sources": out_sources, "divergence": comparison.get("divergence", {})}
