"""
run_analysis.py — callable wrapper around GDELT + Rose Glass pipeline
Returns structured dict matching the Supabase schema.
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


def run_analysis(topic: str, date_str: str, limit: int = 10) -> dict:
    """
    Full pipeline: GDELT → article fetch → Rose Glass scoring.
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
    comparison = compare_sources(raw_sources, engine)

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
                "authenticity_score": getattr(score, "veritas_score", None),
                "flags": getattr(score, "veritas_flags", []),
            },
        })

    return {"sources": out_sources, "divergence": comparison.get("divergence", {})}
