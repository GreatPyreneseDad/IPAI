#!/usr/bin/env python3
"""
GDELT + Rose Glass News Comparison
====================================

Queries GDELT's Global Knowledge Graph via BigQuery for articles on a topic,
fetches article text, and runs each through the Rose Glass engine with
culturally appropriate calibrations.

Usage:
    python3 scripts/gdelt_news_compare.py --topic IRAN --date 2026-03-05
    python3 scripts/gdelt_news_compare.py --topic CLIMATE --date 2026-03-04 --limit 15
"""

import argparse
import re
import sys
import os
from urllib.parse import urlparse

import requests
from google.cloud import bigquery

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.rose_glass_v2 import RoseGlassEngine
from scripts.news_compare import (
    SOURCE_CALIBRATIONS,
    compare_sources,
    print_calibration_map,
    print_per_source,
    print_comparison_table,
    print_divergence,
    print_summary,
)

PROJECT_ID = "project-cbd5d6c3-e99a-41b4-bf5"

# Domain -> source_type mapping for known outlets
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


def _extract_domain(url: str) -> str:
    """Extract root domain from URL (e.g. 'www.cnn.com' -> 'cnn.com')."""
    hostname = urlparse(url).hostname or ""
    parts = hostname.split(".")
    if len(parts) >= 3 and parts[-1] in ("uk", "au", "nz"):
        return ".".join(parts[-3:])
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


def map_source_to_calibration(source_domain: str) -> str:
    """Map a source domain to its Rose Glass calibration via source_type."""
    source_type = DOMAIN_SOURCE_MAP.get(source_domain, "mainstream_secular")
    return SOURCE_CALIBRATIONS[source_type]


def map_source_to_type(source_domain: str) -> str:
    """Map a source domain to its source_type."""
    return DOMAIN_SOURCE_MAP.get(source_domain, "mainstream_secular")


# ---------------------------------------------------------------------------
# GDELT BigQuery
# ---------------------------------------------------------------------------

def gdelt_query(topic: str, date: str, limit: int = 10) -> list[dict]:
    """
    Query GDELT GKG for articles matching a topic on a given date.

    Args:
        topic: Theme keyword (e.g. 'IRAN', 'CLIMATE')
        date: Date string 'YYYY-MM-DD'
        limit: Max articles to return

    Returns:
        List of {url, source, v2tone, date} dicts.
    """
    client = bigquery.Client(project=PROJECT_ID)

    query = """
        SELECT
            DocumentIdentifier AS url,
            SourceCommonName AS source,
            V2Tone AS v2tone,
            DATE AS date
        FROM `gdelt-bq.gdeltv2.gkg_partitioned`
        WHERE _PARTITIONTIME = TIMESTAMP(@partition_date)
            AND V2Themes LIKE @theme_pattern
        LIMIT @limit
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("partition_date", "STRING", date),
            bigquery.ScalarQueryParameter("theme_pattern", "STRING", f"%{topic.upper()}%"),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )

    results = client.query(query, job_config=job_config).result()

    articles = []
    for row in results:
        tone_parts = (row.v2tone or "").split(",")
        tone = {
            "overall": float(tone_parts[0]) if len(tone_parts) > 0 and tone_parts[0] else 0.0,
            "positive": float(tone_parts[1]) if len(tone_parts) > 1 and tone_parts[1] else 0.0,
            "negative": float(tone_parts[2]) if len(tone_parts) > 2 and tone_parts[2] else 0.0,
            "polarity": float(tone_parts[3]) if len(tone_parts) > 3 and tone_parts[3] else 0.0,
            "activity_density": float(tone_parts[4]) if len(tone_parts) > 4 and tone_parts[4] else 0.0,
            "self_ref": float(tone_parts[5]) if len(tone_parts) > 5 and tone_parts[5] else 0.0,
        }
        articles.append({
            "url": row.url,
            "source": row.source or _extract_domain(row.url),
            "v2tone": tone,
            "date": str(row.date),
        })

    return articles


# ---------------------------------------------------------------------------
# Article fetching
# ---------------------------------------------------------------------------

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")


def fetch_article_text(url: str, timeout: int = 10) -> str:
    """
    Fetch article text from a URL using simple HTML stripping.
    Returns empty string on any failure.
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        html = resp.text

        # Strip script and style blocks
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)

        # Strip all HTML tags
        text = _TAG_RE.sub(" ", html)
        # Normalize whitespace
        text = _WHITESPACE_RE.sub(" ", text).strip()
        # Take first 1000 chars
        return text[:1000]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

def run_gdelt_comparison(topic: str, date_str: str, limit: int = 10):
    """
    Full pipeline: query GDELT, fetch articles, run Rose Glass comparison.
    """
    print()
    print(f"  Querying GDELT for '{topic}' on {date_str} (limit={limit})...")
    articles = gdelt_query(topic, date_str, limit=limit * 3)  # over-fetch for dedup

    if not articles:
        print("  No articles found. Check topic/date and try again.")
        return

    print(f"  Found {len(articles)} raw articles from GDELT.")

    # Deduplicate by domain — one article per domain
    seen_domains = set()
    deduped = []
    for article in articles:
        domain = _extract_domain(article["url"])
        if domain and domain not in seen_domains:
            seen_domains.add(domain)
            deduped.append({**article, "domain": domain})
        if len(deduped) >= limit:
            break

    print(f"  {len(deduped)} unique domains after deduplication.")
    print()

    # Fetch article text
    sources = []
    for article in deduped:
        domain = article["domain"]
        print(f"  Fetching: {domain} ...", end=" ", flush=True)
        text = fetch_article_text(article["url"])
        if not text:
            print("(failed, skipping)")
            continue
        print(f"({len(text)} chars)")

        source_type = map_source_to_type(domain)
        sources.append({
            "source_name": f"{article['source']} ({domain})",
            "source_type": source_type,
            "text": text,
            "v2tone": article["v2tone"],
        })

    if len(sources) < 2:
        print("\n  Not enough articles with text for comparison (need >= 2).")
        return

    # Print GDELT tone data before Rose Glass analysis
    print()
    print("=" * 78)
    print("  GDELT V2TONE (raw sentiment from GDELT)")
    print("=" * 78)
    for src in sources:
        tone = src["v2tone"]
        print(f"\n  {src['source_name']}")
        print(f"    Overall: {tone['overall']:+.2f}  "
              f"Pos: {tone['positive']:.2f}  Neg: {tone['negative']:.2f}  "
              f"Polarity: {tone['polarity']:.2f}")
    print()

    # Run Rose Glass comparison
    engine = RoseGlassEngine()

    print(f"  ROSE GLASS NEWS COMPARISON (via GDELT)")
    print(f"  Topic: {topic}")
    print(f"  Date: {date_str}")
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
    parser = argparse.ArgumentParser(
        description="GDELT + Rose Glass news comparison"
    )
    parser.add_argument("--topic", required=True, help="Theme keyword (e.g. IRAN, CLIMATE)")
    parser.add_argument("--date", required=True, help="Date in YYYY-MM-DD format")
    parser.add_argument("--limit", type=int, default=10, help="Max articles to compare (default: 10)")
    args = parser.parse_args()

    run_gdelt_comparison(args.topic, args.date, args.limit)
