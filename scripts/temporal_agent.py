"""
temporal_agent.py — Temporal Intelligence Engine
=================================================

Core engine for roseglass.news temporal intelligence:
- Adaptive interval calculation (12-40 points, Fibonacci weighted)
- Shared pool: check before scoring, deposit after scoring
- GDELT + CERATA bridge for recent dates (< 2 years)
- Haiku recall for historical dates (> 2 years, dead URLs)
- Parallel scoring with ThreadPoolExecutor
- Pattern extraction (drift, spikes, fragmentation, coupling)

Author: Christopher MacGregor bin Joseph
ROSE Corp. | MacGregor Holding Company
"""

import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import psycopg2
import psycopg2.extras
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── config (lazy — env vars must be inside functions on Railway) ──────────────

def _cfg(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val

def get_db():
    return psycopg2.connect(_cfg("SUPABASE_DB_URL"), sslmode="require")

CERATA_BRIDGE_URL_DEFAULT = "https://cerata-nematocysts-production.up.railway.app"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# ── Fibonacci helpers ─────────────────────────────────────────────────────────

def calculate_intervals(start: date, end: date) -> list[date]:
    """Adaptive intervals: keep total points between 12-40. Recent = denser."""
    span = (end - start).days
    if span <= 14:      interval = 1
    elif span <= 90:    interval = 3
    elif span <= 180:   interval = 7
    elif span <= 730:   interval = 30
    else:               interval = 90

    points = []
    current = start
    while current <= end:
        points.append(current)
        current += timedelta(days=interval)
    if points[-1] != end:
        points.append(end)
    return points


def assign_fib_positions(date_points: list[date]) -> list[dict]:
    """Most recent date → fib position 1 (highest weight)."""
    n = len(date_points)
    if n == 0:
        return []
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    fibs = fibs[:n]

    sorted_dates = sorted(date_points, reverse=True)
    return [
        {"date": d, "fib_position": fibs[i], "fib_weight": 1.0 / fibs[i]}
        for i, d in enumerate(sorted_dates)
    ]


def articles_per_point(fib_position: int) -> int:
    """More articles for recent (low fib) positions."""
    if fib_position <= 3:
        return 5
    elif fib_position <= 13:
        return 3
    else:
        return 2


def use_haiku_recall(point_date: date) -> bool:
    """Historical dates (> 2 years ago) use Haiku recall instead of GDELT."""
    cutoff = date.today() - timedelta(days=730)
    return point_date < cutoff


# ── Shared pool operations ────────────────────────────────────────────────────

def pool_has_point(conn, topic: str, point_date: date) -> Optional[dict]:
    """Check if shared pool already has this topic+date. Returns row or None."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT * FROM timeline_points WHERE topic = %s AND point_date = %s",
        (topic.upper(), point_date)
    )
    row = cur.fetchone()
    cur.close()
    return dict(row) if row else None


def increment_search_count(conn, topic: str, point_date: date):
    """Bump search_count for an existing pool point."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE timeline_points
           SET search_count = search_count + 1, last_searched = now()
           WHERE topic = %s AND point_date = %s""",
        (topic.upper(), point_date)
    )
    conn.commit()
    cur.close()


def store_point(conn, topic: str, point_date: date, result: dict,
                fib_position: int, fib_weight: float) -> str:
    """Insert a new scored point into the shared pool. Returns point id."""
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO timeline_points
           (topic, point_date, fib_position, fib_weight,
            psi, rho, q, f, tau, lambda_val, coherence,
            source_count, article_urls, article_summaries,
            compute_source, provenance)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
           ON CONFLICT (topic, point_date) DO UPDATE SET
             search_count = timeline_points.search_count + 1,
             last_searched = now()
           RETURNING id""",
        (
            topic.upper(), point_date, fib_position, fib_weight,
            result.get("psi"), result.get("rho"), result.get("q"),
            result.get("f"), result.get("tau"), result.get("lambda_val"),
            result.get("coherence"), result.get("source_count", 0),
            result.get("article_urls", []),
            json.dumps(result.get("article_summaries", [])),
            result.get("compute_source", "cerata-nematocysts-v2"),
            result.get("provenance", "live"),
        )
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    return str(row[0])


def link_build_point(conn, build_id: str, point_id: str,
                     fib_position: int, fib_weight: float):
    """Link a build to a shared pool point (many-to-many)."""
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO build_points (build_id, point_id, fib_position, fib_weight)
           VALUES (%s, %s, %s, %s)
           ON CONFLICT DO NOTHING""",
        (build_id, point_id, fib_position, fib_weight)
    )
    conn.commit()
    cur.close()


# ── CERATA bridge scoring ────────────────────────────────────────────────────

def bridge_score(text: str, timeout: int = 15) -> Optional[dict]:
    """Call CERATA v2 bridge. Returns dimension dict or None on failure."""
    bridge_url = os.environ.get("CERATA_BRIDGE_URL", CERATA_BRIDGE_URL_DEFAULT)
    try:
        resp = requests.post(
            f"{bridge_url}/cx",
            json={"text": text[:3000]},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            return None

        zones = data.get("zones", {})
        details = data.get("zone_details", {})
        psi_d = details.get("psi", {})
        rho_d = details.get("rho", {})
        q_d = details.get("q", {})
        f_d = details.get("f", {})

        psi = psi_d.get("psi", zones.get("psi", {}).get("A", 0.5))
        rho = rho_d.get("wisdom_score", zones.get("rho", {}).get("A", 0.3))
        q = q_d.get("raw_q", q_d.get("optimized_q", zones.get("q", {}).get("A", 0.3)))
        f = f_d.get("belonging_score", zones.get("f", {}).get("A", 0.1))
        tau = data.get("tau", 0.5)
        lam = data.get("lambda", 0.3)

        coherence = (psi * 0.25 + rho * 0.25 + (1 - abs(q - 0.5) * 2) * 0.2
                     + f * 0.15 + tau * 0.1 + (1 - lam) * 0.05)

        return {
            "psi": psi, "rho": rho, "q": q, "f": f,
            "tau": tau, "lambda_val": lam,
            "coherence": round(coherence, 4),
            "compute_source": "cerata-nematocysts-v2",
        }
    except Exception as e:
        print(f"[temporal_agent] bridge error: {e}")
        return None


# ── Article extraction ────────────────────────────────────────────────────────

import re

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def _extract_article_body(html: str) -> str:
    """Extract article body from HTML. Find largest <p> cluster."""
    # Strip scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Strip nav, header, footer, aside
    for tag in ("nav", "header", "footer", "aside"):
        html = re.sub(rf"<{tag}[^>]*>.*?</{tag}>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Extract text from remaining
    text = _TAG_RE.sub(" ", html)
    text = _WS_RE.sub(" ", text).strip()
    return text[:2000]


_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def fetch_and_extract(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch URL, extract article body. Returns text or None."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        return _extract_article_body(resp.text)
    except Exception:
        return None


# ── GDELT query (reused from run_analysis.py) ─────────────────────────────────

def gdelt_query(topic: str, date_str: str, limit: int = 5) -> list[dict]:
    """Query GDELT BigQuery for articles on topic+date."""
    try:
        from google.cloud import bigquery
    except ImportError:
        print("[temporal_agent] google-cloud-bigquery not installed")
        return []

    project_id = os.environ.get("GCP_PROJECT_ID", "triad-web-analyzer")
    try:
        client = bigquery.Client(project=project_id)
        topic_title = topic.title()
        topic_upper = topic.upper()
        query = """
            SELECT DocumentIdentifier AS url, SourceCommonName AS source
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`
            WHERE _PARTITIONTIME = TIMESTAMP(@partition_date)
              AND (V2Locations LIKE @topic_title
                   OR V2Themes LIKE @topic_upper
                   OR V2Persons LIKE @topic_title
                   OR V2Organizations LIKE @topic_title)
            LIMIT @limit
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("partition_date", "STRING", date_str),
                bigquery.ScalarQueryParameter("topic_title", "STRING", f"%{topic_title}%"),
                bigquery.ScalarQueryParameter("topic_upper", "STRING", f"%{topic_upper}%"),
                bigquery.ScalarQueryParameter("limit", "INT64", limit * 3),
            ]
        )
        rows = client.query(query, job_config=job_config).result()
        from urllib.parse import urlparse
        seen = set()
        articles = []
        for row in rows:
            domain = urlparse(row.url).hostname or ""
            parts = domain.split(".")
            key = ".".join(parts[-2:]) if len(parts) >= 2 else domain
            if key in seen:
                continue
            seen.add(key)
            articles.append({"url": row.url, "source": row.source or key})
            if len(articles) >= limit:
                break
        return articles
    except Exception as e:
        print(f"[temporal_agent] GDELT error: {e}")
        return []


# ── Haiku recall (historical dates, dead URLs) ────────────────────────────────

def haiku_recall(topic: str, point_date: date, num_articles: int = 3) -> list[dict]:
    """
    Ask Haiku for its compressed memory of news coverage.
    Returns list of {source, title, summary, tone}.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("[temporal_agent] No ANTHROPIC_API_KEY for Haiku recall")
        return []

    date_str = point_date.strftime("%B %Y")
    prompt = (
        f"What were the major news stories about {topic} around {date_str}? "
        f"Name specific outlets, events, and the tone of coverage. "
        f"Return JSON only, no markdown: "
        f'{{"articles": [{{"source": "...", "title": "...", '
        f'"summary": "...", "tone": "..."}}]}}. '
        f"Return up to {num_articles} articles."
    )

    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "max_tokens": 1500,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        data = json.loads(raw)
        return data.get("articles", [])
    except Exception as e:
        print(f"[temporal_agent] Haiku recall error: {e}")
        return []


# ── Score a single date point ─────────────────────────────────────────────────

def score_single_point(topic: str, point_date: date,
                       fib_weight: float, fib_position: int) -> dict:
    """
    Score one timeline point. Two modes:
    - Recent (< 2yr): GDELT → fetch article → CERATA bridge
    - Historical (> 2yr): Haiku recall → score summaries through bridge
    Dead URL fallback: if fetch fails, ask Haiku for that source+date.
    """
    date_str = point_date.isoformat()
    num_articles = articles_per_point(fib_position)
    is_historical = use_haiku_recall(point_date)

    scored_dims = []
    urls = []
    summaries = []
    provenance = "live"

    if is_historical:
        # Mode B: Haiku recall
        provenance = "haiku-recall"
        articles = haiku_recall(topic, point_date, num_articles)
        for art in articles:
            summary = art.get("summary", "")
            if not summary:
                continue
            dims = bridge_score(summary)
            if dims:
                dims["compute_source"] = "cerata-nematocysts-v2"
                scored_dims.append(dims)
                summaries.append({
                    "source": art.get("source", "unknown"),
                    "title": art.get("title", ""),
                    "summary": summary,
                    "tone": art.get("tone", ""),
                })
    else:
        # Mode A: GDELT + bridge
        articles = gdelt_query(topic, date_str, num_articles)
        for art in articles:
            text = fetch_and_extract(art["url"])
            if text and len(text) > 100:
                dims = bridge_score(text)
                if dims:
                    scored_dims.append(dims)
                    urls.append(art["url"])
            else:
                # Dead URL fallback — ask Haiku
                fallback = haiku_recall(topic, point_date, 1)
                if fallback:
                    fb_summary = fallback[0].get("summary", "")
                    if fb_summary:
                        dims = bridge_score(fb_summary)
                        if dims:
                            dims["compute_source"] = "cerata-nematocysts-v2"
                            scored_dims.append(dims)
                            summaries.append(fallback[0])
                            if provenance == "live":
                                provenance = "mixed"

    if not scored_dims:
        return None

    # Average dimensions across scored articles
    n = len(scored_dims)
    avg = {}
    for key in ("psi", "rho", "q", "f", "tau", "lambda_val", "coherence"):
        vals = [d.get(key, 0) or 0 for d in scored_dims]
        avg[key] = round(sum(vals) / n, 4)

    avg["source_count"] = n
    avg["article_urls"] = urls
    avg["article_summaries"] = summaries
    avg["provenance"] = provenance
    avg["compute_source"] = scored_dims[0].get("compute_source", "cerata-nematocysts-v2")
    return avg


# ── Build progress tracking ───────────────────────────────────────────────────

def update_build_progress(conn, build_id: str, completed: int, reused: int,
                          status: str = "building"):
    """Update build progress for frontend polling."""
    cur = conn.cursor()
    cur.execute(
        """UPDATE timeline_builds
           SET points_completed = %s, points_reused = %s, status = %s
           WHERE id = %s""",
        (completed, reused, status, build_id)
    )
    conn.commit()
    cur.close()


# ── Main build function ──────────────────────────────────────────────────────

def build_timeline(topic: str, start_date: date, end_date: date) -> dict:
    """
    Main entry point. Creates a build, calculates intervals,
    assigns Fibonacci weights, scores points in parallel (reusing
    shared pool), extracts patterns, returns build summary.
    """
    topic = topic.upper().strip()
    conn = get_db()
    cur = conn.cursor()

    # Calculate intervals and Fibonacci assignments
    date_points = calculate_intervals(start_date, end_date)
    fib_assignments = assign_fib_positions(date_points)
    span_days = (end_date - start_date).days
    interval_days = max(1, span_days // max(len(date_points), 1))

    # Create or get build record
    try:
        cur.execute(
            """INSERT INTO timeline_builds
               (topic, start_date, end_date, interval_days,
                total_points, status)
               VALUES (%s, %s, %s, %s, %s, 'building')
               ON CONFLICT (topic, start_date, end_date) DO UPDATE
                 SET status = 'building',
                     points_completed = 0,
                     points_reused = 0,
                     completed_at = NULL
               RETURNING id""",
            (topic, start_date, end_date, interval_days, len(fib_assignments))
        )
        build_id = str(cur.fetchone()[0])
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to create build: {e}")

    # Phase 1: Check shared pool, identify gaps
    to_score = []
    reused = 0
    reused_points = []

    for dp in fib_assignments:
        existing = pool_has_point(conn, topic, dp["date"])
        if existing:
            reused += 1
            increment_search_count(conn, topic, dp["date"])
            link_build_point(conn, build_id, str(existing["id"]),
                           dp["fib_position"], dp["fib_weight"])
            reused_points.append(existing)
        else:
            to_score.append(dp)

    update_build_progress(conn, build_id, reused, reused, "building")
    print(f"[build] {topic}: {reused} reused, {len(to_score)} to score")

    # Phase 2: Score gaps in parallel
    completed = reused
    scored_points = list(reused_points)

    def _score_and_store(dp):
        """Score one point and store in shared pool. Thread-safe."""
        nonlocal completed
        try:
            result = score_single_point(
                topic, dp["date"], dp["fib_weight"], dp["fib_position"]
            )
            if result is None:
                return None
            # Each thread gets its own connection
            thread_conn = get_db()
            point_id = store_point(
                thread_conn, topic, dp["date"], result,
                dp["fib_position"], dp["fib_weight"]
            )
            link_build_point(
                thread_conn, build_id, point_id,
                dp["fib_position"], dp["fib_weight"]
            )
            thread_conn.close()
            return {**result, "id": point_id, "point_date": dp["date"]}
        except Exception as e:
            print(f"[build] Error scoring {dp['date']}: {e}")
            traceback.print_exc()
            return None

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_score_and_store, dp): dp
            for dp in to_score
        }
        for future in as_completed(futures):
            result = future.result()
            completed += 1
            if result:
                scored_points.append(result)
            # Update progress
            try:
                prog_conn = get_db()
                update_build_progress(prog_conn, build_id, completed,
                                     reused, "building")
                prog_conn.close()
            except Exception:
                pass

    # Phase 3: Pattern extraction
    patterns = extract_patterns(conn, topic, start_date, end_date)

    # Phase 4: Generate pattern summary via Sonnet (if enough data)
    pattern_summary = None
    if patterns and len(scored_points) >= 5:
        pattern_summary = generate_pattern_summary(topic, patterns,
                                                    start_date, end_date)

    # Finalize build
    cur = conn.cursor()
    cur.execute(
        """UPDATE timeline_builds
           SET status = 'complete',
               points_completed = %s,
               points_reused = %s,
               pattern_summary = %s,
               completed_at = now()
           WHERE id = %s""",
        (completed, reused, pattern_summary, build_id)
    )
    conn.commit()
    conn.close()

    return {
        "build_id": build_id,
        "topic": topic,
        "start_date": str(start_date),
        "end_date": str(end_date),
        "total_points": len(fib_assignments),
        "points_scored": len(scored_points),
        "points_reused": reused,
        "status": "complete",
        "pattern_summary": pattern_summary,
        "patterns": patterns,
    }


# ── Pattern extraction ────────────────────────────────────────────────────────

def extract_patterns(conn, topic: str, start_date: date,
                     end_date: date) -> list[dict]:
    """
    Extract dimensional patterns from the shared pool.
    - Drift: weighted linear regression per dimension
    - Spikes: points > 2σ from Fibonacci-weighted mean
    - Fragmentation: Ψ variance over time
    - Coupling: cross-dimensional correlations
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT point_date, psi, rho, q, f, tau, lambda_val,
                  coherence, fib_weight, provenance
           FROM timeline_points
           WHERE topic = %s AND point_date BETWEEN %s AND %s
           ORDER BY point_date ASC""",
        (topic.upper(), start_date, end_date)
    )
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()

    if len(rows) < 3:
        return []

    patterns = []
    dims = ["psi", "rho", "q", "f", "tau", "lambda_val"]

    # Drift detection: Fibonacci-weighted linear regression per dimension
    for dim in dims:
        values = [(r["point_date"], r.get(dim) or 0, r.get("fib_weight") or 1.0)
                  for r in rows if r.get(dim) is not None]
        if len(values) < 3:
            continue
        # Weighted least squares
        x = [(v[0] - values[0][0]).days for v in values]
        y = [v[1] for v in values]
        w = [v[2] for v in values]
        w_sum = sum(w)
        if w_sum == 0:
            continue
        x_mean = sum(xi * wi for xi, wi in zip(x, w)) / w_sum
        y_mean = sum(yi * wi for yi, wi in zip(y, w)) / w_sum
        num = sum(wi * (xi - x_mean) * (yi - y_mean)
                  for xi, yi, wi in zip(x, y, w))
        den = sum(wi * (xi - x_mean) ** 2 for xi, wi in zip(x, w))
        slope = num / den if den != 0 else 0

        if abs(slope) > 0.0005:  # meaningful drift
            direction = "increasing" if slope > 0 else "decreasing"
            dim_label = {"psi": "Ψ consistency", "rho": "ρ wisdom",
                        "q": "q activation", "f": "f belonging",
                        "tau": "τ temporal", "lambda_val": "λ interference"
                        }.get(dim, dim)
            patterns.append({
                "dimension": dim,
                "pattern_type": "drift",
                "magnitude": round(slope * 1000, 4),
                "description": f"{dim_label} {direction} over period "
                               f"(slope: {slope:.6f}/day)",
                "start_date": str(start_date),
                "end_date": str(end_date),
            })

    # Spike detection: points > 2σ from weighted mean
    for dim in dims:
        values = [(r["point_date"], r.get(dim) or 0, r.get("fib_weight") or 1.0)
                  for r in rows if r.get(dim) is not None]
        if len(values) < 5:
            continue
        w = [v[2] for v in values]
        y = [v[1] for v in values]
        w_sum = sum(w)
        if w_sum == 0:
            continue
        w_mean = sum(yi * wi for yi, wi in zip(y, w)) / w_sum
        w_var = sum(wi * (yi - w_mean) ** 2 for yi, wi in zip(y, w)) / w_sum
        w_std = math.sqrt(w_var) if w_var > 0 else 0
        if w_std < 0.01:
            continue

        for pt_date, val, _ in values:
            z = (val - w_mean) / w_std
            if abs(z) > 2.0:
                direction = "spike" if z > 0 else "dip"
                dim_label = {"psi": "Ψ", "rho": "ρ", "q": "q",
                            "f": "f", "tau": "τ", "lambda_val": "λ"
                            }.get(dim, dim)
                patterns.append({
                    "dimension": dim,
                    "pattern_type": "spike",
                    "magnitude": round(z, 2),
                    "description": f"{dim_label} {direction} on {pt_date} "
                                   f"(z={z:.2f}, value={val:.3f})",
                    "start_date": str(pt_date),
                    "end_date": str(pt_date),
                })

    # Narrative fragmentation: Ψ variance over time windows
    psi_vals = [(r["point_date"], r.get("psi") or 0) for r in rows
                if r.get("psi") is not None]
    if len(psi_vals) >= 6:
        mid = len(psi_vals) // 2
        first_half = [v[1] for v in psi_vals[:mid]]
        second_half = [v[1] for v in psi_vals[mid:]]
        var_first = sum((v - sum(first_half)/len(first_half))**2
                       for v in first_half) / len(first_half)
        var_second = sum((v - sum(second_half)/len(second_half))**2
                        for v in second_half) / len(second_half)
        if var_second > var_first * 2:
            patterns.append({
                "dimension": "psi",
                "pattern_type": "fragmentation",
                "magnitude": round(var_second / max(var_first, 0.001), 2),
                "description": "Narrative fragmentation increasing — "
                               "Ψ variance growing over period",
                "start_date": str(psi_vals[mid][0]),
                "end_date": str(psi_vals[-1][0]),
            })
        elif var_first > var_second * 2:
            patterns.append({
                "dimension": "psi",
                "pattern_type": "convergence",
                "magnitude": round(var_first / max(var_second, 0.001), 2),
                "description": "Narrative convergence — "
                               "Ψ variance decreasing, consensus forming",
                "start_date": str(psi_vals[mid][0]),
                "end_date": str(psi_vals[-1][0]),
            })

    # Cross-dimensional coupling: q↑ → ρ↓ (reactive coverage)
    if len(rows) >= 5:
        q_vals = [r.get("q") or 0 for r in rows]
        rho_vals = [r.get("rho") or 0 for r in rows]
        if len(q_vals) == len(rho_vals) and len(q_vals) >= 5:
            n = len(q_vals)
            q_mean = sum(q_vals) / n
            rho_mean = sum(rho_vals) / n
            cov = sum((q_vals[i] - q_mean) * (rho_vals[i] - rho_mean)
                      for i in range(n)) / n
            q_std = math.sqrt(sum((v - q_mean)**2 for v in q_vals) / n)
            rho_std = math.sqrt(sum((v - rho_mean)**2 for v in rho_vals) / n)
            if q_std > 0.01 and rho_std > 0.01:
                corr = cov / (q_std * rho_std)
                if corr < -0.5:
                    patterns.append({
                        "dimension": "q-rho",
                        "pattern_type": "coupling",
                        "magnitude": round(corr, 3),
                        "description": "q↑ ρ↓ coupling detected — "
                                       "emotional activation suppressing "
                                       "wisdom depth (reactive coverage)",
                        "start_date": str(start_date),
                        "end_date": str(end_date),
                    })

    # λ trajectory: topic attention decay/rekindling
    lam_vals = [(r["point_date"], r.get("lambda_val") or 0) for r in rows
                if r.get("lambda_val") is not None]
    if len(lam_vals) >= 4:
        recent_lam = [v[1] for v in lam_vals[-3:]]
        early_lam = [v[1] for v in lam_vals[:3]]
        avg_recent = sum(recent_lam) / len(recent_lam)
        avg_early = sum(early_lam) / len(early_lam)
        if avg_recent > avg_early + 0.1:
            patterns.append({
                "dimension": "lambda_val",
                "pattern_type": "decay",
                "magnitude": round(avg_recent - avg_early, 3),
                "description": "λ rising — topic leaving collective "
                               "consciousness",
                "start_date": str(lam_vals[0][0]),
                "end_date": str(lam_vals[-1][0]),
            })
        elif avg_early > avg_recent + 0.1:
            patterns.append({
                "dimension": "lambda_val",
                "pattern_type": "rekindling",
                "magnitude": round(avg_early - avg_recent, 3),
                "description": "λ dropping — renewed attention, "
                               "topic re-entering consciousness",
                "start_date": str(lam_vals[0][0]),
                "end_date": str(lam_vals[-1][0]),
            })

    # Store patterns in DB
    cur = conn.cursor()
    for p in patterns:
        cur.execute(
            """INSERT INTO timeline_patterns
               (topic, dimension, pattern_type, start_date, end_date,
                magnitude, description)
               VALUES (%s,%s,%s,%s,%s,%s,%s)""",
            (topic, p["dimension"], p["pattern_type"],
             p.get("start_date"), p.get("end_date"),
             p.get("magnitude"), p["description"])
        )
    conn.commit()
    cur.close()
    return patterns


# ── Pattern summary generation (Sonnet) ───────────────────────────────────────

def generate_pattern_summary(topic: str, patterns: list[dict],
                              start_date: date, end_date: date) -> Optional[str]:
    """Ask Sonnet for a plain-language synthesis of detected patterns."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    pattern_text = "\n".join(
        f"- [{p['pattern_type']}] {p['description']}"
        for p in patterns
    )
    prompt = (
        f"You are a Rose Glass temporal analyst. Below are dimensional "
        f"patterns detected in news coverage of {topic} from "
        f"{start_date} to {end_date}.\n\n"
        f"{pattern_text}\n\n"
        f"Synthesize these into 2-3 plain language sentences. "
        f"Describe what the patterns reveal about how coverage "
        f"of this topic evolved. No jargon. No dimension names. "
        f"Just what happened and what it means."
    )
    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"[temporal_agent] Pattern summary error: {e}")
        return None


# ── Build status query (for frontend polling) ─────────────────────────────────

def get_build_status(build_id: str) -> Optional[dict]:
    """Get current build status for frontend progress polling."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute("SELECT * FROM timeline_builds WHERE id = %s", (build_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return None
    return {
        "build_id": str(row["id"]),
        "topic": row["topic"],
        "status": row["status"],
        "total_points": row["total_points"],
        "points_completed": row["points_completed"],
        "points_reused": row["points_reused"],
        "pattern_summary": row["pattern_summary"],
        "start_date": str(row["start_date"]),
        "end_date": str(row["end_date"]),
    }


# ── Fetch timeline points (for frontend chart) ────────────────────────────────

def get_timeline_points(topic: str, start_date: date = None,
                        end_date: date = None) -> list[dict]:
    """Get all shared pool points for a topic within date range."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    query = "SELECT * FROM timeline_points WHERE topic = %s"
    params = [topic.upper()]
    if start_date:
        query += " AND point_date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND point_date <= %s"
        params.append(end_date)
    query += " ORDER BY point_date ASC"
    cur.execute(query, params)
    rows = [dict(r) for r in cur.fetchall()]
    cur.close()
    conn.close()
    # Serialize for JSON
    for r in rows:
        r["id"] = str(r["id"])
        r["point_date"] = str(r["point_date"])
        r["created_at"] = str(r["created_at"]) if r.get("created_at") else None
        r["last_searched"] = str(r["last_searched"]) if r.get("last_searched") else None
    return rows


# ── Annotate (user write-back) ────────────────────────────────────────────────

def add_annotation(topic: str, point_date: str, annotation_type: str,
                   content: str, cross_topic: str = None,
                   start_date: str = None, end_date: str = None) -> str:
    """Add a user annotation to the shared pool. Returns annotation id."""
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO user_annotations
           (topic, point_date, start_date, end_date,
            annotation_type, content, cross_topic)
           VALUES (%s,%s,%s,%s,%s,%s,%s)
           RETURNING id""",
        (topic.upper(), point_date, start_date, end_date,
         annotation_type, content, cross_topic)
    )
    ann_id = str(cur.fetchone()[0])
    conn.commit()
    conn.close()
    return ann_id
