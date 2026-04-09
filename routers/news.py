"""
IPAI News Router — /news endpoints
===================================
POST /news/ingest   GDELT → score → Supabase
POST /news/poem     unpoemed sources → Claude → Supabase
GET  /news/status   row counts
"""

import concurrent.futures
import os
import re
import threading
from datetime import date as date_type
from typing import Optional

import psycopg2
import psycopg2.extras
import requests
from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from scripts.run_analysis import run_analysis

# ── config — lazy so missing env vars don't crash at import time ──────────────
def _cfg(key: str) -> str:
    val = os.environ.get(key, "")
    if not val:
        raise RuntimeError(f"Missing required env var: {key}")
    return val

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL         = "claude-sonnet-4-20250514"

STANDING_TOPICS = [
    "IRAN","UKRAINE","CLIMATE","ELECTION","ECONOMY",
    "CHINA","ISRAEL","AI","NATO","NIGERIA",
    "BRAZIL","AUSTRALIA","INDIA","MEXICO",
]

router = APIRouter(prefix="/news", tags=["news"])

# ── db ────────────────────────────────────────────────────────────────────────
def get_db():
    return psycopg2.connect(_cfg("SUPABASE_DB_URL"), sslmode="require")

def is_cached(conn, topic: str, date_str: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM analyses WHERE UPPER(topic)=UPPER(%s) AND date=%s LIMIT 1",
        (topic, date_str)
    )
    return cur.fetchone() is not None

def save_analysis(conn, topic: str, date_str: str, result: dict) -> Optional[str]:
    sources = result.get("sources", [])
    if len(sources) < 2:
        return None
    avg_coh = sum(s["coherence"] for s in sources) / len(sources)
    if avg_coh < 0.3:
        return None
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO analyses (topic, date) VALUES (%s,%s) ON CONFLICT DO NOTHING RETURNING id",
            (topic.upper(), date_str)
        )
        row = cur.fetchone()
        if not row:
            conn.rollback()
            return None
        analysis_id = row[0]
        for s in sources:
            dims    = s.get("dimensions", {})
            veritas = s.get("veritas") or {}
            cur.execute("""
                INSERT INTO sources
                  (analysis_id, source_name, source_type, calibration, url, article_text,
                   psi, rho, q, f, tau, lambda_val, coherence,
                   veritas_score, veritas_assessment)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                analysis_id,
                s.get("source_name"), s.get("source_type"), s.get("calibration"),
                s.get("url"), s.get("article_text"),
                dims.get("psi"), dims.get("rho"), dims.get("q"),
                dims.get("f"), dims.get("tau"), dims.get("lambda"),
                s.get("coherence"),
                veritas.get("authenticity_score"),
                ", ".join(veritas.get("flags", [])) or None,
            ))
        for dim, info in result.get("divergence", {}).items():
            cur.execute(
                "INSERT INTO divergence (analysis_id, dimension, mean_val, std_dev, variance)"
                " VALUES (%s,%s,%s,%s,%s)",
                (analysis_id, dim, info["mean"], info["std_dev"], info["variance"])
            )
        conn.commit()
        return str(analysis_id)
    except Exception as e:
        conn.rollback()
        raise e

# ── poem helpers ──────────────────────────────────────────────────────────────
CULTURAL_LENSES = [
    "western_liberal","state_nationalist","pan_islamic","humanitarian",
    "realpolitik","indigenous_oral","revolutionary","technocratic",
    "apocalyptic","grievance",
]

POEM_SYSTEM = """You are a Rose Glass agent. You read news articles and write compressed witness poems.

Your output is ALWAYS exactly this format:

LENS: [one lens name]
POEM:
[line 1]
[line 2]
[line 3]
[optional line 4]
[optional line 5]

Rules:
- MINIMUM 3 lines. Never fewer.
- Name real actors, places, actions — no vague abstractions
- Write from inside the detected lens, not about it
- Free verse, no rhyme required
- Do not reference "lens" or "framing" in the poem

Lens options: western_liberal, state_nationalist, pan_islamic, humanitarian,
realpolitik, indigenous_oral, revolutionary, technocratic, apocalyptic, grievance"""

def _parse_poem(raw: str):
    lens_m  = re.search(r"LENS:\s*(\S+)", raw)
    poem_m  = re.search(r"POEM:\s*\n+([\s\S]+?)$", raw.strip())
    if not lens_m or not poem_m:
        return None, None
    lens  = lens_m.group(1).strip().lower().rstrip(".,:")
    lines = [l for l in poem_m.group(1).split("\n") if l.strip()]
    if len(lines) < 3:
        return None, None
    if lens not in CULTURAL_LENSES:
        for known in CULTURAL_LENSES:
            if known.startswith(lens[:6]) or lens.startswith(known[:6]):
                lens = known
                break
        else:
            lens = "western_liberal"
    return lens, "\n".join(lines)

def _generate_poem(source: dict) -> tuple:
    dims = source
    dim_str = (f"Ψ={dims.get('psi',0) or 0:.2f} ρ={dims.get('rho',0) or 0:.2f} "
               f"q={dims.get('q',0) or 0:.2f} f={dims.get('f',0) or 0:.2f} "
               f"τ={dims.get('tau',0) or 0:.2f} λ={dims.get('lambda_val',0) or 0:.2f}")
    prompt = (f"Source: {source.get('source_name','unknown')}\n"
              f"Topic: {source.get('topic','')}\nDate: {source.get('date','')}\n"
              f"Dimensions: {dim_str}\n\n"
              f"Article:\n{(source.get('article_text') or '')[:2500]}\n\n"
              f"Write the witness poem.")
    try:
        resp = requests.post(
            ANTHROPIC_URL,
            headers={
                "x-api-key": _cfg("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": MODEL,
                "max_tokens": 400,
                "system": POEM_SYSTEM,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        raw = resp.json()["content"][0]["text"].strip()
        return _parse_poem(raw)
    except Exception:
        return None, None

# ── request models ────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    topics:  Optional[list[str]] = None
    date:    Optional[str]       = None
    limit:   int                 = 10

class PoemRequest(BaseModel):
    topic:   Optional[str] = None
    limit:   int           = 100
    workers: int           = 8

# ── endpoints ─────────────────────────────────────────────────────────────────
@router.get("/status")
def news_status():
    conn = get_db()
    cur  = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM analyses"); analyses = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sources");  sources  = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM sources WHERE poem IS NOT NULL"); poems = cur.fetchone()[0]
    conn.close()
    return {"analyses": analyses, "sources": sources, "poems": poems}

@router.post("/ingest")
def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
    topics   = [t.upper() for t in req.topics] if req.topics else list(STANDING_TOPICS)
    date_str = req.date or str(date_type.today())

    def _run():
        conn    = get_db()
        results = {"saved": [], "skipped": [], "errors": []}
        for topic in topics:
            if is_cached(conn, topic, date_str):
                results["skipped"].append(topic)
                continue
            try:
                result = run_analysis(topic, date_str, limit=req.limit)
                if not result.get("sources"):
                    results["skipped"].append(f"{topic}(no_articles)")
                    continue
                aid = save_analysis(conn, topic, date_str, result)
                if aid:
                    results["saved"].append({"topic": topic, "id": aid,
                                              "sources": len(result["sources"])})
                else:
                    results["skipped"].append(f"{topic}(low_coherence)")
            except Exception as e:
                results["errors"].append({"topic": topic, "error": str(e)})
        conn.close()
        return results

    if len(topics) <= 3:
        results = _run()
        return {"date": date_str, "topics": topics, **results}
    else:
        background_tasks.add_task(_run)
        return {"date": date_str, "topics": topics, "status": "running_background"}

@router.post("/poem")
def poem_ingest(req: PoemRequest):
    conn = get_db()
    cur  = conn.cursor()
    query = """
        SELECT s.id, s.source_name, s.article_text,
               s.psi, s.rho, s.q, s.f, s.tau, s.lambda_val,
               a.topic, a.date::text
        FROM sources s JOIN analyses a ON s.analysis_id = a.id
        WHERE s.article_text IS NOT NULL
          AND LENGTH(s.article_text) > 100
          AND s.poem IS NULL
    """
    params: list = []
    if req.topic:
        query  += " AND UPPER(a.topic)=UPPER(%s)"
        params.append(req.topic)
    query  += " ORDER BY a.date DESC LIMIT %s"
    params.append(req.limit)
    cur.execute(query, params)
    cols    = [d[0] for d in cur.description]
    sources = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()

    if not sources:
        return {"processed": 0, "saved": 0, "failed": 0}

    stats = {"saved": 0, "failed": 0}
    lock  = threading.Lock()

    def _process(source):
        lens, poem = _generate_poem(source)
        if not lens or not poem:
            with lock: stats["failed"] += 1
            return
        try:
            c   = get_db()
            cur = c.cursor()
            cur.execute(
                "UPDATE sources SET poem=%s, cultural_lens=%s, poem_generated_at=NOW() WHERE id=%s",
                (poem, lens, source["id"])
            )
            c.commit()
            c.close()
            with lock: stats["saved"] += 1
        except Exception:
            with lock: stats["failed"] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=req.workers) as ex:
        concurrent.futures.wait([ex.submit(_process, s) for s in sources])

    return {"processed": len(sources), **stats}


# ── temporal intelligence endpoints ───────────────────────────────────────────

from scripts.temporal_agent import (
    build_timeline, get_build_status, get_timeline_points,
    add_annotation,
)
from datetime import date as _date, datetime as _datetime


class TimelineBuildRequest(BaseModel):
    topic: str
    start_date: str   # YYYY-MM-DD
    end_date: str     # YYYY-MM-DD


class AnnotateRequest(BaseModel):
    topic: str
    point_date: str
    annotation_type: str  # context, pattern_label, cross_topic
    content: str
    cross_topic: Optional[str] = None


@router.post("/timeline-build")
def timeline_build(req: TimelineBuildRequest, background_tasks: BackgroundTasks):
    """
    Start a temporal timeline build. Runs in background for long spans.
    Returns build_id immediately for progress polling.
    """
    topic = req.topic.upper().strip()
    try:
        sd = _date.fromisoformat(req.start_date)
        ed = _date.fromisoformat(req.end_date)
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD."}, 400

    if ed <= sd:
        return {"error": "end_date must be after start_date."}, 400

    span_days = (ed - sd).days

    # Short spans: run synchronously. Long spans: background.
    if span_days <= 30:
        try:
            result = build_timeline(topic, sd, ed)
            return result
        except Exception as e:
            return {"error": str(e)}, 500
    else:
        # Create build record first, return ID for polling
        conn = get_db()
        cur = conn.cursor()
        from scripts.temporal_agent import calculate_intervals
        points = calculate_intervals(sd, ed)
        interval = max(1, span_days // max(len(points), 1))
        cur.execute(
            """INSERT INTO timeline_builds
               (topic, start_date, end_date, interval_days,
                total_points, status)
               VALUES (%s,%s,%s,%s,%s,'pending')
               ON CONFLICT (topic, start_date, end_date) DO UPDATE
                 SET status='pending', points_completed=0,
                     points_reused=0, completed_at=NULL
               RETURNING id""",
            (topic, sd, ed, interval, len(points))
        )
        build_id = str(cur.fetchone()[0])
        conn.commit()
        conn.close()

        background_tasks.add_task(build_timeline, topic, sd, ed)
        return {"build_id": build_id, "status": "building", "topic": topic}


@router.get("/timeline-status/{build_id}")
def timeline_status(build_id: str):
    """Poll build progress. Frontend calls this every 2 seconds."""
    status = get_build_status(build_id)
    if not status:
        return {"error": "Build not found"}, 404
    return status


@router.get("/timeline-points/{topic}")
def timeline_points_endpoint(topic: str, start: str = None, end: str = None):
    """Get all shared pool points for a topic."""
    sd = _date.fromisoformat(start) if start else None
    ed = _date.fromisoformat(end) if end else None
    points = get_timeline_points(topic, sd, ed)
    return {"topic": topic.upper(), "points": points}


@router.post("/annotate")
def annotate(req: AnnotateRequest):
    """User write-back: add annotation to shared pool."""
    if req.annotation_type not in ("context", "pattern_label", "cross_topic"):
        return {"error": "Invalid annotation_type"}, 400
    ann_id = add_annotation(
        req.topic, req.point_date, req.annotation_type,
        req.content, req.cross_topic,
    )
    return {"annotation_id": ann_id, "status": "stored"}


@router.get("/annotations/{topic}")
def get_annotations(topic: str):
    """Get all annotations for a topic."""
    conn = get_db()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """SELECT id, topic, point_date::text, start_date::text,
                  end_date::text, annotation_type, content,
                  cross_topic, upvotes, created_at::text
           FROM user_annotations
           WHERE topic = %s ORDER BY point_date ASC""",
        (topic.upper(),)
    )
    rows = [dict(r) for r in cur.fetchall()]
    for r in rows:
        r["id"] = str(r["id"])
    conn.close()
    return {"topic": topic.upper(), "annotations": rows}
