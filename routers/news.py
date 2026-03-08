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
