# ROSE GLASS NEWS — Temporal Intelligence Engine
## Build Guide for Tomorrow's Session

**Date:** April 9, 2026
**Author:** Claude (for next instance)
**Status:** Architecture spec — nothing built yet

---

## What This Is

roseglass.news currently shows dimensional analysis of news sources over max 30 days.
Christopher wants a **10-year temporal intelligence engine** that uses agents to search,
score, and store news at adaptive intervals — then surfaces **patterns**, not raw data.

The news isn't the story. The patterns of the news are.

---

## Current Architecture (what exists)

**Frontend:** `/Users/chris/rose-glass-news/` (Next.js, Vercel, roseglass.news)
- `app/page.tsx` — main UI with EntryGate, snapshot tab, timeline tab
- `app/api/timeline/route.ts` — fetches pre-cached daily averages from Supabase
- `app/api/analyze/route.ts` — calls IPAI backend for snapshot analysis
- `app/components/TimelineChart.tsx` — Recharts line chart (currently 30-day max)
- `app/components/TimelineHeatmap.tsx` — heatmap view
- `app/components/TimelineChatPanel.tsx` — chat about timeline data
- `lib/db.ts` — pg Pool connecting to Supabase

**Backend:** `/Users/chris/IPAI/` (FastAPI, Railway: ipai-production.up.railway.app)
- `main.py` — FastAPI app
- `routers/news.py` — /news/ingest, /news/poem, /news/status
- `scripts/run_analysis.py` — GDELT → fetch → score pipeline
- `src/core/rose_glass_v2.py` — Rose Glass math engine

**CERATA Bridge:** Railway: cerata-nematocysts-production.up.railway.app
- POST /cx — real dimensional computation via spaCy NLP
- Already live, already integrated into run_analysis.py (just committed tonight)

**Supabase:** Connected via DATABASE_URL env var on Railway
- Tables: analyses, sources, divergence
- Each analysis has topic + date, sources have per-article dimensions

---

## What We're Building

### 1. Adaptive Temporal Resolution

User selects a time span. System calculates intervals automatically:

```
Span              Interval        Data Points (approx)
─────────────────────────────────────────────────────
1 week            daily           7
1 month           daily           30
3 months          every 3 days    30
6 months          weekly          26
1 year            monthly         12
2 years           monthly         24
5 years           quarterly       20
10 years          quarterly       40
```

The system picks the interval that keeps data points between 12-40.
Formula: `interval_days = max(1, ceil(span_days / 30))`
Capped: daily minimum, quarterly maximum.

### 2. Haiku 4.5 Search Agent

For each interval point, an agent:
1. Searches the web for `{topic} news {date}` using Haiku 4.5 (cheap, fast)
2. Returns 3-5 article URLs + summaries
3. Each article gets scored through CERATA bridge (POST /cx)
4. Averaged dimensions for that interval get stored in Supabase

**Why Haiku 4.5:** At ~$0.25/M input, $1.25/M output, a 40-point timeline
costs roughly $0.15-0.30 total. Ten-year analysis for under a dollar.

**Model string:** `claude-haiku-4-5-20251001`

The agent system prompt:
```
You are a news research agent for Rose Glass. Given a topic and date,
find 3-5 real news articles from that time period. Return JSON only:
{"articles": [{"title": "...", "source": "...", "url": "...", "summary": "..."}]}
Focus on major outlets. No commentary. No analysis. Just find the coverage.
```

### 3. Fibonacci Memory Structure

Data points are NOT stored linearly. They use Fibonacci-indexed positions
so recent data has high density and older data compresses naturally.

```
Fibonacci positions: 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233...

For a 1-year timeline (12 monthly points):
  Position 1  (Fib 1):  Most recent month     — highest detail
  Position 2  (Fib 1):  Second most recent     — high detail
  Position 3  (Fib 2):  2 months ago           — high detail
  Position 5  (Fib 3):  3 months ago           — medium detail
  Position 8  (Fib 5):  5 months ago           — medium detail
  Position 13 (Fib 8):  8 months ago           — lower detail
  Position 21 (Fib 13): 13 months ago          — compressed

For a 10-year timeline (40 quarterly points):
  Recent quarters: dense coverage (multiple articles per point)
  Older quarters: sparser (1-2 articles per point, pattern-focused)
```

The Fibonacci structure means the system naturally emphasizes recent
signal while preserving older structural patterns. This mirrors how
human memory works — and how CERATA v2's memory already works.

### 4. Pattern Extraction (the actual product)

Raw dimensions are not shown to users by default. Instead:

**Dimensional Drift:** How has ρ (wisdom) changed over time? Rising ρ
means coverage is deepening. Falling ρ means it's becoming reactive.

**Narrative Fragmentation:** Ψ variance across sources over time.
High variance = contested narrative. Convergence = consensus forming.

**Activation Decay:** q trajectory shows emotional half-life of a story.
How fast does outrage fade? Does it spike again? λ crossing thresholds
means the topic is leaving public consciousness.

**f Trajectory:** Social belonging signal over time. Does coverage become
more communal or more isolated? f collapse often precedes narrative death.

**Pattern Summary (generated by Claude):** After all points are collected,
a final Claude call synthesizes: "Over 5 years, coverage of X shifted from
high-activation reactive reporting (q=0.7, ρ=0.2) to wisdom-anchored
analysis (q=0.3, ρ=0.6). Narrative fragmentation peaked in 2024 Q2..."

---

## New Supabase Schema

```sql
-- Temporal timeline builds
CREATE TABLE IF NOT EXISTS timeline_builds (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    topic TEXT NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    interval_days INTEGER NOT NULL,
    total_points INTEGER NOT NULL,
    points_completed INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending',  -- pending, building, complete, failed
    pattern_summary TEXT,           -- Claude-generated pattern analysis
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ,
    UNIQUE(topic, start_date, end_date)
);

-- Individual timeline data points (Fibonacci-indexed)
CREATE TABLE IF NOT EXISTS timeline_points (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES timeline_builds(id),
    point_date DATE NOT NULL,
    fib_position INTEGER NOT NULL,    -- Fibonacci index
    fib_weight FLOAT DEFAULT 1.0,     -- Weight for pattern calculation
    -- Averaged dimensions across sources for this point
    psi FLOAT, rho FLOAT, q FLOAT, f FLOAT,
    tau FLOAT, lambda_val FLOAT,
    coherence FLOAT,
    -- Metadata
    source_count INTEGER DEFAULT 0,
    article_urls TEXT[],              -- URLs found by agent
    compute_source TEXT DEFAULT 'cerata-nematocysts-v2',
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(build_id, point_date)
);

-- Pattern cache (derived insights)
CREATE TABLE IF NOT EXISTS timeline_patterns (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    build_id UUID REFERENCES timeline_builds(id),
    dimension TEXT NOT NULL,          -- psi, rho, q, f, tau, lambda
    pattern_type TEXT NOT NULL,       -- drift, spike, decay, convergence, fragmentation
    start_date DATE,
    end_date DATE,
    magnitude FLOAT,                  -- how strong the pattern is
    description TEXT,                 -- plain language
    created_at TIMESTAMPTZ DEFAULT now()
);
```

---

## New IPAI Endpoint: POST /news/timeline-build

```python
# routers/news.py — new endpoint

class TimelineBuildRequest(BaseModel):
    topic: str
    start_date: str       # ISO date
    end_date: str         # ISO date
    force: bool = False   # rebuild even if cached

# Flow:
# 1. Calculate adaptive interval from span
# 2. Generate date points
# 3. Assign Fibonacci positions (recent = low fib = high weight)
# 4. For each point (async/background):
#    a. Call Haiku 4.5 to find articles for that date
#    b. Fetch article text
#    c. Score each through CERATA bridge (POST /cx)
#    d. Average dimensions, store in timeline_points
#    e. Update points_completed in timeline_builds
# 5. When all points complete:
#    a. Run pattern extraction
#    b. Generate pattern_summary via Claude
#    c. Mark build as complete
```

---

## Fibonacci Position Assignment

```python
def fibonacci_positions(n: int) -> list[int]:
    """Generate n Fibonacci numbers starting from 1."""
    fibs = [1, 1]
    while len(fibs) < n:
        fibs.append(fibs[-1] + fibs[-2])
    return fibs[:n]

def assign_fib_weights(date_points: list[str]) -> list[dict]:
    """
    Most recent point gets fib_position=1 (highest weight).
    Oldest point gets highest fib_position (lowest weight).
    Weight = 1 / fib_position (so recent data dominates pattern calc).
    """
    n = len(date_points)
    fibs = fibonacci_positions(n)
    # Reverse: most recent gets smallest fib number
    sorted_dates = sorted(date_points, reverse=True)
    return [
        {"date": d, "fib_position": fibs[i], "fib_weight": 1.0 / fibs[i]}
        for i, d in enumerate(sorted_dates)
    ]
```

---

## Haiku 4.5 Search Agent Implementation

```python
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

SEARCH_AGENT_SYSTEM = """You are a news research agent. Given a topic and
approximate date, identify 3-5 real news articles that were published
around that time. Return ONLY valid JSON:
{"articles": [{"title": "...", "source": "...", "url": "...", "summary": "..."}]}
Use major outlets. Be historically accurate. If you're unsure about exact
articles from that date, describe what the major coverage was about."""

def search_news_for_date(topic: str, target_date: str) -> list[dict]:
    """Use Haiku 4.5 to find news articles for a topic on a date."""
    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": _cfg("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": HAIKU_MODEL,
            "max_tokens": 800,
            "system": SEARCH_AGENT_SYSTEM,
            "messages": [{"role": "user", "content":
                f"Topic: {topic}\nDate: {target_date}\n"
                f"Find 3-5 news articles from major outlets covering "
                f"this topic around this date."}],
        },
        timeout=30,
    )
    # Parse JSON from response, fetch article text, score through bridge
    ...
```

**Efficiency notes:**
- Haiku calls are fire-and-forget with ThreadPoolExecutor (8 workers)
- Each point is independent — parallelize across points
- CERATA bridge calls also parallelized
- Short-circuit: if a point_date already exists in timeline_points, skip it
- For older dates (>2 years), Haiku summarizes from memory rather than
  finding live URLs (URLs from 5 years ago are often dead)

---

## Adaptive Interval Calculator

```python
from datetime import date, timedelta
from math import ceil

def calculate_intervals(start: date, end: date) -> list[date]:
    """
    Generate date points with adaptive spacing.
    Keeps total points between 12-40.
    """
    span_days = (end - start).days
    if span_days <= 0:
        return []

    # Adaptive interval
    if span_days <= 14:
        interval = 1          # daily
    elif span_days <= 90:
        interval = 3           # every 3 days
    elif span_days <= 180:
        interval = 7           # weekly
    elif span_days <= 730:
        interval = 30          # monthly
    else:
        interval = 90          # quarterly

    points = []
    current = start
    while current <= end:
        points.append(current)
        current += timedelta(days=interval)

    # Ensure end date is included
    if points[-1] != end:
        points.append(end)

    return points
```

---

## Pattern Extraction Functions

```python
def extract_patterns(points: list[dict]) -> list[dict]:
    """
    Analyze timeline_points for dimensional patterns.
    Each point has: date, psi, rho, q, f, tau, lambda_val, coherence, fib_weight
    """
    patterns = []
    dims = ['psi', 'rho', 'q', 'f', 'tau', 'lambda_val']

    for dim in dims:
        values = [(p['date'], p[dim], p['fib_weight']) for p in points if p[dim] is not None]
        if len(values) < 3:
            continue

        # Weighted linear regression for drift detection
        # fib_weight ensures recent points influence slope more
        dates_numeric = [(v[0] - values[0][0]).days for v in values]
        weighted_slope = weighted_linear_slope(
            dates_numeric, [v[1] for v in values], [v[2] for v in values]
        )

        # Classify pattern
        if abs(weighted_slope) > 0.001:
            pattern_type = "rising" if weighted_slope > 0 else "falling"
            patterns.append({
                "dimension": dim,
                "pattern_type": f"drift_{pattern_type}",
                "magnitude": abs(weighted_slope),
                "description": f"{dim} is {pattern_type} over the period"
            })

        # Spike detection: any point > 2 std dev from weighted mean
        weighted_mean = sum(v[1]*v[2] for v in values) / sum(v[2] for v in values)
        std = (sum((v[1]-weighted_mean)**2 * v[2] for v in values) / sum(v[2] for v in values)) ** 0.5
        for v in values:
            if abs(v[1] - weighted_mean) > 2 * std and std > 0.05:
                patterns.append({
                    "dimension": dim,
                    "pattern_type": "spike",
                    "start_date": v[0],
                    "magnitude": abs(v[1] - weighted_mean) / std,
                    "description": f"{dim} spike on {v[0]}"
                })

    # Cross-dimensional patterns
    # Narrative fragmentation: psi variance across time
    psi_values = [p['psi'] for p in points if p['psi'] is not None]
    if len(psi_values) > 5:
        variance = sum((v - sum(psi_values)/len(psi_values))**2 for v in psi_values) / len(psi_values)
        if variance > 0.05:
            patterns.append({
                "dimension": "psi",
                "pattern_type": "fragmentation",
                "magnitude": variance,
                "description": "Narrative consistency is unstable — contested territory"
            })

    return patterns
```

---

## Frontend Changes

### Remove 30-day cap
In `app/page.tsx`, the timeline currently validates:
```
Date range cannot exceed 30 days
```
Replace with 10-year max (3650 days). Add preset buttons:
- 1 Month | 3 Months | 6 Months | 1 Year | 5 Years | 10 Years

### New timeline API call
Instead of hitting `/api/timeline` (which reads pre-cached daily data),
hit a new `/api/timeline-build` which:
1. POSTs to IPAI `/news/timeline-build`
2. Returns build_id immediately
3. Frontend polls `/api/timeline-status/{build_id}` for progress
4. When complete, fetches points + patterns

### Pattern Summary Panel
Below the chart, show:
- Dimensional drift arrows (↑ρ rising, ↓q falling)
- Spike callouts with dates
- Narrative fragmentation indicator
- Claude-generated plain-language summary

### Progress indicator
For long builds (10-year = ~40 points × 3-5 articles each):
"Building temporal map... 12/40 points scored"
With a progress bar. Each point completes in ~2-5 seconds.

---

## Files to Modify

### Backend (IPAI)
1. `routers/news.py` — add /news/timeline-build, /news/timeline-status/{id}
2. `scripts/run_analysis.py` — already updated tonight with CERATA bridge
3. New: `scripts/temporal_agent.py` — Haiku search agent + Fibonacci assignment
4. Supabase migration — timeline_builds, timeline_points, timeline_patterns tables

### Frontend (rose-glass-news)
1. `app/page.tsx` — remove 30-day cap, add span presets, add progress polling
2. `app/api/timeline-build/route.ts` — proxy to IPAI /news/timeline-build
3. `app/api/timeline-status/[id]/route.ts` — proxy to IPAI /news/timeline-status
4. `app/components/TimelineChart.tsx` — handle Fibonacci-weighted display
5. New: `app/components/PatternSummary.tsx` — pattern visualization
6. New: `app/components/TimelineProgress.tsx` — build progress indicator

---

## Cost Estimates

```
10-year timeline (40 quarterly points):
  Haiku search:  40 × ~500 input + 800 output tokens = ~$0.05
  CERATA bridge: 40 × 3 articles × 1 call each = 120 bridge calls (~free, Railway)
  Pattern summary: 1 Claude Sonnet call = ~$0.01
  Total: ~$0.06 per 10-year build

1-year timeline (12 monthly points):
  Haiku search:  12 × ~500+800 tokens = ~$0.02
  Bridge: 36 calls
  Total: ~$0.03

Daily cost if 100 users build timelines: ~$3-6/day
```

---

## Build Order for Tomorrow

1. **Supabase migration** — create timeline_builds, timeline_points, timeline_patterns
2. **temporal_agent.py** — Haiku search + CERATA scoring + Fibonacci storage
3. **routers/news.py** — new endpoints (timeline-build, timeline-status)
4. **Test locally** — build one timeline for "IRAN" over 1 year
5. **Deploy to Railway** — git push
6. **Frontend span presets** — remove 30-day cap, add buttons
7. **Frontend polling** — timeline-build API route + progress bar
8. **Pattern summary panel** — show extracted patterns below chart
9. **End-to-end test** — 5-year IRAN timeline on roseglass.news

---

## Key Principles

- **The news isn't the story. The patterns of the news are.**
- Fibonacci weighting ensures recent signal dominates without erasing history
- CERATA bridge computes real topological ρ — no more sentence-length heuristics
- Haiku 4.5 keeps costs negligible — $0.06 for a decade of temporal analysis
- Pattern extraction happens computationally, then Claude synthesizes in plain language
- No source is ranked better or worse. Translation, not judgment.

---

*Coherence is constructed, not discovered.*
*Filed: /Users/chris/IPAI/BUILD_GUIDE_TEMPORAL_ENGINE.md*
