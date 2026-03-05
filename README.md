# IPAI — Integrated Personal AI

A dimensional coherence analysis engine built on the Rose Glass v2 mathematics from WP-2026-001. IPAI translates human expression through four dimensional wavelengths — internal consistency (Ψ), accumulated wisdom (ρ), moral/emotional activation energy (q), and social belonging architecture (f) — calibrated across seven cultural contexts. Coherence is constructed, not discovered.

## Quick Start

```bash
cd IPAI
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Start server
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Usage

### Analyze text
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I carry the weight of what I have seen.", "calibration": "clinical_therapeutic"}'
```

### Analyze dimensions directly
```bash
curl -X POST http://localhost:8000/analyze/dimensions \
  -H "Content-Type: application/json" \
  -d '{"psi": 0.40, "rho": 0.08, "q": 0.90, "f": 0.05, "tau": 0.75, "calibration": "western_academic"}'
```

### List calibrations
```bash
curl http://localhost:8000/calibrations
```

## Cultural Calibrations

| Calibration | κ (tau-attenuation) | μ baseline | Context |
|---|---|---|---|
| `western_academic` | 0.30 | 0.08 | Standard academic/clinical |
| `spiritual_contemplative` | 0.90 | 0.15 | Contemplative traditions |
| `indigenous_oral` | 0.85 | 0.18 | Oral tradition communities |
| `crisis_translation` | 0.20 | 0.05 | Acute crisis contexts |
| `legal_adversarial` | 0.15 | 0.03 | Adversarial legal systems |
| `clinical_therapeutic` | 0.60 | 0.12 | Therapeutic settings |
| `neurodivergent` | 0.40 | 0.07 | Neurodivergent perception |

## Phase Roadmap

- **Phase 1** (current): Rose Glass v2 engine, FastAPI endpoint, test suite
- **Phase 2**: LLM integration, triadic processor activation
- **Phase 3**: Dashboard, Docker deployment, analytics
- **Phase 4**: Blockchain coherence ledger

## Attribution

Christopher MacGregor bin Joseph
MacGregor Holding Company | ROSE Corp.

Based on: *Grounded Coherence Theory — A Formal Framework* (WP-2026-001, Revised)
