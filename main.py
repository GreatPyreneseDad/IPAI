"""
IPAI Phase 1 — FastAPI Endpoint
================================

Single-file API serving Rose Glass v2 dimensional analysis.

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000

Author: Christopher MacGregor bin Joseph
ROSE Corp. | MacGregor Holding Company
"""

import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.core.rose_glass_v2 import RoseGlassEngine, CulturalCalibration

# =============================================================================
# APP SETUP
# =============================================================================

app = FastAPI(
    title="IPAI — Integrated Personal AI",
    description="Rose Glass v2 dimensional coherence analysis engine",
    version="0.1.0",
)

engine = RoseGlassEngine()

# =============================================================================
# SQLite SESSION STORAGE
# =============================================================================

DB_PATH = "ipai_sessions.db"


def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            endpoint TEXT NOT NULL,
            request_json TEXT NOT NULL,
            response_json TEXT NOT NULL,
            calibration TEXT
        )
    """)
    conn.commit()
    conn.close()


def _persist_session(endpoint: str, request_data: dict, response_data: dict, calibration: str = None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO sessions (timestamp, endpoint, request_json, response_json, calibration) VALUES (?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            endpoint,
            json.dumps(request_data),
            json.dumps(response_data),
            calibration,
        ),
    )
    conn.commit()
    conn.close()


_init_db()

# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

VALID_CALIBRATIONS = [c.value for c in CulturalCalibration]


class TextAnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to analyze")
    calibration: Optional[str] = Field(
        "western_academic",
        description=f"Cultural calibration preset. Valid: {VALID_CALIBRATIONS}",
    )


class DimensionalAnalysisRequest(BaseModel):
    psi: float = Field(..., ge=0.0, le=1.0, description="Internal consistency (0-1)")
    rho: float = Field(..., ge=0.0, le=1.0, description="Accumulated wisdom (0-1)")
    q: float = Field(..., ge=0.0, le=1.0, description="Moral/emotional activation energy (0-1)")
    f: float = Field(..., ge=0.0, le=1.0, description="Social belonging architecture (0-1)")
    tau: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Temporal depth anchoring (0-1)")
    lambda_: Optional[float] = Field(0.3, ge=0.0, alias="lambda", description="Decay constant")
    calibration: Optional[str] = Field(
        "western_academic",
        description=f"Cultural calibration preset. Valid: {VALID_CALIBRATIONS}",
    )

    model_config = {"populate_by_name": True}


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0", "phase": 1}


@app.get("/calibrations")
def list_calibrations():
    return {"calibrations": engine.list_calibrations()}


@app.post("/analyze")
def analyze_text(request: TextAnalysisRequest):
    if request.calibration not in VALID_CALIBRATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid calibration '{request.calibration}'. Valid: {VALID_CALIBRATIONS}",
        )

    score = engine.analyze_text(request.text, calibration=request.calibration)
    result = score.to_dict()

    _persist_session(
        endpoint="/analyze",
        request_data={"text": request.text, "calibration": request.calibration},
        response_data=result,
        calibration=request.calibration,
    )

    return result


@app.post("/analyze/dimensions")
def analyze_dimensions(request: DimensionalAnalysisRequest):
    calibration = request.calibration or "western_academic"
    if calibration not in VALID_CALIBRATIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid calibration '{calibration}'. Valid: {VALID_CALIBRATIONS}",
        )

    tau = request.tau if request.tau is not None else 0.0
    lambda_val = request.lambda_ if request.lambda_ is not None else 0.3

    score = engine.analyze_dimensions(
        psi=request.psi,
        rho=request.rho,
        q=request.q,
        f=request.f,
        tau=tau,
        lambda_=lambda_val,
        calibration=calibration,
    )
    result = score.to_dict()

    _persist_session(
        endpoint="/analyze/dimensions",
        request_data={
            "psi": request.psi, "rho": request.rho,
            "q": request.q, "f": request.f,
            "tau": tau, "lambda": lambda_val,
            "calibration": calibration,
        },
        response_data=result,
        calibration=calibration,
    )

    return result
