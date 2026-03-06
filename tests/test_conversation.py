"""
Phase 3 — Conversational gradient tracking tests.
"""

from unittest.mock import patch, MagicMock
import json

import requests

from src.core.rose_glass_v2 import (
    RoseGlassEngine,
    RoseGlassScore,
    ConversationSession,
    GradientAnalysis,
    analyze_conversation,
)


def _make_score(psi, rho, q, f, tau=0.1, lambda_=0.2, coherence=1.0):
    """Build a minimal RoseGlassScore for testing."""
    return RoseGlassScore(
        psi=psi, rho=rho, q_raw=q, q=q * 0.8, f=f,
        tau=tau, lambda_=lambda_, coherence=coherence,
        calibration="clinical_therapeutic",
        decomposition={"coherence": coherence},
    )


class TestPerformedStability:
    """Performed stability detection requires 3+ turns of high-psi/low-q."""

    def test_performed_stability_detected(self):
        """3 turns of high psi + low q should flag performed stability."""
        session = ConversationSession(
            session_id="test-ps",
            calibration="clinical_therapeutic",
        )
        session.add_turn("I'm fine.", _make_score(psi=0.8, rho=0.3, q=0.1, f=0.1, coherence=1.2))
        session.add_turn("Really, I'm handling it.", _make_score(psi=0.82, rho=0.3, q=0.12, f=0.1, coherence=1.2))
        session.add_turn("Everything is under control.", _make_score(psi=0.79, rho=0.3, q=0.1, f=0.1, coherence=1.2))

        result = analyze_conversation(session)
        assert result.performed_stability_flag is True
        assert "performed stability" in result.signal.lower()
        assert result.turns_analyzed == 3

    def test_not_flagged_with_high_q(self):
        """If q is not consistently low, no performed stability."""
        session = ConversationSession(session_id="test-nps", calibration="clinical_therapeutic")
        session.add_turn("I'm fine.", _make_score(psi=0.8, rho=0.3, q=0.1, f=0.1))
        session.add_turn("Actually I'm angry.", _make_score(psi=0.8, rho=0.3, q=0.7, f=0.1))
        session.add_turn("Very angry.", _make_score(psi=0.8, rho=0.3, q=0.8, f=0.1))

        result = analyze_conversation(session)
        assert result.performed_stability_flag is False

    def test_not_flagged_with_only_two_turns(self):
        """Performed stability requires 3+ turns."""
        session = ConversationSession(session_id="test-2t", calibration="clinical_therapeutic")
        session.add_turn("I'm fine.", _make_score(psi=0.8, rho=0.3, q=0.1, f=0.1))
        session.add_turn("Still fine.", _make_score(psi=0.82, rho=0.3, q=0.1, f=0.1))

        result = analyze_conversation(session)
        assert result.performed_stability_flag is False


class TestTrajectory:
    """Dimension trajectory classification."""

    def test_rising_q_trajectory(self):
        """q increasing across turns should be classified as rising."""
        session = ConversationSession(session_id="test-rq", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.3, q=0.1, f=0.3))
        session.add_turn("t2", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3))
        session.add_turn("t3", _make_score(psi=0.5, rho=0.3, q=0.5, f=0.3))
        session.add_turn("t4", _make_score(psi=0.5, rho=0.3, q=0.7, f=0.3))

        result = analyze_conversation(session)
        assert result.trajectory["q_raw"] == "rising"

    def test_stable_dimensions(self):
        """Flat dimensions should be classified as stable."""
        session = ConversationSession(session_id="test-st", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.5, q=0.5, f=0.5))
        session.add_turn("t2", _make_score(psi=0.5, rho=0.5, q=0.5, f=0.5))
        session.add_turn("t3", _make_score(psi=0.5, rho=0.5, q=0.5, f=0.5))

        result = analyze_conversation(session)
        assert result.trajectory["psi"] == "stable"
        assert result.trajectory["q_raw"] == "stable"

    def test_falling_psi_trajectory(self):
        """psi decreasing should be classified as falling."""
        session = ConversationSession(session_id="test-fp", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.9, rho=0.5, q=0.3, f=0.3))
        session.add_turn("t2", _make_score(psi=0.7, rho=0.5, q=0.3, f=0.3))
        session.add_turn("t3", _make_score(psi=0.5, rho=0.5, q=0.3, f=0.3))
        session.add_turn("t4", _make_score(psi=0.3, rho=0.5, q=0.3, f=0.3))

        result = analyze_conversation(session)
        assert result.trajectory["psi"] == "falling"


class TestCoherenceTrend:
    """Coherence trend computation."""

    def test_improving_coherence(self):
        session = ConversationSession(session_id="test-ic", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.0))
        session.add_turn("t2", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.2))
        session.add_turn("t3", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.4))

        result = analyze_conversation(session)
        assert result.coherence_trend == "improving"

    def test_degrading_coherence(self):
        session = ConversationSession(session_id="test-dc", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.4))
        session.add_turn("t2", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.2))
        session.add_turn("t3", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=0.9))

        result = analyze_conversation(session)
        assert result.coherence_trend == "degrading"

    def test_stable_coherence(self):
        session = ConversationSession(session_id="test-sc", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.2))
        session.add_turn("t2", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.2))
        session.add_turn("t3", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3, coherence=1.2))

        result = analyze_conversation(session)
        assert result.coherence_trend == "stable"


class TestGradientAnalysisStructure:
    """GradientAnalysis output structure."""

    def test_to_dict_keys(self):
        session = ConversationSession(session_id="test-k", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3))
        session.add_turn("t2", _make_score(psi=0.6, rho=0.3, q=0.4, f=0.3))

        result = analyze_conversation(session)
        d = result.to_dict()
        assert "dimension_deltas" in d
        assert "trajectory" in d
        assert "performed_stability_flag" in d
        assert "coherence_trend" in d
        assert "turns_analyzed" in d
        assert "signal" in d
        assert d["turns_analyzed"] == 2

    def test_dimension_deltas_has_all_dims(self):
        session = ConversationSession(session_id="test-dd", calibration="western_academic")
        session.add_turn("t1", _make_score(psi=0.5, rho=0.3, q=0.3, f=0.3))
        session.add_turn("t2", _make_score(psi=0.6, rho=0.4, q=0.4, f=0.4))

        result = analyze_conversation(session)
        for dim in ("psi", "rho", "q_raw", "f", "tau", "lambda_"):
            assert dim in result.dimension_deltas
            assert len(result.dimension_deltas[dim]) == 1


class TestConversationAPI:
    """API endpoint structure tests."""

    def test_start_conversation(self, api_client):
        response = api_client.post(
            "/conversation/start",
            json={"calibration": "clinical_therapeutic"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["calibration"] == "clinical_therapeutic"

    def test_start_default_calibration(self, api_client):
        response = api_client.post("/conversation/start", json={})
        assert response.status_code == 200
        assert response.json()["calibration"] == "western_academic"

    def test_start_invalid_calibration(self, api_client):
        response = api_client.post(
            "/conversation/start",
            json={"calibration": "nonexistent"},
        )
        assert response.status_code == 422

    @patch("src.core.rose_glass_v2.requests.post")
    def test_turn_returns_score(self, mock_post, api_client):
        mock_post.side_effect = requests.ConnectionError("no ollama")

        start = api_client.post(
            "/conversation/start", json={"calibration": "western_academic"},
        )
        sid = start.json()["session_id"]

        resp = api_client.post(
            "/conversation/turn",
            json={"session_id": sid, "text": "Hello world."},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["turn"] == 1
        assert "score" in data
        assert "psi" in data["score"]
        # No gradient for first turn
        assert "gradient" not in data

    @patch("src.core.rose_glass_v2.requests.post")
    def test_turn_returns_gradient_after_two(self, mock_post, api_client):
        mock_post.side_effect = requests.ConnectionError("no ollama")

        start = api_client.post(
            "/conversation/start", json={"calibration": "western_academic"},
        )
        sid = start.json()["session_id"]

        api_client.post("/conversation/turn", json={"session_id": sid, "text": "First."})
        resp = api_client.post("/conversation/turn", json={"session_id": sid, "text": "Second."})

        assert resp.status_code == 200
        data = resp.json()
        assert data["turn"] == 2
        assert "gradient" in data
        assert "trajectory" in data["gradient"]
        assert "signal" in data["gradient"]

    @patch("src.core.rose_glass_v2.requests.post")
    def test_get_session(self, mock_post, api_client):
        mock_post.side_effect = requests.ConnectionError("no ollama")

        start = api_client.post(
            "/conversation/start", json={"calibration": "clinical_therapeutic"},
        )
        sid = start.json()["session_id"]

        api_client.post("/conversation/turn", json={"session_id": sid, "text": "Turn one."})
        api_client.post("/conversation/turn", json={"session_id": sid, "text": "Turn two."})

        resp = api_client.get(f"/conversation/{sid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == sid
        assert data["turns_count"] == 2
        assert len(data["turns"]) == 2
        assert "gradient" in data

    def test_get_nonexistent_session(self, api_client):
        resp = api_client.get("/conversation/nonexistent-id")
        assert resp.status_code == 404

    def test_turn_nonexistent_session(self, api_client):
        resp = api_client.post(
            "/conversation/turn",
            json={"session_id": "nonexistent", "text": "Hello"},
        )
        assert resp.status_code == 404
