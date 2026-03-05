"""
End-to-end integration tests.
"""


class TestTextToScores:
    """Text in -> dimensional scores out -> verify structure + plausible values."""

    def test_full_pipeline_text(self, api_client):
        response = api_client.post(
            "/analyze",
            json={
                "text": "I carry the weight of what I have seen.",
                "calibration": "clinical_therapeutic",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Structure checks
        assert "psi" in data
        assert "rho" in data
        assert "q_raw" in data
        assert "q_optimized" in data
        assert "f" in data
        assert "tau" in data
        assert "lambda" in data
        assert "coherence" in data
        assert "calibration" in data
        assert "decomposition" in data
        assert "resilience" in data
        assert "veritas" in data

        # Plausibility checks — dimensions should be in [0, 1]
        assert 0.0 <= data["psi"] <= 1.0
        assert 0.0 <= data["rho"] <= 1.0
        assert 0.0 <= data["q_raw"] <= 1.0
        assert 0.0 <= data["q_optimized"] <= 1.0
        assert 0.0 <= data["f"] <= 1.0
        assert 0.0 <= data["tau"] <= 1.0
        assert data["lambda"] >= 0.0

        # Coherence should be in [0, 4]
        assert 0.0 <= data["coherence"] <= 4.0

        # Resilience sub-structure
        resil = data["resilience"]
        assert "resilience" in resil
        assert "lambda_effective" in resil
        assert "half_life_years" in resil
        assert "net_trajectory" in resil
        assert resil["net_trajectory"] in ("decaying", "restoring", "equilibrium")

        # Veritas sub-structure
        ver = data["veritas"]
        assert "authenticity_score" in ver
        assert "flags" in ver
        assert 0.0 <= ver["authenticity_score"] <= 1.0

    def test_full_pipeline_dimensions(self, api_client):
        response = api_client.post(
            "/analyze/dimensions",
            json={
                "psi": 0.40, "rho": 0.08, "q": 0.90, "f": 0.05,
                "tau": 0.75, "lambda": 0.4,
                "calibration": "western_academic",
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Input values preserved
        assert data["psi"] == 0.40
        assert data["rho"] == 0.08
        assert data["q_raw"] == 0.90
        assert data["f"] == 0.05

        # q_optimized should be < q_raw (biological optimization)
        assert data["q_optimized"] < data["q_raw"]

        # Coherence > 0
        assert data["coherence"] > 0

        # With these low psi/rho/f and high q, coherence should be modest
        assert data["coherence"] < 2.5

    def test_emotional_text_has_higher_q(self, api_client):
        """Text with emotional language should estimate higher q."""
        neutral = api_client.post(
            "/analyze",
            json={"text": "The report contains data about the quarterly results."},
        )
        emotional = api_client.post(
            "/analyze",
            json={"text": "I feel the pain and grief of losing everything I love and fear."},
        )
        assert neutral.status_code == 200
        assert emotional.status_code == 200

        n_data = neutral.json()
        e_data = emotional.json()

        assert e_data["q_raw"] > n_data["q_raw"]

    def test_social_text_has_higher_f(self, api_client):
        """Text with social/relational language should estimate higher f."""
        solo = api_client.post(
            "/analyze",
            json={"text": "The algorithm processes data efficiently."},
        )
        social = api_client.post(
            "/analyze",
            json={"text": "We stand together as a community with our family and children."},
        )
        assert solo.status_code == 200
        assert social.status_code == 200

        assert social.json()["f"] > solo.json()["f"]

    def test_calibration_consistency(self, engine):
        """Engine should return consistent results for same input."""
        s1 = engine.analyze_dimensions(psi=0.5, rho=0.5, q=0.5, f=0.5)
        s2 = engine.analyze_dimensions(psi=0.5, rho=0.5, q=0.5, f=0.5)
        assert s1.coherence == s2.coherence
        assert s1.q == s2.q
