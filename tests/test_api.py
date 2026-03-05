"""
API endpoint validation tests.
"""

import pytest


class TestHealthEndpoint:

    def test_health_returns_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert data["phase"] == 1


class TestCalibrationsEndpoint:

    def test_calibrations_returns_7(self, api_client):
        response = api_client.get("/calibrations")
        assert response.status_code == 200
        data = response.json()
        assert len(data["calibrations"]) == 7

    def test_calibrations_include_parameters(self, api_client):
        response = api_client.get("/calibrations")
        data = response.json()
        for cal in data["calibrations"]:
            assert "name" in cal
            assert "parameters" in cal
            assert "km" in cal["parameters"]
            assert "kappa" in cal["parameters"]


class TestAnalyzeTextEndpoint:

    def test_analyze_text_success(self, api_client):
        response = api_client.post(
            "/analyze",
            json={"text": "I carry the weight of what I have seen.", "calibration": "clinical_therapeutic"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "psi" in data
        assert "rho" in data
        assert "q_raw" in data
        assert "q_optimized" in data
        assert "f" in data
        assert "coherence" in data
        assert "calibration" in data
        assert "decomposition" in data
        assert data["calibration"] == "clinical_therapeutic"

    def test_analyze_text_default_calibration(self, api_client):
        response = api_client.post(
            "/analyze",
            json={"text": "The world is full of wonder."},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["calibration"] == "western_academic"

    def test_analyze_text_invalid_calibration(self, api_client):
        response = api_client.post(
            "/analyze",
            json={"text": "Hello world", "calibration": "invalid_calibration"},
        )
        assert response.status_code == 422

    def test_analyze_text_empty_text(self, api_client):
        response = api_client.post(
            "/analyze",
            json={"text": ""},
        )
        assert response.status_code == 422

    def test_analyze_text_includes_veritas(self, api_client):
        response = api_client.post(
            "/analyze",
            json={"text": "I feel the pain of generations of loss and displacement."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "veritas" in data


class TestAnalyzeDimensionsEndpoint:

    def test_dimensions_success(self, api_client):
        response = api_client.post(
            "/analyze/dimensions",
            json={
                "psi": 0.40, "rho": 0.08, "q": 0.90, "f": 0.05,
                "tau": 0.75, "calibration": "western_academic",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "psi" in data
        assert data["psi"] == 0.40
        assert "coherence" in data
        assert "decomposition" in data
        assert "resilience" in data

    def test_dimensions_with_lambda(self, api_client):
        response = api_client.post(
            "/analyze/dimensions",
            json={
                "psi": 0.5, "rho": 0.5, "q": 0.5, "f": 0.5,
                "lambda": 0.5,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["lambda"] == 0.5

    def test_dimensions_invalid_calibration(self, api_client):
        response = api_client.post(
            "/analyze/dimensions",
            json={
                "psi": 0.5, "rho": 0.5, "q": 0.5, "f": 0.5,
                "calibration": "nonexistent",
            },
        )
        assert response.status_code == 422

    def test_dimensions_out_of_range(self, api_client):
        response = api_client.post(
            "/analyze/dimensions",
            json={"psi": 1.5, "rho": 0.5, "q": 0.5, "f": 0.5},
        )
        assert response.status_code == 422

    def test_dimensions_all_calibrations(self, api_client):
        """All 7 calibrations should work."""
        calibrations = [
            "western_academic", "spiritual_contemplative", "indigenous_oral",
            "crisis_translation", "legal_adversarial", "clinical_therapeutic",
            "neurodivergent",
        ]
        for cal in calibrations:
            response = api_client.post(
                "/analyze/dimensions",
                json={"psi": 0.5, "rho": 0.5, "q": 0.5, "f": 0.5, "calibration": cal},
            )
            assert response.status_code == 200, f"Failed for calibration: {cal}"
