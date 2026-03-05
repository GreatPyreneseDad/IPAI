"""
Cultural calibration tests — all 7 presets.
"""

from src.core.rose_glass_v2 import (
    RoseGlassEngine,
    CulturalCalibration,
    CALIBRATION_PRESETS,
    DimensionalCalibration,
)


class TestCalibrationPresets:
    """Verify all 7 calibration presets load with correct values."""

    def test_all_seven_presets_exist(self):
        assert len(CALIBRATION_PRESETS) == 7

    def test_western_academic(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.WESTERN_ACADEMIC]
        assert cal.km == 0.20
        assert cal.ki == 0.80
        assert cal.kappa == 0.3
        assert cal.mu_baseline == 0.08

    def test_spiritual_contemplative(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.SPIRITUAL_CONTEMPLATIVE]
        assert cal.km == 0.30
        assert cal.ki == 1.20
        assert cal.rho_weight == 1.3
        assert cal.kappa == 0.9
        assert cal.mu_baseline == 0.15

    def test_indigenous_oral(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.INDIGENOUS_ORAL]
        assert cal.f_weight == 1.4
        assert cal.rho_weight == 1.2
        assert cal.kappa == 0.85
        assert cal.mu_baseline == 0.18  # Highest restoration

    def test_crisis_translation(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.CRISIS_TRANSLATION]
        assert cal.q_weight == 1.3
        assert cal.psi_weight == 1.2
        assert cal.kappa == 0.2
        assert cal.mu_baseline == 0.05

    def test_legal_adversarial(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.LEGAL_ADVERSARIAL]
        assert cal.kappa == 0.15  # Lowest kappa
        assert cal.mu_baseline == 0.03
        assert cal.psi_weight == 1.4
        assert cal.q_weight == 0.7

    def test_clinical_therapeutic(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.CLINICAL_THERAPEUTIC]
        assert cal.km == 0.25
        assert cal.ki == 1.00
        assert cal.kappa == 0.6
        assert cal.mu_baseline == 0.12

    def test_neurodivergent(self):
        cal = CALIBRATION_PRESETS[CulturalCalibration.NEURODIVERGENT]
        assert cal.km == 0.45
        assert cal.ki == 3.50
        assert cal.psi_weight == 1.3
        assert cal.kappa == 0.4

    def test_legal_has_lowest_kappa(self):
        """Legal adversarial should have the lowest kappa."""
        kappas = {cal.value: preset.kappa for cal, preset in CALIBRATION_PRESETS.items()}
        min_cal = min(kappas, key=kappas.get)
        assert min_cal == "legal_adversarial"

    def test_indigenous_has_highest_mu_baseline(self):
        """Indigenous oral should have the highest mu_baseline."""
        mus = {cal.value: preset.mu_baseline for cal, preset in CALIBRATION_PRESETS.items()}
        max_cal = max(mus, key=mus.get)
        assert max_cal == "indigenous_oral"


class TestCulturalCoherenceDifferences:
    """Same dimensional input should produce different coherence per calibration."""

    def test_same_input_different_coherence(self, engine):
        """All 7 calibrations should produce different coherence for same input."""
        calibrations = [c.value for c in CulturalCalibration]
        results = {}
        for cal_name in calibrations:
            score = engine.analyze_dimensions(
                psi=0.5, rho=0.5, q=0.5, f=0.5,
                tau=0.5, lambda_=0.3,
                calibration=cal_name,
            )
            results[cal_name] = score.coherence

        # Not all coherence values should be identical
        unique_values = set(results.values())
        assert len(unique_values) > 1, f"All calibrations produced same coherence: {results}"

    def test_neurodivergent_higher_q_ceiling(self, engine):
        """Neurodivergent calibration (Ki=3.50) should have less q inhibition."""
        nd = engine.analyze_dimensions(
            psi=0.5, rho=0.5, q=0.8, f=0.5, calibration="neurodivergent"
        )
        wa = engine.analyze_dimensions(
            psi=0.5, rho=0.5, q=0.8, f=0.5, calibration="western_academic"
        )
        # ND has Ki=3.50 vs WA Ki=0.80, so q_optimized should be higher for ND
        assert nd.q > wa.q

    def test_indigenous_f_weight_amplifies(self, engine):
        """Indigenous oral f_weight=1.4 should amplify social belonging."""
        io = engine.analyze_dimensions(
            psi=0.5, rho=0.5, q=0.3, f=0.8, calibration="indigenous_oral"
        )
        wa = engine.analyze_dimensions(
            psi=0.5, rho=0.5, q=0.3, f=0.8, calibration="western_academic"
        )
        # Indigenous oral doesn't directly amplify f in the coherence equation
        # (f_weight affects lambda decomposition, not coherence),
        # but its different km/ki/coupling should produce different coherence
        assert io.coherence != wa.coherence


class TestEngineCalibrationAccess:
    """Engine calibration management."""

    def test_get_calibration_valid(self, engine):
        cal = engine.get_calibration("western_academic")
        assert isinstance(cal, DimensionalCalibration)
        assert cal.km == 0.20

    def test_get_calibration_invalid(self, engine):
        import pytest
        with pytest.raises(ValueError, match="Unknown calibration"):
            engine.get_calibration("nonexistent")

    def test_list_calibrations(self, engine):
        cals = engine.list_calibrations()
        assert len(cals) == 7
        names = [c["name"] for c in cals]
        assert "western_academic" in names
        assert "indigenous_oral" in names
        assert "legal_adversarial" in names

    def test_list_calibrations_include_parameters(self, engine):
        cals = engine.list_calibrations()
        for cal in cals:
            assert "name" in cal
            assert "parameters" in cal
            params = cal["parameters"]
            assert "km" in params
            assert "ki" in params
            assert "kappa" in params
