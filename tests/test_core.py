"""
Core math validation tests for Rose Glass v2.
"""

from src.core.rose_glass_v2 import (
    biological_optimization,
    calculate_coherence,
    tau_attenuated_lambda,
    restoration_integral,
    resilience_complete,
    lambda_decomposition,
    veritas_check,
    DimensionalCalibration,
)


class TestBiologicalOptimization:
    """Michaelis-Menten with substrate inhibition."""

    def test_zero_input(self):
        assert biological_optimization(0.0) == 0.0

    def test_negative_input(self):
        assert biological_optimization(-0.5) == 0.0

    def test_moderate_q_saturation(self):
        """q=0.5 should produce output < q (saturation works)."""
        result = biological_optimization(0.5)
        assert result < 0.5
        assert result > 0.0

    def test_high_q_inhibition(self):
        """q=0.9 should be well below 0.9 (inhibition kicks in)."""
        result = biological_optimization(0.9)
        assert result < 0.9
        # With Km=0.20 and Ki=0.80, inhibition is significant at high q
        assert result < 0.5

    def test_very_high_q_strong_inhibition(self):
        """Extreme q should show strong inhibition effect."""
        result_low = biological_optimization(0.3)
        result_high = biological_optimization(0.95)
        # Due to substrate inhibition, very high q can produce
        # lower output than moderate q
        assert result_high < 0.5

    def test_custom_parameters(self):
        """Custom km and ki should shift the curve."""
        default = biological_optimization(0.5)
        high_km = biological_optimization(0.5, km=0.5)
        # Higher km = more saturation needed = lower output
        assert high_km < default

    def test_output_always_positive(self):
        """Output should always be positive for positive input."""
        for q in [0.01, 0.1, 0.5, 0.9, 0.99]:
            assert biological_optimization(q) > 0


class TestCoherence:
    """Coherence equation: C = Ψ + (ρ×Ψ) + q_opt + (f×Ψ) + coupling."""

    def test_known_inputs_expected_range(self):
        """Known inputs should produce coherence in expected range."""
        result = calculate_coherence(psi=0.7, rho=0.5, q_raw=0.6, f=0.4)
        c = result["coherence"]
        # With these moderate inputs, coherence should be > 0 and < 4
        assert 0.5 < c < 3.0

    def test_zero_inputs(self):
        result = calculate_coherence(psi=0.0, rho=0.0, q_raw=0.0, f=0.0)
        assert result["coherence"] == 0.0

    def test_decomposition_sums_correctly(self):
        """Decomposition components should sum to coherence."""
        result = calculate_coherence(psi=0.6, rho=0.4, q_raw=0.5, f=0.3)
        total = (
            result["base_psi"] +
            result["wisdom_amplification"] +
            result["q_optimized"] +
            result["social_amplification"] +
            result["coupling"]
        )
        assert abs(total - result["coherence"]) < 0.01

    def test_coherence_capped_at_4(self):
        """Coherence should never exceed 4.0."""
        result = calculate_coherence(psi=1.0, rho=1.0, q_raw=1.0, f=1.0)
        assert result["coherence"] <= 4.0

    def test_pattern_intensity_range(self):
        """Pattern intensity = coherence/4.0, should be 0-1."""
        result = calculate_coherence(psi=0.5, rho=0.5, q_raw=0.5, f=0.5)
        assert 0.0 <= result["pattern_intensity"] <= 1.0

    def test_calibration_affects_output(self):
        """Different calibration should produce different results."""
        default_cal = DimensionalCalibration()
        custom_cal = DimensionalCalibration(km=0.45, ki=3.50, coupling_strength=0.30)
        r1 = calculate_coherence(0.5, 0.5, 0.5, 0.5, default_cal)
        r2 = calculate_coherence(0.5, 0.5, 0.5, 0.5, custom_cal)
        assert r1["coherence"] != r2["coherence"]


class TestTauAttenuation:
    """λ_eff = λ₀ / (1 + κτ)."""

    def test_zero_tau_no_attenuation(self):
        """With tau=0, lambda_eff should equal lambda_base."""
        result = tau_attenuated_lambda(0.5, tau=0.0, kappa=0.5)
        assert abs(result - 0.5) < 0.001

    def test_higher_tau_lower_lambda(self):
        """Higher tau should produce lower effective lambda."""
        low_tau = tau_attenuated_lambda(0.5, tau=0.2, kappa=0.5)
        high_tau = tau_attenuated_lambda(0.5, tau=0.8, kappa=0.5)
        assert high_tau < low_tau

    def test_zero_lambda_returns_zero(self):
        assert tau_attenuated_lambda(0.0, tau=0.5, kappa=0.5) == 0.0

    def test_negative_lambda_returns_zero(self):
        assert tau_attenuated_lambda(-0.1, tau=0.5, kappa=0.5) == 0.0

    def test_high_kappa_stronger_attenuation(self):
        """Higher kappa = stronger τ effect."""
        low_k = tau_attenuated_lambda(0.5, tau=0.5, kappa=0.2)
        high_k = tau_attenuated_lambda(0.5, tau=0.5, kappa=0.9)
        assert high_k < low_k


class TestRestorationIntegral:
    """∫₀ᵗ μ(s) × e^(-λ_eff(t-s)) ds."""

    def test_no_events_returns_zero(self):
        assert restoration_integral(0.3, t=5.0, translation_events=[]) == 0.0

    def test_recent_event_contributes_more(self):
        """Recent events should contribute more than distant ones."""
        # Event at t=4 (recent) vs event at t=1 (distant), evaluated at t=5
        recent = restoration_integral(0.3, t=5.0, translation_events=[(4.0, 0.5)])
        distant = restoration_integral(0.3, t=5.0, translation_events=[(1.0, 0.5)])
        assert recent > distant

    def test_future_events_ignored(self):
        """Events after current time should not contribute."""
        result = restoration_integral(0.3, t=5.0, translation_events=[(10.0, 0.5)])
        assert result == 0.0


class TestResilienceComplete:
    """Full resilience model: R(t) = N₀e^(-λ_eff×t) + ∫₀ᵗ μ(s)e^(-λ_eff(t-s)) ds."""

    def test_decay_without_restoration(self):
        """Resilience should decay over time without translation events."""
        result = resilience_complete(n0=0.8, lambda_base=0.3, t=5.0)
        assert result["resilience"] < 0.8
        assert result["restoration_component"] == 0.0
        assert result["net_trajectory"] == "decaying"

    def test_restoration_slows_decay(self):
        """Translation events should increase resilience vs pure decay."""
        no_events = resilience_complete(n0=0.8, lambda_base=0.3, t=5.0)
        with_events = resilience_complete(
            n0=0.8, lambda_base=0.3, t=5.0,
            translation_events=[(2.0, 0.5), (3.0, 0.3)]
        )
        assert with_events["resilience"] > no_events["resilience"]

    def test_tau_slows_decay(self):
        """Higher tau should slow decay (via lambda attenuation)."""
        no_tau = resilience_complete(n0=0.8, lambda_base=0.3, t=5.0, tau=0.0)
        with_tau = resilience_complete(n0=0.8, lambda_base=0.3, t=5.0, tau=0.8, kappa=0.5)
        assert with_tau["resilience"] > no_tau["resilience"]

    def test_resilience_bounded(self):
        """Resilience should be capped at [0, 1]."""
        result = resilience_complete(n0=0.9, lambda_base=0.01, t=0.1,
                                     translation_events=[(0.0, 1.0), (0.05, 1.0)])
        assert result["resilience"] <= 1.0
        assert result["resilience"] >= 0.0

    def test_half_life_computation(self):
        """Half-life should be ln(2)/lambda_eff."""
        result = resilience_complete(n0=0.8, lambda_base=0.3, t=1.0, tau=0.5, kappa=0.4)
        import math
        expected_half_life = math.log(2) / result["lambda_effective"]
        assert abs(result["half_life_years"] - expected_half_life) < 0.01


class TestLambdaDecomposition:
    """λ = w₁λΨ + w₂λρ + w₃λq + w₄λf."""

    def test_equal_lambdas(self):
        result = lambda_decomposition(0.1, 0.1, 0.1, 0.1)
        assert abs(result["lambda_total"] - 0.4) < 0.01

    def test_dominant_vector(self):
        """Should identify the dominant attack vector."""
        result = lambda_decomposition(0.1, 0.1, 0.5, 0.1)
        assert result["dominant_attack_vector"] == "q"

    def test_weights_affect_result(self):
        """Calibration weights should shift contributions."""
        cal = DimensionalCalibration(psi_weight=2.0)
        result = lambda_decomposition(0.3, 0.1, 0.1, 0.1, cal)
        assert result["psi_contribution"] > result["rho_contribution"]


class TestVeritasCheck:
    """Authenticity detection."""

    def test_genuine_expression(self):
        """Balanced but textured dimensions should pass."""
        result = veritas_check(0.6, 0.5, 0.4, 0.7)
        assert result["assessment"] == "genuine"
        assert result["authenticity_score"] > 0.8

    def test_performed_emotion_flagged(self):
        """High q with low psi should flag."""
        result = veritas_check(psi=0.1, rho=0.3, q=0.9, f=0.3)
        assert "high_activation_low_consistency" in result["flags"]
        assert result["authenticity_score"] < 1.0

    def test_theoretical_without_experience_flagged(self):
        """High psi with zero rho should flag."""
        result = veritas_check(psi=0.9, rho=0.05, q=0.3, f=0.3)
        assert "consistency_without_experience" in result["flags"]

    def test_dimensional_texture(self):
        """Genuine expression should have dimensional texture > 0."""
        result = veritas_check(0.3, 0.7, 0.5, 0.2)
        assert result["dimensional_texture"] > 0.0
