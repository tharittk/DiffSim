"""Unit tests for diffsim.data.seismic module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.data.seismic import (
    DEFAULT_AI,
    DEFAULT_ROCK_PROPERTIES,
    compute_reflectivity_vertical,
    compute_rms_cube,
    compute_synthetic_seismic_3d,
    facies_to_ai,
    generate_rms_from_facies_3d,
    normalize_rms,
    ricker_wavelet,
)


# ---------------------------------------------------------------------------
# ricker_wavelet
# ---------------------------------------------------------------------------
class TestRickerWavelet:
    def test_peak_at_center(self):
        t, w = ricker_wavelet(25.0, 0.001, 0.1)
        # Peak should be at t≈0
        peak_idx = np.argmax(w)
        assert abs(t[peak_idx]) < 0.001 + 1e-12

    def test_peak_value_is_one(self):
        _, w = ricker_wavelet(25.0, 0.001, 0.1)
        assert abs(w.max() - 1.0) < 1e-10

    def test_symmetric(self):
        _, w = ricker_wavelet(30.0, 0.001, 0.08)
        # Should be symmetric about center
        np.testing.assert_allclose(w, w[::-1], atol=1e-12)

    def test_has_negative_sidelobes(self):
        _, w = ricker_wavelet(25.0, 0.001, 0.2)
        assert w.min() < 0, "Ricker wavelet should have negative sidelobes"

    def test_length_scales_with_duration(self):
        _, w_short = ricker_wavelet(25.0, 0.001, 0.05)
        _, w_long = ricker_wavelet(25.0, 0.001, 0.20)
        assert len(w_long) > len(w_short)

    def test_returns_time_and_wavelet(self):
        t, w = ricker_wavelet(25.0, 0.01, 0.1)
        assert t.shape == w.shape
        assert len(t) > 0


# ---------------------------------------------------------------------------
# facies_to_ai — deterministic mode
# ---------------------------------------------------------------------------
class TestFaciesToAiDeterministic:
    def test_default_ai_values(self):
        facies = np.array([[0, 1, 2]], dtype=np.int8)
        ai = facies_to_ai(facies)
        np.testing.assert_array_equal(
            ai, [[DEFAULT_AI[0], DEFAULT_AI[1], DEFAULT_AI[2]]]
        )

    def test_custom_ai_values(self):
        custom = {0: 100.0, 1: 200.0, 2: 300.0}
        facies = np.array([0, 1, 2, 0], dtype=np.int8)
        ai = facies_to_ai(facies, ai_values=custom)
        np.testing.assert_array_equal(ai, [100.0, 200.0, 300.0, 100.0])

    def test_output_dtype_float32(self):
        facies = np.array([0, 1], dtype=np.int8)
        assert facies_to_ai(facies).dtype == np.float32

    def test_preserves_shape_2d(self):
        facies = np.zeros((5, 7), dtype=np.int8)
        assert facies_to_ai(facies).shape == (5, 7)

    def test_preserves_shape_3d(self):
        facies = np.ones((3, 4, 5), dtype=np.int8)
        assert facies_to_ai(facies).shape == (3, 4, 5)

    def test_uniform_facies(self):
        facies = np.full((4, 4), 2, dtype=np.int8)
        ai = facies_to_ai(facies)
        np.testing.assert_array_equal(ai, np.full((4, 4), DEFAULT_AI[2]))


# ---------------------------------------------------------------------------
# facies_to_ai — stochastic mode
# ---------------------------------------------------------------------------
class TestFaciesToAiStochastic:
    def test_stochastic_varies_per_voxel(self):
        facies = np.zeros((10, 10), dtype=np.int8)  # all mud
        rng = np.random.default_rng(42)
        ai = facies_to_ai(facies, rock_properties=DEFAULT_ROCK_PROPERTIES, rng=rng)
        assert len(set(ai.ravel())) > 1, "Stochastic AI should vary per voxel"

    def test_stochastic_reproducible(self):
        facies = np.array([0, 1, 2, 0, 1, 2], dtype=np.int8)
        ai1 = facies_to_ai(
            facies,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(7),
        )
        ai2 = facies_to_ai(
            facies,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(ai1, ai2)

    def test_stochastic_different_seeds_differ(self):
        facies = np.array([0, 1, 2], dtype=np.int8)
        ai1 = facies_to_ai(
            facies,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(1),
        )
        ai2 = facies_to_ai(
            facies,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(2),
        )
        assert not np.array_equal(ai1, ai2)

    def test_stochastic_physically_sensible(self):
        facies = np.array([0, 1, 2] * 100, dtype=np.int8)
        rng = np.random.default_rng(0)
        ai = facies_to_ai(facies, rock_properties=DEFAULT_ROCK_PROPERTIES, rng=rng)
        # AI = rhob * vp; with defaults roughly in 5000–12000 range
        assert ai.min() > 4000
        assert ai.max() < 15000

    def test_stochastic_dtype_float32(self):
        facies = np.array([0, 1, 2], dtype=np.int8)
        ai = facies_to_ai(
            facies,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(0),
        )
        assert ai.dtype == np.float32

    def test_stochastic_skips_missing_code(self):
        # Only code 0 present — codes 1, 2 in rock_properties should be harmlessly skipped
        facies = np.zeros((3, 3), dtype=np.int8)
        ai = facies_to_ai(
            facies,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(0),
        )
        assert ai.shape == (3, 3)
        assert np.all(ai > 0)

    def test_rock_properties_takes_precedence_over_ai_values(self):
        facies = np.array([0, 1, 2], dtype=np.int8)
        ai_det = facies_to_ai(facies, ai_values=DEFAULT_AI)
        ai_stoch = facies_to_ai(
            facies,
            ai_values=DEFAULT_AI,
            rock_properties=DEFAULT_ROCK_PROPERTIES,
            rng=np.random.default_rng(0),
        )
        # Stochastic should override — values won't match the constants
        assert not np.array_equal(ai_det, ai_stoch)


# ---------------------------------------------------------------------------
# compute_reflectivity_vertical
# ---------------------------------------------------------------------------
class TestComputeReflectivityVertical:
    def test_output_shape(self):
        ai = np.ones((4, 5, 10), dtype=np.float32)
        r = compute_reflectivity_vertical(ai)
        assert r.shape == (4, 5, 9)

    def test_uniform_ai_gives_zero_reflectivity(self):
        ai = np.full((3, 3, 8), 8000.0, dtype=np.float32)
        r = compute_reflectivity_vertical(ai)
        np.testing.assert_allclose(r, 0.0, atol=1e-10)

    def test_known_two_layer(self):
        # Two layers: AI=10000 above, AI=7000 below → R = (7000-10000)/(7000+10000) = -3000/17000
        ai = np.zeros((1, 1, 2), dtype=np.float32)
        ai[0, 0, 0] = 10000.0
        ai[0, 0, 1] = 7000.0
        r = compute_reflectivity_vertical(ai)
        expected = (7000.0 - 10000.0) / (7000.0 + 10000.0)
        np.testing.assert_allclose(r[0, 0, 0], expected, rtol=1e-5)

    def test_reflectivity_bounded(self):
        # R ∈ [-1, 1] for any positive AI
        rng = np.random.default_rng(42)
        ai = rng.uniform(1000, 15000, size=(5, 5, 20)).astype(np.float32)
        r = compute_reflectivity_vertical(ai)
        assert np.all(np.abs(r) <= 1.0 + 1e-6)

    def test_sign_convention(self):
        # Increasing impedance → positive reflectivity
        ai = np.zeros((1, 1, 2), dtype=np.float32)
        ai[0, 0, 0] = 5000.0
        ai[0, 0, 1] = 10000.0
        r = compute_reflectivity_vertical(ai)
        assert r[0, 0, 0] > 0


# ---------------------------------------------------------------------------
# compute_synthetic_seismic_3d
# ---------------------------------------------------------------------------
class TestComputeSyntheticSeismic3d:
    def test_output_shape_matches_input(self):
        refl = np.random.default_rng(0).standard_normal((4, 5, 30)).astype(np.float32)
        synth = compute_synthetic_seismic_3d(refl, f_dominant=25.0, dz=0.1)
        assert synth.shape == refl.shape

    def test_output_dtype_float32(self):
        refl = np.zeros((2, 2, 20), dtype=np.float32)
        synth = compute_synthetic_seismic_3d(refl, f_dominant=25.0, dz=0.1)
        assert synth.dtype == np.float32

    def test_zero_reflectivity_gives_zero_output(self):
        refl = np.zeros((3, 3, 15), dtype=np.float32)
        synth = compute_synthetic_seismic_3d(refl, f_dominant=25.0, dz=0.1)
        np.testing.assert_array_equal(synth, 0.0)

    def test_invalid_f_dominant(self):
        refl = np.zeros((2, 2, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="f_dominant must be positive"):
            compute_synthetic_seismic_3d(refl, f_dominant=0.0)
        with pytest.raises(ValueError, match="f_dominant must be positive"):
            compute_synthetic_seismic_3d(refl, f_dominant=-1.0)

    def test_invalid_dz(self):
        refl = np.zeros((2, 2, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="dz must be positive"):
            compute_synthetic_seismic_3d(refl, dz=0.0)

    def test_invalid_velocity(self):
        refl = np.zeros((2, 2, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="velocity must be positive"):
            compute_synthetic_seismic_3d(refl, velocity=-100.0)

    def test_invalid_wavelet_duration(self):
        refl = np.zeros((2, 2, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="wavelet_duration must be positive"):
            compute_synthetic_seismic_3d(refl, wavelet_duration=-1.0)

    def test_spike_produces_wavelet_shape(self):
        # Single spike → output should resemble the wavelet
        refl = np.zeros((1, 1, 51), dtype=np.float32)
        refl[0, 0, 25] = 1.0
        synth = compute_synthetic_seismic_3d(refl, f_dominant=25.0, dz=0.1)
        # Peak should be near the spike location
        peak_z = np.argmax(np.abs(synth[0, 0, :]))
        assert abs(peak_z - 25) <= 1

    def test_short_trace_wavelet_longer_than_trace(self):
        # Regression: wavelet longer than trace should not crash
        refl = np.zeros((1, 1, 3), dtype=np.float32)
        refl[0, 0, 1] = 1.0
        synth = compute_synthetic_seismic_3d(refl, f_dominant=25.0, dz=0.1)
        assert synth.shape == (1, 1, 3)


# ---------------------------------------------------------------------------
# compute_rms_cube
# ---------------------------------------------------------------------------
class TestComputeRmsCube:
    def test_output_shape(self):
        data = np.random.default_rng(0).standard_normal((4, 5, 10)).astype(np.float32)
        rms = compute_rms_cube(data, window_half=2)
        assert rms.shape == data.shape

    def test_output_dtype_float32(self):
        data = np.ones((2, 2, 5), dtype=np.float32)
        assert compute_rms_cube(data, window_half=1).dtype == np.float32

    def test_constant_input_returns_abs_value(self):
        val = 3.0
        data = np.full((2, 2, 10), val, dtype=np.float32)
        rms = compute_rms_cube(data, window_half=2)
        np.testing.assert_allclose(rms, val, atol=1e-5)

    def test_zero_window(self):
        # window_half=0 → each sample is its own RMS = |value|
        data = np.array([[[1.0, -2.0, 3.0]]], dtype=np.float32)
        rms = compute_rms_cube(data, window_half=0)
        np.testing.assert_allclose(rms, np.abs(data), atol=1e-5)

    def test_negative_window_raises(self):
        data = np.ones((2, 2, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="window_half must be >= 0"):
            compute_rms_cube(data, window_half=-1)

    def test_rms_nonnegative(self):
        rng = np.random.default_rng(42)
        data = rng.standard_normal((5, 5, 20)).astype(np.float32)
        rms = compute_rms_cube(data, window_half=3)
        assert np.all(rms >= 0)

    def test_known_manual_rms(self):
        # 1x1x5 data: [1, 2, 3, 4, 5], window_half=1
        # At z=2: window covers z=[1,2,3] → values [2, 3, 4] → RMS = sqrt((4+9+16)/3)
        data = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
        rms = compute_rms_cube(data, window_half=1)
        expected_z2 = np.sqrt((4.0 + 9.0 + 16.0) / 3.0)
        np.testing.assert_allclose(rms[0, 0, 2], expected_z2, rtol=1e-5)


# ---------------------------------------------------------------------------
# generate_rms_from_facies_3d
# ---------------------------------------------------------------------------
class TestGenerateRmsFromFacies3d:
    @pytest.fixture()
    def simple_facies(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 3, size=(8, 8, 12), dtype=np.int8)

    def test_output_shape(self, simple_facies):
        rms = generate_rms_from_facies_3d(simple_facies, rng=np.random.default_rng(0))
        # nz-1 because reflectivity reduces depth by 1
        assert rms.shape == (8, 8, 11)

    def test_output_dtype(self, simple_facies):
        rms = generate_rms_from_facies_3d(simple_facies, rng=np.random.default_rng(0))
        assert rms.dtype == np.float32

    def test_output_nonnegative_no_noise(self, simple_facies):
        rms = generate_rms_from_facies_3d(
            simple_facies, noise_level=0.0, rng=np.random.default_rng(0)
        )
        assert np.all(rms >= 0)

    def test_deterministic_mode(self, simple_facies):
        rms = generate_rms_from_facies_3d(
            simple_facies, ai_values=DEFAULT_AI, noise_level=0.0, smooth_sigma=0.0
        )
        # Deterministic → no rng dependency, should be reproducible
        rms2 = generate_rms_from_facies_3d(
            simple_facies, ai_values=DEFAULT_AI, noise_level=0.0, smooth_sigma=0.0
        )
        np.testing.assert_array_equal(rms, rms2)

    def test_stochastic_default(self, simple_facies):
        # Neither ai_values nor rock_properties → defaults to stochastic
        rms1 = generate_rms_from_facies_3d(
            simple_facies,
            noise_level=0.0,
            smooth_sigma=0.0,
            rng=np.random.default_rng(1),
        )
        rms2 = generate_rms_from_facies_3d(
            simple_facies,
            noise_level=0.0,
            smooth_sigma=0.0,
            rng=np.random.default_rng(2),
        )
        # Different seeds → different RMS cubes
        assert not np.array_equal(rms1, rms2)

    def test_stochastic_reproducible(self, simple_facies):
        rms1 = generate_rms_from_facies_3d(
            simple_facies, noise_level=0.0, rng=np.random.default_rng(42)
        )
        rms2 = generate_rms_from_facies_3d(
            simple_facies, noise_level=0.0, rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(rms1, rms2)

    def test_noise_adds_variation(self, simple_facies):
        rms_clean = generate_rms_from_facies_3d(
            simple_facies, ai_values=DEFAULT_AI, noise_level=0.0, smooth_sigma=0.0
        )
        rms_noisy = generate_rms_from_facies_3d(
            simple_facies,
            ai_values=DEFAULT_AI,
            noise_level=0.1,
            smooth_sigma=0.0,
            rng=np.random.default_rng(0),
        )
        assert not np.array_equal(rms_clean, rms_noisy)

    def test_smoothing_reduces_variance(self, simple_facies):
        rms_raw = generate_rms_from_facies_3d(
            simple_facies, ai_values=DEFAULT_AI, noise_level=0.0, smooth_sigma=0.0
        )
        rms_smooth = generate_rms_from_facies_3d(
            simple_facies, ai_values=DEFAULT_AI, noise_level=0.0, smooth_sigma=2.0
        )
        assert np.std(rms_smooth) < np.std(rms_raw)

    def test_invalid_facies_codes_raises(self):
        bad_facies = np.array([[[0, 1, 5]]], dtype=np.int8)
        with pytest.raises(ValueError, match="unexpected codes"):
            generate_rms_from_facies_3d(bad_facies)

    def test_uniform_facies_zero_reflectivity(self):
        # All same facies → zero reflectivity → zero synthetic → zero RMS
        facies = np.zeros((4, 4, 10), dtype=np.int8)
        rms = generate_rms_from_facies_3d(
            facies, ai_values=DEFAULT_AI, noise_level=0.0, smooth_sigma=0.0
        )
        np.testing.assert_allclose(rms, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# normalize_cube_to_range
# ---------------------------------------------------------------------------
class TestNormalizeCubeToRange:
    def test_default_range(self):
        arr = np.array([0.0, 5.0, 10.0])
        result = normalize_rms(arr)
        np.testing.assert_allclose(result, [-1.0, 0.0, 1.0])

    def test_custom_range(self):
        arr = np.array([0.0, 10.0])
        result = normalize_rms(arr, vmin=0.0, vmax=1.0)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_constant_array_returns_midpoint(self):
        arr = np.full((3, 3), 42.0)
        result = normalize_rms(arr, vmin=-1.0, vmax=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_preserves_shape(self):
        arr = np.random.default_rng(0).uniform(0, 10, (3, 4, 5))
        result = normalize_rms(arr)
        assert result.shape == (3, 4, 5)

    def test_output_within_range(self):
        arr = np.random.default_rng(0).uniform(-100, 100, (10, 10))
        result = normalize_rms(arr, vmin=-1.0, vmax=1.0)
        assert result.min() >= -1.0 - 1e-10
        assert result.max() <= 1.0 + 1e-10

    def test_min_maps_to_vmin_max_maps_to_vmax(self):
        arr = np.array([3.0, 7.0, 5.0])
        result = normalize_rms(arr, vmin=0.0, vmax=1.0)
        np.testing.assert_allclose(result[0], 0.0, atol=1e-10)  # min → vmin
        np.testing.assert_allclose(result[1], 1.0, atol=1e-10)  # max → vmax

    def test_near_constant_returns_midpoint(self):
        arr = np.full((3,), 5.0)
        arr[1] = 5.0 + 1e-12  # tiny difference, below threshold
        result = normalize_rms(arr, vmin=-1.0, vmax=1.0)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
