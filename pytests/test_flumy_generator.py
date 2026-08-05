"""Unit tests for diffsim.data.flumy_generator module."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffsim.data.flumy_generator import (
    FACIES_OVERBANK,
    FACIES_CHANNEL,
    FACIES_LEVEE,
    FACIES_CRAVASSE,
    FACIES_COAL,
    FACIES_OTHERS,
    FACIES_NAMES,
    FACIES_NORMALIZED,
    FlumyGenerator,
    denormalize_facies,
    normalize_facies,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
class TestConstants:
    def test_facies_codes(self):
        assert FACIES_OVERBANK == 0
        assert FACIES_CHANNEL == 1
        assert FACIES_LEVEE == 2
        assert FACIES_CRAVASSE == 3
        assert FACIES_COAL == 4
        assert FACIES_OTHERS == 5

    def test_facies_names_keys(self):
        assert set(FACIES_NAMES.keys()) == {0, 1, 2, 3, 4, 5}

    def test_facies_normalized_values(self):
        assert FACIES_NORMALIZED[FACIES_OVERBANK] == -1.0
        assert FACIES_NORMALIZED[FACIES_CHANNEL] == -0.6
        assert FACIES_NORMALIZED[FACIES_LEVEE] == -0.2
        assert FACIES_NORMALIZED[FACIES_CRAVASSE] == 0.2
        assert FACIES_NORMALIZED[FACIES_COAL] == 0.6
        assert FACIES_NORMALIZED[FACIES_OTHERS] == 1.0


# ---------------------------------------------------------------------------
# FlumyGenerator.__init__
# ---------------------------------------------------------------------------
class TestFlumyGeneratorInit:
    def test_default_params(self):
        gen = FlumyGenerator()
        assert gen.nx == 256
        assert gen.ny == 256
        assert gen.mesh == 20
        assert gen.hmax == 3.0
        assert gen.ng == 50
        assert gen.isbx == 80
        assert gen.zul == 9.0  # 3 * hmax
        assert gen.dz == 1
        assert gen.nz == 9
        assert gen.verbose is False

    def test_custom_params(self):
        gen = FlumyGenerator(
            nx=64, ny=128, mesh=10, hmax=5.0, ng=40, isbx=60, zul=20, dz=2, verbose=True
        )
        assert gen.nx == 64
        assert gen.ny == 128
        assert gen.mesh == 10
        assert gen.hmax == 5.0
        assert gen.ng == 40
        assert gen.isbx == 60
        assert gen.zul == 20
        assert gen.dz == 2
        assert gen.nz == 10  # 20 / 2
        assert gen.verbose is True

    def test_zul_default_from_hmax(self):
        gen = FlumyGenerator(hmax=7.0)
        assert gen.zul == 21.0

    def test_nz_computation(self):
        gen = FlumyGenerator(hmax=6.0, dz=0.5)
        # zul = 3*6 = 18, nz = 18/0.5 = 36
        assert gen.nz == 36


# ---------------------------------------------------------------------------
# FlumyGenerator.generate (requires flumy package)
# ---------------------------------------------------------------------------
class TestFlumyGeneratorGenerate:
    @pytest.fixture()
    def small_generator(self):
        return FlumyGenerator(nx=24, ny=24, mesh=10, hmax=3.0, ng=50, isbx=80)

    def test_generate_returns_3d_int8(self, small_generator):
        block = small_generator.generate(seed=42)
        assert block.ndim == 3
        assert block.dtype == np.int8

    def test_generate_shape(self, small_generator):
        block = small_generator.generate(seed=42)
        assert block.shape[0] == small_generator.nx
        assert block.shape[1] == small_generator.ny
        assert block.shape[2] == small_generator.nz

    def test_generate_reproducible(self, small_generator):
        b1 = small_generator.generate(seed=123)
        b2 = small_generator.generate(seed=123)
        np.testing.assert_array_equal(b1, b2)

    def test_generate_different_seeds_differ(self, small_generator):
        b1 = small_generator.generate(seed=1)
        b2 = small_generator.generate(seed=2)
        assert not np.array_equal(b1, b2)


# ---------------------------------------------------------------------------
# reclassify_facies
# ---------------------------------------------------------------------------
class TestReclassify:
    def test_undefined_and_overbank_map_to_overbank(self):
        raw = np.array([0, 8, 9, 10, 11, 12, 13], dtype=np.int8)
        result = FlumyGenerator.reclassify_facies(raw)
        np.testing.assert_array_equal(result, [FACIES_OVERBANK] * 7)

    def test_channel_codes(self):
        for code in [1, 2, 3]:
            raw = np.array([code], dtype=np.int8)
            result = FlumyGenerator.reclassify_facies(raw)
            assert result[0] == FACIES_CHANNEL, f"Code {code} should be CHANNEL"

    def test_cravasse_codes(self):
        for code in [4, 5, 6]:
            raw = np.array([code], dtype=np.int8)
            result = FlumyGenerator.reclassify_facies(raw)
            assert result[0] == FACIES_CRAVASSE, f"Code {code} should be CRAVASSE"

    def test_levee_code(self):
        raw = np.array([7], dtype=np.int8)
        result = FlumyGenerator.reclassify_facies(raw)
        assert result[0] == FACIES_LEVEE

    def test_mixed_input(self):
        raw = np.array([[0, 1, 4], [7, 6, 9]], dtype=np.int8)
        expected = np.array(
            [
                [FACIES_OVERBANK, FACIES_CHANNEL, FACIES_CRAVASSE],
                [FACIES_LEVEE, FACIES_CRAVASSE, FACIES_OVERBANK],
            ],
            dtype=np.int8,
        )
        result = FlumyGenerator.reclassify_facies(raw)
        np.testing.assert_array_equal(result, expected)

    def test_output_dtype_is_int8(self):
        raw = np.zeros((4, 4), dtype=np.int32)
        result = FlumyGenerator.reclassify_facies(raw)
        assert result.dtype == np.int8

    def test_preserves_shape(self):
        raw = np.zeros((3, 5, 7), dtype=np.int8)
        result = FlumyGenerator.reclassify_facies(raw)
        assert result.shape == (3, 5, 7)

    def test_deprecated_alias_still_works(self):
        raw = np.array([0, 1, 7], dtype=np.int8)
        assert np.array_equal(
            FlumyGenerator.reclassify_to_three_facies(raw),
            FlumyGenerator.reclassify_facies(raw),
        )


# ---------------------------------------------------------------------------
# normalize_facies
# ---------------------------------------------------------------------------
class TestNormalizeFacies:
    def test_maps_codes_correctly(self):
        facies = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int8)
        result = normalize_facies(facies)
        expected = np.array([[-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]], dtype=np.float32)
        np.testing.assert_allclose(result, expected)

    def test_output_dtype(self):
        facies = np.array([0, 1, 2], dtype=np.int8)
        assert normalize_facies(facies).dtype == np.float32

    def test_all_overbank(self):
        facies = np.zeros((3, 3), dtype=np.int8)
        result = normalize_facies(facies)
        np.testing.assert_array_equal(result, np.full((3, 3), -1.0))

    def test_all_others(self):
        facies = np.full((2, 2), FACIES_OTHERS, dtype=np.int8)
        result = normalize_facies(facies)
        np.testing.assert_array_equal(result, np.ones((2, 2)))

    def test_preserves_shape(self):
        facies = np.zeros((5, 7), dtype=np.int8)
        assert normalize_facies(facies).shape == (5, 7)


# ---------------------------------------------------------------------------
# denormalize_facies
# ---------------------------------------------------------------------------
class TestDenormalizeFacies:
    def test_exact_values_roundtrip(self):
        facies = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.int8)
        normalized = normalize_facies(facies)
        recovered = denormalize_facies(normalized)
        np.testing.assert_array_equal(recovered, facies)

    def test_output_dtype(self):
        normalized = np.array([-1.0, -0.6, -0.2, 0.2, 0.6, 1.0], dtype=np.float32)
        assert denormalize_facies(normalized).dtype == np.int8

    def test_noisy_values_snap_to_nearest(self):
        # Values clearly in each facies region (not at midpoints)
        normalized = np.array([[-0.9, -0.55, 0.15]], dtype=np.float32)
        result = denormalize_facies(normalized)
        # -0.9 → OVERBANK(-1.0), -0.55 → CHANNEL(-0.6), 0.15 → CRAVASSE(0.2)
        expected = np.array([[FACIES_OVERBANK, FACIES_CHANNEL, FACIES_CRAVASSE]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_extreme_values_clamp(self):
        normalized = np.array([[-5.0, 5.0]], dtype=np.float32)
        result = denormalize_facies(normalized)
        expected = np.array([[FACIES_OVERBANK, FACIES_OTHERS]], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_preserves_shape(self):
        normalized = np.zeros((3, 4), dtype=np.float32)
        assert denormalize_facies(normalized).shape == (3, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
