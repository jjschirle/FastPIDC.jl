import numpy as np
import pytest

from fastpidc.discretizers import (
    LinearDiscretizer,
    binedges_bayesian_blocks,
    binedges_uniform_count,
    binedges_uniform_width,
    get_bin_ids,
)


def test_linear_discretizer_basic():
    ld = LinearDiscretizer(np.array([0.0, 1.0, 2.0, 3.0]))
    assert ld.nbins == 3
    np.testing.assert_array_equal(ld.encode(np.array([0.0, 0.5, 1.0, 2.9, 3.0])), [0, 0, 1, 2, 2])


def test_linear_discretizer_clamps_outliers():
    ld = LinearDiscretizer(np.array([0.0, 1.0, 2.0]))
    np.testing.assert_array_equal(ld.encode(np.array([-5.0, 5.0])), [0, 1])


def test_linear_discretizer_rejects_unsorted_edges():
    with pytest.raises(ValueError):
        LinearDiscretizer(np.array([0.0, 2.0, 1.0]))


def test_linear_discretizer_rejects_too_few_edges():
    with pytest.raises(ValueError):
        LinearDiscretizer(np.array([0.0]))


def test_binedges_uniform_width():
    edges = binedges_uniform_width(np.array([0.0, 10.0, 5.0]), 5)
    np.testing.assert_allclose(edges, [0, 2, 4, 6, 8, 10])


def test_binedges_uniform_count_equal_bins():
    data = np.arange(10, dtype=np.float64)
    edges = binedges_uniform_count(data, 5)
    ld = LinearDiscretizer(edges)
    ids = ld.encode(data)
    counts = np.bincount(ids, minlength=5)
    np.testing.assert_array_equal(counts, [2, 2, 2, 2, 2])


def test_binedges_uniform_count_too_many_bins_raises():
    with pytest.raises(ValueError):
        binedges_uniform_count(np.array([1.0, 2.0]), 5)


def test_binedges_bayesian_blocks_constant_block_for_uniform_data():
    # Evenly-spaced data with no interesting structure should collapse to
    # relatively few, evenly distributed blocks.
    rng = np.random.default_rng(0)
    data = rng.normal(size=500)
    edges = binedges_bayesian_blocks(data)
    assert edges[0] == data.min()
    assert edges[-1] == data.max()
    assert np.all(np.diff(edges) > 0)


def test_get_bin_ids_constant_values_single_bin():
    ids, nbins = get_bin_ids(np.array([5.0, 5.0, 5.0]), "bayesian_blocks", 10)
    assert nbins == 1
    np.testing.assert_array_equal(ids, [0, 0, 0])


def test_get_bin_ids_binarize():
    ids, nbins = get_bin_ids(np.array([0.0, 1.0, 0.0, 2.5]), "binarize", 10)
    assert nbins == 2
    np.testing.assert_array_equal(ids, [0, 1, 0, 1])


def test_get_bin_ids_unknown_mode_warns_and_falls_back_to_uniform_width():
    with pytest.warns(RuntimeWarning, match="doesn't exist"):
        ids_fallback, nbins_fallback = get_bin_ids(np.arange(10.0), "nonsense", 5)
    ids_uw, nbins_uw = get_bin_ids(np.arange(10.0), "uniform_width", 5)
    assert nbins_fallback == nbins_uw
    np.testing.assert_array_equal(ids_fallback, ids_uw)


def test_get_bin_ids_uniform_count_warns_and_falls_back_to_uniform_width():
    # Heavily repeated values make the equal-count edges non-unique.
    values = np.array([0.0] * 9 + [1.0])
    with pytest.warns(RuntimeWarning, match="Uniform count failed"):
        ids_fallback, nbins_fallback = get_bin_ids(values, "uniform_count", 5)
    ids_uw, nbins_uw = get_bin_ids(values, "uniform_width", 5)
    assert nbins_fallback == nbins_uw
    np.testing.assert_array_equal(ids_fallback, ids_uw)


def test_get_bin_ids_bin_range():
    rng = np.random.default_rng(1)
    values = rng.normal(size=200)
    for mode in ("uniform_width", "uniform_count", "bayesian_blocks"):
        ids, nbins = get_bin_ids(values, mode, 8)
        assert ids.min() >= 0
        assert ids.max() < nbins
