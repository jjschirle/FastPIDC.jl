import numpy as np
import pytest

from fastpidc.information import (
    apply_mutual_information_formula,
    apply_redundancy_formula,
    apply_specific_information_formula,
    get_frequencies_from_bin_ids,
    get_joint_frequencies_from_bin_ids,
    get_lambda,
    get_probabilities,
)


def test_get_frequencies_from_bin_ids():
    freqs = get_frequencies_from_bin_ids(np.array([0, 0, 1, 2, 2, 2]), 3)
    np.testing.assert_array_equal(freqs, [2, 1, 3])


def test_get_joint_frequencies_from_bin_ids():
    x = np.array([0, 0, 1, 1])
    y = np.array([0, 1, 0, 0])
    freqs = get_joint_frequencies_from_bin_ids(x, y, 2, 2)
    np.testing.assert_array_equal(freqs, [[1, 1], [2, 0]])


def test_get_probabilities_maximum_likelihood():
    probs = get_probabilities("maximum_likelihood", np.array([1.0, 3.0]))
    np.testing.assert_allclose(probs, [0.25, 0.75])


def test_get_probabilities_dirichlet():
    probs = get_probabilities("dirichlet", np.array([0.0, 0.0]), prior=1.0)
    np.testing.assert_allclose(probs, [0.5, 0.5])


def test_get_probabilities_shrinkage_towards_uniform():
    # A perfectly uniform sample should need no shrinkage.
    probs = get_probabilities("shrinkage", np.array([5.0, 5.0]))
    np.testing.assert_allclose(probs, [0.5, 0.5])


def test_get_probabilities_unknown_estimator_raises():
    with pytest.raises(ValueError):
        get_probabilities("nonsense", np.array([1.0, 1.0]))


def test_get_lambda_bounds():
    assert get_lambda(np.array([0.5, 0.5]), 0.5, 0) == 1.0
    assert get_lambda(np.array([0.5, 0.5]), 0.5, 1) == 1.0
    lam = get_lambda(np.array([0.5, 0.5]), 0.5, 100)
    assert 0.0 <= lam <= 1.0


def test_mutual_information_zero_for_independent_variables():
    # p(x,y) = p(x) p(y): MI should be (numerically) zero.
    p_x = np.array([[0.5], [0.5]])
    p_y = np.array([[0.5, 0.5]])
    p_xy = p_x * p_y
    mi = apply_mutual_information_formula(p_xy, p_x, p_y, 2)
    assert mi == pytest.approx(0.0, abs=1e-12)


def test_mutual_information_positive_for_dependent_variables():
    p_xy = np.array([[0.5, 0.0], [0.0, 0.5]])
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    mi = apply_mutual_information_formula(p_xy, p_x, p_y, 2)
    assert mi == pytest.approx(1.0)  # one bit of information


def test_specific_information_matches_manual_expectation():
    # Perfectly correlated binary variables: knowing x fully determines z.
    p_xz = np.array([[0.5, 0.0], [0.0, 0.5]])
    p_x = p_xz.sum(axis=1, keepdims=True)
    p_z = p_xz.sum(axis=0, keepdims=True)
    si = apply_specific_information_formula(p_xz, p_x, p_z, 0, 2)
    np.testing.assert_allclose(si, [1.0, 1.0])


def test_redundancy_is_bounded_by_min_specific_information():
    p_z = np.array([0.5, 0.5])
    si1 = np.array([0.2, 0.8])
    si2 = np.array([0.5, 0.5])
    redundancy = apply_redundancy_formula(p_z, si1, si2, 2)
    assert redundancy == pytest.approx(0.5 * 0.2 + 0.5 * 0.5)
