import numpy as np
import pytest

from fastpidc.pid import combined_node, pid_triple
from fastpidc.types import Node


def _node(label: str, bin_ids: np.ndarray, number_of_bins: int, estimator="maximum_likelihood") -> Node:
    from fastpidc.information import get_frequencies_from_bin_ids, get_probabilities

    freq = get_frequencies_from_bin_ids(bin_ids.astype(np.int64), number_of_bins)
    probs = get_probabilities(estimator, freq)
    return Node(label, bin_ids.astype(np.int64), number_of_bins, probs)


def test_combined_node_bin_ids_encode_both_sources():
    n1 = _node("A", np.array([0, 0, 1, 1]), 2)
    n2 = _node("B", np.array([0, 1, 0, 1]), 2)
    joint = combined_node(n1, n2, "maximum_likelihood")
    assert joint.number_of_bins == 4
    np.testing.assert_array_equal(joint.binned_values, [0, 1, 2, 3])
    assert joint.probabilities.sum() == pytest.approx(1.0)


def test_pid_triple_decomposition_sums_to_joint_mi():
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    z = x ^ y  # target: pure synergy between x and y

    nx = _node("X", x, 2)
    ny = _node("Y", y, 2)
    nz = _node("Z", z, 2)

    result = pid_triple(nx, ny, nz)
    assert result.mi_joint == pytest.approx(
        result.redundancy + result.unique1 + result.unique2 + result.synergy, abs=1e-9
    )


def test_pid_triple_xor_is_pure_synergy():
    # Classic XOR gate: X, Y independent fair bits, Z = X XOR Y. Individually
    # X and Y carry zero information about Z; jointly they determine it
    # exactly (MI_joint = 1 bit). Redundancy and both uniques should be ~0,
    # synergy should be ~1 bit.
    rng = np.random.default_rng(1)
    n = 20000
    x = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    z = x ^ y

    nx = _node("X", x, 2)
    ny = _node("Y", y, 2)
    nz = _node("Z", z, 2)

    result = pid_triple(nx, ny, nz)
    assert result.mi1 == pytest.approx(0.0, abs=1e-2)
    assert result.mi2 == pytest.approx(0.0, abs=1e-2)
    assert result.redundancy == pytest.approx(0.0, abs=1e-2)
    assert result.unique1 == pytest.approx(0.0, abs=1e-2)
    assert result.unique2 == pytest.approx(0.0, abs=1e-2)
    assert result.synergy == pytest.approx(1.0, abs=1e-2)


def test_pid_triple_identical_sources_is_pure_redundancy():
    # X == Y (identical copies), Z = X. All information both sources carry
    # about Z is fully redundant: unique1 == unique2 == synergy ~= 0, and
    # redundancy == mi1 == mi2 == MI(X, Z) == 1 bit for a fair bit.
    rng = np.random.default_rng(2)
    n = 4000
    x = rng.integers(0, 2, size=n)
    z = x.copy()

    nx = _node("X", x, 2)
    ny = _node("Y", x.copy(), 2)  # identical to nx
    nz = _node("Z", z, 2)

    result = pid_triple(nx, ny, nz)
    assert result.unique1 == pytest.approx(0.0, abs=1e-9)
    assert result.unique2 == pytest.approx(0.0, abs=1e-9)
    assert result.synergy == pytest.approx(0.0, abs=1e-9)
    assert result.redundancy == pytest.approx(result.mi1, abs=1e-9)
    assert result.redundancy == pytest.approx(1.0, abs=0.05)


def test_pid_triple_independent_of_everything_is_all_zero():
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.integers(0, 2, size=n)
    y = rng.integers(0, 2, size=n)
    z = rng.integers(0, 2, size=n)  # independent of both x and y

    nx = _node("X", x, 2)
    ny = _node("Y", y, 2)
    nz = _node("Z", z, 2)

    result = pid_triple(nx, ny, nz)
    for value in (result.redundancy, result.unique1, result.unique2, result.synergy, result.mi_joint):
        assert value == pytest.approx(0.0, abs=0.05)
