import numpy as np
import pytest

from fastpidc.network import (
    CLRNetworkInference,
    MINetworkInference,
    PIDCNetworkInference,
    PUCNetworkInference,
    build_sorted_edges,
    get_joint_probabilities,
    get_mi_and_si,
    get_mi_scores,
    infer_network_from_nodes,
)
from fastpidc.puc import compute_puc_full
from fastpidc.types import Node, PIDCConfig


def _make_nodes(n_nodes=6, n_samples=300, n_bins=3, seed=0) -> list[Node]:
    rng = np.random.default_rng(seed)
    nodes = []
    base = rng.integers(0, n_bins, size=n_samples)
    for i in range(n_nodes):
        # Node 0 correlates strongly with the shared `base` signal; the rest
        # are noisier mixtures of it, so there is real (non-independent)
        # structure to detect.
        noise = rng.integers(0, n_bins, size=n_samples)
        mix = np.where(rng.random(n_samples) < 0.7 - 0.1 * i, base, noise) % n_bins
        node = Node.from_raw_values(f"N{i}", mix.astype(np.float64), "uniform_width", "maximum_likelihood", n_bins)
        nodes.append(node)
    return nodes


def test_get_joint_probabilities_sums_to_one():
    nodes = _make_nodes()
    probs, p1, p2 = get_joint_probabilities(nodes[0], nodes[1], "maximum_likelihood")
    assert probs.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(p1.sum(), 1.0)
    np.testing.assert_allclose(p2.sum(), 1.0)


def test_mi_symmetric_and_nonnegative():
    nodes = _make_nodes()
    mi_ab, _, _ = get_mi_and_si(nodes[0], nodes[1], "maximum_likelihood", 2)
    mi_ba, _, _ = get_mi_and_si(nodes[1], nodes[0], "maximum_likelihood", 2)
    assert mi_ab == pytest.approx(mi_ba)
    assert mi_ab >= -1e-12


def test_get_mi_scores_matrix_is_symmetric_with_zero_diagonal():
    nodes = _make_nodes()
    scores = get_mi_scores(nodes, "maximum_likelihood", 2)
    np.testing.assert_allclose(scores, scores.T)
    np.testing.assert_array_equal(np.diag(scores), np.zeros(len(nodes)))


def test_build_sorted_edges_descending():
    nodes = _make_nodes(n_nodes=4)
    weights = np.array(
        [
            [0, 1, 5, 2],
            [1, 0, 3, 4],
            [5, 3, 0, 6],
            [2, 4, 6, 0],
        ],
        dtype=float,
    )
    edges = build_sorted_edges(nodes, weights)
    assert len(edges) == 6
    assert [e.weight for e in edges] == sorted((e.weight for e in edges), reverse=True)


@pytest.mark.parametrize(
    "inference_cls", [MINetworkInference, CLRNetworkInference, PUCNetworkInference, PIDCNetworkInference]
)
def test_infer_network_from_nodes_produces_full_edge_set(inference_cls):
    nodes = _make_nodes(n_nodes=6)
    network = infer_network_from_nodes(inference_cls(), nodes, config=PIDCConfig(backend="cpu"))
    assert len(network.edges) == len(nodes) * (len(nodes) - 1) // 2
    assert all(np.isfinite(e.weight) for e in network.edges)
    weights = [e.weight for e in network.edges]
    assert weights == sorted(weights, reverse=True)


def test_puc_scores_are_symmetric_and_nonnegative():
    nodes = _make_nodes(n_nodes=6)
    mi_scores, puc_scores = compute_puc_full(nodes, config=PIDCConfig(backend="cpu"))
    np.testing.assert_allclose(mi_scores, mi_scores.T)
    np.testing.assert_allclose(puc_scores, puc_scores.T)
    assert np.all(puc_scores >= -1e-9)
    np.testing.assert_array_equal(np.diag(puc_scores), np.zeros(len(nodes)))
