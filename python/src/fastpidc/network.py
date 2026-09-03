"""Network inference algorithms and the :class:`InferredNetwork` builder.

MI, CLR, PUC and PIDC are explained in Chan, Stumpf & Babtie (2017)
(http://biorxiv.org/content/early/2017/04/26/082099), along with terms such
as specific information, proportional unique contribution, context, etc.

Ported from FastPIDC.jl's ``src/network_inference.jl`` and ``src/infer_network.jl``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .information import (
    apply_mutual_information_formula,
    apply_specific_information_formula,
    get_joint_frequencies_from_bin_ids,
    get_probabilities,
)
from .types import Edge, InferredNetwork, Node, PIDCConfig

__all__ = [
    "AbstractNetworkInference",
    "CLRNetworkInference",
    "MINetworkInference",
    "PIDCNetworkInference",
    "PUCNetworkInference",
    "build_sorted_edges",
    "get_joint_probabilities",
    "get_mi_and_si",
    "get_mi_scores",
    "get_weights",
    "infer_network_from_nodes",
]


@dataclass(frozen=True, slots=True)
class AbstractNetworkInference:
    """Base class for network inference algorithm selectors.

    Concrete subclasses select behavior via the :attr:`apply_context` and
    :attr:`get_puc` class attributes.
    """

    apply_context: bool = False
    get_puc: bool = False


class MINetworkInference(AbstractNetworkInference):
    """Mutual information (MI): raw pairwise MI as edge weights."""

    def __init__(self) -> None:
        super().__init__(apply_context=False, get_puc=False)


class CLRNetworkInference(AbstractNetworkInference):
    """Context Likelihood of Relatedness (CLR): MI weights with per-node
    background context applied."""

    def __init__(self) -> None:
        super().__init__(apply_context=True, get_puc=False)


class PUCNetworkInference(AbstractNetworkInference):
    """Proportional Unique Contribution (PUC): redundancy-corrected MI,
    without context."""

    def __init__(self) -> None:
        super().__init__(apply_context=False, get_puc=True)


class PIDCNetworkInference(AbstractNetworkInference):
    """Partial Information Decomposition and Context (PIDC): PUC scores with
    per-node background context applied."""

    def __init__(self) -> None:
        super().__init__(apply_context=True, get_puc=True)


def get_joint_probabilities(node1: Node, node2: Node, estimator: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the joint probability distribution for ``node1`` and
    ``node2`` (a matrix over their bin ids) using ``estimator``, along with
    the marginal distributions recovered by summing over the other node's
    bins."""
    frequencies = get_joint_frequencies_from_bin_ids(
        node1.binned_values, node2.binned_values, node1.number_of_bins, node2.number_of_bins
    )
    probabilities = get_probabilities(estimator, frequencies)
    # `probabilities` is already a property of each Node, but recomputing it
    # gets the correct array shapes and, for MI/CLR, means we don't assume
    # the marginal for a node is always the same no matter the second node
    # (allowing estimators other than maximum_likelihood). We still can't do
    # this for PUC/PIDC, since 3-node joint distributions in `puc.py` do
    # make that assumption.
    probabilities1 = probabilities.sum(axis=1, keepdims=True)
    probabilities2 = probabilities.sum(axis=0, keepdims=True)
    return probabilities, probabilities1, probabilities2


def get_mi_and_si(node1: Node, node2: Node, estimator: str, base: float) -> tuple[float, np.ndarray, np.ndarray]:
    """Mutual information between ``node1`` and ``node2``, plus the specific
    information of each node with respect to the other."""
    probabilities, p1, p2 = get_joint_probabilities(node1, node2, estimator)
    mi = apply_mutual_information_formula(probabilities, p1, p2, base)
    # probabilities has shape (node1.number_of_bins, node2.number_of_bins);
    # axis 0 is node1's (the source, for si1) axis, axis 1 is node2's.
    si1 = apply_specific_information_formula(probabilities, p1, p2, 0, base)
    si2 = apply_specific_information_formula(probabilities, p2, p1, 1, base)
    return mi, si1, si2


def get_mi_scores(nodes: list[Node], estimator: str, base: float) -> np.ndarray:
    """Pairwise mutual information between all ``nodes``, as a symmetric
    ``len(nodes)`` by ``len(nodes)`` matrix (the diagonal is left as zero)."""
    n = len(nodes)
    scores = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mi, _, _ = get_mi_and_si(nodes[i], nodes[j], estimator, base)
            scores[i, j] = mi
            scores[j, i] = mi
    return scores


def _fit_gamma(scores_i: np.ndarray) -> tuple[float, float] | None:
    """MLE-fit a zero-located Gamma distribution to ``scores_i``, returning
    ``(shape, scale)``, or ``None`` if the fit is not viable (mirroring the
    Julia implementation's try/catch fallback to CLR-style z-scores)."""
    if scores_i.size < 2 or np.any(scores_i <= 0) or np.var(scores_i) == 0:
        return None
    try:
        shape, _loc, scale = stats.gamma.fit(scores_i, floc=0)
    except Exception:
        return None
    if not (np.isfinite(shape) and np.isfinite(scale) and shape > 0 and scale > 0):
        return None
    return float(shape), float(scale)


def get_weights(
    inference: PIDCNetworkInference | CLRNetworkInference,
    scores: np.ndarray,
    nodes: list[Node],
) -> np.ndarray:
    """Apply background-context weighting to raw pairwise ``scores`` (MI for
    :class:`CLRNetworkInference`, PUC for :class:`PIDCNetworkInference`).

    For each node, a background distribution of its scores against all
    other nodes is used to standardize its scores: for
    :class:`PIDCNetworkInference`, a Gamma distribution is fit to the
    background (falling back to a CLR-style z-score if the fit fails for
    either node in a pair); for :class:`CLRNetworkInference`, a z-score
    against the background mean/variance is always used. See Chan, Stumpf &
    Babtie (2017) for details.
    """
    n = len(nodes)
    is_pidc = isinstance(inference, PIDCNetworkInference)

    use_gamma = np.zeros(n, dtype=bool)
    gamma_shape = np.zeros(n, dtype=np.float64)
    gamma_scale = np.zeros(n, dtype=np.float64)
    clr_mean = np.zeros(n, dtype=np.float64)
    clr_var = np.zeros(n, dtype=np.float64)

    for i in range(n):
        scores_i = np.delete(scores[:, i], i)
        clr_mean[i] = scores_i.mean()
        clr_var[i] = scores_i.var(ddof=1)

        if is_pidc:
            fit = _fit_gamma(scores_i)
            if fit is not None:
                gamma_shape[i], gamma_scale[i] = fit
                use_gamma[i] = True

    weights = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            score = scores[i, j]
            if is_pidc and use_gamma[i] and use_gamma[j]:
                w = stats.gamma.cdf(score, gamma_shape[i], scale=gamma_scale[i]) + stats.gamma.cdf(
                    score, gamma_shape[j], scale=gamma_scale[j]
                )
            else:
                diff_i = score - clr_mean[i]
                diff_j = score - clr_mean[j]
                term_i = 0.0 if (clr_var[i] == 0 or diff_i < 0) else diff_i**2 / clr_var[i]
                term_j = 0.0 if (clr_var[j] == 0 or diff_j < 0) else diff_j**2 / clr_var[j]
                w = np.sqrt(term_i + term_j)
            weights[i, j] = w
            weights[j, i] = w
    return weights


def build_sorted_edges(nodes: list[Node], weights: np.ndarray) -> list[Edge]:
    """Build an :class:`Edge` for every pair of ``nodes``, weighted by the
    corresponding entry of ``weights``, sorted in descending order of
    weight."""
    n = len(nodes)
    edges = [Edge((nodes[i], nodes[j]), float(weights[i, j])) for i in range(n) for j in range(i + 1, n)]
    edges.sort(key=lambda e: e.weight, reverse=True)
    return edges


def infer_network_from_nodes(
    inference: AbstractNetworkInference,
    nodes: list[Node],
    *,
    estimator: str = "maximum_likelihood",
    base: float = 2,
    config: PIDCConfig | None = None,
) -> InferredNetwork:
    """Construct an :class:`InferredNetwork` given a network inference
    algorithm and a list of :class:`Node`, computing pairwise scores (MI or
    PUC, depending on ``inference``), optionally applying context weighting,
    and sorting the resulting edges by descending weight.

    ``estimator="maximum_likelihood"`` is recommended for PUC and PIDC: the
    package's speedups assume the marginal probability distribution for a
    node, in the joint distribution with any two other nodes, is always the
    same. Using other estimators violates that assumption for PUC/PIDC (in
    :func:`get_joint_probabilities` and the PUC computation).
    """
    if config is None:
        config = PIDCConfig()

    if inference.get_puc:
        from . import dump
        from .puc import compute_puc_full

        mi_scores, scores = compute_puc_full(nodes, estimator=estimator, base=base, config=config)

        if isinstance(inference, PIDCNetworkInference) and config.dump_mi_path is not None:
            if config.verbose:
                print("[fastpidc] Writing MI scores.")
            dump.dump_mi_scores(mi_scores, nodes, config)

        if config.dump_puc_path is not None:
            if config.verbose:
                print("[fastpidc] Writing pre-context PUC scores.")
            dump.dump_puc_scores(scores, nodes, config)

        if inference.apply_context:
            if config.verbose:
                print("[fastpidc] Context weighting.")
            weights = get_weights(inference, scores, nodes)
        else:
            weights = scores
    else:
        scores = get_mi_scores(nodes, estimator, base)
        weights = get_weights(inference, scores, nodes) if inference.apply_context else scores

    return InferredNetwork(nodes, build_sorted_edges(nodes, weights))
