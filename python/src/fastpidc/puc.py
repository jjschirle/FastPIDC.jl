"""Proportional Unique Contribution (PUC) score computation.

Ported from FastPIDC.jl's ``src/puc_full.jl``. Computes, for every ordered
triple ``(x, y, z)`` of distinct nodes, the redundancy between sources ``x``
and ``y`` with respect to target ``z`` from cached specific-information
values, and accumulates::

    PUC(x, z) += max(0, (MI(x, z) - redundancy) / MI(x, z))

symmetrically (skipped when ``MI(x, z)`` is ~0).

The CPU backend mirrors the reference (non-distributed) Julia
implementation directly: it caches pairwise MI and specific-information
values for all node pairs, then vectorizes the inner "for each other node
y" loop with NumPy. It is intended for small-to-moderate networks; for
large gene sets, use ``config.backend = "cuda"`` (see :mod:`fastpidc.cuda`),
which mirrors the chunked GPU kernel used by FastPIDC.jl.
"""

from __future__ import annotations

import numpy as np

from .network import get_mi_and_si
from .types import Node, PIDCConfig

__all__ = ["compute_puc_full"]

_MI_EPSILON = 1e-12


def _increment(x: int, z: int, mi: float, redundancy: float, puc: np.ndarray) -> None:
    if mi <= _MI_EPSILON:
        return
    score = (mi - redundancy) / mi
    score = score if (np.isfinite(score) and score >= 0) else 0.0
    puc[x, z] += score
    puc[z, x] += score


def _compute_puc_full_cpu(nodes: list[Node], estimator: str, base: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(nodes)
    mi = np.zeros((n, n), dtype=np.float64)
    # si_by_target[z] is an (n, nodes[z].number_of_bins) array; row x holds
    # the specific information of source x with respect to target z.
    si_by_target = [np.zeros((n, node.number_of_bins), dtype=np.float64) for node in nodes]

    for x in range(n):
        for z in range(x + 1, n):
            mi_val, si1, si2 = get_mi_and_si(nodes[x], nodes[z], estimator, base)
            mi[x, z] = mi[z, x] = mi_val
            si_by_target[z][x, :] = si1
            si_by_target[x][z, :] = si2

    puc = np.zeros((n, n), dtype=np.float64)
    for x in range(n):
        for z in range(x + 1, n):
            mi_xz = mi[x, z]
            if mi_xz <= _MI_EPSILON:
                continue

            # Sources x & y, target z.
            si_target_z = si_by_target[z]
            redundancy_z = np.minimum(si_target_z[x, :], si_target_z) @ nodes[z].probabilities
            scores_z = (mi_xz - redundancy_z) / mi_xz
            scores_z = np.where(np.isfinite(scores_z) & (scores_z >= 0), scores_z, 0.0)
            scores_z[x] = scores_z[z] = 0.0

            # Sources y & z, target x.
            si_target_x = si_by_target[x]
            redundancy_x = np.minimum(si_target_x[z, :], si_target_x) @ nodes[x].probabilities
            scores_x = (mi_xz - redundancy_x) / mi_xz
            scores_x = np.where(np.isfinite(scores_x) & (scores_x >= 0), scores_x, 0.0)
            scores_x[x] = scores_x[z] = 0.0

            total = float(scores_z.sum() + scores_x.sum())
            puc[x, z] = puc[z, x] = total

    return mi, puc


def compute_puc_full(
    nodes: list[Node],
    *,
    estimator: str = "maximum_likelihood",
    base: float = 2,
    config: PIDCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the full pairwise MI matrix and pre-context PUC matrix for
    ``nodes``.

    Returns
    -------
    (mi_scores, puc_scores) : each a dense ``len(nodes)``-by-``len(nodes)``
        matrix.
    """
    if config is None:
        config = PIDCConfig()

    if config.backend == "cuda":
        from .cuda import compute_puc_full_cuda

        if config.verbose:
            print(f"[fastpidc] Computing PUC scores. Backend: {config.backend}")
        return compute_puc_full_cuda(nodes, base=base, verbose=config.verbose)

    if config.verbose:
        print(f"[fastpidc] Computing PUC scores. Backend: {config.backend}")
    return _compute_puc_full_cpu(nodes, estimator, base)
