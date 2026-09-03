"""Binary diagnostic score dumps, shared by MI and PUC.

Ported from FastPIDC.jl's ``src/mi_dump.jl`` and ``src/puc_dump.jl``.
"""

from __future__ import annotations

import numpy as np

from .types import Node, PIDCConfig, score_genes_path, score_output_path, write_genes_file

__all__ = ["dump_mi_scores", "dump_puc_scores"]


def _dump_score_matrix(scores: np.ndarray, nodes: list[Node], file_path: str, score_name: str) -> str:
    n = len(nodes)
    if scores.shape != (n, n):
        raise ValueError(f"score matrix shape {scores.shape} does not match {n} nodes")

    output_path = score_output_path(file_path, score_name)
    np.save(output_path, scores)
    write_genes_file(score_genes_path(output_path, score_name), nodes)
    return output_path


def dump_mi_scores(mi_scores: np.ndarray, nodes: list[Node], config: PIDCConfig) -> str | None:
    """Write the full symmetric MI score matrix to ``<stem>_mi.npy``
    (float64), alongside a ``<stem>_genes.txt`` row/column sidecar."""
    if config.dump_mi_path is None:
        return None
    return _dump_score_matrix(mi_scores, nodes, config.dump_mi_path, "mi")


def dump_puc_scores(scores: np.ndarray, nodes: list[Node], config: PIDCConfig) -> str | None:
    """Write the full symmetric pre-context PUC score matrix to
    ``<stem>_puc.npy`` (float64), alongside a ``<stem>_genes.txt``
    row/column sidecar."""
    if config.dump_puc_path is None:
        return None
    return _dump_score_matrix(scores, nodes, config.dump_puc_path, "puc")
