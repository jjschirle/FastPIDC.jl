"""Basic types for inferring a network.

Ported from FastPIDC.jl's ``src/common.jl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .discretizers import get_bin_ids
from .information import get_frequencies_from_bin_ids, get_probabilities

__all__ = ["Edge", "InferredNetwork", "Node", "PIDCConfig"]

_VALID_BACKENDS = ("cpu", "cuda")


@dataclass(frozen=True, slots=True)
class PIDCConfig:
    """Runtime configuration for PUC/PIDC network inference.

    Parameters
    ----------
    backend : computation backend for PUC/PIDC, either ``"cuda"`` (default,
        requires an optional GPU dependency and a functional GPU) or
        ``"cpu"``.
    bb_backend : computation backend for the Bayesian-blocks dynamic program,
        either ``"cuda"`` (default) or ``"cpu"``.
    discretizer, estimator : mirror the defaults used by :func:`get_nodes`.
    dump_mi_path, dump_puc_path : if set, the pairwise MI / pre-context PUC
        score matrix is written to ``<stem>_mi.npy`` / ``<stem>_puc.npy``
        (see :mod:`fastpidc.dump`).
    verbose : print progress information while inferring the network.

    Notes
    -----
    The two backends fail differently, mirroring FastPIDC.jl: requesting
    ``backend="cuda"`` without a usable GPU raises, whereas
    ``bb_backend="cuda"`` warns and falls back to the CPU solver, since the two
    Bayesian-block solvers produce the same bin edges either way.
    """

    backend: str = "cuda"
    bb_backend: str = "cuda"
    discretizer: str = "bayesian_blocks"
    estimator: str = "maximum_likelihood"
    dump_mi_path: str | None = None
    dump_puc_path: str | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {self.backend!r}")
        if self.bb_backend not in _VALID_BACKENDS:
            raise ValueError(f"bb_backend must be one of {_VALID_BACKENDS}, got {self.bb_backend!r}")


@dataclass(frozen=True, slots=True)
class Node:
    """A node (e.g. gene) with its discretized data and metadata.

    Attributes
    ----------
    label : unique identifying label.
    binned_values : data values discretized into (0-indexed) bins.
    number_of_bins : number of bins the data were discretized into.
    probabilities : probability distribution across the bins.
    """

    label: str
    binned_values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    number_of_bins: int = 0
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))

    @classmethod
    def from_raw_values(
        cls,
        label: str,
        raw_values: np.ndarray,
        discretizer: str,
        estimator: str,
        number_of_bins: int,
    ) -> Node:
        """Construct a :class:`Node` from raw (continuous) data values,
        discretizing with ``discretizer`` and estimating bin probabilities
        with ``estimator``."""
        raw_values = np.asarray(raw_values, dtype=np.float64)
        binned_values, number_of_bins = get_bin_ids(raw_values, discretizer, number_of_bins)
        frequencies = get_frequencies_from_bin_ids(binned_values, number_of_bins)
        probabilities = get_probabilities(estimator, frequencies)
        return cls(str(label), binned_values, number_of_bins, probabilities)


@dataclass(frozen=True, slots=True)
class Edge:
    """Undirected edge between two nodes.

    ``weight`` indicates confidence of the edge existing in the true
    network. Weights are used to rank edges; different algorithms may use a
    different scale, so relative weights within one inferred network are
    more meaningful than the absolute weight out of context.
    """

    nodes: tuple[Node, Node]
    weight: float


@dataclass(frozen=True, slots=True)
class InferredNetwork:
    """A weighted, fully connected network.

    An edge's weight indicates the relative confidence of that edge existing
    in the true network.

    Attributes
    ----------
    nodes : all the nodes, in an arbitrary order.
    edges : all the edges, in descending order of weight.
    """

    nodes: list[Node]
    edges: list[Edge]


# --- Output path helpers -----------------------------------------------


def npy_output_path(file_path: str) -> str:
    """Replace the extension of ``file_path`` with ``.npy``."""
    return str(Path(file_path).with_suffix(".npy"))


def score_output_path(file_path: str, score_name: str) -> str:
    """Build the ``.npy`` output path for a score dump named ``score_name``
    (``"mi"`` or ``"puc"``), appending a ``_mi``/``_puc`` suffix to the stem
    of ``file_path`` unless it is already present."""
    if score_name not in ("mi", "puc"):
        raise ValueError(f"score_name must be 'mi' or 'puc', got {score_name!r}")
    npy_path = Path(npy_output_path(file_path))
    suffix = f"_{score_name}"
    stem = npy_path.stem
    if not stem.endswith(suffix):
        stem = stem + suffix
    return str(npy_path.with_name(stem + ".npy"))


def network_genes_path(file_path: str) -> str:
    """Path of the gene-label sidecar file for an inferred-network ``.npy``
    dump at ``file_path``."""
    return str(Path(npy_output_path(file_path)).with_name(Path(npy_output_path(file_path)).stem + "_genes.txt"))


def score_genes_path(file_path: str, score_name: str) -> str:
    """Path of the gene-label sidecar file for the ``score_name`` (``"mi"``
    or ``"puc"``) ``.npy`` dump derived from ``file_path``."""
    score_path = Path(score_output_path(file_path, score_name))
    suffix = f"_{score_name}"
    stem = score_path.stem
    stem = stem.removesuffix(suffix)
    return str(score_path.with_name(stem + "_genes.txt"))


def write_genes_file(file_path: str, nodes) -> None:
    """Write one :class:`Node` label per line to ``file_path``, in the order
    given by ``nodes``, to serve as the row/column key for a companion
    ``.npy`` matrix."""
    with open(file_path, "w") as fh:
        fh.writelines(f"{node.label}\n" for node in nodes)
