"""Reading data files into :class:`~fastpidc.types.Node`s, and reading/writing
inferred networks.

Ported from FastPIDC.jl's ``src/infer_network.jl``.
"""

from __future__ import annotations

import csv

import h5py
import numpy as np
from scipy import sparse

from .types import (
    Edge,
    InferredNetwork,
    Node,
    network_genes_path,
    npy_output_path,
    write_genes_file,
)

__all__ = [
    "get_adjacency_matrix",
    "get_nodes",
    "read_network_file",
    "write_network_file",
    "write_network_npy",
]

_SPARSE_KEY = "matrix_sparse_csc"
_MATRIX_KEYS = (_SPARSE_KEY, "matrix_dense", "X", "matrix", "data")


def get_nodes(
    data_file_path: str,
    *,
    delim: str | None = None,
    discretizer: str = "bayesian_blocks",
    estimator: str = "maximum_likelihood",
    number_of_bins: int = 10,
) -> list[Node]:
    """Get a list of all :class:`Node`s from a data file.

    Dispatches to :func:`_get_nodes_h5` for ``.h5`` files, or to the
    whitespace/delimited text loader otherwise.
    """
    if str(data_file_path).endswith(".h5"):
        return _get_nodes_h5(
            data_file_path, discretizer=discretizer, estimator=estimator, number_of_bins=number_of_bins
        )
    return _get_nodes_text(
        data_file_path,
        delim=delim,
        discretizer=discretizer,
        estimator=estimator,
        number_of_bins=number_of_bins,
    )


def _get_nodes_text(
    data_file_path: str,
    *,
    delim: str | None = None,
    discretizer: str = "bayesian_blocks",
    estimator: str = "maximum_likelihood",
    number_of_bins: int = 10,
) -> list[Node]:
    """Get a list of all Nodes from a data file.

    The first line of the file is assumed to be headers (discarded), and
    each subsequent line represents one node::

        Label    data_value1  data_value2 ...

    though a different delimiter may be specified. ``"maximum_likelihood"``
    is recommended as the estimator for PUC and PIDC.
    """
    with open(data_file_path, newline="") as fh:
        reader = csv.reader(fh, delimiter=delim) if delim else _WhitespaceReader(fh)
        next(reader)  # discard header line
        rows = [row for row in reader if row]

    nodes = [
        Node.from_raw_values(row[0], np.array(row[1:], dtype=np.float64), discretizer, estimator, number_of_bins)
        for row in rows
    ]
    return nodes


class _WhitespaceReader:
    """Minimal whitespace-delimited line reader, mirroring
    ``DelimitedFiles.readdlm``'s default (any run of whitespace splits
    fields)."""

    def __init__(self, fh):
        self._fh = fh

    def __iter__(self):
        return self

    def __next__(self):
        line = next(self._fh)
        return line.split()


def _get_nodes_h5(
    data_file_path: str,
    *,
    discretizer: str = "bayesian_blocks",
    estimator: str = "maximum_likelihood",
    number_of_bins: int = 10,
) -> list[Node]:
    """Get a list of all :class:`Node`s from an HDF5 (``.h5``) expression file.

    The file must contain a ``"gene_names"`` dataset, and an expression
    matrix under one of ``"matrix_sparse_csc"``, ``"matrix_dense"``, ``"X"``,
    ``"matrix"`` or ``"data"`` (checked in that order). The matrix may be a
    dense HDF5 dataset, with the logical shape ``(cells, genes)`` as written
    by AnnData/scanpy and reported by h5py, or an HDF5 group holding a CSC
    sparse matrix (``"data"``, ``"indices"``, ``"indptr"`` datasets plus a
    ``"shape"`` attribute) in that same ``(cells, genes)`` orientation.
    """
    with h5py.File(data_file_path, "r") as f:
        if "gene_names" not in f:
            raise ValueError(f"Invalid HDF5 schema in {data_file_path}. Missing required dataset: 'gene_names'")
        gene_names = [name.decode() if isinstance(name, bytes) else str(name) for name in f["gene_names"][()]]
        number_of_nodes = len(gene_names)

        matrix_key = next((key for key in _MATRIX_KEYS if key in f), None)
        if matrix_key is None:
            raise ValueError("Could not find expression data. Expected key 'X' or similar.")

        obj = f[matrix_key]

        if isinstance(obj, h5py.Group):
            required = ("data", "indices", "indptr")
            if not all(k in obj for k in required):
                raise ValueError(
                    f"Sparse matrix group '{matrix_key}' is missing required datasets: 'data', 'indices', or 'indptr'."
                )
            if "shape" not in obj.attrs:
                raise ValueError(f"Sparse matrix group '{matrix_key}' is missing the required attribute: 'shape'.")
            shape = tuple(int(s) for s in obj.attrs["shape"])
            x_raw = sparse.csc_matrix((obj["data"][()], obj["indices"][()], obj["indptr"][()]), shape=shape)
        elif isinstance(obj, h5py.Dataset):
            # h5py reports the dataset's logical shape as stored, (cells, genes).
            x_raw = obj[()]
        else:
            raise ValueError(f"Object at '{matrix_key}' is neither an HDF5 group nor a dataset.")

    nodes: list[Node] = [None] * number_of_nodes  # type: ignore[list-item]
    for i in range(number_of_nodes):
        column = x_raw[:, i]
        if sparse.issparse(column):
            column = np.asarray(column.todense()).ravel()
        else:
            column = np.asarray(column).ravel()
        nodes[i] = Node.from_raw_values(
            gene_names[i], column.astype(np.float64), discretizer, estimator, number_of_bins
        )

    return nodes


def write_network_file(file_path: str, inferred_network: InferredNetwork) -> None:
    """Write a network file from an :class:`InferredNetwork`.

    Each line contains an edge; since networks are undirected, each edge is
    written in both directions with the same weight::

        LabelX   LabelY  WeightXY
        LabelY   LabelX  WeightXY
        ...
    """
    with open(file_path, "w") as out:
        for edge in inferred_network.edges:
            n1, n2 = edge.nodes
            out.write(f"{n1.label}\t{n2.label}\t{edge.weight}\n")
            out.write(f"{n2.label}\t{n1.label}\t{edge.weight}\n")


def write_network_npy(file_path: str, inferred_network: InferredNetwork) -> None:
    """Write an inferred undirected weighted network as a dense NumPy binary
    file (``.npy``) plus a sidecar gene list file preserving row/column order.

    Outputs
    -------
    file_path : ``N x N`` dense weighted adjacency matrix (float32).
    ``<stem>_genes.txt`` : one gene label per line, matching matrix
        row/column order.
    """
    file_path = npy_output_path(file_path)

    labels = [node.label for node in inferred_network.nodes]
    labels_to_ids = {label: i for i, label in enumerate(labels)}
    n = len(labels)

    adjacency = np.zeros((n, n), dtype=np.float32)
    for edge in inferred_network.edges:
        i = labels_to_ids[edge.nodes[0].label]
        j = labels_to_ids[edge.nodes[1].label]
        w = np.float32(edge.weight)
        adjacency[i, j] = w
        adjacency[j, i] = w

    np.save(file_path, adjacency)
    write_genes_file(network_genes_path(file_path), inferred_network.nodes)


def read_network_file(file_path: str) -> InferredNetwork:
    """Read a network file and create an :class:`InferredNetwork`.

    Assumes each line contains an edge and each edge is written in both
    directions with the same weight (see :func:`write_network_file`).
    """
    labels_to_nodes: dict[str, Node] = {}
    edges: list[Edge] = []

    def _get_node(label: str) -> Node:
        if label not in labels_to_nodes:
            labels_to_nodes[label] = Node(label)
        return labels_to_nodes[label]

    with open(file_path) as fh:
        for line_number, line in enumerate(fh):
            if line_number % 2 != 0:
                continue  # each edge is written twice; keep only one direction
            if not line.strip():
                continue
            label1, label2, weight = line.split()
            n1, n2 = _get_node(label1), _get_node(label2)
            edges.append(Edge((n1, n2), float(weight)))

    return InferredNetwork(list(labels_to_nodes.values()), edges)


def get_adjacency_matrix(
    inferred_network: InferredNetwork, threshold: float = 0.1, *, absolute: bool = False
) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    """Get an adjacency matrix given an :class:`InferredNetwork` and a
    threshold.

    If ``absolute`` is ``False`` (default), ``threshold`` is the proportion
    of (highest-weighted) edges to keep; otherwise it is interpreted as an
    absolute confidence score, and all edges with weight ``>= threshold``
    are kept.
    """
    nodes = inferred_network.nodes
    n = len(nodes)
    adjacency = np.zeros((n, n), dtype=bool)

    labels_to_ids = {node.label: i for i, node in enumerate(nodes)}
    ids_to_labels = {i: node.label for i, node in enumerate(nodes)}

    edges = inferred_network.edges
    if absolute:
        number_of_edges = next((i for i, e in enumerate(edges) if e.weight < threshold), len(edges))
    else:
        number_of_edges = round(len(edges) * threshold)

    for edge in edges[:number_of_edges]:
        i = labels_to_ids[edge.nodes[0].label]
        j = labels_to_ids[edge.nodes[1].label]
        adjacency[i, j] = True
        adjacency[j, i] = True

    return adjacency, labels_to_ids, ids_to_labels
