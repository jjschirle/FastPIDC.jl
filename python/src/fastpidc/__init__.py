"""fastpidc: information-theoretic network inference (MI, CLR, PUC, PIDC).

A package for inferring undirected gene regulatory (or other) networks from
per-node measurements, using the information-theoretic algorithms MI, CLR,
PUC and PIDC (Chan, Stumpf & Babtie 2017). This is a Python port of
`FastPIDC.jl <https://github.com/meyer-lab/FastPIDC.jl>`_; see that
repository's ``python/`` directory for how the two packages are kept in
sync.

The main entry points are :func:`get_nodes`/:func:`infer_network` to build
an :class:`InferredNetwork` from a data file, and
:func:`write_network_file`/:func:`write_network_npy` to save the result.
"""

from .api import infer_network
from .io import get_adjacency_matrix, get_nodes, read_network_file, write_network_file, write_network_npy
from .network import (
    AbstractNetworkInference,
    CLRNetworkInference,
    MINetworkInference,
    PIDCNetworkInference,
    PUCNetworkInference,
)
from .pid import PIDTriple, combined_node, pid_triple
from .types import Edge, InferredNetwork, Node, PIDCConfig

__all__ = [
    # Common types
    "Node",
    "Edge",
    "InferredNetwork",
    "PIDCConfig",
    # Network inference algorithms
    "AbstractNetworkInference",
    "MINetworkInference",
    "CLRNetworkInference",
    "PUCNetworkInference",
    "PIDCNetworkInference",
    # Functions for inferring networks
    "get_nodes",
    "write_network_file",
    "write_network_npy",
    "read_network_file",
    "get_adjacency_matrix",
    "infer_network",
    # Triple partial information decomposition
    "PIDTriple",
    "combined_node",
    "pid_triple",
]

__version__ = "0.1.0"
