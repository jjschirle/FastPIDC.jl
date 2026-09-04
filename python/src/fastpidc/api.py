"""Top-level convenience entry point.

Ported from FastPIDC.jl's ``infer_network`` (``src/infer_network.jl``).
"""

from __future__ import annotations

from .io import get_nodes, write_network_file, write_network_npy
from .network import AbstractNetworkInference, infer_network_from_nodes
from .types import InferredNetwork, PIDCConfig

__all__ = ["infer_network"]

_OUTPUT_FORMATS = ("tsv", "npy")


def infer_network(
    data_file_path: str,
    inference: AbstractNetworkInference,
    *,
    delim: str | None = None,
    discretizer: str = "bayesian_blocks",
    estimator: str = "maximum_likelihood",
    number_of_bins: int = 10,
    base: float = 2,
    out_file_path: str | None = None,
    output_format: str = "tsv",
    config: PIDCConfig | None = None,
) -> InferredNetwork:
    """Infer a network, given a data file and a network inference algorithm.

    The first line of the file is assumed to be headers (discarded), and
    each subsequent line represents one node::

        Label    data_value1  data_value2 ...

    though a different delimiter may be specified. ``"maximum_likelihood"``
    is recommended as the estimator for PUC and PIDC.

    Parameters
    ----------
    out_file_path : path to the output file. If ``None`` (default), no file
        is written.
    output_format : ``"tsv"`` or ``"npy"``.
    """
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output_format={output_format!r}. Use 'tsv' or 'npy'.")
    if config is None:
        config = PIDCConfig()

    if config.verbose:
        print("[fastpidc] Getting nodes...")
    nodes = get_nodes(
        data_file_path,
        delim=delim,
        discretizer=discretizer,
        estimator=estimator,
        number_of_bins=number_of_bins,
        bb_backend=config.bb_backend,
        verbose=config.verbose,
    )

    if config.verbose:
        print("[fastpidc] Inferring network...")
    inferred_network = infer_network_from_nodes(inference, nodes, estimator=estimator, base=base, config=config)

    if out_file_path:
        if config.verbose:
            print("[fastpidc] Writing network to file...")
        if output_format == "tsv":
            write_network_file(out_file_path, inferred_network)
        else:
            write_network_npy(out_file_path, inferred_network)

    return inferred_network
