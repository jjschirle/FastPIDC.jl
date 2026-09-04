"""Cross-validate the Python port against a freshly run FastPIDC.jl.

These tests shell out to `julia --project=<repo root>` and are skipped
automatically if a working `julia` executable is not on PATH (e.g. in CI
jobs that don't set up Julia). They are the main correctness guarantee that
this package produces the same results as FastPIDC.jl, without depending on
Julia at runtime for normal use.

Note: `test/baseline_outputs/*.tsv` in the repository are pre-existing
snapshot files that may be stale relative to the current Julia source (this
was true for at least one gene in `pidc_toy_edges.tsv` at the time this
suite was written - see the git history for details), so these tests always
run Julia fresh rather than trusting those files.
"""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from fastpidc import PIDCConfig, get_nodes
from fastpidc.network import (
    CLRNetworkInference,
    MINetworkInference,
    PIDCNetworkInference,
    PUCNetworkInference,
    get_mi_scores,
    infer_network_from_nodes,
)

pytestmark = pytest.mark.julia

_ALGORITHMS = (
    ("mi", MINetworkInference),
    ("clr", CLRNetworkInference),
    ("puc", PUCNetworkInference),
    ("pidc", PIDCNetworkInference),
)


def _run_julia_networks(run_julia, data_file: str, out_dir):
    """Infer all four networks with FastPIDC.jl, returning the output paths and
    the bin count Julia's discretizer chose for each node.

    Bayesian blocks is pinned to Julia's CPU solver: its `:cuda` default only
    takes effect once the caller has done `using CUDA` (which loads the
    package extension), so leaving it here would just emit a fallback warning.
    FastPIDC.jl's own suite covers its CPU/GPU Bayesian-block equivalence.
    """
    out_paths = {name: str(out_dir / f"julia_{name}.tsv") for name, _ in _ALGORITHMS}
    inference_exprs = "\n".join(
        f'write_network_file("{out_paths[name]}", InferredNetwork({cls.__name__}(), nodes; config=cfg))'
        for name, cls in _ALGORITHMS
    )
    output = run_julia(
        textwrap.dedent(
            f"""
            using FastPIDC
            nodes = get_nodes("{data_file}"; bb_backend=:cpu)
            println("NBINS:", join([n.number_of_bins for n in nodes], ","))
            cfg = PIDCConfig(backend=:cpu)
            {inference_exprs}
            """
        )
    )
    nbins_line = next(line for line in output.splitlines() if line.startswith("NBINS:"))
    nbins = [int(x) for x in nbins_line.removeprefix("NBINS:").split(",")]
    return out_paths, nbins


def _load_edge_map(path: str) -> dict:
    edges = {}
    with open(path) as fh:
        for line in fh:
            a, b, w = line.split()
            edges[frozenset((a, b))] = float(w)
    return edges


def _edge_map(network) -> dict:
    return {frozenset((e.nodes[0].label, e.nodes[1].label)): e.weight for e in network.edges}


def _assert_matches(python_map: dict, julia_map: dict, atol: float = 1e-8) -> None:
    """Compare edge weights keyed by node-label pair.

    The default tolerance is far above the differences actually observed
    (~1e-13 for PIDC, ~1e-15 for MI/PUC), which come only from the two
    implementations' summation order.
    """
    assert set(python_map) <= set(julia_map)
    max_diff = max(abs(python_map[k] - julia_map[k]) for k in python_map)
    assert max_diff < atol, f"max diff {max_diff} exceeds tolerance"


# CLR standardizes each raw MI score against that node's background
# distribution as (score - mean) / sqrt(variance). A node whose mutual
# information against every other node is "zero up to floating point" (e.g. a
# near-constant column) has a background variance around 1e-32, which turns
# ordinary float64 rounding differences between the two implementations'
# summation order into O(1) differences in the standardized score. That is
# numerical ill-conditioning inherent to dividing by a near-zero variance, not
# a porting bug, so edges touching such a node are checked only for rank
# concordance.
_DEGENERATE_BACKGROUND_VARIANCE = 1e-20


def _ill_conditioned_labels(nodes) -> set[str]:
    mi = get_mi_scores(nodes, "maximum_likelihood", 2)
    return {
        node.label
        for i, node in enumerate(nodes)
        if np.delete(mi[:, i], i).var(ddof=1) < _DEGENERATE_BACKGROUND_VARIANCE
    }


@pytest.mark.parametrize("data_file_name", ["yeast1_10_data.txt", "toy_small_200.txt"])
def test_all_algorithms_match_julia(run_julia, julia_test_data, data_file_name, tmp_path):
    data_file = julia_test_data / data_file_name
    out_paths, julia_nbins = _run_julia_networks(run_julia, str(data_file), tmp_path)

    nodes = get_nodes(str(data_file))
    python_nbins = [n.number_of_bins for n in nodes]

    # FastPIDC.jl's Bayesian-blocks change-point backtracking used to have a
    # narrow off-by-one edge case (a BoundsError, caught and silently
    # downgraded to a uniform-width fallback) for data with very few unique
    # values whose optimal segmentation assigned every point its own block -
    # e.g. mostly constant data with a single outlier - which this Python
    # port never reproduced. Now that the off-by-one is fixed in the Julia
    # source too, bin counts should always agree; this guards that.
    assert python_nbins == julia_nbins

    ill_conditioned = _ill_conditioned_labels(nodes)
    config = PIDCConfig(backend="cpu")
    for name, inference_cls in _ALGORITHMS:
        network = infer_network_from_nodes(inference_cls(), nodes, config=config)
        python_edges = _edge_map(network)
        julia_edges = _load_edge_map(out_paths[name])
        assert set(python_edges) == set(julia_edges)

        if name == "clr":
            well_conditioned = {k: v for k, v in python_edges.items() if not (k & ill_conditioned)}
            _assert_matches(well_conditioned, julia_edges)

            py_vals = [python_edges[k] for k in python_edges]
            jl_vals = [julia_edges[k] for k in python_edges]
            correlation = float(np.corrcoef(py_vals, jl_vals)[0, 1])
            assert correlation > 0.98, f"{name}: correlation {correlation} too low"
        else:
            _assert_matches(python_edges, julia_edges)


def test_gpu_backend_matches_julia(run_julia, julia_test_data, tmp_path):
    from fastpidc.cuda import cuda_available

    if not cuda_available():
        pytest.skip("no functional GPU / cupy backend available")

    data_file = julia_test_data / "toy_small_200.txt"
    out_paths, _ = _run_julia_networks(run_julia, str(data_file), tmp_path)

    nodes = get_nodes(str(data_file))
    gpu_network = infer_network_from_nodes(PIDCNetworkInference(), nodes, config=PIDCConfig(backend="cuda"))
    _assert_matches(_edge_map(gpu_network), _load_edge_map(out_paths["pidc"]), atol=1e-6)
