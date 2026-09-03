# fastpidc

Python port of [FastPIDC.jl](https://github.com/meyer-lab/FastPIDC.jl):
information-theoretic network inference (MI, CLR, PUC, PIDC) from per-node
(e.g. per-gene) measurements, following Chan, Stumpf & Babtie (2017),
*Gene Regulatory Network Inference from Single-Cell Data Using Multivariate
Information Measures*, Cell Systems.

This package is a standalone, native Python/NumPy implementation - it does
not call out to Julia. See the repository root README for how this package
and the Julia package are organized and released together.

## Installation

This package is managed with [uv](https://docs.astral.sh/uv/). From this
directory:

```sh
uv sync                 # install (CPU-only)
uv sync --extra cuda    # also install the optional GPU (cupy) backend
```

## Basic usage

```python
from fastpidc import get_nodes, infer_network, PIDCNetworkInference, PIDCConfig

# One step: infer directly from a data file.
network = infer_network(
    "data.txt",
    PIDCNetworkInference(),
    config=PIDCConfig(backend="cpu"),  # or "cuda", if cupy + a GPU are available
)

# Multiple steps: build nodes, then infer.
from fastpidc.network import infer_network_from_nodes

nodes = get_nodes("data.txt")
network = infer_network_from_nodes(PIDCNetworkInference(), nodes)
```

The input data file's first line is treated as a header (discarded); each
subsequent line is `label value1 value2 ...`.

`network.edges` is sorted by descending weight (confidence). Use
`fastpidc.write_network_file`/`write_network_npy` to save it, and
`fastpidc.get_adjacency_matrix` to threshold it into an adjacency matrix.

## Command line

```sh
uv run fastpidc --infile data.txt --outfile edges.tsv --backend cpu
```

Run `uv run fastpidc --help` for the full option list (discretizer,
estimator, output format, diagnostic score dumps, etc.) - it mirrors
`CLI_fastpidc.jl` in the Julia package.

## Backends

* `backend="cpu"` (works everywhere): a direct NumPy port of the reference
  algorithm. Recommended for small-to-moderate networks (tens to low
  hundreds of nodes); PUC/PIDC are inherently `O(N^3)`.
* `backend="cuda"` (optional `cupy` dependency + a functional GPU):
  chunked GPU kernels for large-scale problems, ported from - and kept
  numerically consistent with - the CUDA kernels in FastPIDC.jl's
  `FastPIDCCUDAExt`. The shared kernel source lives at
  `src/fastpidc/kernels/pidc_kernels.cu`; see that file's header comment for
  the sharing strategy between the two packages.

## Testing

```sh
uv run pytest
```

Some tests cross-check results against the Julia package's baseline
outputs and, when a working `julia` executable is on `PATH`, against a
freshly run `FastPIDC.jl`; those are skipped automatically otherwise.
