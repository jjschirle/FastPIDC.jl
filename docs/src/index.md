```@meta
CurrentModule = FastPIDC
```

# FastPIDC.jl

FastPIDC is a package for inferring (undirected) networks, given a set of
measurements for each node. The main output is the [`InferredNetwork`](@ref)
type, which represents a fully connected, weighted network, where an edge's
weight indicates the relative confidence of that edge existing in the true
network.

Some things to note:
* The package was originally written for inferring biological networks using
  gene expression data, hence the use of "network" instead of "graph".
  However, these methods could be applied to other types of data.
* Four network inference algorithms are implemented: MI, CLR, PUC and PIDC
  (explained in [1](#References)).
* Networks are assumed to be **undirected**, since all the algorithms
  included so far infer undirected networks. Hence:
  * in the [`Edge`](@ref) type, the order of the nodes is arbitrary
  * when a network is written to a TSV file, edges are written in both
    directions, because downstream analyses sometimes require this
  * the [`InferredNetwork`](@ref) type contains a list of edges, with one
    edge for each pair of nodes, in which the order of the nodes is
    arbitrary

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/meyer-lab/FastPIDC.jl")
```

## Basic usage

First load the package:

```julia
using FastPIDC
```

### One step

Given a data file and an inference algorithm, you can infer a network with a
single function call:

```julia
infer_network(<path to data file>, PIDCNetworkInference())
```

This will return an [`InferredNetwork`](@ref). You can also write the
inferred network to file, using the `out_file_path` keyword argument. See
[`infer_network`](@ref) for the full set of options.

### Multiple steps

First make an array of [`Node`](@ref)s from your data:

```julia
nodes = get_nodes(<path to data file>)
```

[`get_nodes`](@ref) accepts either a delimited text file or an HDF5 (`.h5`)
file (see [`get_nodes_h5`](@ref) for the expected schema). For text files,
the format is:
* line 1: headers (these are discarded for now)
* other lines: `NodeLabel value1 value2 value3 ...`

Then infer a network:

```julia
inferred_network = InferredNetwork(PIDCNetworkInference(), nodes)
```

An [`InferredNetwork`](@ref) has an array of nodes and an array of edges
between all possible node pairs (sorted in descending order of edge weight,
i.e. confidence of the edge existing in the true network).

You can write the network to a TSV file:

```julia
write_network_file(<path to output file>, inferred_network)
```

or to a NumPy-compatible dense binary matrix (plus a gene-label sidecar
file), for fast loading from Python:

```julia
write_network_npy(<path to output file>, inferred_network)
```

## Configuration and GPU acceleration

[`PIDCConfig`](@ref) controls the computation backend (`:cuda`, the default,
or `:cpu`), diagnostic score dumps, and verbosity, and can be passed to
[`infer_network`](@ref) or [`InferredNetwork`](@ref) via the `config`
keyword argument:

```julia
using CUDA  # loads the GPU backend, FastPIDCCUDAExt

cfg = PIDCConfig(backend = :cuda, verbose = true)
infer_network(<path to data file>, PIDCNetworkInference(); config = cfg)
```

If `backend = :cuda` is requested but `CUDA.jl` has not been loaded (or no
functional GPU is available), an error is raised suggesting `:cpu` instead.

## Command-line interface

`CLI_fastpidc.jl` at the repository root wraps [`infer_network`](@ref) as a
standalone script, useful for running PIDC network inference without writing
any Julia code:

```sh
julia --project=. CLI_fastpidc.jl --infile X.txt --outfile edges.tsv --backend cuda
```

Run with `--help` for the full list of options.

## Options

The following keyword arguments can be passed to [`infer_network`](@ref):

**delim** (`Union{Char,Bool}`) Column delimiter
* `false` (default) Delimiter is whitespace

**discretizer** (`String`) Method for discretizing
* `"bayesian_blocks"` (default) Adaptive discretizer with a variable number of bins
* `"uniform_width"` Use this if Bayesian blocks fails, or if a constant number of bins is required
* `"uniform_count"`

**estimator** (`String`) Estimator for estimating the probability distribution
* `"maximum_likelihood"` (default) Highly recommended for PUC and PIDC
* `"dirichlet"`
* `"shrinkage"`

**number_of_bins** (`Integer`)
* `10` (default) (ignored when using Bayesian blocks discretization)

**base** (`Number`) Base of the logarithm, i.e. the units for entropy
* `2` (default)

**out_file_path** (`String`) Path to the output network file
* `""` (default) No file will be written

**output_format** (`Symbol`) Format for `out_file_path`
* `:tsv` (default) Plain-text edge list, see [`write_network_file`](@ref)
* `:npy` NumPy-compatible dense matrix, see [`write_network_npy`](@ref)

**config** ([`PIDCConfig`](@ref)) Computation backend and diagnostics settings

## Scope

This package is not designed for analysing networks/graphs or calculating
network/graph metrics. In order to do such analyses, another package should
be used (e.g. [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl)). Of
course, the edge list or the [`InferredNetwork`](@ref) will need to be
parsed into the appropriate data structure first; [`get_adjacency_matrix`](@ref)
may help with this.

Note that the [`InferredNetwork`](@ref) type contains a list of every
possible edge, and the confidence of each edge existing in the true network.
For analysing the properties of an inferred network, you may first want to
define a partially connected, unweighted network by classifying each edge as
"in the network" or "not in the network", based on the confidences. The
simplest ways to do this are either to decide that the top x percent of
edges are "in the network", or to define a threshold confidence, above which
edges are "in the network".

You can pass a threshold into [`get_adjacency_matrix`](@ref) to get the
adjacency matrix of a thresholded network (as well as dictionaries to map the
node labels to their numerical IDs within the matrix, and vice versa):

```julia
get_adjacency_matrix(inferred_network, 0.1) # Keeps top 10% edges with the largest weights
get_adjacency_matrix(inferred_network, 0.1, absolute = true) # Keeps all edges with weights >= 0.1
```

## Performance

It may be possible to speed up an analysis, particularly for large datasets,
by using [multiple processes](https://docs.julialang.org/en/v1/manual/distributed-computing/)
for the MI backend, or a CUDA-capable GPU for the PUC/PIDC backend (the
default; see [`PIDCConfig`](@ref)).

If multiple processes are available, FastPIDC will distribute the most
costly CPU calculations across the processes (the for loops in
[`get_mi_scores`](@ref) and [`compute_puc_full`](@ref)):

```sh
julia -p 3 --project=.
```

```julia
using FastPIDC
infer_network(<path to data file>, PIDCNetworkInference())
```

This starts the Julia REPL with 3 extra worker processes (so 4 in total).
Note that the performance gain from distributing calculations is offset by
communicating between the processes, so for small datasets it is more
efficient to use one process. For the same reason, using too many processes
will degrade performance, so it is a good idea to do some timing tests with
different numbers of processes.

## References

[1] Chan, Stumpf and Babtie (2017) [Gene Regulatory Network Inference from
Single-Cell Data Using Multivariate Information
Measures](http://www.cell.com/cell-systems/fulltext/S2405-4712(17)30386-1)
Cell Systems
