# Basic types for inferring a network

"""
Node with metadata

Fields:
* `label`: unique identifying label
* `binned_values`: data values discretized into bins
* `number_of_bins`: no. bins the data were discretized into
* `probabilities`: probability distribution across the bins
"""
struct Node
    label::String
    binned_values::Array{Int64}
    number_of_bins::Int64
    probabilities::Array{Float64}
end

"""
    PIDCConfig(; backend=:cuda, discretizer="bayesian_blocks",
                 estimator="maximum_likelihood", dump_mi_path=nothing,
                 dump_puc_path=nothing, verbose=false)

Runtime configuration for PUC/PIDC network inference.

# Fields
* `backend::Symbol`: computation backend, either `:cuda` (default, requires
  `using CUDA`, a functional GPU, and `nvcc` on `PATH` to compile the
  shared kernel source the first time it's used - see `ext/FastPIDCCUDAExt`)
  or `:cpu`.
* `discretizer::String`: discretization method, mirrors the default used by
  [`get_nodes`](@ref).
* `estimator::String`: probability estimator, mirrors the default used by
  [`get_nodes`](@ref).
* `dump_mi_path::Union{Nothing,String}`: if set, the pairwise MI score matrix
  is written to `<stem>_mi.npy` (see [`dump_mi_scores`](@ref)).
* `dump_puc_path::Union{Nothing,String}`: if set, the pre-context PUC score
  matrix is written to `<stem>_puc.npy` (see [`dump_puc_scores`](@ref)).
* `verbose::Bool`: print progress information while inferring the network.

Throws an `ArgumentError` if `backend` is not `:cpu` or `:cuda`.
"""
Base.@kwdef struct PIDCConfig
    backend::Symbol = :cuda                     # PUC backend: :cuda or :cpu
    bb_backend::Symbol = :cuda                  # Bayesian blocks backend: :cuda or :cpu
    discretizer::String = "bayesian_blocks"     # mirrors existing default
    estimator::String = "maximum_likelihood"    # mirrors existing default
    dump_mi_path::Union{Nothing,String} = nothing   # Output stem/path; writes *_mi.npy
    dump_puc_path::Union{Nothing,String} = nothing  # Output stem/path; writes *_puc.npy
    verbose::Bool = false
    # Inner constructor for automatic validation
    function PIDCConfig(
        backend,
        bb_backend,
        discretizer,
        estimator,
        dump_mi_path,
        dump_puc_path,
        verbose,
    )

        if !(backend in (:cpu, :cuda))
            throw(ArgumentError("backend must be :cpu or :cuda, got :$backend"))
        end
        if !(bb_backend in (:cpu, :cuda))
            throw(ArgumentError("bb_backend must be :cpu or :cuda, got :$bb_backend"))
        end
        new(
            backend,
            bb_backend,
            discretizer,
            estimator,
            dump_mi_path,
            dump_puc_path,
            verbose,
        )
    end
end

# --- NumPy output helpers -----------------------------------------

"""
    _npy_output_path(file_path) -> String

Replace the extension of `file_path` with `.npy`, preserving the stem.
"""
function _npy_output_path(file_path::AbstractString)
    stem, _ = splitext(String(file_path))
    return stem * ".npy"
end

"""
    _score_output_path(file_path, score_name) -> String

Build the `.npy` output path for a score dump named `score_name`
(`:mi` or `:puc`), appending a `_mi`/`_puc` suffix to the stem of
`file_path` unless it is already present. Throws an `ArgumentError` for any
other `score_name`.
"""
function _score_output_path(file_path::AbstractString, score_name::Symbol)
    score_name in (:mi, :puc) ||
        throw(ArgumentError("score_name must be :mi or :puc, got :$score_name"))

    npy_path = _npy_output_path(file_path)
    stem, _ = splitext(npy_path)
    suffix = "_$(score_name)"
    return endswith(stem, suffix) ? npy_path : stem * suffix * ".npy"
end

"""
    _network_genes_path(file_path) -> String

Path of the gene-label sidecar file (`<stem>_genes.txt`) that accompanies
an inferred-network `.npy` dump at `file_path`.
"""
function _network_genes_path(file_path::AbstractString)
    stem, _ = splitext(_npy_output_path(file_path))
    return stem * "_genes.txt"
end

"""
    _score_genes_path(file_path, score_name) -> String

Path of the gene-label sidecar file (`<stem>_genes.txt`) that accompanies
the `score_name` (`:mi` or `:puc`) `.npy` dump derived from `file_path`.
"""
function _score_genes_path(file_path::AbstractString, score_name::Symbol)
    score_path = _score_output_path(file_path, score_name)
    stem, _ = splitext(score_path)
    suffix = "_$(score_name)"
    return chop(stem; tail = length(suffix)) * "_genes.txt"
end

"""
    _write_genes_file(file_path, nodes)

Write one [`Node`](@ref) label per line to `file_path`, in the order given
by `nodes`, to serve as the row/column key for a companion `.npy` matrix.
"""
function _write_genes_file(file_path::AbstractString, nodes)
    open(file_path, "w") do io
        for node in nodes
            println(io, String(node.label))
        end
    end
    return nothing
end


"""
    Node(label::AbstractString, raw_values::AbstractVector{<:Real},
         discretizer, estimator, number_of_bins)

Construct a `Node` directly from a label and typed numeric values. This path
avoids materializing the legacy mixed-type `Matrix{Any}` row when loading HDF5
columns, reducing allocation and type instability without changing the
underlying discretization or probability calculations.
"""
function Node(
    label::AbstractString,
    raw_values::AbstractVector{<:Real},
    discretizer,
    estimator,
    number_of_bins,
)
    values = collect(Float64, raw_values)

    # Raw values are mapped to their bin IDs.
    binned_values = zeros(Int, length(values))

    # If the discretizer is Bayesian blocks, number_of_bins will be
    # overwritten by the ideal number of bins. Otherwise, it will remain
    # the same as the value passed in.
    number_of_bins = get_bin_ids!(values, discretizer, number_of_bins, binned_values)

    probabilities = get_probabilities(
        estimator,
        get_frequencies_from_bin_ids(binned_values, number_of_bins),
    )

    return Node(String(label), binned_values, number_of_bins, probabilities)
end

"""
    Node(line::AbstractArray, discretizer, estimator, number_of_bins) -> Node

Construct a [`Node`](@ref) from one row of a data file: `line` is an array
whose first element is the node's label and whose remaining elements are
its raw (continuous) data values. The raw values are discretized using
`discretizer` (overwriting `number_of_bins` if it is `"bayesian_blocks"`,
which chooses its own bin count) and the per-bin probabilities are then
estimated using `estimator`.
"""
function Node(line::AbstractArray, discretizer, estimator, number_of_bins)
    return Node(
        string(line[1]),
        collect(Float64, line[2:end]),
        discretizer,
        estimator,
        number_of_bins,
    )
end


"""
    get_mi_and_si(node1::Node, node2::Node, estimator, base) -> (mi, si1, si2)

Compute the mutual information `mi` between `node1` and `node2`, along with
the specific information of each node with respect to the other (`si1` for
`node1`, `si2` for `node2`), using probabilities estimated with `estimator`
and logarithms of base `base`.
"""
function get_mi_and_si(node1::Node, node2::Node, estimator, base)
    probabilities, probabilities1, probabilities2 =
        get_joint_probabilities(node1, node2, estimator)
    mi = apply_mutual_information_formula(
        probabilities,
        probabilities1,
        probabilities2,
        base,
    )
    si1 = apply_specific_information_formula(
        probabilities,
        probabilities1,
        probabilities2,
        1,
        base,
    )
    si2 = apply_specific_information_formula(
        probabilities,
        probabilities2,
        probabilities1,
        2,
        base,
    )
    return mi, si1, si2
end

"""
Cache of information measures for an ordered pair of nodes.

Fields:
* `mi`: mutual information between the two nodes
* `si`: specific information of the first node with respect to the second
"""
struct NodePair
    mi::Float64
    si::Array{Float64}
end

"""
Undirected edge

Fields:
* `nodes`: the two nodes, in an arbitrary order
* `weight`: weight indicating confidence of edge existing in the true network
Weights are used to rank the edges, and different algorithms may have a
different scale. The relative weights within one inferred network are
therefore more meaningful than the absolute weight out of context.
"""
struct Edge
    nodes::Tuple{Node,Node}
    weight::Float64
end
