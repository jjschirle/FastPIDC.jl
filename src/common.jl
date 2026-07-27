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

# --- PIDC configuration -------------------------------------------
Base.@kwdef struct PIDCConfig
    backend::Symbol = :cuda                     # :cuda (default) or :cpu
    discretizer::String = "bayesian_blocks"     # mirrors existing default
    estimator::String = "maximum_likelihood"    # mirrors existing default
    dump_mi_path::Union{Nothing,String} = nothing   # Output stem/path; writes *_mi.npy
    dump_puc_path::Union{Nothing,String} = nothing  # Output stem/path; writes *_puc.npy
    verbose::Bool = false
    # Inner constructor for automatic validation
    function PIDCConfig(
        backend,
        discretizer,
        estimator,
        dump_mi_path,
        dump_puc_path,
        verbose,
    )

        if !(backend in (:cpu, :cuda))
            throw(ArgumentError("backend must be :cpu or :cuda, got :$backend"))
        end
        new(
            backend,
            discretizer,
            estimator,
            dump_mi_path,
            dump_puc_path,
            verbose,
        )
    end
end

# --- NumPy output helpers -----------------------------------------

function _npy_output_path(file_path::AbstractString)
    stem, _ = splitext(String(file_path))
    return stem * ".npy"
end

function _score_output_path(file_path::AbstractString, score_name::Symbol)
    score_name in (:mi, :puc) ||
        throw(ArgumentError("score_name must be :mi or :puc, got :$score_name"))

    npy_path = _npy_output_path(file_path)
    stem, _ = splitext(npy_path)
    suffix = "_$(score_name)"
    return endswith(stem, suffix) ? npy_path : stem * suffix * ".npy"
end

function _network_genes_path(file_path::AbstractString)
    stem, _ = splitext(_npy_output_path(file_path))
    return stem * "_genes.txt"
end

function _score_genes_path(file_path::AbstractString, score_name::Symbol)
    score_path = _score_output_path(file_path, score_name)
    stem, _ = splitext(score_path)
    suffix = "_$(score_name)"
    return chop(stem; tail = length(suffix)) * "_genes.txt"
end

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

# Constructs a Node from the legacy mixed-type row format: the label is the
# first element and all remaining elements are raw numeric values.
function Node(line::AbstractArray, discretizer, estimator, number_of_bins)
    return Node(
        string(line[1]),
        collect(Float64, line[2:end]),
        discretizer,
        estimator,
        number_of_bins,
    )
end


# Mutual information and specific information for a node pair
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

# Type for caching information between pairs of nodes:
# - mi: mutual information
# - si: specific information
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
