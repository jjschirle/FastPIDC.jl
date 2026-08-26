"""
    FastPIDC

A package for inferring undirected gene regulatory (or other) networks from
per-node measurements, using information-theoretic algorithms: MI, CLR, PUC
and PIDC (Chan, Stumpf & Babtie 2017). The main entry points are
[`get_nodes`](@ref)/[`infer_network`](@ref) to build an [`InferredNetwork`](@ref)
from a data file, and [`write_network_file`](@ref)/[`write_network_npy`](@ref)
to save the result.
"""
module FastPIDC

using Distributions
using Distributed
using DelimitedFiles
using SharedArrays
using SparseArrays
using NPZ
using HDF5

export
    # Common types
    Node,
    Edge,
    InferredNetwork,
    # Network inference algorithms
    AbstractNetworkInference,
    MINetworkInference,
    CLRNetworkInference,
    PUCNetworkInference,
    PIDCNetworkInference,
    # Functions for inferring networks
    get_nodes,
    write_network_file,
    write_network_npy,
    read_network_file,
    get_adjacency_matrix,
    infer_network

include("discretizers.jl")
include("information_measures.jl")
include("common.jl")
export PIDCConfig # New addition
include("puc_full.jl")
include("mi_dump.jl")
include("puc_dump.jl")
include("network_inference.jl")
include("infer_network.jl")
include("empirical_bayes_glue.jl")


# Optional exports
if EB_EXISTS
    export
        # Empirical Bayes glue functions
        to_index,
        make_priors,
        empirical_bayes
end

end # module
