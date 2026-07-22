# Vendored from Discretizers.jl (https://github.com/sisl/Discretizers.jl),
# trimmed to the discretization algorithms and encoding path used by FastPIDC
# (uniform-width, uniform-count and Bayesian blocks binning of Float64 data).
#
# Discretizers.jl is licensed under the MIT "Expat" License:
# Copyright (c) 2015: Tim Wheeler.

# --- Generic types -----------------------------------------------------

abstract type AbstractDiscretizer{N,D} end
    # N indicates the decoded, or natural type
    # D indicates the encoded, or discrete type

abstract type DiscretizationAlgorithm end

# --- LinearDiscretizer ---------------------------------------------------
# Encodes (typically continuous) values into discrete bins. A univariate
# domain is divided into discrete bins by edges. A value V is encoded into
# bin B if V ∈ [Bₗ, Bᵣ) (or V ∈ [Bₗ, Bᵣ] if B is the rightmost bin). Values
# outside of binedges are shunted into the nearest bin.

const DEFAULT_LIN_DISC_FORCE_OUTLIERS_TO_CLOSEST = true

struct LinearDiscretizer{N<:Real,D<:Integer} <: AbstractDiscretizer{N,D}
    binedges::Vector{N}   # list of bin edges, sorted smallest to largest
    nbins::Int
    force_outliers_to_closest::Bool
end

function LinearDiscretizer(
    binedges::AbstractArray{N},
    ::Type{D} = Int;
    force_outliers_to_closest::Bool = DEFAULT_LIN_DISC_FORCE_OUTLIERS_TO_CLOSEST,
) where {N<:Real,D<:Integer}

    length(binedges) > 1 || error("bin edges must contain at least 2 values")

    if N <: AbstractFloat
        findfirst(i -> binedges[i-1] ≥ binedges[i], 2:length(binedges)) == nothing ||
            error("Bin edges must be sorted in increasing order")
    else # for integers, bins of unit width require repeated values
        (
            findfirst(i -> binedges[i-1] ≥ binedges[i], 2:length(binedges)-1) ==
            nothing &&
            findfirst(i -> binedges[i-1] > binedges[i], 2:length(binedges)) == nothing
        ) || error("Bin edges must be sorted in increasing order")
    end

    nbins = length(binedges) - 1

    return LinearDiscretizer{N,D}(
        convert(Vector{N}, binedges),
        nbins,
        force_outliers_to_closest,
    )
end

function encode(ld::LinearDiscretizer{N,D}, x::N) where {N<:Real,D<:Integer}
    !isnan(x) || error("cannot encode NaN values")

    if x < ld.binedges[1]
        return ld.force_outliers_to_closest ? convert(D, 1) : throw(BoundsError())
    elseif x > ld.binedges[end]
        return ld.force_outliers_to_closest ? convert(D, ld.nbins) : throw(BoundsError())
    else
        # run bisection search
        binedges = ld.binedges
        a, b = 1, length(binedges)
        va, vb = binedges[a], binedges[b]
        while b - a > 1
            c = div(a + b, 2)
            vc = binedges[c]
            if x < vc
                b, vb = c, vc
            else
                a, va = c, vc
            end
        end
        return convert(D, a)
    end
end
encode(ld::LinearDiscretizer{N,D}, x) where {N<:Real,D<:Integer} =
    encode(ld, convert(N, x))::D
function encode(ld::LinearDiscretizer{N,D}, data::AbstractArray) where {N<:Real,D<:Integer}
    arr = [encode(ld, x) for x in data]
    reshape(arr, size(data))
end

# --- Uniform-width discretization -----------------------------------------

struct DiscretizeUniformWidth <: DiscretizationAlgorithm
    nbins::Int
end

function binedges(alg::DiscretizeUniformWidth, data::AbstractArray{N}) where {N<:AbstractFloat}
    lo, hi = extrema(data)
    @assert(hi > lo)
    convert(Vector{N}, collect(range(lo, stop = hi, length = alg.nbins + 1)))
end
function binedges(alg::DiscretizeUniformWidth, data::AbstractArray{N}) where {N<:Integer}
    lo, hi = extrema(data)
    @assert(hi > lo)
    collect(range(lo, stop = hi, length = alg.nbins + 1))
end

# --- Uniform-count discretization -----------------------------------------

struct DiscretizeUniformCount <: DiscretizationAlgorithm
    nbins::Int
end

function binedges(alg::DiscretizeUniformCount, data::AbstractArray{N}) where {N<:AbstractFloat}

    nbins = alg.nbins

    n = length(data)
    n ≥ nbins || error("too many bins requested")

    p = sortperm(data)
    counts_per_bin, remainder = div(n, nbins), rem(n, nbins)
    retval = Array{N}(undef, nbins + 1)
    retval[1] = data[p[1]]
    retval[end] = data[p[end]]

    ind = 0
    for i = 2:nbins
        counts = counts_per_bin + (remainder > 0.0 ? 1 : 0)
        remainder -= 1.0
        ind += counts
        retval[i] = (data[p[ind]] + data[p[ind+1]]) / 2
        retval[i-1] != retval[i] || error("binedges non-unique")
    end

    retval
end
function binedges(alg::DiscretizeUniformCount, data::AbstractArray{N}) where {N<:Integer}

    nbins = alg.nbins

    n = length(data)
    n ≥ nbins || error("too many bins requested")

    p = sortperm(data)
    counts_per_bin, remainder = div(n, nbins), rem(n, nbins)
    retval = Array{N}(undef, nbins + 1)
    retval[1] = data[p[1]]
    retval[end] = data[p[end]]

    ind = 0
    for i = 2:nbins
        counts = counts_per_bin + (remainder > 0.0 ? 1 : 0)
        remainder -= 1.0
        ind += counts
        retval[i] = ceil(Int, (data[p[ind]] + data[p[ind+1]]) / 2) # V will be placed in bin B if V ∈ [Bₗ, Bᵣ)
        retval[i-1] != retval[i] || error("binedges non-unique")
    end

    retval
end

# --- Bayesian blocks discretization -----------------------------------------
# Implementation by Michael P.H. Stumpf and T. Chan.
# Based on the Python code of Jake Vanderplas.
#
# This version implements Bayesian blocks for histograms, where the data are
# sorted, then treated as event data (see Scargle 2012). Defaults as
# suggested in Scargle 2012 are used.
#
# References:
# Scargle 2012: http://adsabs.harvard.edu/abs/2012arXiv1207.5578S
# Python implementation: https://github.com/astroML/astroML/blob/master/astroML/density_estimation/bayesian_blocks.py

struct DiscretizeBayesianBlocks <: DiscretizationAlgorithm end

function binedges(alg::DiscretizeBayesianBlocks, data::AbstractArray{N}) where {N<:AbstractFloat}

    # Single sorted pass to get unique values together with their multiplicities,
    # rather than sorting/uniquing separately and then re-scanning the full
    # data array once per unique value (which is O(n_unique * length(data))).
    sorted_data = sort(vec(data))
    m = length(sorted_data)

    unique_data = Vector{Float64}(undef, m)
    nn_vec = Vector{Float64}(undef, m)
    n = 0
    i = 1
    @inbounds while i <= m
        v = sorted_data[i]
        j = i + 1
        while j <= m && sorted_data[j] == v
            j += 1
        end
        n += 1
        unique_data[n] = v
        nn_vec[n] = j - i
        i = j
    end
    resize!(unique_data, n)
    resize!(nn_vec, n)

    edges = zeros(n + 1)
    edges[1] = unique_data[1]
    for i = 1:(n-1)
        edges[i+1] = 0.5 * (unique_data[i] + unique_data[i+1])
    end
    edges[end] = unique_data[end]
    block_length = unique_data[end] .- edges

    count_vec = zeros(n)
    best = zeros(n)
    last = zeros(Int64, n)

    # Reused across iterations so the O(n^2) DP does not also pay for O(n^2)
    # bytes of temporary array allocation (one fresh slice/broadcast per K).
    widths = Vector{Float64}(undef, n)
    fit_vec = Vector{Float64}(undef, n)

    @inbounds for K = 1:n
        block_length_K1 = block_length[K+1]
        for i = 1:K
            widths[i] = block_length[i] - block_length_K1
        end
        for i = 1:K
            count_vec[i] += nn_vec[K]
        end

        # Prior (eq. 21 from Scargle 2012)
        prior = 4 - log(73.53 * 0.05 * ((K)^-0.478))
        # Fitness function (eq. 19 from Scargle 2012)
        for i = 1:K
            fit_vec[i] = count_vec[i] * log(count_vec[i] / widths[i]) - prior
        end
        for i = 2:K
            fit_vec[i] += best[i-1]
        end

        i_max = 1
        best_val = fit_vec[1]
        for i = 2:K
            if fit_vec[i] > best_val
                best_val = fit_vec[i]
                i_max = i
            end
        end
        last[K] = i_max
        best[K] = best_val
    end

    change_points = zeros(Int64, n)
    i_cp = n + 1
    ind = n + 1
    while true
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 1
            break
        end
        ind = last[ind-1]
    end
    change_points = change_points[i_cp:end]
    edges[change_points]

end
function binedges(alg::DiscretizeBayesianBlocks, data::AbstractArray{N}) where {N<:Integer}
    data = convert(Array{Float64}, data)
    return binedges(alg, data)
end
