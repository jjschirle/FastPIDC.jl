# Vendored from InformationMeasures.jl (https://github.com/Tchanders/InformationMeasures.jl),
# trimmed to the discretization, probability estimation and information-theoretic
# formulae actually used by FastPIDC.
#
# InformationMeasures.jl is licensed under the MIT "Expat" License:
# Copyright (c) 2015: Thalia Chan.

# --- Discretization --------------------------------------------------------
# (originally InformationMeasures.jl's Discretization.jl)

"""
    get_bin_ids!(values_x, mode, number_of_bins, bin_ids) -> Int

Discretize `values_x` in place into `bin_ids` (a pre-allocated array of the
same length), using discretization method `mode`.

# Arguments
* `values_x`: array of raw (continuous) data values.
* `mode`: discretization method — one of `"bayesian_blocks"`,
  `"uniform_width"`, `"uniform_count"` or `"binarize"`. Falls back to
  `"uniform_width"` (with a printed message) if `mode` is unrecognized, or
  if the requested method fails on this data.
* `number_of_bins`: number of bins to use; ignored (and overwritten in the
  return value) when `mode == "bayesian_blocks"`, since that method chooses
  its own bin count.
* `bin_ids`: pre-allocated output array, filled in place with the bin id of
  each value in `values_x`.

Returns the actual number of bins used, which differs from `number_of_bins`
when `mode == "bayesian_blocks"` or when all values in `values_x` are equal
(in which case a single bin is used).
"""
function get_bin_ids!(values_x, mode, number_of_bins, bin_ids)
    min, max = extrema(values_x)
    if min == max
        # If values are all the same, assign them all to bin 1
        number_of_bins = 1
        bin_ids[1:end] .= 1
    elseif mode == "uniform_width"
        bin_ids[1:end] = encode(
            LinearDiscretizer(binedges(DiscretizeUniformWidth(number_of_bins), values_x)),
            values_x,
        )
    elseif mode == "binarize"
        number_of_bins = 2
        bin_ids[1:end] = map(v -> v == 0 ? 1 : 2, values_x)
    elseif mode == "uniform_count"
        try
            bin_ids[1:end] = encode(
                LinearDiscretizer(
                    binedges(
                        DiscretizeUniformCount(number_of_bins),
                        reshape(values_x, length(values_x)),
                    ),
                ),
                values_x,
            )
        catch
            bin_ids = encode(
                LinearDiscretizer(
                    binedges(DiscretizeUniformWidth(number_of_bins), values_x),
                ),
                values_x,
            )
            println("Uniform count failed, fell back to uniform width")
        end
    elseif mode == "bayesian_blocks"
        try
            edges = binedges(DiscretizeBayesianBlocks(), values_x)
            bin_ids[1:end] = encode(LinearDiscretizer(edges), values_x)
            number_of_bins = length(edges) - 1
        catch
            bin_ids[1:end] = encode(
                LinearDiscretizer(
                    binedges(DiscretizeUniformWidth(number_of_bins), values_x),
                ),
                values_x,
            )
            println("Bayesian blocks failed, fell back to uniform width")
        end
    else
        println("Mode doesn't exist, fell back to uniform width")
        bin_ids[1:end] = encode(
            LinearDiscretizer(binedges(DiscretizeUniformWidth(number_of_bins), values_x)),
            values_x,
        )
    end

    return number_of_bins
end

"""
    get_frequencies_from_bin_ids(bin_ids_x, number_of_bins_x) -> Vector{Int}

Count how many values fall into each of `number_of_bins_x` bins, given
`bin_ids_x`, the bin id assigned to each value.
"""
function get_frequencies_from_bin_ids(bin_ids_x, number_of_bins_x)
    n = size(bin_ids_x)[1]
    frequencies = zeros(Int, number_of_bins_x)
    for i = 1:n
        frequencies[bin_ids_x[i]] += 1
    end
    return frequencies
end
"""
    get_frequencies_from_bin_ids(bin_ids_x, bin_ids_y, number_of_bins_x, number_of_bins_y) -> Matrix{Int}

Count the joint frequency of each `(bin_ids_x, bin_ids_y)` pair, returning a
`number_of_bins_x` by `number_of_bins_y` matrix of counts.
"""
function get_frequencies_from_bin_ids(
    bin_ids_x,
    bin_ids_y,
    number_of_bins_x,
    number_of_bins_y,
)
    n = size(bin_ids_x)[1]
    frequencies = zeros(Int, (number_of_bins_x, number_of_bins_y))
    for i = 1:n
        frequencies[bin_ids_x[i], bin_ids_y[i]] += 1
    end
    return frequencies
end

# --- Probability estimators -------------------------------------------------
# (originally InformationMeasures.jl's Estimators.jl)
#
# Estimators are described in:
# Hausser, Jean; Strimmer, Korbinian (2009-01-01).
# "Entropy Inference and the James-Stein Estimator, with Application to Nonlinear Gene Association Networks"
# https://arxiv.org/abs/0811.3579
#
# R implementation of estimators:
# https://cran.r-project.org/web/packages/entropy/

"""
    get_probabilities_dirichlet(frequencies, prior) -> Array{Float64}

Estimate probabilities from `frequencies` using a Dirichlet estimator with
concentration `prior` added to every bin before normalizing.
"""
function get_probabilities_dirichlet(frequencies, prior)
    prior_frequencies = fill(prior, size(frequencies))
    return (frequencies + prior_frequencies) / (sum(frequencies) + sum(prior_frequencies))
end

"""
    get_probabilities_maximum_likelihood(frequencies) -> Array{Float64}

Estimate probabilities as the simple relative frequencies
`frequencies / sum(frequencies)`.
"""
function get_probabilities_maximum_likelihood(frequencies)
    return frequencies / sum(frequencies)
end

"""
    get_probabilities_shrinkage(frequencies, lambda::Nothing) -> Array{Float64}

Estimate probabilities via James-Stein shrinkage towards the uniform
distribution, estimating the shrinkage intensity automatically (via
[`get_lambda`](@ref)) since `lambda` is `nothing`.
"""
function get_probabilities_shrinkage(frequencies, lambda::Nothing)
    target = get_uniform_distribution(frequencies)
    n = sum(frequencies)
    normalized_frequencies = frequencies / n
    calculated_lambda = get_lambda(normalized_frequencies, target, n)
    return apply_shrinkage_formula(normalized_frequencies, target, calculated_lambda)
end
"""
    get_probabilities_shrinkage(frequencies, lambda) -> Array{Float64}

Estimate probabilities via James-Stein shrinkage towards the uniform
distribution, using the fixed shrinkage intensity `lambda`.
"""
function get_probabilities_shrinkage(frequencies, lambda)
    target = get_uniform_distribution(frequencies)
    normalized_frequencies = get_normalized_frequencies(frequencies)
    return apply_shrinkage_formula(normalized_frequencies, target, lambda)
end

"""
    apply_shrinkage_formula(normalized_frequencies, target, lambda)

Blend `normalized_frequencies` with `target` by shrinkage intensity
`lambda`: `lambda * target + (1 - lambda) * normalized_frequencies`.
"""
function apply_shrinkage_formula(normalized_frequencies, target, lambda)
    return lambda * target .+ (1 - lambda) * normalized_frequencies
end

"""
    get_uniform_distribution(frequencies) -> Float64

Probability of a single bin under a uniform distribution over
`length(frequencies)` bins, i.e. `1 / length(frequencies)`.
"""
function get_uniform_distribution(frequencies)
    return 1 / length(frequencies)
end

"""
    get_normalized_frequencies(frequencies) -> Array{Float64}

Relative frequencies `frequencies / sum(frequencies)`.
"""
function get_normalized_frequencies(frequencies)
    return frequencies / sum(frequencies)
end

"""
    get_lambda(normalized_frequencies, target, n) -> Float64

Estimate the James-Stein shrinkage intensity given already-normalized
frequencies, a `target` distribution, and sample size `n`, following
Hausser & Strimmer (2009). Returns `1.0` when `n` is `0` or `1`. The result
is clamped to `[0, 1]`.
"""
function get_lambda(normalized_frequencies, target, n)
    if n == 1 || n == 0
        return 1.0
    end
    # Unbiased estimator of variance of u
    varu = normalized_frequencies .* (1 .- normalized_frequencies) / (n - 1)
    msp = sum((normalized_frequencies .- target) .^ 2) # misspecification

    # Estimate shrinkage intensity
    lambda = msp == 0 ? 1.0 : sum(varu) / msp

    # Make lambda be between 0 and 1 inclusive
    return lambda > 1 ? 1.0 : (lambda < 0 ? 0.0 : lambda)

end
"""
    get_lambda(frequencies, get_target=get_uniform_distribution) -> Float64

Estimate the James-Stein shrinkage intensity directly from raw
`frequencies`, computing the target distribution via `get_target` and
normalizing internally. Returns `1` when the total count is `0` or `1`.
"""
function get_lambda(frequencies, get_target = get_uniform_distribution)
    n = sum(frequencies)
    if n == 1 || n == 0
        return 1
    end

    target = get_target(frequencies)
    normalized_frequencies = frequencies / n

    return get_lambda(normalized_frequencies, target, n)
end

"""
    get_probabilities(estimator, frequencies; <keyword arguments>)

Estimate probabilities from a set of discrete values.

# Arguments:
* `estimator`: the entropy estimator.
* `frequencies`: the bin frequencies for the discretized data values.
* `lambda=nothing`: the shrinkage instensity, only used if `estimator` is `"shrinkage"`.
* `prior=1`: the Dirichlet prior, only used if `estimator` is `"dirichlet"`.
"""
function get_probabilities(estimator, frequencies; lambda = nothing, prior = 1)

    if estimator == "maximum_likelihood" || estimator == "miller_madow"
        probabilities = get_probabilities_maximum_likelihood(frequencies)
    elseif estimator == "shrinkage"
        probabilities = get_probabilities_shrinkage(frequencies, lambda)
    elseif estimator == "dirichlet"
        probabilities = get_probabilities_dirichlet(frequencies, prior)
    end

    return probabilities
end

# --- Formulae ----------------------------------------------------------
# (originally InformationMeasures.jl's Formulae.jl)
#
# Information measures are reviewed in:
# Timme, Nicholas; Alford, Wesley; Flecker, Benjamin; Beggs, John M. (2013-07-03).
# "Synergy, redundancy, and multivariate information measures: an experimentalist's perspective".
# Journal of Computational Neuroscience. 36 (2): 119-140.
# http://link.springer.com/article/10.1007%2Fs10827-013-0458-4

"""
    remove_non_finite(x)

Return `x` unchanged if finite, otherwise `zero(x)`. Used to silence
`NaN`/`Inf` contributions (e.g. `0 * log(0)` terms) in information-measure
sums.
"""
function remove_non_finite(x)
    return isfinite(x) ? x : zero(x)
end

"""
    apply_mutual_information_formula(p_xy, p_x, p_y, base) -> Float64

Compute the mutual information `sum(p_xy .* log(base, p_xy ./ (p_x .* p_y)))`
between two variables, given their joint probabilities `p_xy` and marginal
probabilities `p_x`, `p_y`, using logarithms of base `base`. Non-finite
terms (arising from zero probabilities) are treated as zero.
"""
function apply_mutual_information_formula(
    p_xy::AbstractArray{T},
    p_x::AbstractArray{T},
    p_y::AbstractArray{T},
    base::R,
) where {T<:AbstractFloat,R<:Real}
    return sum(remove_non_finite.(p_xy .* log.(base, p_xy ./ (p_x .* p_y))))
end

"""
    apply_specific_information_formula(p_xz, p_x, p_z, dim_sum, base) -> Vector{Float64}

Compute the specific information of a source variable with respect to a
target variable, given their joint probabilities `p_xz` and marginals
`p_x` (source) and `p_z` (target). Summation is performed along dimension
`dim_sum` of `p_xz` (the source's axis), so the result has one value per
target bin. Logarithms use base `base`; non-finite terms are treated as
zero.
"""
function apply_specific_information_formula(p_xz, p_x, p_z, dim_sum, base)
    return vec(
        sum(
            remove_non_finite.((p_xz ./ p_z) .* log.(base, p_xz ./ (p_x .* p_z))),
            dims = dim_sum,
        ),
    )
end

"""
    apply_redundancy_formula(p_z, specific_information_1, specific_information_2, base) -> Float64

Compute the redundancy between two source variables with respect to a
common target, as the expectation (over the target's marginal
distribution `p_z`) of the minimum of their specific informations
`specific_information_1` and `specific_information_2`. `base` is accepted
for a consistent call signature with the other information-measure
formulae but does not affect the computation (the specific informations
already encode the logarithm base they were computed with).
"""
function apply_redundancy_formula(
    p_z::AbstractArray{T},
    specific_information_1::AbstractArray{T},
    specific_information_2::AbstractArray{T},
    base::R,
) where {T<:AbstractFloat,R<:Real}
    minimum_specific_information = min.(specific_information_1, specific_information_2)
    return sum(vec(p_z) .* vec(minimum_specific_information))
end
