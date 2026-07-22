# Vendored from InformationMeasures.jl (https://github.com/Tchanders/InformationMeasures.jl),
# trimmed to the discretization, probability estimation and information-theoretic
# formulae actually used by FastPIDC.
#
# InformationMeasures.jl is licensed under the MIT "Expat" License:
# Copyright (c) 2015: Thalia Chan.

# --- Discretization --------------------------------------------------------
# (originally InformationMeasures.jl's Discretization.jl)

# Parameters:
# 	- values_x, array of floats
# 	- mode, string
#	- number_of_bins, integer
#	- bin_ids, 1-dimensional array of bin ids for each value
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

# Parameters:
# 	- bin_ids_x, array of ints
# 	- number_of_bins_x, number
function get_frequencies_from_bin_ids(bin_ids_x, number_of_bins_x)
    n = size(bin_ids_x)[1]
    frequencies = zeros(Int, number_of_bins_x)
    for i = 1:n
        frequencies[bin_ids_x[i]] += 1
    end
    return frequencies
end
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

# Parameters:
# 	- frequencies, integer array
# 	- prior, number
function get_probabilities_dirichlet(frequencies, prior)
    prior_frequencies = fill(prior, size(frequencies))
    return (frequencies + prior_frequencies) / (sum(frequencies) + sum(prior_frequencies))
end

# Parameters:
# 	- frequencies, integer array
function get_probabilities_maximum_likelihood(frequencies)
    return frequencies / sum(frequencies)
end

# Parameters:
# 	- frequencies, integer array
# 	- lambda, void
function get_probabilities_shrinkage(frequencies, lambda::Nothing)
    target = get_uniform_distribution(frequencies)
    n = sum(frequencies)
    normalized_frequencies = frequencies / n
    calculated_lambda = get_lambda(normalized_frequencies, target, n)
    return apply_shrinkage_formula(normalized_frequencies, target, calculated_lambda)
end
# Parameters:
# 	- frequencies, integer array
# 	- lambda, number
function get_probabilities_shrinkage(frequencies, lambda)
    target = get_uniform_distribution(frequencies)
    normalized_frequencies = get_normalized_frequencies(frequencies)
    return apply_shrinkage_formula(normalized_frequencies, target, lambda)
end

function apply_shrinkage_formula(normalized_frequencies, target, lambda)
    return lambda * target .+ (1 - lambda) * normalized_frequencies
end

function get_uniform_distribution(frequencies)
    return 1 / length(frequencies)
end

function get_normalized_frequencies(frequencies)
    return frequencies / sum(frequencies)
end

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

function remove_non_finite(x)
    return isfinite(x) ? x : zero(x)
end

# Parameters:
# 	- joint probabilities, array of floats
# 	- probabilities (first variable), array of floats
# 	- probabilities (second variable), array of floats
# 	- base, number
function apply_mutual_information_formula(
    p_xy::AbstractArray{T},
    p_x::AbstractArray{T},
    p_y::AbstractArray{T},
    base::R,
) where {T<:AbstractFloat,R<:Real}
    return sum(remove_non_finite.(p_xy .* log.(base, p_xy ./ (p_x .* p_y))))
end

# Parameters:
# 	- joint probabilities, array of floats
# 	- probabilities (source), array of floats
# 	- probabilities (target), array of floats
# 	- dimension along which to sum, integer
# 	- base, number
function apply_specific_information_formula(p_xz, p_x, p_z, dim_sum, base)
    return vec(
        sum(
            remove_non_finite.((p_xz ./ p_z) .* log.(base, p_xz ./ (p_x .* p_z))),
            dims = dim_sum,
        ),
    )
end

# Parameters:
# 	- probabilities (target), array of floats
# 	- specific information of source 1 and target, array of floats
# 	- specific information of source 2 and target, array of floats
# 	- base, number
function apply_redundancy_formula(
    p_z::AbstractArray{T},
    specific_information_1::AbstractArray{T},
    specific_information_2::AbstractArray{T},
    base::R,
) where {T<:AbstractFloat,R<:Real}
    minimum_specific_information = min.(specific_information_1, specific_information_2)
    return sum(vec(p_z) .* vec(minimum_specific_information))
end
