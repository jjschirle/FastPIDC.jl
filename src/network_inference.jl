# Network inference algorithms and the InferredNetwork type. The algorithms MI, CLR, PUC and
# PIDC are explained in http://biorxiv.org/content/early/2017/04/26/082099 - along with terms
# such as specific information, proportional unique contribution, context, etc.

"""
    AbstractNetworkInference

Supertype for the network inference algorithms: [`MINetworkInference`](@ref),
[`CLRNetworkInference`](@ref), [`PUCNetworkInference`](@ref) and
[`PIDCNetworkInference`](@ref). Used to select behavior in
[`InferredNetwork`](@ref) via the [`apply_context`](@ref) and
[`get_puc`](@ref) traits.
"""
abstract type AbstractNetworkInference end

"Mutual information (MI) network inference: raw pairwise MI as edge weights."
struct MINetworkInference <: AbstractNetworkInference end

"Context Likelihood of Relatedness (CLR): MI weights with per-node background context applied."
struct CLRNetworkInference <: AbstractNetworkInference end

"Proportional Unique Contribution (PUC): redundancy-corrected MI, without context."
struct PUCNetworkInference <: AbstractNetworkInference end

"Partial Information Decomposition and Context (PIDC): PUC scores with per-node background context applied."
struct PIDCNetworkInference <: AbstractNetworkInference end

"""
    apply_context(inference::AbstractNetworkInference) -> Bool

Whether `inference` applies background-context weighting (via
[`get_weights`](@ref)) to the raw scores. `true` for
[`CLRNetworkInference`](@ref) and [`PIDCNetworkInference`](@ref).
"""
apply_context(::MINetworkInference) = false
apply_context(::CLRNetworkInference) = true
apply_context(::PUCNetworkInference) = false
apply_context(::PIDCNetworkInference) = true

"""
    get_puc(inference::AbstractNetworkInference) -> Bool

Whether `inference` computes PUC (redundancy-corrected) scores rather than
raw MI. `true` for [`PUCNetworkInference`](@ref) and
[`PIDCNetworkInference`](@ref).
"""
get_puc(::MINetworkInference) = false
get_puc(::CLRNetworkInference) = false
get_puc(::PUCNetworkInference) = true
get_puc(::PIDCNetworkInference) = true

"""
    get_weight(edge::Edge) -> Float64

Accessor returning `edge.weight`, used as the sort key when ordering edges
by confidence.
"""
get_weight(edge::Edge) = edge.weight

"""
    get_joint_probabilities(node1, node2, estimator) -> (probabilities, probabilities1, probabilities2)

Estimate the joint probability distribution `probabilities` for `node1` and
`node2` (a matrix over their bin ids) using `estimator`, along with the
marginal distributions `probabilities1` and `probabilities2` recovered by
summing over the other node's bins.
"""
function get_joint_probabilities(node1, node2, estimator)

    frequencies = get_frequencies_from_bin_ids(
        node1.binned_values,
        node2.binned_values,
        node1.number_of_bins,
        node2.number_of_bins,
    )

    probabilities = get_probabilities(estimator, frequencies)
    # probabilities is already property of a node, but doing this gets correct array shapes.
    # Also, for MI and CLR, it means that we don't assume that the marginal probabilities for
    # a node are always the same, no matter what the second node is, meaning that we can use
    # estimators other than maximum likelihood. (We still can't do this for PUC and PIDC,
    # because we do make that assumption for 3-node joint distributions, in get_puc.)
    probabilities1 = sum(probabilities, dims = 2)
    probabilities2 = sum(probabilities, dims = 1)

    return (probabilities, probabilities1, probabilities2)

end

"""
    get_mi_scores(nodes, number_of_nodes, estimator, base; config=PIDCConfig()) -> SharedMatrix{Float64}

Compute the pairwise mutual information between all `nodes`, returning a
symmetric `number_of_nodes` by `number_of_nodes` matrix (the diagonal is
left as zero). `estimator` selects the probability estimator and `base`
the logarithm base. The computation is distributed across worker
processes if any are available; `config.verbose` enables progress
printouts.
"""
function get_mi_scores(
    nodes,
    number_of_nodes,
    estimator,
    base;
    config::PIDCConfig = PIDCConfig(),
)
    # Legacy path
    function get_mi(node1, node2, i, j, base, mi_scores)
        probabilities, probabilities1, probabilities2 =
            get_joint_probabilities(node1, node2, estimator)
        mi = apply_mutual_information_formula(
            probabilities,
            probabilities1,
            probabilities2,
            base,
        )
        mi_scores[i, j] = mi
        mi_scores[j, i] = mi
    end

    mi_scores = SharedArray{Float64}(number_of_nodes, number_of_nodes)

    @sync @distributed for i = 1:number_of_nodes
        if config.verbose && i % 500 == 0
            println("[FastPIDC] Distributed MI progress: x = $i / $number_of_nodes")
        end
        for j = (i+1):number_of_nodes
            get_mi(nodes[i], nodes[j], i, j, base, mi_scores)
        end
    end

    return mi_scores

end


"""
    get_puc_scores(nodes, number_of_nodes, estimator, base; config=PIDCConfig()) -> (mi_scores, puc_scores)

Compute the pre-context Proportional Unique Contribution (PUC) scores for
all pairs of `nodes` (see [`compute_puc_full`](@ref) for the algorithm),
returning both the pairwise MI matrix `mi_scores` and the PUC score matrix
`puc_scores`. `estimator` selects the probability estimator, `base` the
logarithm base, and `config` selects the computation backend (`:cpu` or
`:cuda`) and enables progress printouts when `config.verbose` is set.
"""
function get_puc_scores(
    nodes,
    number_of_nodes,
    estimator,
    base;
    config::PIDCConfig = PIDCConfig(),
)
    if config.verbose
        println("[FastPIDC] Computing PUC scores. Backend: $(config.backend)")
    end

    return compute_puc_full(nodes; estimator = estimator, base = base, config = config)
end


"""
    get_weights(inference, scores, number_of_nodes, nodes) -> SharedMatrix{Float64}

Apply background-context weighting to raw pairwise `scores` (MI for
[`CLRNetworkInference`](@ref), PUC for [`PIDCNetworkInference`](@ref)),
returning a new score matrix of edge weights.

For each node, a background distribution of its scores against all other
nodes is used to standardize its scores: for [`PIDCNetworkInference`](@ref)
a Gamma distribution is fit to the background (falling back to a
CLR-style z-score if the fit fails for either node in a pair), and for
[`CLRNetworkInference`](@ref) a z-score against the background mean/variance
is always used. See Chan, Stumpf & Babtie (2017) for details.
"""
function get_weights(inference::Union{PIDCNetworkInference, CLRNetworkInference}, scores, number_of_nodes, nodes)

    # Pre-allocate parameter storage
    use_gamma = falses(number_of_nodes)
    gamma_alpha = zeros(Float64, number_of_nodes)
    gamma_theta = zeros(Float64, number_of_nodes)
    clr_mean = zeros(Float64, number_of_nodes)
    clr_var = zeros(Float64, number_of_nodes)

    # Pre-computation pass O(N) complexity
    for i in 1:number_of_nodes
        # Remove the diagonal element scores[i, i] (the self-score)
        # Doing this vcat N times (instead of N^2 times)
        scores_i = vcat(scores[1:i-1, i], scores[i+1:end, i])
        
        # Precompute CLR parameters for the fallback / pure CLR
        clr_mean[i] = mean(scores_i)
        clr_var[i] = var(scores_i)

        if isa(inference, PIDCNetworkInference)
            try
                # Attempt Gamma MLE fit on the background scores
                g = fit(Gamma, scores_i)
                gamma_alpha[i] = shape(g)
                gamma_theta[i] = scale(g)
                use_gamma[i] = true
            catch
                use_gamma[i] = false
            end
        end
    end

    weights = SharedArray{Float64}(number_of_nodes, number_of_nodes)

    # Edge weighting pass: O(N^2) complexity, but on fast math operations
    @sync @distributed for i in 1:number_of_nodes
        for j in i+1:number_of_nodes
            score = scores[i, j]
            
            # PIDC Logic: If Gamma succeeded for BOTH genes, use Gamma CDF sum
            # (Matches original code: a try/catch on the sum forces a fallback if *either* fails)
            if isa(inference, PIDCNetworkInference) && use_gamma[i] && use_gamma[j]
                weights[i, j] = cdf(Gamma(gamma_alpha[i], gamma_theta[i]), score) + 
                                cdf(Gamma(gamma_alpha[j], gamma_theta[j]), score)
            
            # Fallback / CLR Logic: If CLR inference, or if Gamma failed for either gene
            else
                diff_i = score - clr_mean[i]
                diff_j = score - clr_mean[j]
                
                term_i = (clr_var[i] == 0 || diff_i < 0) ? 0.0 : (diff_i^2 / clr_var[i])
                term_j = (clr_var[j] == 0 || diff_j < 0) ? 0.0 : (diff_j^2 / clr_var[j])
                
                weights[i, j] = sqrt(term_i + term_j)
            end
        end
    end

    return weights
end


"""
InferredNetwork type. Represents a weighted, fully connected network, where an
edges's weight indicates the relative confidence of that edge existing in the true
network.

Fields:
* nodes: array of all the nodes, in an arbitrary order
* edges: array of all the edges, in descending order of weight
"""
struct InferredNetwork
    nodes::Array{Node}
    edges::Array{Edge}
end

"""
    build_sorted_edges(nodes, weights) -> Vector{Edge}

Build an [`Edge`](@ref) for every pair of `nodes`, weighted by the
corresponding entry of the `weights` matrix, and return them sorted in
descending order of weight.
"""
function build_sorted_edges(nodes, weights)
    number_of_nodes = length(nodes)
    edges = Edge[]
    sizehint!(edges, binomial(number_of_nodes, 2))
    for i = 1:number_of_nodes
        for j = (i+1):number_of_nodes
            push!(edges, Edge((nodes[i], nodes[j]), weights[i, j]))
        end
    end
    sort!(edges; by = get_weight, rev = true)
    return edges
end

"""
    InferredNetwork(inference::AbstractNetworkInference, nodes::Array{Node};
                     estimator="maximum_likelihood", base=2, config=PIDCConfig())

Construct an [`InferredNetwork`](@ref) given a network inference algorithm
and an array of [`Node`](@ref)s, computing pairwise scores (MI or PUC,
depending on `inference`), optionally applying context weighting, and
sorting the resulting edges by descending weight.

# Arguments
* `inference`: network inference algorithm, e.g. `PIDCNetworkInference()`.
* `estimator="maximum_likelihood"`: algorithm for estimating the
  probability distribution. This is recommended for PUC and PIDC, because
  speedups are made there based on the assumption that the marginal
  probability distribution for a node, from the joint distribution with
  any two other nodes, is always the same. Using other estimators violates
  this assumption for PUC and PIDC (in `get_joint_probabilities` and the
  PUC computation).
* `base=2`: base for the information measures.
* `config=PIDCConfig()`: computation backend and diagnostic-dump settings,
  used by the PUC/PIDC code path.
"""
function InferredNetwork(
    inference::AbstractNetworkInference,
    nodes::Array{Node};
    estimator::String = "maximum_likelihood",
    base::Int = 2,
    config::PIDCConfig = PIDCConfig(),
)
    number_of_nodes = length(nodes)

    if get_puc(inference)
        # ===== PUC / PIDC branch =====
        mi_scores, scores =
            get_puc_scores(nodes, number_of_nodes, estimator, base; config = config)

        # Optional MI dump (PIDC only)
        if isa(inference, PIDCNetworkInference) && config.dump_mi_path !== nothing
            if config.verbose
                println("[FastPIDC] Writing MI scores.")
            end
            dump_mi_scores(mi_scores, nodes, config)
        end

        # Optional pre-context PUC dump
        if config.dump_puc_path !== nothing
            if config.verbose
                println("[FastPIDC] Writing pre-context PUC scores.")
            end
            dump_puc_scores(scores, nodes, config)
        end

        # Apply context if necessary (PIDC = true, PUC = false)
        if apply_context(inference)
            if config.verbose
                println("[FastPIDC] Context weighting.")
            end

            weights = get_weights(inference, scores, number_of_nodes, nodes)
        else
            weights = scores
        end

        return InferredNetwork(nodes, build_sorted_edges(nodes, weights))

    else
        # ===== MI / CLR branch (no PUC) =====
        scores = get_mi_scores(nodes, number_of_nodes, estimator, base; config = config)

        if apply_context(inference)
            weights = get_weights(inference, scores, number_of_nodes, nodes)
        else
            weights = scores
        end

        return InferredNetwork(nodes, build_sorted_edges(nodes, weights))
    end
end
